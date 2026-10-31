"""Directory profiles around native config files and native Python CLI options."""
import argparse
import json
from pathlib import Path
import re
import shlex

from . import ROOT

# Display groups share their file's native key namespace. Group names do not
# rename keys or change include/override order.
SECTION_GROUPS = {
    'selfplay': ('logging', 'writer', 'game_forks', 'policy_init', 'side_positions',
                 'cheap_search', 'reduce_visits', 'pda', 'surprise_weighting',
                 'parallelism', 'inference', 'handicap', 'result_utility',
                 'search', 'limits', 'temperature', 'policy_target', 'dirichlet_noise',
               'forced_playouts', 'symmetry', 'lcb', 'utility', 'puct', 'fpu',
               'value_weighting', 'subtree_value_bias', 'graph_search'),
    'train': ('training', 'optimizer', 'learning_rate', 'weight_decay', 'lookahead',
              'swa', 'precision', 'loss', 'validation', 'checkpoints', 'data',
              'brenorm', 'attention', 'native_options'),
    'shuffle': ('replay', 'workers', 'output', 'filtering', 'native_options'),
}


def profile_path(path, filename):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    path = path.resolve()
    return path / filename if path.is_dir() else path


def read_sections(path, allowed, *, origins=None):
    """Outer sections with source-order @include and file-local duplicate checks.

    Engine configs still go directly to KataGo. This adapter only adds includes
    to the outer/CLI option files; it does not interpret engine parameters.
    Origins are needed only for the few paths owned by the outer controller.
    """
    sections = {}
    aliases = {group: owner for owner in allowed for group in SECTION_GROUPS.get(owner, ())}
    origins = {} if origins is None else origins

    def read(source, ancestors=(), local=True):
        source = source.resolve()
        if source in ancestors:
            raise ValueError('Configuration include cycle: ' + ' -> '.join(map(str, (*ancestors, source))))
        section, seen = None, set()
        for number, line in enumerate(source.read_text().splitlines(), 1):
            line = line.strip()
            location = f'{source}:{number}'
            if not line or line.startswith(('#', ';')):
                continue
            if line.startswith('@'):
                include = re.fullmatch(r'@include[\s=]+(.+)', line)
                words = shlex.split(include[1], comments=True) if include else []
                if len(words) != 1:
                    raise ValueError(f'{location}: expected @include path')
                read(source.parent / words[0], (*ancestors, source))
                continue
            if line.startswith('['):
                header = re.fullmatch(r'\[([^]]+)\]\s*(?:[#;].*)?', line)
                name = aliases.get(header[1], header[1]) if header else None
                if name not in allowed:
                    raise ValueError(f'{location}: expected sections {", ".join(allowed)}')
                section = name
                sections.setdefault(section, {})
                continue
            key, separator, value = line.partition('=')
            key = key.strip()
            if section is None or not key or any(c.isspace() for c in key):
                raise ValueError(f'{location}: expected a key inside an allowed section')
            entry = (section, key)
            if entry in seen:
                raise ValueError(f'{location}: duplicate key {section}.{key}')
            seen.add(entry)
            sections[section][key] = value.strip() if separator else None
            origins[entry] = source
        override = source.with_name(source.name + '.local')
        if local and override.is_file():
            read(override, (*ancestors, source), local=False)

    read(Path(path))
    return sections


def boolean(value):
    if value not in ('true', 'false'):
        raise ValueError('Boolean settings must be true or false')
    return value == 'true'


def settings(raw, converters, path):
    unknown = set(raw) - converters.keys()
    if unknown:
        raise ValueError(f'{path}: unknown outer settings: {sorted(unknown)}')
    values = {}
    for key, value in raw.items():
        try:
            words = shlex.split(value or '', comments=True)
            if len(words) != 1:
                raise ValueError('expected one value (quote paths containing spaces)')
            values[key] = converters[key](words[0])
        except (ValueError, TypeError) as error:
            raise ValueError(f'{path}: {key}: {error}') from error
    return values


def native_arguments(path, section, *, path_options=()):
    """Keep names/values in the source CLI's spelling, without a field registry."""
    origins = {}
    sections = read_sections(path, [section], origins=origins)
    args = []
    for key, value in sections.get(section, {}).items():
        if not key or key.startswith('-') or any(c.isspace() for c in key):
            raise ValueError(f'{path}: use native option names without the leading dash: {key!r}')
        words = shlex.split(value or '', comments=True)
        # Bare flags enable a native option; an explicit empty assignment removes
        # an inherited option, e.g. `no-compile =` enables source compilation.
        if value is not None and not words:
            continue
        if key in path_options and len(words) == 1:
            words = [str((origins[section, key].parent / words[0]).resolve())]
        args.extend(['-' + key, *words])
    return args


class BoundaryParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)

    def _get_option_tuples(self, option_string):
        # argparse's allow_abbrev=False only covers double-dash options; native
        # options use one dash. Consume exact spellings only at this boundary.
        return []


def load_run_profile(path):
    from .config import RunConfig
    path = profile_path(path, 'run.cfg')
    origins = {}
    sections = read_sections(path, ['run'], origins=origins)
    def resolve(key, value):
        return str((origins.get(('run', key), path).parent / value).resolve())
    converters = dict(game=str, algorithm=str, bootstrap_games=int, train_steps=int,
                      replay_ratio=float, bootstrap_game_multiplier=float, publish_weights=str,
                      gatekeeper_config=str,
                      gatekeeper_required_win_prop=float, auto_plot=boolean,
                      selfplay_config=str, train_config=str, shuffle_config=str,
                      iterations=int, max_hours=float, run_dir=str)
    values = settings(sections.get('run', {}), converters, path)
    launch = {key: values.pop(key) for key in ('iterations', 'max_hours', 'run_dir') if key in values}
    if 'run_dir' in launch:
        launch['run_dir'] = resolve('run_dir', launch['run_dir'])
    for key, default in (('selfplay_config', 'selfplay.cfg'), ('gatekeeper_config', None)):
        value = values.get(key, default)
        if value is not None:
            values[key] = resolve(key, value)
    train_path = Path(resolve('train_config', values.pop('train_config', 'train.cfg')))
    shuffle_path = Path(resolve('shuffle_config', values.pop('shuffle_config', 'shuffle.cfg')))
    structured = (path.parent / 'env.cfg').exists() or (path.parent / 'net.cfg').exists()
    # Only stage-boundary options are consumed here. All other arguments pass
    # unchanged to the source parser, which owns their types/defaults/constraints.
    train = BoundaryParser(add_help=False, allow_abbrev=False)
    train.add_argument('-pos-len', dest='canvas', type=int, required=not structured)
    train.add_argument('-batch-size', dest='batch_size', type=int, required=True)
    train.add_argument('-model-kind', dest='model_kind')
    train.add_argument('-muzero-config', dest='muzero_network')
    owned, extra = train.parse_known_args(native_arguments(train_path, 'train', path_options=('muzero-config',)))
    values.update(vars(owned), train_args=extra)
    if values['muzero_network'] is not None:
        values['muzero_network'] = json.loads(Path(values['muzero_network']).read_text())
    shuffle = BoundaryParser(add_help=False, allow_abbrev=False)
    shuffle.add_argument('-num-processes', dest='shuffle_processes', type=int, default=1)
    shuffle.add_argument('-approx-rows-per-out-file', dest='shuffle_rows_per_file', type=int, default=70000)
    shuffle.add_argument('-keep-target-rows', dest='shuffle_keep_rows', type=int, default=600000)
    shuffle.add_argument('-min-rows', dest='min_rows', type=int, default=100000)
    shuffle.add_argument('-muzero-unroll-steps', dest='muzero_unroll_steps', type=int)
    owned, extra = shuffle.parse_known_args(native_arguments(shuffle_path, 'shuffle'))
    values.update(vars(owned), shuffle_args=extra)
    if structured:
        from .translation import translate_profile
        for key in ('game', 'algorithm', 'canvas', 'model_kind', 'muzero_network', 'muzero_unroll_steps'):
            if values.get(key) is not None:
                raise ValueError(f'{path}: {key} belongs in env.cfg, net.cfg or selfplay.cfg')
        values.update(translate_profile(path.parent, Path(values['selfplay_config'])))
    for key in ('bootstrap_games', 'train_steps'):
        if key not in values:
            raise ValueError(f'{path}: run.{key} is required')
    return RunConfig(**values).validate(), launch


def default_output(path, filename):
    directory = profile_path(path, filename).parent
    try:
        relative = directory.relative_to(ROOT / 'configs')
    except ValueError as error:
        raise ValueError('Profiles outside this version\'s configs/ need an explicit output directory') from error
    return ROOT / 'data' / relative


def load_experiment_profile(path):
    path = profile_path(path, 'exp.cfg')
    if path.suffix == '.json':
        return path, json.loads(path.read_text())
    origins = {}
    sections = read_sections(path, ['experiment', 'arms', 'evaluation'], origins=origins)
    values = settings(sections.get('experiment', {}),
                      dict(iterations=int, max_hours=float, shared_init=boolean, gpus=str), path)
    if 'gpus' in values:
        values['gpus'] = [gpu.strip() for gpu in values['gpus'].split(',')]
    if 'arms' in sections:
        values['arms'] = settings(sections['arms'], {name: str for name in sections['arms']}, path)
        values['arms'] = {name: str((origins['arms', name].parent / value).resolve())
                          for name, value in values['arms'].items()}
    else:
        values['arms'] = {p.name: p.name for p in sorted(path.parent.iterdir())
                          if p.is_dir() and (p / 'run.cfg').is_file()}
    if 'evaluation' in sections:
        values['evaluation'] = settings(sections['evaluation'],
            dict(config=str, stride=int, neighbors=int, games_per_pair=int, sample_seconds=float,
                 cross_seconds=float, anchor=str, final_cross=boolean, include_first=boolean,
                 prior_games=float, first_player_advantage=boolean, max_loaded_models=int), path)
        if 'config' in values['evaluation']:
            values['evaluation']['config'] = str((origins['evaluation', 'config'].parent /
                                                 values['evaluation']['config']).resolve())
    return path, values
