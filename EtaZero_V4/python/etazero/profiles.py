"""Directory profiles around native config files and native Python CLI options."""
import argparse
import configparser
import json
from pathlib import Path
import shlex

from . import ROOT


def profile_path(path, filename):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    path = path.resolve()
    return path / filename if path.is_dir() else path


def read_sections(path, allowed):
    parser = configparser.ConfigParser(interpolation=None, allow_no_value=True,
                                       delimiters=('=',), comment_prefixes=('#', ';'))
    parser.optionxform = str
    try:
        with path.open() as source:
            parser.read_file(source)
        local = path.with_name(path.name + '.local')
        if local.is_file():
            with local.open() as source:
                parser.read_file(source)
    except configparser.Error as error:
        raise ValueError(f'{path}: {error}') from error
    if parser.defaults() or set(parser.sections()) - set(allowed):
        raise ValueError(f'{path}: expected sections {", ".join(allowed)}; DEFAULT is not supported')
    return {section: dict(parser[section]) for section in parser.sections()}


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


def native_arguments(path, section):
    """Keep names/values in the source CLI's spelling, without a field registry."""
    sections = read_sections(path, [section])
    args = []
    for key, value in sections.get(section, {}).items():
        if not key or key.startswith('-') or any(c.isspace() for c in key):
            raise ValueError(f'{path}: use native option names without the leading dash: {key!r}')
        words = shlex.split(value or '', comments=True)
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
    sections = read_sections(path, ['run'])
    converters = dict(game=str, algorithm=str, games_per_round=int, publish_weights=str,
                      replay_ratio=float, bootstrap_min_rows=int, gatekeeper_config=str,
                      gatekeeper_required_win_prop=float, auto_plot=boolean,
                      selfplay_config=str, train_config=str, shuffle_config=str,
                      iterations=int, max_hours=float, run_dir=str)
    values = settings(sections.get('run', {}), converters, path)
    launch = {key: values.pop(key) for key in ('iterations', 'max_hours', 'run_dir') if key in values}
    if 'run_dir' in launch:
        launch['run_dir'] = str((path.parent / launch['run_dir']).resolve())
    for key, default in (('selfplay_config', 'selfplay.cfg'), ('gatekeeper_config', None)):
        value = values.get(key, default)
        if value is not None:
            values[key] = str((path.parent / value).resolve())
    train_path = (path.parent / values.pop('train_config', 'train.cfg')).resolve()
    shuffle_path = (path.parent / values.pop('shuffle_config', 'shuffle.cfg')).resolve()
    # Only stage-boundary options are consumed here. All other arguments pass
    # unchanged to the source parser, which owns their types/defaults/constraints.
    train = BoundaryParser(add_help=False, allow_abbrev=False)
    train.add_argument('-pos-len', dest='canvas', type=int, required=True)
    train.add_argument('-batch-size', dest='batch_size', type=int, required=True)
    train.add_argument('-samples-per-epoch', dest='samples_per_epoch', type=int, required=True)
    train.add_argument('-max-epochs-this-instance', dest='epochs_per_round', type=int, default=1)
    train.add_argument('-model-kind', dest='model_kind')
    train.add_argument('-muzero-config', dest='muzero_network')
    owned, extra = train.parse_known_args(native_arguments(train_path, 'train'))
    values.update(vars(owned), train_args=extra)
    if values['muzero_network'] is not None:
        values['muzero_network'] = json.loads((train_path.parent / values['muzero_network']).read_text())
    shuffle = BoundaryParser(add_help=False, allow_abbrev=False)
    shuffle.add_argument('-num-processes', dest='shuffle_processes', type=int, default=1)
    shuffle.add_argument('-approx-rows-per-out-file', dest='shuffle_rows_per_file', type=int, default=70000)
    shuffle.add_argument('-keep-target-rows', dest='shuffle_keep_rows', type=int, default=600000)
    shuffle.add_argument('-muzero-unroll-steps', dest='muzero_unroll_steps', type=int)
    owned, extra = shuffle.parse_known_args(native_arguments(shuffle_path, 'shuffle'))
    values.update(vars(owned), shuffle_args=extra)
    if 'games_per_round' not in values:
        raise ValueError(f'{path}: run.games_per_round is required')
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
    sections = read_sections(path, ['experiment', 'arms', 'evaluation'])
    values = settings(sections.get('experiment', {}),
                      dict(iterations=int, max_hours=float, shared_init=boolean, gpus=str), path)
    if 'gpus' in values:
        values['gpus'] = [gpu.strip() for gpu in values['gpus'].split(',')]
    if 'arms' in sections:
        values['arms'] = settings(sections['arms'], {name: str for name in sections['arms']}, path)
    else:
        values['arms'] = {p.name: p.name for p in sorted(path.parent.iterdir())
                          if p.is_dir() and (p / 'run.cfg').is_file()}
    if 'evaluation' in sections:
        values['evaluation'] = settings(sections['evaluation'],
            dict(config=str, stride=int, neighbors=int, games_per_pair=int, sample_seconds=float,
                 cross_seconds=float, anchor=str, final_cross=boolean, include_first=boolean,
                 prior_games=float, first_player_advantage=boolean, max_loaded_models=int), path)
    return path, values
