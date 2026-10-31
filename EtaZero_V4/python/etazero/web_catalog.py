"""Workbench model discovery from V4 publication commits, without loading nets."""
import json
import math
from pathlib import Path

from . import ROOT


def read_json(path):
    return json.loads(path.read_text())


def native_config(directory, saved, override):
    if override is not None:
        return Path(override).resolve()
    candidates = []
    try:
        candidates.append(ROOT / 'configs' / directory.relative_to((ROOT / 'data').resolve()) / 'gtp.cfg')
    except ValueError:
        pass
    if saved.get('selfplay_config'):
        candidates.append(Path(saved['selfplay_config']).parent / 'gtp.cfg')
    return next((path.resolve() for path in candidates if path.is_file()), None)


def run_catalog(directory, *, config=None):
    directory = Path(directory).resolve()
    saved = read_json(directory / 'config.json')
    state = read_json(directory / 'state.json')
    count = state['completed_rounds']
    if type(count) is not int or count < -1:
        raise ValueError(f'Invalid completed-round count: {directory}')
    try:
        label = str(directory.relative_to((ROOT / 'data').resolve()))
    except ValueError:
        label = directory.name
    source = native_config(directory, saved, config)
    result = dict(id=str(directory), label=label, game=saved['game'], algorithm=saved['algorithm'],
                  canvas=saved['canvas'], config=str(source) if source is not None else None,
                  completed_rounds=count, current_model=None, models=[])
    models = {}
    previous = None
    first = 0 if (directory / 'rounds/000000').exists() else 1
    for number in range(first, count + 1):
        receipt = read_json(directory / 'rounds' / f'{number:06d}' / 'commit.json')
        committed = receipt['state']
        if committed['completed_rounds'] != number:
            raise ValueError(f'Publication round does not match its commit: {directory}, {number}')
        accepted = committed['accepted_model']
        for relative in dict.fromkeys([*receipt.get('candidate_models', []), receipt.get('candidate_model'), accepted]):
            if relative is None:
                continue
            bundle = (directory / relative).resolve()
            muzero = 'muzero' in saved['algorithm']
            path = bundle if muzero else bundle / 'model.bin'
            key = str(path)
            if key in models:
                continue
            files = ['model.json', 'representation.bin', 'dynamics.bin', 'prediction.bin'] if muzero else ['model.bin']
            if not all((bundle / name).is_file() for name in files):
                raise ValueError(f'Incomplete committed model: {bundle}')
            if muzero:
                manifest = read_json(bundle / 'model.json')
                if manifest['game'] != saved['game'] or manifest['pos_len'] != saved['canvas']:
                    raise ValueError(f'MuZero model differs from its published run: {bundle}')
            seconds = committed.get('elapsed_seconds')
            # Invalid timing does not hide a valid model or invent a time axis.
            timed = (committed.get('timing_valid') is True and
                     committed.get('timing_basis') == 'committed_wall_excluding_compile' and
                     type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0)
            models[key] = dict(id=key, path=key, run=str(directory), iteration=number,
                               game=saved['game'], algorithm=saved['algorithm'], canvas=saved['canvas'],
                               weights='ordinary' if bundle.name == 'initial' else saved['publish_weights'], accepted=relative == accepted,
                               training_seconds=seconds if timed else None,
                               timing_source='EtaZero controller: committed wall time excluding compilation' if timed else None)
        previous = accepted
    if count >= 0 and previous != state['accepted_model']:
        raise ValueError(f'Published model pointer does not match the committed rounds: {directory}')
    if previous is not None:
        path = (directory / previous).resolve()
        if 'muzero' not in saved['algorithm']:
            path /= 'model.bin'
        result['current_model'] = str(path)
    result['models'] = sorted(models.values(), key=lambda model: model['iteration'], reverse=True)
    return result


def discover_catalog(root=None, *, config=None):
    root = Path(root or ROOT / 'data').resolve()
    if config is not None and not Path(config).is_file():
        raise ValueError(f'Native GTP configuration does not exist: {config}')
    if not root.exists():
        return dict(runs=[], errors=[])
    if not root.is_dir():
        raise ValueError(f'Run discovery root is not a directory: {root}')
    # A run's state pointer bounds this snapshot. Models published by an active
    # but uncommitted round are intentionally absent until the next refresh.
    paths = sorted(path.parent for path in root.rglob('state.json')
                   if (path.parent / 'config.json').is_file() and
                   not any(part.startswith('.') for part in path.relative_to(root).parts))
    runs, errors = [], []
    for directory in paths:
        try:
            runs.append(run_catalog(directory, config=config))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(run=str(directory), error=str(error)))
    return dict(runs=runs, errors=errors)
