"""Incremental experiment Elo with fixed checkpoints and reusable paired matches."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from .arena import (PairStore, build_schedule, discover_players, identity,
                    native_config, run_arena)
from .config import ROOT, boolean
from .config_files import read_config_file
from .engine_config import load_engine_config
from .elo_cache import prepare_pairs, retain_pairs
from .process import install_signals
from .build import verify_build
from .storage import atomic_write, load_json, run_lock, save_json, sha256, sync_directory


ELO_FIELDS = {'sample_seconds': float, 'neighbors': int, 'cross_seconds': float,
              'final_cross': boolean, 'games_per_pair': int, 'bootstrap_samples': int,
              'anchor': str, 'game_threads': int, 'max_loaded_models': int,
              'model_load_workers': int}


def config_directory(directory):
    path = Path(directory)
    if not path.is_absolute() and not path.exists():
        path = ROOT/path
    return path.resolve()


def load_elo_config(directory, environ=None, overrides=None):
    directory = config_directory(directory)
    entries = read_config_file(ROOT/'configs/baseline/elo.cfg', {'elo': ELO_FIELDS})
    for path in (directory/'elo.cfg', directory/'elo.cfg.local'):
        if path.is_file():
            entries.update(read_config_file(path, {'elo': ELO_FIELDS}))
    values = {key: value for (section, key), value in entries.items() if section == 'elo'}
    env = os.environ if environ is None else environ
    for key in ELO_FIELDS:
        if 'ELO_'+key.upper() in env:
            values[key] = env['ELO_'+key.upper()]
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    if set(values) != set(ELO_FIELDS):
        raise ValueError('Missing or unknown Elo configuration fields')
    settings = {key: convert(str(values[key])) for key, convert in ELO_FIELDS.items()}
    for key in ('neighbors', 'game_threads', 'model_load_workers'):
        if settings[key] < 1:
            raise ValueError(f'elo.{key} must be positive')
    if settings['max_loaded_models'] < 2:
        raise ValueError('elo.max_loaded_models must be at least two')
    if settings['bootstrap_samples'] < 2:
        raise ValueError('elo.bootstrap_samples must be at least two')
    for key in ('sample_seconds', 'cross_seconds'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'elo.{key} must be finite and positive')
    PairStore(Path('.'), settings['games_per_pair'])
    return settings


def load_elo_profile(directory, environ=None, overrides=None, *, gpu=None):
    """Resolve one shared match protocol and its Elo-owned execution limits."""
    settings = load_elo_config(directory, environ, overrides)
    config = load_engine_config(config_directory(directory)/'match.cfg', match=True, environ=environ,
                                overrides={'match': {'games': settings['games_per_pair'],
                                                     'game_threads': settings['game_threads']}})
    if gpu is not None and config['match']['device'] != 'cpu':
        config['match']['device'] = 'cuda:0'
    env = os.environ if environ is None else environ
    return {'settings': settings, 'config': config,
            'cuda_visible_devices': str(gpu) if gpu is not None else env.get('CUDA_VISIBLE_DEVICES')}


def default_output(directory, arms=None, data=None):
    if data is not None:
        return Path(data).resolve()/'elo'
    paths = [Path(a['run_dir']) for a in arms]
    if len({p.parent for p in paths}) == 1:
        return paths[0].parent/'elo'
    # Custom output locations need not share a parent. Keep one central result
    # under the controller's normal experiment directory in that case.
    tag = identity(str(config_directory(directory)))[:12]
    return ROOT/'data/experiments'/f'{Path(directory).name}_{tag}'/'elo'


def autoelo_plan(directory, binary, *, data=None, arms=None, output=None,
                 overrides=None, environ=None, gpu=None, profile=None):
    directory = config_directory(directory)
    if profile is None:
        profile = load_elo_profile(directory, environ, overrides, gpu=gpu)
    elif overrides is not None or environ is not None or gpu is not None:
        raise ValueError('A resolved Elo profile cannot be combined with new overrides')
    profile = deepcopy(profile)
    settings, config = profile['settings'], profile['config']
    if arms is None and data is None:
        from .experiment import load_experiment_arms
        arms = load_experiment_arms(directory)
    roots = [(a['name'], Path(a['run_dir'])) for a in arms] if arms is not None else None
    base = Path(output).resolve() if output is not None else default_output(directory, arms, data)
    anchor_path = base/'anchor.json'
    pinned = load_json(anchor_path) if anchor_path.exists() else None
    anchor = settings['anchor'] or (pinned['id'] if pinned else '')
    players = discover_players(Path(data).resolve() if data is not None else directory,
                               sample_seconds=settings['sample_seconds'], arms=roots, include_first=False,
                               required_ids=(anchor,) if anchor else ())
    native_config(config, players)
    schedule = build_schedule(players, settings['neighbors'],
                              cross_seconds=settings['cross_seconds'], sample_seconds=settings['sample_seconds'],
                              final_cross=settings['final_cross'])
    if not anchor:
        reference = [p for p in players if p.arm == players[0].arm]
        anchor = reference[0].id
    if anchor not in {p.id for p in players}:
        raise ValueError(f'Unknown anchor: {anchor}')
    anchor_player = next(p for p in players if p.id == anchor)
    anchor_record = {'id': anchor, 'model': anchor_player.model}
    if pinned and anchor == pinned['id'] and anchor_record != pinned:
        raise ValueError('Pinned Elo anchor model changed; use a separate result root')
    manifest = {'version': 1, 'players': [asdict(p) for p in players], 'config': config,
                'binary_sha256': verify_build(Path(binary)), 'games_per_pair': settings['games_per_pair'],
                'anchor': anchor, 'pairs': [{'id': identity(pair), 'players': list(pair)} for pair in schedule],
                'sampling': {key: settings[key] for key in ('sample_seconds', 'neighbors', 'cross_seconds', 'final_cross')},
                'execution': {'max_loaded_models': settings['max_loaded_models'],
                              'model_load_workers': settings['model_load_workers'],
                              'cuda_visible_devices': profile['cuda_visible_devices']}}
    sources = {name: sha256(Path(__file__).parent/name)
               for name in ('autoelo.py', 'arena.py', 'elo.py', 'elo_cache.py', 'engine_config.py')}
    # Store JSON-native values so in-memory and reloaded manifests compare equal.
    manifest = json.loads(json.dumps(manifest))
    session = base/identity(manifest)[:16]
    names = {p.id: p.arm for p in players}
    return {'directory': str(directory), 'base': str(base), 'output': str(session),
            'settings': settings, 'manifest': manifest,
            'source_sha256': sources,
            'anchor_record': anchor_record,
            'summary': {'players_per_arm': {name: sum(p.arm == name for p in players)
                       for name in sorted({p.arm for p in players})}, 'pairs': len(schedule),
                       'within_arm_pairs': sum(names[a] == names[b] for a, b in schedule),
                       'cross_arm_pairs': sum(names[a] != names[b] for a, b in schedule),
                       'games': len(schedule)*settings['games_per_pair']}}


def publish_latest(base, output):
    temporary = base/('.latest.'+uuid.uuid4().hex)
    try:
        temporary.symlink_to(output.relative_to(base), target_is_directory=True)
        os.replace(temporary, base/'latest')
        sync_directory(base)
    finally:
        temporary.unlink(missing_ok=True)


def run_autoelo(plan, binary):
    base, output = Path(plan['base']), Path(plan['output'])
    with run_lock(base/'autoelo.lock'):
        anchor_path = base/'anchor.json'
        if not anchor_path.exists():
            save_json(anchor_path, plan['anchor_record'], immutable=True)
        elif not plan['settings']['anchor'] and load_json(anchor_path) != plan['anchor_record']:
            raise ValueError('Pinned Elo anchor changed after planning; recreate the plan')
        source_id = identity(plan['source_sha256'])
        for name in plan['source_sha256']:
            source = Path(__file__).parent/name
            destination = output/'source'/source_id/name
            if not destination.exists():
                atomic_write(destination, lambda p, source=source: p.write_bytes(source.read_bytes()), immutable=True)
        save_json(output/'plan.json', plan)
        save_json(output/'status.json', {'status': 'running'})
        started = time.monotonic()
        previous_stats = set((output/'performance').glob('*.json'))
        prepared = False
        try:
            before = prepare_pairs(base, output, plan['manifest'])
            prepared = True
            print(f"autoelo: reused_games={before}/{plan['summary']['games']}", flush=True)
            env = dict(os.environ)
            visibility = plan['manifest']['execution']['cuda_visible_devices']
            if visibility is None:
                env.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                env['CUDA_VISIBLE_DEVICES'] = visibility
            result = run_arena(output, plan['manifest'], Path(binary),
                               plan['settings']['bootstrap_samples'], plan['settings']['max_loaded_models'], env)
        except BaseException as error:
            save_json(output/'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt)
                      else 'failed', 'error': str(error)})
            raise
        finally:
            if prepared:
                retain_pairs(base, output, plan['manifest'])
        seconds = time.monotonic()-started
        count = result['games']-before
        performance = {'seconds': seconds, 'new_games': count,
                       'source_id': source_id, 'source_sha256': plan['source_sha256'],
                       'reused_games': before,
                       'new_games_per_second': count/seconds if seconds else 0,
                       'game_threads': plan['settings']['game_threads'],
                       'max_loaded_models': plan['settings']['max_loaded_models'],
                       'model_load_workers': plan['settings']['model_load_workers'],
                       'native_stats': [load_json(p) for p in sorted(
                           set((output/'performance').glob('*.json')) - previous_stats)]}
        record = output/'invocations'/f'{uuid.uuid4().hex}.json'
        save_json(record, performance, immutable=True)
        save_json(output/'status.json', {'status': 'complete', 'performance': str(record)})
        publish_latest(base, output)
    print(f'autoelo: {output / "elo.png"}; new_games={count} seconds={seconds:.2f}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Incremental native Elo for an experiment umbrella')
    parser.add_argument('--config-dir', default=os.environ.get('CONFIG_DIR'))
    parser.add_argument('--data', type=Path, help='Discover actual committed arms here, independently of config arms')
    parser.add_argument('--output', type=Path, help='Result root; immutable schedules get separate subdirectories')
    parser.add_argument('--binary', type=Path, default=ROOT/'build/etazero')
    parser.add_argument('--dry-run', action='store_true', default=os.environ.get('DRY_RUN') == '1')
    parser.add_argument('--fit-only', action='store_true', help='Refit an existing --output session without matches')
    for key in ELO_FIELDS:
        parser.add_argument('--'+key.replace('_', '-'), type=str)
    args = parser.parse_args()
    if args.fit_only:
        if args.output is None:
            parser.error('--fit-only requires --output pointing to an existing session')
        from .elo import write_ratings
        samples = int(args.bootstrap_samples or 300)
        if args.dry_run:
            print(json.dumps(load_json(args.output/'manifest.json'), indent=2))
            return 0
        with run_lock(args.output/'arena.lock'):
            write_ratings(args.output, samples)
        return 0
    if not args.config_dir:
        parser.error('Set CONFIG_DIR or --config-dir to the experiment umbrella')
    plan = autoelo_plan(args.config_dir, args.binary.resolve(), data=args.data, output=args.output,
                        overrides={key: getattr(args, key) for key in ELO_FIELDS})
    print(json.dumps({'summary': plan['summary'], 'output': plan['output']}, indent=2), flush=True)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    install_signals()
    run_autoelo(plan, args.binary.resolve())
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
