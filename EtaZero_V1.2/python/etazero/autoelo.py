"""Incremental experiment Elo with fixed checkpoints and reusable paired matches."""
import argparse
import configparser
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
from .eval_config import load_evaluation_config
from .elo_cache import prepare_pairs, retain_pairs
from .process import install_signals
from .runtime import verify_build
from .storage import atomic_write, load_json, run_lock, save_json, sha256, sync_directory


ELO_FIELDS = {'stride': int, 'neighbors': int, 'cross_seconds': float,
              'final_cross': boolean, 'games_per_pair': int, 'bootstrap_samples': int,
              'anchor': str, 'pair_workers': int}


def config_directory(directory):
    path = Path(directory)
    if not path.is_absolute() and not path.exists():
        path = ROOT/path
    return path.resolve()


def load_elo_config(directory, environ=None, overrides=None):
    directory = config_directory(directory)
    values = {}
    for path in (ROOT/'configs/baseline/elo.cfg', directory/'elo.cfg', directory/'elo.cfg.local'):
        if not path.is_file():
            continue
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string(path.read_text())
        if parser.defaults() or parser.sections() != ['elo'] or set(parser['elo']) - set(ELO_FIELDS):
            raise ValueError(f'Unknown Elo sections or fields: {path}')
        values.update(parser['elo'])
    env = os.environ if environ is None else environ
    for key in ELO_FIELDS:
        if 'ELO_'+key.upper() in env:
            values[key] = env['ELO_'+key.upper()]
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    if set(values) != set(ELO_FIELDS):
        raise ValueError('Missing or unknown Elo configuration fields')
    settings = {key: convert(str(values[key])) for key, convert in ELO_FIELDS.items()}
    for key in ('stride', 'neighbors', 'pair_workers'):
        if settings[key] < 1:
            raise ValueError(f'elo.{key} must be positive')
    if settings['bootstrap_samples'] < 2:
        raise ValueError('elo.bootstrap_samples must be at least two')
    if not math.isfinite(settings['cross_seconds']) or settings['cross_seconds'] <= 0:
        raise ValueError('elo.cross_seconds must be finite and positive')
    PairStore(Path('.'), settings['games_per_pair'])
    return settings


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
                 overrides=None, environ=None, gpu=None):
    directory = config_directory(directory)
    settings = load_elo_config(directory, environ, overrides)
    config = load_evaluation_config(directory, match=True, environ=environ, umbrella=True)
    # Elo owns the number of games; match.cfg supplies only the match protocol.
    config['match']['games'] = settings['games_per_pair']
    if gpu is not None and config['match']['device'] != 'cpu':
        config['match']['device'] = 'cuda:0'
    if arms is None and data is None:
        from .experiment import experiment_plan
        arms = experiment_plan(directory, environ=environ)['arms']
    roots = [(a['name'], Path(a['run_dir'])) for a in arms] if arms is not None else None
    base = Path(output).resolve() if output is not None else default_output(directory, arms, data)
    anchor_path = base/'anchor.json'
    pinned = load_json(anchor_path) if anchor_path.exists() else None
    anchor = settings['anchor'] or (pinned['id'] if pinned else '')
    players = discover_players(Path(data).resolve() if data is not None else directory,
                               stride=settings['stride'], arms=roots, include_first=False,
                               required_ids=(anchor,) if anchor else ())
    native_config(config, players)
    schedule = build_schedule(players, settings['neighbors'],
                              cross_seconds=settings['cross_seconds'], stride=settings['stride'],
                              final_cross=settings['final_cross'])
    if not anchor:
        reference = [p for p in players if p.arm == players[0].arm]
        anchor = reference[0].id
    if anchor not in {p.id for p in players}:
        raise ValueError(f'Unknown anchor: {anchor}')
    anchor_player = next(p for p in players if p.id == anchor)
    anchor_record = {'id': anchor, 'sha256': anchor_player.sha256}
    if pinned and anchor == pinned['id'] and anchor_record != pinned:
        raise ValueError('Pinned Elo anchor model changed; use a separate result root')
    manifest = {'version': 1, 'players': [asdict(p) for p in players], 'config': config,
                'binary_sha256': verify_build(Path(binary)), 'games_per_pair': settings['games_per_pair'],
                'anchor': anchor, 'pairs': [{'id': identity(pair), 'players': list(pair)} for pair in schedule],
                'sampling': {key: settings[key] for key in ('stride', 'neighbors', 'cross_seconds', 'final_cross')},
                'execution': {'pair_workers': settings['pair_workers'],
                              'cuda_visible_devices': str(gpu) if gpu is not None else
                              (os.environ if environ is None else environ).get('CUDA_VISIBLE_DEVICES')}}
    manifest['source_sha256'] = {name: sha256(Path(__file__).parent/name)
                               for name in ('autoelo.py', 'arena.py', 'elo.py', 'elo_cache.py', 'eval_config.py')}
    # Store JSON-native values so in-memory and reloaded manifests compare equal.
    manifest = json.loads(json.dumps(manifest))
    session = base/identity(manifest)[:16]
    names = {p.id: p.arm for p in players}
    return {'directory': str(directory), 'base': str(base), 'output': str(session),
            'settings': settings, 'manifest': manifest,
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
        for name, checksum in plan['manifest']['source_sha256'].items():
            source = Path(__file__).parent/name
            if sha256(source) != checksum:
                raise ValueError(f'Elo source changed after planning: {name}; recreate the plan')
            destination = output/'source'/name
            if not destination.exists():
                atomic_write(destination, lambda p, source=source: p.write_bytes(source.read_bytes()), immutable=True)
            elif sha256(destination) != checksum:
                raise ValueError(f'Saved Elo source changed: {destination}')
        save_json(output/'plan.json', plan)
        save_json(output/'status.json', {'status': 'running'})
        started = time.monotonic()
        prepared = False
        try:
            before = prepare_pairs(base, output, plan['manifest'])
            prepared = True
            print(f"autoelo: reused_games={before}/{plan['summary']['games']}", flush=True)
            env = dict(os.environ)
            visibility = plan['manifest']['execution']['cuda_visible_devices']
            if visibility is not None:
                env['CUDA_VISIBLE_DEVICES'] = visibility
            result = run_arena(output, plan['manifest'], Path(binary),
                               plan['settings']['bootstrap_samples'], plan['settings']['pair_workers'], env)
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
                       'reused_games': before,
                       'new_games_per_second': count/seconds if seconds else 0,
                       'pair_workers': plan['settings']['pair_workers'],
                       'native_stats': [load_json(p) for p in sorted((output/'pairs').glob('*/performance/*.json'))]}
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
