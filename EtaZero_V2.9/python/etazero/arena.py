"""Resumable paired matches and equal committed-time Elo, adapted from MuZero_V2."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid
from threading import Event, Thread
from tqdm.auto import tqdm

from .config import ROOT, native_text
from .engine_config import load_engine_config
from .analysis import model_info
from .build import verify_build
from .process import install_signals, stop_process
from .storage import run_lock, write_json


@dataclass(frozen=True)
class Player:
    id: str
    arm: str
    iteration: int
    seconds: float
    model: str
    sample_horizon: float | None = None


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        comparable = {k: v for k, v in manifest.items() if k != 'execution'}
        previous.pop('execution', None)
        if previous['games_per_pair'] > manifest['games_per_pair'] or (
                previous | {'games_per_pair': manifest['games_per_pair']}) != comparable:
            raise ValueError('Models, evaluator or schedule changed; use a separate output directory')
    write_json(path, manifest)


def time_sample_indices(times: list[float], sample_seconds: float, *,
                        horizon: float | None = None) -> list[int]:
    """Nearest published model to each covered time target, with earlier ties."""
    if not math.isfinite(sample_seconds) or sample_seconds <= 0:
        raise ValueError('Sample seconds must be finite and positive')
    selected = set()
    for index in range(1, math.floor((times[-1] if horizon is None else horizon) / sample_seconds)+1):
        target = index * sample_seconds
        selected.add(min(range(len(times)), key=lambda i: (abs(times[i]-target), i)))
    return sorted(selected)


def inspect_models(paths):
    """Read metadata for each unique model path."""
    return {path: model_info(path) for path in dict.fromkeys(paths)}


def discover_players(root: Path, stride: int = 1, *,
                     arms: list[tuple[str, Path]] | None = None,
                     include_first: bool = True, required_ids: tuple[str, ...] = (),
                     sample_seconds: float | None = None) -> list[Player]:
    if stride < 1:
        raise ValueError('Stride must be positive')
    candidates = []
    sample_horizons = {}
    if arms is None:
        paths = [root] if (root/'.internal/state.json').is_file() else sorted(root.iterdir())
        arms = [(p.name, p) for p in paths if (p/'.internal/state.json').is_file()]
    if len({name for name, _ in arms}) != len(arms):
        raise ValueError('Duplicate arm names')
    for name, arm in arms:
        state_path = arm/'.internal/state.json'
        if not state_path.is_file():
            raise ValueError(f'Arm has no committed state: {arm}')
        state = json.loads(state_path.read_text())
        if not state.get('timing_valid', True):
            raise ValueError(f'Run has invalid compiler timing and cannot be compared by training time: {arm}')
        previous = 0
        rows = []
        for iteration in range(state['iteration']):
            path = arm/'.internal/iterations'/f'{iteration:06d}'/'status.json'
            row = json.loads(path.read_text())['commit']
            seconds = row['elapsed_seconds']
            if row['iteration'] != iteration or not math.isfinite(seconds) or seconds < previous:
                raise ValueError(f'Invalid committed iteration timing: {path}')
            previous = seconds
            if row['model']:
                rows.append(row)
        if previous != state['elapsed_seconds']:
            raise ValueError(f'Committed time does not match state: {arm}')
        if not rows:
            raise ValueError(f'Arm has no committed models: {arm}')
        if sample_seconds is not None:
            times = [row['elapsed_seconds'] for row in rows]
            sample_horizons[name] = times[-1]
            selected = set(time_sample_indices(times, sample_seconds))
            # Keep a short history usable, but avoid a near-duplicate tail point
            # when a fixed target already represents the end of training.
            if not selected or times[-1] - times[max(selected)] > sample_seconds / 2:
                selected.add(len(rows)-1)
        else:
            selected = {i for i, row in enumerate(rows) if row['iteration'] % stride == 0}
            selected.add(len(rows)-1)
        if include_first:
            selected.add(0)
        for i, row in enumerate(rows):
            number = row['iteration']
            if i not in selected and f'{name}:{number:08d}' not in required_ids:
                continue
            candidates.append((name, arm, row))
    infos = inspect_models([arm/row['model']['path'] for _, arm, row in candidates])
    players = []
    for name, arm, row in candidates:
        number = row['iteration']
        model, info = infos[arm/row['model']['path']]
        if info != row['model']:
            raise ValueError(f'Model differs from committed record: {model}')
        players.append(Player(f'{name}:{number:08d}', name, number, row['elapsed_seconds'],
                              str(model), sample_horizons.get(name)))
    if len(players) < 2:
        raise ValueError('Need at least two committed models')
    return players


def build_schedule(players: list[Player], neighbors: int, *,
                   final_cross: bool = False, cross_seconds: float | None = None,
                   stride: int = 1, sample_seconds: float | None = None) -> list[tuple[str, str]]:
    if neighbors < 1:
        raise ValueError('Neighbors must be positive')
    if not players or len({p.id for p in players}) != len(players):
        raise ValueError('Need uniquely identified players')
    if cross_seconds is not None and (not math.isfinite(cross_seconds) or cross_seconds <= 0 or stride < 1):
        raise ValueError('Cross seconds and stride must be positive')
    arms = {}
    for player in players:
        arms.setdefault(player.arm, []).append(player)
    edges = set()
    for rows in arms.values():
        rows.sort(key=lambda p: p.iteration)
        for i, player in enumerate(rows):
            for other in rows[max(0, i-neighbors):i]:
                edges.add(tuple(sorted((player.id, other.id))))
    for a, b in combinations(arms.values(), 2):
        if cross_seconds is not None:
            # Preserve the original coverage even when the final model is
            # omitted and the nearest fixed sample lies before its target.
            if sample_seconds is not None:
                horizon_a = a[-1].seconds if a[-1].sample_horizon is None else a[-1].sample_horizon
                horizon_b = b[-1].seconds if b[-1].sample_horizon is None else b[-1].sample_horizon
                stable_a = [a[i] for i in time_sample_indices([p.seconds for p in a], sample_seconds,
                                                            horizon=horizon_a)]
                stable_b = [b[i] for i in time_sample_indices([p.seconds for p in b], sample_seconds,
                                                            horizon=horizon_b)]
                pair_horizon = min(math.floor(horizon_a/sample_seconds),
                                   math.floor(horizon_b/sample_seconds)) * sample_seconds
            else:
                stable_a = [p for p in a if p.iteration % stride == 0]
                stable_b = [p for p in b if p.iteration % stride == 0]
                pair_horizon = min(stable_a[-1].seconds, stable_b[-1].seconds) if stable_a and stable_b else 0
            if stable_a and stable_b:
                for index in range(1, math.floor(pair_horizon / cross_seconds)+1):
                    target = index * cross_seconds
                    left = min(stable_a, key=lambda p: abs(p.seconds-target))
                    right = min(stable_b, key=lambda p: abs(p.seconds-target))
                    edges.add(tuple(sorted((left.id, right.id))))
        else:
            pair_horizon = min(a[-1].seconds, b[-1].seconds)
            for source, target in ((a, b), (b, a)):
                for player in source:
                    if player.seconds > pair_horizon:
                        continue
                    other = min(target, key=lambda p: abs(p.seconds - player.seconds))
                    edges.add(tuple(sorted((player.id, other.id))))
        if final_cross:
            edges.add(tuple(sorted((a[-1].id, b[-1].id))))
    reached = {players[0].id}
    while True:
        before = len(reached)
        for a, b in edges:
            if a in reached or b in reached:
                reached.update((a, b))
        if len(reached) == before:
            break
    if reached != {p.id for p in players}:
        raise ValueError('Selected models do not form a connected comparison graph')
    return sorted(edges)


class PairStore:
    def __init__(self, directory: Path, games: int):
        if games < 4 or games % 4:
            raise ValueError('Games per pair must be a positive multiple of 4')
        self.directory = directory
        self.total = games

    def _validate(self, event, openings=None):
        kind, index = event['type'], event['id']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid event ID')
        moves = event['moves']
        if not isinstance(moves, list) or not moves or any(type(m) is not int or m < 0 for m in moves):
            raise ValueError('Invalid move history')
        if kind == 'opening':
            if index >= self.total // 2 or event['generator'] != index % 2 or not math.isfinite(event['value']):
                raise ValueError('Invalid opening assignment')
            folder = 'openings'
        elif kind == 'game':
            if index >= self.total or event['opening_id'] != index // 2 or event['black_a'] is not (index % 2 == 0) or event['winner'] not in (-1, 0, 1):
                raise ValueError('Invalid game assignment')
            opening = (openings[index//2] if openings is not None else
                       json.loads((self.directory / 'openings' / f'{index//2:08d}.json').read_text()))
            if moves[:len(opening['moves'])] != opening['moves'] or len(moves) <= len(opening['moves']):
                raise ValueError('Game does not continue its opening')
            if not math.isfinite(event['seconds']) or event['seconds'] < 0:
                raise ValueError('Invalid game duration')
            folder = 'games'
        else:
            raise ValueError(f'Unknown match event: {kind}')
        return folder

    def accept(self, event: dict) -> None:
        if event['type'] == 'match_stats':
            # Each invocation (including a resumed partial pair) is retained.
            write_json(self.directory/'performance'/f'{identity(event)}.json', event)
            return
        folder = self._validate(event)
        index = event['id']
        path = self.directory / folder / f'{index:08d}.json'
        if path.exists():
            if json.loads(path.read_text()) != event:
                raise ValueError(f'Conflicting completed event: {path}')
        else:
            write_json(path, event)

    def read(self):
        """Load and validate each immutable record once for this operation."""
        openings, games = {}, {}
        for folder, kind, rows in (('openings', 'opening', openings), ('games', 'game', games)):
            for path in sorted((self.directory/folder).glob('*.json')):
                row = json.loads(path.read_text())
                if row['type'] != kind or path.name != f"{row['id']:08d}.json":
                    raise ValueError(f'Invalid result identity: {path}')
                self._validate(row, openings)
                rows[row['id']] = row
        return openings, games

    def tasks(self, seed: int, records=None) -> list[dict]:
        openings, games = self.read() if records is None else records
        tasks = []
        for index in range(self.total // 2):
            mask = sum(1 << color for color in (0, 1)
                       if 2*index+color not in games)
            opening_seed = (seed + index * 0x9e3779b97f4a7c15) % (1 << 64)
            moves = None
            if index in openings:
                opening = openings[index]
                if opening['seed'] != opening_seed or opening['generator'] != index % 2:
                    raise ValueError('Saved opening does not match scheduled seed/generator')
                moves = opening['moves']
            if not mask:
                continue
            tasks.append({'id': index, 'seed': opening_seed, 'generator': index % 2, 'mask': mask, 'moves': moves})
        return tasks

    def games(self) -> list[dict]:
        return list(self.read()[1].values())


def _run_native(command, log_path, accept, cancel, environ):
    """Persist events while running and draining an interrupted native match process."""
    with log_path.open('a') as log:
        log.write(json.dumps({'command': command})+'\n'); log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True,
                                   start_new_session=True, env=environ)
        done = Event()
        def watch_cancel():
            while not done.wait(0.1):
                if cancel.is_set():
                    stop_process(process)
                    return
        watcher = Thread(target=watch_cancel) if cancel is not None else None
        if watcher is not None:
            watcher.start()
        try:
            for line in process.stdout:
                accept(json.loads(line))
            code = process.wait()
            if code:
                raise RuntimeError(f'{command[1].title()} exited {code}: {log_path}')
        finally:
            stopper = Thread(target=stop_process, args=(process,))
            stopper.start()
            try:
                for line in process.stdout:
                    accept(json.loads(line))
            finally:
                process.stdout.close()
                stopper.join()
                done.set()
                if watcher is not None:
                    watcher.join()


def execute_pair(binary: Path, config: Path, players: tuple[Player, Player], store: PairStore,
                 seed: int, cancel: Event | None = None, environ: dict | None = None) -> None:
    if cancel is not None and cancel.is_set():
        return
    records = store.read()
    completed = len(records[1])
    tasks = store.tasks(seed, records)
    if not tasks:
        return
    task_path = store.directory / 'tasks.txt'
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(''.join(
        f"{t['id']} {t['seed']} {t['generator']} {t['mask']} " +
        ('-1' if t['moves'] is None else f"{len(t['moves'])} " + ' '.join(map(str, t['moves']))) + '\n'
        for t in tasks))
    a, b = players
    profile = json.loads((store.directory.parents[1]/'manifest.json').read_text())['config']['match']
    def accept(event):
        nonlocal completed
        store.accept(event)
        if event['type'] == 'game':
            completed += 1
            print(f"  {a.id} vs {b.id} games={completed}/{store.total}", flush=True)
    _run_native([str(binary), 'match', '--config', str(config),
                 '--model', a.model, '--model-id', model_info(a.model)[1]['id'],
                 '--model-b', b.model, '--model-b-id', model_info(b.model)[1]['id'],
                 '--device', profile['device'], '--size', str(profile['board_size']),
                 '--rule', profile['rule'], '--tasks', str(task_path)],
                store.directory/'native.log', accept, cancel, environ)
    if store.tasks(seed):
        raise RuntimeError('Match exited without completing its assignments')


def load_results(output: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((output / 'manifest.json').read_text())
    games = []
    for pair in manifest['pairs']:
        a, b = pair['players']
        store = PairStore(output / 'pairs' / pair['id'], manifest['games_per_pair'])
        for row in store.games():
            games.append({'pair': pair['id'], 'opening_id': row['opening_id'],
                          'black': a if row['black_a'] else b, 'white': b if row['black_a'] else a,
                          'score': (row['winner'] + 1) / 2})
    return manifest, games


def native_config(config, players):
    infos = inspect_models([player.model for player in players])
    canvas = validate_match_models(config, [infos[player.model][1] for player in players])
    return {**config, 'network': {'canvas': canvas}}


def validate_match_models(config, models):
    """Check shared match conditions against model metadata or planned networks."""
    canvases = set()
    for info in models:
        canvases.add(info['canvas'])
        if info.get('algorithm') == 'muzero' and (config['match']['reuse_tree'] or
                config['match']['use_graph_search'] or config['match']['root_num_symmetries_to_sample'] != 1):
            raise ValueError('MuZero matches require reuse_tree=false, use_graph_search=false, '
                             'root_num_symmetries_to_sample=1')
    if len(canvases) != 1:
        raise ValueError('All evaluated models must have the same canvas')
    canvas = canvases.pop()
    if config['match']['board_size'] > canvas:
        raise ValueError('Match board exceeds model canvas')
    return canvas


def execute_arena(binary: Path, config: Path, output: Path, manifest: dict,
                  max_loaded_models: int, cancel: Event, environ: dict | None = None) -> None:
    """One native process, model services shared across every pending pair."""
    players = {p['id']: Player(**p) for p in manifest['players']}
    stores, models, indices, assignments = {}, [], {}, []
    profile = manifest['config']['match']
    # Pair count can greatly exceed model count; share metadata across edges.
    infos = inspect_models([player.model for player in players.values()])
    completed = 0
    for pair in manifest['pairs']:
        store = PairStore(output/'pairs'/pair['id'], manifest['games_per_pair'])
        records = store.read()
        completed += len(records[1])
        seed = (profile['seed'] + int(pair['id'][:16], 16)) % (1 << 64)
        tasks = store.tasks(seed, records)
        if not tasks:
            continue
        stores[pair['id']] = store
        slots = []
        for player_id in pair['players']:
            player = players[player_id]
            path, _ = infos[player.model]
            key = str(path)
            if key not in indices:
                indices[key] = len(models)
                # Like an explicit KataGo nnRandSeed: model contents and the
                # order of pending pairs do not determine the inference seed.
                models.append(dict(id=key, path=key, seed=profile['seed']))
            slots.append(indices[key])
        for task in tasks:
            suffix = '-1' if task['moves'] is None else f"{len(task['moves'])} " + ' '.join(map(str, task['moves']))
            assignments.append(f"{pair['id']} {slots[0]} {slots[1]} {task['id']} {task['seed']} "
                               f"{task['generator']} {task['mask']} {suffix}\n")
    if not assignments or cancel.is_set():
        return
    def quoted(value):
        if '\n' in value or '\r' in value:
            raise ValueError('Native model fields cannot contain line breaks')
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    model_path, task_path = output/'models.txt', output/'tasks.txt'
    model_path.write_text(''.join(f"{quoted(m['id'])} {quoted(m['path'])} {m['seed']}\n" for m in models))
    task_path.write_text(''.join(assignments))
    statistics = []
    total = len(manifest['pairs'])*manifest['games_per_pair']
    with tqdm(total=total, initial=completed, desc='Arena games', unit='game',
              dynamic_ncols=True, disable=None) as progress:
        def accept(event):
            nonlocal completed
            if event['type'] == 'arena_progress':
                stage = event['stage']
                if stage == 'loading':
                    progress.set_postfix_str(f"window={event['window']} preparing 0/{event['models']} models")
                    if progress.disable:
                        print(f"arena: window={event['window']} preparing {event['models']} models "
                              f"with {event['workers']} load workers", flush=True)
                elif stage == 'model_ready':
                    progress.set_postfix_str(f"window={event['window']} prepared {event['loaded']}/{event['models']} models")
                elif stage == 'playing':
                    progress.set_postfix_str(f"window={event['window']} playing")
                    if progress.disable:
                        print(f"arena: window={event['window']} ready in {event['seconds']:.2f}s; "
                              f"games={completed}/{total}", flush=True)
                statistics.append(event)
                return
            if event['type'] in ('model_prepared', 'model_stats', 'arena_stats'):
                statistics.append(event)
                return
            pair_id = event.pop('pair')
            store = stores[pair_id]
            new_game = (event['type'] == 'game' and
                        not (store.directory/'games'/f"{event['id']:08d}.json").exists())
            store.accept(event)
            if new_game:
                completed += 1
                progress.update(1)
                if progress.disable and (completed % 100 == 0 or completed == total):
                    print(f'arena: games={completed}/{total}', flush=True)
        try:
            _run_native([str(binary), 'arena', '--config', str(config),
                         '--models', str(model_path), '--tasks', str(task_path), '--max-models', str(max_loaded_models),
                         '--model-load-workers', str(manifest.get('execution', {}).get('model_load_workers', 4)),
                         '--device', profile['device'], '--size', str(profile['board_size']), '--rule', profile['rule']],
                        output/'native.log', accept, cancel, environ)
        finally:
            write_json(output/'performance'/f'{uuid.uuid4().hex}.json', {'events': statistics})
    for pair_id, store in stores.items():
        seed = (profile['seed'] + int(pair_id[:16], 16)) % (1 << 64)
        if store.tasks(seed):
            raise RuntimeError('Arena exited without completing its assignments')


def run_arena(output: Path, manifest: dict, binary: Path, samples: int,
              max_loaded_models: int = 32, environ: dict | None = None) -> dict:
    """Global native game pool; persist partial results and never fit an incomplete schedule."""
    if max_loaded_models < 2:
        raise ValueError('Arena needs at least two resident model slots')
    execution = dict(manifest.get('execution', {}))
    if execution.get('max_loaded_models', max_loaded_models) != max_loaded_models:
        raise ValueError('Arena model limit differs from manifest')
    if execution.get('model_load_workers', 4) < 1:
        raise ValueError('Arena model load workers must be positive')
    manifest = {**manifest, 'execution': {**execution, 'max_loaded_models': max_loaded_models}}
    players = [Player(**p) for p in manifest['players']]
    native = native_config(manifest['config'], players)
    with run_lock(output/'arena.lock'):
        save_manifest(output/'manifest.json', manifest)
        config_path = output/'resolved.cfg'
        config_path.write_text(native_text(native))
        cancel = Event()
        try:
            execute_arena(binary, config_path, output, manifest, max_loaded_models, cancel, environ)
        finally:
            cancel.set()
        from .elo import write_ratings
        return write_ratings(output, samples)


def single_match(config, binary, model_a, model_b, output, games=None):
    import copy
    config = copy.deepcopy(config)
    from .engine_config import validate_engine_config
    validate_engine_config(config, True)
    players = []
    for label, path in (('a', model_a), ('b', model_b)):
        path, info = model_info(path)
        players.append(Player(label, label, info['checkpoint']['iteration'], 0, str(path)))
    native = native_config(config, players)
    count = config['match']['games'] if games is None else games
    PairStore(Path(output),count)
    pair = {'id': identity(('a','b')), 'players': ['a','b']}
    verify_build(binary)
    manifest = {'version': 1, 'players': [asdict(p) for p in players], 'config': config,
                'games_per_pair': count, 'anchor': 'a', 'pairs': [pair]}
    output = Path(output).resolve()
    with run_lock(output/'arena.lock'):
        save_manifest(output/'manifest.json', manifest)
        path = output/'resolved.cfg'; path.write_text(native_text(native))
        store = PairStore(output/'pairs'/pair['id'], count)
        execute_pair(binary, path, tuple(players), store, (config['match']['seed']+int(pair['id'][:16],16)) % (1<<64))
        return output, store.games()


def main():
    parser = argparse.ArgumentParser(description='Resumable native matches and equal-wall-time Elo')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config-dir', default='configs/baseline')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/etazero')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--neighbors', type=int, default=2)
    parser.add_argument('--games', type=int, help='Games per pair; defaults to match.cfg')
    parser.add_argument('--anchor', default='')
    parser.add_argument('--bootstrap-samples', type=int, default=300)
    parser.add_argument('--game-threads', type=int, default=512)
    parser.add_argument('--max-loaded-models', type=int, default=32)
    parser.add_argument('--model-load-workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    if args.bootstrap_samples < 2:
        parser.error('--bootstrap-samples must be at least two')
    output = (args.output or args.data / 'elo').resolve()
    if args.fit_only:
        from .elo import write_ratings
        with run_lock(output/'arena.lock'):
            write_ratings(output, args.bootstrap_samples)
        return
    config = load_engine_config(args.config_dir, match=True)
    if args.game_threads < 1 or args.max_loaded_models < 2 or args.model_load_workers < 1:
        parser.error('game-threads and model-load-workers must be positive and max-loaded-models at least two')
    config['match']['game_threads'] = args.game_threads
    if args.games is None:
        args.games = config['match']['games']
    binary = args.binary.resolve()
    players = discover_players(args.data.resolve(), args.stride)
    native_config(config, players)
    schedule = build_schedule(players, args.neighbors)
    anchor = args.anchor
    if not anchor:
        baseline = [p for p in players if p.arm == 'exp_baseline'] or [p for p in players if p.arm == players[0].arm]
        anchor = min(baseline, key=lambda p: abs(p.seconds - baseline[-1].seconds / 2)).id
    if anchor not in {p.id for p in players}:
        raise ValueError(f'Unknown anchor: {anchor}')
    PairStore(output, args.games)
    verify_build(binary)
    manifest = {'version': 1, 'players': [asdict(p) for p in players], 'config': config,
                'games_per_pair': args.games, 'anchor': anchor,
                'pairs': [{'id': identity(pair), 'players': list(pair)} for pair in schedule],
                'execution': {'max_loaded_models': args.max_loaded_models,
                              'model_load_workers': args.model_load_workers}}
    print(f'players={len(players)} pairs={len(schedule)} games={len(schedule)*args.games} anchor={anchor}', flush=True)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    install_signals()
    run_arena(output, manifest, binary, args.bootstrap_samples, args.max_loaded_models)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
