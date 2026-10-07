"""Resumable paired matches and equal committed-time Elo, adapted from MuZero_V2."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import ROOT, native_text
from .engine_config import load_engine_config
from .analysis import model_info
from .runtime import verify_build
from .process import install_signals, stop_process
from .storage import run_lock, write_json


@dataclass(frozen=True)
class Player:
    id: str
    arm: str
    iteration: int
    seconds: float
    model: str
    sha256: str


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['games_per_pair'] > manifest['games_per_pair'] or (
                previous | {'games_per_pair': manifest['games_per_pair']}) != manifest:
            raise ValueError('Models, evaluator, binary or schedule changed; use a separate output directory')
    write_json(path, manifest)


def discover_players(root: Path, stride: int = 1, *,
                     arms: list[tuple[str, Path]] | None = None,
                     include_first: bool = True, required_ids: tuple[str, ...] = ()) -> list[Player]:
    if stride < 1:
        raise ValueError('Stride must be positive')
    players = []
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
        previous = 0
        rows = []
        for iteration in range(state['iteration']):
            path = arm/'logs/iterations'/f'{iteration:06d}.json'
            row = json.loads(path.read_text())
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
        for row in rows:
            number = row['iteration']
            if (number != rows[-1]['iteration'] and number % stride
                    and not (include_first and number == rows[0]['iteration'])
                    and f'{name}:{number:08d}' not in required_ids):
                continue
            model, info = model_info(arm/row['model']['path'])
            if info != row['model']:
                raise ValueError(f'Model differs from committed record: {model}')
            players.append(Player(f'{name}:{number:08d}', name, number, row['elapsed_seconds'],
                                  str(model), info['sha256']))
    if len(players) < 2:
        raise ValueError('Need at least two committed models')
    return players


def build_schedule(players: list[Player], neighbors: int, *,
                   final_cross: bool = False, cross_seconds: float | None = None,
                   stride: int = 1) -> list[tuple[str, str]]:
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
            # Only regular checkpoints participate in fixed-time bridges. Once
            # both histories bracket a target, extending them cannot move it.
            stable_a = [p for p in a if p.iteration % stride == 0]
            stable_b = [p for p in b if p.iteration % stride == 0]
            if stable_a and stable_b:
                pair_horizon = min(stable_a[-1].seconds, stable_b[-1].seconds)
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

    def accept(self, event: dict) -> None:
        if event['type'] == 'match_stats':
            # Each invocation (including a resumed partial pair) is retained.
            write_json(self.directory/'performance'/f'{identity(event)}.json', event)
            return
        kind, index = event['type'], event['id']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid event ID')
        moves = event['moves']
        if not isinstance(moves, list) or not moves or any(type(m) is not int or m < 0 for m in moves) or len(set(moves)) != len(moves):
            raise ValueError('Invalid move history')
        if kind == 'opening':
            if index >= self.total // 2 or event['generator'] != index % 2 or not math.isfinite(event['value']):
                raise ValueError('Invalid opening assignment')
            folder = 'openings'
        elif kind == 'game':
            if index >= self.total or event['opening_id'] != index // 2 or event['black_a'] is not (index % 2 == 0) or event['winner'] not in (-1, 0, 1):
                raise ValueError('Invalid game assignment')
            opening = json.loads((self.directory / 'openings' / f'{index//2:08d}.json').read_text())
            if moves[:len(opening['moves'])] != opening['moves'] or len(moves) <= len(opening['moves']):
                raise ValueError('Game does not continue its opening')
            if not math.isfinite(event['seconds']) or event['seconds'] < 0:
                raise ValueError('Invalid game duration')
            folder = 'games'
        else:
            raise ValueError(f'Unknown match event: {kind}')
        path = self.directory / folder / f'{index:08d}.json'
        if path.exists():
            if json.loads(path.read_text()) != event:
                raise ValueError(f'Conflicting completed event: {path}')
        else:
            write_json(path, event)

    def tasks(self, seed: int) -> list[dict]:
        tasks = []
        for index in range(self.total // 2):
            mask = sum(1 << color for color in (0, 1)
                       if not (self.directory / 'games' / f'{2*index+color:08d}.json').exists())
            opening_path = self.directory / 'openings' / f'{index:08d}.json'
            opening_seed = (seed + index * 0x9e3779b97f4a7c15) % (1 << 64)
            moves = None
            if opening_path.exists():
                opening = json.loads(opening_path.read_text())
                self.accept(opening)
                if opening['seed'] != opening_seed or opening['generator'] != index % 2:
                    raise ValueError('Saved opening does not match scheduled seed/generator')
                moves = opening['moves']
            if not mask:
                continue
            tasks.append({'id': index, 'seed': opening_seed, 'generator': index % 2, 'mask': mask, 'moves': moves})
        return tasks

    def games(self) -> list[dict]:
        rows = []
        for path in sorted((self.directory / 'games').glob('*.json')):
            row = json.loads(path.read_text())
            if row['type'] != 'game' or path.name != f"{row['id']:08d}.json":
                raise ValueError(f'Invalid result identity: {path}')
            self.accept(row)
            rows.append(row)
        return rows


def execute_pair(binary: Path, config: Path, players: tuple[Player, Player], store: PairStore,
                 seed: int, cancel: Event | None = None, environ: dict | None = None) -> None:
    if cancel is not None and cancel.is_set():
        return
    completed = len(store.games())
    tasks = store.tasks(seed)
    if not tasks:
        return
    task_path = store.directory / 'tasks.txt'
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(''.join(
        f"{t['id']} {t['seed']} {t['generator']} {t['mask']} " +
        ('-1' if t['moves'] is None else f"{len(t['moves'])} " + ' '.join(map(str, t['moves']))) + '\n'
        for t in tasks))
    with (store.directory / 'native.log').open('a') as log:
        a, b = players
        profile = json.loads((store.directory.parents[1]/'manifest.json').read_text())['config']['match']
        process = subprocess.Popen([str(binary), 'match', '--config', str(config),
                                   '--model', a.model, '--model-id', model_info(a.model)[1]['id'],
                                   '--model-b', b.model, '--model-b-id', model_info(b.model)[1]['id'],
                                   '--device', profile['device'], '--size', str(profile['board_size']),
                                   '--rule', profile['rule'], '--tasks', str(task_path)],
                                   stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True,
                                   env=environ)
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
                event = json.loads(line)
                store.accept(event)
                if event['type'] == 'game':
                    completed += 1
                    print(f"  {a.id} vs {b.id} games={completed}/{store.total}", flush=True)
            code = process.wait()
            if code:
                raise RuntimeError(f'Match exited {code}: {store.directory / "native.log"}')
        finally:
            stopper = Thread(target=stop_process, args=(process,))
            stopper.start()
            try:
                for line in process.stdout:
                    store.accept(json.loads(line))
            finally:
                process.stdout.close()
                stopper.join()
                done.set()
                if watcher is not None:
                    watcher.join()
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
    canvases = set()
    for player in players:
        _, info = model_info(player.model)
        if info['sha256'] != player.sha256:
            raise ValueError(f'Model changed: {player.model}')
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
    return {**config, 'network': {'canvas': canvas}}


def run_arena(output: Path, manifest: dict, binary: Path, samples: int,
              pair_workers: int = 1, environ: dict | None = None) -> dict:
    """Bounded concurrent C++ matches; cancel and flush all children on failure."""
    if pair_workers < 1:
        raise ValueError('Pair workers must be positive')
    players = [Player(**p) for p in manifest['players']]
    native = native_config(manifest['config'], players)
    with run_lock(output/'arena.lock'):
        save_manifest(output/'manifest.json', manifest)
        config_path = output/'resolved.cfg'
        config_path.write_text(native_text(native))
        by_id = {p.id: p for p in players}
        cancel = Event()
        executor = ThreadPoolExecutor(max_workers=pair_workers)
        try:
            futures = []
            for pair in manifest['pairs']:
                store = PairStore(output/'pairs'/pair['id'], manifest['games_per_pair'])
                seed = (manifest['config']['match']['seed'] + int(pair['id'][:16], 16)) % (1 << 64)
                futures.append(executor.submit(execute_pair, binary, config_path,
                               tuple(by_id[p] for p in pair['players']), store, seed, cancel, environ))
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                print(f'completed pairs={count}/{len(futures)}', flush=True)
        finally:
            cancel.set()
            executor.shutdown(wait=True, cancel_futures=True)
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
        players.append(Player(label, label, info['checkpoint']['iteration'], 0, str(path), info['sha256']))
    native = native_config(config, players)
    count = config['match']['games'] if games is None else games
    PairStore(Path(output),count)
    pair = {'id': identity(('a','b')), 'players': ['a','b']}
    manifest = {'version': 1, 'players': [asdict(p) for p in players], 'config': config,
                'binary_sha256': verify_build(binary), 'games_per_pair': count, 'anchor': 'a', 'pairs': [pair]}
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
    manifest = {'version': 1, 'players': [asdict(p) for p in players], 'config': config,
                'binary_sha256': verify_build(binary), 'games_per_pair': args.games, 'anchor': anchor,
                'pairs': [{'id': identity(pair), 'players': list(pair)} for pair in schedule]}
    print(f'players={len(players)} pairs={len(schedule)} games={len(schedule)*args.games} anchor={anchor}', flush=True)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    install_signals()
    run_arena(output, manifest, binary, args.bootstrap_samples)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
