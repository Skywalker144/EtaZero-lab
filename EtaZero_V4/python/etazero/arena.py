"""Committed model discovery and source-native historical tournaments.

Time sampling and graph construction are adapted from V3.4's arena. Search,
pair shuffling, games, and Elo are executed by the in-tree KataGo programs.
"""
from dataclasses import asdict, dataclass
import fcntl
from itertools import combinations
import json
import math
import os
from pathlib import Path
import uuid

from .evaluation import run_match, validate_fit_options, validate_model_window
from .run import save_json


@dataclass(frozen=True)
class Player:
    id: str
    arm: str
    iteration: int
    seconds: float
    model: str
    game: str
    muzero: bool
    sample_horizon: float | None = None


def discover_players(root, stride=1, *, arms=None, include_first=True, required_ids=(), sample_seconds=None):
    if type(stride) is not int or stride < 1:
        raise ValueError('Stride must be a positive integer')
    if sample_seconds is not None and (not math.isfinite(sample_seconds) or sample_seconds <= 0):
        raise ValueError('Sample seconds must be finite and positive')
    root = Path(root).resolve()
    if arms is None:
        paths = [root] if (root / 'state.json').is_file() else sorted(root.iterdir())
        arms = [(path.name, path) for path in paths if (path / 'state.json').is_file()]
    if not arms or len({name for name, _ in arms}) != len(arms):
        raise ValueError('Need nonempty, uniquely named training runs')
    players = []
    for name, directory in arms:
        directory = Path(directory).resolve()
        config = json.loads((directory / 'config.json').read_text())
        state = json.loads((directory / 'state.json').read_text())
        # Snapshot the root pointer once. A concurrently training controller may
        # publish more rounds afterwards; those belong to a later evaluation.
        previous, rows, seen_models = 0., [], set()
        last = None
        first = 0 if (directory / 'rounds/000000').exists() else 1
        for number in range(first, state['completed_rounds'] + 1):
            path = directory / 'rounds' / f'{number:06d}' / 'commit.json'
            last = json.loads(path.read_text())['state']
            seconds = last['elapsed_seconds']
            if (last['completed_rounds'] != number or last['timing_valid'] is not True or
                    last['timing_basis'] != 'committed_wall_excluding_compile' or
                    type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < previous):
                raise ValueError(f'Invalid committed training time; cannot compare by time: {path}')
            previous = seconds
            accepted = last['accepted_model']
            if accepted:
                model = directory / accepted
                if 'muzero' not in config['algorithm']:
                    model /= 'model.bin'
                model = str(model.resolve())
                if model not in seen_models:
                    rows.append(Player(f'{name}:{number:08d}', name, number, seconds, model,
                                       config['game'], 'muzero' in config['algorithm'],
                                       state['elapsed_seconds'] if sample_seconds is not None else None))
                    seen_models.add(model)
        if (last is None or previous != state['elapsed_seconds'] or
                last['accepted_model'] != state['accepted_model'] or not state['timing_valid']):
            raise ValueError(f'Committed model/time does not match state: {directory}')
        if not rows:
            raise ValueError(f'Run has no committed accepted models: {directory}')
        if sample_seconds is None:
            selected = {i for i, player in enumerate(rows) if player.iteration % stride == 0}
            selected.add(len(rows) - 1)
        else:
            times = [player.seconds for player in rows]
            selected = set(time_sample_indices(times, sample_seconds, horizon=previous))
            if not selected or times[-1] - times[max(selected)] > sample_seconds / 2:
                selected.add(len(rows) - 1)
        if include_first:
            selected.add(0)
        for i, player in enumerate(rows):
            if i not in selected and player.id not in required_ids:
                continue
            model = Path(player.model)
            if player.muzero:
                manifest = json.loads((model / 'model.json').read_text())
                if manifest['game'] != player.game:
                    raise ValueError(f'MuZero model game differs from committed run: {model}')
            elif not model.is_file():
                raise ValueError(f'Missing committed model: {model}')
            players.append(player)
    if len(players) < 2 or len({p.id for p in players}) != len(players):
        raise ValueError('Need at least two uniquely identified committed models')
    if len({p.game for p in players}) != 1:
        raise ValueError('Arena participants must use the same game')
    return players


def make_plan(data, *, stride=1, sample_seconds=None, neighbors=1, cross_seconds=None,
              final_cross=False, games_per_pair=40, anchor=None, include_first=True, arms=None):
    if type(games_per_pair) is not int or games_per_pair < 1:
        raise ValueError('Games per pair must be a positive integer')
    players = discover_players(data, stride, arms=arms, include_first=include_first,
                               required_ids=() if anchor is None else (anchor,), sample_seconds=sample_seconds)
    pairs = build_schedule(players, neighbors, final_cross=final_cross, cross_seconds=cross_seconds,
                           stride=stride, sample_seconds=sample_seconds)
    anchor = players[0].id if anchor is None else anchor
    if anchor not in {p.id for p in players}:
        raise ValueError(f'Anchor is not a committed participant: {anchor}')
    return dict(players=[asdict(player) for player in players], pairs=[list(pair) for pair in pairs],
                games_per_pair=games_per_pair, anchor=anchor,
                sampling=dict(stride=stride, sample_seconds=sample_seconds, neighbors=neighbors,
                              cross_seconds=cross_seconds, final_cross=final_cross, include_first=include_first))


def match_overlay(plan):
    players = plan['players']
    indices = {player['id']: index for index, player in enumerate(players)}
    values = dict(numBots=len(players), game=players[0]['game'], includeBots='0',
                  extraPairs=','.join(f'{indices[a]}-{indices[b]}' for a, b in plan['pairs']),
                  extraPairsAreOneSidedBW=False, numGamesPerPair=plan['games_per_pair'],
                  numGamesTotal=plan['games_per_pair'] * len(plan['pairs']))
    for index, player in enumerate(players):
        values.update({f'botName{index}': player['id'], f'nnModelFile{index}': player['model'],
                       f'useMuZero{index}': player['muzero']})
    lines = []
    for key, value in values.items():
        if isinstance(value, str):
            if any(ord(char) < 32 for char in value):
                raise ValueError(f'Control characters are not supported in native config values: {key}')
            value = json.dumps(value, ensure_ascii=False)
        elif type(value) is bool:
            value = str(value).lower()
        lines.append(f'{key} = {value}\n')
    return ''.join(lines)


def plot_timeline(plan, ratings, output):
    import matplotlib as mpl
    from .plotting import THEME, COLORS, _figure
    indexed = {player['name']: player for player in ratings['players']}
    arms = {}
    for player in plan['players']:
        arms.setdefault(player['arm'], []).append(player)
    with mpl.rc_context(THEME):
        figure, (axis,) = _figure(['Committed training time'], [('EtaZero controller training time (h)', 'Relative Elo')],
                                  'EtaZero V4 · historical models', columns=1, figsize=(10, 7))
        from matplotlib.ticker import MaxNLocator
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        for index, (arm, players) in enumerate(arms.items()):
            players.sort(key=lambda player: player['iteration'])
            axis.errorbar([p['seconds'] / 3600 for p in players], [indexed[p['id']]['elo'] for p in players],
                          yerr=[indexed[p['id']]['stderr'] for p in players], label=arm,
                          color=COLORS[index % len(COLORS)], marker='None', capsize=3)
        axis.legend()
        figure.supxlabel(f'Anchor: {ratings["anchor"]} · error bars: approximately one standard error\n'
                         'Points use actual commit times; connecting lines do not measure intermediate strength.', fontsize=9)
        try:
            for extension in ('png', 'svg'):
                destination = output / f'elo.{extension}'
                temporary = destination.with_name(destination.name + '.tmp.' + uuid.uuid4().hex)
                try:
                    figure.savefig(temporary, format=extension, dpi=160, bbox_inches='tight')
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            figure.clear()


def run_arena(output, config=None, *, data=None, fit_only=False, dry_run=False, binary=None,
              prior_games=2., first_player_advantage=False, max_loaded_models=None, **selection):
    validate_fit_options(prior_games, first_player_advantage)
    validate_model_window(max_loaded_models)
    if fit_only and max_loaded_models is not None:
        raise ValueError('fit-only does not load models or take a residency window')
    output = Path(output).resolve()
    if fit_only:
        if dry_run or config is not None or data is not None or selection:
            raise ValueError('--fit-only uses the saved arena plan; do not pass new selection/config options')
        plan = json.loads((output / 'plan.json').read_text())
    else:
        if data is None or config is None:
            raise ValueError('arena requires --data and --config, or --fit-only')
        config = Path(config).resolve()
        if not config.is_file():
            raise ValueError(f'Missing source match config: {config}')
        plan = make_plan(data, **selection)
        plan['config'] = str(config)
        overlay = match_overlay(plan)
        if dry_run:
            return dict(plan, execution=dict(max_loaded_models=max_loaded_models))
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another arena controller is using this output directory') from error
        if not fit_only:
            path = output / 'plan.json'
            if path.exists():
                if json.loads(path.read_text()) != plan:
                    raise ValueError('Arena participants or schedule changed; use a new output directory')
            else:
                if any(p.name != '.lock' for p in output.iterdir()):
                    raise ValueError('A new arena requires an empty output directory')
                save_json(path, plan)
            extra = output / 'participants.cfg'
            if extra.exists():
                if extra.read_text() != overlay:
                    raise ValueError('Generated arena config differs from its saved plan')
            else:
                temporary = extra.with_suffix('.tmp')
                temporary.write_text(overlay)
                os.replace(temporary, extra)
        ratings = run_match(output / 'match', config, binary=binary, fit_only=fit_only,
                            anchor=plan['anchor'], prior_games=prior_games, first_player_advantage=first_player_advantage,
                            extra_config=None if fit_only else extra, inherited_fds=(lock.fileno(),),
                            max_loaded_models=max_loaded_models)
        history = {p['id']: p for p in plan['players']}
        ratings['players'] = [dict(player, **{key: history[player['name']][key]
                                            for key in ('arm', 'iteration', 'seconds')}) for player in ratings['players']]
        save_json(output / 'elo.json', ratings)
        plot_timeline(plan, ratings, output)
        return ratings


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
