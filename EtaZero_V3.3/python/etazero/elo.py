"""KataGo game-result ratings with equal-time plots."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .plotting import BLUE, GREEN, GREY, ORANGE, RED, THEME
from .storage import atomic_path, write_json


def fit_ratings(games: list[dict], anchor: str, prior_games: float = 2.0,
                estimate_first_player_advantage: bool = False) -> dict:
    """Pinned KataGo likelihood, prior, Hessian covariance and Student-t LOS.

    Relative ratings are a display transformation; the source prior is applied
    before subtracting the anchor. Draw/NR score half a point for match results,
    independently of the four categorical training labels.
    """
    from ._vendor.katago import elo as source
    if not games or not np.isfinite(prior_games) or prior_games < 0:
        raise ValueError('Need completed games and a finite nonnegative Elo prior')
    players = sorted({g[side] for g in games for side in ('black', 'white')})
    if anchor not in players or source.P1_ADVANTAGE_NAME in players:
        raise ValueError('Invalid Elo anchor or reserved player name')
    adjacent = {p: set() for p in players}
    counts = defaultdict(lambda: [0, 0.0])
    records = {p: dict(games=0, wins=0, draws=0, losses=0) for p in players}
    pair_scores = defaultdict(list)
    for g in games:
        black, white, score = g['black'], g['white'], g['score']
        if black == white or score not in (0, 0.5, 1):
            raise ValueError('Invalid game score or opponent')
        adjacent[black].add(white); adjacent[white].add(black)
        counts[black, white][0] += 1
        counts[black, white][1] += score
        a, b = sorted((black, white))
        pair_scores[a, b].append(score if black == a else 1-score)
        for player, result in ((black, score), (white, 1-score)):
            records[player]['games'] += 1
            records[player]['wins' if result == 1 else 'losses' if result == 0 else 'draws'] += 1
    reached, pending = set(), [anchor]
    while pending:
        player = pending.pop()
        if player not in reached:
            reached.add(player); pending.extend(adjacent[player]-reached)
    if reached != set(players):
        raise ValueError('Match graph must be connected before a joint Elo fit')
    data = []
    for (black, white), (n, w) in sorted(counts.items()):
        data.extend(source.likelihood_of_games(black, white, n, w/n, estimate_first_player_advantage))
    for player in players:
        data.extend(source.make_single_player_prior(player, prior_games, 0))
    data.extend(source.make_center_elos_prior(players, 0))
    if estimate_first_player_advantage:
        data.extend(source.make_single_player_prior(source.P1_ADVANTAGE_NAME, (1+prior_games)*2, 0))
    info = source.compute_elos(data, verbose=False)
    rows = []
    for player in players:
        value = info.get_elo_difference(player, anchor)
        stderr = info.get_approx_elo_difference_stderr(player, anchor) if player != anchor else 0.0
        rows.append(dict(player=player, elo=value, stderr=stderr, lower=value-stderr, upper=value+stderr,
                         effective_game_count=info.effective_game_count[player], **records[player]))
    advantage = source.P1_ADVANTAGE_NAME
    return dict(anchor=anchor, games=len(games), ratings=rows,
                first_player_advantage_elo=info.get_elo(advantage) if estimate_first_player_advantage else 0.0,
                first_player_advantage_stderr=info.get_approx_elo_stderr(advantage) if estimate_first_player_advantage else 0.0,
                estimate_first_player_advantage=estimate_first_player_advantage,
                interval_method='KataGo approximate one standard error (relative covariance)',
                prior_games=prior_games,
                likelihood_of_superiority={a: {b: float(info.get_approx_likelihood_of_superiority(a,b))
                                               for b in players} for a in players},
                separated_pairs=[dict(a=a,b=b,games=len(scores),score_a=sum(scores))
                                 for (a,b),scores in pair_scores.items() if sum(scores) in (0,len(scores))])


def write_ratings(output: Path, prior_games: float = 2.0, estimate_first_player_advantage: bool = False) -> dict:
    print('Elo: validating completed games and fitting ratings', flush=True)
    from .arena import load_results
    manifest, games = load_results(output)
    expected = len(manifest['pairs']) * manifest['games_per_pair']
    if len(games) != expected:
        raise ValueError(f'Incomplete schedule: {len(games)}/{expected} games; resume the arena first')
    result = fit_ratings(games, manifest['anchor'], prior_games, estimate_first_player_advantage)
    metadata = {p['id']: p for p in manifest['players']}
    for row in result['ratings']:
        row.update({k: v for k, v in metadata[row['player']].items() if k != 'id'})
    result['config'] = manifest['config']
    write_json(output / 'elo.json', result)
    with atomic_path(output / 'elo.csv') as temporary:
        with temporary.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['ratings'][0]))
            writer.writeheader()
            writer.writerows(result['ratings'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with matplotlib.rc_context(THEME):
        figure, axis = plt.subplots(figsize=(11, 6.5))
        arms = defaultdict(list)
        for row in result['ratings']:
            arms[row['arm']].append(row)
        # Distinct hues first; avoid adjacent dark/light variants of one hue.
        colors = (BLUE, ORANGE, GREEN, '#c678dd', RED,
                  GREY, '#56b6c2', '#e58ac4', '#c69c7b', '#e5c07b')
        markers = ('o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h')
        for index, (arm, rows) in enumerate(sorted(arms.items())):
            rows.sort(key=lambda r: r['seconds'])
            hours = [r['seconds'] / 3600 for r in rows]
            line, = axis.plot(hours, [r['elo'] for r in rows], marker=markers[index % len(markers)],
                              color=colors[index % len(colors)], markersize=4, label=arm)
            axis.fill_between(hours, [r['lower'] for r in rows], [r['upper'] for r in rows],
                              color=line.get_color(), alpha=0.16)
        config = manifest['config']['match']
        axis.set(xlabel='Cumulative committed training wall time (hours)',
                 ylabel=f"Relative Elo ({manifest['anchor']} = 0)",
                 title=f"{config['board_size']}×{config['board_size']} {config['rule'].capitalize()} · "
                       f"{config['visits']} visits · {len(games):,} games")
        axis.axhline(0, color=GREY, linewidth=0.8, alpha=0.5)
        axis.grid(alpha=0.55)
        axis.legend(fontsize=9)
        note = 'Shading: KataGo approximate ±1 standard error; fixed evaluation search for every model.'
        if result['separated_pairs']:
            note += '\nSwept pairings present: rating magnitudes and intervals can be prior-sensitive.'
        figure.text(0.5, 0.015, note, ha='center', fontsize=8)
        figure.tight_layout(rect=(0, 0.065, 1, 1))
        for suffix in ('png', 'svg'):
            with atomic_path(output / f'elo.{suffix}') as temporary:
                figure.savefig(temporary, format=suffix, dpi=160)
        plt.close(figure)
    print(f"Elo: {output / 'elo.png'}", flush=True)
    return result
