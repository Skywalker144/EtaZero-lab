"""Committed histories, measured time sampling, and source-native tournaments."""
from collections import Counter, defaultdict
from dataclasses import replace
import json
import os

import pytest

from etazero.arena import discover_players, make_plan, run_arena, time_sample_indices
from etazero.evaluation import load_games
from etazero.run import run
from test_controller_recovery import recovery_config
from test_gatekeeper import gate_config

gpu = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA training and match')


def history(root, times, *, accepted=None, valid=True, game='renju'):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(dict(game=game, algorithm='alphazero')))
    for number, seconds in enumerate(times, 1):
        model = f'models/{number}' if accepted is None else accepted[number - 1]
        if model:
            directory = root / model
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'model.bin').touch()
        state = dict(completed_rounds=number, elapsed_seconds=seconds, accepted_model=model,
                     timing_valid=valid, timing_basis='committed_wall_excluding_compile')
        directory = root / 'rounds' / f'{number:06d}'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'commit.json').write_text(json.dumps(dict(state=state)))
    (root / 'state.json').write_text(json.dumps(state))
    return root


def test_discovery_uses_commits_first_acceptance_and_ignores_uncommitted_candidates(tmp_path):
    root = history(tmp_path / 'a', [10, 20, 30, 40, 50],
                   accepted=['models/1', 'models/2', 'models/2', 'models/4', 'models/4'])
    # No metrics or source selfplay config is required for discovery.
    uncommitted = root / 'models/6'; uncommitted.mkdir(); (uncommitted / 'model.bin').touch()
    players = discover_players(root)
    assert [p.iteration for p in players] == [1, 2, 4]
    assert [p.seconds for p in players] == [10, 20, 40]
    assert all(p.sample_horizon is None for p in players)
    assert all(p.sample_horizon == 50 for p in discover_players(root, sample_seconds=10))
    assert [p.iteration for p in discover_players(root, stride=3)] == [1, 4]
    old_plan = make_plan(root)
    history(root, [10, 20, 30, 40, 50, 60],
            accepted=['models/1', 'models/2', 'models/2', 'models/4', 'models/4', 'models/4'])
    assert make_plan(root) == old_plan


def test_time_grid_earlier_ties_actual_time_tail_and_required_anchor(tmp_path):
    root = history(tmp_path / 'a', [5, 15, 21, 34, 38])
    assert time_sample_indices([5, 15, 21, 34, 38], 10) == [0, 2, 3]
    selected = discover_players(root, sample_seconds=10, include_first=False)
    assert [p.seconds for p in selected] == [5, 21, 34]  # Tail is within half an interval.
    pinned = discover_players(root, sample_seconds=10, include_first=False, required_ids=('a:00000002',))
    assert [p.seconds for p in pinned] == [5, 15, 21, 34]
    history(tmp_path / 'b', [7, 25, 44])
    plan = make_plan(tmp_path, sample_seconds=10, cross_seconds=20, neighbors=1, games_per_pair=3)
    assert plan['games_per_pair'] == 3
    assert ['a:00000003', 'b:00000002'] in plan['pairs']
    assert all(p['seconds'] not in (10, 20, 30, 40) for p in plan['players'])


@pytest.mark.parametrize('defect', ['invalid', 'nan', 'backwards', 'state', 'model', 'game'])
def test_invalid_equal_time_evidence_is_rejected(tmp_path, defect):
    root = history(tmp_path / 'a', [10, 20])
    if defect == 'invalid':
        history(root, [10, 20], valid=False)
    elif defect == 'nan':
        history(root, [10, float('nan')])
    elif defect == 'backwards':
        history(root, [10, 9])
    elif defect == 'state':
        state = json.loads((root / 'state.json').read_text()); state['elapsed_seconds'] = 21
        (root / 'state.json').write_text(json.dumps(state))
    elif defect == 'model':
        (root / 'models/1/model.bin').unlink()
    else:
        history(tmp_path / 'b', [10, 20], game='hex')
    with pytest.raises(ValueError):
        discover_players(tmp_path)


def test_dry_run_does_not_create_output_and_rejects_disconnected_graph(tmp_path):
    data = tmp_path / 'data'
    history(data / 'a', [10, 20]); history(data / 'b', [100, 200])
    profile = gate_config(tmp_path / 'match.cfg')
    output = tmp_path / 'output'
    plan = run_arena(output, profile, data=data, dry_run=True, final_cross=True, games_per_pair=1)
    assert len(plan['players']) == 4 and not output.exists()
    with pytest.raises(ValueError, match='connected'):
        make_plan(data, cross_seconds=1000)
    assert not output.exists()


@gpu
def test_real_training_histories_share_source_arena_and_fit_without_training_inputs(tmp_path):
    data = tmp_path / 'data'
    for name, algorithm in [('a', 'alphazero'), ('b', 'gumbel_muzero')]:
        cfgdir = tmp_path / ('config-' + name); cfgdir.mkdir()
        config = replace(recovery_config(cfgdir, algorithm), auto_plot=False)
        state = run(data / name, config, 2)
        assert state['timing_valid'] and state['completed_rounds'] == 2
    check_trained_arena(tmp_path)


def check_trained_arena(tmp_path, output=None):
    data = tmp_path / 'data'
    # Both bot labels and model paths must survive source config quoting.
    (data / 'a').rename(data / 'a # ordinary')
    profile = gate_config(tmp_path / 'match.cfg', moves=100)
    output = tmp_path / 'arena' if output is None else output
    result = run_arena(output, profile, data=data, games_per_pair=3, final_cross=True)
    plan = json.loads((output / 'plan.json').read_text())
    native, games = load_games(output / 'match/results')
    counts, colors = Counter(), defaultdict(Counter)
    for game in games:
        a, b = native['bots'][game['black']]['name'], native['bots'][game['white']]['name']
        pair = tuple(sorted((a, b)))
        counts[pair] += 1; colors[pair][a] += 1
    assert counts == Counter({tuple(pair): 3 for pair in plan['pairs']})
    assert all(sorted(count.values()) == [1, 2] for count in colors.values())
    assert result['games'] == len(plan['pairs']) * 3
    assert all(p['seconds'] > 0 for p in result['players'])
    assert (output / 'elo.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (output / 'elo.svg').read_text()
    saved = {p: p.read_bytes() for p in (output / 'match/results').glob('game-*.json')}
    again = run_arena(output, profile, data=data, games_per_pair=3, final_cross=True)
    assert result == again
    assert all(p.read_bytes() == value for p, value in saved.items())
    with pytest.raises(ValueError, match='changed'):
        run_arena(output, profile, data=data, games_per_pair=4, final_cross=True)
    # Saved-time plots and ratings must remain usable after moving the runs.
    data.rename(tmp_path / 'archived-training'); profile.unlink()
    assert run_arena(output, fit_only=True, binary='/missing') == result
