"""Mixed native models, source pairings, immutable game recovery and Elo."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

from etazero import ROOT
from etazero.evaluation import estimate, fit_match, load_games, run_match, summarize_match
from export_muzero import export_components
from katago.train.model_pytorch import Model
from katago.utils import elo
from match_dispatch_worker import worker_command
from test_controller_recovery import wait_until
from test_gatekeeper import gate_config
from test_muzero_model import make_model, component_config, limited_cpu_threads

gpu = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA')


def sample_games():
    plan = {'bots': [{'name': 'A', 'model': '/models/a'}, {'name': 'B', 'model': '/models/b'}],
            'matchups': [[0, 1], [1, 0], [0, 1], [1, 0]]}
    games = [dict(game_index=i, black=pair[0], white=pair[1], outcome=outcome)
             for i, (pair, outcome) in enumerate(zip(plan['matchups'], ['B', 'W', 'D', 'NR']))]
    return plan, games


def test_four_results_and_color_mapping_use_source_elo_covariance_and_los():
    plan, games = sample_games()
    result = estimate(plan, games, anchor='B')
    # Independent grouped counts: A scores 3/4 points, both colors represented.
    reference = elo.GameResultSummary(2., False)
    reference.add_game_record(elo.GameRecord('A', 'B', win=1, draw=1))
    reference.add_game_record(elo.GameRecord('B', 'A', loss=1, draw=1))
    info = reference.get_elos()
    assert result['outcomes'] == {'B': 1, 'W': 1, 'D': 1, 'NR': 1}
    a, b = result['players']
    assert a['elo'] > 0 and b['elo'] == b['stderr'] == 0
    assert a['elo'] == pytest.approx(info.get_elo_difference('A', 'B'))
    assert a['stderr'] == pytest.approx(info.get_approx_elo_difference_stderr('A', 'B'))
    assert result['los']['A']['B'] == pytest.approx(info.get_approx_likelihood_of_superiority('A', 'B'))
    games[-1]['outcome'] = 'D'
    assert estimate(plan, games, anchor='B')['players'] == result['players']


def test_first_player_advantage_does_not_become_model_advantage():
    plan, games = sample_games()
    games = [dict(game, outcome='B') for _ in range(10) for game in games]
    result = estimate(plan, games, first_player_advantage=True)
    assert result['first_player_advantage']['elo'] > 0
    assert result['players'][0]['elo'] == 0
    assert abs(result['players'][1]['elo']) < 1e-7


def test_incomplete_and_disconnected_results_do_not_publish_ratings(tmp_path):
    plan, games = sample_games()
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    (tmp_path / 'game-0.json').write_text(json.dumps(games[0]))
    with pytest.raises(ValueError, match='incomplete'):
        fit_match(tmp_path, tmp_path / 'rating')
    assert not (tmp_path / 'rating').exists()
    plan['bots'] += [{'name': 'C', 'model': '/c'}, {'name': 'D', 'model': '/d'}]
    games.append(dict(black=2, white=3, outcome='D'))
    with pytest.raises(ValueError, match='disconnected'):
        estimate(plan, games)


@pytest.mark.parametrize('prior', [-1, True, float('nan'), float('inf')])
def test_invalid_fit_options_fail_before_starting_match(tmp_path, prior):
    with pytest.raises(ValueError, match='prior_games'):
        run_match(tmp_path / 'output', 'missing.cfg', prior_games=prior)
    assert not (tmp_path / 'output').exists()


def test_fit_only_uses_raw_games_without_binary_models_or_summary(tmp_path):
    root = tmp_path / 'evaluation'
    data = root / 'results'; data.mkdir(parents=True)
    plan, games = sample_games()
    (data / 'plan.json').write_text(json.dumps(plan))
    for game in games:
        (data / f'game-{game["game_index"]}.json').write_text(json.dumps(game))
    originals = {p: p.read_bytes() for p in data.iterdir()}
    result = run_match(root, fit_only=True, binary='/nonexistent', anchor='B')
    assert result['games'] == 4
    assert (root / 'elo.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (root / 'elo.svg').read_text()
    assert 'stderr' in (root / 'elo.csv').read_text()
    assert all(p.read_bytes() == value for p, value in originals.items())
    assert not (data / 'summary.json').exists()


def test_outer_summary_counts_partial_games_by_color_and_preserves_raw_records(tmp_path):
    plan, games = sample_games()
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    # Publication order may differ from schedule order. Ignore uncommitted and
    # unplanned files, and keep a missing game distinct from a draw.
    for index in (0, 2, 3):
        (tmp_path / f'game-{index}.json').write_text(json.dumps(games[index]))
    (tmp_path / 'game-1.json.tmp').write_text('uncommitted')
    (tmp_path / 'game-99.json').write_text('not in the schedule')
    originals = {p: p.read_bytes() for p in tmp_path.iterdir()}
    partial = summarize_match(tmp_path)
    assert partial == dict(completed=False, games_requested=4, games_completed=3, results=[
        dict(black=0, white=1, black_wins=1, white_wins=0, draws=1, no_results=0),
        dict(black=1, white=0, black_wins=0, white_wins=0, draws=0, no_results=1)])
    (tmp_path / 'game-1.json').write_text(json.dumps(games[1]))
    complete = summarize_match(tmp_path)
    assert complete == dict(completed=True, games_requested=4, games_completed=4, results=[
        dict(black=0, white=1, black_wins=1, white_wins=0, draws=1, no_results=0),
        dict(black=1, white=0, black_wins=0, white_wins=1, draws=0, no_results=1)])
    assert json.loads((tmp_path / 'summary.json').read_text()) == complete
    assert all(p.read_bytes() == value for p, value in originals.items())
    (tmp_path / 'game-1.json').write_text(json.dumps(dict(games[1], black=0)))
    with pytest.raises(ValueError, match='committed schedule'):
        summarize_match(tmp_path)


def test_summary_write_failure_warns_without_blocking_native_results_or_elo(tmp_path, monkeypatch):
    from etazero import evaluation
    from test_xiangqi_match import match_config as xiangqi_match_config
    config = xiangqi_match_config(tmp_path / 'match.cfg', ['/dev/null'] * 2, random=True)
    save = evaluation.save_json

    def fail_summary(path, value):
        if path.name == 'summary.json':
            raise OSError('summary write failed')
        save(path, value)

    monkeypatch.setattr(evaluation, 'save_json', fail_summary)
    root = tmp_path / 'evaluation'
    with pytest.warns(RuntimeWarning, match='Could not rebuild match summary.*summary write failed'):
        result = run_match(root, config)
    assert result['games'] == len(load_games(root / 'results')[1]) == 6
    assert result['outcomes'] == dict(B=0, W=0, D=6, NR=0)
    assert (root / 'elo.json').is_file()
    assert not (root / 'results/summary.json').exists()


def match_config(path, models, game='renju', games=6, moves=100):
    gate_config(path, game, moves=moves)
    with path.open('a') as stream:
        stream.write(f'numBots = {len(models)}\nnumGamesTotal = {games}\n'
                     'nnRandSeed0 = ordinary-seed\n'
                     'nnRandSeed1 = muzero-seed\n')
        for i, (model, muzero) in enumerate(models):
            stream.write(f'botName{i} = bot{i}\nnnModelFile{i} = {model}\nuseMuZero{i} = {str(muzero).lower()}\n')
        stream.write('rootSearchAlgorithm1 = gumbel\n')
    return path


def test_native_match_lock_and_unrelated_directory_rejected_before_inference(tmp_path):
    config = match_config(tmp_path / 'match.cfg', [('/dev/null', False)] * 2)
    data = tmp_path / 'results'; data.mkdir()
    command = [str(ROOT / 'build/katago'), 'match', '-config', str(config), '-results-dir', str(data)]
    with (data / '.native.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(worker_command(command), capture_output=True, text=True, timeout=15)
        assert result.returncode != 0 and 'Another match' in result.stdout + result.stderr
    (data / 'keep.txt').write_text('unrelated')
    result = subprocess.run(worker_command(command), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and 'empty results directory' in result.stdout + result.stderr
    assert (data / 'keep.txt').read_text() == 'unrelated'


def export_models(root, game):
    ordinary = Model(component_config(game), 5); ordinary.initialize()
    checkpoint = root / 'ordinary.ckpt'
    torch.save({'config': ordinary.config, 'model': ordinary.state_dict()}, checkpoint)
    exported = root / 'ordinary'
    result = subprocess.run([sys.executable, str(ROOT / 'python/export_model_pytorch.py'),
        '-checkpoint', str(checkpoint), '-export-dir', str(exported), '-model-name', 'ordinary',
        '-filename-prefix', 'model'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    muzero = root / 'muzero'
    export_components(make_model(game), muzero, 'muzero')
    return exported / 'model.bin', muzero


@gpu
@pytest.mark.parametrize('game', ['renju', 'hex', 'go'])
def test_mixed_tournament_source_color_rounds_dedup_and_cached_completion(tmp_path, game):
    ordinary, muzero = export_models(tmp_path, game)
    config = match_config(tmp_path / 'match.cfg', [(ordinary, False), (muzero, True), (muzero, True)],
                          game=game, games=6, moves=400)
    root = tmp_path / 'evaluation'
    result = run_match(root, config, anchor='bot0')
    plan, games = load_games(root / 'results')
    assert result['games'] == len(games) == 6
    assert Counter(tuple(pair) for pair in plan['matchups']) == Counter(
        (black, white) for black in range(3) for white in range(3) if black != white)
    assert all(record['sgf'].startswith('(;') and 'RE[' in record['sgf'] for record in games)
    if game == 'renju':
        assert result['outcomes']['D'] == 6
        assert all(abs(p['elo']) < 1e-6 for p in result['players'])
    native_log = next((root / 'invocations').glob('*.native.log')).read_text()
    assert 'nnRandSeed0 = ordinary-seed' in native_log
    assert 'nnRandSeed1 = muzero-seed' in native_log
    assert 'nnRandSeed2 =' not in native_log  # Same MuZero bundle is loaded once.
    originals = {path: path.read_bytes() for path in (root / 'results').glob('game-*.json')}
    assert json.loads((root / 'results/summary.json').read_text())['completed']
    (root / 'results/summary.json').unlink()  # Simulate interruption after the final per-game commit.
    command = [str(ROOT / 'build/katago'), 'match', '-config', str(config), '-results-dir', str(root / 'results')]
    cached = subprocess.run(worker_command(command), cwd=config.parent, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                            capture_output=True, text=True, timeout=15)
    assert cached.returncode == 0 and 'no inference needed' in cached.stdout
    assert not (root / 'results/summary.json').exists()
    assert all(path.read_bytes() == data for path, data in originals.items())
    # The original on-demand pairing path remains usable without durable results.
    plain_sgfs = tmp_path / 'plain-sgfs'
    plain = subprocess.run(command[:4] + ['-sgf-output-dir', str(plain_sgfs)], cwd=config.parent,
                           capture_output=True, text=True, timeout=60)
    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert sum(path.read_text().count('RE[') for path in plain_sgfs.glob('*.sgfs')) == 6
    config.write_text(config.read_text().replace('maxVisits = 8\n', 'maxVisits = 9\n'))
    changed = subprocess.run(worker_command(command), cwd=config.parent, capture_output=True, text=True, timeout=15)
    assert changed.returncode != 0 and 'configuration differs' in changed.stdout + changed.stderr


@gpu
def test_native_match_resumes_only_unfinished_games_after_signal(tmp_path):
    ordinary, muzero = export_models(tmp_path, 'renju')
    config = match_config(tmp_path / 'match.cfg', [(ordinary, False), (muzero, True)], games=40)
    config.write_text(config.read_text().replace('maxVisits = 8\n', 'maxVisits = 512\n')
                      .replace('numGameThreads = 2\n', 'numGameThreads = 1\n'))
    data = tmp_path / 'results'
    command = [str(ROOT / 'build/katago'), 'match', '-config', str(config), '-results-dir', str(data)]
    with (tmp_path / 'first.log').open('w') as log:
        process = subprocess.Popen(worker_command(command), cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: any(data.glob('game-*.json')) or process.poll() is not None, seconds=60)
            assert process.poll() is None
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    before = {p: p.read_bytes() for p in data.glob('game-*.json')}
    assert 0 < len(before) < 40
    plan = (data / 'plan.json').read_bytes()
    assert not (data / 'summary.json').exists()
    assert not summarize_match(data)['completed']
    with pytest.raises(ValueError, match='incomplete'):
        load_games(data)
    resumed = subprocess.run(worker_command(command), cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    _, games = load_games(data)
    assert len(games) == 40
    assert (data / 'plan.json').read_bytes() == plan
    assert all(path.read_bytes() == content for path, content in before.items())


@gpu
def test_nonterminal_match_is_not_a_draw_or_no_result(tmp_path):
    ordinary, muzero = export_models(tmp_path, 'renju')
    config = match_config(tmp_path / 'match.cfg', [(ordinary, False), (muzero, True)], games=2, moves=1)
    data = tmp_path / 'results'
    result = subprocess.run(worker_command([str(ROOT / 'build/katago'), 'match', '-config', str(config), '-results-dir', str(data)]),
                             cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and 'without a rule result' in result.stdout + result.stderr
    assert not list(data.glob('game-*.json'))
