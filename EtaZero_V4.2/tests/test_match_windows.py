"""Resident service limits change dispatch order, not native games or fit inputs."""
from collections import Counter
import json
import os
import re
import shutil
import signal
import subprocess
import sys

import pytest

from etazero import ROOT
from etazero import evaluation
from etazero.arena import run_arena
from etazero.evaluation import load_games, run_match
from etazero.match_dispatch import plan_windows
from test_arena import history
from test_controller_recovery import wait_until
from test_match import export_models, match_config, gpu


def window_config(path, models=None, *, random=True, quota=3, visits=8):
    if models is None:
        models = [(path.parent/f'model{i}.bin', False) for i in range(4)]
    config = match_config(path, [*models, models[0]], games=quota*10)
    config.write_text(config.read_text().replace('maxVisits = 8\n',f'maxVisits = {visits}\n')
        .replace('nnRandSeed0 = ordinary-seed\n','nnRandSeed0 = fixed-model-0\n')
        .replace('nnRandSeed1 = muzero-seed\n','nnRandSeed1 = fixed-model-1\n'))
    with config.open('a') as out:
        out.write(f'numGamesPerPair = {quota}\n'
                  f'debugSkipNeuralNet = {str(random).lower()}\nlogToStdout = true\n')
        for i in range(2,4): out.write(f'nnRandSeed{i} = fixed-model-{i}\n')
    return config


def native(config, results=None, capacity=None):
    args = [str(ROOT/'build/katago'), 'match', '-config', str(config)]
    if results is not None: args += ['-results-dir', str(results)]
    if capacity is not None: args.append('-external-dispatch')
    return [sys.executable, str(ROOT/'tests/match_dispatch_worker.py'), str(capacity), *args]


@pytest.mark.parametrize('capacity,expected', [
    (2, [[0, 1, 12], [3, 4], [8]]),
    (3, [[0, 1, 3, 4, 12], [8]]),
    (4, [[0, 1, 3, 4, 8, 12]]),
    (None, [[0, 1, 3, 4, 8, 12]])])
def test_outer_windows_prioritize_shared_and_resident_services(capacity, expected):
    # Missing IDs belong to already committed games; two bots may share service 0.
    games = [[0, 0, 1], [1, 1, 0], [3, 1, 2], [4, 2, 1], [8, 2, 3], [12, 0, 0]]
    assert plan_windows(games, capacity) == expected
    assert plan_windows([], capacity) == []


def check_games(directory, quota):
    plan, games = load_games(directory)
    counts = Counter(tuple(sorted((g['black'],g['white']))) for g in games)
    assert counts == {(a,b):quota for a in range(5) for b in range(a+1,5)}
    for a,b in counts:
        colors = Counter(g['black'] for g in games if {g['black'],g['white']} == {a,b})
        assert abs(colors[a]-colors[b]) <= 1
    assert all(g['outcome'] == 'D' and 'RE[0]' in g['sgf'] for g in games)
    return plan, games


def windows(text):
    return [tuple(map(int,m)) for m in re.findall(
        r'Model window (\d+): resident=(\d+) loaded=(\d+) retained=(\d+) games=(\d+)', text)]


@pytest.mark.parametrize('capacity', [2,3,4])
def test_saved_schedule_counts_colors_and_same_path_services(tmp_path, capacity):
    config = window_config(tmp_path/'match.cfg')
    results = tmp_path/'results'
    result = subprocess.run(native(config, results, capacity), cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    check_games(results, 3)
    stats = windows(result.stdout)
    assert stats and sum(s[4] for s in stats) == 30
    assert all(1 <= s[1] <= capacity and s[1] == s[2]+s[3] for s in stats)
    assert max(s[1] for s in stats) == capacity
    assert sum(s[2] for s in stats) >= 4
    if capacity == 4:
        assert len(stats) == 1 and stats[0][2] == 4  # Five bots, four services.
    else:
        assert len(stats) > 1 and any(s[3] > 0 for s in stats[1:])
    assert all(f'nnRandSeed{i} = fixed-model-{i}' in result.stdout for i in range(4))
    assert 'nnRandSeed4 =' not in result.stdout
    originals = {p:p.read_bytes() for p in results.glob('*.json')}
    cached = subprocess.run(native(config, results, 2), cwd=tmp_path, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                            capture_output=True, text=True, timeout=20)
    assert cached.returncode == 0 and 'no inference needed' in cached.stdout
    assert not windows(cached.stdout)
    assert all(p.read_bytes() == content for p,content in originals.items())


@pytest.mark.parametrize('capacity', [0,1,-1])
def test_invalid_dispatch_capacity_rejected_before_creating_results(tmp_path, capacity):
    config = window_config(tmp_path/'match.cfg')
    result = subprocess.run(native(config,tmp_path/'results',capacity), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and 'at least 2' in result.stdout+result.stderr
    assert not (tmp_path/'results').exists()


def test_window_requires_saved_schedule(tmp_path):
    config = window_config(tmp_path/'match.cfg')
    result = subprocess.run(native(config,capacity=2), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and 'requires results-dir' in result.stdout+result.stderr


@pytest.mark.parametrize('response,message', [
    (dict(windows=[]), 'omitted unfinished'),
    (dict(windows=[list(range(30)) + [0]]), 'repeated game ID'),
    (dict(windows=[list(range(1, 31))]), 'unknown, completed'),
    (dict(windows=[[True], list(range(1, 30))]), 'IDs must be integers'),
    (dict(windows=[[], list(range(30))]), 'nonempty game arrays')])
def test_native_dispatch_rejects_invalid_schedule_before_loading_models(tmp_path, monkeypatch, response, message):
    config = window_config(tmp_path / 'match.cfg')
    root = tmp_path / 'evaluation'
    snapshot = tmp_path / 'source.tar.gz'; snapshot.touch()
    monkeypatch.setattr(evaluation, 'plan_windows', lambda *args: response['windows'])
    with pytest.raises(RuntimeError, match='Native match failed'):
        run_match(root, config, max_loaded_models=2, source_snapshot=snapshot, fit=False)
    log = next(p for p in (root / 'invocations').glob('*.log') if not p.name.endswith('.native.log')).read_text()
    assert message in log
    assert not windows(log) and not list((root / 'results').glob('game-*.json'))


def test_native_results_require_the_outer_schedule_controller(tmp_path):
    config = window_config(tmp_path / 'match.cfg')
    results = tmp_path / 'results'
    command = [str(ROOT / 'build/katago'), 'match', '-config', str(config),
               '-results-dir', str(results), '-external-dispatch']
    result = subprocess.run(command, input='', cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0 and 'closed before receiving the outer schedule' in result.stdout + result.stderr
    assert not (results / 'plan.json').exists() and not list(results.glob('game-*.json'))


def test_outer_planning_failure_releases_native_process_and_results_lock(tmp_path, monkeypatch):
    config = window_config(tmp_path / 'match.cfg')
    root = tmp_path / 'evaluation'
    snapshot = tmp_path / 'source.tar.gz'; snapshot.touch()

    def fail_plan(*args):
        # Native stdout was directly visible in the log before adding the
        # dispatch pipe; it must remain visible while the process is alive.
        stdout_log = next(p for p in (root / 'invocations').glob('*.log') if not p.name.endswith('.native.log'))
        assert 'ETAZERO_MATCH_DISPATCH ' in stdout_log.read_text()
        raise ValueError('dispatch planning failed')

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, 'plan_windows', fail_plan)
        with pytest.raises(ValueError, match='dispatch planning failed'):
            run_match(root, config, max_loaded_models=2, source_snapshot=snapshot, fit=False)
    assert not list((root / 'results').glob('game-*.json'))
    run_match(root, config, max_loaded_models=3, source_snapshot=snapshot, fit=False)
    check_games(root / 'results', 3)


def test_native_executes_the_outer_batches_without_choosing_its_own_windows(tmp_path, monkeypatch):
    config = window_config(tmp_path / 'match.cfg', quota=1)
    root = tmp_path / 'evaluation'
    snapshot = tmp_path / 'source.tar.gz'; snapshot.touch()
    monkeypatch.setattr(evaluation, 'plan_windows', lambda games, capacity: [[g[0]] for g in reversed(games)])
    run_match(root, config, max_loaded_models=2, source_snapshot=snapshot, fit=False)
    check_games(root / 'results', 1)
    report = json.loads(next((root / 'invocations/dispatch').glob('*.json')).read_text())
    assert report['windows'] == [[i] for i in reversed(range(10))]
    stats = windows(next((root / 'invocations').glob('*.native.log')).read_text())
    assert len(stats) == 10 and all(row[4] == 1 for row in stats)


@pytest.mark.parametrize('capacity', [True,0,1,2.5])
def test_invalid_outer_window_rejected_before_output(tmp_path, capacity):
    for func in (run_match, run_arena):
        with pytest.raises(ValueError, match='max_loaded_models'):
            func(tmp_path/'output', max_loaded_models=capacity)
    assert not (tmp_path/'output').exists()


def test_arena_passes_execution_window_without_changing_its_plan(tmp_path):
    data = tmp_path/'data'
    history(data/'a', [10,20,30,40])
    config = window_config(tmp_path/'match.cfg')
    output = tmp_path/'arena'
    first = run_arena(output, config, data=data, games_per_pair=3, max_loaded_models=2)
    native_log = next((output/'match/invocations').glob('*.native.log')).read_text()
    assert windows(native_log) and all(s[1] <= 2 for s in windows(native_log))
    saved = {p:p.read_bytes() for p in (output/'match/results').glob('*.json')}
    plan = (output/'plan.json').read_bytes()
    again = run_arena(output, config, data=data, games_per_pair=3, max_loaded_models=3)
    assert again == first and (output/'plan.json').read_bytes() == plan
    assert all(p.read_bytes() == value for p,value in saved.items())
    assert run_arena(output, fit_only=True) == first


@gpu
def test_real_mixed_models_are_released_between_windows_and_reused(tmp_path):
    ordinary, muzero = export_models(tmp_path, 'renju')
    other = tmp_path/'ordinary2.bin'; shutil.copyfile(ordinary,other)
    other_mu = tmp_path/'muzero2'; shutil.copytree(muzero,other_mu)
    models = [(ordinary,False),(muzero,True),(other,False),(other_mu,True)]
    config = window_config(tmp_path/'match.cfg', models, random=False)
    result = subprocess.run(native(config,tmp_path/'results',2), cwd=tmp_path,
                            capture_output=True, text=True, timeout=90)
    (tmp_path/'gpu.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    check_games(tmp_path/'results',3)
    stats = windows(result.stdout)
    assert len(stats) > 1 and all(s[1] <= 2 for s in stats)
    assert any(s[3] > 0 for s in stats[1:])
    rows = [int(n) for n in re.findall(r'NN rows: (\d+)',result.stdout)]
    assert len(rows) == sum(s[2] for s in stats) and all(n > 0 for n in rows)
    assert 'All cleaned up, quitting' in result.stdout


@gpu
def test_interrupted_windows_resume_same_ids_with_a_different_capacity(tmp_path):
    ordinary, muzero = export_models(tmp_path, 'renju')
    other = tmp_path/'ordinary2.bin'; shutil.copyfile(ordinary,other)
    other_mu = tmp_path/'muzero2'; shutil.copytree(muzero,other_mu)
    config = window_config(tmp_path/'match.cfg', [(ordinary,False),(muzero,True),(other,False),(other_mu,True)],
                           random=False, quota=5, visits=256)
    results = tmp_path/'results'
    with (tmp_path/'interrupted.log').open('w') as log:
        proc = subprocess.Popen(native(config,results,2), cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: len(list(results.glob('game-*.json'))) >= 1 or proc.poll() is not None, seconds=60)
            assert proc.poll() is None
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=30) == 0
        finally:
            if proc.poll() is None: proc.kill(); proc.wait(timeout=10)
    original = {p:p.read_bytes() for p in results.glob('game-*.json')}
    assert 0 < len(original) < 50
    plan = (results/'plan.json').read_bytes()
    resumed = subprocess.run(native(config,results,3), cwd=tmp_path, capture_output=True, text=True, timeout=180)
    (tmp_path/'resumed.log').write_text(resumed.stdout+resumed.stderr)
    assert resumed.returncode == 0, resumed.stdout+resumed.stderr
    check_games(results,5)
    assert (results/'plan.json').read_bytes() == plan
    assert all(p.read_bytes() == content for p,content in original.items())
    assert sum(s[4] for s in windows(resumed.stdout)) == 50-len(original)


@gpu
def test_xiangqi_windows_keep_full_rule_history_with_mixed_services(tmp_path):
    from export_muzero import export_components
    from test_muzero_model import make_model
    from test_xiangqi_match import exported, match_config as xiangqi_config
    ordinary = exported(tmp_path/'ordinary')
    muzero = tmp_path/'muzero'; export_components(make_model('xiangqi'),muzero,'window-xiangqi')
    other = tmp_path/'ordinary2.bin'; shutil.copyfile(ordinary,other)
    other_mu = tmp_path/'muzero2'; shutil.copytree(muzero,other_mu)
    config = xiangqi_config(tmp_path/'match.cfg', [ordinary,muzero,other,other_mu],
                            position='long_check', useGraphSearch='false')
    config.write_text(config.read_text().replace('numGamesTotal = 6\n','numGamesTotal = 18\n')
                       + 'numGamesPerPair = 3\nuseMuZero1 = true\nuseMuZero3 = true\n')
    result = subprocess.run(native(config,tmp_path/'results',2), cwd=tmp_path,
                            capture_output=True, text=True, timeout=90)
    (tmp_path/'xiangqi.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    _,games = load_games(tmp_path/'results')
    assert len(games) == 18 and {g['outcome'] for g in games} == {'W'}
    assert all('XQHIST[e8f8][f9e9][f8e8][e9f9][e8f8][f9e9][f8e8]' in g['sgf'] for g in games)
    stats = windows(result.stdout)
    assert sum(s[4] for s in stats) == 18 and all(s[1] <= 2 for s in stats)
