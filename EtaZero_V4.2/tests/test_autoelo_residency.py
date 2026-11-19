"""Shared native residency keeps pair-local options, schedules and raw records."""
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
from etazero.autoelo import archive_prefix, run_autoelo
from etazero.evaluation import load_games, run_match, run_match_batch
from etazero.experiment import experiment_plan
from test_arena import history
from test_controller_recovery import wait_until
from test_experiment import plan_for
from test_gatekeeper import gate_config
from test_match import export_models, gpu
from test_match_windows import windows


def profile(path, *, random=True, options=''):
    gate_config(path, moves=100)
    with path.open('a') as out:
        out.write(f'debugSkipNeuralNet = {str(random).lower()}\nlogToStdout = true\n'
                  'rootSearchAlgorithm1 = gumbel\n' + options)
    return path


def pair_tasks(root, models=None, quota=3):
    if models is None:
        models = [(root / f'model{i}.bin', False) for i in range(3)]
    tasks = []
    for index, pair in enumerate(((0, 1), (1, 2), (0, 2))):
        cache = root / f'pair{index}'
        cache.mkdir(exist_ok=True)
        (cache / 'match').mkdir(exist_ok=True)
        extra = cache / 'participants.cfg'
        text = f'numBots = 2\nnumGamesTotal = {quota}\nnumGamesPerPair = {quota}\n'
        for local, player in enumerate(pair):
            model, muzero = models[player]
            text += f'botName{local} = player{player}\nnnModelFile{local} = {model}\nuseMuZero{local} = {str(muzero).lower()}\n'
        extra.write_text(text)
        tasks.append((cache / 'match', extra))
    return tasks


def native(config, tasks, capacity):
    manifest = config.parent / 'pair-jobs.json'
    manifest.write_text(json.dumps([dict(extra_config=os.path.relpath(extra, config.parent),
                                       results_dir=str(output / 'results'), sgf_output_dir=str(output / 'sgfs'))
                                   for output, extra in tasks]))
    return [sys.executable, str(ROOT / 'tests/match_dispatch_worker.py'), str(capacity),
            str(ROOT / 'build/katago'), 'match', '-config', str(config), '-pair-jobs', str(manifest),
            '-external-dispatch']


def check_pairs(tasks, quota):
    for output, _ in tasks:
        plan, games = load_games(output / 'results', quota)
        assert len(games) == quota
        assert len(plan['bots']) == 2
        assert all({g['black'], g['white']} == {0, 1} for g in games)
        counts = Counter(g['black'] for g in games)
        assert abs(counts[0] - counts[1]) <= 1
        assert all(g['outcome'] == 'D' and 'RE[0]' in g['sgf'] for g in games)
        assert sum(len(p.read_text().splitlines()) for p in (output / 'sgfs').glob('*.sgfs')) >= quota


@pytest.mark.parametrize('options,services', [('', 3), ('nnRandSeed = shared\n', 3),
    ('nnRandSeed0 = same\nnnRandSeed1 = same\n', 3),
    ('nnRandSeed0 = first\nnnRandSeed1 = second\n', 4),
    ('cudaUseFP16-0 = true\ncudaUseFP16-1 = false\n', 4),
    ('cudaDeviceToUseModel0 = 0\ncudaDeviceToUseModel1 = 1\n', 4)])
def test_sharing_uses_resolved_pair_local_inference_options(tmp_path, options, services):
    config = profile(tmp_path / 'match.cfg', options=options)
    tasks = pair_tasks(tmp_path)
    result = subprocess.run(native(config, tasks, 4), cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    check_pairs(tasks, 3)
    assert windows(result.stdout) == [(1, services, services, 0, 9)]
    if 'first' in options:
        assert 'nnRandSeed0 = first' in result.stdout and 'nnRandSeed1 = second' in result.stdout
        assert 'nnRandSeed2 =' not in result.stdout


def test_reject_different_pair_profiles_before_opening_results(tmp_path):
    config = profile(tmp_path / 'match.cfg')
    tasks = pair_tasks(tmp_path)
    with tasks[-1][1].open('a') as out:
        out.write('maxVisits = 9\n')
    result = subprocess.run(native(config, tasks, 2), cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0 and 'must share one profile' in result.stdout + result.stderr
    assert all(not (output / 'results').exists() for output, _ in tasks)


def test_identical_paths_within_a_pair_keep_source_model_index_zero(tmp_path):
    config = profile(tmp_path / 'match.cfg', options='nnRandSeed0 = first\nnnRandSeed1 = second\n')
    same = (tmp_path / 'same.bin', False)
    tasks = pair_tasks(tmp_path, [same, same, (tmp_path / 'other.bin', False)])
    result = subprocess.run(native(config, tasks, 4), cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    check_pairs(tasks, 3)
    assert windows(result.stdout) == [(1, 2, 2, 0, 9)]


def test_direct_native_pair_batch_without_outer_dispatch_retains_all_services(tmp_path):
    config = profile(tmp_path / 'match.cfg')
    tasks = pair_tasks(tmp_path)
    # Schedule persistence still uses the outer controller; only bounded
    # residency dispatch is omitted here.
    command = native(config, tasks, None)[:-1]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    check_pairs(tasks, 3)
    assert 'ETAZERO_MATCH_DISPATCH ' not in result.stdout
    assert windows(result.stdout) == [(1, 3, 3, 0, 9)]


def test_single_pair_caches_extend_in_batch_and_keep_old_session_records(tmp_path):
    config = profile(tmp_path / 'match.cfg')
    tasks = pair_tasks(tmp_path)
    snapshot = tmp_path / 'source.tar.gz'
    snapshot.touch()
    # An original single-pair invocation and the batch must resolve exactly the
    # same conditions and retain the old local bot IDs and unmatched color.
    output, extra = tasks[0]
    run_match(output, config, extra_config=extra, extend_pair=True, fit=False, source_snapshot=snapshot)
    original = {p: p.read_bytes() for p in (output / 'results').glob('game-*.json')}
    archive_prefix(output / 'results', tmp_path / 'old-session', 3)
    archive = {p: p.read_bytes() for p in (tmp_path / 'old-session').glob('*.json')}
    tasks = pair_tasks(tmp_path, quota=6)
    run_match_batch(tasks, config, tmp_path / 'session', source_snapshot=snapshot, max_loaded_models=2)
    check_pairs(tasks, 6)
    for pair_output, _ in tasks:
        summary = json.loads((pair_output / 'results/summary.json').read_text())
        assert summary['completed'] and summary['games_completed'] == summary['games_requested'] == 6
        assert sum(row['draws'] for row in summary['results']) == 6
    stats = windows(next((tmp_path / 'session/invocations').glob('*.native.log')).read_text())
    assert sum(w[4] for w in stats) == 15 and any(w[3] for w in stats[1:])
    assert all(p.read_bytes() == value for p, value in original.items())
    assert all(p.read_bytes() == value for p, value in archive.items())
    plan, _ = load_games(output / 'results')
    assert plan['matchups'][3] == list(reversed(plan['matchups'][2]))
    before = {p: p.read_bytes() for out, _ in tasks for p in (out / 'results').glob('*.json')}
    tasks = pair_tasks(tmp_path, quota=2)
    run_match_batch(tasks, config, tmp_path / 'cached', source_snapshot=snapshot,
                    max_loaded_models=3, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    log = next((tmp_path / 'cached/invocations').glob('*.log')).read_text()
    assert 'no inference needed' in log and not windows(log)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_autoelo_uses_one_native_process_and_records_actual_command(tmp_path):
    data = history(tmp_path / 'runs/a', [10, 20, 30])
    config = profile(tmp_path / 'match.cfg')
    output = tmp_path / 'elo'
    first = run_autoelo(output, config, data=data, neighbors=2, games_per_pair=3)
    session = (output / 'latest').resolve()
    records = list((session / 'invocations').glob('*.native.log'))
    assert len(records) == 1
    stats = windows(records[0].read_text())
    assert stats and all(w[1] <= 2 for w in stats) and any(w[3] for w in stats[1:])
    invocation_path = records[0].with_name(records[0].name.replace('.native.log', '.json'))
    invocation = json.loads(invocation_path.read_text())
    assert '-pair-jobs' in invocation['command']
    assert invocation['command'][-1] == '-external-dispatch'
    assert invocation['max_loaded_models'] == 2
    dispatch = json.loads(next((session / 'invocations/dispatch').glob('*.json')).read_text())
    assert sorted(index for batch in dispatch['windows'] for index in batch) == sorted(
        game[0] for game in dispatch['games'])
    assert len(dispatch['windows']) == len(stats)
    for p in (output / 'pair_cache').glob('*/match/invocations/*.json'):
        assert json.loads(p.read_text())['command'] == invocation['command']
    before = {p: p.read_bytes() for p in (session / 'pairs').glob('*/*.json')}
    again = run_autoelo(output, config, data=data, neighbors=2, games_per_pair=3, max_loaded_models=3,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert first == again and (output / 'latest').resolve() == session
    assert all(p.read_bytes() == value for p, value in before.items())
    dry = run_autoelo(output, config, data=data, dry_run=True, max_loaded_models=4)
    assert dry['execution'] == dict(max_loaded_models=4)
    assert 'execution' not in json.loads((session / 'plan.json').read_text())


@pytest.mark.parametrize('capacity', [True, 1, -1, 2.5])
def test_invalid_capacity_rejected_before_output(tmp_path, capacity):
    with pytest.raises(ValueError, match='max_loaded_models'):
        run_autoelo(tmp_path / 'elo', max_loaded_models=capacity)
    assert not (tmp_path / 'elo').exists()


def test_experiment_accepts_residency_in_native_style_profile(tmp_path):
    plan_for(tmp_path)
    profile(tmp_path / 'match.cfg')
    config = tmp_path / 'exp.cfg'
    config.write_text('[experiment]\niterations = 1\n[arms]\na = a.json\n'
                      '[evaluation]\nconfig = match.cfg\nmax_loaded_models = 3\n')
    # Use the run config created by the existing experiment fixture.
    arm = next(iter(json.loads((tmp_path / 'exp.json').read_text())['arms'].values()))
    config.write_text(config.read_text().replace('a = a.json', f'a = {arm}'))
    assert experiment_plan(config, tmp_path / 'work')['evaluation']['max_loaded_models'] == 3


@gpu
def test_mixed_cuda_batch_interruption_and_capacity_change_preserve_prefixes(tmp_path):
    ordinary, muzero = export_models(tmp_path, 'renju')
    other = tmp_path / 'ordinary2.bin'
    shutil.copyfile(ordinary, other)
    models = [(ordinary, False), (muzero, True), (other, False)]
    config = profile(tmp_path / 'match.cfg', random=False, options='nnRandSeed = fixed\n')
    config.write_text(config.read_text().replace('maxVisits = 8\n', 'maxVisits = 256\n'))
    tasks = pair_tasks(tmp_path, models, quota=8)
    with (tmp_path / 'interrupted.log').open('w') as log:
        process = subprocess.Popen(native(config, tasks, 2), cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: bool(list(tmp_path.glob('pair*/match/results/game-*.json'))) or process.poll() is not None,
                       seconds=60)
            assert process.poll() is None
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    original = {p: p.read_bytes() for p in tmp_path.glob('pair*/match/results/game-*.json')}
    plans = {p: p.read_bytes() for p in tmp_path.glob('pair*/match/results/plan.json')}
    assert 0 < len(original) < 24
    result = subprocess.run(native(config, tasks, 3), cwd=tmp_path, capture_output=True, text=True, timeout=120)
    (tmp_path / 'resumed.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    check_pairs(tasks, 8)
    assert all(p.read_bytes() == value for p, value in {**plans, **original}.items())
    assert windows(result.stdout) == [(1, 3, 3, 0, 24 - len(original))]
    assert len(re.findall(r'NN rows: [1-9][0-9]*', result.stdout)) == 3
    # Cached completion does not even open the now unavailable MuZero bundle.
    shutil.rmtree(muzero)
    result = subprocess.run(native(config, tasks, 2), cwd=tmp_path, capture_output=True, text=True, timeout=20,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert result.returncode == 0 and 'no inference needed' in result.stdout
