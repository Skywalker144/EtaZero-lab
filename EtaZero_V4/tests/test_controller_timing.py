"""Budget boundaries, compiler accounting and committed model/time recovery."""
from dataclasses import replace
import json
import os

import pytest

import etazero.run as controller
from etazero.timing import RoundClock, committed_timing, compiler_seconds
from test_controller_recovery import recovery_config


def test_clock_excludes_restart_downtime_and_counts_each_interval_once():
    progress = {'wall_seconds': 0.0}
    now = iter([10., 13., 18.])
    clock = RoundClock(progress, now=lambda: next(now))
    clock.settle()
    clock.settle()
    assert progress['wall_seconds'] == 8
    resumed = iter([1000., 1002.])
    clock = RoundClock(progress, now=lambda: next(resumed))
    clock.settle()
    assert progress['wall_seconds'] == 10


def test_cumulative_invalid_timing_uses_wall_without_mixing_net_time():
    prior = dict(wall_seconds=100, compile_seconds=30, timing_valid=True)
    current = dict(wall_seconds=50, compile_seconds=10, timing_valid=True)
    valid = committed_timing(prior, current)
    assert valid['elapsed_seconds'] == 110
    assert valid['compile_seconds'] == 40
    current['timing_valid'] = False
    invalid = committed_timing(prior, current)
    assert not invalid['timing_valid']
    assert invalid['elapsed_seconds'] == 150
    assert invalid['timing_basis'] == 'observed_wall_incomplete'
    later = committed_timing(invalid, dict(wall_seconds=20, compile_seconds=5, timing_valid=True))
    assert not later['timing_valid'] and later['elapsed_seconds'] == 170
    current['compile_seconds'] = None
    assert committed_timing(prior, current)['compile_seconds'] is None


@pytest.mark.parametrize('record', [None, {}, {'compile_seconds': -1},
    dict(compile_seconds=True, wall_seconds=5, timing_valid=True),
    dict(compile_seconds=6, wall_seconds=5, timing_valid=True),
    dict(compile_seconds=1, wall_seconds=11, timing_valid=True),
    dict(compile_seconds=float('nan'), wall_seconds=5, timing_valid=True),
    dict(compile_seconds=1, wall_seconds=5, timing_valid=False)])
def test_missing_or_invalid_compiler_receipt_is_not_zero(tmp_path, record):
    path = tmp_path / 'timing.json'
    if record is not None:
        path.write_text(json.dumps(record))
    with pytest.warns(RuntimeWarning, match='equal-time comparisons'):
        assert compiler_seconds(path, 10) is None


def test_compiler_receipt_uses_process_counter_not_stage_wall(tmp_path):
    path = tmp_path / 'timing.json'
    path.write_text(json.dumps(dict(compile_seconds=2, wall_seconds=8, timing_valid=True)))
    assert compiler_seconds(path, 10) == 2


@pytest.mark.parametrize('budget', [0, -1, True, float('nan'), float('inf')])
def test_invalid_time_limit_rejected_before_creating_run(tmp_path, budget):
    config = recovery_config(tmp_path, 'alphazero')
    with pytest.raises(ValueError, match='max_seconds'):
        controller.run(tmp_path / 'run', config, max_seconds=budget)
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('resume', [False, True])
def test_no_limits_runs_until_interrupted(tmp_path, monkeypatch, resume):
    config = recovery_config(tmp_path, 'alphazero')
    root = tmp_path / 'run'
    completed = []

    def run_round(root, config, state, binary, lock_fd):
        if state['completed_rounds'] == 4:
            raise KeyboardInterrupt
        state['completed_rounds'] += 1
        state['elapsed_seconds'] += 3600
        completed.append(state['completed_rounds'])
        controller.save_json(root / 'state.json', state)

    monkeypatch.setattr(controller, 'run_round', run_round)
    monkeypatch.setattr(controller, 'refresh_plots', lambda *args: None)
    if resume:
        assert controller.run(root, config, 2, binary='/bin/false')['completed_rounds'] == 2
    with pytest.raises(KeyboardInterrupt):
        controller.run(root, config, binary='/bin/false')
    assert completed == [0, 1, 2, 3, 4]
    state = json.loads((root / 'state.json').read_text())
    assert state['completed_rounds'] == 4 and state['elapsed_seconds'] == 18000


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and learner')
def test_real_unlimited_training_and_resume(tmp_path, monkeypatch):
    config = replace(recovery_config(tmp_path, 'gumbel_alphazero'),
                     auto_plot=False)
    root = tmp_path / 'run'
    stop_after = [2]

    def interrupt_after_commit(root, config):
        state = json.loads((root / 'state.json').read_text())
        if state['completed_rounds'] == stop_after[0]:
            raise KeyboardInterrupt

    monkeypatch.setattr(controller, 'refresh_plots', interrupt_after_commit)
    with pytest.raises(KeyboardInterrupt):
        controller.run(root, config)
    first = json.loads((root / 'state.json').read_text())
    assert first['completed_rounds'] == 2 and first['total_samples'] > 0
    assert first['accepted_model'] is not None
    stop_after[0] = 3
    with pytest.raises(KeyboardInterrupt):
        controller.run(root, config)
    resumed = json.loads((root / 'state.json').read_text())
    assert resumed['completed_rounds'] == 3 and resumed['total_samples'] > first['total_samples']
    result = json.loads((root / 'rounds/000003/result.json').read_text())
    assert result['input_model'] == first['accepted_model'].split('/')[-1]
    assert resumed['active_round'] is None


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and learner')
@pytest.mark.parametrize('algorithm,compiled,muon', [
    ('alphazero', False, False),
    ('alphazero', False, True),
    ('gumbel_muzero', True, False),
])
def test_real_time_budget_and_model_commit(tmp_path, monkeypatch, algorithm, compiled, muon):
    config = recovery_config(tmp_path, algorithm)
    args = list(config.train_args)
    if compiled:
        args.remove('-no-compile')
    if muon:
        args[args.index('-use-adamw')] = '-use-muon'
    config = replace(config, train_args=tuple(args))
    root = tmp_path / 'run'
    controller.run(root, config, 0)
    actual_save = controller.save_json
    cold = json.loads((root / 'state.json').read_text())
    budget = cold['elapsed_seconds'] + 0.000001

    def fail_pointer(path, value):
        if path == root / 'state.json' and value.get('completed_rounds') == 1:
            raise OSError('injected after model/time commit')
        actual_save(path, value)

    if compiled:
        monkeypatch.setattr(controller, 'save_json', fail_pointer)
        with pytest.raises(OSError, match='after model/time commit'):
            controller.run(root, config, max_seconds=budget)
        commit = json.loads((root / 'rounds/000001/commit.json').read_text())
        assert json.loads((root / 'state.json').read_text())['completed_rounds'] == 0
        monkeypatch.setattr(controller, 'save_json', actual_save)

        def no_replay(*args, **kwargs):
            pytest.fail('committed round must be adopted without repeating a source stage')

        monkeypatch.setattr(controller, 'execute', no_replay)
        state = controller.run(root, config, max_seconds=budget)
        assert state == commit['state']
        assert not (root / 'rounds/000001/result.json').exists()
    else:
        state = controller.run(root, config, max_seconds=budget)
        commit = json.loads((root / 'rounds/000001/commit.json').read_text())
        assert state == commit['state']
        report = json.loads((root / 'rounds/000001/result.json').read_text())
        assert report['round_timing'] == commit['round_timing']
        assert report['elapsed_seconds'] == state['elapsed_seconds']
    assert state['completed_rounds'] == 1 and state['active_round'] is None
    assert state['total_samples'] > 0 and state['timing_valid']
    assert state['wall_seconds'] >= state['elapsed_seconds'] > 0.000001
    assert state['elapsed_seconds'] == pytest.approx(state['wall_seconds'] - state['compile_seconds'])
    receipts = [json.loads(path.read_text()) for path in (root / 'rounds/000001/timing').glob('*.json')]
    assert receipts and all(record['timing_valid'] for record in receipts)
    assert state['compile_seconds'] == pytest.approx(cold['compile_seconds'] + sum(record['compile_seconds'] for record in receipts))
    assert (state['compile_seconds'] > 0) == (compiled or muon)
    checkpoint = root / 'train/checkpoint.ckpt'
    modified = checkpoint.stat().st_mtime_ns
    # The time limit is cumulative and may coexist with a larger round limit.
    assert controller.run(root, config, 2, max_seconds=state['elapsed_seconds']) == state
    assert checkpoint.stat().st_mtime_ns == modified
