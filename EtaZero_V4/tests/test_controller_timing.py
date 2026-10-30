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
    config = replace(config, epochs_per_round=1, train_args=tuple(args))
    root = tmp_path / 'run'
    actual_save = controller.save_json

    def fail_pointer(path, value):
        if path == root / 'state.json' and value.get('completed_rounds') == 1:
            raise OSError('injected after model/time commit')
        actual_save(path, value)

    if compiled:
        monkeypatch.setattr(controller, 'save_json', fail_pointer)
        with pytest.raises(OSError, match='after model/time commit'):
            controller.run(root, config, max_seconds=0.000001)
        commit = json.loads((root / 'rounds/000001/commit.json').read_text())
        assert json.loads((root / 'state.json').read_text())['completed_rounds'] == 0
        monkeypatch.setattr(controller, 'save_json', actual_save)

        def no_replay(*args, **kwargs):
            pytest.fail('committed round must be adopted without repeating a source stage')

        monkeypatch.setattr(controller, 'execute', no_replay)
        state = controller.run(root, config, max_seconds=0.000001)
        assert state == commit['state']
        assert not (root / 'rounds/000001/result.json').exists()
    else:
        state = controller.run(root, config, max_seconds=0.000001)
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
    assert state['compile_seconds'] == pytest.approx(sum(record['compile_seconds'] for record in receipts))
    assert (state['compile_seconds'] > 0) == (compiled or muon)
    checkpoint = root / 'train/checkpoint.ckpt'
    modified = checkpoint.stat().st_mtime_ns
    # The time limit is cumulative and may coexist with a larger round limit.
    assert controller.run(root, config, 2, max_seconds=state['elapsed_seconds']) == state
    assert checkpoint.stat().st_mtime_ns == modified
