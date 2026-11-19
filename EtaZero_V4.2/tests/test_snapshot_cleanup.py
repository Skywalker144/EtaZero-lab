"""Bound derived snapshots without deleting inputs needed by active rounds."""
from dataclasses import replace
import fcntl
import importlib
import json
import os

import pytest

from test_controller_recovery import recovery_config, load_checkpoint

controller = importlib.import_module('etazero.run')


def snapshot(root, number):
    path = root / 'shuffleddata' / f'{number:06d}'
    (path / 'train').mkdir(parents=True)
    (path / 'train/data.npz').write_bytes(b'derived data')
    (path / 'train.json').write_text('{}')
    return path


def test_only_old_committed_snapshots_are_removed(tmp_path):
    for number in (1, 2, 10, 11, 12):
        snapshot(tmp_path, number)
    raw = tmp_path / 'selfplay'
    raw.mkdir()
    (raw / 'original.npz').write_bytes(b'original')
    (tmp_path / 'shuffleddata/000000').symlink_to(raw, target_is_directory=True)
    pending = tmp_path / 'shuffleddata/000013.tmp'
    pending.mkdir()
    state = dict(completed_rounds=10, active_round=11)
    assert controller.prune_shuffle_snapshots(tmp_path, state) == ['000001']
    assert all((tmp_path / 'shuffleddata' / f'{n:06d}').is_dir() for n in (2, 10, 11, 12))
    assert (raw / 'original.npz').read_bytes() == b'original'
    assert pending.is_dir()
    assert state == dict(completed_rounds=10, active_round=11)


def test_explicit_reader_dependencies_survive_cleanup(tmp_path):
    for number in range(1, 6):
        snapshot(tmp_path, number)
    state = dict(completed_rounds=5, active_round=None)
    assert controller.prune_shuffle_snapshots(tmp_path, state, protected={'000001'}) == ['000002', '000003']
    assert (tmp_path / 'shuffleddata/000001/train/data.npz').is_file()


def test_partial_cleanup_failure_warns_and_is_retried(tmp_path, monkeypatch, capsys):
    first = snapshot(tmp_path, 1)
    snapshot(tmp_path, 2)
    snapshot(tmp_path, 3)
    actual = controller.shutil.rmtree
    def fail(path):
        (path / 'train.json').unlink()
        raise OSError('injected removal failure')
    monkeypatch.setattr(controller.shutil, 'rmtree', fail)
    state = dict(completed_rounds=3, active_round=None)
    assert controller.prune_shuffle_snapshots(tmp_path, state) == []
    assert 'injected removal failure' in capsys.readouterr().err
    monkeypatch.setattr(controller.shutil, 'rmtree', actual)
    assert controller.prune_shuffle_snapshots(tmp_path, state) == ['000001']
    assert not first.exists()


def existing_run(root, config, *, completed=3, active=None):
    root.mkdir(exist_ok=True)
    state = dict(completed_rounds=completed, active_round=active, timing_valid=True)
    (root / 'state.json').write_text(json.dumps(state))
    (root / 'config.json').write_text(json.dumps(config.serializable()))
    for number in range(1, completed + 1):
        snapshot(root, number)
    return state


def test_restart_cleans_even_when_training_budget_is_already_reached(tmp_path):
    config = replace(recovery_config(tmp_path, 'alphazero'), auto_plot=False)
    state = existing_run(tmp_path, config)
    assert controller.run(tmp_path, config, 3, binary='/bin/false') == state
    assert sorted(p.name for p in (tmp_path / 'shuffleddata').iterdir()) == ['000002', '000003']


def test_committed_round_recovery_cleans_after_advancing_state(tmp_path):
    config = replace(recovery_config(tmp_path, 'alphazero'), auto_plot=False)
    state = existing_run(tmp_path, config, completed=2, active=3)
    snapshot(tmp_path, 3)
    directory = tmp_path / 'rounds/000003'
    directory.mkdir(parents=True)
    committed = {**state, 'completed_rounds': 3, 'active_round': None}
    (directory / 'commit.json').write_text(json.dumps(dict(state=committed)))
    assert controller.run(tmp_path, config, 3, binary='/bin/false') == committed
    assert sorted(p.name for p in (tmp_path / 'shuffleddata').iterdir()) == ['000002', '000003']


def test_busy_run_cannot_be_cleaned_by_another_controller(tmp_path):
    config = replace(recovery_config(tmp_path, 'alphazero'), auto_plot=False)
    existing_run(tmp_path, config)
    with (tmp_path / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another controller'):
            controller.run(tmp_path, config, 3, binary='/bin/false')
    assert (tmp_path / 'shuffleddata/000001/train/data.npz').is_file()


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and learner')
@pytest.mark.parametrize('algorithm', ['alphazero', 'gumbel_muzero'])
def test_cuda_rounds_resume_after_snapshot_pruning(tmp_path, monkeypatch, algorithm):
    config = replace(recovery_config(tmp_path, algorithm), auto_plot=False,
                     bootstrap_games=16, min_rows=64, train_steps=12)
    root = tmp_path / 'run'
    before = controller.run(root, config, 3)
    assert sorted(p.name for p in (root / 'shuffleddata').iterdir()) == ['000002', '000003']
    actual_save = controller.save_json
    def interrupt(path, value):
        actual_save(path, value)
        if (path == root / 'state.json' and value['active_round'] == 4
                and 'shuffle' in value['round_progress']['done']):
            raise RuntimeError('injected after shuffle publication')
    monkeypatch.setattr(controller, 'save_json', interrupt)
    with pytest.raises(RuntimeError, match='injected after shuffle publication'):
        controller.run(root, config, 4)
    assert (root / 'shuffleddata/000004/train.json').is_file()
    monkeypatch.setattr(controller, 'save_json', actual_save)
    after = controller.run(root, config, 4)
    assert after['completed_rounds'] == 4 and after['total_samples'] > before['total_samples']
    assert sorted(p.name for p in (root / 'shuffleddata').iterdir()) == ['000003', '000004']
    checkpoint = load_checkpoint(root / 'train/checkpoint.ckpt')
    assert 'epoch_cursor' not in checkpoint['train_state']
    assert checkpoint['train_state']['export_cycle_counter'] == 4
    assert list((root / 'selfplay/000000/random/tdata').glob('*.npz'))
