"""Adaptive game quotas, V3.4 sample credit and real learner continuation."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

from etazero import ROOT
from etazero.replay import plan_sampling, plan_training, commit_replay
from etazero.run import learner_command
from test_controller_recovery import recovery_config, load_checkpoint

controller = importlib.import_module('etazero.run')
gpu = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires CUDA')


def test_game_plans_use_bootstrap_then_pooled_previous_two_rounds(tmp_path):
    config = replace(recovery_config(tmp_path, 'alphazero'), bootstrap_games=11,
                     min_rows=100, batch_size=4, train_steps=30, replay_ratio=3,
                     bootstrap_game_multiplier=1.05)
    replay = dict(rows=20, credit=0, history=[dict(rows=20, games=5)])
    assert plan_sampling(config, replay, 0)['games'] == 11
    assert plan_sampling(config, replay, 1)['games'] == 21  # ceil(80 / 4 * 1.05)
    replay = dict(rows=100, credit=999, history=[dict(rows=20, games=5), dict(rows=80, games=5)])
    assert plan_sampling(config, replay, 2)['games'] == 4  # 120 / 3 / 10
    assert plan_sampling(config, replay, 2)['target_rows'] == 140


def test_training_budget_caps_steps_by_credit_and_per_file_batches(tmp_path):
    config = replace(recovery_config(tmp_path, 'alphazero'), train_steps=20, batch_size=4, replay_ratio=2)
    snapshot = tmp_path / 'snapshot'
    (snapshot / 'train').mkdir(parents=True)
    for name, rows in [('a', 43), ('b', 39)]:
        (snapshot / 'train' / f'{name}.npz').touch()
        (snapshot / 'train' / f'{name}.json').write_text(json.dumps(dict(num_rows=rows)))
    replay = dict(rows=100, credit=3, history=[])
    assert plan_training(config, replay, 1, 0, snapshot)['train_steps'] == 19
    small = plan_training(config, replay, 2, 11, snapshot)
    assert small['train_steps'] == 6
    committed = commit_replay(replay, 2, dict(rows=11, games=3), small, 24)
    assert committed['credit'] == 1
    large = plan_training(config, committed, 3, 100, snapshot)
    assert large['train_steps'] == 19
    assert commit_replay(committed, 3, dict(rows=100, games=10), large, 76)['credit'] == 125


def test_outer_command_owns_exact_budget_and_epoch_size(tmp_path):
    config = recovery_config(tmp_path, 'alphazero')
    command = list(map(str, learner_command(tmp_path, config, stop_samples=100)))
    assert command[command.index('-stop-at-samples') + 1] == '100'
    assert command[command.index('-samples-per-epoch') + 1] == str(config.train_steps * config.batch_size)
    assert '-no-export' in command
    assert '-no-repeat-files' not in command and '-max-train-bucket-per-new-data' not in command


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True])
def test_invalid_replay_ratio_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='replay_ratio'):
        replace(recovery_config(tmp_path, 'alphazero'), replay_ratio=value).validate()


def assert_same_state(left, right, path='state'):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_same_state(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same_state(a, b, f'{path}[{index}]')
    else:
        assert left == right, path


@gpu
@pytest.mark.parametrize('algorithm,subepochs', [('alphazero', 1), ('gumbel_muzero', 1), ('alphazero', 2)])
def test_cuda_pause_restores_fast_slow_rng_scaler_and_file_cursor(tmp_path, algorithm, subepochs, monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    config = replace(recovery_config(tmp_path, algorithm), bootstrap_games=64, train_steps=12,
                     shuffle_rows_per_file=32, auto_plot=False)
    args = list(config.train_args)
    args[args.index('-lookahead-k') + 1] = '6'
    args[args.index('-swa-period-samples') + 1] = '12'
    config = replace(config, train_args=(*args, '-sub-epochs', str(subepochs)))
    root = tmp_path / 'source'
    bootstrap = controller.run(root, config, 0)
    assert bootstrap['total_samples'] == 0 and bootstrap['replay']['history'][0]['games'] == 64
    snapshot = root / 'shuffleddata/000001'
    snapshot.mkdir(parents=True)
    controller.execute([sys.executable, ROOT / 'python/shuffle.py', root / 'selfplay',
        '-out-dir', snapshot / 'train', '-out-tmp-dir', root / 'scratch', '-num-processes', 1,
        '-approx-rows-per-out-file', 32, '-keep-target-rows', 512, '-min-rows', 64,
        *(['-muzero-unroll-steps', config.muzero_unroll_steps] if config.is_muzero else [])], root / 'shuffle.log')
    # Fix the file boundary using real shuffled rows. Native whole-file epoch
    # selection is stochastic; otherwise a 5-batch file can end exactly at the
    # intended mid-epoch pause and correctly reset Lookahead there.
    files = sorted((snapshot / 'train').glob('*.npz'))
    arrays = []
    for path in files:
        with np.load(path) as data:
            arrays.append(dict(data))
    combined = {key: np.concatenate([data[key] for data in arrays])[:64] for key in arrays[0]}
    assert len(combined['globalInputNC']) == 64
    for path in files:
        path.unlink()
    np.savez(snapshot / 'train/fixed.npz', **combined)
    (snapshot / 'train/index.json').write_text(json.dumps(dict(files=[dict(name='fixed.npz', num_rows=64)])))
    if not (root / 'train/checkpoint.ckpt').exists():
        controller.execute([*learner_command(root, config), '-initialize-only'], root / 'initialize.log')
    initial = root / 'train/checkpoint.ckpt'
    branches = []
    for name in ('continuous', 'split'):
        branch = tmp_path / name
        (branch / 'train').mkdir(parents=True)
        shutil.copyfile(initial, branch / 'train/checkpoint.ckpt')
        (branch / 'shuffleddata').symlink_to(root / 'shuffleddata', target_is_directory=True)
        if config.is_muzero:
            shutil.copyfile(root / 'muzero_network.json', branch / 'muzero_network.json')
        branches.append(branch)
    continuous, split = branches
    def train(branch, stop, log):
        command = learner_command(branch, config, stop_samples=stop)
        # Fix kernel choice as well as RNG so this tests exact continuation,
        # independently of cuDNN's default per-process benchmark selection.
        wrapper = ('import torch,sys,runpy; torch.use_deterministic_algorithms(True); '
                   'sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")')
        controller.execute([sys.executable, '-c', wrapper, *command[1:]], branch / log)
    train(continuous, 76, 'train.log')
    train(split, 20, 'first.log')
    first = load_checkpoint(split / 'train/checkpoint.ckpt')
    assert first['lookahead']['counter'] == 5
    assert first['train_state']['epoch_cursor']['batches'] == 5
    assert first['swa_model']['n_averaged'] == 0
    assert first['train_state']['swa_sample_accum'] == 20
    train(split, 28, 'second.log')
    second = load_checkpoint(split / 'train/checkpoint.ckpt')
    assert second['swa_model']['n_averaged'] == 1
    assert second['train_state']['swa_sample_accum'] == 4
    if subepochs == 2:
        train(split, 64, 'subepoch.log')
        boundary = load_checkpoint(split / 'train/checkpoint.ckpt')['train_state']['epoch_cursor']
        assert boundary['subepoch'] == 1 and boundary['files'] is None
    train(split, 76, 'third.log')
    expected, resumed = [load_checkpoint(p / 'train/checkpoint.ckpt') for p in branches]
    for key in ('model', 'optimizer', 'lookahead', 'swa_model', 'train_state', 'learner_runtime'):
        assert_same_state(expected[key], resumed[key], key)


@gpu
def test_bootstrap_fill_and_unready_swa_preserve_training_progress(tmp_path):
    config = replace(recovery_config(tmp_path, 'alphazero'), bootstrap_games=2,
                     min_rows=80, train_steps=7, auto_plot=False)
    args = list(config.train_args)
    args[args.index('-swa-period-samples') + 1] = '100000'
    config = replace(config, train_args=tuple(args))
    root = tmp_path / 'run'
    cold = controller.run(root, config, 0)
    assert cold['replay']['rows'] < 80 and cold['total_samples'] == 0
    first = controller.run(root, config, 1)
    assert first['replay']['rows'] >= 80 and first['total_samples'] == 28
    assert first['accepted_model'] is None
    second = controller.run(root, config, 2)
    assert second['total_samples'] > first['total_samples']
    assert second['accepted_model'] is None
    ckpt = load_checkpoint(root / 'train/checkpoint.ckpt')
    assert ckpt['swa_model']['n_averaged'] == 0
    report = json.loads((root / 'rounds/000002/result.json').read_text())
    assert report['published_weights'] is None and report['gatekeeper']['reason'] == 'swa_not_ready'
    assert second['replay']['credit'] == report['training_plan']['available_credit'] - report['training_steps'] * 4
