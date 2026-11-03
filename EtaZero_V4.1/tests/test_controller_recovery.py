"""Recovery at real source publication boundaries, including subprocess lifetime."""
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import fcntl
import pytest
import torch

from etazero import ROOT
from etazero.config import RunConfig
from katago.train.load_model import load_checkpoint

controller = importlib.import_module('etazero.run')
gpu = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and learner')


def wait_until(predicate, seconds=15):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for subprocess boundary')
        time.sleep(.02)


def test_stage_keeps_directory_locked_after_controller_is_killed(tmp_path):
    # The stage is an actual child process, orphaned by SIGKILL of its controller.
    # Completion is controlled by a file so the lock lifetime has no timing race.
    stage = tmp_path / 'stage.py'
    stage.write_text('import os, time\nfrom pathlib import Path\n'
                     f'root = Path({str(tmp_path)!r})\n'
                     '(root / "ready").write_text(str(os.getpid()))\n'
                     'while not (root / "release").exists(): time.sleep(.02)\n')
    parent_code = (
        'import fcntl, sys\nfrom pathlib import Path\nfrom etazero.run import execute\n'
        f'root = Path({str(tmp_path)!r})\n'
        'with (root / ".lock").open("a") as lock:\n'
        ' fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
        ' execute([sys.executable, root / "stage.py"], root / "stage.log", lock_fd=lock.fileno())\n')
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'))
    parent = subprocess.Popen([sys.executable, '-c', parent_code], env=env, stdout=subprocess.DEVNULL)
    try:
        wait_until(lambda: (tmp_path / 'ready').exists())
        parent.kill()
        parent.wait(timeout=10)
        with (tmp_path / '.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (tmp_path / 'release').touch()

            def acquire():
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    return False

            wait_until(acquire)
    finally:
        (tmp_path / 'release').touch()
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=10)


def recovery_config(tmp_path, algorithm, *, fp16=True):
    cfg = tmp_path / 'selfplay.cfg'
    cfg.write_text(f'@include {os.path.relpath(ROOT / "configs/checks/source_go.cfg", tmp_path)}\n'
                   'game = renju\nkomiMean = 0\nbSizes = 3\nmaxMovesPerGame = 9\nbalanceOpeningProb = 0\n'
                   'maxVisits = 8\nuseGraphSearch = false\nsubtreeValueBiasFactor = 0\n'
                   'rootSymmetryPruning = false\ngumbelMaxConsideredActions = 4\n'
                   'cudaUseFP16 = true\n')
    if not fp16:
        with cfg.open('a') as f:
            f.write('inputsUseNHWC = true\neigenUseFP16 = false\n')
    network = {}
    if 'muzero' in algorithm:
        from test_muzero_model import make_model
        network = dict(muzero_network=asdict(make_model('renju').network_config), muzero_unroll_steps=3)
    else:
        network = dict(model_kind='b2c16-fson-mish')
    return RunConfig(selfplay_config=str(cfg), algorithm=algorithm, game='renju',
        canvas=5, batch_size=4, bootstrap_games=4, train_steps=4,
        publish_weights='swa', shuffle_rows_per_file=128, shuffle_keep_rows=512,
        min_rows=1, train_args=('-use-adamw', *(('-use-fp16',) if fp16 else ()), '-no-compile',
            '-lookahead-k', '2', '-swa-period-samples', '8', '-no-longterm-checkpoints'), **network).validate()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('algorithm', ['alphazero', 'gumbel_muzero'])
def test_recover_partial_games_epochs_and_atomic_publications(tmp_path, monkeypatch, algorithm, device):
    flag = 'ETAZERO_GPU_TESTS' if device == 'cuda' else 'ETAZERO_EIGEN_TESTS'
    if os.environ.get(flag) != '1':
        pytest.skip('requires native ' + device + ' backend')
    if device == 'cpu':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    binary = ROOT / ('build/katago' if device == 'cuda' else 'build-eigen/katago')
    config = recovery_config(tmp_path, algorithm, fp16=device == 'cuda')
    root = tmp_path / 'run'
    fault = ['partial_selfplay']
    actual_execute, actual_save = controller.execute, controller.save_json
    calls = []
    requested_games = []

    def execute(command, log, **kwargs):
        stage = Path(log).name.split('.')[0]
        calls.append(stage)
        command = list(command)
        if stage == 'selfplay':
            index = command.index('-max-games-total') + 1
            requested_games.append(int(command[index]))
            if fault[0] == 'partial_selfplay':
                # Commit one actual game, but lose the process completion receipt.
                command[index] = 1
                actual_execute(command, log, **kwargs)
                Path(command[command.index('-result-file') + 1]).unlink()
                raise RuntimeError('injected partial selfplay')
        if stage == 'train':
            assert '-stop-at-samples' in command
            assert '-no-repeat-files' not in command
            assert '-no-export' in command
        actual_execute(command, log, **kwargs)
        if (stage, fault[0]) in [('selfplay', 'receipt'), ('train', 'partial_train'),
                                ('export', 'partial_export'),
                                ('shuffle', 'partial_shuffle')]:
            raise RuntimeError('injected after source commit')

    def save(path, value):
        # Fail between atomic directory publication and the controller marker.
        if Path(path).name == 'state.json' and value.get('active_round') is not None:
            finished = value['round_progress']['done']
            if (fault[0] == 'train_receipt' and 'train' in finished or
                    fault[0] == 'shuffle_publication' and 'shuffle' in finished or
                    fault[0] == 'model_publication' and 'export' in finished):
                raise RuntimeError('injected after atomic publication')
        actual_save(path, value)

    monkeypatch.setattr(controller, 'execute', execute)
    monkeypatch.setattr(controller, 'save_json', save)

    def interrupted(name):
        fault[0] = name
        with pytest.raises(RuntimeError, match='injected'):
            controller.run(root, config, 1, binary=binary)
        state = json.loads((root / 'state.json').read_text())
        cold = name in ('partial_selfplay', 'receipt')
        assert state['completed_rounds'] == (-1 if cold else 0)
        assert state['active_round'] == (0 if cold else 1)

    interrupted('partial_selfplay')
    input_name = 'initial' if config.is_muzero else 'random'
    data = root / 'selfplay/000000' / input_name / 'tdata'
    assert controller.published_games(data) == 1
    initial_shards = {p: p.read_bytes() for p in data.glob('*.npz')}
    interrupted('receipt')
    assert requested_games == [4, 3]
    assert controller.published_games(data) == 4
    assert all(p.read_bytes() == content for p, content in initial_shards.items())
    interrupted('partial_shuffle')
    assert calls.count('selfplay') == 2  # Receipt avoids replay after normal native exit.
    interrupted('shuffle_publication')
    snapshot = root / 'shuffleddata/000001'
    shuffled = {p: p.read_bytes() for p in snapshot.rglob('*') if p.is_file()}
    interrupted('partial_train')
    first = load_checkpoint(root / 'train/checkpoint.ckpt')
    assert first['train_state']['global_step_samples'] > 0
    assert first['train_state']['global_step_samples'] == config.train_steps * config.batch_size
    assert 'lookahead' in first
    assert bool(first['learner_runtime']['scaler']) == (device == 'cuda')
    interrupted('train_receipt')
    second = load_checkpoint(root / 'train/checkpoint.ckpt')
    # An absolute stop target prevents replay when the controller loses the receipt.
    assert second['train_state']['global_step_samples'] == first['train_state']['global_step_samples']
    assert second['swa_model']['n_averaged'] == first['swa_model']['n_averaged']
    assert all(torch.equal(value, first['model'][key]) for key, value in second['model'].items())
    checkpoint_bytes = (root / 'train/checkpoint.ckpt').read_bytes()
    interrupted('partial_export')
    interrupted('model_publication')
    published = sorted((root / 'models').glob('round-000001-s*'))[-1]
    published_bytes = {p: p.read_bytes() for p in published.iterdir()}
    call_count = len(calls)
    fault[0] = None
    result = controller.run(root, config, 1, binary=binary)
    assert len(calls) == call_count  # Adopt the already published candidate.
    assert result['completed_rounds'] == 1 and result['active_round'] is None
    assert (root / 'train/checkpoint.ckpt').read_bytes() == checkpoint_bytes
    assert all(p.read_bytes() == content for p, content in published_bytes.items())
    assert all(p.read_bytes() == content for p, content in shuffled.items())
    assert all(p.read_bytes() == content for p, content in initial_shards.items())
    assert calls.count('train') == 1 and calls.count('shuffle') == 2 and calls.count('export') == 2
    if config.is_muzero:
        assert calls.count('initialize') == calls.count('initial_export') == 1
    discarded = root / 'rounds/000001/discarded'
    assert any(discarded.glob('000001.tmp.*')) and any(discarded.glob('round-000001-s*.tmp.*'))
    report = json.loads((root / 'rounds/000001/result.json').read_text())
    # Injected state-write failures left the controller clock open on disk.
    # The model/batch commits remain recoverable, but the missing wall interval
    # cannot be reconstructed from source artifacts or reported as exact time.
    assert not report['timing_valid']
    assert report['timing_basis'] == 'observed_wall_incomplete'
    assert report['stage_attempts']['train'] == 1
    assert controller.run(root, config, 1, binary=binary) == result


@gpu
def test_native_interruption_receipt_does_not_claim_quota_completed(tmp_path):
    config = recovery_config(tmp_path, 'alphazero')
    models = tmp_path / 'models'; models.mkdir()
    output = tmp_path / 'data'; output.mkdir()
    receipt = tmp_path / 'completion.json'
    log = tmp_path / 'selfplay.log'
    with log.open('w') as stream:
        process = subprocess.Popen([str(ROOT / 'build/katago'), 'selfplay', '-config', config.selfplay_config,
            '-models-dir', str(models), '-output-dir', str(output), '-max-games-total', '1000000',
            '-result-file', str(receipt)], stdout=stream, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: 'starting game on new neural net' in log.read_text(), seconds=30)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert json.loads(receipt.read_text()) == {'games_requested': 1000000, 'completed': False}
