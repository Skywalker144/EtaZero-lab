"""Real signal delivery, drained stages and checkpoint-backed observation history."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from etazero import ROOT
from etazero.observations import learner_log, selfplay_metrics
from katago.train.metric_history import checkpoint_offsets, record_resume
from test_controller_recovery import recovery_config, wait_until, load_checkpoint


@pytest.mark.parametrize('sig,group', [(signal.SIGINT, True), (signal.SIGTERM, False)])
def test_controller_waits_for_drained_stage_and_preserves_time(tmp_path, sig, group):
    config = replace(recovery_config(tmp_path, 'alphazero'), auto_plot=False)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config.serializable()))
    stage = tmp_path / 'stage'
    ready, drained = tmp_path / 'ready', tmp_path / 'drained'
    stage.write_text(f'#!{sys.executable}\nimport signal,time\nfrom pathlib import Path\n'
        'stop=False\ndef request(*_):\n global stop\n stop=True\n'
        'signal.signal(signal.SIGINT,request)\n'
        f'Path({str(ready)!r}).touch()\n'
        'while not stop: time.sleep(.01)\n'
        'time.sleep(.6)\n'
        f'Path({str(drained)!r}).touch()\n')
    stage.chmod(0o755)
    root = tmp_path / 'run'
    code = ('from etazero.config import load_config\nimport etazero.run as c\n'
            'c.snapshot_source=lambda p:None\n'
            f'config=load_config({str(config_path)!r})\n'
            f'try: c.run({str(root)!r},config,0,binary={str(stage)!r})\n'
            'except KeyboardInterrupt: raise SystemExit(130)\n')
    with (tmp_path / 'controller.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT / 'python')), stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_until(ready.exists)
            if group:
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
            assert process.wait(timeout=10) == 130
            assert drained.exists()
            state = json.loads((root / 'state.json').read_text())
            progress = state['round_progress']
            assert progress['timing_valid'] and state['timing_valid']
            assert progress['wall_seconds'] >= .6
            assert progress['running'] is None and not progress['clock_running']
            assert state['completed_rounds'] == -1 and progress['done'] == []
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def test_recovery_filters_only_uncommitted_suffix_and_preserves_original_bytes(tmp_path):
    train, val = tmp_path / 'metrics_train.json', tmp_path / 'metrics_val.json'
    with train.open('a') as a, val.open('a') as b:
        a.write('{"nsamp":100,"loss":1}\n')
        b.write('{"nsamp_train":100,"loss":2}\n')
        offsets = checkpoint_offsets(a, b)
        a.write('{"nsamp":200,"loss":999}\n')
        # Validation at the same sample count can also belong to an abandoned attempt.
        b.write('{"nsamp_train":100,"loss":999}\n')
    original = [p.read_bytes() for p in (train, val)]
    with train.open('a') as a, val.open('a') as b:
        record_resume(tmp_path, offsets, 100, a, b)
        a.write('{"nsamp":200,"loss":3}\n')
        b.write('{"nsamp_train":200,"loss":4}\n')
    assert learner_log(train) == [dict(nsamp=100, loss=1), dict(nsamp=200, loss=3)]
    assert learner_log(val) == [dict(nsamp_train=100, loss=2), dict(nsamp_train=200, loss=4)]
    assert all(p.read_bytes().startswith(content) for p, content in zip((train, val), original))
    # A second crash before another checkpoint abandons both attempts' suffixes.
    with train.open('a') as a, val.open('a') as b:
        record_resume(tmp_path, offsets, 100, a, b)
        a.write('{"nsamp":200,"loss":5}\n')
    assert learner_log(train) == [dict(nsamp=100, loss=1), dict(nsamp=200, loss=5)]
    assert learner_log(val) == [dict(nsamp_train=100, loss=2)]


def test_existing_checkpoint_without_log_positions_uses_saved_sample_boundary(tmp_path):
    path = tmp_path / 'metrics_train.json'
    path.write_text('{"nsamp":100,"loss":1}\n{"nsamp":200,"loss":999}\n')
    with path.open('a') as stream:
        record_resume(tmp_path, None, 100, stream)
        stream.write('{"nsamp":200,"loss":2}\n')
    assert learner_log(path) == [dict(nsamp=100, loss=1), dict(nsamp=200, loss=2)]


def test_corrupt_recovery_evidence_leaves_metrics_missing(tmp_path):
    path = tmp_path / 'metrics_train.json'
    path.write_text('{"nsamp":100,"loss":999}\n')
    (tmp_path / 'metrics_recovery.jsonl').write_text('{partial')
    with pytest.warns(RuntimeWarning, match='metric log unavailable'):
        assert learner_log(path) == []


def test_stop_before_learner_launch_does_not_require_a_compiler_receipt(tmp_path, monkeypatch):
    import etazero.run as controller
    from etazero.stopping import StopRequested
    config = replace(recovery_config(tmp_path, 'alphazero'), auto_plot=False)
    root = tmp_path / 'run'
    monkeypatch.setattr(controller, 'snapshot_source', lambda p: None)
    with pytest.raises(RuntimeError, match='Source stage exited'):
        controller.run(root, config, 0, binary='/bin/false')
    state = json.loads((root / 'state.json').read_text())
    state['round_progress'].update(done=['selfplay', 'shuffle'], attempts={},
        sampling=dict(rows=0, games=0), train_start=dict(samples=0, epoch=0),
        training_plan=dict(train_steps=1, files=[]))
    def not_launched(*args, **kwargs):
        raise StopRequested(started=False)
    monkeypatch.setattr(controller, 'execute', not_launched)
    with pytest.raises(StopRequested):
        controller.run_round(root, config, state, Path('/bin/false'))
    progress = json.loads((root / 'state.json').read_text())['round_progress']
    assert progress['timing_valid'] and progress['compile_seconds'] == 0
    assert progress['attempts'] == {} and progress['stage_compile_seconds'] == {}


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and compiled learner')
def test_compiled_controller_stop_keeps_checkpoint_receipt_and_resumes_budget(tmp_path):
    import etazero.run as controller
    config = replace(recovery_config(tmp_path, 'alphazero'), bootstrap_games=64,
                     train_steps=12, auto_plot=False)
    config = replace(config, train_args=tuple(arg for arg in config.train_args if arg != '-no-compile'))
    root = tmp_path / 'run'
    cold = controller.run(root, config, 0)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config.serializable()))
    log_path = root / 'rounds/000001/logs/train.log'
    with (tmp_path / 'controller.log').open('w') as output:
        process = subprocess.Popen([sys.executable, '-m', 'etazero', 'run', '--config', str(config_path),
            '--run-dir', str(root), '--iterations', '1'], start_new_session=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT / 'python')), stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: log_path.exists() and 'Beginning synchronous epoch!' in log_path.read_text(), seconds=60)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=180) == 130
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    interrupted = json.loads((root / 'state.json').read_text())
    progress = interrupted['round_progress']
    assert progress['timing_valid'] and not progress['clock_running'] and progress['running'] is None
    checkpoint = load_checkpoint(root / 'train/checkpoint.ckpt')
    saved = checkpoint['train_state']['global_step_samples']
    target = progress['train_start']['samples'] + progress['training_plan']['train_steps'] * config.batch_size
    assert 0 < saved < target
    assert checkpoint['metrics_log_offsets']
    receipt = json.loads((root / 'rounds/000001/timing/train.1.json').read_text())
    assert receipt['timing_valid'] and receipt['compile_seconds'] > 0
    time.sleep(.2)  # Restart downtime must not be added by the round clock.
    completed = controller.run(root, config, 1)
    assert completed['timing_valid'] and completed['total_samples'] == target
    report = json.loads((root / 'rounds/000001/result.json').read_text())
    assert report['stage_attempts']['train'] == 2
    assert report['round_timing']['wall_seconds'] >= progress['wall_seconds']
    receipts = [json.loads(p.read_text()) for p in (root / 'rounds/000001/timing').glob('train.*.json')]
    assert completed['compile_seconds'] == pytest.approx(cold['compile_seconds'] + sum(r['compile_seconds'] for r in receipts))
    assert completed['elapsed_seconds'] == pytest.approx(completed['wall_seconds'] - completed['compile_seconds'])


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA selfplay')
def test_launcher_ctrl_c_drains_native_selfplay_and_keeps_summary(tmp_path):
    config = replace(recovery_config(tmp_path, 'alphazero'), bootstrap_games=1000000, auto_plot=False)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config.serializable()))
    root = tmp_path / 'run'
    native_log = root / 'rounds/000000/logs/selfplay.log'
    with (tmp_path / 'controller.log').open('w') as output:
        process = subprocess.Popen(['bash', str(ROOT / 'scripts/run.sh'), '--config', str(config_path),
            '--run-dir', str(root), '--iterations', '0'], start_new_session=True,
            stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: native_log.exists() and 'starting game on new neural net' in native_log.read_text(), seconds=60)
            os.killpg(process.pid, signal.SIGINT)
            assert process.wait(timeout=30) == 130
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    progress = json.loads((root / 'state.json').read_text())['round_progress']
    assert progress['timing_valid'] and not progress['clock_running']
    receipt = json.loads((root / 'rounds/000000/selfplay-1.json').read_text())
    assert not receipt['completed']
    measured = selfplay_metrics(root / 'selfplay/000000', 1)
    assert measured['seconds'] > 0 and measured['rows'] >= 0
    assert sum(measured[key] for key in ('black_wins', 'white_wins', 'draws', 'no_results', 'truncated')) == measured['games']
