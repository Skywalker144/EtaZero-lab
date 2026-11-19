"""Scheduler orchestration uses real processes; native training is checked separately."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import sys

import pytest

from etazero import experiment
from etazero import ROOT
from etazero.run import save_json
from test_controller_recovery import recovery_config, wait_until


def plan_for(tmp_path, **settings):
    config = recovery_config(tmp_path, 'alphazero')
    save_json(tmp_path / 'run.json', config.serializable())
    save_json(tmp_path / 'exp.json', dict(arms={'a space': 'run.json', 'b': 'run.json', 'c': 'run.json'},
                                         iterations=2, shared_init=False, **settings))
    return experiment.experiment_plan(tmp_path / 'exp.json', tmp_path / 'work')


@pytest.mark.parametrize('change', [dict(iterations=None), dict(iterations=True), dict(max_hours=float('inf')),
    dict(gpus=[]), dict(gpus=['0', '0']), dict(gpus='0,1'), dict(shared_init='true'),
    dict(arms={'../escape': 'run.json'}), dict(unknown=True)])
def test_invalid_experiment_settings(tmp_path, change):
    plan_for(tmp_path)
    path = tmp_path / 'exp.json'
    save_json(path, {**json.loads(path.read_text()), **change})
    with pytest.raises(ValueError):
        experiment.experiment_plan(path, tmp_path / 'work')


def test_progress_requires_idle_committed_state_and_matching_config(tmp_path):
    plan = plan_for(tmp_path)
    arm = plan['arms'][0]
    root = Path(arm['run_dir'])
    root.mkdir(parents=True)
    save_json(root / 'config.json', arm['config'])
    state = dict(completed_rounds=2, active_round=3, elapsed_seconds=0)
    save_json(root / 'state.json', state)
    assert not experiment.arm_complete(arm, plan)
    state['active_round'] = None
    save_json(root / 'state.json', state)
    assert experiment.arm_complete(arm, plan)
    assert not experiment.arm_complete(arm, {**plan, 'iterations': 3})
    state['elapsed_seconds'] = 3600
    save_json(root / 'state.json', state)
    assert experiment.arm_complete(arm, {**plan, 'iterations': 3, 'max_hours': .5})
    save_json(root / 'config.json', {})
    with pytest.raises(ValueError, match='differs'):
        experiment.arm_complete(arm, plan)


def fake_source_children(tmp_path, monkeypatch, failing=False):
    script = tmp_path / 'worker.py'
    script.write_text('''import fcntl, json, os, sys, time
from pathlib import Path
args = sys.argv
root = Path(args[args.index('--run-dir') + 1])
config = Path(args[args.index('--config') + 1])
root.mkdir(parents=True, exist_ok=True)
# The experiment lease must survive in every arm process.
with (root.parent.parent / '.lock').open('a') as lock:
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: pass
    else: raise RuntimeError('experiment lease was lost')
start = time.monotonic()
time.sleep(.2)
(root / 'config.json').write_bytes(config.read_bytes())
(root / 'observed.json').write_text(json.dumps(dict(start=start, end=time.monotonic(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))))
if os.environ.get('FAIL_ARM') == root.name: sys.exit(7)
(root / 'state.json').write_text(json.dumps(dict(completed_rounds=int(args[args.index('--iterations') + 1]), active_round=None, elapsed_seconds=1)))
''')
    original = subprocess.Popen
    def popen(command, **kwargs):
        if failing:
            kwargs['env']['FAIL_ARM'] = 'a space'
        return original([sys.executable, str(script), *command[1:]], **kwargs)
    monkeypatch.setattr(experiment.subprocess, 'Popen', popen)


def test_slots_are_exclusive_and_resume_uses_actual_state(tmp_path, monkeypatch):
    plan = plan_for(tmp_path, gpus=['GPU-unit-a', 'GPU-unit-b'])
    fake_source_children(tmp_path, monkeypatch)
    assert experiment.run_experiment(plan, '/bin/true') == 0
    work = Path(plan['work_dir'])
    records = [json.loads((Path(a['run_dir']) / 'observed.json').read_text()) for a in plan['arms']]
    assert {r['gpu'] for r in records} == set(plan['slots'])
    for index, record in enumerate(records):
        for other in records[:index]:
            if record['gpu'] == other['gpu']:
                assert other['end'] <= record['start'] or record['end'] <= other['start']
    original = {p: p.read_bytes() for p in (work / 'runs').glob('*/observed.json')}
    save_json(work / 'status.json', {'arms': {'a space': 'failed'}})
    assert experiment.run_experiment(plan, '/bin/true') == 0
    assert all(p.read_bytes() == value for p, value in original.items())
    assert experiment.run_experiment({**plan, 'iterations': 3}, '/bin/true') == 0
    assert all(experiment.arm_complete(arm, {**plan, 'iterations': 3}) for arm in plan['arms'])


def test_failure_stops_queue_and_lock_rejects_concurrent_scheduler(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    work = Path(plan['work_dir'])
    work.mkdir()
    with (work / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another scheduler'):
            experiment.run_experiment(plan, '/bin/true')
    fake_source_children(tmp_path, monkeypatch, failing=True)
    with pytest.raises(RuntimeError, match='exited before'):
        experiment.run_experiment(plan, '/bin/true')
    state = json.loads((work / 'status.json').read_text())
    assert state['arms']['a space']['status'] == 'failed'
    assert state['arms']['b']['status'] == 'pending'
    assert not (work / 'runs/b').exists()


@pytest.mark.parametrize('termination', ['term', 'kill'])
def test_scheduler_signal_and_orphaned_stage_lease(tmp_path, termination):
    plan = plan_for(tmp_path)
    stage = tmp_path / 'native-stage'
    ready = tmp_path / 'stage-ready'
    stage.write_text(f'#!{sys.executable}\nimport os, signal, time\nfrom pathlib import Path\n'
                     f'Path({str(ready)!r}).write_text(str(os.getpid()))\n'
                     'signal.signal(signal.SIGINT, lambda *_: exit(0))\n'
                     'while True: time.sleep(.05)\n')
    stage.chmod(0o755)
    code = ('from etazero.experiment import experiment_plan, run_experiment\n'
            f'plan = experiment_plan({str(tmp_path / "exp.json")!r}, {plan["work_dir"]!r})\n'
            f'raise SystemExit(run_experiment(plan, {str(stage)!r}))\n')
    output = (tmp_path / 'scheduler.log').open('w')
    parent = subprocess.Popen([sys.executable, '-c', code],
        env=dict(os.environ, PYTHONPATH=str(ROOT / 'python')), stdout=output, stderr=subprocess.STDOUT)
    group = None
    try:
        wait_until(ready.exists, seconds=45)
        work = Path(plan['work_dir'])
        status = json.loads((work / 'status.json').read_text())
        group = status['arms']['a space']['pid']
        if termination == 'term':
            parent.terminate()
            assert parent.wait(timeout=25) == 130
            status = json.loads((work / 'status.json').read_text())
            assert status['stopping']
            assert status['arms']['a space']['status'] == 'interrupted'
        else:
            parent.kill()
            parent.wait(timeout=10)
            # Both the arm controller and native stage outlive their scheduler.
            # Kill the arm controller too: the stage alone must retain the lease.
            os.kill(group, signal.SIGKILL)
            with pytest.raises(RuntimeError, match='Another scheduler'):
                experiment.run_experiment(plan, '/bin/true')
            os.killpg(int(ready.read_text()), signal.SIGINT)
        assert not (work / 'runs/b').exists()
        with (work / '.lock').open('a') as lock:
            def acquire():
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    return False
            wait_until(acquire)
    finally:
        if ready.exists():
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if group is not None:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=10)
        output.close()
