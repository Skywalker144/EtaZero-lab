"""Finite experiment arms; source learning remains inside each round controller."""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import ROOT
from .config import RunConfig, load_config
from .initialization import prepare_shared_initializations
from .profiles import default_output, load_experiment_profile
from .run import save_json


def experiment_plan(config_path, work_dir=None):
    path, settings = load_experiment_profile(config_path)
    allowed = {'arms', 'iterations', 'max_hours', 'gpus', 'shared_init', 'evaluation'}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError('Experiment accepts arms, iterations, max_hours, gpus, shared_init and evaluation')
    iterations, hours = settings.get('iterations'), settings.get('max_hours')
    if iterations is not None and (type(iterations) is not int or iterations < 0):
        raise ValueError('iterations must be a nonnegative final iteration (0 is bootstrap)')
    if hours is not None and (type(hours) not in (int, float) or not math.isfinite(hours * 3600) or hours <= 0):
        raise ValueError('max_hours must be finite and positive')
    if iterations is None and hours is None:
        raise ValueError('Each experiment requires a finite round or time budget')
    shared = settings.get('shared_init', True)
    if type(shared) is not bool:
        raise ValueError('shared_init must be boolean')
    gpus = settings.get('gpus')
    if gpus is not None:
        if (not isinstance(gpus, list) or not gpus or
                any(not isinstance(g, str) or not (g.isdecimal() or g.startswith(('GPU-', 'MIG-'))) for g in gpus)):
            raise ValueError('gpus must be a nonempty list of CUDA index or UUID strings')
        if len(set(gpus)) != len(gpus):
            raise ValueError('gpus must not repeat a slot')
    selected = settings.get('arms')
    if not isinstance(selected, dict) or not selected:
        raise ValueError('arms must map names to run-config paths')
    evaluation = settings.get('evaluation')
    if evaluation is not None:
        from .autoelo import SELECTION, FIT, EXECUTION, validate_options
        if not isinstance(evaluation, dict) or set(evaluation) - (SELECTION | FIT | EXECUTION | {'config'}) or 'config' not in evaluation:
            raise ValueError('evaluation requires config and accepts autoelo selection, fitting and residency settings')
        if not isinstance(evaluation['config'], str):
            raise ValueError('evaluation.config must be a source match config path')
        validate_options(evaluation)
        evaluation = dict(evaluation, config=str((path.parent / evaluation['config']).resolve()))
        if not Path(evaluation['config']).is_file():
            raise ValueError('Missing independent evaluation config')
    work = Path(work_dir).resolve() if work_dir is not None else default_output(path, 'exp.cfg')
    arms = []
    for name, config in sorted(selected.items()):
        if not name or name in {'.', '..'} or Path(name).name != name or '\\' in name:
            raise ValueError(f'Arm name must be a directory name: {name!r}')
        if not isinstance(config, str):
            raise ValueError(f'Arm {name} needs a run-config path')
        config = (path.parent / config).resolve()
        arms.append(dict(name=name, config_path=str(config), config=load_config(config).serializable(),
                         run_dir=str(work / 'runs' / name)))
    if evaluation and len({arm['config']['game'] for arm in arms}) != 1:
        raise ValueError('Automatic Elo requires all arms to use the same game')
    return dict(config_path=str(path), work_dir=str(work), arms=arms, slots=gpus or [None],
                iterations=iterations, max_hours=hours, shared_init=shared, evaluation=evaluation)


def arm_complete(arm, plan):
    root = Path(arm['run_dir'])
    if not (root / 'state.json').is_file():
        return False
    if json.loads((root / 'config.json').read_text()) != arm['config']:
        raise ValueError(f'Run configuration differs from existing arm {arm["name"]}')
    state = json.loads((root / 'state.json').read_text())
    if state['active_round'] is not None:
        return False
    return ((plan['iterations'] is not None and state['completed_rounds'] >= plan['iterations']) or
            (plan['max_hours'] is not None and state['elapsed_seconds'] >= 3600 * plan['max_hours']))


def signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


class Scheduler:
    def __init__(self, plan, binary, initializations, lock_fd):
        self.plan, self.binary, self.initializations, self.lock_fd = plan, binary, initializations, lock_fd
        self.work = Path(plan['work_dir'])
        self.running, self.states = [], {}
        self.stopping = False

    def signal(self, *_):
        self.stopping = True

    def save(self):
        save_json(self.work / 'status.json', dict(arms=self.states, stopping=self.stopping))

    def start(self, arm, slot, gpu):
        config_path = self.work / 'configs' / (arm['name'] + '.json')
        config_path.parent.mkdir(exist_ok=True)
        save_json(config_path, arm['config'])
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'python') + os.pathsep + os.environ.get('PYTHONPATH', ''))
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = gpu
        command = [sys.executable, '-m', 'etazero', 'run', '--config', str(config_path),
                   '--run-dir', arm['run_dir'], '--binary', str(self.binary), '--lease-fd', str(self.lock_fd)]
        for option, key in (('--iterations', 'iterations'), ('--max-hours', 'max_hours')):
            if self.plan[key] is not None:
                command += [option, str(self.plan[key])]
        if arm['name'] in self.initializations:
            command += ['--weights', str(self.initializations[arm['name']])]
        log = self.work / 'logs' / (arm['name'] + '.log')
        log.parent.mkdir(exist_ok=True)
        output = log.open('a')
        try:
            process = subprocess.Popen(command, cwd=ROOT / 'python', env=env, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(self.lock_fd,))
        except BaseException:
            output.close()
            raise
        self.running.append(dict(process=process, output=output, arm=arm, slot=slot))
        self.states[arm['name']] = dict(status='running', gpu=gpu, pid=process.pid, log=str(log), command=command)
        self.save()
        print(f'Started {arm["name"]} in GPU slot {gpu}', flush=True)

    def cleanup(self):
        # Signal the entire arm group, including source children whose controller
        # may already have exited. Inherited leases prevent premature takeover.
        for item in self.running:
            signal_group(item['process'], signal.SIGINT)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            alive = False
            for item in self.running:
                item['process'].poll()
                try:
                    os.killpg(item['process'].pid, 0)
                    alive = True
                except ProcessLookupError:
                    pass
            if not alive:
                break
            time.sleep(.1)
        for item in self.running:
            process = item['process']
            signal_group(process, signal.SIGKILL)
            process.wait()
            item['output'].close()
            state = self.states[item['arm']['name']]
            if state['status'] != 'failed':
                state['status'] = 'interrupted'
            state['returncode'] = process.returncode
        self.save()

    def run(self):
        queue = []
        for arm in self.plan['arms']:
            complete = arm_complete(arm, self.plan)
            self.states[arm['name']] = dict(status='complete' if complete else 'pending')
            if not complete:
                queue.append(arm)
        self.save()
        handlers = {sig: signal.signal(sig, self.signal) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while (queue or self.running) and not self.stopping:
                for item in list(self.running):
                    code = item['process'].poll()
                    if code is None:
                        continue
                    arm = item['arm']
                    complete = code == 0 and arm_complete(arm, self.plan)
                    self.states[arm['name']].update(status='complete' if complete else 'failed', returncode=code)
                    self.save()
                    if not complete:
                        raise RuntimeError(f'Experiment arm {arm["name"]} exited before its budget completed; see {self.states[arm["name"]]["log"]}')
                    item['output'].close()
                    self.running.remove(item)
                occupied = {item['slot'] for item in self.running}
                for slot, gpu in enumerate(self.plan['slots']):
                    if queue and slot not in occupied and not self.stopping:
                        self.start(queue.pop(0), slot, gpu)
                if self.running:
                    time.sleep(.2)
            return 130 if self.stopping else 0
        finally:
            self.cleanup()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def run_experiment(plan, binary=None):
    binary = Path(binary or ROOT / 'build/katago').resolve()
    if not binary.is_file():
        raise ValueError(f'Build the native engine first: {binary}')
    work = Path(plan['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    with (work / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another scheduler or arm is using {work}') from error
        # Validate existing arms before allocating initialization or GPU work.
        for arm in plan['arms']:
            arm_complete(arm, plan)
        initializations = prepare_shared_initializations(work,
            [{**arm, 'config': RunConfig(**arm['config'])} for arm in plan['arms']],
            lock_fd=lock.fileno()) if plan['shared_init'] else {}
        save_json(work / 'plan.json', {**plan, 'binary': str(binary),
            'initializations': {name: str(path) for name, path in initializations.items()}})
        code = Scheduler(plan, binary, initializations, lock.fileno()).run()
        if code == 0 and plan.get('evaluation') is not None:
            from .autoelo import run_autoelo
            status = work / 'elo_status.json'
            save_json(status, dict(status='running', training='complete'))
            env = dict(os.environ)
            if plan['slots'][0] is not None:
                env['CUDA_VISIBLE_DEVICES'] = plan['slots'][0]
            # The arm controllers are already done. Evaluator failure is tracked
            # independently, so retry never re-trains arms that reached budget.
            def interrupt_evaluation(*_):
                raise KeyboardInterrupt
            handlers = {sig: signal.signal(sig, interrupt_evaluation)
                        for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                run_autoelo(work / 'elo', data=work / 'runs', binary=binary,
                            arms=[(a['name'], a['run_dir']) for a in plan['arms']],
                            inherited_fds=(lock.fileno(),), env=env, **plan['evaluation'])
                save_json(status, dict(status='complete', training='complete'))
            except BaseException as error:
                save_json(status, dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                       training='complete', error=str(error)))
                if isinstance(error, KeyboardInterrupt):
                    return 130
                raise
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        return code
