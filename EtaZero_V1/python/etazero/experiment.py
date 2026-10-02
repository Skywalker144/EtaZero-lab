"""Finite, resumable experiment arms with one controller per GPU slot."""
import argparse
import configparser
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .config import ROOT, FILES, FIELDS, boolean, fingerprint, load_config, validate
from .schema import CONTRACT_ID
from .storage import atomic_write, load_json, save_json, sha256

EXP_FIELDS = {'max_iteration': int, 'max_seconds': float, 'arm_gpus': str, 'shared_init': boolean}
ENV_FIELDS = {'MAX_ITERS': 'max_iteration', 'MAX_TIME_SECONDS': 'max_seconds',
              'ARM_GPUS': 'arm_gpus', 'SHARED_INIT': 'shared_init'}


def experiment_plan(directory, environ=None, work_dir=None):
    env = os.environ if environ is None else environ
    directory = Path(directory)
    if not directory.is_absolute() and not directory.exists():
        directory = ROOT/directory
    directory = directory.resolve()
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.read_string((directory/'exp.cfg').read_text())
    if parser.defaults() or parser.sections() != ['experiment'] or set(parser['experiment']) != set(EXP_FIELDS):
        raise ValueError('exp.cfg requires exactly [experiment] and '+', '.join(EXP_FIELDS))
    raw = dict(parser['experiment'])
    for source, target in ENV_FIELDS.items():
        if source in env:
            raw[target] = env[source]
    settings = {key: convert(raw[key]) for key, convert in EXP_FIELDS.items()}
    if settings['max_iteration'] < 0 or not math.isfinite(settings['max_seconds']) or settings['max_seconds'] < 0:
        raise ValueError('Experiment budgets must be nonnegative and finite')
    if not settings['max_iteration'] and not settings['max_seconds']:
        raise ValueError('Each experiment requires a finite iteration or time budget')
    if settings['arm_gpus'].strip():
        slots = [value.strip() for value in settings['arm_gpus'].split(',')]
        if any(not value or not (value.isdecimal() or value.startswith(('GPU-', 'MIG-'))) for value in slots):
            raise ValueError('arm_gpus requires CUDA device indices or GPU/MIG UUIDs')
        if len(set(slots)) != len(slots):
            raise ValueError('arm_gpus must not repeat a GPU slot')
    else:
        slots = [None]
    tag = hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    work = Path(work_dir).resolve() if work_dir else ROOT/'data/experiments'/f'{directory.name}_{tag}'
    arms = []
    for selected in sorted(directory.iterdir()):
        if not selected.is_dir() or not (selected/'run.cfg').is_file():
            continue
        config = load_config(selected)
        if slots != [None]:
            devices = config['devices']
            if len(devices['selfplay'].split(',')) != 1:
                raise ValueError('GPU slot scheduling requires one selfplay device per arm')
            devices['train'] = 'cpu' if devices['train'] == 'cpu' else 'cuda:0'
            devices['selfplay'] = 'cpu' if devices['selfplay'] == 'cpu' else 'cuda:0'
            validate(config)
        destination = (ROOT/config['run']['run_dir']).resolve()
        # Resolve symlinks before overlap checks and before writing the effective config.
        config['run']['run_dir'] = str(destination)
        arms.append({'name': selected.name, 'config_dir': str(selected), 'run_dir': str(destination), 'config': config})
    if not arms:
        raise ValueError(f'No experiment arm directories with run.cfg in {directory}')
    destinations = [Path(a['run_dir']) for a in arms]
    for i, destination in enumerate(destinations):
        if destination == work or destination in work.parents:
            raise ValueError('An arm run directory must not contain the experiment controller directory')
        for other in destinations[:i]:
            if destination == other or destination in other.parents or other in destination.parents:
                raise ValueError('Experiment arms require independent, non-overlapping run directories')
    return {'umbrella': str(directory), 'work_dir': str(work), 'settings': settings, 'slots': slots, 'arms': arms}


def initialization_key(config):
    identity = [CONTRACT_ID, config['network'], config['run']['seed']]
    if config['agent']['algorithm'] == 'muzero':
        identity.append({'algorithm': 'muzero', 'muzero_config': config['muzero']})
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def prepare_initializations(work, arms):
    import torch
    from .network import make_network
    from .training import model_identity
    references = {}
    for arm in arms:
        key = initialization_key(arm['config'])
        path = work/'.internal/initializations'/f'{key}.pt'
        record = path.with_suffix('.json')
        if record.exists():
            reference = load_json(record)
            if reference['key'] != key or reference['sha256'] != sha256(path):
                raise ValueError(f'Shared initialization checksum mismatch: {path}')
        else:
            # CPU-only creation does not consume any arm's learner RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(arm['config']['run']['seed'])
                model = make_network(arm['config'])
                value = {'contract': CONTRACT_ID, **model_identity(arm['config']),
                         'model': model.state_dict()}
                if path.exists():
                    # A crash between payload and sidecar publication is repairable.
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    if saved['contract'] != value['contract'] or saved['network_config'] != value['network_config'] or \
                            set(saved['model']) != set(value['model']) or any(
                                not torch.equal(saved['model'][k], v) for k, v in value['model'].items()):
                        raise ValueError(f'Uncommitted initialization differs from its seed: {path}')
                else:
                    atomic_write(path, lambda p: torch.save(value, p), immutable=True)
            reference = {'key': key, 'path': str(path), 'sha256': sha256(path)}
            save_json(record, reference, immutable=True)
        references[arm['name']] = reference
    return references


def arm_progress(arm, settings, initialization=None):
    root = Path(arm['run_dir'])
    info_path = root/'.internal/run.json'
    if not info_path.exists():
        if root.exists() and any(root.iterdir()):
            # The run lock can precede run.json if startup was interrupted.
            leftovers = list(root.rglob('*'))
            if any(p.is_file() and p != root/'.internal/run.lock' for p in leftovers):
                raise ValueError(f'Nonempty arm directory has no run record: {root}')
        return 'pending'
    info = load_json(info_path)
    if info['config_id'] != fingerprint(arm['config']) or info['config'] != arm['config']:
        raise ValueError(f'Arm configuration differs from saved run: {arm["name"]}')
    origin = info['weights_initialization']
    if initialization != origin:
        raise ValueError(f'Arm initialization differs from saved run: {arm["name"]}')
    if origin and sha256(origin['path']) != origin['sha256']:
        raise ValueError(f'Arm initialization payload changed: {arm["name"]}')
    state_path = root/'.internal/state.json'
    state = load_json(state_path) if state_path.exists() else {'iteration': 0, 'elapsed_seconds': 0.0}
    if settings['max_iteration'] and state['iteration'] > settings['max_iteration']:
        return 'complete'
    if settings['max_seconds'] and state['elapsed_seconds'] >= settings['max_seconds']:
        return 'complete'
    return 'pending'


def write_arm_config(config, directory):
    for name in FILES:
        parser = configparser.ConfigParser(interpolation=None)
        for section, (owner, _) in FIELDS.items():
            if owner == name and section in config:
                parser[section] = {key: str(value).lower() if isinstance(value, bool) else str(value)
                                   for key, value in config[section].items()}
        stream = io.StringIO(); parser.write(stream)
        atomic_write(directory/(name+'.cfg'), lambda p: p.write_text(stream.getvalue()))


class Scheduler:
    def __init__(self, work, plan, binary, initializations):
        self.work, self.plan, self.binary, self.initializations = work, plan, Path(binary), initializations
        self.running = []; self.stop = False
        self.states = {}; self.handlers = {}

    def signal(self, *_):
        self.stop = True

    def save(self):
        save_json(self.work/'.internal/status.json', {'arms': self.states, 'stopping': self.stop})

    def start(self, arm, slot, gpu):
        directory = self.work/'configs'/arm['name']
        write_arm_config(arm['config'], directory)
        env = dict(os.environ)
        env.pop('CONFIG_DIR', None)
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = gpu
        env['PYTHONPATH'] = str(ROOT/'python')+os.pathsep+env.get('PYTHONPATH', '')
        command = [sys.executable, '-m', 'etazero', 'run', '--config-dir', str(directory),
                   '--run-dir', arm['run_dir'], '--binary', str(self.binary),
                   '--iterations', str(self.plan['settings']['max_iteration']),
                   '--max-seconds', str(self.plan['settings']['max_seconds'])]
        # A resumed run already records its weight origin and rejects --weights.
        origin = self.initializations.get(arm['name'])
        if origin and not (Path(arm['run_dir'])/'.internal/run.json').exists():
            command += ['--weights', origin['path']]
        log = self.work/'logs'/f'{arm["name"]}.runner.log'; log.parent.mkdir(parents=True, exist_ok=True)
        output = log.open('a', buffering=1)
        try:
            process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            output.close(); raise
        self.running.append({'process': process, 'output': output, 'slot': slot, 'arm': arm})
        self.states[arm['name']] = {'status': 'running', 'gpu': gpu, 'pid': process.pid, 'command': command, 'log': str(log)}
        self.save()
        print(f'started {arm["name"]} gpu={gpu if gpu is not None else arm["config"]["devices"]["train"]}', flush=True)

    def cleanup(self):
        # Native workers own separate process groups; signal the Python controller
        # so it can flush games, commit updates and close its workers itself.
        for item in self.running:
            if item['process'].poll() is None:
                item['process'].send_signal(signal.SIGINT)
        deadline = time.monotonic()+60
        while any(item['process'].poll() is None for item in self.running) and time.monotonic() < deadline:
            time.sleep(.1)
        for item in self.running:
            process = item['process']
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            item['output'].close()
            self.states[item['arm']['name']]['status'] = 'interrupted'
            self.states[item['arm']['name']]['returncode'] = process.returncode
        self.save()

    def run(self):
        settings = self.plan['settings']; queue = []
        for arm in self.plan['arms']:
            origin = self.initializations.get(arm['name'])
            # run.json stores just path/checksum, while the umbrella also stores the group key.
            origin = {k: origin[k] for k in ('path', 'sha256')} if origin else None
            status = arm_progress(arm, settings, origin)
            self.states[arm['name']] = {'status': status}
            if status == 'pending':
                queue.append(arm)
            else:
                print(f'skipped completed {arm["name"]}', flush=True)
        self.save()
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.handlers[sig] = signal.signal(sig, self.signal)
        try:
            while (queue or self.running) and not self.stop:
                for item in list(self.running):
                    code = item['process'].poll()
                    if code is None:
                        continue
                    item['output'].close(); self.running.remove(item)
                    arm = item['arm']; origin = self.initializations.get(arm['name'])
                    origin = {k: origin[k] for k in ('path', 'sha256')} if origin else None
                    complete = code == 0 and arm_progress(arm, settings, origin) == 'complete'
                    self.states[arm['name']].update(status='complete' if complete else 'failed', returncode=code)
                    self.save()
                    if not complete:
                        raise RuntimeError(f'Experiment {arm["name"]} exited before its budget was completed '
                                           f'(code={code}); see {self.work/"logs"/(arm["name"]+".runner.log")}')
                    print(f'completed {arm["name"]}', flush=True)
                occupied = {item['slot'] for item in self.running}
                for slot, gpu in enumerate(self.plan['slots']):
                    if queue and slot not in occupied and not self.stop:
                        self.start(queue.pop(0), slot, gpu)
                if self.running:
                    time.sleep(.2)
            return 130 if self.stop else 0
        finally:
            self.cleanup()
            for sig, handler in self.handlers.items():
                signal.signal(sig, handler)


def run_experiment(plan, binary):
    from .runtime import verify_build
    verify_build(binary)
    work = Path(plan['work_dir']); (work/'.internal').mkdir(parents=True, exist_ok=True)
    with (work/'.internal/experiment.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another scheduler owns this experiment') from error
        identity = {'umbrella': plan['umbrella'], 'arms': plan['arms'], 'shared_init': plan['settings']['shared_init']}
        manifest = work/'.internal/identity.json'
        if manifest.exists():
            if load_json(manifest) != identity:
                raise ValueError('Experiment arms or shared initialization changed; use a new experiment directory')
        else:
            save_json(manifest, identity, immutable=True)
        initializations = prepare_initializations(work, plan['arms']) if identity['shared_init'] else {}
        save_json(work/'.internal/plan.json', {**plan, 'initializations': initializations,
                                             'binary_sha256': sha256(binary)})
        return Scheduler(work, plan, binary, initializations).run()


def main():
    parser = argparse.ArgumentParser(description='Queue finite EtaZero experiments from an exp.cfg umbrella')
    parser.add_argument('--config-dir', default=os.environ.get('CONFIG_DIR'))
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/etazero')
    parser.add_argument('--dry-run', action='store_true', default=os.environ.get('DRY_RUN') == '1')
    args = parser.parse_args()
    if not args.config_dir:
        parser.error('Set CONFIG_DIR or --config-dir to an experiment umbrella')
    plan = experiment_plan(args.config_dir, work_dir=args.work_dir)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return 0
    return run_experiment(plan, args.binary.resolve())


if __name__ == '__main__':
    sys.exit(main())
