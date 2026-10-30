"""Synchronous rounds using the source programs, with explicit stage ownership.

Recovery follows published source shards, epoch checkpoints and atomic directory
publication. Uncommitted work is recomputed without replaying committed stages.
The source learner owns checkpoint state, file selection, and optimizer clocks.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

from . import ROOT, __version__
from .replay import count_rows, plan_games, plan_training, commit_replay
from .timing import RoundClock, compiler_seconds, committed_timing


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execute(argv, log, *, lock_fd=None):
    """List arguments avoid shell expansion; source programs run at their source cwd."""
    print('Running:', ' '.join(map(str, argv)), flush=True)
    with Path(log).open('w') as stream:
        try:
            subprocess.run(list(map(str, argv)), cwd=ROOT / 'python', stdout=stream,
                           stderr=subprocess.STDOUT, check=True,
                           pass_fds=() if lock_fd is None else ((lock_fd,) if isinstance(lock_fd, int) else lock_fd))
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'Source stage exited with {error.returncode}; see {log}') from error


def checkpoint_progress(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return (int(checkpoint['train_state']['global_step_samples']),
            int(checkpoint['train_state']['export_cycle_counter']))


def published_games(data):
    """Count source game hashes in atomically published NPZs, only on recovery.

    Rows already contain targets from a finished game. An interrupted writer can
    lose buffered rows, but no published row is rewritten or turned into a fake
    partial-game target. MuZero shards retain the full referenced trajectories.
    """
    import numpy as np
    games = set()
    for path in sorted(data.glob('*.npz')):
        with np.load(path) as shard:
            hashes = np.unique(shard['globalTargetsNC'][:, 41:47], axis=0)
            games.update(map(tuple, hashes))
    return len(games)


def learner_command(root, config, *, epochs=None):
    network = (['-muzero-config', root / 'muzero_network.json'] if config.is_muzero else
               ['-model-kind', config.model_kind])
    return [sys.executable, ROOT / 'python/train.py',
        '-traindir', root / 'train', '-latestdatadir', root / 'shuffleddata',
        '-pos-len', config.canvas, '-batch-size', config.batch_size, *network,
        '-input-features', config.game if config.game in {'go', 'xiangqi'} else 'placement',
        '-samples-per-epoch', config.samples_per_epoch,
        '-max-epochs-this-instance', config.epochs_per_round if epochs is None else epochs, '-no-export', '-quit-if-no-data',
        *config.train_args,
        *(['-initial-weights', root / 'initial/weights.ckpt'] if (root / 'initial/weights.ckpt').is_file() else [])]


def snapshot_source(destination):
    # One archive at run creation records uncommitted/untracked adaptations too.
    # It is evidence, not a hash contract checked on every read or restart.
    def include(info):
        if any(part in {'__pycache__', '.pytest_cache', 'scratch'} for part in Path(info.name).parts):
            return None
        return info
    with tarfile.open(destination, 'w:gz') as archive:
        for name in ('cpp', 'python', 'scripts', 'configs', 'LICENSE', 'licenses', 'VERSION', 'reference_sources.json'):
            archive.add(ROOT / name, arcname=name, filter=include)


def run_round(root, config, state, binary, lock_fd=None):
    number = state['completed_rounds'] + 1
    commit_path = root / 'rounds' / f'{number:06d}' / 'commit.json'
    if commit_path.is_file():
        # A completed round's time/model record is authoritative even when the
        # following state-pointer write was interrupted. Do not time it twice.
        commit = json.loads(commit_path.read_text())
        state.clear()
        state.update(commit['state'])
        save_json(root / 'state.json', state)
        print(f'Recovered committed round {number}', flush=True)
        return
    if state['active_round'] is None:
        state.update(active_round=number, phase='prepare', round_progress={
            'done': [], 'attempts': {}, 'stage_seconds': {}, 'stage_compile_seconds': {},
            'running': None, 'clock_running': False, 'wall_seconds': 0.0, 'compile_seconds': 0.0,
            'timing_valid': True})
    progress = state['round_progress']
    if progress['running'] is not None or progress['clock_running']:
        # The previous controller did not settle its last interval, whether
        # inside a child or between stages. The inherited lease prevents
        # takeover while a child still lives, but cannot recover missing time.
        if progress['running'] in ('train', 'initialize'):
            progress['stage_compile_seconds'][progress['running']] = None
            progress['compile_seconds'] = None
        progress.update(running=None, timing_valid=False)
        print('WARNING: resuming an unobserved controller interval; elapsed stage time is incomplete',
              file=sys.stderr, flush=True)
    clock = RoundClock(progress)
    def persist():
        clock.settle()
        save_json(root / 'state.json', state)
    progress['clock_running'] = True
    try:
        persist()
        _run_round(root, config, state, binary, lock_fd, progress, persist)
    finally:
        if state['active_round'] is not None:
            progress['clock_running'] = False
            persist()


def _run_round(root, config, state, binary, lock_fd, progress, persist):
    number = state['completed_rounds'] + 1
    replay = state.get('replay')
    directory = root / 'rounds' / f'{number:06d}'
    directory.mkdir(parents=True, exist_ok=True)
    logs = directory / 'logs'
    logs.mkdir(exist_ok=True)

    def done(name):
        return name in progress['done']

    def finish(name):
        if not done(name):
            progress['done'].append(name)
            persist()

    def stage(name, command):
        attempt = progress['attempts'].get(name, 0) + 1
        progress['attempts'][name] = attempt
        progress['running'] = name
        state['phase'] = name
        receipt = None
        if name in ('train', 'initialize'):
            timing_dir = directory / 'timing'
            timing_dir.mkdir(exist_ok=True)
            receipt = timing_dir / f'{name}.{attempt}.json'
            command = [command[0], '-m', 'etazero.timed_learner', receipt, *command[1:]]
        persist()
        log = logs / (f'{name}.log' if attempt == 1 else f'{name}.{attempt}.log')
        started = time.monotonic()
        try:
            execute(command, log, lock_fd=lock_fd)
        except BaseException as error:
            if not isinstance(error, RuntimeError):
                progress['timing_valid'] = False
            raise
        finally:
            wall = time.monotonic() - started
            progress['stage_seconds'][name] = progress['stage_seconds'].get(name, 0.0) + wall
            if receipt is not None:
                compiled = compiler_seconds(receipt, wall)
                old = progress['stage_compile_seconds'].get(name, 0.0)
                progress['stage_compile_seconds'][name] = None if compiled is None or old is None else old + compiled
                if compiled is None:
                    progress['compile_seconds'] = None
                    progress['timing_valid'] = False
                elif progress['compile_seconds'] is not None:
                    progress['compile_seconds'] += compiled
            progress['running'] = None
            persist()

    def retain_partial(path):
        # Only unpublished derived artifacts are moved aside. Raw data and
        # checkpoints stay where their producers committed them.
        if path.exists():
            discarded = directory / 'discarded'
            discarded.mkdir(exist_ok=True)
            path.rename(discarded / (path.name + '.' + uuid.uuid4().hex))

    model_files = (('representation.bin', 'dynamics.bin', 'prediction.bin', 'model.json')
                   if config.is_muzero else ('model.bin',))

    def publish(name, checkpoint, log_name, weights):
        published = root / 'models' / name
        if published.is_dir():
            finish(log_name)
            return published
        candidate = published.with_name(published.name + '.tmp')
        candidate.parent.mkdir(parents=True, exist_ok=True)
        retain_partial(candidate)
        exporter = 'export_muzero.py' if config.is_muzero else 'export_model_pytorch.py'
        layout = ['-pos-len', config.canvas] if config.is_muzero else ['-filename-prefix', 'model']
        stage(log_name, [sys.executable, ROOT / 'python' / exporter,
            '-checkpoint', checkpoint, '-export-dir', candidate,
            '-model-name', f'etazero-v{__version__.replace(".", "_")}-{name}', *layout,
            *(['-use-swa'] if weights == 'swa' else [])])
        if not all((candidate / file).is_file() for file in model_files):
            raise RuntimeError('Source export did not produce a complete model')
        candidate.rename(published)
        finish(log_name)
        return published

    def gate(candidate, baseline):
        # Native gatekeeper moves a hardlinked copy. Published model artifacts
        # remain immutable, including candidates that fail acceptance.
        gate_dir = directory / 'gatekeeper'
        gate_dir.mkdir(exist_ok=True)
        receipt = gate_dir / 'result.json'
        testing = gate_dir / 'test' / candidate.name
        if not receipt.is_file():
            for source, target in ((candidate, testing), (baseline, gate_dir / 'accepted' / baseline.name)):
                target.mkdir(parents=True, exist_ok=True)
                for file in model_files:
                    if not (target / file).exists():
                        os.link(source / file, target / file)
            attempt = progress['attempts'].get('gatekeeper', 0) + 1
            matches = gate_dir / 'matches' / str(attempt)
            matches.mkdir(parents=True, exist_ok=True)
            stage('gatekeeper', [binary, 'gatekeeper', '-config', config.gatekeeper_config,
                '-test-models-dir', testing.parent, '-accepted-models-dir', gate_dir / 'accepted',
                '-rejected-models-dir', gate_dir / 'rejected', '-sgf-output-dir', matches,
                '-required-candidate-win-prop', config.gatekeeper_required_win_prop,
                '-no-autoreject-old-models', '-result-file', receipt,
                '-override-config', f'game={config.game},useMuZero={str(config.is_muzero).lower()}'])
        if not receipt.is_file():
            raise RuntimeError('Gatekeeper did not commit a decision; candidate and previous accepted model are preserved')
        result = json.loads(receipt.read_text())
        if result['candidate'] != candidate.name or result['baseline'] != baseline.name:
            raise RuntimeError('Gatekeeper receipt does not match this round candidate and baseline')
        destination = gate_dir / ('accepted' if result['accepted'] else 'rejected') / candidate.name
        if not destination.is_dir():
            # Receipt precedes the source directory move, so a killed process
            # can leave the already-decided candidate at its input location.
            destination.parent.mkdir(exist_ok=True)
            testing.rename(destination)
        finish('gatekeeper')
        return (candidate if result['accepted'] else baseline), result

    checkpoint = root / 'train/checkpoint.ckpt'
    accepted = root / state['accepted_model'] if state['accepted_model'] else None
    if (config.is_muzero or (root / 'initial/weights.ckpt').is_file()) and accepted is None:
        if not done('initialize'):
            if not checkpoint.is_file():
                stage('initialize', [*learner_command(root, config), '-initialize-only'])
            if checkpoint_progress(checkpoint) != (0, 0):
                raise RuntimeError('Initial learner checkpoint advanced the training clock')
            initial = root / 'initial'
            initial.mkdir(exist_ok=True)
            if not (initial / 'checkpoint.ckpt').exists():
                os.link(checkpoint, initial / 'checkpoint.ckpt')
            finish('initialize')
        accepted = publish('initial', root / 'initial/checkpoint.ckpt', 'initial_export', 'ordinary')

    # Only this fixed model is visible to the source polling loop throughout
    # this round, including all attempts after an interrupted selfplay process.
    models = directory / 'input_models'
    models.mkdir(exist_ok=True)
    input_name = 'random'
    if accepted is not None:
        input_name = accepted.name
        frozen = models / input_name
        frozen.mkdir(exist_ok=True)
        for file in model_files:
            if not (frozen / file).exists():
                os.link(accepted / file, frozen / file)
    output = root / 'selfplay' / f'{number:06d}'
    output.mkdir(parents=True, exist_ok=True)
    data = output / input_name / 'tdata'
    if 'selfplay_target' not in progress:
        progress.update(selfplay_target=plan_games(config, replay['history'] if replay else []),
                        selfplay_base_games=0, selfplay_base_hashes=0)
        persist()
    while not done('selfplay'):
        attempt = progress['attempts'].get('selfplay', 0)
        receipt = directory / f'selfplay-{attempt}.json'
        completed = None
        if receipt.is_file():
            result = json.loads(receipt.read_text())
            if result['games_requested'] != progress['selfplay_requested']:
                raise RuntimeError('Selfplay completion receipt differs from the recorded request')
            if result['completed']:
                completed = progress['selfplay_attempt_target']
        if completed is None:
            completed = progress['selfplay_base_games'] + published_games(data) - progress['selfplay_base_hashes']
        if completed < progress['selfplay_target']:
            remaining = progress['selfplay_target'] - completed
            progress['selfplay_requested'] = remaining
            progress['selfplay_attempt_target'] = progress['selfplay_target']
            receipt = directory / f'selfplay-{attempt + 1}.json'
            selfplay_config = config.selfplay_config
            if config.native_selfplay is not None:
                selfplay_config = root / 'selfplay.resolved.cfg'
                selfplay_config.write_text(config.native_selfplay)
            stage('selfplay', [binary, 'selfplay', '-config', selfplay_config,
                '-models-dir', models, '-output-dir', output, '-max-games-total', remaining,
                '-result-file', receipt,
                '-override-config', f'game={config.game},dataBoardLen={config.canvas},switchNetsMidGame=false,'
                f'useMuZero={str(config.is_muzero).lower()},'
                f'rootSearchAlgorithm={"gumbel" if config.is_gumbel else "puct"}'])
            result = json.loads(receipt.read_text())
            if result['games_requested'] != remaining or not result['completed']:
                raise RuntimeError('Selfplay was interrupted before its quota completed; published samples were preserved')
            completed = progress['selfplay_target']
        if replay is not None:
            rows = count_rows(data)
            if rows == 0 and replay['origin_rows'] is None:
                raise RuntimeError('Bootstrap selfplay produced no training rows; cannot estimate rows per game')
            minimum = config.bootstrap_min_rows or config.samples_per_epoch
            if replay['origin_rows'] is None and rows < minimum:
                progress.update(selfplay_target=completed + math.ceil((minimum - rows) * completed / rows),
                                selfplay_base_games=completed, selfplay_base_hashes=published_games(data))
                persist()
                continue
            progress['sampling'] = {'rows': rows, 'games': completed}
        finish('selfplay')
    if (replay is None or replay['origin_rows'] is None) and not any(data.glob('*.npz')):
        raise RuntimeError(f'Selfplay produced no training rows for the required model {input_name}; see the source selfplay log')

    shuffle = root / 'shuffleddata' / f'{number:06d}'
    scratch = directory / 'shuffle_scratch'
    if not done('shuffle'):
        if not shuffle.is_dir():
            pending = shuffle.with_name(shuffle.name + '.tmp')
            retain_partial(pending)
            retain_partial(scratch)
            pending.mkdir(parents=True)
            stage('shuffle', [sys.executable, ROOT / 'python/shuffle.py', root / 'selfplay',
                '-out-dir', pending / 'train', '-out-tmp-dir', scratch,
                '-num-processes', config.shuffle_processes,
                '-approx-rows-per-out-file', config.shuffle_rows_per_file,
                '-keep-target-rows', config.shuffle_keep_rows,
                *(['-muzero-unroll-steps', config.muzero_unroll_steps] if config.is_muzero else []),
                *config.shuffle_args])
            if not (pending / 'train.json').is_file() or not any((pending / 'train').glob('*.npz')):
                raise RuntimeError('Shuffle did not produce a usable source training snapshot')
            pending.rename(shuffle)
        finish('shuffle')
    if scratch.exists():
        try:
            shutil.rmtree(scratch)
        except OSError as error:
            print(f'WARNING: committed shuffle; scratch cleanup failed: {error}', file=sys.stderr, flush=True)

    after, epoch = checkpoint_progress(checkpoint) if checkpoint.is_file() else (0, 0)
    if 'train_start' not in progress:
        progress['train_start'] = {'samples': after, 'epoch': epoch}
        progress['training_plan'] = plan_training(config, replay, progress.get('sampling', {}).get('rows', 0))
        persist()
    before = progress['train_start']['samples']
    training_plan = progress['training_plan']
    target_epoch = progress['train_start']['epoch'] + training_plan['epochs']
    if not progress['train_start']['epoch'] <= epoch <= target_epoch:
        raise RuntimeError('Learner epoch counter is outside the recorded round budget')
    if epoch < target_epoch:
        stage('train', learner_command(root, config, epochs=target_epoch - epoch))
        after, epoch = checkpoint_progress(checkpoint)
    if epoch != target_epoch or (training_plan['epochs'] > 0 and after <= before):
        raise RuntimeError('Learner did not commit the planned epochs and new samples; no model will be published')
    finish('train')

    published = (publish(f'round-{number:06d}', checkpoint, 'export', config.publish_weights)
                 if training_plan['epochs'] else accepted)
    selected = published
    decision = {'accepted': None, 'reason': 'no_training'}
    if training_plan['epochs']:
        if config.gatekeeper_config is not None and state['accepted_model'] is not None:
            selected, decision = gate(published, accepted)
        else:
            decision = {'accepted': True, 'reason': 'initial_model' if state['accepted_model'] is None else 'disabled'}
    persist()
    final_state = {**state, **committed_timing(state, progress),
                   'completed_rounds': number, 'accepted_model': str(selected.relative_to(root)),
                   'active_round': None, 'phase': 'idle', 'total_samples': after}
    if replay is not None:
        final_state['replay'] = commit_replay(replay, training_plan, progress['sampling'], after - before)
    del final_state['round_progress']
    round_timing = dict(wall_seconds=progress['wall_seconds'], compile_seconds=progress['compile_seconds'],
                        seconds=(progress['wall_seconds'] - progress['compile_seconds']) if progress['timing_valid'] else None,
                        timing_valid=progress['timing_valid'])
    candidate_model = str(published.relative_to(root)) if training_plan['epochs'] else None
    save_json(directory / 'commit.json', {'state': final_state, 'round_timing': round_timing,
                                         'candidate_model': candidate_model, 'gatekeeper': decision})
    state.clear()
    state.update(final_state)
    save_json(root / 'state.json', state)
    try:
        save_json(directory / 'result.json', {'round': number, 'input_model': input_name, 'samples_before': before,
            'samples_after': after, 'model': str(selected.relative_to(root)),
            'candidate_model': candidate_model,
            'gatekeeper': decision, 'published_weights': config.publish_weights,
            'stage_seconds': progress['stage_seconds'], 'stage_compile_seconds': progress['stage_compile_seconds'],
            'round_timing': round_timing, 'timing_valid': state['timing_valid'],
            'elapsed_seconds': state['elapsed_seconds'], 'timing_basis': state['timing_basis'],
            'stage_attempts': progress['attempts'], 'training_epochs': training_plan['epochs'],
            **({'sampling': progress['sampling'], 'replay_credit': state['replay']['credit'],
                'actual_replay_ratio': after / state['replay']['rows']} if replay is not None else {})})
    except OSError as error:
        print(f'WARNING: round {number} committed, but result.json could not be written: {error}', file=sys.stderr, flush=True)
    print(f'Completed round {number}: source learner samples {before} -> {after}; '
          f'elapsed {state["elapsed_seconds"]:.3f}s ({state["timing_basis"]})', flush=True)


def validate_budget(iterations, max_seconds):
    if iterations is not None and (type(iterations) is not int or iterations < 1):
        raise ValueError('iterations must be a positive cumulative round limit')
    if max_seconds is not None and (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be finite and positive')
    if iterations is None and max_seconds is None:
        raise ValueError('Specify iterations or a time budget')


def run(root, config, iterations=None, binary=None, *, max_seconds=None, weights=None, inherited_fds=()):
    config.validate()
    validate_budget(iterations, max_seconds)
    binary = Path(binary or ROOT / 'build/katago').resolve()
    if not binary.is_file():
        raise ValueError(f'Build the native engine first: {binary}')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another controller is using {root}') from error
        state_path = root / 'state.json'
        if state_path.exists():
            if weights is not None:
                raise ValueError('An existing run owns its initialization; resume without weights')
            state = json.loads(state_path.read_text())
            if json.loads((root / 'config.json').read_text()) != config.serializable():
                raise ValueError('Run configuration differs from the existing run')
        else:
            if any(p.name != '.lock' for p in root.iterdir()):
                raise ValueError('A new run requires an empty output directory')
            origin = None if weights is None else Path(weights).resolve()
            if origin is not None and not origin.is_file():
                raise ValueError(f'Initial weights not found: {origin}')
            save_json(root / 'config.json', config.serializable())
            if config.is_muzero:
                save_json(root / 'muzero_network.json', config.muzero_network)
            source = json.loads((ROOT / 'reference_sources.json').read_text())
            revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
            snapshot_source(root / 'source.tar.gz')
            save_json(root / 'source.json', {'etazero_version': __version__, 'repository_commit': revision,
                                            'source': source, 'binary': str(binary), 'snapshot': 'source.tar.gz',
                                            'initial_weights': str(origin) if origin is not None else None})
            if origin is not None:
                (root / 'initial').mkdir()
                shutil.copyfile(origin, root / 'initial/weights.ckpt')
            state = dict(completed_rounds=0, accepted_model=None, active_round=None, phase='idle', total_samples=0,
                         wall_seconds=0.0, compile_seconds=0.0, elapsed_seconds=0.0, timing_valid=True,
                         timing_basis='committed_wall_excluding_compile')
            if config.replay_ratio is not None:
                state['replay'] = dict(rows=0, origin_rows=None, credit=0.0, history=[])
            save_json(state_path, state)
        if not state['timing_valid']:
            print('WARNING: cumulative timing is incomplete; budgets use observed wall time, not equal-time comparison data',
                  file=sys.stderr, flush=True)
        plotted = False
        while state['active_round'] is not None or iterations is None or state['completed_rounds'] < iterations:
            if state['active_round'] is None and max_seconds is not None and state['elapsed_seconds'] >= max_seconds:
                print(f'Time budget reached after round {state["completed_rounds"]}: '
                      f'{state["elapsed_seconds"]:.3f}s >= {max_seconds:.3f}s', flush=True)
                break
            run_round(root, config, state, binary, (lock.fileno(), *inherited_fds))
            refresh_plots(root, config)
            plotted = True
        if not plotted and state['completed_rounds']:
            refresh_plots(root, config)
        return state


def refresh_plots(root, config):
    if config.auto_plot:
        try:
            from .plotting import plot_run
            plot_run(root)
        except Exception as error:
            print(f'WARNING: plotting failed; committed training remains valid: {error}', file=sys.stderr, flush=True)
