from config_samples import CONFIGS
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from etazero.config import ROOT, load_config
from etazero.experiment import (experiment_plan, initialization_key, prepare_initializations,
                                arm_progress, write_arm_config, run_experiment, Scheduler)
from etazero.plotting import (run_history, training_figure, loss_figure, performance_figure,
                             journal_events, METRICS)
from etazero.storage import save_json, load_json, sha256


def umbrella(tmp_path):
    directory = tmp_path/'umbrella'; directory.mkdir()
    (directory/'exp.cfg').write_text('[experiment]\nmax_iteration = 2\nmax_seconds = 0\narm_gpus = 0\nshared_init = true\nautoelo = false\n')
    for name in ('a', 'b'):
        arm = directory/name; arm.mkdir()
        (arm/'run.cfg').write_text(f'[run]\nextends = smoke_test\nrun_dir = {tmp_path/name}\n')
    return directory


def test_plan_is_read_only_and_resolves_inheritance(tmp_path):
    directory = umbrella(tmp_path)
    plan = experiment_plan(directory, environ={}, work_dir=tmp_path/'work')
    assert [a['name'] for a in plan['arms']] == ['a', 'b']
    assert plan['slots'] == ['0']
    assert plan['arms'][0]['config']['training']['train_steps'] == 4
    assert plan['arms'][0]['config']['devices'] == {'train': 'cuda:0', 'selfplay': 'cuda:0'}
    assert not (tmp_path/'work').exists() and not (tmp_path/'a').exists()
    resolved = tmp_path/'resolved'
    write_arm_config(plan['arms'][0]['config'], resolved)
    assert load_config(resolved) == plan['arms'][0]['config']
    with pytest.raises(ValueError, match='finite iteration or time'):
        experiment_plan(directory, environ={'MAX_ITERS': '0', 'MAX_TIME_SECONDS': '0'})
    with pytest.raises(ValueError, match='repeat'):
        experiment_plan(directory, environ={'ARM_GPUS': '0,0'})
    with pytest.raises(ValueError, match='nonnegative'):
        experiment_plan(directory, environ={'MAX_TIME_SECONDS': 'nan'})
    with pytest.raises(ValueError, match='non-overlapping'):
        (directory/'b/run.cfg').write_text(f'[run]\nextends = smoke_test\nrun_dir = {tmp_path/"a/subdir"}\n')
        experiment_plan(directory, environ={})


def test_shared_weights_group_by_network_seed_and_verify_payload(tmp_path):
    config = load_config(CONFIGS / 'smoke_test')
    alternate = copy.deepcopy(config); alternate['optimizer']['lr_scale'] *= 2
    different_seed = copy.deepcopy(config); different_seed['run']['seed'] += 1
    arms = [{'name': name, 'config': c} for name, c in [('a', config), ('b', alternate), ('c', different_seed)]]
    before = torch.get_rng_state().clone()
    origins = prepare_initializations(tmp_path, arms)
    assert torch.equal(before, torch.get_rng_state())
    assert origins['a'] == origins['b'] and origins['a'] != origins['c']
    assert initialization_key(config) == initialization_key(alternate)
    assert prepare_initializations(tmp_path, arms) == origins
    path = Path(origins['a']['path']); path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checksum'):
        prepare_initializations(tmp_path, arms)


def journal(root, events):
    path = root/'logs/events.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    return path


def test_history_counts_amp_consumption_and_finite_gradients_separately(tmp_path):
    save_json(tmp_path/'.internal/state.json', {'iteration': 2, 'checkpoint': {'id': 'c', 'path': 'checkpoints/c.pt'}})
    save_json(tmp_path/'checkpoints/c.json', {'parent': None, 'committed_updates': ['skip', 'success']})
    update = {'event': 'update', 'iteration': 1, 'policy_loss': 1, 'opponent_policy_loss': 1,
              'soft_policy_loss': 1, 'soft_opponent_policy_loss': 1, 'value_loss': 1}
    update.update({k:0 for k in ('td_value_long_loss','td_value_mid_loss','td_value_short_loss',
                                'long_optimistic_policy_loss','short_optimistic_policy_loss','shortterm_value_error_loss')})
    journal(tmp_path, [{**update, 'update_id': 'skip', 'amp_skipped': True, 'loss': 6, 'grad_norm': float('inf')},
                       {**update, 'update_id': 'success', 'amp_skipped': False, 'loss': 10, 'grad_norm': 5}])
    row, = run_history(tmp_path)
    assert row['steps'] == 2 and row['amp_skipped_steps'] == 1 and row['gradient_steps'] == 1
    assert row['loss'] == 8 and row['grad_norm'] == 5


def test_history_uses_committed_updates_and_completed_selfplay(tmp_path):
    save_json(tmp_path/'.internal/state.json', {'iteration': 2, 'checkpoint': {'id': 'base', 'path': 'checkpoints/base.pt', 'total_steps': 1}})
    save_json(tmp_path/'checkpoints/base.json', {'parent': None, 'committed_updates': ['kept']})
    save_json(tmp_path/'.internal/iterations/000002/learner.json', {'checkpoint': {'id': 'progress', 'path': 'checkpoints/progress.pt', 'total_steps': 2}})
    save_json(tmp_path/'checkpoints/progress.json', {'parent': {'id': 'base', 'path': 'checkpoints/base.pt'}, 'committed_updates': ['current']})
    stats = {'event': 'selfplay_statistics', 'iteration': 1, 'games': 4, 'rows': 40, 'plies': 48,
             'avg_game_length': 12, 'avg_rows_per_game': 10, 'black_wins': 2, 'white_wins': 1, 'draws': 1}
    update = {'event': 'update', 'iteration': 1, 'total_steps': 1, 'update_id': 'kept',
              'loss': 6, 'policy_loss': 2, 'opponent_policy_loss': .5,
              'soft_policy_loss': 2, 'soft_opponent_policy_loss': .5, 'value_loss': 1, 'grad_norm': 5}
    update.update({k:0 for k in ('td_value_long_loss','td_value_mid_loss','td_value_short_loss',
                                'long_optimistic_policy_loss','short_optimistic_policy_loss','shortterm_value_error_loss')})
    infer = {'event': 'inference', 'evaluator': 'network', 'iteration': 1, 'attempt': 'a', 'worker_id': 0,
             'requests': 8, 'batches': 2, 'queue_wait_us': 24, 'submitted': 10, 'cache_hits': 2}
    path = journal(tmp_path, [stats, stats, {'event': 'iteration_complete', 'iteration': 1, 'unique_rows': 40},
                             update, update, {**update, 'update_id': 'abandoned', 'loss': 999},
                             {**update, 'iteration': 2, 'update_id': 'current', 'loss': 7},
                             {**stats, 'iteration': 2}, infer, infer,
                             {'event': 'phase_end', 'iteration': 1, 'phase': 'selfplay', 'seconds': 2},
                             {'event': 'phase_end', 'iteration': 1, 'phase': 'selfplay', 'seconds': 3}])
    with path.open('a') as file:
        file.write('{"event":')
    rows = run_history(tmp_path)
    assert len(rows) == 1 and rows[0]['steps'] == 1 and rows[0]['loss'] == 6
    assert rows[0]['games'] == 4 and rows[0]['requests'] == 8 and rows[0]['phases']['selfplay'] == 5
    figure = training_figure(rows)
    assert len(figure.axes) == 6
    assert list(figure.axes[0].lines[0].get_ydata()) == [.5]
    assert list(figure.axes[1].lines[0].get_ydata()) == [12]
    assert list(figure.axes[4].lines[0].get_ydata()) == [5]
    assert list(figure.axes[5].lines[0].get_ydata()) == [.2]
    assert [line.get_label() for line in figure.axes[2].lines] == [
        'Policy', 'Opponent policy', 'Soft policy', 'Soft opponent policy',
        'Long optimistic', 'Short optimistic']
    assert [line.get_label() for line in figure.axes[3].lines] == [
        'Value', 'TD long', 'TD mid', 'TD short', 'Value error']
    assert sum(line.get_ydata()[0] for axis in figure.axes[2:4] for line in axis.lines) == 6
    performance = performance_figure(rows, 8)
    assert list(performance.axes[1].lines[0].get_ydata()) == [8]
    assert list(performance.axes[3].lines[0].get_ydata()) == [4]
    figure.clear(); performance.clear()
    path.write_text('{broken\n'+json.dumps(stats)+'\n')
    with pytest.raises(ValueError, match='Corrupt journal'):
        list(journal_events(path))


def test_validation_history_excludes_abandoned_and_uncommitted_rounds(tmp_path):
    save_json(tmp_path/'.internal/state.json', {'iteration': 4, 'checkpoint': {'id': 'c', 'path': 'checkpoints/c.pt'}})
    save_json(tmp_path/'checkpoints/c.json', {'parent': None, 'committed_updates': ['one', 'two', 'three']})
    values = dict.fromkeys(METRICS, 1.0); values['loss'] = 11.0
    def update(iteration, identity):
        return dict(values, event='update', iteration=iteration, update_id=identity)
    def complete(iteration):
        return dict(event='iteration_complete', iteration=iteration, unique_rows=100*iteration)
    def validation(iteration, loss):
        return dict(values, event='validation', iteration=iteration, samples=16, batches=2, loss=loss)
    journal(tmp_path, [update(1, 'abandoned'), validation(1, 999),
                       dict(event='plan', iteration=1), update(1, 'one'), complete(1),
                       update(2, 'two'), validation(2, 12), validation(2, 12), complete(2),
                       validation(3, 999), dict(event='plan', iteration=3), update(3, 'three'),
                       dict(event='validation', iteration=3, samples=0, batches=0,
                            reason='no_complete_validation_batch'), complete(3),
                       update(4, 'current'), validation(4, 999), complete(4)])
    history = run_history(tmp_path)
    assert [r['iteration'] for r in history] == [1, 2, 3]
    assert 'validation' not in history[0]
    assert history[1]['validation']['loss'] == 12
    assert history[2]['validation']['samples'] == 0 and 'loss' not in history[2]['validation']
    figure = loss_figure(history)
    assert len(figure.axes) == 12
    train, val = figure.axes[0].lines
    assert list(train.get_ydata()) == [11, 11, 11]
    np.testing.assert_equal(val.get_ydata(), [np.nan, 12, np.nan])
    assert val.get_linestyle() == '--' and val.get_marker() == 'o'
    figure.clear()
    history[1]['q_winloss_loss'] = .25
    history[1]['validation']['q_winloss_loss'] = .5
    figure = loss_figure(history)
    assert len(figure.axes) == 13 and figure.axes[-1].get_title() == 'W-L Q'
    np.testing.assert_equal(figure.axes[-1].lines[0].get_ydata(), [np.nan, .25, np.nan])
    assert figure.axes[-1].lines[1].get_ydata()[1] == .5
    figure.clear()


def test_budget_completion_requires_matching_run(tmp_path):
    directory = umbrella(tmp_path)
    plan = experiment_plan(directory, environ={})
    arm = plan['arms'][0]; root = Path(arm['run_dir'])
    assert arm_progress(arm, plan['settings']) == 'pending'
    from etazero.config import fingerprint
    save_json(root/'.internal/run.json', {'config': arm['config'], 'config_id': fingerprint(arm['config']),
                                          'weights_initialization': None})
    save_json(root/'.internal/state.json', {'iteration': 2, 'elapsed_seconds': 0})
    journal(root, [{'event': 'heartbeat', 'active_seconds': 10}])
    assert arm_progress(arm, plan['settings']) == 'pending'
    save_json(root/'.internal/state.json', {'iteration': 3, 'elapsed_seconds': 10})
    assert arm_progress(arm, plan['settings']) == 'complete'
    assert arm_progress(arm, {**plan['settings'], 'max_iteration': 4, 'max_seconds': 9}) == 'complete'
    arm['config']['run']['seed'] += 1
    with pytest.raises(ValueError, match='configuration differs'):
        arm_progress(arm, plan['settings'])


@pytest.mark.parametrize('shared_init', [True, False])
@pytest.mark.parametrize('budget', ['iterations', 'seconds'])
def test_experiment_adds_arms_skips_complete_and_resumes_pending(tmp_path, monkeypatch, capsys,
                                                               shared_init, budget):
    from etazero.config import fingerprint
    directory = umbrella(tmp_path)
    env = {'SHARED_INIT': str(shared_init).lower()}
    if budget == 'seconds':
        env.update(MAX_ITERS='0', MAX_TIME_SECONDS='10')
    work = tmp_path/'work'
    binary = tmp_path/'binary'; binary.write_bytes(b'test binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: None)
    started = []

    def finish_arm(scheduler, arm, slot, gpu):
        # Exercise the real scheduler and on-disk resume checks without training.
        started.append(arm['name'])
        origin = scheduler.initializations.get(arm['name'])
        origin = {k: origin[k] for k in ('path', 'sha256')} if origin else None
        root = Path(arm['run_dir'])
        save_json(root/'.internal/run.json', {'config': arm['config'],
                  'config_id': fingerprint(arm['config']), 'weights_initialization': origin})
        save_json(root/'.internal/state.json', {'iteration': 3, 'elapsed_seconds': 10})
        scheduler.states[arm['name']] = {'status': 'complete'}

    monkeypatch.setattr(Scheduler, 'start', finish_arm)
    first = experiment_plan(directory, environ=env, work_dir=work)
    assert run_experiment(first, binary) == 0
    assert started == ['a', 'b']
    old_identity = load_json(work/'.internal/identity.json')
    old_origins = load_json(work/'.internal/plan.json')['initializations']
    old_payloads = {v['path']: Path(v['path']).read_bytes() for v in old_origins.values()}
    complete_state = (tmp_path/'a/.internal/state.json').read_bytes()
    # b was interrupted below the budget; a has already reached it.
    save_json(tmp_path/'b/.internal/state.json', {'iteration': 2, 'elapsed_seconds': 5})
    seed = first['arms'][0]['config']['run']['seed']+1
    for name, arm_seed in [('0_new', None), ('c', seed)]:
        arm_dir = directory/name; arm_dir.mkdir()
        extra = f'seed = {arm_seed}\n' if arm_seed is not None else ''
        (arm_dir/'run.cfg').write_text(f'[run]\nextends = smoke_test\nrun_dir = {tmp_path/name}\n'+extra)
    started.clear(); capsys.readouterr()
    expanded = experiment_plan(directory, environ=env, work_dir=work)
    assert run_experiment(expanded, binary) == 0
    assert started == ['0_new', 'b', 'c']
    assert 'skipped completed a' in capsys.readouterr().out
    assert (tmp_path/'a/.internal/state.json').read_bytes() == complete_state
    identity = load_json(work/'.internal/identity.json')
    assert [a['name'] for a in identity['arms']] == ['0_new', 'a', 'b', 'c']
    assert [a for a in identity['arms'] if a['name'] in {'a', 'b'}] == old_identity['arms']
    origins = load_json(work/'.internal/plan.json')['initializations']
    if shared_init:
        assert origins['a'] == origins['b'] == origins['0_new'] == old_origins['a']
        assert origins['c'] != origins['a']
    else:
        assert origins == {}
    assert all(Path(path).read_bytes() == payload for path, payload in old_payloads.items())
    assert all(s['status'] == 'complete' for s in load_json(work/'.internal/status.json')['arms'].values())
    started.clear()
    assert run_experiment(expanded, binary) == 0
    assert started == []


@pytest.mark.parametrize('change', ['config', 'run_dir', 'removed', 'shared_init', 'umbrella'])
def test_experiment_rejects_existing_identity_changes(tmp_path, monkeypatch, change):
    directory = umbrella(tmp_path)
    work = tmp_path/'work'
    binary = tmp_path/'binary'; binary.write_bytes(b'test binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: None)
    calls = []
    monkeypatch.setattr(Scheduler, 'run', lambda _: calls.append(True) or 0)
    plan = experiment_plan(directory, environ={}, work_dir=work)
    assert run_experiment(plan, binary) == 0
    manifest = work/'.internal/identity.json'
    original = manifest.read_bytes()
    if change == 'config':
        plan['arms'][0]['config']['run']['seed'] += 1
    elif change == 'run_dir':
        plan['arms'][0]['run_dir'] = str(tmp_path/'other')
    elif change == 'removed':
        plan['arms'].pop(0)
    elif change == 'shared_init':
        plan['settings']['shared_init'] = False
    else:
        plan['umbrella'] = str(tmp_path/'other')
    with pytest.raises(ValueError, match='changed|removed'):
        run_experiment(plan, binary)
    assert calls == [True]
    assert manifest.read_bytes() == original


def test_scheduler_failure_interrupts_other_running_arms(tmp_path, monkeypatch):
    directory = umbrella(tmp_path)
    plan = experiment_plan(directory, environ={'ARM_GPUS': '0,1'}, work_dir=tmp_path/'work')
    scheduler = Scheduler(Path(plan['work_dir']), plan, ROOT/'build/etazero', {})
    processes = []
    actual_popen = subprocess.Popen
    def popen(command, **kwargs):
        # Fail one arm while another is genuinely running; no GPU work is needed
        # to verify process ownership, fail-fast and signal handling.
        config_dir = Path(command[command.index('--config-dir')+1])
        code = 'import time; time.sleep(.4); raise SystemExit(7)' if config_dir.name == 'a' else 'import time; time.sleep(30)'
        process = actual_popen([sys.executable, '-c', code], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr('etazero.experiment.subprocess.Popen', popen)
    with pytest.raises(RuntimeError, match='Experiment a'):
        scheduler.run()
    assert len(processes) == 2 and all(p.poll() is not None for p in processes)
    status = load_json(Path(plan['work_dir'])/'.internal/status.json')
    assert status['arms']['a']['status'] == 'failed'
    assert status['arms']['b']['status'] == 'interrupted'
