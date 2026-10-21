from config_samples import CONFIGS, derive_config
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
from etazero.plotting import (run_history, round_history, training_figure, loss_figure, performance_figure,
                             journal_events, METRICS)
from etazero.storage import save_json, load_json, sha256


def umbrella(tmp_path):
    directory = tmp_path/'umbrella'; directory.mkdir()
    (directory/'exp.cfg').write_text('[experiment]\nmax_iteration = 2\nmax_hours = 0\narm_gpus = 0\nshared_init = true\nautoelo = false\n')
    for name in ('a', 'b'):
        arm = directory/name; arm.mkdir()
        derive_config(arm, CONFIGS/'smoke_test', f'run_dir = {tmp_path/name}\n')
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
        experiment_plan(directory, environ={'MAX_ITERS': '0', 'MAX_TIME_HOURS': '0'})
    with pytest.raises(ValueError, match='repeat'):
        experiment_plan(directory, environ={'ARM_GPUS': '0,0'})
    with pytest.raises(ValueError, match='nonnegative'):
        experiment_plan(directory, environ={'MAX_TIME_HOURS': 'nan'})
    with pytest.raises(ValueError, match='non-overlapping'):
        derive_config(directory/'b', CONFIGS/'smoke_test', f'run_dir = {tmp_path/"a/subdir"}\n')
        experiment_plan(directory, environ={})


def test_scheduler_local_budget_and_disabled_evaluation_scope(tmp_path):
    directory = umbrella(tmp_path)
    for name in ('elo.cfg', 'match.cfg'):
        (directory/name).write_text('Unused when automatic evaluation is disabled.\n')
    (directory/'exp.cfg.local').write_text('[experiment]\nmax_iteration=3\nmax_hours=4.5\n')
    plan = experiment_plan(directory, environ={})
    assert plan['settings']['max_iteration'] == 3
    assert plan['settings']['max_hours'] == 4.5
    assert 'elo' not in plan
    overridden = experiment_plan(directory, environ={'MAX_ITERS': '4', 'MAX_TIME_HOURS': '6'})
    assert overridden['settings']['max_iteration'] == 4
    assert overridden['settings']['max_hours'] == 6


@pytest.mark.parametrize('hours', [6, 10, 4.5])
def test_experiment_parses_hour_budget_from_config(tmp_path, hours):
    directory = umbrella(tmp_path)
    path = directory/'exp.cfg'
    path.write_text(path.read_text().replace('max_iteration = 2', 'max_iteration = 0')
                   .replace('max_hours = 0', f'max_hours = {hours}'))
    plan = experiment_plan(directory, environ={})
    assert plan['settings']['max_hours'] == hours


@pytest.mark.parametrize('hours', ['-0.5', 'nan', 'inf', '-inf', '1e308'])
def test_experiment_rejects_invalid_hour_budget(tmp_path, hours):
    directory = umbrella(tmp_path)
    with pytest.raises(ValueError, match='nonnegative and finite'):
        experiment_plan(directory, environ={'MAX_TIME_HOURS': hours})


@pytest.mark.parametrize('elapsed, expected', [(16199.5, 'pending'), (16200, 'complete'),
                                              (16200.5, 'complete')])
def test_hour_budget_completion_uses_elapsed_seconds(tmp_path, elapsed, expected):
    save_json(tmp_path/'.internal/run.json', {})
    save_json(tmp_path/'.internal/state.json', {'iteration': 1, 'elapsed_seconds': elapsed})
    assert arm_progress({'run_dir': str(tmp_path)}, {'max_iteration': 0, 'max_hours': 4.5}) == expected


@pytest.mark.parametrize('section', ['elo', 'match', 'opening', 'hex_opening'])
def test_scheduler_rejects_evaluation_sections(tmp_path, section):
    directory = umbrella(tmp_path)
    with (directory/'exp.cfg').open('a') as stream:
        stream.write(f'\n[{section}]\n')
    with pytest.raises(ValueError, match='Wrong section/file ownership'):
        experiment_plan(directory, environ={})


def test_evaluation_is_resolved_before_training_and_dry_run_reuses_plan(tmp_path, monkeypatch, capsys):
    from etazero import experiment
    directory = umbrella(tmp_path)
    (directory/'match.cfg').write_text(f'@include "{CONFIGS / "baseline/match.cfg"}"\n'
                                      '[match]\nboard_size=5\nvisits=17\n')
    (directory/'elo.cfg').write_text('[elo]\ngames_per_pair=12\n')
    plan = experiment_plan(directory, environ={'AUTOELO': 'true'}, work_dir=tmp_path/'work')
    assert plan['elo']['config']['match']['games'] == 12
    assert plan['elo']['config']['match']['visits'] == 17
    assert plan['elo']['cuda_visible_devices'] == '0'
    assert not (tmp_path/'work').exists()
    (directory/'match.cfg').unlink()
    monkeypatch.setattr(experiment, 'experiment_plan', lambda *args, **kwargs: plan)
    monkeypatch.setattr(sys, 'argv', ['autoexp', '--config-dir', str(directory), '--dry-run'])
    assert experiment.main() == 0
    assert json.loads(capsys.readouterr().out) == plan


@pytest.mark.parametrize('failure', ['missing_match', 'invalid_elo', 'board', 'canvas',
                                    'reuse_tree', 'use_graph_search', 'root_num_symmetries_to_sample'])
def test_evaluation_errors_are_rejected_before_creating_work(tmp_path, failure):
    directory = umbrella(tmp_path)
    protocol = ('[match]\nboard_size=5\nreuse_tree=false\nuse_graph_search=false\n'
                'root_num_symmetries_to_sample=1\n')
    if failure == 'invalid_elo':
        (directory/'elo.cfg').write_text('[elo]\nsample_seconds=0\n')
    elif failure == 'board':
        protocol = protocol.replace('board_size=5', 'board_size=7')
    elif failure == 'canvas':
        (directory/'b/net.cfg.local').write_text('[network]\ncanvas=7\n')
    elif failure in ('reuse_tree', 'use_graph_search', 'root_num_symmetries_to_sample'):
        derive_config(directory/'b', CONFIGS/'muzero', f'run_dir={tmp_path/"b"}\n')
        if failure == 'root_num_symmetries_to_sample':
            protocol = protocol.replace(f'{failure}=1', f'{failure}=2')
        else:
            protocol = protocol.replace(f'{failure}=false', f'{failure}=true')
    if failure != 'missing_match':
        (directory/'match.cfg').write_text(f'@include "{CONFIGS / "baseline/match.cfg"}"\n'+protocol)
    with pytest.raises((FileNotFoundError, ValueError), match={
            'missing_match': 'match.cfg', 'invalid_elo': 'sample_seconds', 'board': 'exceeds model canvas',
            'canvas': 'same canvas'}.get(failure, 'MuZero matches require')):
        experiment_plan(directory, environ={'AUTOELO': 'true'}, work_dir=tmp_path/'work')
    assert not (tmp_path/'work').exists()


def test_shared_weights_group_by_network_seed_and_reuse_atomic_payload(tmp_path):
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
    assert not list((tmp_path/'.internal/initializations').glob('*.json'))
    assert all(set(origin) == {'key', 'path'} for origin in origins.values())


def test_experiment_initializes_and_schedules_all_arms(tmp_path, monkeypatch):
    directory = umbrella(tmp_path)
    work = tmp_path/'work'
    binary = tmp_path/'binary'; binary.write_bytes(b'test binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: None)
    initialized, started = [], []
    def initialize(work, arms):
        initialized.append([arm['name'] for arm in arms])
        return {}
    monkeypatch.setattr('etazero.experiment.prepare_initializations', initialize)
    monkeypatch.setattr(Scheduler, 'start', lambda self, arm, slot, gpu: started.append(arm['name']))
    plan = experiment_plan(directory, environ={}, work_dir=work)
    assert run_experiment(plan, binary) == 0
    assert initialized == [['a', 'b']]
    assert started == ['a', 'b']


def journal(root, events):
    # Synthetic per-batch fixtures are serialized as the current accumulated format.
    from etazero.training_metrics import TrainingMetrics
    seen = {}
    for event in events:
        if event['event'] != 'update':
            continue
        batches = seen.setdefault(event['iteration'], {})
        batches[event['step']] = event
        metrics = TrainingMetrics()
        for step, batch in sorted(batches.items()):
            if step > event['step']:
                continue
            metrics.add({k: batch[k] for k in METRICS if k != 'grad_norm'},
                        batch['grad_norm'], batch.get('amp_skipped', False), False,
                        batch.get('step_losses'))
        event['round_metrics'] = metrics.summary()
    path = root/'logs/events.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    return path


def test_history_counts_amp_consumption_and_finite_gradients_separately(tmp_path):
    save_json(tmp_path/'.internal/state.json', {'iteration': 2, 'checkpoint': {'id': 'c', 'path': 'checkpoints/c.pt'}})
    update = {'event': 'update', 'iteration': 1, 'policy_loss': 1, 'opponent_policy_loss': 1,
              'soft_policy_loss': 1, 'soft_opponent_policy_loss': 1, 'value_loss': 1}
    update.update({k:0 for k in ('td_value_long_loss','td_value_mid_loss','td_value_short_loss',
                                'long_optimistic_policy_loss','short_optimistic_policy_loss','shortterm_value_error_loss')})
    journal(tmp_path, [{**update, 'step': 1, 'amp_skipped': True, 'loss': 6, 'grad_norm': float('inf')},
                       {**update, 'step': 2, 'amp_skipped': False, 'loss': 10, 'grad_norm': 5}])
    row = round_history(tmp_path,1,2,0)
    assert row['steps'] == 2 and row['amp_skipped_steps'] == 1 and row['gradient_steps'] == 1
    assert row['loss'] == 8 and row['grad_norm'] == 5


def test_history_uses_committed_updates_and_completed_selfplay(tmp_path):
    save_json(tmp_path/'.internal/state.json', {'iteration': 2, 'checkpoint': {'id': 'base', 'path': 'checkpoints/base.pt', 'total_steps': 1}})
    save_json(tmp_path/'.internal/iterations/000002/learner.json', {'checkpoint': {'id': 'progress', 'path': 'checkpoints/progress.pt', 'total_steps': 2}})
    stats = {'event': 'selfplay_statistics', 'iteration': 1, 'games': 4, 'rows': 40, 'plies': 48,
             'avg_game_length': 12, 'avg_rows_per_game': 10, 'black_wins': 2, 'white_wins': 1, 'draws': 1}
    update = {'event': 'update', 'iteration': 1, 'total_steps': 1, 'step': 1,
              'loss': 6, 'policy_loss': 2, 'opponent_policy_loss': .5,
              'soft_policy_loss': 2, 'soft_opponent_policy_loss': .5, 'value_loss': 1, 'grad_norm': 5}
    update.update({k:0 for k in ('td_value_long_loss','td_value_mid_loss','td_value_short_loss',
                                'long_optimistic_policy_loss','short_optimistic_policy_loss','shortterm_value_error_loss')})
    infer = {'event': 'inference', 'evaluator': 'network', 'iteration': 1, 'attempt': 'a', 'worker_id': 0,
             'requests': 8, 'batches': 2, 'queue_wait_us': 24, 'submitted': 10, 'cache_hits': 2}
    path = journal(tmp_path, [stats, stats, {'event': 'iteration_complete', 'iteration': 1, 'unique_rows': 40, 'total_samples': 8},
                             update, update, {**update, 'step': 2, 'loss': 999},
                             {**update, 'iteration': 2, 'step': 1, 'loss': 7},
                             {**stats, 'iteration': 2}, infer, infer,
                             {'event': 'phase_end', 'iteration': 1, 'phase': 'selfplay', 'seconds': 2},
                             {'event': 'phase_end', 'iteration': 1, 'phase': 'selfplay', 'seconds': 3}])
    with path.open('a') as file:
        file.write('{"event":')
    rows = [round_history(tmp_path,1,1,0)]
    assert len(rows) == 1 and rows[0]['steps'] == 1 and rows[0]['loss'] == 6
    assert rows[0]['total_samples'] == 8
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
    values = dict.fromkeys(METRICS, 1.0); values['loss'] = 11.0
    def update(iteration, identity):
        return dict(values, event='update', iteration=iteration, step=1)
    def complete(iteration):
        return dict(event='iteration_complete', iteration=iteration, unique_rows=100*iteration,
                    total_samples=16*iteration)
    def validation(iteration, loss):
        return dict(values, event='validation', iteration=iteration, samples=16, batches=2, loss=loss)
    journal(tmp_path, [update(1, 'abandoned'),
                       dict(event='plan', iteration=1), update(1, 'one'), complete(1),
                       update(2, 'two'), validation(2, 12), validation(2, 12), complete(2),
                       validation(3, 999), dict(event='plan', iteration=3), update(3, 'three'),
                       dict(event='validation', iteration=3, samples=0, batches=0,
                            reason='no_complete_validation_batch'), complete(3),
                       update(4, 'current'), validation(4, 999), complete(4)])
    history = [round_history(tmp_path,i,1,0) for i in (1,2,3)]
    assert [r['iteration'] for r in history] == [1, 2, 3]
    assert 'validation' not in history[0]
    assert history[1]['validation']['loss'] == 12
    assert history[2]['validation']['samples'] == 0 and 'loss' not in history[2]['validation']
    figure = loss_figure(history)
    assert len(figure.axes) == 12
    train, val = figure.axes[0].lines
    assert list(train.get_xdata()) == [16, 32, 48]
    assert list(train.get_ydata()) == [11, 11, 11]
    np.testing.assert_equal(val.get_ydata(), [np.nan, 12, np.nan])
    assert val.get_linestyle() == '-' and val.get_marker() == 'None'
    figure.clear()
    history[1]['q_winloss_loss'] = .25
    history[1]['validation']['q_winloss_loss'] = .5
    figure = loss_figure(history)
    assert len(figure.axes) == 13 and figure.axes[-1].get_title() == 'W-L Q'
    np.testing.assert_equal(figure.axes[-1].lines[0].get_ydata(), [np.nan, .25, np.nan])
    assert figure.axes[-1].lines[1].get_ydata()[1] == .5
    figure.clear()


def test_budget_completion_only_reads_saved_progress(tmp_path):
    directory = umbrella(tmp_path)
    plan = experiment_plan(directory, environ={})
    arm = plan['arms'][0]; root = Path(arm['run_dir'])
    assert arm_progress(arm, plan['settings']) == 'pending'
    save_json(root/'.internal/run.json', {'config': {'old': 'configuration'},
              'weights_initialization': {'path': str(tmp_path/'missing.pt'), 'sha256': 'old'}})
    save_json(root/'.internal/state.json', {'iteration': 2, 'elapsed_seconds': 0})
    journal(root, [{'event': 'session_start', 'utc': '2026-01-01T00:00:00+00:00'}])
    assert arm_progress(arm, plan['settings']) == 'pending'
    save_json(root/'.internal/state.json', {'iteration': 3, 'elapsed_seconds': 10})
    assert arm_progress(arm, plan['settings']) == 'complete'
    assert arm_progress(arm, {**plan['settings'], 'max_iteration': 4, 'max_hours': 9/3600}) == 'complete'
    arm['config']['run']['seed'] += 1
    assert arm_progress(arm, plan['settings']) == 'complete'
    (root/'.internal/run.json').unlink()
    assert arm_progress(arm, plan['settings']) == 'pending'


@pytest.mark.parametrize('shared_init', [True, False])
@pytest.mark.parametrize('budget', ['iterations', 'hours'])
def test_experiment_adds_arms_skips_complete_and_resumes_pending(tmp_path, monkeypatch, capsys,
                                                               shared_init, budget):
    from etazero.config import fingerprint
    directory = umbrella(tmp_path)
    env = {'SHARED_INIT': str(shared_init).lower()}
    if budget == 'hours':
        env.update(MAX_ITERS='0', MAX_TIME_HOURS=str(10/3600))
    work = tmp_path/'work'
    binary = tmp_path/'binary'; binary.write_bytes(b'test binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: None)
    started = []

    def finish_arm(scheduler, arm, slot, gpu):
        # Exercise scheduling and budgets without training; runs own their origin.
        started.append(arm['name'])
        origin = scheduler.initializations.get(arm['name'])
        origin = {k: origin[k] for k in ('path',)} if origin else None
        root = Path(arm['run_dir'])
        if not (root/'.internal/run.json').exists():
            save_json(root/'.internal/run.json', {'config': arm['config'],
                      'config_id': fingerprint(arm['config']), 'weights_initialization': origin})
        save_json(root/'.internal/state.json', {'iteration': 3, 'elapsed_seconds': 10})
        scheduler.states[arm['name']] = {'status': 'complete'}

    monkeypatch.setattr(Scheduler, 'start', finish_arm)
    first = experiment_plan(directory, environ=env, work_dir=work)
    assert run_experiment(first, binary) == 0
    assert started == ['a', 'b']
    old_plan = load_json(work/'.internal/plan.json')
    old_origins = load_json(work/'.internal/plan.json')['initializations']
    old_payloads = {v['path']: Path(v['path']).read_bytes() for v in old_origins.values()}
    complete_state = (tmp_path/'a/.internal/state.json').read_bytes()
    # b was interrupted below the budget; a has already reached it.
    save_json(tmp_path/'b/.internal/state.json', {'iteration': 2, 'elapsed_seconds': 5})
    seed = first['arms'][0]['config']['run']['seed']+1
    for name, arm_seed in [('0_new', None), ('c', seed)]:
        arm_dir = directory/name; arm_dir.mkdir()
        extra = f'seed = {arm_seed}\n' if arm_seed is not None else ''
        derive_config(arm_dir, CONFIGS/'smoke_test', f'run_dir = {tmp_path/name}\n'+extra)
    started.clear(); capsys.readouterr()
    expanded = experiment_plan(directory, environ=env, work_dir=work)
    assert run_experiment(expanded, binary) == 0
    assert started == ['0_new', 'b', 'c']
    assert 'skipped completed a' in capsys.readouterr().out
    assert (tmp_path/'a/.internal/state.json').read_bytes() == complete_state
    current = load_json(work/'.internal/plan.json')
    assert [a['name'] for a in current['arms']] == ['0_new', 'a', 'b', 'c']
    assert [a for a in current['arms'] if a['name'] in {'a', 'b'}] == old_plan['arms']
    origins = load_json(work/'.internal/plan.json')['initializations']
    if shared_init:
        assert set(origins) == {'0_new', 'c'}
        assert origins['0_new'] == old_origins['a']
        assert origins['c'] != origins['0_new']
    else:
        assert origins == {}
    assert all(Path(path).read_bytes() == payload for path, payload in old_payloads.items())
    assert all(s['status'] == 'complete' for s in load_json(work/'.internal/status.json')['arms'].values())
    started.clear()
    assert run_experiment(expanded, binary) == 0
    assert started == []


@pytest.mark.parametrize('change', ['config', 'architecture', 'run_dir', 'removed', 'shared_init', 'umbrella'])
def test_experiment_uses_current_plan_and_ignores_old_identity(tmp_path, monkeypatch, change):
    directory = umbrella(tmp_path)
    work = tmp_path/'work'
    binary = tmp_path/'binary'; binary.write_bytes(b'test binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: None)
    calls = []
    monkeypatch.setattr(Scheduler, 'run', lambda _: calls.append(True) or 0)
    plan = experiment_plan(directory, environ={}, work_dir=work)
    assert run_experiment(plan, binary) == 0
    manifest = work/'.internal/identity.json'
    assert not manifest.exists()
    save_json(manifest, {'umbrella': plan['umbrella'], 'arms': plan['arms'],
                         'shared_init': plan['settings']['shared_init']})
    original = manifest.read_bytes()
    if change == 'config':
        plan['arms'][0]['config']['run']['seed'] += 1
    elif change == 'architecture':
        plan['arms'][0]['config']['network'].update(architecture='plain', channels=128, blocks=10)
    elif change == 'run_dir':
        plan['arms'][0]['run_dir'] = str(tmp_path/'other')
    elif change == 'removed':
        plan['arms'].pop(0)
    elif change == 'shared_init':
        plan['settings']['shared_init'] = False
    else:
        plan['umbrella'] = str(tmp_path/'other')
    assert run_experiment(plan, binary) == 0
    assert calls == [True, True]
    current = load_json(work/'.internal/plan.json')
    assert current['arms'] == plan['arms']
    assert current['umbrella'] == plan['umbrella']
    assert current['settings'] == plan['settings']
    assert manifest.read_bytes() == original


def test_experiment_accepts_execution_changes(tmp_path, monkeypatch):
    directory = umbrella(tmp_path)
    binary = tmp_path/'binary'
    binary.write_bytes(b'test binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: None)
    monkeypatch.setattr(Scheduler, 'run', lambda _: 0)
    plan = experiment_plan(directory, environ={}, work_dir=tmp_path/'work')
    assert run_experiment(plan, binary) == 0
    plan['arms'][0]['config']['run']['cpu_threads'] = 3
    plan['arms'][0]['config']['training']['checkpoint_every'] = 1
    assert run_experiment(plan, binary) == 0
    current = load_json(Path(plan['work_dir'])/'.internal/plan.json')
    assert current['arms'][0]['config']['run']['cpu_threads'] == 3


def test_experiment_restarts_archived_arm_with_new_architecture(tmp_path, monkeypatch, capsys):
    directory = umbrella(tmp_path)
    work = tmp_path/'work'
    binary = tmp_path/'binary'; binary.write_bytes(b'test binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: None)
    started = []

    def finish_arm(scheduler, arm, slot, gpu):
        started.append(arm['name'])
        root = Path(arm['run_dir'])
        origin = scheduler.initializations[arm['name']]
        save_json(root/'.internal/run.json', {'config': arm['config'],
                  'weights_initialization': {k: origin[k] for k in ('path',)}}, immutable=True)
        save_json(root/'.internal/state.json', {'iteration': 3, 'elapsed_seconds': 10})
        scheduler.states[arm['name']] = {'status': 'complete'}

    monkeypatch.setattr(Scheduler, 'start', finish_arm)
    first = experiment_plan(directory, environ={}, work_dir=work)
    assert run_experiment(first, binary) == 0
    old_origins = load_json(work/'.internal/plan.json')['initializations']
    root = tmp_path/'a'
    old_record = (root/'.internal/run.json').read_bytes()
    archive = tmp_path/'a.old'; root.rename(archive)
    with (directory/'a/net.cfg').open('a') as config:
        config.write('[network]\narchitecture = plain\nchannels = 128\nblocks = 10\n')
    started.clear(); capsys.readouterr()
    current = experiment_plan(directory, environ={}, work_dir=work)
    assert run_experiment(current, binary) == 0
    assert started == ['a']
    assert 'skipped completed b' in capsys.readouterr().out
    assert (archive/'.internal/run.json').read_bytes() == old_record
    assert load_json(root/'.internal/run.json')['config']['network']['architecture'] == 'plain'
    origins = load_json(work/'.internal/plan.json')['initializations']
    assert set(origins) == {'a'} and origins['a'] != old_origins['a']
    assert Path(old_origins['a']['path']).is_file()


@pytest.mark.parametrize('shared_init', [True, False])
def test_existing_runs_resume_without_scheduler_initialization(tmp_path, monkeypatch, shared_init):
    directory = umbrella(tmp_path)
    plan = experiment_plan(directory, environ={'SHARED_INIT': str(shared_init).lower()},
                           work_dir=tmp_path/'new_work')
    for arm in plan['arms']:
        root = Path(arm['run_dir'])
        save_json(root/'.internal/run.json', {'config': arm['config'],
                  'weights_initialization': {'path': str(tmp_path/'old_work/initial.pt'), 'sha256': 'old'}})
        save_json(root/'.internal/state.json', {'iteration': 2, 'elapsed_seconds': 5})
    binary = tmp_path/'binary'; binary.write_bytes(b'test binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: None)
    monkeypatch.setattr('etazero.experiment.prepare_initializations',
                        lambda work, arms: {} if not arms else pytest.fail('Existing runs requested initialization'))
    started = []

    def finish_arm(scheduler, arm, slot, gpu):
        assert scheduler.initializations == {}
        started.append(arm['name'])
        save_json(Path(arm['run_dir'])/'.internal/state.json', {'iteration': 3, 'elapsed_seconds': 10})
        scheduler.states[arm['name']] = {'status': 'complete'}

    monkeypatch.setattr(Scheduler, 'start', finish_arm)
    assert run_experiment(plan, binary) == 0
    assert started == ['a', 'b']
    assert load_json(Path(plan['work_dir'])/'.internal/plan.json')['initializations'] == {}


def test_scheduler_failure_interrupts_other_running_arms(tmp_path, monkeypatch):
    directory = umbrella(tmp_path)
    plan = experiment_plan(directory, environ={'ARM_GPUS': '0,1', 'MAX_TIME_HOURS': '4.5'},
                           work_dir=tmp_path/'work')
    scheduler = Scheduler(Path(plan['work_dir']), plan, ROOT/'build/etazero', {})
    processes = []
    actual_popen = subprocess.Popen
    def popen(command, **kwargs):
        assert float(command[command.index('--max-seconds')+1]) == 16200
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
