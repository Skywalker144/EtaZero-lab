import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest
import torch
from etazero.config import ROOT, load_config
from etazero.experiment import (experiment_plan, initialization_key, prepare_initializations,
                                arm_progress, write_arm_config, Scheduler)
from etazero.plotting import run_history, training_figure, performance_figure, journal_events
from etazero.storage import save_json, load_json, sha256


def umbrella(tmp_path):
    directory = tmp_path/'umbrella'; directory.mkdir()
    (directory/'exp.cfg').write_text('[experiment]\nmax_iteration = 2\nmax_seconds = 0\narm_gpus = 0\nshared_init = true\n')
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
    config = load_config(ROOT/'configs/smoke_test')
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
    assert list(figure.axes[4].lines[0].get_ydata()) == [10]
    assert list(figure.axes[2].lines[0].get_ydata()) == [6]
    assert [line.get_label() for line in figure.axes[3].lines] == [
        'Policy', 'Opponent policy', 'Soft policy', 'Soft opponent policy', 'Value']
    assert sum(line.get_ydata()[0] for line in figure.axes[3].lines) == 6
    performance = performance_figure(rows, 8)
    assert list(performance.axes[1].lines[0].get_ydata()) == [8]
    assert list(performance.axes[3].lines[0].get_ydata()) == [4]
    figure.clear(); performance.clear()
    path.write_text('{broken\n'+json.dumps(stats)+'\n')
    with pytest.raises(ValueError, match='Corrupt journal'):
        list(journal_events(path))


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
