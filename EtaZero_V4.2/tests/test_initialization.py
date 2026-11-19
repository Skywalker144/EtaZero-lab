"""Weight sharing must not share a learner's optimizer, clocks or RNG state."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from etazero import ROOT
from etazero.initialization import initialization_key, prepare_shared_initializations
from etazero.experiment import experiment_plan, run_experiment
from katago.train.load_model import load_checkpoint, load_model_state_dict
from test_controller_recovery import recovery_config

controller = importlib.import_module('etazero.run')


def initialize(root, config, weights):
    root.mkdir()
    (root / 'initial').mkdir()
    (root / 'initial/weights.ckpt').write_bytes(weights.read_bytes())
    if config.is_muzero:
        controller.save_json(root / 'muzero_network.json', config.muzero_network)
    command = [*controller.learner_command(root, config), '-initialize-only']
    with (root / 'initialize.log').open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT / 'python',
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
            stdout=log, stderr=subprocess.STDOUT)
    return result


def test_initialization_groups_share_networks_not_search_or_optimizer(tmp_path):
    config = recovery_config(tmp_path, 'alphazero')
    assert initialization_key(config) == initialization_key(replace(config,
        algorithm='gumbel_alphazero', train_args=('-no-compile',), bootstrap_games=20))
    assert initialization_key(config) != initialization_key(replace(config, game='hex'))
    assert initialization_key(config) != initialization_key(replace(config, canvas=7))
    mu = recovery_config(tmp_path, 'muzero')
    assert initialization_key(mu) == initialization_key(replace(mu,
        algorithm='gumbel_muzero', muzero_unroll_steps=5))
    assert initialization_key(config) != initialization_key(mu)


@pytest.mark.parametrize('algorithm', ['alphazero', 'muzero'])
def test_source_weights_only_initialization_resets_all_training_state(tmp_path, algorithm):
    config = recovery_config(tmp_path, algorithm)
    arms = [dict(name='a', run_dir=tmp_path / 'a', config=config),
            dict(name='b', run_dir=tmp_path / 'b', config=replace(config, algorithm='gumbel_' + algorithm))]
    shared = prepare_shared_initializations(tmp_path / 'experiment', arms)
    assert shared['a'] == shared['b']
    original = shared['a'].read_bytes()
    assert prepare_shared_initializations(tmp_path / 'experiment', arms) == shared
    assert shared['a'].read_bytes() == original
    donor = load_checkpoint(shared['a'])
    assert donor['train_state']['global_step_samples'] == donor['train_state']['export_cycle_counter'] == 0
    # Deliberately poison every non-weight state. This must still initialize an
    # independent learner; adopting an initial checkpoint would fail or inherit it.
    weights = tmp_path / 'donor.ckpt'
    donor.update(optimizer='must not be read', train_state={'global_step_samples': 1234567},
                 swa_model='must not be read', learner_runtime='must not be read',
                 running_metrics={'poison': True}, last_val_metrics={'poison': True})
    torch.save(donor, weights)
    independent = replace(config, train_args=('-no-compile', '-lookahead-k', '2',
                                               '-no-longterm-checkpoints'))  # SGD, donor used AdamW.
    result = initialize(tmp_path / 'learner', independent, weights)
    assert result.returncode == 0, (tmp_path / 'learner/initialize.log').read_text()
    fresh = load_checkpoint(tmp_path / 'learner/train/checkpoint.ckpt')
    assert fresh['train_state']['global_step_samples'] == fresh['train_state']['export_cycle_counter'] == 0
    assert fresh['train_state']['optimizer_name'] == 'SGD'
    assert fresh['optimizer']['state'] == {}
    assert fresh['swa_model']['n_averaged'] == 0
    assert 'poison' not in fresh['running_metrics'] and fresh['last_val_metrics'] == {}
    assert fresh['running_metrics']['sums']['nsamp'] == 0
    assert isinstance(fresh['learner_runtime'], dict)
    expected = load_model_state_dict(donor)
    actual = load_model_state_dict(fresh)
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    # Existing arm state makes the shared artifact unnecessary on resume.
    for arm in arms:
        Path(arm['run_dir']).mkdir()
        (Path(arm['run_dir']) / 'state.json').write_text('{}')
    shared['a'].unlink()
    assert prepare_shared_initializations(tmp_path / 'experiment', arms) == {}


def test_initial_weights_reject_different_network_and_reserved_cli(tmp_path):
    config = recovery_config(tmp_path, 'alphazero')
    with pytest.raises(ValueError, match='outer controller'):
        replace(config, train_args=('-initial-weights=x',)).validate()
    weights = tmp_path / 'wrong.ckpt'
    torch.save({'config': {'version': -1}, 'model': {}}, weights)
    result = initialize(tmp_path / 'learner', config, weights)
    assert result.returncode != 0
    assert 'does not match the requested network' in (tmp_path / 'learner/initialize.log').read_text()
    assert not (tmp_path / 'learner/train/checkpoint.ckpt').exists()
    with pytest.raises(ValueError, match='not found'):
        controller.run(tmp_path / 'run', config, 1, binary='/bin/false', weights=tmp_path / 'absent')
    assert not (tmp_path / 'run/config.json').exists()


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real native CUDA and learner')
@pytest.mark.parametrize('algorithm', ['alphazero', 'muzero'])
def test_two_real_arms_share_weights_then_resume_independently(tmp_path, algorithm):
    config = replace(recovery_config(tmp_path, algorithm), auto_plot=False)
    work = tmp_path / 'experiment'
    arms = [dict(name='a', run_dir=work / 'runs/a', config=config),
            dict(name='b', run_dir=work / 'runs/b', config=replace(config, algorithm='gumbel_' + algorithm))]
    for arm in arms:
        controller.save_json(tmp_path / (arm['name'] + '.json'), arm['config'].serializable())
    experiment = tmp_path / 'experiment.json'
    controller.save_json(experiment, dict(arms={arm['name']: arm['name'] + '.json' for arm in arms}, iterations=1))
    plan = experiment_plan(experiment, work)
    assert run_experiment(plan) == 0
    origins = {name: Path(path) for name, path in json.loads((work / 'plan.json').read_text())['initializations'].items()}
    expected = load_model_state_dict(load_checkpoint(origins['a']))
    for arm in arms:
        state = json.loads((arm['run_dir'] / 'state.json').read_text())
        assert state['completed_rounds'] == 1 and state['total_samples'] > 0
        initial = load_checkpoint(arm['run_dir'] / 'initial/checkpoint.ckpt')
        assert initial['train_state']['global_step_samples'] == 0
        for key, value in load_model_state_dict(initial).items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        result = json.loads((arm['run_dir'] / 'rounds/000001/result.json').read_text())
        assert result['input_model'] == 'initial'
        assert json.loads((arm['run_dir'] / 'rounds/000000/result.json').read_text())['stage_seconds']['initialize'] > 0
        trained = load_model_state_dict(load_checkpoint(arm['run_dir'] / 'train/checkpoint.ckpt'))
        assert any(not torch.equal(trained[key], value) for key, value in expected.items())
        with pytest.raises(ValueError, match='resume without weights'):
            controller.run(arm['run_dir'], arm['config'], 2, weights=origins[arm['name']])
    origins['a'].unlink()
    saved = {arm['name']: ((arm['run_dir'] / 'initial/checkpoint.ckpt').read_bytes(),
             load_checkpoint(arm['run_dir'] / 'train/checkpoint.ckpt')) for arm in arms}
    # A completed experiment is a no-op even without its shared origin; increasing
    # the target resumes each arm's own learner, without regenerating weights.
    before = {arm['name']: (arm['run_dir'] / 'train/checkpoint.ckpt').stat().st_mtime_ns for arm in arms}
    assert run_experiment(plan) == 0
    assert all((arm['run_dir'] / 'train/checkpoint.ckpt').stat().st_mtime_ns == before[arm['name']] for arm in arms)
    assert run_experiment({**plan, 'iterations': 2}) == 0
    for arm in arms:
        initial_bytes, first = saved[arm['name']]
        state = json.loads((arm['run_dir'] / 'state.json').read_text())
        second = load_checkpoint(arm['run_dir'] / 'train/checkpoint.ckpt')
        assert state['total_samples'] > first['train_state']['global_step_samples']
        assert (arm['run_dir'] / 'initial/checkpoint.ckpt').read_bytes() == initial_bytes
        assert 'Loading initial weights only:' not in (arm['run_dir'] / 'rounds/000002/logs/train.log').read_text()
