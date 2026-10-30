"""Actual source learner clocks, continuation, and MuZero model boundaries."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from etazero import ROOT
from katago.train import muzero_learner as learner, checkpoint_runtime
from katago.train.load_model import load_checkpoint, load_model
from katago.train.metrics_pytorch import Metrics
from katago.train.muzero_training import unroll_loss
from test_muzero_model import make_model, batch_for, metric_options, component_config, limited_cpu_threads
from katago.train.muzero_model import MuZeroConfig, MuZeroModel
from test_muzero_data import raw_data, shuffle_source, load


def test_forward_preserves_gradients_and_root_sample_clock():
    model = make_model(intermediate=True).train()
    reference = deepcopy(model)
    root, targets = batch_for(model)
    actions = torch.tensor([[0, 1], [-1, -1], [2, -1]])
    active = torch.tensor([[True, True, True], [True, False, False], [True, True, False]])
    wrapper = learner.MuZeroForward(model, no_compile=True)
    metrics = wrapper((root, targets, actions, active),
                      metrics_fn=Metrics(1, model.prediction).metrics_dict_batchwise,
                      metric_options=metric_options(True))
    loss, logs = unroll_loss(reference, root, targets, actions, active_steps=active, **metric_options(True))
    torch.testing.assert_close(metrics['loss_sum'], loss, rtol=0, atol=0)
    assert metrics['nsamp'] == 3
    assert metrics['mu_recurrent_samples_sum'] == 3
    for index, step_metrics in enumerate(logs):
        torch.testing.assert_close(metrics[f'mu_step_{index}_loss_sum'], step_metrics['loss_sum'])
        assert not metrics[f'mu_step_{index}_loss_sum'].requires_grad
    torch.testing.assert_close(metrics['wsum'], logs[0]['wsum'])
    metrics['loss_sum'].backward(); loss.backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert (actual.grad is None) == (expected.grad is None)
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert set(wrapper.state_dict()) == set(model.state_dict())
    assert len(list(wrapper.parameters())) == len(list(model.parameters()))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_independent_component_norms_weight_decay_and_clipping(device):
    if device == 'cuda' and os.environ.get('ETAZERO_GPU_TESTS') != '1':
        pytest.skip('requires real CUDA')
    model = make_model().to(device)
    state = {}
    learner.initialize_component_baselines(model, state)
    running = {'sums': {}, 'weights': {}}
    for index, (name, component) in enumerate(learner.components(model).items(), 1):
        running['sums'][name + '_norm_normal_batch'] = float(index)
        running['weights'][name + '_norm_normal_batch'] = 2.
        selected, context, metrics = learner.weight_decay_context(model, state, running, name)
        assert selected is component
        assert context['modelnorm_normal_baseline'] == Metrics.get_model_norms(component)['normal']
        assert metrics['sums'] == {'norm_normal_batch': float(index)}
        assert metrics['weights'] == {'norm_normal_batch': 2.}
    # Independently check norm clipping, including a component with no gradients
    # (a batch containing only side positions need not traverse dynamics).
    parameters = [next(c.parameters()) for c in learner.components(model).values()]
    parameters[0].grad = torch.full_like(parameters[0], 1e5)
    parameters[2].grad = torch.full_like(parameters[2], 1e5)
    metrics = {}
    cap = learner.clip_component_gradients(model, metrics, 4, adaptive_optimizer=False,
                                           clip_scale=.2, lr_scale=4.)
    expected_cap = 2500 * .2 * math.sqrt(4/256) / 2
    assert cap == pytest.approx(math.sqrt(3) * expected_cap)
    for parameter in (parameters[0], parameters[2]):
        assert parameter.grad.norm().item() == pytest.approx(expected_cap)
    assert parameters[1].grad is None
    assert all(value.device == model.device for value in metrics.values() if isinstance(value, torch.Tensor))
    from katago.train.trainloop_helpers import detensorify_metrics
    assert detensorify_metrics(metrics)['dynamics_gnorm_batch'] == 0


def test_runtime_rng_roundtrip(tmp_path):
    first, validation = np.random.default_rng(22), np.random.default_rng(33)
    device = torch.device('cpu')
    saved = checkpoint_runtime.capture_runtime(first, validation, None, device)
    path = tmp_path / 'runtime.ckpt'
    torch.save(saved, path)
    expected = (first.random(5), validation.random(5), random.random(), np.random.rand(5), torch.rand(5))
    checkpoint_runtime.restore_runtime(load_checkpoint(path), first, validation, None, device)
    actual = (first.random(5), validation.random(5), random.random(), np.random.rand(5), torch.rand(5))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('norm', [float('nan'), float('inf'), -float('inf'), 1e30, 1e29, 0., 2.])
def test_component_gradient_metrics_follow_source_overflow_rule(monkeypatch, device, norm):
    if device == 'cuda' and os.environ.get('ETAZERO_GPU_TESTS') != '1':
        pytest.skip('requires real CUDA')
    from katago.train import trainloop_helpers

    metrics = {'nsamp': 4, 'loss_sum': torch.tensor(3., device=device),
               'gnorm_batch': torch.tensor(norm, device=device), 'exgnorm_sum': torch.tensor(norm, device=device)}
    for name, value in [('representation', norm), ('dynamics', 0.), ('prediction', norm)]:
        metrics.update({name + '_gnorm_batch': torch.tensor(value, device=device),
                        name + '_exgnorm_sum': torch.tensor(value, device=device),
                        name + '_norm_normal_batch': torch.tensor(7., device=device)})
    # Each component must have exactly the metrics the native helper would
    # retain if it were an ordinary standalone source network.
    expected = trainloop_helpers.detensorify_metrics(metrics)
    for name in ('representation', 'dynamics', 'prediction'):
        local = {key: metrics[name + '_' + key] for key in ('gnorm_batch', 'exgnorm_sum')}
        retained = trainloop_helpers.detensorify_metrics(local)
        for key in local:
            expected.pop(name + '_' + key, None)
        expected.update({name + '_' + key: value for key, value in retained.items()})
    transfers = []
    native_cpu = torch.Tensor.cpu
    def cpu(tensor, *args, **kwargs):
        transfers.append(tensor.device.type)
        return native_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    actual = learner.detensorify_metrics(metrics)
    assert actual == expected
    assert transfers == [device]
    assert actual['dynamics_gnorm_batch'] == 0


def test_overflow_does_not_poison_later_component_metric_averages():
    from collections import defaultdict
    from katago.train.metrics_logging import accumulate_metrics

    sums, weights = defaultdict(float), defaultdict(float)
    for norm in (2., float('inf'), 4.):
        values = learner.detensorify_metrics({'prediction_gnorm_batch': torch.tensor(norm),
                                              'prediction_exgnorm_sum': torch.tensor(norm), 'nsamp': 4})
        accumulate_metrics(sums, weights, values, 4, decay=1., new_weight=1.)
    assert sums['nsamp'] == 12
    assert weights['prediction_gnorm_batch'] == 2
    assert sums['prediction_gnorm_batch'] / weights['prediction_gnorm_batch'] == 3
    assert weights['prediction_exgnorm_sum'] == 8
    assert sums['prediction_exgnorm_sum'] / weights['prediction_exgnorm_sum'] == .75


def test_recurrent_attention_penalty_on_side_only_and_active_batches(monkeypatch):
    monkeypatch.setenv('KATAGO_FLEX_ATTENTION', '0')
    config = component_config()
    model = MuZeroModel(MuZeroConfig('renju', 6, config, component_config(kind='transformer'), config), 5)
    model.initialize()
    model.dynamics.trunk.attn_logit_penalty_cap = 0.01
    model.dynamics.trunk.attn_logit_penalty_batch_frac = 1.
    wrapper = learner.MuZeroForward(model, no_compile=True)
    root, targets = batch_for(model)
    actions = torch.zeros(3, 2, dtype=torch.long)
    active = torch.zeros(3, 3, dtype=torch.bool); active[:, 0] = True
    kwargs = dict(metrics_fn=Metrics(1, model.prediction).metrics_dict_batchwise, metric_options=metric_options())
    wrapper((root, targets, actions, active), **kwargs)
    assert wrapper.attention_loss is None
    active[:, :] = True
    result = wrapper((root, targets, actions, active), **kwargs)
    assert torch.isfinite(wrapper.attention_loss) and wrapper.attention_loss.requires_grad
    assert result['alogitpen_sum'] >= 0
    (result['loss_sum'] + wrapper.attention_loss).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def run_cli(arguments, log, env):
    with log.open('w') as output:
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, env=env,
                                stdout=output, stderr=subprocess.STDOUT, timeout=600)
    assert result.returncode == 0, log.read_text()[-12000:]


@pytest.mark.parametrize('device,optimizer,compiled,game', [
    ('cpu', 'sgd', False, 'renju'), ('cpu', 'adamw', False, 'renju'),
    ('cuda', 'adamw', True, 'renju'), ('cuda', 'muon', False, 'renju'),
    ('cpu', 'sgd', False, 'xiangqi'), ('cuda', 'adamw', True, 'xiangqi'),
])
def test_source_learner_checkpoint_resume_and_bundle(raw_data, tmp_path, device, optimizer, compiled, game):
    if device == 'cuda' and os.environ.get('ETAZERO_GPU_TESTS') != '1':
        pytest.skip('requires real CUDA')
    canvas = 12 if game == 'xiangqi' else 5
    output = shuffle_source(raw_data / ('9' if game == 'xiangqi' else '3') / 'mu', tmp_path, steps=3, rows_per_file=1000, files_per_bucket=1)
    data = tmp_path / 'data'; data.mkdir()
    shutil.copytree(output, data / 'train')
    shutil.copytree(output, data / 'val')
    shutil.copyfile(output.with_suffix('.json'), data / 'train.json')
    shutil.copyfile(output.with_suffix('.json'), data / 'val.json')
    rows = sum(len(load(p)['globalInputNC']) // 4 * 4 for p in output.glob('*.npz'))
    config = tmp_path / 'model.json'
    network = asdict(make_model(game, pos_len=canvas).network_config)
    if optimizer == 'sgd':
        network['representation']['norm_kind'] = 'bnorm'
        network['dynamics']['norm_kind'] = 'brenorm'
    config.write_text(json.dumps(network))
    train = tmp_path / 'train'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    command = [ROOT / 'python/train.py', '-traindir', train, '-datadir', data,
               '-pos-len', str(canvas), '-batch-size', '4', '-samples-per-epoch', '8',
               '-max-epochs-this-instance', '1', '-lookahead-k', '2', '-swa-period-samples', '8',
               '-no-export', '-no-longterm-checkpoints', '-quit-if-no-data']
    if optimizer != 'sgd': command += ['-use-' + optimizer]
    if optimizer == 'sgd':
        command += ['-brenorm-target-rmax', '3', '-brenorm-target-dmax', '5',
                    '-brenorm-avg-momentum', '.01', '-brenorm-adjustment-scale', '100']
    if device == 'cuda': command += ['-use-fp16']
    if not compiled: command += ['-no-compile']
    configured = [*command, '-muzero-config', config, '-muzero-hidden-gradient-scale', '.3']
    run_cli([*configured, '-initialize-only'], tmp_path / 'initial.log', env)
    initial = load_checkpoint(train / 'checkpoint.ckpt')
    assert initial['train_state']['global_step_samples'] == 0
    assert initial['train_state']['export_cycle_counter'] == 0
    assert initial['swa_model']['n_averaged'] == 0
    assert not initial['optimizer']['state']
    assert initial['learner_runtime']['train_rng']
    assert 'BEGINNING NEXT EPOCH' not in (tmp_path / 'initial.log').read_text()
    before_retry = (train / 'checkpoint.ckpt').read_bytes()
    result = subprocess.run([sys.executable, *map(str, configured), '-initialize-only'],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'will not overwrite' in result.stdout + result.stderr
    assert (train / 'checkpoint.ckpt').read_bytes() == before_retry
    run_cli(configured, tmp_path / 'first.log', env)
    first = load_checkpoint(train / 'checkpoint.ckpt')
    run_cli(command, tmp_path / 'resume.log', env)
    second = load_checkpoint(train / 'checkpoint.ckpt')
    assert first['train_state']['global_step_samples'] == rows
    assert second['train_state']['global_step_samples'] == 2 * rows
    assert second['train_state']['export_cycle_counter'] == 2
    assert second['train_state']['muzero_hidden_gradient_scale'] == .3
    assert first['swa_model']['n_averaged'] == rows // 8
    assert second['swa_model']['n_averaged'] == 2 * (rows // 8)
    assert second['train_state']['muzero_modelnorm_baselines'] == first['train_state']['muzero_modelnorm_baselines']
    assert first['train_state']['muzero_modelnorm_baselines'] == initial['train_state']['muzero_modelnorm_baselines']
    for name in ('representation.', 'dynamics.', 'prediction.'):
        assert any(not torch.equal(initial['model'][k], v) for k, v in first['model'].items() if k.startswith(name))
    if optimizer == 'sgd': assert 'brenorm_rmax' in second['train_state']
    assert {g['component_name'] for g in second['optimizer']['param_groups']} == {'representation', 'dynamics', 'prediction'}
    for name in ('representation.', 'dynamics.', 'prediction.'):
        assert any(not torch.equal(first['model'][k], v) for k, v in second['model'].items() if k.startswith(name))
    if optimizer == 'adamw':
        for key, value in second['optimizer']['state'].items():
            assert value['step'] > first['optimizer']['state'][key]['step']
    assert second['learner_runtime']['train_rng'] != first['learner_runtime']['train_rng']
    assert second['learner_runtime']['val_rng'] != first['learner_runtime']['val_rng']
    if device == 'cuda':
        assert second['learner_runtime']['scaler']
        assert second['learner_runtime']['cuda_rng'] is not None
    assert 'dropping old optimizer state' not in (tmp_path / 'resume.log').read_text()
    for swa in (False, True):
        directory = tmp_path / ('swa' if swa else 'ordinary')
        run_cli([ROOT / 'python/export_muzero.py', '-checkpoint', train / 'checkpoint.ckpt',
                 '-export-dir', directory, '-model-name', 'check', '-pos-len', str(canvas),
                 *(['-use-swa'] if swa else [])], tmp_path / f'export-{swa}.log', env)
        assert {p.name for p in directory.iterdir()} == {'representation.bin', 'dynamics.bin', 'prediction.bin', 'model.json'}
        assert json.loads((directory / 'model.json').read_text())['game'] == game
    model, swa, _ = load_model(train / 'checkpoint.ckpt', True, 'cpu', pos_len=canvas)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, second['model'][name], rtol=0, atol=0)
        torch.testing.assert_close(swa.state_dict()[name], second['swa_model']['module.' + name], rtol=0, atol=0)
    if device == 'cuda':
        # Read the actual checkpoint exports in the native engine. Compare both
        # root and learned successor tensors against the selected saved weights.
        for name, selected in (('ordinary', model), ('swa', swa)):
            root, _ = batch_for(selected)
            request = dict(batch=3, fp16=True, spatial=root['binaryInputNCHW'].flatten().tolist(),
                           steps=[dict(states=[[0, 0], [0, 2]], actions=[0, 1])])
            request['global'] = root['globalInputNC'].flatten().tolist()
            request_path = tmp_path / f'{name}-request.json'
            response_path = tmp_path / f'{name}-response.json'
            request_path.write_text(json.dumps(request))
            result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-muzero-inference',
                                     str(tmp_path / name), str(request_path), str(response_path)],
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
            selected.eval()
            with torch.no_grad():
                initial = selected.initial(root['binaryInputNCHW'], root['globalInputNC'])
                recurrent = selected.recurrent(initial[0][[0, 2]], torch.tensor([0, 1]))
            for rows, (hidden, heads) in zip(json.loads(response_path.read_text()), (initial, recurrent)):
                policy, value, misc, more, _ = heads[0]
                for key, tensor in dict(latent=hidden, policy=policy[:, [0, 5, 6]], value=value,
                                        auxiliary=torch.cat((misc[:, :1], more[:, :1]), 1)).items():
                    actual = np.array([row[key] for row in rows]).reshape(tensor.shape)
                    np.testing.assert_allclose(actual, tensor.numpy(), rtol=.015, atol=.012, err_msg=name + ':' + key)
