"""MuZero component checks, independent of the gated selfplay/learner runtime."""
from config_samples import CONFIGS
import copy
from dataclasses import replace
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from etazero.config import ROOT, load_config, validate
from etazero.muzero.network import (NetworkConfig, MuZeroNet, inference_network,
                                    normalize_hidden_state, scale_gradient)
from etazero.muzero.optimization import parameter_groups, optimizer_for
from etazero.network import PolicyHead, ValueHead, losses
from etazero.optimization import Optimization


@pytest.fixture
def dimensions():
    torch.set_num_threads(1)
    return NetworkConfig(6, 16, 24, 1, 32, 2, 48, 1)


def inputs(device='cpu'):
    obs = torch.zeros(2, 5, 6, 6, device=device)
    obs[0, 0, :5, :5] = 1
    obs[1, 0] = 1
    obs[:, 1, 0, 0] = 1
    obs[:, 2, 1, 0] = 1
    globals = torch.tensor([[0, 1, -1, 1, 1, .5]] * 2, dtype=torch.float32, device=device)
    return obs, globals


def test_normalization_hand_calculated_and_constant():
    hidden = torch.tensor([[[[-2., 4., 1000.]], [[1., -2., -1000.]]],
                           [[[3., 3., 5.]], [[3., 3., 8.]]]], requires_grad=True)
    mask = torch.tensor([[[[1., 1., 0.]]]]).expand(2, -1, -1, -1)
    expected = torch.tensor([[[[0., 1., 0.]], [[.5, 0., 0.]]],
                             [[[0., 0., 0.]], [[0., 0., 0.]]]])
    actual = normalize_hidden_state(hidden, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad[:, :, :, -1].eq(0).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('span', [0., 2e-12, 5e-6, 1e-5, 2e-5])
def test_normalization_small_span_values_and_gradients(device, span):
    if device == 'cuda' and os.environ.get('ETAZERO_GPU_TESTS') != '1':
        pytest.skip('Requires host CUDA')
    hidden = torch.tensor([[[[0., span / 4., span, 1e6]]]],
                          device=device, requires_grad=True)
    mask = torch.tensor([[[[1., 1., 1., 0.]]]], device=device)
    denominator = span + 1e-5 if span < 1e-5 else span
    expected = torch.tensor([[[[0., span / (4. * denominator),
                               span / denominator, 0.]]]], device=device)
    actual = normalize_hidden_state(hidden, mask)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    actual[0, 0, 0, 1].backward()
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad[0, 0, 0, -1] == 0
    assert hidden.grad.abs().max() <= 1e5
    if span:
        # Analytic derivative of (x1 - x0) / (x2 - x0 + epsilon).
        expected_grad = torch.tensor([[[[-1. / denominator + span / (4. * denominator**2),
                                         1. / denominator,
                                         -span / (4. * denominator**2), 0.]]]], device=device)
        torch.testing.assert_close(hidden.grad, expected_grad)


def test_gradient_scaling_is_identity_and_scales_only_backward():
    x = torch.tensor([2., -3.], requires_grad=True)
    output = scale_gradient(x, .25)
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    output.square().sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([1., -1.5]), rtol=0, atol=0)


@pytest.mark.parametrize('changes', [dict(canvas=4), dict(latent_channels=0),
    dict(dynamics_channels=25), dict(prediction_blocks=-1), dict(representation_blocks=True),
    dict(predict_q_values=1)])
def test_invalid_dimensions(dimensions, changes):
    with pytest.raises(ValueError):
        replace(dimensions, **changes)


@pytest.mark.parametrize('q', [False, True])
@pytest.mark.parametrize('architecture', ['nbt', 'resnet'])
def test_independent_trunks_complete_heads_and_mask(dimensions, q, architecture):
    c = replace(dimensions, predict_q_values=q, architecture=architecture)
    model = MuZeroNet(c).eval()
    assert isinstance(model.prediction.policy_head, PolicyHead)
    assert isinstance(model.prediction.value_head, ValueHead)
    assert [len(getattr(model, part).trunk.blocks) for part in
            ('representation', 'dynamics', 'prediction')] == [1, 2, 1]
    obs, globals = inputs()
    with torch.no_grad():
        hidden = model.representation(obs, globals)
        for step in range(4):
            assert hidden.shape == (2, 17, 6, 6)
            assert hidden.dtype == torch.float32
            torch.testing.assert_close(hidden[:, -1:], obs[:, :1], rtol=0, atol=0)
            assert hidden[:, :-1].min() >= 0 and hidden[:, :-1].max() <= 1
            assert hidden[0, :-1, 5].eq(0).all() and hidden[0, :-1, :, 5].eq(0).all()
            policy, wdl, td, error = model.prediction(hidden)
            assert policy.shape == (2, 6 + q, 36)
            assert wdl.shape == (2, 3) and td.shape == (2, 3, 3) and error.shape == (2,)
            parent = hidden.clone()
            # Includes an occupied point and repeats it: recurrent sees no real legality.
            hidden = model.dynamics(hidden, torch.tensor([0, 0]))
            torch.testing.assert_close(parent[:, -1:], hidden[:, -1:], rtol=0, atol=0)
        noisy = obs.clone()
        noisy[0, 1:, 5] = 1000
        noisy[0, 1:, :, 5] = -1000
        torch.testing.assert_close(model.representation(noisy, globals), model.representation(obs, globals), rtol=0, atol=0)
        changed = globals.clone(); changed[:, -1] *= -1
        assert not torch.equal(model.representation(obs, globals), model.representation(obs, changed))


@pytest.mark.parametrize('q', [False, True])
@pytest.mark.parametrize('architecture', ['nbt', 'resnet'])
def test_scripted_initial_and_recurrent_roundtrip(tmp_path, dimensions, q, architecture):
    model = MuZeroNet(replace(dimensions, predict_q_values=q, architecture=architecture)).eval()
    inference = inference_network(model)
    scripted = torch.jit.script(inference)
    path = tmp_path / 'model.pt'; scripted.save(str(path))
    restored = torch.jit.load(str(path))
    assert restored.metadata()[2:] == ('muzero', 16)
    obs, globals = inputs()
    with torch.no_grad():
        original = model(obs, globals)
        actual = restored.initial(obs, globals)
        torch.testing.assert_close(actual[1], original[0][:, 0])
        torch.testing.assert_close(actual[2], original[1])
        torch.testing.assert_close(actual[3], original[0][:, 5])
        expected = inference.initial(obs, globals)
        for step in range(5):
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
            actions = torch.tensor([step, step + 6])
            actual = restored.recurrent(actual[0], actions)
            expected = inference.recurrent(expected[0], actions)


def test_optimizer_roles_and_no_alpha_runtime_fallback(dimensions):
    model = MuZeroNet(dimensions)
    roles = {id(p): g['group_name'] for g in parameter_groups(model) for p in g['params']}
    assert len(roles) == len(list(model.parameters()))
    assert roles[id(model.representation.stem.weight)] == 'input'
    assert roles[id(model.dynamics.stem.weight)] == 'input'
    assert roles[id(model.dynamics.trunk.norm.weight)] == 'normal_gamma'
    assert roles[id(model.prediction.trunk.norm.weight)] == 'output'
    assert roles[id(model.prediction.value_head.out.weight)] == 'output'
    model.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match='every model parameter'):
        parameter_groups(model)
    config = load_config(CONFIGS / 'smoke_test')
    config['agent']['algorithm'] = 'muzero'
    with pytest.raises(ValueError, match='explicit muzero and unroll'):
        validate(config)


def update(model, optimizer, optimization, device):
    """Component exercise using existing head losses; not a trajectory trainer."""
    optimizer.zero_grad(set_to_none=True)
    obs, globals = inputs(device)
    policy = obs[:, 0].flatten(1)
    target = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device)
    hidden = model.representation(obs, globals)
    total = torch.zeros((), device=device)
    for step in range(3):
        p, v, td, error = model.prediction(hidden)
        loss = losses(p, v, td, error, obs, policy, policy, torch.ones(2, device=device),
                      target, target[:, None].expand(-1, 3, -1), torch.ones(2, device=device), 8., False,
                      torch.zeros_like(policy), policy)[0]
        total = total + scale_gradient(loss, 1. if step == 0 else .5)
        if step < 2:
            if step:
                hidden = scale_gradient(hidden, .5)
            hidden = model.dynamics(hidden, torch.tensor([1, 7], device=device))
    total.backward()
    for part in ('representation', 'dynamics', 'prediction'):
        gradients = [p.grad for p in getattr(model, part).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    optimization.configure()
    optimizer.step(); optimization.after_step()
    return total.detach()


@pytest.mark.parametrize('kind', ['sgd', 'adamw'])
def test_component_update_state_restore_next_update(tmp_path, dimensions, kind):
    config = load_config(CONFIGS / 'smoke_test')
    config['optimizer']['kind'] = kind
    config['training']['batch_size'] = 2
    torch.manual_seed(19)
    model = MuZeroNet(replace(dimensions, predict_q_values=True)).train()
    optimizer = optimizer_for(model, config)
    optimization = Optimization(model, config, optimizer)
    update(model, optimizer, optimization, 'cpu')
    path = tmp_path / 'component.pt'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    optimization=optimization.state_dict()), path)
    state = torch.load(path, weights_only=False)
    restored = copy.deepcopy(model)
    restored.load_state_dict(state['model'])
    other_optimizer = optimizer_for(restored, config); other_optimizer.load_state_dict(state['optimizer'])
    other_optimization = Optimization(restored, config, other_optimizer, state['optimization'])
    torch.testing.assert_close(update(model, optimizer, optimization, 'cpu'),
                               update(restored, other_optimizer, other_optimization, 'cpu'), rtol=0, atol=0)
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize('device,precision', [('cpu', 'float32'), ('cuda:0', 'float32'), ('cuda:0', 'float16')])
@pytest.mark.parametrize('architecture', ['nbt', 'resnet'])
def test_native_multistep_parity_and_guards(tmp_path, dimensions, device, precision, architecture):
    if device.startswith('cuda') and os.environ.get('ETAZERO_GPU_TESTS') != '1':
        pytest.skip('Enable ETAZERO_GPU_TESTS=1 with host CUDA access')
    binary = ROOT / 'build/muzero_inference_probe'
    if not binary.exists():
        pytest.skip('Build the native MuZero inference probe first')
    model = MuZeroNet(replace(dimensions, architecture=architecture)).to(device).eval()
    # Nontrivial BN buffers ensure all three export conversions are exercised.
    with torch.no_grad():
        if architecture == 'nbt':
            for part in (model.representation, model.dynamics, model.prediction):
                part.trunk.norm.running_mean.uniform_(-.3, .3)
                part.trunk.norm.running_std.uniform_(.5, 1.5)
    inference = inference_network(model)
    path = tmp_path / 'model.pt'; torch.jit.script(inference).save(str(path))
    result = subprocess.run([str(binary), str(path), device, precision], capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    obs, globals = inputs(device)
    with torch.no_grad(), torch.autocast('cuda', enabled=precision == 'float16', dtype=torch.float16):
        prediction = inference.initial(obs, globals)
        initial = prediction
        for step in range(5):
            if step == 4:
                prediction = inference.recurrent(initial[0], torch.tensor([1, 7], device=device))
            hidden, policy, value, optimistic, stdev = prediction
            expected = (hidden.flatten(1), policy.float(), value.float().softmax(1), optimistic.float(), stdev)
            for row in range(2):
                for key, tensor in zip(('latent', 'policy', 'wdl', 'optimistic', 'stdev'), expected):
                    np.testing.assert_allclose(actual[step][row][key], tensor[row].cpu().numpy(),
                                               rtol=3e-3 if precision == 'float16' else 2e-4,
                                               atol=3e-3 if precision == 'float16' else 2e-5)
            if step < 3:
                prediction = inference.recurrent(hidden, torch.tensor([step + 1, step + 7], device=device))
    if architecture == 'nbt':
        assert actual[1] == actual[4]
    else:
        # JIT profiling can fuse dynamic norm reductions after the first call.
        # Parent reuse must preserve predictions within inference precision;
        # unlike precomputed NBT normalization, bitwise identity is not expected.
        for first, repeated in zip(actual[1], actual[4]):
            for key in first:
                np.testing.assert_allclose(first[key], repeated[key],
                                           rtol=3e-3 if precision == 'float16' else 2e-4,
                                           atol=3e-3 if precision == 'float16' else 2e-5)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Requires host CUDA')
@pytest.mark.parametrize('amp', [False, True])
def test_cuda_unroll_gradients(dimensions, amp):
    config = load_config(CONFIGS / 'smoke_test')
    config['training']['batch_size'] = 2
    model = MuZeroNet(replace(dimensions, predict_q_values=True)).cuda().train()
    optimizer = optimizer_for(model, config)
    optimization = Optimization(model, config, optimizer)
    with torch.autocast('cuda', enabled=amp, dtype=torch.float16):
        output = model(*inputs('cuda'))
        assert all(x.dtype == torch.float32 for x in output)
        loss = update(model, optimizer, optimization, 'cuda')
    assert torch.isfinite(loss)
