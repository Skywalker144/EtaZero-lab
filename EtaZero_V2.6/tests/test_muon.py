"""Muon parameter routing, decay semantics, schedules and state restoration."""
import copy

import pytest
import torch

from config_samples import CONFIGS
from etazero.config import load_config, validate
from etazero.muon import Muon
from etazero.network import make_network
from etazero.optimization import group_settings, optimization_for, optimizer_for


def muon_config(algorithm='alphazero', architecture='nbt', plain=False):
    if algorithm == 'muzero':
        from test_muzero_pipeline import small_config
        config = small_config()
    else:
        config = load_config(CONFIGS/'smoke_test')
    config['network']['architecture'] = architecture
    if architecture == 'plain':
        config['network'].update(channels=128, blocks=10)
    elif architecture == 'transformer':
        config['network'].update(channels=192, blocks=5)
    config['optimizer']['kind'] = 'muon'
    if plain:
        options = dict(learning_rate=.002, weight_decay=.0007)
        if algorithm == 'muzero':
            config['muzero_training'] = dict(auxiliary_losses=True, katago_optimizer=False, **options)
        else:
            config['plain_optimizer'] = options
    validate(config)
    return config


@pytest.fixture(autouse=True)
def scalar_cpu_kernels(monkeypatch):
    # Small CPU checks exercise the source scalar reference; CUDA integration
    # separately uses its production compiled/batched Newton-Schulz kernels.
    monkeypatch.setenv('KATAGO_MUON_BATCHED_NS', '0')
    torch.set_num_threads(1)


@pytest.mark.parametrize('algorithm,architecture,plain', [
    ('alphazero', 'nbt', False), ('alphazero', 'plain', False),
    ('alphazero', 'transformer', False),
    ('alphazero', 'nbt', True), ('muzero', 'nbt', False),
    ('muzero', 'nbt', True), ('muzero', 'resnet', True),
])
def test_model_parameter_routing_and_fixed_plain_rates(algorithm, architecture, plain):
    config = muon_config(algorithm, architecture, plain)
    model = make_network(config)
    optimizer = optimizer_for(model, config)
    assert isinstance(optimizer, Muon)
    registered = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(registered) == len(set(registered)) == len(list(model.parameters()))
    uses_muon = {id(p): group['use_muon'] for group in optimizer.param_groups for p in group['params']}
    if algorithm == 'alphazero':
        assert not uses_muon[id(model.stem.weight)]
        head = model.policy_head
        trunk = model.blocks
    else:
        assert not uses_muon[id(model.representation.stem.weight)]
        assert not uses_muon[id(model.dynamics.stem.weight)]
        head = model.prediction.policy_head
        trunk = model.dynamics.trunk.blocks
    assert all(not uses_muon[id(p)] for p in head.parameters())
    assert any(uses_muon[id(p)] for p in trunk.parameters())
    for layer in trunk.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            assert uses_muon[id(layer.weight)]
        else:
            assert all(not uses_muon[id(p)] for p in layer.parameters(recurse=False))
    if plain:
        optimization = optimization_for(model, config, optimizer)
        optimization.begin_round()
        assert optimization.backward_scale == 1 and optimization.state_dict()['swa'] is None
        assert all(g['lr'] == .002 and g['weight_decay'] == .0007 for g in optimizer.param_groups)


@pytest.mark.parametrize('foreach', ['0', '1'])
def test_auxiliary_adam_is_decoupled_adamw(foreach, monkeypatch):
    monkeypatch.setenv('KATAGO_AUX_ADAM_FOREACH', foreach)
    parameter = torch.nn.Parameter(torch.tensor([1., -2., .3]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = Muon([dict(group_name='noreg', params=[parameter], lr=.01, weight_decay=.2)])
    adamw = torch.optim.AdamW([reference], lr=.01, weight_decay=.2,
                             betas=(.95, .995), eps=1e-6, foreach=False)
    for gradient in ([.7, -1.2, .4], [-.1, .9, -.3], [0., 0., 0.]):
        parameter.grad = torch.tensor(gradient)
        reference.grad = parameter.grad.clone()
        optimizer.step(); adamw.step()
        torch.testing.assert_close(parameter, reference, rtol=1e-6, atol=1e-7)
        for name in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(optimizer.state[parameter][name], adamw.state[reference][name])


def test_muon_schedule_matches_katago_group_formulas():
    options = muon_config()['optimizer']
    options.update(lr_scale=4, lr_warmup=False, lookahead_alpha=.5)
    base = 1.33 * .00003 * 4 / .5 * .5  # sqrt(64/256)
    assert group_settings('normal', options, 64, 0, {}, {}) == pytest.approx(
        (base * 2, .020 * .5 * 4**.70))
    assert group_settings('input', options, 64, 0, {}, {}) == pytest.approx(
        (base, .009 * .5 * 4**.70 * 2/3))
    assert group_settings('normal_gamma', options, 64, 0, {}, {})[1] == pytest.approx(.009 * .5 * 4**.70 / 4)
    assert group_settings('normal_attn', options, 64, 0, {}, {}, 'fixup') == pytest.approx(
        (base * 2, .005 * .5 / 2))
    assert group_settings('noreg', options, 64, 0, {}, {})[1] == pytest.approx(1e-6 * .5 * 4**.75)
    changed_norm = group_settings('normal', options, 64, 0, {'normal': 2}, {'normal': 1})[1]
    assert changed_norm > group_settings('normal', options, 64, 0, {}, {})[1]


def test_serialized_momentum_moments_and_next_update(tmp_path):
    config = muon_config()
    model = make_network(config)
    optimizer = optimizer_for(model, config)
    for _ in range(2):
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    path = tmp_path/'optimizer.pt'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), path)
    saved = torch.load(path, weights_only=False)
    restored = make_network(config); restored.load_state_dict(saved['model'])
    other = optimizer_for(restored, config); other.load_state_dict(saved['optimizer'])
    for parameter, replica in zip(model.parameters(), restored.parameters()):
        parameter.grad = torch.randn_like(parameter); replica.grad = parameter.grad.clone()
    optimizer.step(); other.step()
    for parameter, replica in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(parameter, replica, rtol=0, atol=0)
    malformed = copy.deepcopy(saved['optimizer'])
    key = next(k for k, state in malformed['state'].items() if 'momentum_buffer' in state)
    malformed['state'][key]['momentum_buffer'] = torch.zeros(1)
    with pytest.raises(ValueError, match='shape differs'):
        other.load_state_dict(malformed)
    malformed = copy.deepcopy(saved['optimizer']); malformed['param_groups'][0]['use_muon'] = True
    with pytest.raises(ValueError, match='parameter groups differ'):
        other.load_state_dict(malformed)


def test_muon_rejects_vector_trunk_and_unsupported_groups():
    for name, parameter in [('normal', torch.nn.Parameter(torch.ones(3))),
                            ('unknown', torch.nn.Parameter(torch.ones(2, 2)))]:
        with pytest.raises(ValueError):
            Muon([dict(group_name=name, params=[parameter])])
