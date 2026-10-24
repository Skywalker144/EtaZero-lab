from etazero.schema import GLOBALS
"""Independent normalization properties and MuZero ResNet configuration guards."""
from config_samples import CONFIGS
import copy
from dataclasses import replace

import pytest
import torch

from etazero.config import ROOT, load_config, validate
from etazero.experiment import initialization_key
from etazero.muzero.network import MuZeroNet, NetworkConfig
from etazero.muzero.resnet import MaskedNorm
from etazero.optimization import optimizer_for, optimization_for


def test_masked_norm_hand_calculation_and_padding_gradient():
    norm = MaskedNorm(2)
    x = torch.tensor([[[[1., 3., 1000.]], [[5., 7., -1000.]]]], requires_grad=True)
    mask = torch.tensor([[[[1., 1., 0.]]]])
    y = norm(x, mask)
    expected = torch.tensor([[[[-3., -1., 0.]], [[1., 3., 0.]]]]) / (5. + 1e-5)**.5
    torch.testing.assert_close(y, expected)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[..., -1].eq(0).all()
    assert not list(norm.buffers())


def test_resnet_is_independent_of_other_batch_rows_and_train_eval_mode():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    c = NetworkConfig(6, 16, 24, 2, 32, 2, 48, 0, architecture='resnet')
    model = MuZeroNet(c).eval()
    obs = torch.randn(2, 5, 6, 6)
    obs[:, 0] = 1; obs[0, 0, 5] = 0; obs[0, 0, :, 5] = 0
    globals = torch.randn(2, len(GLOBALS))
    hidden = model.representation(obs, globals)
    single = model.representation(obs[:1], globals[:1])
    torch.testing.assert_close(hidden[:1], single, rtol=2e-5, atol=2e-6)
    actions = torch.tensor([1, 7])
    recurrent = model.dynamics(hidden, actions)
    torch.testing.assert_close(recurrent[:1], model.dynamics(single, actions[:1]), rtol=2e-5, atol=2e-6)
    before = model.prediction(recurrent)
    model.eval()
    after = model.prediction(model.dynamics(model.representation(obs, globals), actions))
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('katago_optimizer', [False, True])
def test_resnet_optimizer_and_initialization_identity(katago_optimizer):
    c = load_config(CONFIGS / 'resnet')
    nbt = copy.deepcopy(c); nbt['network']['architecture'] = 'nbt'
    assert initialization_key(c) != initialization_key(nbt)
    c['muzero_training']['katago_optimizer'] = katago_optimizer
    c['optimizer']['kind'] = 'sgd' if katago_optimizer else 'adamw'
    validate(c)
    invalid = load_config(CONFIGS / 'baseline'); invalid['network']['architecture'] = 'resnet'
    with pytest.raises(ValueError, match='Unsupported network architecture'):
        validate(invalid)
    with pytest.raises(ValueError, match='architecture must be'):
        replace(NetworkConfig(6, 16, 24, 1, 24, 1, 24, 0), architecture='unknown')
    model = MuZeroNet(NetworkConfig(6, 16, 24, 1, 24, 1, 24, 0, architecture='resnet'))
    opt = optimizer_for(model, c)
    control = optimization_for(model, c, opt)
    assert isinstance(opt, torch.optim.SGD if katago_optimizer else torch.optim.AdamW)
    assert control.backward_scale == (c['training']['batch_size'] if katago_optimizer else 1)
    assert control.swa_count == 0
    if katago_optimizer:
        assert all(g['momentum'] == .9 for g in opt.param_groups)
        roles = {id(p): g['group_name'] for g in opt.param_groups for p in g['params']}
        assert roles[id(model.representation.body.blocks[0].normactconv1.norm.gamma)] == 'normal_gamma'
        assert roles[id(model.dynamics.body.blocks[0].normactconv1.norm.beta)] == 'noreg'
        assert roles[id(model.prediction.policy_head.conv2p.weight)] == 'output'
    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in model.parameters()}
