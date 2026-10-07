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
    model = MuZeroNet(c)
    obs = torch.randn(2, 5, 6, 6)
    obs[:, 0] = 1; obs[0, 0, 5] = 0; obs[0, 0, :, 5] = 0
    globals = torch.randn(2, 6)
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
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_resnet_guards_optimizer_and_initialization_identity():
    c = load_config(CONFIGS / 'resnet')
    nbt = copy.deepcopy(c); nbt['network']['architecture'] = 'nbt'
    assert initialization_key(c) != initialization_key(nbt)
    invalid = copy.deepcopy(c); invalid['muzero_training']['katago_optimizer'] = True
    with pytest.raises(ValueError, match='katago_optimizer=false'):
        validate(invalid)
    invalid = load_config(CONFIGS / 'baseline'); invalid['network']['architecture'] = 'resnet'
    with pytest.raises(ValueError, match='Unsupported network architecture'):
        validate(invalid)
    with pytest.raises(ValueError, match='architecture must be'):
        replace(NetworkConfig(6, 16, 24, 1, 24, 1, 24, 0), architecture='unknown')
    model = MuZeroNet(NetworkConfig(6, 16, 24, 1, 24, 1, 24, 0, architecture='resnet'))
    opt = optimizer_for(model, c)
    control = optimization_for(model, c, opt)
    assert isinstance(opt, torch.optim.AdamW)
    assert control.backward_scale == 1 and control.swa_count == 0
    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in model.parameters()}
