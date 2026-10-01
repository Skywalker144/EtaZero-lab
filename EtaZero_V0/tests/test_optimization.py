import copy
import math

import pytest
import torch

from etazero.config import ROOT, load_config, validate
from etazero.network import make_network
from etazero.optimization import Optimization, group_settings, inference_weights, optimizer_for, parameter_groups, warmup_factor
from etazero.symmetry import apply_symmetry, augment_batch


@pytest.fixture
def config():
    return load_config(ROOT / 'configs/smoke_test')


@pytest.mark.parametrize('symmetry,expected', [
    (0, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    (1, [[3, 6, 9], [2, 5, 8], [1, 4, 7]]),
    (2, [[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
    (3, [[7, 4, 1], [8, 5, 2], [9, 6, 3]]),
    (4, [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
    (5, [[3, 2, 1], [6, 5, 4], [9, 8, 7]]),
    (6, [[9, 6, 3], [8, 5, 2], [7, 4, 1]]),
    (7, [[7, 8, 9], [4, 5, 6], [1, 2, 3]]),
])
def test_d4_coordinates(symmetry, expected):
    grid = torch.arange(1, 10).reshape(3, 3)
    torch.testing.assert_close(apply_symmetry(grid, symmetry), torch.tensor(expected))


def test_d4_mixed_size_masks_features_and_targets():
    obs = torch.zeros(2, 5, 6, 6)
    obs[0, 0, :5, :5] = 1
    obs[1, 0] = 1
    for channel in range(1, 5):
        obs[:, channel, channel, 4] = 1
    policy = obs[:, 3].flatten(1)
    batch = {'obs': obs, 'policy': policy, 'globals': torch.randn(2, 4), 'value': torch.tensor([-1., 1.])}
    original = {k: v.clone() for k, v in batch.items()}
    variants = []
    for symmetry in range(8):
        result = augment_batch(batch, symmetry)
        torch.testing.assert_close(result['obs'][:, 3].flatten(1), result['policy'])
        assert result['obs'][0, 0].sum() == 25 and result['obs'][1, 0].sum() == 36
        assert not (result['policy'].bool() & ~result['obs'][:, 0].flatten(1).bool()).any()
        assert result['globals'] is batch['globals'] and result['value'] is batch['value']
        assert result['obs'].is_contiguous() and result['policy'].is_contiguous()
        variants.append(result['obs'])
    assert len({v.numpy().tobytes() for v in variants}) == 8
    for k in original:
        torch.testing.assert_close(batch[k], original[k])


def test_parameter_group_coverage_and_roles(config):
    model = make_network(config)
    groups = {g['group_name']: g['params'] for g in parameter_groups(model)}
    ids = [id(p) for params in groups.values() for p in params]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert {id(p) for p in groups['input']} == {id(model.stem.weight), id(model.linear_global.weight)}
    assert id(model.blocks[0].bn1.weight) in {id(p) for p in groups['normal_gamma']}
    assert id(model.blocks[0].bn1.bias) in {id(p) for p in groups['noreg']}
    assert id(model.value_out.bias) in {id(p) for p in groups['output_noreg']}
    assert id(model.policy_bn.weight) in {id(p) for p in groups['output']}
    model.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match='every model parameter'):
        parameter_groups(model)


def test_native_lr_and_decay_formulas(config):
    options = config['optimizer']
    lr, wd = group_settings('normal', options, 256, 0, {}, {'normal': 2})
    assert lr == pytest.approx(3e-6)
    assert wd == pytest.approx(0.00125 * 0.05 ** 0.75)
    head_lr, head_wd = group_settings('output', options, 256, 0, {}, {'normal': 2})
    assert head_lr == pytest.approx(1.5e-6) and head_wd == pytest.approx(1e-6)
    gamma_wd = group_settings('normal_gamma', options, 256, 0, {}, {'normal': 2})[1]
    assert gamma_wd == pytest.approx(wd / 8)
    adaptive = group_settings('normal', options, 256, 2000000, {'normal': 4}, {'normal': 2})[1]
    assert adaptive == pytest.approx(0.00125 * 2 ** (2 * math.tanh(math.log(2) * 3)))
    adamw = {**options, 'kind': 'adamw'}
    lr, wd = group_settings('input', adamw, 64, 2000000, {}, {'input': 2})
    assert lr == pytest.approx(1.33 * 3e-5 * 0.5 / 0.5)
    assert wd == pytest.approx(0.009 * 0.5 * 2 / 3)
    assert group_settings('normal_gamma', adamw, 256, 2000000, {}, {'normal': 2})[1] == pytest.approx(0.009 / 4)


def test_warmup_boundaries():
    for index, denominator in enumerate((20, 14, 10, 7, 5, 3, 2, 1.4, 1)):
        assert warmup_factor(index * 250000) == pytest.approx(1 / denominator)
        if index:
            assert warmup_factor(index * 250000 - 1) < warmup_factor(index * 250000)
    assert warmup_factor(0, False) == 1


def test_lookahead_swa_and_round_boundary(config):
    config['optimizer'].update(lookahead_k=2, swa_period_samples=8, swa_scale=4)
    model = make_network(config)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    optimization = Optimization(model, config, optimizer_for(model, config))
    parameter = model.stem.weight
    with torch.no_grad():
        parameter.fill_(2)
    optimization.after_step()
    assert optimization.swa.n_averaged == 0
    with torch.no_grad():
        parameter.fill_(6)
        model.stem_bn.running_mean.fill_(7)
    optimization.after_step()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 3))
    torch.testing.assert_close(optimization.swa.module.stem.weight, torch.full_like(parameter, 3))
    assert optimization.swa.n_averaged == 1
    with torch.no_grad():
        parameter.fill_(10)
    optimization.after_step()
    with torch.no_grad():
        parameter.fill_(15)
        model.stem_bn.running_mean.fill_(11)
    optimization.after_step()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 9))
    torch.testing.assert_close(optimization.swa.module.stem.weight, torch.full_like(parameter, 4.5))
    torch.testing.assert_close(optimization.swa.module.stem_bn.running_mean, torch.full_like(model.stem_bn.running_mean, 11))
    with torch.no_grad():
        parameter.fill_(99)
    optimization.after_step()
    optimization.finish_round()
    assert optimization.counter == 0
    torch.testing.assert_close(parameter, torch.full_like(parameter, 9))
    saved = {'model': model.state_dict(), 'optimization': optimization.state_dict()}
    torch.testing.assert_close(inference_weights(saved)['stem.weight'], torch.full_like(parameter, 4.5))


def test_optimizer_validation(config):
    for key, value in [('kind', 'muon'), ('lookahead_alpha', 0), ('lookahead_alpha', 1.1),
                       ('swa_scale', 0.5), ('norm_interval', 0), ('lr_scale', float('nan'))]:
        changed = copy.deepcopy(config)
        changed['optimizer'][key] = value
        with pytest.raises(ValueError):
            validate(changed)
