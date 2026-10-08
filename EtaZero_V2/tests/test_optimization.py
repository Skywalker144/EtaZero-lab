from etazero.schema import GLOBALS
from config_samples import CONFIGS
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
    return load_config(CONFIGS / 'smoke_test')


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
    batch = {'obs': obs, 'policy': policy, 'opponent_policy': obs[:,2].flatten(1),
             'opponent_policy_weight': torch.tensor([1.,0.]),
             'globals': torch.randn(2, len(GLOBALS)), 'value': torch.tensor([-1., 1.])}
    batch['globals'][:,6:]=0
    original = {k: v.clone() for k, v in batch.items()}
    variants = []
    for symmetry in range(8):
        result = augment_batch(batch, symmetry)
        torch.testing.assert_close(result['obs'][:, 3].flatten(1), result['policy'])
        torch.testing.assert_close(result['obs'][:, 2].flatten(1), result['opponent_policy'])
        assert result['opponent_policy_weight'] is batch['opponent_policy_weight']
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
    assert id(model.blocks[0].pre.norm.weight) in {id(p) for p in groups['normal_gamma']}
    assert id(model.blocks[0].pre.norm.bias) in {id(p) for p in groups['noreg']}
    assert id(model.value_head.out.bias) in {id(p) for p in groups['output_noreg']}
    assert id(model.trunk_norm.weight) in {id(p) for p in groups['output']}
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


@pytest.mark.parametrize('kind',['sgd','adamw'])
def test_transformer_selects_source_fixup_decay(config,kind):
    config['network'].update(architecture='transformer',channels=192,blocks=5)
    config['optimizer'].update(kind=kind,normal_attn_wd_factor=0.7)
    model=make_network(config);optimizer=optimizer_for(model,config)
    optimization=Optimization(model,config,optimizer)
    scale=math.sqrt(8/256) if kind=='adamw' else 8/256
    regular=(0.005 if kind=='adamw' else 1e-6)*scale
    expected={name:regular for name in ('input','normal','normal_gamma','output')}
    expected.update(normal_attn=regular*0.5*0.7,noreg=1e-8*scale,output_noreg=1e-8*scale)
    for samples in (0,250000,2000000):
        optimization.consumed_samples=samples
        optimization.norms={k:4*v for k,v in optimization.baselines.items()}
        optimization.configure()
        for g in optimizer.param_groups:
            assert g['weight_decay']==pytest.approx(expected[g['group_name']])


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
        model.trunk_norm.running_mean.fill_(7)
    optimization.after_step()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 3))
    torch.testing.assert_close(optimization.swa.module.stem.weight, torch.full_like(parameter, 3))
    assert optimization.swa.n_averaged == 1
    with torch.no_grad():
        parameter.fill_(10)
    optimization.after_step()
    with torch.no_grad():
        parameter.fill_(15)
        model.trunk_norm.running_mean.fill_(11)
    optimization.after_step()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 9))
    torch.testing.assert_close(optimization.swa.module.stem.weight, torch.full_like(parameter, 4.5))
    torch.testing.assert_close(optimization.swa.module.trunk_norm.running_mean, torch.full_like(model.trunk_norm.running_mean, 11))
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


def small_optimization(config):
    """A scalar weight gives hand-calculable norms and slow/fast updates."""
    model = torch.nn.Linear(1, 1, bias=False)
    model.norm_kind = 'fixscaleonenorm'
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = torch.optim.SGD([{'params': [model.weight], 'group_name': 'normal'}], lr=1)
    return model, optimizer, Optimization(model, config, optimizer)


@pytest.mark.parametrize('kind,base', [('sgd', 2500), ('adamw', 11000)])
def test_gradient_cap_and_explicit_mean_override(config, kind, base):
    config['optimizer'].update(kind=kind, lr_scale=4)
    config['training']['batch_size'] = 128
    _, _, optimization = small_optimization(config)
    assert optimization.gradient_cap(0) == pytest.approx(base / math.sqrt(8))
    assert optimization.gradient_cap(3) == 384


@pytest.mark.parametrize('threshold', [250000*i for i in range(1, 9)])
def test_warmup_refresh_uses_post_consumption_and_next_batch(config, threshold):
    config['optimizer']['lookahead_alpha'] = 1
    config['training']['batch_size'] = 128
    _, optimizer, optimization = small_optimization(config)
    optimization.consumed_samples = threshold - 3*128
    optimization.begin_round()
    old_lr = optimizer.param_groups[0]['lr']
    for batch in range(1, 6):
        optimization.before_step()
        assert optimizer.param_groups[0]['lr'] == old_lr
        optimization.after_step(successful=batch != 3)
        if batch < 5:
            assert optimizer.param_groups[0]['lr'] == old_lr
    assert optimizer.param_groups[0]['lr'] > old_lr
    assert optimization.optimizer_steps == 4
    assert optimization.consumed_samples == threshold + 256


@pytest.mark.parametrize('start,refresh', [(200000000-5*8, 5), (200000000-4*8, 50), (200000000, 50)])
def test_200m_refresh_boundary_and_norm_before_update(config, start, refresh):
    config['optimizer'].update(lookahead_alpha=1, norm_interval=100)
    model, optimizer, optimization = small_optimization(config)
    optimization.consumed_samples = start
    optimization.begin_round()
    old_wd = optimizer.param_groups[0]['weight_decay']
    optimization.norms = {'normal': 2}
    for batch in range(1, refresh+1):
        optimization.before_step()
        optimization.after_step()
        if batch < refresh:
            assert optimizer.param_groups[0]['weight_decay'] == old_wd
    assert optimizer.param_groups[0]['weight_decay'] > old_wd
    optimization.begin_round()
    for batch in range(1, 101):
        optimization.before_step()
        if batch == 100:
            with torch.no_grad():
                model.weight.fill_(9)
        optimization.after_step()
    assert optimization.norms['normal'] == 1  # pre-step snapshot, not 9


@pytest.mark.parametrize('lookahead_print', [False, True])
def test_all_batch_norm_running_average_print_decay(config, lookahead_print):
    config['optimizer'].update(norm_interval=2, norm_only_at_print=False,
                               lookahead_print=lookahead_print, lookahead_k=2)
    model, _, optimization = small_optimization(config)
    for value in (2., 4., 8.):
        with torch.no_grad():
            model.weight.fill_(value)
        optimization.before_step()
        optimization.after_step()
    # *_batch norms do not have per-batch .995 decay in the source.
    # A print shrinks BOTH historical sum and count by .001.
    expected = (2*.001+8)/(1*.001+1) if lookahead_print else ((2+4)*.001+8)/(2*.001+1)
    assert optimization.norms['normal'] == pytest.approx(expected)
    assert optimization.norm_weights['normal'] == pytest.approx(1.001 if lookahead_print else 1.002)


def test_skipped_update_lookahead_swa_and_subepoch_reset(config):
    config['optimizer'].update(lookahead_k=2, swa_period_samples=16)
    model, _, optimization = small_optimization(config)
    optimization.begin_round()
    with torch.no_grad():
        model.weight.fill_(5)
    optimization.before_step()
    optimization.after_step()
    optimization.before_step()
    optimization.after_step(successful=False)
    assert model.weight.item() == 3  # skipped optimizer still triggers slow synchronization
    assert optimization.optimizer_steps == 1 and optimization.consumed_samples == 16
    assert optimization.swa.n_averaged == 1 and optimization.swa.module.weight.item() == 3
    with torch.no_grad():
        model.weight.fill_(7)
    optimization.before_step(); optimization.after_step()
    optimization.begin_subepoch()
    assert optimization.counter == 0 and model.weight.item() == 7
    optimization.before_step(); optimization.after_step()
    assert optimization.counter == 1  # subepoch reset changes next sync, keeping fast weights
    optimization.finish_round()
    assert model.weight.item() == 3 and optimization.swa_samples == 16


def test_optimization_state_restores_pending_refresh_and_norm_history(config):
    config['optimizer'].update(norm_only_at_print=False, norm_interval=2)
    model, optimizer, optimization = small_optimization(config)
    optimization.begin_round()
    for _ in range(4):
        optimization.before_step(); optimization.after_step()
    saved = copy.deepcopy(optimization.state_dict())
    restored = Optimization(model, config, optimizer, saved)
    assert restored.round_batches == 4 and restored.optimizer_steps == 4
    for obj in (optimization, restored):
        obj.before_step(); obj.after_step(successful=False)
    a, b = optimization.state_dict(), restored.state_dict()
    for name in ('norms', 'norm_sums', 'norm_weights', 'consumed_samples', 'optimizer_steps',
                 'round_batches', 'lookahead_counter', 'swa_samples'):
        assert a[name] == b[name]
    for name in a['swa']:
        torch.testing.assert_close(a['swa'][name], b['swa'][name], rtol=0, atol=0)
