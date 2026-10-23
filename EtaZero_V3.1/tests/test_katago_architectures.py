"""Preset selection, source architecture coverage and game-pipeline integration."""
import pytest
import torch
from etazero.config import load_config, validate
from etazero.katago_configs import BASE_NAMES, resolve
from etazero.katago_network import KataGoNet, inference_network, ExportComplexDepthwiseConv2d
from etazero.network import TrainingForward, make_network, losses
from etazero.optimization import optimizer_for
from config_samples import CONFIGS


@pytest.mark.parametrize('name', BASE_NAMES)
def test_every_handwritten_architecture_constructs_with_complete_optimizer_roles(name):
    # Full-size models, including the largest transformer, without allocating
    # training storage. Verify actual module construction and all role assignments.
    with torch.device('meta'):
        model = KataGoNet(name, 5)
    config = resolve(name)
    assert model.network.block_kind == config['block_kind']
    assert model.network.c_trunk == config['trunk_num_channels']
    assert model.network.c_mid == config['mid_num_channels']
    assert model.network.c_p1 == config['p1_num_channels']
    assert model.network.c_g1 == config['g1_num_channels']
    assert model.network.c_v1 == config['v1_num_channels']
    assert model.network.c_v2 == config['v2_size']
    groups = model.parameter_groups()
    assert sum(len(g['params']) for g in groups) == len(list(model.parameters()))


@pytest.mark.parametrize('suffix', ['', '-fs', '-bn', '-brn', '-fbrn', '-bng', '-brng', '-fbrng',
                                    '-fson-mish-rvgl-tcg-trbnr-bnh-meta', '-silu-rsnh'])
def test_config_suffix_selection_is_independent(suffix):
    config = resolve('b5c192nbt' + suffix)
    config['block_kind'][0][1] = 'corrupted'
    assert resolve('b5c192nbt' + suffix)['block_kind'][0][1] != 'corrupted'


@pytest.mark.parametrize('name', ['missing', 'b5c192nbt-bnh', 'b5c192nbt-mish-fson',
                                 'b5c192nbt-mish-mish', 'b5c192nbt--mish', 'b5c192nbt-'])
def test_invalid_names_fail(name):
    with pytest.raises(ValueError):
        resolve(name)


def test_named_config_dimensions_and_q_validation():
    config = load_config(CONFIGS / 'smoke_test')
    config['network'].update(architecture='b7c96h6kv3qk32v16tflrs-fson-silu', channels=96, blocks=14,
                             predict_q_values=True)
    validate(config)
    assert make_network(config).model_version == 17
    config['network']['blocks'] = 7
    with pytest.raises(ValueError, match='fixes network.blocks=14'):
        validate(config)
    config['network'].update(architecture='b2c16', channels=16, blocks=2)
    with pytest.raises(ValueError, match='v17'):
        validate(config)


def test_named_dimensions_can_be_omitted(tmp_path):
    for name in ('run', 'env', 'train', 'selfplay'):
        (tmp_path / (name + '.cfg')).write_text('@include ' + str(CONFIGS / 'smoke_test' / (name + '.cfg')) + '\n')
    (tmp_path / 'net.cfg').write_text('[network]\narchitecture=b2c16-fson-mish\ncanvas=6\npredict_q_values=false\n')
    config = load_config(tmp_path, run_dir=tmp_path / 'output')
    assert config['network']['channels'] == 16 and config['network']['blocks'] == 2


@pytest.mark.parametrize('name', ['b1c6nbt-fson-mish', 'b2c16-fson-mish-rvgl-bnh',
                                 'b2c16-brng-gelu-tcg-trbnr-ih-meta',
                                 'b7c96h6kv3qk32v16tflrs-fson-silu'])
def test_real_training_heads_and_variable_batch_export(name):
    torch.set_num_threads(2)
    model = KataGoNet(name, 6)
    obs = torch.randn(2, 5, 6, 6); obs[:, 0] = 0; obs[:, 0, :5, :5] = 1
    globals = torch.randn(2, 8)
    heads = model.training_outputs(obs, globals)
    assert len(heads) == len(model.head_loss_weights)
    loss = sum(weight * sum(t.square().mean() for t in head)
               for weight, head in zip(model.head_loss_weights, heads))
    loss.backward()
    if len(heads) == 2:
        assert model.network.intermediate_policy_head.conv2p.weight.grad.abs().sum() > 0
    model.eval()
    scripted = torch.jit.script(inference_network(model))
    assert scripted.metadata() == model.metadata()
    for count in (1, 2, 5):
        inputs = obs[:1].expand(count, -1, -1, -1).contiguous(), globals[:1].expand(count, -1).contiguous()
        outputs = model.forward_all(*inputs)
        actual = scripted(*inputs)
        expected = outputs[0][:, 0], outputs[1], outputs[0][:, 5], 0.5 * torch.nn.functional.softplus(outputs[3] * 0.5)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('norm', ['', '-fs', '-bn', '-brn', '-fbrn', '-bng', '-brng', '-fbrng', '-fson'])
@pytest.mark.parametrize('kind', ['sgd', 'adamw', 'muon'])
def test_all_norm_optimizer_combinations(norm, kind):
    model = KataGoNet('b2c16' + norm + '-mish', 6)
    config = load_config(CONFIGS / 'smoke_test')
    config['optimizer']['kind'] = kind
    optimizer = optimizer_for(model, config)
    assert sum(len(g['params']) for g in optimizer.param_groups) == len(list(model.parameters()))


def test_intermediate_loss_source_default_weighting_and_repvgg_gradient_scale():
    model = KataGoNet('b2c16-fson-mish-rvglr-bnh', 6)
    assert model.head_loss_weights == (0.2, 0.8)
    obs = torch.randn(2, 5, 6, 6); obs[:, 0] = 1
    globals = torch.randn(2, 8)
    policy = torch.full((2, 36), 1/36)
    value = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
    td = value[:, None].expand(-1, 3, -1)
    args = (policy, policy, torch.ones(2), value, td, torch.ones(2))
    model.eval()
    manual = [losses(*head[:4], obs, *args, 0.6, False)
              for head in model.training_outputs(obs, globals)]
    actual = TrainingForward(model, 0.6, False)(obs, globals, *args)
    for i, output in enumerate(actual):
        torch.testing.assert_close(output, 0.2 * manual[0][i] + 0.8 * manual[1][i])
    parameter = model.network.blocks[0].normactconv1.conv.weight
    parameter.grad = torch.ones_like(parameter)
    model.scale_repvgg_gradients()
    assert parameter.grad[:, :, 1, 1].eq(2).all()
    assert parameter.grad[:, :, 0, 0].eq(1).all()


def test_batch_renorm_schedule_and_metadata_default():
    model = KataGoNet('b2c16-brng-meta', 6)
    from etazero.brenorm import BoundsSchedule
    schedule = BoundsSchedule(model, dict(avg_momentum=.01, target_rmax=3., target_dmax=2., adjustment_scale=10.))
    schedule.update(10)
    from etazero._vendor.katago.model_pytorch import NormMask
    for norm in (module for module in model.modules() if isinstance(module, NormMask)):
        assert norm.rmax == pytest.approx(3 - 2 / torch.e)
        assert norm.dmax == pytest.approx(2 - 2 / torch.e)
        assert norm.renorm_avg_momentum == .01
    assert isinstance(model.network.metadata_encoder.act1, torch.nn.ReLU)
    obs = torch.zeros(1, 5, 6, 6); obs[:, 0] = 1
    assert all(torch.isfinite(value).all() for value in model.forward_all(obs, torch.zeros(1, 8)))


def test_batch_renorm_config_is_validated_and_part_of_resume_identity():
    from etazero.config import resume_config
    config = load_config(CONFIGS / 'smoke_test')
    config['network'].update(architecture='b2c16-brng', channels=16, blocks=2)
    config['brenorm'] = dict(avg_momentum=.01, target_rmax=2., target_dmax=1., adjustment_scale=10.)
    validate(config)
    assert resume_config(config)['brenorm'] == config['brenorm']
    config['brenorm']['target_rmax'] = .5
    with pytest.raises(ValueError, match='batch-renorm schedule'):
        validate(config)


def test_repvgg_export_folds_branches_without_changing_training_weights():
    model = KataGoNet('b2c16-fson-mish-rvgl', 6).eval()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    assert any('conv1x1.weight' in name for name in original)
    exported = inference_network(model)
    assert not any('conv1x1.weight' in name for name, _ in exported.named_parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_source_grouped_rmsnorm_dimension_constraint_is_not_changed():
    with pytest.raises(ValueError, match='divisible'):
        KataGoNet('b1c6nbt-rgnh', 6)


@pytest.mark.parametrize('dilation', [1, 3])
def test_complex_depthwise_export_preserves_source_convolution_and_rng(dilation):
    from etazero._vendor.katago.model_pytorch import ComplexDepthwiseConv2d
    torch.set_num_threads(2)
    source = ComplexDepthwiseConv2d(3, dilation=dilation).double()
    source.initialize('relu')
    original = {name: value.clone() for name, value in source.state_dict().items()}
    rng = torch.get_rng_state().clone()
    exported = ExportComplexDepthwiseConv2d(source)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    x = torch.randn(2, 6, 7, 9, dtype=torch.float64)
    # Independent complex convolution verifies the real/imaginary mapping.
    expected = torch.nn.functional.conv2d(
        torch.complex(x[:, :3], x[:, 3:]),
        torch.complex(source.real_kernel, source.imag_kernel),
        padding=dilation, dilation=dilation, groups=3)
    actual = exported(x)
    torch.testing.assert_close(actual[:, :3], expected.real, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual[:, 3:], expected.imag, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, source(x), rtol=0, atol=0)
