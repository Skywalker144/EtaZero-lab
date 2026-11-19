"""Source trunk fidelity and MuZero recurrence/loss boundaries, without a fake game transition."""
from copy import deepcopy
from dataclasses import asdict
import os

import pytest
import torch

from katago.train import modelconfigs
from katago.train.model_pytorch import Model
from katago.train.metrics_pytorch import Metrics
from katago.train.muzero_model import MuZeroConfig, MuZeroModel, normalize_hidden_state, scale_gradient
from katago.train.muzero_training import unroll_loss
from katago.game import xiangqi as xq


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def component_config(game='renju', kind='regular', intermediate=False):
    if kind == 'plain_transformer':
        config = deepcopy(modelconfigs.config_of_name['b22c192h6tfrs-fson-mish'])
        config['block_kind'] = config['block_kind'][:2]
    elif kind == 'transformer':
        config = deepcopy(modelconfigs.config_of_name['b5c192h3nbttfrs-fson-mish'])
        config['block_kind'] = config['block_kind'][:1]
    else:
        config = deepcopy(modelconfigs.config_of_name['b2c16-fson-mish'])
        if kind == 'nbt':
            config['block_kind'] = [['r1', 'bottlenest2'], ['r2', 'bottlenest2gpool']]
    config.update(version=17 if game == 'go' else 19 if game == 'xiangqi' else 18, predict_q_values=True,
                  has_intermediate_head=intermediate, intermediate_head_blocks=1)
    return config


def make_model(game='renju', intermediate=False, kind='regular', pos_len=None):
    config = component_config(game, kind, intermediate)
    model = MuZeroModel(MuZeroConfig(game, 6, config, config, config),
                        (10 if game == 'xiangqi' else 5) if pos_len is None else pos_len)
    model.initialize()
    return model


def batch_for(model, device='cpu'):
    config = model.network_config.representation
    canvas = model.pos_len
    spatial = torch.randn(3, modelconfigs.get_num_bin_input_features(config), canvas, canvas, device=device)
    if model.game == 'xiangqi':
        spatial[:, 0] = 0
        spatial[:, 0, :10, :9] = 1
        spatial[:, :, 10:, :] = 0
        spatial[:, :, :, 9:] = 0
    else:
        spatial[:, 0] = 1
        spatial[1, :, -1] = 0  # Smaller board; off-board latent features must stay zero.
    global_input = torch.zeros(3, modelconfigs.get_num_global_input_features(config), device=device)
    root = dict(binaryInputNCHW=spatial, globalInputNC=global_input)
    steps = []
    for step in range(3):
        targets = torch.zeros(3, 80, device=device)
        for columns in ([0,70,1], [4,71,5], [8,72,9], [12,73,13]):
            for row in range(3):
                targets[row, columns[(row + step) % 3]] = 1
        targets[:, 25:29] = 1
        targets[:, 22] = 2
        moves = 2086 if model.game == 'xiangqi' else canvas*canvas + int(model.game == 'go')
        q = torch.zeros(3, 2, moves, device=device); q[:, 1] = 1
        steps.append(dict(globalTargetsNC=targets, policyTargetsNCMove=torch.ones(3, 2, moves, device=device),
                          valueTargetsNCHW=torch.zeros(3, 2, canvas, canvas, device=device), qValueTargetsNCMove=q))
    return root, steps


def metric_options(intermediate=False):
    return dict(is_training=True, soft_policy_weight_scale=1., disable_optimistic_policy=False,
                meta_kata_only_soft_policy=False, value_loss_scale=1., td_value_loss_scales=[1., 1., 1.],
                variance_time_loss_scale=.5, main_loss_scale=1., intermediate_loss_scale=.5 if intermediate else None)


@pytest.mark.parametrize('kind', ['regular', 'nbt', 'transformer', 'plain_transformer'])
@pytest.mark.parametrize('training', [False, True])
def test_component_trunk_equals_source_outputs_and_gradients(kind, training, monkeypatch):
    monkeypatch.setenv('KATAGO_FLEX_ATTENTION', '0')
    config = component_config(kind=kind)
    full = Model(config, 5); full.initialize(); full.train(training)
    trunk = Model(config, 5, trunk_only=True)
    trunk.initialize()
    trunk.load_state_dict({k: v for k, v in full.state_dict().items() if k in trunk.state_dict()})
    trunk.train(training)
    if kind == 'plain_transformer':
        assert trunk.supports_per_block_compile()
        trunk.compile_per_block_trunk = True
    observed = []
    hook = full.act_trunkfinal.register_forward_hook(lambda _module, _args, out: observed.append(out))
    spatial = torch.randn(2, 5, 5, 5); spatial[:, 0] = 1; spatial[1, :, 4] = 0
    global_input = torch.randn(2, 8)
    full(spatial, global_input)
    actual = trunk(spatial, global_input)
    hook.remove()
    torch.testing.assert_close(actual, observed[0], rtol=0, atol=0)
    observed[0].square().sum().backward(); actual.square().sum().backward()
    full_params = dict(full.named_parameters())
    for name, parameter in trunk.named_parameters():
        expected = full_params[name].grad
        assert (parameter.grad is None) == (expected is None), name
        if expected is not None:
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)


def test_latent_mask_small_span_and_gradient_scaling():
    mask = torch.tensor([[[[1., 1., 0.]]]])
    hidden = torch.tensor([[[[2., 4., -100.]], [[3., 2., 100.]]]], requires_grad=True)
    actual = normalize_hidden_state(hidden, mask)
    torch.testing.assert_close(actual, torch.tensor([[[[0., 1., 0.]], [[.5, 0., 0.]]]]))
    actual.sum().backward()
    assert torch.count_nonzero(hidden.grad[..., 2]) == 0
    tiny = normalize_hidden_state(torch.tensor([[[[0., 1e-6, 100.]]]]), mask)
    assert tiny[0,0,0,1].item() == pytest.approx(1/11)
    assert torch.count_nonzero(normalize_hidden_state(torch.ones(1, 3, 1, 3), mask)) == 0
    x = torch.tensor(2., requires_grad=True)
    y = scale_gradient(x.square(), .25)
    assert y.item() == 4
    y.backward(); assert x.grad.item() == 1


@pytest.mark.parametrize('canvas', [10, 12])
def test_xiangqi_joint_actions_are_two_fixed_root_planes(canvas):
    model = make_model('xiangqi', pos_len=canvas).eval()
    root, _ = batch_for(model)
    # Horse, long rook/cannon, advisor: these are geometry, not a legal-move
    # list for the random observation. Repeating the same action is permitted.
    moves = [(1, 0, 2, 2), (8, 9, 8, 0), (4, 8, 5, 9)]
    actions = torch.tensor([xq.INDEX[move] for move in moves])
    recorded = []
    hook = model.dynamics.register_forward_pre_hook(lambda _m, args: recorded.append(args[0].clone()))
    with torch.no_grad():
        hidden, _ = model.initial(root['binaryInputNCHW'], root['globalInputNC'])
        root_hidden = hidden
        expected_hidden = hidden.clone()
        for _ in range(2):
            previous = hidden
            hidden, outputs = model.recurrent(hidden, actions)
            packed = recorded[-1]
            torch.testing.assert_close(packed[:, :1], previous[:, -1:], rtol=0, atol=0)
            torch.testing.assert_close(packed[:, 1:-2], previous[:, :-1], rtol=0, atol=0)
            for row, (x, y, u, v) in enumerate(moves):
                expected = torch.zeros(2, canvas, canvas)
                expected[0, y, x] = expected[1, v, u] = 1
                torch.testing.assert_close(packed[row, -2:], expected, rtol=0, atol=0)
            assert outputs[0][0].shape == (3, 7, 2086)
            torch.testing.assert_close(hidden[:, -1:], expected_hidden[:, -1:], rtol=0, atol=0)
        # Raw root state is immutable across different children.
        torch.testing.assert_close(root_hidden, expected_hidden, rtol=0, atol=0)
    hook.remove()
    assert 'move_squares' not in model.state_dict()
    assert not any('pass' in name for name in model.state_dict())
    for invalid in (-1, 2086):
        with pytest.raises((IndexError, RuntimeError)):
            model.recurrent(hidden, torch.full((3,), invalid, dtype=torch.int64))


@pytest.mark.parametrize('game,version', [('go', 19), ('renju', 19), ('xiangqi', 18), ('xiangqi', 17)])
def test_muzero_component_game_family_cannot_be_reinterpreted(game, version):
    config = component_config(game)
    for name in ('representation', 'dynamics', 'prediction'):
        components = {key: deepcopy(config) for key in ('representation', 'dynamics', 'prediction')}
        components[name]['version'] = version
        with pytest.raises(ValueError, match='versions must match'):
            MuZeroConfig(game, 6, **components).validate()


def test_xiangqi_muzero_canvas_contains_physical_board():
    with pytest.raises(ValueError, match='9 x 10'):
        make_model('xiangqi', pos_len=9)


@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('intermediate', [False, True])
def test_unroll_heads_roles_side_rows_and_source_losses(game, intermediate):
    model = make_model(game, intermediate).train()
    root, targets = batch_for(model)
    actions = torch.tensor([[0, 0], [-1, -1], [2, -1]])
    active = torch.tensor([[True, True, True], [True, False, False], [True, True, False]])
    sizes = []
    hook = model.dynamics.register_forward_pre_hook(lambda _m, args: sizes.append(args[0].shape[0]))
    result = list(model.unroll(root['binaryInputNCHW'], root['globalInputNC'], actions, active_steps=active))
    hook.remove()
    assert sizes == [2, 1]  # Inactive rows with invalid actions never enter dynamics or BN.
    assert [step.rows.tolist() for step in result] == [[0,1,2], [0,2], [0]]
    for step in result:
        assert step.hidden.dtype == torch.float32
        torch.testing.assert_close(step.hidden[:, -1:], root['binaryInputNCHW'][step.rows, :1])
        assert step.hidden[:, :-1].min() >= 0 and step.hidden[:, :-1].max() <= 1
        outputs = model.prediction.postprocess_output(step.outputs)
        assert len(outputs) == (2 if intermediate else 1)
        for output in outputs:
            assert output[0].shape[-1] == (2086 if game == 'xiangqi' else 25 + int(game == 'go'))
            assert output[1].shape[-1] == 4
            if game != 'go': assert torch.count_nonzero(output[1].softmax(-1)[:, 3]) == 0
    groups = {}; model.add_reg_dict(groups)
    grouped = [id(parameter) for group in groups.values() for parameter in group]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(parameter) for parameter in model.parameters()}
    local = {}; model.representation.trunk.add_reg_dict(local)
    for parameter in local['output']:
        assert any(parameter is p for p in groups['normal_gamma'])
    loss, logged = unroll_loss(model, root, targets, actions, active_steps=active, **metric_options(intermediate))
    torch.testing.assert_close(loss.detach(), sum(step['loss_sum'] for step in logged))
    loss.backward()
    for component in (model.representation, model.dynamics, model.prediction):
        grads = [p.grad for p in component.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


def test_hidden_gradient_boundary_and_repeated_pass():
    base = make_model('go').eval()
    root, _ = batch_for(base)
    actions = torch.full((3, 2), 25, dtype=torch.long)
    gradients = []
    outputs = []
    for factor in (1., .5):
        model = deepcopy(base)
        result = list(model.unroll(root['binaryInputNCHW'], root['globalInputNC'], actions,
                                   hidden_gradient_scale=factor))
        outputs.append(result[-1].outputs[0][1].detach())
        result[-1].outputs[0][1].square().sum().backward()
        gradients.append(model.representation.projection.weight.grad)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[1], gradients[0] * .5)
    assert gradients[0].abs().sum() > 0


def test_unroll_loss_future_backward_weight_keeps_unscaled_logs():
    base = make_model().eval()
    root, targets = batch_for(base)
    targets[0]['globalTargetsNC'][:, 25] = 0
    actions = torch.tensor([[0,1], [1,2], [2,3]])
    actual = deepcopy(base)
    loss, logged = unroll_loss(actual, root, targets, actions, hidden_gradient_scale=1., **metric_options())
    loss.backward()
    reference = deepcopy(base)
    hidden, _ = reference.initial(root['binaryInputNCHW'], root['globalInputNC'])
    costs = []
    metrics = Metrics(1, reference.prediction)
    for step in (1, 2):
        hidden, outputs = reference.recurrent(hidden, actions[:, step - 1])
        batch = dict(root, **targets[step])
        result = metrics.metrics_dict_batchwise(reference.prediction,
            reference.prediction.postprocess_output(outputs), None, batch,
            include_model_norms=False, **metric_options())
        costs.append(result['loss_sum'])
    # K=2: the independent reference uses ordinary source losses with half weight.
    reference_loss = (costs[0] + costs[1]) / 2
    reference_loss.backward()
    assert logged[0]['loss_sum'] == 0
    torch.testing.assert_close(loss.detach(), 2 * reference_loss.detach())
    for p, q in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_source_go_configuration_versions_remain_available():
    config = deepcopy(modelconfigs.config_of_name['b2c16-fson-mish'])
    model = MuZeroModel(MuZeroConfig('go', 4, config, config, config), 5)
    model.initialize()
    assert model.prediction.config['version'] == config['version']
    with pytest.raises(ValueError, match='versions must match'):
        MuZeroModel(MuZeroConfig('hex', 4, config, config, config), 5)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires CUDA')
@pytest.mark.parametrize('game', ['renju', 'xiangqi'])
def test_muzero_cuda_amp_update_and_reload(tmp_path, game):
    torch.manual_seed(13)
    model = make_model(game, True, 'nbt').cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=.9)
    scaler = torch.amp.GradScaler('cuda')
    root, targets = batch_for(model, 'cuda')
    actions = torch.tensor([[0,0], [1,2], [3,4]], device='cuda')
    if game == 'xiangqi':
        actions = torch.tensor([[2085,2085], [xq.INDEX[1,0,2,2], xq.INDEX[2,0,4,2]],
                                [xq.INDEX[8,9,8,0], xq.INDEX[4,8,5,9]]], device='cuda')
    before = model.representation.projection.weight.detach().clone()
    updates = 0
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            hidden, prediction = model.initial(root['binaryInputNCHW'], root['globalInputNC'])
            assert hidden.dtype == torch.float32 and prediction[0][1].dtype == torch.float32
            loss, _ = unroll_loss(model, root, targets, actions, **metric_options(True))
        assert torch.isfinite(loss)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        old_scale = scaler.get_scale()
        old_weight = model.representation.projection.weight.detach().clone()
        scaler.step(optimizer); scaler.update()
        if finite:
            updates += 1
            if updates == 3:
                break
        else:
            assert scaler.get_scale() < old_scale
            torch.testing.assert_close(model.representation.projection.weight, old_weight, rtol=0, atol=0)
    assert updates == 3
    assert not torch.equal(before, model.representation.projection.weight)
    path = tmp_path / 'component-checkpoint.pt'
    torch.save(dict(config=asdict(model.network_config), model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict()), path)
    saved = torch.load(path, map_location='cuda', weights_only=False)
    restored = MuZeroModel(MuZeroConfig(**saved['config']), model.pos_len).cuda()
    restored.initialize()
    restored.load_state_dict(saved['model'])
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=1e-4, momentum=.9)
    restored_optimizer.load_state_dict(saved['optimizer'])
    model.eval(); restored.eval()
    with torch.no_grad():
        expected = model.initial(root['binaryInputNCHW'], root['globalInputNC'])
        actual = restored.initial(root['binaryInputNCHW'], root['globalInputNC'])
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for a, b in zip(actual[1][0], expected[1][0]): torch.testing.assert_close(a, b, rtol=0, atol=0)
    for parameter, original in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(restored_optimizer.state[parameter]['momentum_buffer'],
                                   optimizer.state[original]['momentum_buffer'], rtol=0, atol=0)
    restored_scaler = torch.amp.GradScaler('cuda')
    restored_scaler.load_state_dict(saved['scaler'])
    before_resume = model.representation.projection.weight.detach().clone()
    for network, opt, amp in ((model, optimizer, scaler), (restored, restored_optimizer, restored_scaler)):
        network.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss, _ = unroll_loss(network, root, targets, actions, **metric_options(True))
        amp.scale(loss).backward(); amp.step(opt); amp.update()
    assert not torch.equal(before_resume, model.representation.projection.weight)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
def test_heads_only_prediction_is_direct_hidden_to_heads(game):
    body = component_config(game, 'nbt')
    prediction = deepcopy(body)
    prediction.update(block_kind=[], trunk_num_channels=6)
    model = MuZeroModel(MuZeroConfig(game, 6, body, body, prediction), 10 if game == 'xiangqi' else 5)
    model.initialize()
    assert model.prediction.conv_spatial is None
    assert model.prediction.norm_trunkfinal is None
    assert not model.prediction.blocks
    assert all(name.startswith(('policy_head.', 'value_head.')) for name, _ in model.prediction.named_parameters())
    from katago.train.muzero_learner import initialize_component_baselines, weight_decay_context
    state = {}; initialize_component_baselines(model, state)
    assert 'normal' not in state['muzero_modelnorm_baselines']['prediction']
    _, prediction_state, _ = weight_decay_context(model, state, {}, 'prediction')
    assert prediction_state['modelnorm_normal_baseline'] is None
    root, targets = batch_for(model)
    hidden = model.representation(root['binaryInputNCHW'], root['globalInputNC']).detach().requires_grad_()
    mask = hidden[:, -1:]
    kwargs = dict(mask=mask, mask_sum_hw=mask.sum((2,3), keepdim=True), mask_sum=mask.sum(), extra_outputs=None)
    expected = (model.prediction.policy_head(hidden[:, :-1], **kwargs),
                *model.prediction.value_head(hidden[:, :-1], input_global=hidden.new_empty((3,0)), **kwargs))
    actual = model.predict(hidden)[0]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(torch.autograd.grad(actual[0].sum()+actual[1].sum(), hidden, retain_graph=True)[0],
                               torch.autograd.grad(expected[0].sum()+expected[1].sum(), hidden)[0])
    groups = {}; model.add_reg_dict(groups)
    assert {id(p) for group in groups.values() for p in group} == {id(p) for p in model.parameters()}
    loss = sum(step.outputs[0][1].square().sum() for step in model.unroll(
        root['binaryInputNCHW'], root['globalInputNC'], torch.zeros(3,5,dtype=torch.int64)))
    loss.backward()
    for component in model.training_components().values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
