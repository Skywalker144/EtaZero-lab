"""Source Model/Metrics with a joint Xiangqi readout; no selfplay claim here."""
import os

import pytest
import torch

from katago.game import xiangqi as xq
from katago.train import modelconfigs
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Model


def model(with_q=False, intermediate=False, pos_len=10):
    torch.manual_seed(131)
    config = dict(modelconfigs.config_of_name['b2c16-fson-mish'], version=19,
                  predict_q_values=with_q, has_intermediate_head=intermediate, intermediate_head_blocks=1)
    net = Model(config, pos_len)
    net.initialize()
    return net


def batch(device='cpu', with_q=False):
    spatial = torch.zeros(2, 113, 10, 10, device=device)
    spatial[:, 0, :, :9] = 1
    spatial[:, 5, 2, 2] = 1
    spatial[:, 12, 7, 6] = 1
    global_input = torch.zeros(2, 8, device=device)
    global_input[:, 0] = 1
    global_input[:, 5] = 1/8
    targets = torch.zeros(2, 80, device=device)
    for columns in ([0, 1], [4, 5], [8, 9], [12, 13]):
        targets[torch.arange(2, device=device), columns] = 1
    targets[:, 25:29] = 1
    targets[:, 33] = 1
    targets[:, 75] = 1  # First future horizon unavailable, second available.
    data = dict(binaryInputNCHW=spatial, globalInputNC=global_input, globalTargetsNC=targets,
                policyTargetsNCMove=torch.ones(2, 2, 2086, device=device),
                valueTargetsNCHW=torch.zeros(2, 2, 10, 10, device=device))
    if with_q:
        data['qValueTargetsNCMove'] = torch.zeros(2, 2, 2086, device=device)
        data['qValueTargetsNCMove'][:, 1] = 1
    return data


def loss(net, outputs, data):
    return Metrics(1, net).metrics_dict_batchwise(
        net, outputs, extra_outputs=None, batch=data, is_training=True,
        soft_policy_weight_scale=1., disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
        value_loss_scale=1., td_value_loss_scales=[1., 1., 1.], variance_time_loss_scale=.5,
        main_loss_scale=1., intermediate_loss_scale=.5 if net.has_intermediate_head else None,
        include_model_norms=False)['loss_sum']


def test_joint_readout_gathers_origin_and_displacement_without_factorizing():
    net = model().eval()
    data = batch()
    def numbered_logits(module, args, out):
        return torch.arange(6*50*100, dtype=out.dtype).reshape(1, 300, 10, 10).expand(2, -1, -1, -1)
    handle = net.policy_head.conv2p.register_forward_hook(numbered_logits)
    try:
        policy = net(data['binaryInputNCHW'], data['globalInputNC'])[0][0]
    finally:
        handle.remove()
    for head in range(6):
        # Independently fixed entries of the published displacement ordering.
        assert policy[0, head, xq.INDEX[1, 0, 2, 2]] == head*5000 + 45*100 + 1  # horse
        assert policy[0, head, xq.INDEX[1, 2, 1, 9]] == head*5000 + 15*100 + 21 # long cannon move
        assert policy[0, head, xq.INDEX[4, 0, 4, 1]] == head*5000 + 9*100 + 4   # king
    assert 'move_indices' not in net.policy_head.state_dict()
    assert not any('pass' in key for key in net.state_dict())


@pytest.mark.parametrize('with_q', [False, True])
@pytest.mark.parametrize('intermediate', [False, True])
def test_source_losses_reach_joint_heads_and_mask_impossible_nr(with_q, intermediate):
    net = model(with_q, intermediate)
    data = batch(with_q=with_q)
    raw = net(data['binaryInputNCHW'], data['globalInputNC'])
    outputs = net.postprocess_output(raw)
    total = loss(net, outputs, data)
    assert torch.isfinite(total)
    for index, output in enumerate(outputs):
        assert output[0].shape == (2, 7 if with_q else 6, 2086)
        assert torch.count_nonzero(output[1].softmax(-1)[:, 3]) == 0
        assert torch.count_nonzero(output[2].softmax(-1)[:, :, 3]) == 0
        grad = torch.autograd.grad(total, raw[index][1], retain_graph=True)[0]
        assert torch.count_nonzero(grad[:, 3]) == 0 and torch.count_nonzero(grad[:, :3]) > 0
        future_grad = torch.autograd.grad(total, raw[index][4], retain_graph=True)[0]
        assert torch.count_nonzero(future_grad[:, 0]) == 0
        assert torch.count_nonzero(future_grad[:, 1]) > 0
    total.backward()
    assert net.conv_spatial.weight.grad.isfinite().all()
    assert net.policy_head.conv2p.weight.grad.isfinite().all()
    assert torch.count_nonzero(net.policy_head.conv2p.weight.grad) > 0


def test_invalid_canvas_and_go_metadata_are_rejected():
    config = dict(modelconfigs.config_of_name['b2c16-fson-mish'], version=19)
    with pytest.raises(ValueError, match='9 x 10'):
        Model(config, 9)
    with pytest.raises(ValueError, match='Go metadata'):
        Model(dict(config, metadata_encoder={}), 10)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA AMP and Inductor')
@pytest.mark.parametrize('compiled', [False, True])
def test_cuda_fp16_joint_network_updates_and_roundtrips(tmp_path, compiled):
    net = model(with_q=True).cuda().train()
    data = batch('cuda', with_q=True)
    forward = torch.compile(net) if compiled else net
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    original = net.policy_head.conv2p.weight.detach().clone()
    updates, attempts = 0, []
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            raw = forward(data['binaryInputNCHW'], data['globalInputNC'])
        assert raw[0][0].dtype == torch.float32  # source heads stay FP32
        # Source train.py leaves autocast before postprocessing and Metrics.
        outputs = net.postprocess_output(raw)
        total = loss(net, outputs, data)
        assert torch.isfinite(total)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        finite = all(p.grad is None or p.grad.isfinite().all() for p in net.parameters())
        before = {name: p.detach().clone() for name, p in net.named_parameters()} if not finite else None
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        attempts.append(dict(scale=scale, finite=bool(finite), next_scale=scaler.get_scale()))
        if finite:
            updates += 1
        else:
            # Default source GradScaler may initially overflow. Verify that it
            # skips the update and backs off, instead of reducing the test's
            # starting scale or silently accepting nonfinite parameter updates.
            assert scaler.get_scale() < scale
            assert all(torch.equal(p, before[name]) for name, p in net.named_parameters())
        if updates == 2:
            break
    assert updates == 2, attempts
    import json
    (tmp_path / 'amp.json').write_text(json.dumps(attempts, indent=2))
    assert not torch.equal(original, net.policy_head.conv2p.weight)
    path = tmp_path / 'network.ckpt'
    torch.save(net.state_dict(), path)
    restored = model(with_q=True).cuda().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    with torch.no_grad():
        before = net(data['binaryInputNCHW'], data['globalInputNC'])[0][0]
        after = restored(data['binaryInputNCHW'], data['globalInputNC'])[0][0]
    torch.testing.assert_close(after, before, rtol=0, atol=0)
