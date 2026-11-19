"""Joint logits through source serialization, descriptor and production backends."""
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from katago.game import xiangqi as xq
from test_xiangqi_network import model
from test_muzero_export import native_desc


def export(tmp_path, with_q):
    net = model(with_q=with_q, pos_len=12 if with_q else 10).eval()
    path = tmp_path / 'model.bin'
    with path.open('wb') as out:
        write_model_binary(net, out, 'xiangqi-joint')
    return net, path


@pytest.mark.parametrize('with_q', [False, True])
def test_source_descriptor_and_unconnected_game_boundary(tmp_path, with_q):
    net, path = export(tmp_path, with_q)
    desc = native_desc(path)
    assert desc['role'] == 0 and desc['spatial_channels'] == 113 and desc['global_channels'] == 8
    assert desc['policy_channels'] == (3 if with_q else 2)
    assert desc['value_channels'] == 4 and desc['aux_channels'] == 2
    assert desc['parameters'] > 0 and desc['scale8']
    heads = [0, 5, 6] if with_q else [0, 5]
    weights = net.policy_head.conv2p.weight.detach().reshape(net.policy_head.num_policy_outputs, 50, -1)
    assert desc['spatial_policy_channels'] == len(heads) * 50
    # scale8 changes intermediate activations and adds a final output multiplier;
    # the last policy convolution's weights retain their original units.
    np.testing.assert_array_equal(desc['policy_weights'], weights[heads].flatten().numpy())
    # The raw backend supports model 19; game-level entry must not misinterpret
    # a joint action as a Go point. The standalone evaluator is tested separately.
    config = tmp_path / 'gtp.cfg'
    config.write_text('rules = tromp-taylor\nnumSearchThreads = 1\nmaxVisits = 4\n'
                      'logAllGTPCommunication = false\nlogSearchInfo = false\nlogToStderr = false\n'
                      'ponderingEnabled = false\nrequireMaxBoardSize = false\nmaxBoardSizeForNNBuffer = 12\n')
    result = subprocess.run([str(ROOT / 'build/katago'), 'gtp', '-config', str(config), '-model', str(path)],
                            input='quit\n', capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'Xiangqi models require game=xiangqi and a supported' in result.stdout + result.stderr


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA build and GPU')
@pytest.mark.parametrize('fp16,nhwc', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('with_q', [False, True])
@pytest.mark.parametrize('inputs_nhwc', [False, True])
def test_joint_export_matches_native_all_actions_and_batch_reuse(tmp_path, fp16, nhwc, with_q, inputs_nhwc):
    check_joint_export(tmp_path, ROOT / 'build/katago', fp16, nhwc, with_q, inputs_nhwc, strict_cuda=True)


def check_joint_export(tmp_path, binary, fp16, nhwc, with_q, inputs_nhwc, *, strict_cuda=False, backend_config=None):
    net, path = export(tmp_path, with_q)
    torch.manual_seed(57)
    side = net.pos_len
    spatial = (torch.rand(3, 113, side, side) < 0.04).float()
    spatial[:, :, :, 9:] = 0
    spatial[:, :, 10:, :] = 0
    spatial[:, 0, :10, :9] = 1
    globals_ = torch.rand(3, 8)
    # Input memory order is independent of the backend's internal tensor layout.
    symmetries, optimisms = [0, 2, 2], [0.0, 1.0, 0.37]
    rows, expected = [], []
    for i, (symmetry, optimism) in enumerate(zip(symmetries, optimisms)):
        original = spatial[i:i+1]
        reflected = original.clone()
        if symmetry == 2:
            reflected[:, :, :, :9] = torch.flip(original[:, :, :, :9], [-1])
        with torch.no_grad():
            policy, value, misc, moremisc, _ = net(reflected, globals_[i:i+1])[0]
        combined = policy[0, 0] + (policy[0, 5] - policy[0, 0]) * optimism
        if symmetry == 2:
            combined = combined[torch.tensor(xq.action_permutation(mirror=True))]
        expected.append(dict(policy=combined.numpy(), value=value[0].numpy(),
                             auxiliary=torch.stack([misc[0, 0], moremisc[0, 0]]).numpy()))
        encoded = original.permute(0, 2, 3, 1) if inputs_nhwc else original
        rows.append(dict(spatial=encoded.flatten().tolist(), global_=globals_[i].tolist(),
                         symmetry=symmetry, optimism=optimism))
        rows[-1]['global'] = rows[-1].pop('global_')
    request = dict(width=side, height=side, rows=rows, fp16=fp16, inputs_nhwc=inputs_nhwc,
                   backend_config={'cudaUseNHWC': str(nhwc).lower(), **(backend_config or {})}, batches=[3, 1, 2, 3])
    request_path, response_path = tmp_path / 'request.json', tmp_path / 'response.json'
    request_path.write_text(json.dumps(request))
    result = subprocess.run([str(binary), 'runtests', '-xiangqi-inference', str(path),
                             str(request_path), str(response_path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(response_path.read_text())
    assert actual['fp16'] == fp16
    assert actual['policy_channels'] == (3 if with_q else 2)
    assert actual['spatial_policy_channels'] == 50 * actual['policy_channels']
    for batch, count in zip(actual['batches'], request['batches']):
        assert len(batch) == count
        for row, reference in zip(batch, expected):
            assert len(row['policy']) == 2086
            for field in reference:
                np.testing.assert_allclose(np.asarray(row[field]) * actual['scale'], reference[field],
                                           atol=3e-3 if fp16 else 5e-4, rtol=3e-3 if fp16 else 5e-4)
    if strict_cuda and not fp16:
        # cuDNN may use TF32 in its default FP32 path. Also compare the same
        # source kernels in IEEE mode; this flag is test-only, not a run default.
        strict_path = tmp_path / 'ieee-response.json'
        strict = subprocess.run([str(binary), 'runtests', '-xiangqi-inference', str(path),
                                 str(request_path), str(strict_path)], capture_output=True, text=True, timeout=120,
                                env=dict(os.environ, NVIDIA_TF32_OVERRIDE='0'))
        assert strict.returncode == 0, strict.stdout + strict.stderr
        strict_output = json.loads(strict_path.read_text())
        for batch in strict_output['batches']:
            for row, reference in zip(batch, expected):
                for field in reference:
                    np.testing.assert_allclose(np.asarray(row[field]) * strict_output['scale'], reference[field],
                                               atol=2e-5, rtol=2e-5)
    request['rows'][0]['symmetry'] = 1
    request_path.write_text(json.dumps(request))
    invalid = subprocess.run([str(binary), 'runtests', '-xiangqi-inference', str(path),
                              str(request_path), str(tmp_path / 'invalid.json')], capture_output=True, text=True, timeout=120)
    assert invalid.returncode != 0 and 'horizontal reflection' in invalid.stderr
