"""Real source CUDA initial/recurrent components vs PyTorch, including retained states."""
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_muzero import export_components
from katago.train.muzero_model import MuZeroModel, MuZeroConfig
from katago.game import xiangqi as xq
from test_muzero_model import component_config, batch_for

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA')


@pytest.fixture(autouse=True)
def ieee_reference():
    threads = torch.get_num_threads()
    cudnn, matmul = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_num_threads(threads)
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = cudnn, matmul


@pytest.mark.parametrize('fp16', [False, True])
@pytest.mark.parametrize('game,variant', [('go', 'regular'), ('renju', 'nbt'), ('hex', 'mixed_transformer'),
                                        ('renju', 'constant'), ('go', 'small_span'), ('go', 'metadata'),
                                        ('xiangqi', 'regular'), ('xiangqi', 'nbt'), ('xiangqi', 'mixed_transformer'), ('go', 'heads_only'), ('renju', 'heads_only'),
                                        ('hex', 'heads_only'), ('xiangqi', 'heads_only')])
def test_native_initial_recurrent_matches_pytorch(tmp_path, game, variant, fp16):
    check_native_initial_recurrent(tmp_path, game, variant, fp16)


def check_native_initial_recurrent(tmp_path, game, variant, fp16, *, binary=None, tensor_device="cuda", inputs_nhwc=None, backend_config=None):
    torch.manual_seed(71)
    representation = component_config(game, 'transformer' if variant == 'mixed_transformer' else
                                      'nbt' if variant == 'nbt' else 'regular')
    dynamics = component_config(game, 'nbt')
    prediction = component_config(game, 'regular')
    if variant == 'heads_only':
        prediction.update(block_kind=[], trunk_num_channels=6)
    if variant == 'metadata':
        representation['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    canvas = (12 if variant == 'nbt' else 10) if game == 'xiangqi' else 5
    model = MuZeroModel(MuZeroConfig(game, 6, representation, dynamics, prediction), canvas)
    model.initialize(); model.eval()
    if variant in ('constant', 'small_span'):
        with torch.no_grad():
            for body in (model.representation, model.dynamics):
                body.projection.weight.mul_(0 if variant == 'constant' else 1e-6)
    root, _ = batch_for(model)
    metadata = torch.randn(3, 192) if variant == 'metadata' else None
    export_components(model, tmp_path / 'model', 'native-test')
    # Reorder rows, expand one parent twice, mix states from different batches,
    # and revisit the same action; no real successor observation is supplied.
    steps = [dict(states=[[0, 2], [0, 0]], actions=[0, 25 if game == 'go' else 1]),
             dict(states=[[0, 0], [1, 0], [0, 1]], actions=[1, 0, 3]),
             dict(states=[[1, 0], [1, 0]], actions=[0, 2]),
             # A second Go pass remains a learned transition, not a real-board
             # terminal. Also cover the one-row batch after larger batches.
             dict(states=[[1, 1]], actions=[25 if game == 'go' else 1])]
    if game == 'xiangqi':
        # Cover both board edges, long moves, horse and palace geometry; the
        # second recurrence also reuses a root encoded before the first batch.
        joint_actions = [[xq.INDEX[8, 9, 8, 0], xq.INDEX[1, 0, 2, 2]],
                         [xq.INDEX[4, 0, 4, 1], xq.INDEX[8, 9, 8, 0], 2085],
                         [xq.INDEX[4, 8, 5, 9], xq.INDEX[2, 0, 4, 2]],
                         [xq.INDEX[1, 0, 2, 2]]]
        for step, actions in zip(steps, joint_actions):
            step['actions'] = actions
    request = dict(batch=3, fp16=fp16, spatial=root['binaryInputNCHW'].flatten().tolist(),
                   global_=root['globalInputNC'].flatten().tolist(), steps=steps,
                   lifecycle=variant == 'regular', inputs_nhwc=(variant not in ('nbt', 'constant')) if inputs_nhwc is None else inputs_nhwc)
    request['warmup'] = game == 'xiangqi' and variant == 'mixed_transformer'
    request['global'] = request.pop('global_')
    if metadata is not None:
        request['metadata'] = metadata.flatten().tolist()
    if not request['inputs_nhwc']:
        # Exercise the source-supported NCHW input/compute combination, including FP16.
        request['cuda_config'] = {'cudaUseNHWC': 'false'}
    if backend_config is not None:
        request['cuda_config'] = backend_config
    (tmp_path / 'request.json').write_text(json.dumps(request))
    command = [str(binary or ROOT / 'build/katago'), 'runtests', '-muzero-inference',
               str(tmp_path / 'model'), str(tmp_path / 'request.json'), str(tmp_path / 'response.json')]
    result = subprocess.run(command,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads((tmp_path / 'response.json').read_text())
    comparisons = [('default', actual, .012 if fp16 else .002 if tensor_device == 'cuda' else 5e-5)]
    if not fp16 and tensor_device == 'cuda':
        # The source cuDNN FP32 path may use TF32. Keep its production default,
        # including per-batch convolution algorithm selection, and independently
        # require IEEE agreement with NVIDIA's diagnostic flag on the same model.
        result = subprocess.run(command[:-1] + [str(tmp_path / 'response-ieee.json')],
                                env=dict(os.environ, NVIDIA_TF32_OVERRIDE='0'), capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        comparisons.append(('ieee', json.loads((tmp_path / 'response-ieee.json').read_text()), 5e-5))
    model.to(tensor_device)
    # Compare against FP32 PyTorch on the reference device; FP16 is allowed its
    # rounding error, never a change in layout, state, mask or value perspective.
    with torch.no_grad():
        expected = [model.initial(root['binaryInputNCHW'].to(tensor_device), root['globalInputNC'].to(tensor_device),
                                  None if metadata is None else metadata.to(tensor_device))]
        for step in steps:
            states = torch.stack([expected[generation][0][row] for generation, row in step['states']])
            expected.append(model.recurrent(states, torch.tensor(step['actions'], device=tensor_device)))
    errors = {}
    for mode, actual, tolerance in comparisons:
        errors[mode] = {}
        for rows, (hidden, heads) in zip(actual, expected):
            policy, value, misc, more, _future = heads[0]
            policy = policy[:, [0, 5, 6]]
            target = dict(latent=hidden, policy=policy, value=value, auxiliary=torch.cat((misc[:, :1], more[:, :1]), 1))
            for key, tensor in target.items():
                tensor = tensor.detach().float().cpu().numpy()
                values = np.array([row[key] for row in rows], dtype=float).reshape(tensor.shape)
                assert np.isfinite(values).all(), key
                errors[mode][key] = max(errors[mode].get(key, 0), float(np.max(np.abs(values - tensor))))
                np.testing.assert_allclose(values, tensor, rtol=.015 if fp16 else 3e-4,
                                           atol=tolerance, err_msg=key)
            masks = hidden[:, -1].cpu().numpy()
            native_hidden = np.array([row['latent'] for row in rows]).reshape(hidden.shape)
            np.testing.assert_array_equal(native_hidden[:, -1], masks)
            assert np.count_nonzero(native_hidden[:, :-1] * (1 - masks[:, None])) == 0
    (tmp_path / 'errors.json').write_text(json.dumps(errors, indent=2))
