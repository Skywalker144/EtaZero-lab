"""Real CUDA request grouping, worker affinity, shutdown and failure propagation."""
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_muzero import export_components
from katago.train.muzero_model import MuZeroModel, MuZeroConfig
from test_muzero_cuda import ieee_reference  # Shared autouse numerical reference fixture.
from test_muzero_model import component_config, batch_for

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA')


@pytest.mark.parametrize('fp16', [False, True])
@pytest.mark.parametrize('game,variant', [('go', 'metadata'), ('renju', 'nbt'), ('hex', 'transformer')])
def test_batcher_concurrent_inference_and_lifecycle(tmp_path, game, variant, fp16):
    check_batcher_concurrent_inference(tmp_path, game, variant, fp16)


def check_batcher_concurrent_inference(tmp_path, game, variant, fp16, *, binary=None, tensor_device="cuda", inputs_nhwc=True, backend_config=None):
    torch.manual_seed(83)
    representation = component_config(game, 'regular' if variant == 'metadata' else variant)
    if variant == 'metadata':
        representation['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    model = MuZeroModel(MuZeroConfig(game, 6, representation, component_config(game, 'nbt'),
                                    component_config(game, 'regular')), 10 if game == 'xiangqi' else 5)
    model.initialize()
    model.eval()
    root, _ = batch_for(model)
    metadata = torch.randn(3, 192) if variant == 'metadata' else None
    export_components(model, tmp_path / 'model', 'batcher-test')
    request = {'batch': 3, 'fp16': fp16, 'spatial': root['binaryInputNCHW'].flatten().tolist(),
               'global': root['globalInputNC'].flatten().tolist(), 'inputs_nhwc': inputs_nhwc}
    if metadata is not None:
        request['metadata'] = metadata.flatten().tolist()
    if variant == 'nbt':
        request['cuda_config'] = {'cudaUseNHWC': 'false'}
    if backend_config is not None:
        request['cuda_config'] = backend_config
    (tmp_path / 'request.json').write_text(json.dumps(request))
    # Disable TF32 only for this strict comparison; production retains source defaults.
    result = subprocess.run([str(binary or ROOT / 'build/katago'), 'runtests', '-muzero-batcher',
                             str(tmp_path / 'model'), str(tmp_path / 'request.json'), str(tmp_path / 'response.json')],
                            env=dict(os.environ, NVIDIA_TF32_OVERRIDE='0'), timeout=180,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads((tmp_path / 'response.json').read_text())
    assert actual['burst']['rows'] == 24*4
    assert sum(actual['burst']['workers']) == actual['burst']['rows']
    assert actual['capped']['batches'] - actual['burst']['batches'] == 24
    assert actual['drained_rows'] > 0
    assert ('exactly the 9 x 10 board' if game == 'xiangqi' else 'nonempty') in actual['runtime_error']
    if variant == 'transformer':
        expected = 'require NHWC' if tensor_device == 'cuda' else 'inputsUseNHWC = false'
        assert expected in actual['startup_error']

    model.to(tensor_device)
    errors = {}
    with torch.no_grad():
        hidden, heads = model.initial(root['binaryInputNCHW'].repeat(8, 1, 1, 1).to(tensor_device),
                                      root['globalInputNC'].repeat(8, 1).to(tensor_device),
                                      None if metadata is None else metadata.repeat(8, 1).to(tensor_device))
        for step in range(5):
            if step:
                actions = torch.tensor([25 if game == 'go' and i % 2 == 0 else (i+step-1) % 5
                                        for i in range(24)], device=tensor_device)
                hidden, heads = model.recurrent(hidden, actions)
            policy, value, misc, more, _future = heads[0]
            expected = dict(policy=policy[:, [0, 5, 6]], value=value,
                            auxiliary=torch.cat((misc[:, :1], more[:, :1]), 1))
            for key, tensor in expected.items():
                reference = tensor.cpu().numpy()
                values = np.array([row[step][key] for row in actual['outputs']]).reshape(reference.shape)
                errors[key] = max(errors.get(key, 0), float(np.max(np.abs(values-reference))))
                np.testing.assert_allclose(values, reference, atol=.012 if fp16 else 5e-5,
                                           rtol=.015 if fp16 else 3e-4, err_msg=f'{step}: {key}')
    (tmp_path / 'errors.json').write_text(json.dumps(errors, indent=2))
