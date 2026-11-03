"""Source position features + fixed latent orientation vs independent PyTorch inference."""
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
from test_muzero_cuda import ieee_reference
from test_muzero_model import component_config

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA')


def transform(tensor, symmetry):
    if symmetry & 1:
        tensor = tensor.flip(-2)
    if symmetry & 2:
        tensor = tensor.flip(-1)
    if symmetry & 4:
        tensor = tensor.transpose(-2, -1)
    return tensor.contiguous()


def untransform(tensor, symmetry):
    if symmetry & 4:
        tensor = tensor.transpose(-2, -1)
    if symmetry & 2:
        tensor = tensor.flip(-1)
    if symmetry & 1:
        tensor = tensor.flip(-2)
    return tensor.contiguous()


@pytest.mark.parametrize('fp16', [False, True])
@pytest.mark.parametrize('game,metadata,canvas', [('go', False, 5), ('go', True, 5), ('renju', False, 5),
                                                ('hex', False, 5), ('xiangqi', False, 10), ('xiangqi', False, 12)])
def test_position_cache_symmetry_and_latent_domain(tmp_path, game, metadata, canvas, fp16):
    check_position_cache_symmetry(tmp_path, game, metadata, canvas, fp16)


def check_position_cache_symmetry(tmp_path, game, metadata, canvas, fp16, *, binary=None, tensor_device="cuda", inputs_nhwc=True, backend_config=None):
    torch.manual_seed(94)
    representation = component_config(game, 'nbt' if game == 'renju' else 'regular')
    if metadata:
        representation['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    model = MuZeroModel(MuZeroConfig(game, 6, representation, component_config(game, 'nbt'),
                                    component_config(game, 'regular')), canvas)
    model.initialize()
    model.eval()
    with torch.no_grad():
        model.prediction.value_head.linear_valuehead.bias.copy_(torch.tensor([.7, -.6, .2, -.3]))
    export_components(model, tmp_path / 'model', 'position-test')
    (tmp_path / 'request.json').write_text(json.dumps(dict(fp16=fp16, inputs_nhwc=inputs_nhwc, cuda_config=backend_config or {})))
    result = subprocess.run([str(binary or ROOT / 'build/katago'), 'runtests', '-muzero-position',
                             str(tmp_path / 'model'), str(tmp_path / 'request.json'), str(tmp_path / 'response.json')],
                            env=dict(os.environ, NVIDIA_TF32_OVERRIDE='0'), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads((tmp_path / 'response.json').read_text())
    model.to(tensor_device)
    errors = {}
    with torch.no_grad():
        for case in actual['cases']:
            symmetry = case['symmetry']
            spatial = torch.tensor(case['spatial'], device=tensor_device).reshape(1, -1, canvas, canvas)
            global_input = torch.tensor(case['global'], device=tensor_device).reshape(1, -1)
            meta = torch.tensor(case['metadata'], device=tensor_device).reshape(1, -1) if metadata else None
            if game == 'xiangqi':
                # Features already use the root player's perspective. Mirror
                # only nine files; padding never moves to the opposite edge.
                oriented = spatial.clone()
                if symmetry == 2:
                    oriented[:, :, :10, :9] = spatial[:, :, :10, :9].flip(-1)
                hidden, heads = model.initial(oriented, global_input)
                static = np.ones(2086, dtype=bool)
                # Independently map absolute actions to the fixed root frame.
                permutation = []
                for x, y, u, v in xq.MOVES:
                    if case['player'] == 2:
                        x, y, u, v = 8-x, 9-y, 8-u, 9-v
                    if symmetry == 2:
                        x, u = 8-x, 8-u
                    permutation.append(xq.INDEX[x, y, u, v])
            else:
                hidden, heads = model.initial(transform(spatial, symmetry), global_input, meta)
                static = np.zeros(26, dtype=bool)
                static[:25].reshape(5, 5)[:case['ysize'], :case['xsize']] = True
                static[25] = game == 'go'
            for step, output in enumerate(case['outputs']):
                if step:
                    action = case['actions'][step-1]
                    if game == 'xiangqi':
                        action = permutation[action]
                    elif action != 25:
                        plane = torch.zeros(5, 5)
                        plane.flatten()[action] = 1
                        action = int(transform(plane, symmetry).flatten().argmax())
                    hidden, heads = model.recurrent(hidden, torch.tensor([action], device=tensor_device))
                policy, value, misc, more, _ = heads[0]
                mixed = policy[0, 0] + (policy[0, 5] - policy[0, 0]) * case['optimism']
                if game == 'xiangqi':
                    logits = mixed[permutation]
                else:
                    logits = torch.empty(26, device=tensor_device)
                    logits[:25] = untransform(mixed[:25].reshape(5, 5), symmetry).flatten()
                    logits[25] = mixed[25] if game == 'go' else -1e30
                legal = static.copy()
                if step == 0:
                    # Fixture has two (black to move) or three (white to move)
                    # live stones on the first row; no captures, ko or suicide.
                    if game == 'xiangqi':
                        legal[:] = False
                        legal[case['legal']] = True
                    else:
                        legal[:2 if case['player'] == 1 else 3] = False
                probs = torch.softmax(logits[legal].double()/case['temperature'], dim=0).cpu().numpy()
                expected_policy = np.full(len(legal), -1.)
                expected_policy[legal] = probs
                values = value[0].double().clone()
                if case['suppress_nr']:
                    values[3] -= 100000
                expected_value = values.softmax(0).cpu().numpy()
                player = case['player'] if step % 2 == 0 else 3-case['player']
                assert output['player'] == player and output['symmetry'] == symmetry
                assert output['root_player'] == case['player']
                if player == 1:
                    expected_value = expected_value[[2, 1, 0, 3]]
                expected_aux = np.array([torch.nn.functional.softplus(misc[0, 0].double()).item()*40,
                                        torch.nn.functional.softplus(more[0, 0].double()*.5).item()*.5])
                for key, expected in dict(policy=expected_policy, value=expected_value, auxiliary=expected_aux).items():
                    values = np.array(output[key])
                    errors[key] = max(errors.get(key, 0), float(np.max(np.abs(values-expected))))
                    np.testing.assert_allclose(values, expected, atol=.015 if fp16 else 5e-5,
                                               rtol=.015 if fp16 else 3e-4, err_msg=f'{symmetry}/{step}: {key}')
                np.testing.assert_array_equal(np.array(output['policy']) >= 0, legal)
                assert output['optimism'] == pytest.approx(case['optimism'])
                assert output['value'][3] == 0 if case['suppress_nr'] else output['value'][3] > 0
    for average in actual['averages']:
        assert len(average['individual']) == (2 if game in ('hex', 'xiangqi') else 8)
        for key in ('policy', 'value', 'auxiliary'):
            reference = np.array([row[key] for row in average['individual']]).mean(axis=0)
            np.testing.assert_allclose(average['average'][key], reference, atol=5e-5, rtol=3e-4)
            np.testing.assert_allclose(average['recurrent'][key], average['selected_recurrent'][key], atol=1e-6)
    (tmp_path / 'errors.json').write_text(json.dumps(errors, indent=2))
