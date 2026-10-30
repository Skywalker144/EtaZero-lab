"""Actual source PUCT/Gumbel tree traversal over native CUDA latent states."""
import json
import os
import subprocess

import pytest
import torch

from etazero import ROOT
from export_muzero import export_components
from katago.train.muzero_model import MuZeroModel, MuZeroConfig
from test_muzero_model import component_config
from test_muzero_cuda import ieee_reference

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA')


@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('fp16', [False, True])
def test_native_latent_puct_and_gumbel(tmp_path, game, fp16):
    check_native_latent_search(tmp_path, game, fp16)


def check_native_latent_search(tmp_path, game, fp16, *, binary=None, inputs_nhwc=True, backend_config=None):
    torch.manual_seed(107)
    model = MuZeroModel(MuZeroConfig(game, 6, component_config(game, 'regular'),
                                    component_config(game, 'nbt'), component_config(game, 'regular')), 10 if game == 'xiangqi' else 5)
    model.initialize()
    model.eval()
    export_components(model, tmp_path / 'model', 'search-test')
    (tmp_path / 'request.json').write_text(json.dumps(dict(fp16=fp16, inputs_nhwc=inputs_nhwc, cuda_config=backend_config or {})))
    result = subprocess.run([str(binary or ROOT / 'build/katago'), 'runtests', '-muzero-search',
                             str(tmp_path / 'model'), str(tmp_path / 'request.json'), str(tmp_path / 'response.json')],
                            env=dict(os.environ, NVIDIA_TF32_OVERRIDE='0'), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    cases = json.loads((tmp_path / 'response.json').read_text())
    assert len(cases) == 12
    assert {case['mode'] for case in cases} == {0, 1, 2}
    assert {case['threads'] for case in cases} == {1, 4}
    for case in cases:
        assert 32 <= case['visits'] <= 32 + case['threads']
        assert case['visits'] == case['nodes'] + case['edgeCatchups']
        if case['threads'] == 1:
            assert case['nodes'] == 32
            assert case['edgeCatchups'] == 0
        if case['chain']:
            assert case['depth'] == case['nodes'] - 1
            assert case['depth'] > 2
