"""Independent secondary latent states within native PUCT/Gumbel HumanSL search."""
import json
import os
import subprocess

import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from export_muzero import export_components
from katago.train.model_pytorch import Model
from katago.train.muzero_model import MuZeroConfig, MuZeroModel
from test_muzero_model import component_config
from test_muzero_analysis import cuda


@cuda
@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('fp16', [False, True])
def test_native_secondary_state_and_human_policy(tmp_path, game, fp16):
    torch.manual_seed(2491)
    canvas = 10 if game == 'xiangqi' else 5
    for name, width in [('primary', 6), ('human', 8)]:
        representation = component_config(game)
        if name == 'human' and game == 'go':
            representation['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
        model = MuZeroModel(MuZeroConfig(game, width, representation, component_config(game, 'nbt'),
                                        component_config(game)), canvas)
        model.initialize(); model.eval()
        export_components(model, tmp_path / name, name)
    settings = component_config(game)
    if game == 'go':
        settings['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    ordinary = Model(settings, canvas)
    ordinary.initialize(); ordinary.eval()
    with (tmp_path / 'human.bin').open('wb') as out:
        write_model_binary(ordinary, out, 'ordinary-human')
    (tmp_path / 'request.json').write_text(json.dumps(dict(
        fp16=fp16, humanBundle=str(tmp_path / 'human'), humanBoardModel=str(tmp_path / 'human.bin'))))
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-muzero-search',
                             str(tmp_path / 'primary'), str(tmp_path / 'request.json'), str(tmp_path / 'response.json')],
                            env=dict(os.environ, NVIDIA_TF32_OVERRIDE='0'), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    cases = json.loads((tmp_path / 'response.json').read_text())
    assert len(cases) == 36
    assert {case['algorithm'] for case in cases} == {0, 1, 2}
    assert {case['threads'] for case in cases} == {1, 4}
    assert {case['mode'] for case in cases} == set(range(6))
    assert all(case['depth'] > 2 for case in cases if case['mode'] == 5)
