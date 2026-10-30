"""Shared source serialization and native component-format boundary checks."""
from copy import deepcopy
import io
import json
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from export_muzero import export_components
from katago.train.muzero_model import MuZeroConfig, MuZeroModel
from test_muzero_model import make_model, component_config


def native_desc(path, *, success=True):
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-muzero-model-desc', str(path)],
                            capture_output=True, text=True)
    if success:
        assert result.returncode == 0, result.stderr + result.stdout
        return json.loads(result.stdout)
    assert result.returncode != 0
    return result.stdout + result.stderr


@pytest.mark.parametrize('game,kind', [('go', 'regular'), ('renju', 'nbt'),
                                      ('hex', 'transformer'), ('go', 'plain_transformer'),
                                      ('xiangqi', 'regular'), ('xiangqi', 'nbt'), ('xiangqi', 'transformer')])
def test_components_use_source_layers_and_native_lifecycle(tmp_path, game, kind):
    model = make_model(game, intermediate=True, kind=kind)
    before = deepcopy(model.state_dict())
    directory = tmp_path / 'components'
    export_components(model, directory, 'test')
    assert json.loads((directory / 'model.json').read_text()) == dict(game=game, pos_len=model.pos_len, latent_channels=6)
    for role, name in enumerate(('representation', 'dynamics', 'prediction'), 1):
        desc = native_desc(directory / f'{name}.bin')
        component = getattr(model, name)
        trunk = component.trunk if role != 3 else component
        assert desc['role'] == role and desc['latent_channels'] == 6
        assert desc['spatial_channels'] == trunk.bin_input_shape[0]
        assert desc['global_channels'] == trunk.global_input_shape[0]
        if role != 3:
            assert desc['policy_channels'] == desc['value_channels'] == desc['aux_channels'] == 0
            # The native parser converts file [y,x,input,output] back to OIYX.
            # Source scale8 must
            # compensate here to retain the small-span guard's absolute units.
            expected = component.projection.weight.detach().flatten().numpy()
            np.testing.assert_array_equal(desc['projection_weights'], expected * (8 if desc['scale8'] else 1))
        else:
            assert (desc['policy_channels'], desc['value_channels'], desc['aux_channels']) == (3, 4, 2)
            if game == 'xiangqi':
                assert desc['spatial_policy_channels'] == 3*50
        assert desc['parameters'] > 0 and desc['convolutions'] > 0
    assert model.training
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        export_components(model, directory, 'test')


@pytest.mark.parametrize('game', ['renju', 'xiangqi'])
def test_native_rejects_component_dimension_role_and_trailing_corruption(tmp_path, game):
    directory = tmp_path / 'components'
    export_components(make_model(game), directory, 'test')
    payload = (directory / 'dynamics.bin').read_bytes()
    # The source header precedes all binary weights. G is latent width, H is role.
    header, rest = payload.split(b'\n', 14)[:14], payload.split(b'\n', 14)[14]
    for field, value, message in [(3, b'1', 'input dimensions'), (12, b'5', 'input dimensions'),
                                  (13, b'0', 'role or latent width'), (13, b'9', 'role or latent width')]:
        changed = header.copy(); changed[field] = value
        path = tmp_path / f'invalid-{field}-{value.decode()}.bin'
        path.write_bytes(b'\n'.join(changed) + b'\n' + rest)
        assert message in native_desc(path, success=False)
    tail = tmp_path / 'tail.bin'; tail.write_bytes(payload + b'old-head\n')
    assert 'trailing model data' in native_desc(tail, success=False)
    truncated = tmp_path / 'truncated.bin'; truncated.write_bytes(payload[:-16])
    native_desc(truncated, success=False)


def test_binary_writer_rejects_role_mismatch():
    model = make_model()
    with pytest.raises(ValueError, match='role'):
        write_model_binary(model.prediction, io.BytesIO(), 'bad', muzero_component=2, latent_channels=6)
    with pytest.raises(ValueError, match='positive latent width'):
        write_model_binary(model.prediction, io.BytesIO(), 'bad', muzero_component=3)


def test_go_version15_components_and_export_attention_guard(tmp_path):
    config = component_config('go')
    config.update(version=15, predict_q_values=False)
    model = MuZeroModel(MuZeroConfig('go', 6, config, config, config), 5)
    model.initialize()
    directory = tmp_path / 'v15'
    export_components(model, directory, 'v15')
    for name in ('representation', 'dynamics', 'prediction'):
        desc = native_desc(directory / f'{name}.bin')
        assert desc['policy_channels'] == (2 if name == 'prediction' else 0)
    rejected = tmp_path / 'unsafe-attention'
    with pytest.raises(Exception, match='Attention logit bound'):
        export_components(make_model(kind='transformer'), rejected, 'unsafe', attn_logit_bound_limit=0)
    assert not (rejected / 'model.json').exists()


def test_native_bundle_rejects_mismatched_components_before_device_creation(tmp_path):
    directory = tmp_path / 'components'
    export_components(make_model(), directory, 'test')
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(dict(batch=1, fp16=False)))
    manifest = directory / 'model.json'
    original = json.loads(manifest.read_text())
    for field, value, message in [('latent_channels', 7, 'widths'), ('game', 'go', 'layouts'),
                                  ('game', 'xiangqi', 'dimensions'), ('game', 'unknown', 'not connected'),
                                  ('pos_len', 0, 'dimensions'),
                                  ('latent_channels', 2**31-1, '2^31')]:
        manifest.write_text(json.dumps(dict(original, **{field: value})))
        result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-muzero-inference',
                                 str(directory), str(request), str(tmp_path / 'response.json')],
                                capture_output=True, text=True)
        if 'MuZero native inference currently requires the CUDA, Eigen or OpenCL backend' in result.stderr:
            pytest.skip('native MuZero backend was not built')
        assert result.returncode != 0 and message in result.stderr, result.stdout + result.stderr


def test_native_xiangqi_bundle_rejects_other_game_components_before_device_creation(tmp_path):
    directory = tmp_path / 'xiangqi'
    placement = tmp_path / 'renju'
    export_components(make_model('xiangqi'), directory, 'joint')
    export_components(make_model('renju', pos_len=10), placement, 'points')
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(dict(batch=1, fp16=False)))
    for component in ('representation', 'dynamics', 'prediction'):
        path = directory / f'{component}.bin'
        original = path.read_bytes()
        path.write_bytes((placement / path.name).read_bytes())
        result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-muzero-inference',
                                 str(directory), str(request), str(tmp_path / 'response.json')],
                                capture_output=True, text=True)
        path.write_bytes(original)
        if 'MuZero native inference currently requires the CUDA, Eigen or OpenCL backend' in result.stderr:
            pytest.skip('native MuZero backend was not built')
        assert result.returncode != 0 and 'game layouts differ' in result.stderr, result.stdout + result.stderr
