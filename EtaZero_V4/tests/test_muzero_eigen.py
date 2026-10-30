"""Native Eigen MuZero, using the existing independent PyTorch and search checks."""
import json
import os
import subprocess

import pytest
import torch

from etazero import ROOT
from test_muzero_cuda import check_native_initial_recurrent
from test_muzero_batcher import check_batcher_concurrent_inference
from test_muzero_search import check_native_latent_search
from test_muzero_position import check_position_cache_symmetry


pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1', reason='requires native Eigen build')
BINARY = ROOT / 'build-eigen/katago'


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('game,variant', [('go', 'regular'), ('freestyle', 'regular'), ('standard', 'regular'),
                                        ('renju', 'nbt'), ('hex', 'mixed_transformer'), ('renju', 'constant'),
                                        ('go', 'small_span'), ('go', 'metadata'), ('xiangqi', 'regular'),
                                        ('xiangqi', 'nbt'), ('xiangqi', 'mixed_transformer')])
def test_initial_recurrent_against_pytorch(tmp_path, game, variant):
    check_native_initial_recurrent(tmp_path, game, variant, False, binary=BINARY,
                                   tensor_device='cpu', inputs_nhwc=True)


@pytest.mark.parametrize('game,variant', [('go', 'metadata'), ('renju', 'nbt'),
                                        ('hex', 'transformer'), ('xiangqi', 'regular')])
def test_concurrent_batching_and_lifecycle(tmp_path, game, variant):
    check_batcher_concurrent_inference(tmp_path, game, variant, False, binary=BINARY, tensor_device='cpu')


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
def test_latent_puct_and_gumbel(tmp_path, game):
    check_native_latent_search(tmp_path, game, False, binary=BINARY)


@pytest.mark.parametrize('game,metadata,canvas', [('go', False, 5), ('go', True, 5), ('freestyle', False, 5),
                                                ('standard', False, 5), ('renju', False, 5), ('hex', False, 5),
                                                ('xiangqi', False, 10), ('xiangqi', False, 12)])
def test_position_cache_symmetry_and_static_domain(tmp_path, game, metadata, canvas):
    check_position_cache_symmetry(tmp_path, game, metadata, canvas, False, binary=BINARY, tensor_device='cpu')


@pytest.mark.parametrize('option,message', [('fp16', 'useFP16 = true not supported'),
                                           ('nchw', 'inputsUseNHWC = false unsupported')])
def test_native_precision_and_layout_boundaries(tmp_path, option, message):
    from export_muzero import export_components
    from test_muzero_model import make_model
    model = make_model('go', pos_len=5).eval()
    export_components(model, tmp_path / 'model', 'eigen-boundary')
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(dict(batch=1, fp16=option == 'fp16', inputs_nhwc=option != 'nchw')))
    result = subprocess.run([str(BINARY), 'runtests', '-muzero-inference', str(tmp_path / 'model'),
                             str(request), str(tmp_path / 'response.json')], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and message in result.stdout + result.stderr


@pytest.mark.parametrize('game,algorithm,interior,weights', [
    ('go', 'muzero', 'puct', 'ordinary'), ('xiangqi', 'gumbel_muzero', 'gumbel', 'swa')])
def test_two_round_training_publication_and_resume(tmp_path, monkeypatch, game, algorithm, interior, weights):
    from test_muzero_controller import check_muzero_two_rounds
    # Source learner selects CPU when CUDA is unavailable. Keep this backend
    # check usable without a GPU, including the learner's zero-sample startup.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    check_muzero_two_rounds(tmp_path, game, algorithm, interior, weights, False, binary=BINARY, fp16=False)
