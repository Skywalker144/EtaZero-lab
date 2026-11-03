"""Native OpenCL latent transitions against independent CPU PyTorch values."""
import os

import pytest

from etazero import ROOT
from test_opencl_backend import precision, binary, cpu_threads
from test_muzero_cuda import check_native_initial_recurrent
from test_muzero_batcher import check_batcher_concurrent_inference
from test_muzero_search import check_native_latent_search
from test_muzero_position import check_position_cache_symmetry
from test_muzero_controller import check_muzero_two_rounds
from test_gpu_error import config_for, run, check
from test_benchmarknn import model_for


pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_OPENCL_TESTS') != '1',
                                reason='requires native OpenCL build and device')


@pytest.mark.parametrize('game,variant', [('go', 'regular'), ('freestyle', 'regular'), ('standard', 'regular'),
                                        ('renju', 'nbt'), ('hex', 'mixed_transformer'), ('renju', 'constant'),
                                        ('go', 'small_span'), ('go', 'metadata'), ('xiangqi', 'regular'),
                                        ('xiangqi', 'nbt'), ('xiangqi', 'mixed_transformer'), ('go', 'heads_only'), ('renju', 'heads_only'),
                                        ('hex', 'heads_only'), ('xiangqi', 'heads_only')])
def test_initial_and_recurrent_against_pytorch(tmp_path, binary, precision, game, variant):
    fp16, config = precision
    check_native_initial_recurrent(tmp_path, game, variant, fp16, binary=binary,
                                   tensor_device='cpu', inputs_nhwc=False, backend_config=config)


@pytest.mark.parametrize('game,variant', [('go', 'metadata'), ('renju', 'nbt'), ('hex', 'transformer'), ('xiangqi', 'regular')])
def test_concurrent_workers_and_lifetime(tmp_path, binary, precision, game, variant):
    fp16, config = precision
    check_batcher_concurrent_inference(tmp_path, game, variant, fp16, binary=binary,
                                       tensor_device='cpu', inputs_nhwc=False, backend_config=config)


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
def test_puct_gumbel_and_latent_static_domain(tmp_path, binary, precision, game):
    fp16, config = precision
    check_native_latent_search(tmp_path, game, fp16, binary=binary, inputs_nhwc=False, backend_config=config)


@pytest.mark.parametrize('game,metadata,canvas', [('go', False, 5), ('go', True, 5), ('freestyle', False, 5),
                                               ('standard', False, 5), ('renju', False, 5), ('hex', False, 5),
                                               ('xiangqi', False, 10), ('xiangqi', False, 12)])
def test_position_symmetry_cache_and_history(tmp_path, binary, precision, game, metadata, canvas):
    fp16, config = precision
    check_position_cache_symmetry(tmp_path, game, metadata, canvas, fp16, binary=binary,
                                  tensor_device='cpu', inputs_nhwc=False, backend_config=config)


@pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1', reason='requires Eigen reference exporter')
@pytest.mark.parametrize('game', ['go', 'xiangqi'])
def test_multidepth_native_reference(tmp_path, binary, precision, game):
    fp16, options = precision
    model = model_for(tmp_path / 'model', game, True,
                      kind='transformer' if game == 'go' else 'regular', metadata=game == 'go')
    cpu = config_for(tmp_path / 'cpu.cfg', game, True, eigenUseFP16='false')
    native = config_for(tmp_path / 'native.cfg', game, True, **options)
    sgf = tmp_path / 'game.sgf'
    sgf.write_text('(;GM[1]SZ[4:5]RU[Japanese]KM[0];B[aa];W[dc];B[];W[])' if game == 'go' else
                   '(;GM[7]SZ[9:10]RU[xiangqi]XQHIST[a3a4][a6a5]XQTURN[97];B[b0c2];W[b9c7])')
    args = ['-sgf', sgf, '-reference-file', tmp_path / 'reference.jsonl']
    check(run(cpu, model, *args, binary=ROOT / 'build-eigen/katago'), muzero=True, fp16=False)
    check(run(native, model, *args, binary=binary), muzero=True, fp16=fp16)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires CUDA learner')
@pytest.mark.parametrize('game,algorithm,interior,weights,precision', [
    ('go', 'muzero', 'puct', 'ordinary', 'fp16_storage'),
    ('xiangqi', 'gumbel_muzero', 'gumbel', 'swa', 'fp16_tensor'),
], indirect=['precision'])
def test_two_rounds_publish_and_resume(tmp_path, binary, precision, game, algorithm, interior, weights):
    _, options = precision
    check_muzero_two_rounds(tmp_path, game, algorithm, interior, weights, False,
                           binary=binary, fp16=True, backend_config=options)
