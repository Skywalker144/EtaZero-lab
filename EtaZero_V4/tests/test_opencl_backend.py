"""Production OpenCL with the existing PyTorch and independent Eigen oracles.

The explicit tiny-kernel profiles test correctness, not performance or tuning.
Tensor-core cases require a device supported by the source WMMA kernels and
ETAZERO_OPENCL_TENSOR_TESTS=1, in addition to ETAZERO_OPENCL_TESTS=1.
"""
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from test_four_results import check_exported_four_logits, model
from test_placement_network import check_placement_export
from test_xiangqi_export import check_joint_export, export
from test_xiangqi_evaluator import check_positions
from test_xiangqi_search import check_joint_search
from test_backend_reference import reference, dump, use_outputs, write_rows, limits, run, configure
from test_benchmarknn import model_for


pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_OPENCL_TESTS') != '1',
                                reason='requires native OpenCL build and device')


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def binary():
    binary = ROOT / 'build-opencl/katago'
    assert binary.is_file(), 'Build USE_BACKEND=OPENCL before requesting its checks'
    return binary


@pytest.fixture(params=['fp32', 'fp16_storage', 'fp16_tensor'])
def precision(request, tmp_path):
    name = request.param
    tensor = name == 'fp16_tensor'
    if tensor and os.environ.get('ETAZERO_OPENCL_TENSOR_TESTS') != '1':
        pytest.skip('requires a device supporting the source WMMA kernels')
    fp16 = name != 'fp32'
    # Native OpenCLTuneParams serialization, version 13. Each parameter line
    # sets one documented member; the source parser supplies the other defaults.
    # Explicit profiles avoid running a performance search during numeric tests.
    capability = [1, 0, int(tensor), int(tensor)]
    parameters = ['WGD=8', 'MWG=8', 'MWG=8', 'MWG=16', 'MWG=16',
                  'INTILE_XSIZE=4', 'INTILE_XSIZE=6', 'XYSTRIDE=1',
                  'ATTN_BLOCK_Q=32', 'WG_C_SIZE=64', 'ELTS_PER_THREAD=1',
                  'XY_ELTS_PER_THREAD=1', 'TILE_SIZE=32']
    tuner = tmp_path / 'opencl-tune.txt'
    tuner.write_text('\n'.join(['VERSION=13', *map(str, capability * 2), *parameters]) + '\n')
    return fp16, {'openclTunerFile': str(tuner), 'openclUseFP16': str(fp16).lower(),
                  'inputsUseNHWC': 'false'}


@pytest.mark.parametrize('with_q', [False, True])
def test_joint_logits_mirror_padding_and_batch_reuse(tmp_path, binary, precision, with_q):
    fp16, config = precision
    check_joint_export(tmp_path, binary, fp16, False, with_q, False, backend_config=config)


@pytest.mark.parametrize('kind', ['regular', 'transformer'])
def test_positions_history_cache_and_parallel_queue(tmp_path, binary, precision, kind):
    fp16, config = precision
    check_positions(tmp_path, binary, fp16, kind, inputs_nhwc=False, backend_config=config)


@pytest.mark.parametrize('gumbel,threads,human_mode', [(False, 1, None), (True, 4, None), (False, 4, 'weightless')])
def test_native_joint_search(tmp_path, binary, precision, gumbel, threads, human_mode):
    fp16, config = precision
    check_joint_search(tmp_path, binary, gumbel, threads, human_mode, fp16,
                       inputs_nhwc=False, backend_config=config)


@pytest.mark.parametrize('with_q', [False, True])
def test_placement_rules_symmetry_and_export(tmp_path, binary, precision, with_q):
    fp16, config = precision
    check_placement_export(tmp_path, binary, fp16, with_q, backend_config=config)


@pytest.mark.parametrize('with_q', [False, True])
def test_go_four_results_rule_mask_and_auxiliary(tmp_path, binary, precision, model, with_q):
    fp16, config = precision
    check_exported_four_logits(tmp_path, binary, model, fp16, with_q, backend_config=config)


@pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1', reason='requires independent Eigen build')
def test_go_metadata_transformer_against_eigen(tmp_path, binary, precision):
    _, config = precision
    model_path = model_for(tmp_path / 'model', 'go', False, kind='transformer', metadata=True)
    ref = tmp_path / 'reference.jsonl'
    rows = reference(ref, 'go')
    eigen_config = configure(tmp_path / 'eigen.cfg', 'go', maxBoardSizeForNNBuffer=5)
    outputs = dump(eigen_config, model_path, ref, tmp_path / 'eigen.jsonl', binary=ROOT / 'build-eigen/katago')
    use_outputs(rows, outputs); write_rows(ref, rows)
    opencl_config = configure(tmp_path / 'opencl.cfg', 'go', maxBoardSizeForNNBuffer=5, **config)
    result = run(opencl_config, model_path, '-reference-data-file', ref,
                 '-reference-limits-file', limits(tmp_path / 'limits.json'), binary=binary)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASSED' in result.stdout and 'ignored' not in result.stdout


def test_input_layout_rejects_nhwc(tmp_path, binary, precision):
    fp16, config = precision
    _, path = export(tmp_path, False)
    request = dict(width=10, height=10, fp16=fp16, inputs_nhwc=True, batches=[1], backend_config=config,
                   rows=[dict(spatial=[0.] * 11300, **{'global': [0.] * 8}, symmetry=0, optimism=0)])
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps(request))
    result = subprocess.run([str(binary), 'runtests', '-xiangqi-inference', str(path),
                             str(request_path), str(tmp_path / 'response.json')],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and 'inputsUseNHWC = false required' in result.stderr


def test_physical_canvas_and_exact_dimensions(tmp_path, binary, precision):
    fp16, config = precision
    _, path = export(tmp_path, False)
    for width, height, exact, message in [(8, 10, False, 'physical 9x10'),
                                         (10, 9, False, 'physical 9x10'),
                                         (10, 10, True, 'requireExactNNLen'),
                                         (9, 10, True, None)]:
        spatial = [1.] * (width * height) + [0.] * (112 * width * height)
        request = dict(width=width, height=height, fp16=fp16, inputs_nhwc=False, batches=[1],
                       backend_config=config, require_exact=exact,
                       rows=[dict(spatial=spatial, **{'global': [0.] * 8}, symmetry=0, optimism=.4)])
        request_path = tmp_path / 'request.json'; request_path.write_text(json.dumps(request))
        output_path = tmp_path / 'response.json'
        result = subprocess.run([str(binary), 'runtests', '-xiangqi-inference', str(path),
                                 str(request_path), str(output_path)], capture_output=True, text=True, timeout=120)
        if message:
            assert result.returncode != 0 and message in result.stderr
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            output = json.loads(output_path.read_text())
            assert output['fp16'] == fp16
            row = output['batches'][0][0]
            assert len(row['policy']) == 2086
            assert all(np.isfinite(row[key]).all() for key in ('policy', 'value', 'auxiliary'))
