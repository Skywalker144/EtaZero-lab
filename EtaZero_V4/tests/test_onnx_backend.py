"""Native ONNX Runtime CPU inference and source graph serialization."""
import gzip
import json
import os
import subprocess

import numpy as np
import pytest

from etazero import ROOT
from test_eigen_backend import cpu_threads
from test_four_results import check_exported_four_logits, model
from test_placement_network import check_placement_export
from test_xiangqi_export import check_joint_export, export
from test_xiangqi_evaluator import check_positions
from test_xiangqi_search import check_joint_search
from test_xiangqi_controller import check_xiangqi_two_rounds
from test_benchmarknn import model_for
from test_backend_reference import reference, dump, configure
from test_gpu_error import config_for, run, check
from test_evalsgf import GAMES, record


pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_ONNX_TESTS') != '1',
                                reason='requires native ONNX Runtime build')
BINARY = ROOT / 'build-onnx/katago'
CONFIG = dict(onnxProvider='cpu', onnxUseFP16='false', inputsUseNHWC='false')


@pytest.mark.parametrize('skip_scale8,pad_batch', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('with_q', [False, True])
def test_joint_logits_symmetry_padding_and_batch_reuse(tmp_path, skip_scale8, pad_batch, with_q):
    config = CONFIG | dict(onnxSkipScale8=str(skip_scale8).lower(), onnxPadBatch=str(pad_batch).lower())
    check_joint_export(tmp_path, BINARY, False, False, with_q, False, backend_config=config)


@pytest.mark.parametrize('with_q', [False, True])
def test_placement_rules_and_four_results(tmp_path, with_q):
    check_placement_export(tmp_path, BINARY, False, with_q, backend_config=CONFIG)


@pytest.mark.parametrize('with_q', [False, True])
def test_go_four_results_and_rule_mask(tmp_path, model, with_q):
    check_exported_four_logits(tmp_path, BINARY, model, False, with_q, backend_config=CONFIG)


@pytest.mark.parametrize('kind,nhwc', [('regular', True), ('transformer', False), ('transformer', True)])
def test_positions_history_cache_and_parallel_queue(tmp_path, kind, nhwc):
    config = CONFIG | dict(onnxTransformerNHWC=str(nhwc).lower())
    check_positions(tmp_path, BINARY, False, kind, inputs_nhwc=False, backend_config=config)


@pytest.mark.parametrize('gumbel,threads,human_mode', [(False, 1, None), (True, 4, None), (False, 4, 'weightless')])
def test_puct_gumbel_and_humansl(tmp_path, gumbel, threads, human_mode):
    check_joint_search(tmp_path, BINARY, gumbel, threads, human_mode, False,
                       inputs_nhwc=False, backend_config=CONFIG)


@pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1', reason='requires independent Eigen reference')
@pytest.mark.parametrize('game', GAMES)
def test_native_precision_comparison_to_eigen(tmp_path, game):
    model = model_for(tmp_path / 'model', game, False)
    native = config_for(tmp_path / 'native.cfg', game, fp16=False, **CONFIG)
    cpu = config_for(tmp_path / 'eigen.cfg', game, fp16=False, eigenUseFP16='false')
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game))
    args = ['-sgf', sgf, '-reference-file', tmp_path / 'reference.jsonl']
    check(run(cpu, model, *args, binary=ROOT / 'build-eigen/katago'), fp16=False, count=3)
    check(run(native, model, *args, binary=BINARY), fp16=False, count=3)


@pytest.mark.parametrize('game,kind,metadata', [('go', 'transformer', True), ('renju', 'nbt', False),
                                             ('xiangqi', 'transformer', False)])
@pytest.mark.parametrize('skip_scale8', [False, True])
def test_dump_graph_and_reload_without_double_scaling(tmp_path, game, kind, metadata, skip_scale8):
    import onnx

    model = model_for(tmp_path / 'model', game, False, kind=kind, metadata=metadata)
    side = 10 if game == 'xiangqi' else 5
    config = CONFIG | dict(maxBoardSizeForNNBuffer=side, onnxSkipScale8=str(skip_scale8).lower())
    cfg = configure(tmp_path / 'native.cfg', game, **config)
    ref = tmp_path / 'positions.jsonl'; reference(ref, game)
    expected = dump(cfg, model, ref, tmp_path / 'binary-values.jsonl', binary=BINARY)
    graph_path = tmp_path / 'model.onnx'
    command = [str(BINARY), 'dumponnx', '-model', str(model), '-out', str(graph_path),
               '-nn-x-len', str(side), '-nn-y-len', str(side)]
    if skip_scale8: command.append('-skip-scale8')
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    graph = onnx.load(graph_path); onnx.checker.check_model(graph)
    outputs = {o.name: o.type.tensor_type.shape.dim[1].dim_value for o in graph.graph.output}
    assert outputs['OutputValue'] == 4 and outputs['OutputAuxValue'] == 2
    assert outputs['OutputPolicy'] == (150 if game == 'xiangqi' else 3)
    assert ('OutputPolicyPass' in outputs) == (game == 'go')
    assert len(outputs) == (4 if game == 'go' else 3)
    # Loaded graphs retain their encoded scale regardless of the current emit
    # preference. Check both the native .onnx and source .onnx.gz reader paths.
    reload_cfg = configure(tmp_path / 'reload.cfg', game, **(config | dict(onnxSkipScale8=str(not skip_scale8).lower())))
    compressed = tmp_path / 'model.onnx.gz'; compressed.write_bytes(gzip.compress(graph_path.read_bytes()))
    for path in (graph_path, compressed):
        actual = dump(reload_cfg, path, ref, tmp_path / f'{path.name}.values.jsonl', binary=BINARY)
        for got, wanted in zip(actual, expected):
            for key in ['policy', 'whiteWinProb', 'whiteDrawProb', 'whiteLossProb', 'whiteNoResultProb',
                        'shorttermWinlossError', 'winrate']:
                np.testing.assert_allclose(got[key], wanted[key], rtol=1e-6, atol=1e-7, err_msg=key)


def test_source_onnx_file_reader(tmp_path):
    result = subprocess.run([str(BINARY), 'runonnxmodelfiletests', str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_canvas_precision_and_input_layout_boundaries(tmp_path):
    _, path = export(tmp_path, False)
    cases = [(8, 10, False, False, False, 'physical 9x10'),
             (10, 9, False, False, False, 'physical 9x10'),
             (10, 10, True, False, False, 'requireExactNNLen'),
             (9, 10, True, False, False, None),
             (10, 10, False, True, False, 'useFP16 = true is not supported'),
             (10, 10, False, False, True, 'inputsUseNHWC = true not supported')]
    for width, height, exact, fp16, nhwc, message in cases:
        spatial = [1.] * (width * height) + [0.] * (112 * width * height)
        request = dict(width=width, height=height, fp16=fp16, inputs_nhwc=nhwc, batches=[1],
                       backend_config=CONFIG, require_exact=exact,
                       rows=[dict(spatial=spatial, **{'global': [0.] * 8}, symmetry=0, optimism=.4)])
        request_path = tmp_path / 'request.json'; request_path.write_text(json.dumps(request))
        output_path = tmp_path / 'response.json'
        result = subprocess.run([str(BINARY), 'runtests', '-xiangqi-inference', str(path),
                                 str(request_path), str(output_path)], capture_output=True, text=True, timeout=60)
        if message:
            assert result.returncode != 0 and message in result.stderr, result.stdout + result.stderr
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            row = json.loads(output_path.read_text())['batches'][0][0]
            assert len(row['policy']) == 2086
            assert all(np.isfinite(row[key]).all() for key in ('policy', 'value', 'auxiliary'))


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires CUDA FP16 learner')
@pytest.mark.parametrize('algorithm,weights,canvas', [('alphazero', 'ordinary', 10), ('gumbel_alphazero', 'swa', 12)])
def test_native_selfplay_two_rounds_publication_and_resume(tmp_path, algorithm, weights, canvas):
    check_xiangqi_two_rounds(tmp_path, algorithm, weights, canvas, binary=BINARY, backend_config=CONFIG)
