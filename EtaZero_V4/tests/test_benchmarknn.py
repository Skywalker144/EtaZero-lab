"""Native forward timing, game inputs, precision and worker lifetimes."""
import json
import math
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


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA inference')
GAMES = ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi']


def config_for(path, game='go', muzero=False, **options):
    values = dict(game=game, useMuZero=muzero, numNNServerThreadsPerModel=2,
                  gpuToUse=0, nnMaxBatchSize=3, cudaUseFP16=False,
                  cudaUseNHWC=True, inputsUseNHWC=True)
    values.update(options)
    path.write_text('\n'.join(f'{key} = {str(value).lower() if isinstance(value, bool) else value}'
                              for key, value in values.items() if value is not None) + '\n')
    return path


def run(config, model='/dev/null', *args, json_output=True, sizes='3,5'):
    command = [str(ROOT / 'build/katago'), 'benchmarknn', '-config', str(config), '-model', str(model),
               '-iterations', '3', '-warmup', '1']
    if sizes is not None:
        command += ['-boardsize', sizes]
    if json_output:
        command += ['-json']
    result = subprocess.run(command + list(args), capture_output=True, text=True, timeout=120)
    attempt = len(list(config.parent.glob(config.stem + '.benchmarknn-*.log')))
    config.with_suffix(f'.benchmarknn-{attempt}.log').write_text(result.stdout + result.stderr)
    return result


def model_for(path, game, muzero, *, kind='regular', metadata=False):
    torch.manual_seed(520)
    canvas = 10 if game == 'xiangqi' else 5
    representation = component_config(game, kind)
    if metadata:
        representation['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    if muzero:
        net = MuZeroModel(MuZeroConfig(game, 6, representation, component_config(game, 'nbt'),
                                      component_config(game)), canvas)
        net.initialize(); net.eval()
        export_components(net, path, 'benchmarknn-test')
    else:
        path = path.with_suffix('.bin')
        net = Model(representation, canvas)
        net.initialize(); net.eval()
        with path.open('wb') as stream:
            write_model_binary(net, stream, 'benchmarknn-test')
    return path


def read_result(result):
    assert result.returncode == 0, result.stdout + result.stderr
    # No backend log may contaminate the machine-readable output.
    info = json.loads(result.stdout)
    assert info['revision'] and info['modelName'] and info['modelFile']
    assert info['gpuIdxs'] == [0]
    return info


def check_metrics(metrics, fp16, *, threads=2, batch=3, iterations=3):
    assert metrics['usingFP16'] is fp16
    assert (metrics['numThreads'], metrics['batchSize'], metrics['numIterations']) == (threads, batch, iterations)
    medians = metrics['perThreadMedianMs']
    rates = metrics['perThreadNNEvalsPerSec']
    assert len(medians) == len(rates) == threads
    assert all(math.isfinite(x) and x > 0 for x in medians + rates)
    assert all(ms * rate == pytest.approx(batch * 1000) for ms, rate in zip(medians, rates))
    assert metrics['sumMedianNNEvalsPerSec'] == pytest.approx(sum(rates))
    wall, rate = metrics['actualWallSeconds'], metrics['actualWallNNEvalsPerSec']
    assert math.isfinite(wall) and wall > 0 and math.isfinite(rate) and rate > 0
    assert wall * rate == pytest.approx(threads * batch * iterations)


@pytest.mark.parametrize('options,args,sizes,error', [
    ({}, ['-iterations', '0'], '5', 'iterations must be positive'),
    ({}, ['-warmup', '-1'], '5', 'warmup must be nonnegative'),
    ({}, ['-batch-size', '0'], '5', 'batch-size must be positive'),
    ({}, ['-muzero-operation', 'invalid'], '5', 'invalid muzero-operation'),
    ({}, ['-muzero-operation', 'initial'], '5', 'requires useMuZero=true'),
    ({}, [], '1', 'invalid board size'),
    ({}, [], '5x5x5', 'invalid board size'),
    ({'game': 'xiangqi'}, [], '10', 'Xiangqi requires a 9x10 board'),
    ({'game': 'hex'}, [], '4x5', 'placement games require square boards'),
    ({'game': 'hex', 'rules': 'renju'}, [], '5', 'config game and rules differ'),
    ({}, ['-require-exact-nnlen'], '3,5', 'needs a single board size'),
    ({'debugSkipNeuralNet': True}, [], '5', 'requires a real neural net model'),
])
def test_invalid_requests(tmp_path, options, args, sizes, error):
    config = config_for(tmp_path / 'nn.cfg', **options)
    # Build a single invocation rather than passing duplicate TCLAP arguments.
    command = [str(ROOT / 'build/katago'), 'benchmarknn', '-config', str(config), '-model', '/dev/null',
               '-boardsize', sizes, '-json'] + args
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
    assert not result.stdout.strip()


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero', [False, True])
@pytest.mark.parametrize('fp16', [False, True])
def test_game_model_precision_and_timing(tmp_path, game, muzero, fp16):
    model = model_for(tmp_path / 'model', game, muzero)
    config = config_for(tmp_path / 'nn.cfg', game, muzero, cudaUseFP16=fp16,
                        inputsUseNHWC=fp16, cudaUseNHWC=fp16)
    sizes = None if game == 'xiangqi' else '3x5,5x5' if game == 'go' else '3,5'
    info = read_result(run(config, model, sizes=sizes))
    expected_shapes = [[9, 10]] if game == 'xiangqi' else [[3, 5], [5, 5]] if game == 'go' else [[3, 3], [5, 5]]
    assert info['game'] == game and info['boardShapes'] == expected_shapes
    assert info['nnBufferSize'] == ([10, 10] if muzero and game == 'xiangqi' else
                                    [9, 10] if game == 'xiangqi' else [5, 5])
    assert info['requireExactNNLen'] is False
    if game not in ('go', 'xiangqi'):
        assert info['boardSizes'] == [3, 5]
    else:
        assert 'boardSizes' not in info
    if muzero:
        assert set(info['operations']) == {'initial', 'recurrent'}
        for metrics in info['operations'].values():
            check_metrics(metrics, fp16)
    else:
        assert 'operations' not in info
        check_metrics(info, fp16)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
@pytest.mark.parametrize('variant', ['metadata', 'transformer'])
def test_metadata_transformer_exact_size_and_text(tmp_path, muzero, variant):
    game = 'go' if variant == 'metadata' else 'hex'
    model = model_for(tmp_path / 'model', game, muzero, metadata=variant == 'metadata',
                      kind='transformer' if variant == 'transformer' else 'regular')
    config = config_for(tmp_path / 'nn.cfg', game, muzero, cudaUseFP16=True)
    info = read_result(run(config, model, '-require-exact-nnlen', '-batch-size', '2', sizes='5'))
    assert info['requireExactNNLen'] is True and info['boardSizes'] == [5]
    for metrics in info['operations'].values() if muzero else [info]:
        check_metrics(metrics, True, batch=2)
    for operation in ['initial', 'recurrent'] if muzero else ['']:
        args = ['-muzero-operation', operation] if muzero else []
        result = run(config, model, *args, json_output=False, sizes='5')
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count('=== benchmarknn') == 1
        assert f'=== benchmarknn{(" " + operation) if operation else ""} ===' in result.stdout
        assert 'overall throughput (wall):' in result.stdout


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_defaults_buffer_and_failures(tmp_path, muzero):
    model = model_for(tmp_path / 'model', 'renju', muzero)
    config = config_for(tmp_path / 'nn.cfg', None, muzero)
    # Legacy ordinary v18 defaults to Freestyle; a bundle supplies its own game
    # and canvas. The ordinary source default remains 19 even for a small net.
    info = read_result(run(config, model, sizes=None))
    assert info['game'] == ('renju' if muzero else 'freestyle')
    assert info['boardSizes'] == ([5] if muzero else [19])
    for metrics in info['operations'].values() if muzero else [info]:
        check_metrics(metrics, False)
    if muzero:
        for operation in ['initial', 'recurrent']:
            info = read_result(run(config, model, '-muzero-operation', operation, sizes='5'))
            assert set(info['operations']) == {operation}
            check_metrics(info['operations'][operation], False)
    else:
        config_for(config, 'renju', maxBoardSizeForNNBuffer=7)
        info = read_result(run(config, model, sizes='5'))
        assert info['nnBufferSize'] == [7, 7]

    cases = [
        (dict(game='go'), [], '5', 'MuZero model and game differ' if muzero else 'model and game rules differ'),
        (dict(game='renju', requireMaxBoardSize=True, maxBoardSizeForNNBuffer=5), [], '3', 'does not fill the NN buffer'),
        (dict(game='renju', maxBoardSizeForNNBuffer=3), [], '5', 'buffer must match model canvas' if muzero else 'out of range'),
    ]
    if muzero:
        cases += [(dict(game='renju'), [], '6', 'board size exceeds MuZero canvas'),
                  (dict(game='renju', debugSkipNeuralNet=True), [], '5', 'requires a real model')]
    for options, args, sizes, error in cases:
        config_for(config, muzero=muzero, **options)
        result = run(config, model, *args, sizes=sizes)
        assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
        assert not result.stdout.strip()
    # One failed worker must release the start barrier and return an error,
    # even when another worker has already prepared and warmed up successfully.
    config_for(config, 'renju', muzero, gpuToUseModel0Thread1=1023)
    result = run(config, model, sizes='5')
    assert result.returncode != 0 and not result.stdout.strip(), result.stdout + result.stderr


@cuda
def test_xiangqi_canvas_exact_and_rules_inference(tmp_path):
    model = model_for(tmp_path / 'model', 'xiangqi', True)
    config = config_for(tmp_path / 'nn.cfg', None, True, rules='xiangqi')
    info = read_result(run(config, model, sizes=None))
    assert info['game'] == 'xiangqi' and info['boardShapes'] == [[9, 10]]
    result = run(config, model, '-require-exact-nnlen', sizes=None)
    assert result.returncode != 0 and 'does not fill the NN buffer' in result.stderr
