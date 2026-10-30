"""Source benchmark sampling, tuning and reports across board/latent searches."""
import os
import math
import re
import subprocess

import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from export_muzero import export_components
from katago.train.model_pytorch import Model
from katago.train.muzero_model import MuZeroConfig, MuZeroModel
from test_muzero_model import component_config
from test_gtp import gtp_config, run_gtp
from test_xiangqi_gtp import config_for_gtp


def config_for(path, game, **options):
    options.setdefault('defaultBoardSize', 7 if game == 'go' else 5)
    if game == 'xiangqi':
        options.pop('defaultBoardSize')
        return config_for_gtp(path, **options)
    return gtp_config(path, game, **options)


def benchmark(config, model='/dev/null', *args, threads='1,4'):
    command = [str(ROOT / 'build/katago'), 'benchmark', '-config', str(config), '-model', str(model),
               '-visits', '8', '-numpositions', '3']
    if threads is not None:
        command += ['-threads', threads]
    result = subprocess.run(command + list(args), capture_output=True, text=True, timeout=180)
    attempt = len(list(config.parent.glob(config.stem + '.benchmark-*.log')))
    config.with_suffix(f'.benchmark-{attempt}.log').write_text(result.stdout + result.stderr)
    return result


def check_results(result, elo, *, synthetic=True, positions=3, neural=False):
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'{positions} / {positions} positions' in result.stdout
    rows = re.findall(r'visits/s = (\S+) nnEvals/s = (\S+) nnBatches/s = (\S+) avgBatchSize = (\S+)', result.stdout)
    assert rows and all(float(row[0]) > 0 for row in rows)
    if neural:
        assert all(math.isfinite(float(v)) and float(v) > 0 for row in rows for v in row)
    else:
        assert all(float(row[1]) == float(row[2]) == 0 for row in rows)
    assert ('EloDiff' in result.stdout) == elo
    assert ('synthetic legal' in result.stdout + result.stderr) == synthetic
    if not elo:
        assert 'no Elo estimate is reported' in result.stdout


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('gumbel', [False, True])
def test_native_benchmark_and_gtp(tmp_path, game, gumbel):
    config = config_for(tmp_path / 'benchmark.cfg', game, gumbel=gumbel)
    result = benchmark(config, '/dev/null', '-no-server-thread-test', '-no-half-batch-size-test')
    check_results(result, game == 'go' and not gumbel, synthetic=game != 'go')
    move = 'b0c2' if game == 'xiangqi' else 'C3'
    replies = run_gtp(config, [f'play B {move}', 'printsgf', 'kata-benchmark 8',
        'printsgf', 'kata-search W', 'kata-benchmark 0', 'kata-benchmark nonsense', 'kata-search W', 'quit'])
    assert [error for error, _ in replies] == [False] * 5 + [True, True, False, False], replies
    assert replies[1][1] == replies[3][1]
    assert 'nnEvals/s' in replies[2][1] and 'EloDiff' not in replies[2][1]


@pytest.mark.parametrize('game,sgf', [
    ('go', '(;GM[1]SZ[5]KM[6.5];B[aa];W[ab];B[ba];W[bb])'),
    ('renju', '(;GM[4]SZ[5]RU[renju];B[aa];W[ab];B[ba];W[bb])'),
    ('hex', '(;GM[11]SZ[5]RU[hex];B[aa];W[ab];B[ba];W[bb])'),
    ('xiangqi', '(;GM[7]SZ[9:10]RU[xiangqi];B[a3a4];W[a6a5];B[b0c2];W[b9c7])'),
])
def test_sgf_game_and_size(tmp_path, game, sgf):
    path = tmp_path / 'positions.sgf'; path.write_text(sgf)
    config = config_for(tmp_path / 'benchmark.cfg', game, gumbel=True)
    result = benchmark(config, '/dev/null', '-sgf', str(path), '-no-server-thread-test', '-no-half-batch-size-test')
    check_results(result, False, synthetic=False)
    text = config.read_text().replace(f'game = {game}\n', 'game = standard\n')
    text = re.sub(r'^rules = .*$', 'rules = standard', text, flags=re.M)
    config.write_text(text)
    result = benchmark(config, '/dev/null', '-sgf', str(path))
    assert result.returncode != 0 and 'config and SGF game differ' in result.stderr


def test_empty_sgf_rejected_and_tuning_runs(tmp_path):
    config = config_for(tmp_path / 'benchmark.cfg', 'hex', gumbel=True)
    path = tmp_path / 'empty.sgf'; path.write_text('(;GM[11]SZ[5]RU[hex])')
    result = benchmark(config, '/dev/null', '-sgf', str(path))
    assert result.returncode != 0 and 'needs moves' in result.stderr
    result = benchmark(config, '/dev/null', '-no-server-thread-test', '-no-half-batch-size-test', threads=None)
    check_results(result, False)
    assert 'Ordered summary of results:' in result.stdout
    assert 'Highest measured throughput:' in result.stdout


def test_rules_inference_and_xiangqi_sgf_history(tmp_path):
    config = config_for(tmp_path / 'benchmark.cfg', 'xiangqi', gumbel=True)
    config.write_text(config.read_text().replace('game = xiangqi\n', ''))
    result = benchmark(config, '/dev/null', '-no-server-thread-test', '-no-half-batch-size-test')
    check_results(result, False)
    path = tmp_path / 'history.sgf'
    # The live board starts after two reversible plies. Replaying the mainline
    # must retain their complete history through the source SGF adapter.
    path.write_text('(;GM[7]SZ[9:10]RU[xiangqi]'
        'XQFEN[rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1]'
        'XQHIST[b0c2][b9c7];B[c2b0];W[c7b9];B[a3a4];W[a6a5])')
    result = benchmark(config, '/dev/null', '-sgf', str(path), '-no-server-thread-test', '-no-half-batch-size-test')
    check_results(result, False, synthetic=False)
    config.write_text(config.read_text().replace('rules = xiangqi\n', ''))
    result = benchmark(config, '/dev/null', '-sgf', str(path), '-no-server-thread-test', '-no-half-batch-size-test')
    check_results(result, False, synthetic=False)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA benchmark')
@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('muzero,fp16,gumbel', [(False, False, False), (False, True, True), (True, False, True), (True, True, False)])
def test_cuda_model_lifetime_and_tuning(tmp_path, game, muzero, fp16, gumbel):
    torch.manual_seed(519)
    canvas = 10 if game == 'xiangqi' else 7 if game == 'go' else 5
    if muzero:
        network = MuZeroModel(MuZeroConfig(game, 6, component_config(game), component_config(game, 'nbt'),
                                           component_config(game)), canvas)
        network.initialize(); network.eval()
        model = tmp_path / 'model'
        export_components(network, model, 'benchmark-model')
    else:
        network = Model(component_config(game), canvas)
        network.initialize(); network.eval()
        model = tmp_path / 'model.bin'
        with model.open('wb') as stream:
            write_model_binary(network, stream, 'benchmark-model')
    config = config_for(tmp_path / 'benchmark.cfg', game, muzero=muzero, random=False,
                        gumbel=gumbel, cudaUseFP16=str(fp16).lower())
    # A single requested thread count makes both extra tests run at four threads
    # independently of noisy timing recommendations.
    result = benchmark(config, model, threads='4')
    check_results(result, game == 'go' and not (muzero or gumbel), synthetic=game != 'go', neural=True)
    assert 'Testing 2 NN server threads per GPU' in result.stdout
    assert 'Testing a max batch size of 2' in result.stdout
    assert 'Benchmark SGF' not in result.stderr
    if game == 'hex' and muzero and fp16:
        result = benchmark(config, model, '-no-server-thread-test', '-no-half-batch-size-test', threads=None)
        check_results(result, False, neural=True)
        assert 'Ordered summary of results:' in result.stdout
    result = benchmark(config, model, '-fixed-batch-size', '2', '-no-server-thread-test', threads='1,4')
    check_results(result, game == 'go' and not (muzero or gumbel), synthetic=game != 'go', neural=True)
    replies = run_gtp(config, ['kata-benchmark 8', 'kata-search B', 'quit'], model)
    assert all(not error for error, _ in replies), replies
    assert 'nnEvals/s' in replies[0][1]
