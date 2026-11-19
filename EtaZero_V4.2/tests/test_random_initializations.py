"""Native random policy prefixes, actual terminal results and bounded diagnostics."""
import csv
import io
import os
import re
import signal
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from export_muzero import export_components
from test_controller_recovery import wait_until
from test_evalsgf import GAMES, make_model
from test_initializations import configure as sample_config


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA initialization')
# Fixed seeds of the source Rand: first requested lengths are 0 and 110. This
# makes real search and rule-terminal prefixes explicit, without tuning a model.
ZERO_SEED, LONG_SEED = 'init-test-261', 'init-test-2'
HEADER = 'requestedInitialMoves,whiteWinLoss,whiteWin,whiteDraw,whiteLoss,whiteNoResult,valueSource,initialMoves'


def configure(path, game, muzero=False, gumbel=False, random=True, **changes):
    settings = dict(defaultBoardXSize=9 if game == 'xiangqi' else 5,
                    defaultBoardYSize=10 if game == 'xiangqi' else 5,
                    fpuReductionMax=0, rootFpuReductionMax=0, rootPolicyOptimism=0, policyOptimism=0)
    settings.update(changes)
    return sample_config(path, game, muzero, gumbel, random, **settings)


def command(config, model='/dev/null', *, count=1, seed=ZERO_SEED, binary=None):
    return [str(binary or ROOT / 'build/katago'), 'evalrandominits', '-config', str(config), '-model', str(model),
            *(['-num', str(count)] if count is not None else []), *(['-seed', seed] if seed is not None else [])]


def rows(text, game, count):
    assert HEADER in text
    result = list(csv.DictReader(io.StringIO(text[text.index(HEADER):])))
    assert len(result) == count
    for row in result:
        assert None not in row and None not in row.values()
        requested, actual = int(row['requestedInitialMoves']), int(row['initialMoves'])
        assert 0 <= actual <= requested < 200
        values = np.array([float(row[f'white{s}']) for s in ['Win', 'Draw', 'Loss', 'NoResult']])
        assert np.isfinite(values).all() and (values >= 0).all()
        assert values.sum() == pytest.approx(1, abs=2e-6)
        assert float(row['whiteWinLoss']) == pytest.approx(values[0] - values[2], abs=2e-6)
        assert row['valueSource'] in ['rule', 'search']
        if game != 'go':
            assert values[-1] == 0
        else:
            # Source permits explicit post-terminal Go analysis.
            assert row['valueSource'] == 'search'
        if row['valueSource'] == 'rule':
            assert sum(values == 1) == 1 and sum(values == 0) == 3
            assert actual > 0
    return result


def run(config, model='/dev/null', **options):
    result = subprocess.run(command(config, model, **options), capture_output=True, text=True, timeout=120)
    index = len(list(config.parent.glob('random-inits-*.log')))
    (config.parent / f'random-inits-{index}.log').write_text(result.stdout + result.stderr)
    return result


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('gumbel', [False, True])
def test_finite_native_random_prefixes(tmp_path, game, gumbel):
    config = configure(tmp_path / 'inits.cfg', game, gumbel=gumbel)
    result = run(config, count=3)
    assert result.returncode == 0, result.stdout + result.stderr
    data = rows(result.stdout, game, 3)
    assert data[0]['requestedInitialMoves'] == data[0]['initialMoves'] == '0'
    assert data[0]['valueSource'] == 'search'
    # A separate fixed stream exercises initialization transitions, including
    # guaranteed rule termination on the 5x5 placement boards.
    result = run(config, seed=LONG_SEED)
    assert result.returncode == 0, result.stdout + result.stderr
    row, = rows(result.stdout, game, 1)
    assert row['requestedInitialMoves'] == '110' and int(row['initialMoves']) > 0
    if game in ['freestyle', 'standard', 'renju', 'hex']:
        assert row['valueSource'] == 'rule' and int(row['initialMoves']) <= 25


def constant_model(path, game, muzero):
    net, _ = make_model(path.with_name('initial'), game, muzero)
    head = net.prediction if muzero else net
    with torch.no_grad():
        head.value_head.linear_valuehead.weight.zero_()
        head.value_head.linear_valuehead.bias.copy_(torch.log(torch.tensor([.2, .5, .2, .1])))
        head.policy_head.conv2p.weight.zero_()
        if head.policy_head.has_pass:
            # Keep the zero-prefix Go oracle away from a two-pass terminal.
            # This changes only the test model, not native sampling behavior.
            head.policy_head.linear_pass.weight.zero_()
            head.policy_head.linear_pass.bias.fill_(1)
            head.policy_head.linear_pass2.weight.fill_(-30 / head.policy_head.linear_pass2.in_features)
    if muzero:
        export_components(net, path, 'random-init-values')
    else:
        path = path.with_suffix('.bin')
        with path.open('wb') as out:
            write_model_binary(net, out, 'random-init-values')
    return path


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_real_models_search_and_policy_transitions(tmp_path, game, muzero, gumbel):
    check_real_models(tmp_path, game, muzero, gumbel)


def check_real_models(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True):
    model = constant_model(tmp_path / 'model', game, muzero)
    config = configure(tmp_path / 'inits.cfg', game, muzero, gumbel, False, cudaUseFP16=str(fp16).lower())
    for seed in [ZERO_SEED, LONG_SEED]:
        result = run(config, model, seed=seed, binary=binary)
        assert result.returncode == 0, result.stdout + result.stderr
        row, = rows(result.stdout, game, 1)
        if seed == ZERO_SEED:
            assert row['initialMoves'] == row['requestedInitialMoves'] == '0' and row['valueSource'] == 'search'
            expected = [.2, .5, .2, .1] if game == 'go' else [2/9, 5/9, 2/9, 0]
            np.testing.assert_allclose([float(row[f'white{s}']) for s in ['Win', 'Draw', 'Loss', 'NoResult']], expected, atol=1e-4)
        elif game in ['freestyle', 'standard', 'renju', 'hex']:
            assert row['valueSource'] == 'rule' and 0 < int(row['initialMoves']) <= 25
        else:
            assert int(row['initialMoves']) > 0


def test_source_go_defaults_and_rule_game_inference(tmp_path):
    config = configure(tmp_path / 'go.cfg', 'go', maxBoardSizeForNNBuffer=19, logToStderr='true')
    config.write_text(re.sub(r'^defaultBoard[XY]Size = .*\n', '', config.read_text(), flags=re.M) + 'rules = chinese\n')
    result = run(config)
    assert result.returncode == 0, result.stdout + result.stderr
    rows(result.stdout, 'go', 1)
    assert re.search(r'Initialization rules: .*scoreTERRITORY.*board 19x19', result.stderr)
    config = configure(tmp_path / 'hex.cfg', 'hex')
    config.write_text(re.sub(r'^game = .*\n', '', config.read_text(), flags=re.M) + 'rules = hex\n')
    result = run(config)
    assert result.returncode == 0, result.stdout + result.stderr
    rows(result.stdout, 'hex', 1)


def test_invalid_count_game_shape_and_bundle(tmp_path):
    for game, changes, count, message in [
        ('hex', {}, -1, '-num must be non-negative'),
        ('hex', {'rules': 'renju'}, 1, 'game and rules differ'),
        ('xiangqi', {'defaultBoardXSize': 10}, 1, '9x10 board'),
        ('renju', {'defaultBoardXSize': 4}, 1, 'square board'),
    ]:
        config = configure(tmp_path / f'{game}-{count}.cfg', game, **changes)
        result = run(config, count=count)
        assert result.returncode != 0 and message in result.stdout + result.stderr
        assert HEADER not in result.stdout
    model = constant_model(tmp_path / 'model', 'hex', True)
    for game, changes, message in [('renju', {}, 'game differs'),
                                  ('hex', {'defaultBoardXSize': 7, 'defaultBoardYSize': 7}, 'canvas must contain')]:
        config = configure(tmp_path / f'{game}.cfg', game, muzero=True, random=False, **changes)
        result = run(config, model)
        assert result.returncode != 0 and message in result.stdout + result.stderr
        assert HEADER not in result.stdout


def check_stop(tmp_path, config, model, count, stop_signal):
    output = tmp_path / 'rows.csv'
    with output.open('w') as stdout, (tmp_path / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command(config, model, count=count, seed=LONG_SEED), stdout=stdout, stderr=stderr)
        try:
            wait_until(lambda: len(output.read_text().splitlines()) >= 3, seconds=30)
            assert process.poll() is None
            process.send_signal(stop_signal)
            assert process.wait(timeout=15) == 0
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    text = output.read_text()
    data = rows(text, 'hex', len(text.splitlines()) - 1)
    assert len(data) >= 2


@pytest.mark.parametrize('count,stop_signal', [(None, signal.SIGINT), (0, signal.SIGTERM)])
def test_default_and_zero_count_run_until_stopped(tmp_path, count, stop_signal):
    config = configure(tmp_path / 'inits.cfg', 'hex', gumbel=True)
    check_stop(tmp_path, config, '/dev/null', count, stop_signal)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_real_model_shutdown_drains_service(tmp_path, muzero):
    model = constant_model(tmp_path / 'model', 'hex', muzero)
    config = configure(tmp_path / 'inits.cfg', 'hex', muzero, True, False)
    check_stop(tmp_path, config, model, None, signal.SIGTERM)
