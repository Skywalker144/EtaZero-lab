"""Native initialization sampling keeps GameRunner, rule history and model services."""
import os
import re
import subprocess

import numpy as np
import pytest

from etazero import ROOT
from test_evalsgf import GAMES
from test_surprise_dump import configure as selfplay_config, constant_model


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA initialization')


def configure(path, game, muzero=False, gumbel=False, random=True, **changes):
    settings = dict(scoringRules='AREA', taxRules='NONE', koRules='POSITIONAL',
        maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5, requireMaxBoardSize='false',
        maxVisits=4, useUncertainty='false', useReanalyze='false', logToStdout='false')
    settings.update(changes)
    return selfplay_config(path, game, muzero, gumbel, random, **settings)


def run(config, *, model='/dev/null', evaluate=True, count=2, binary=None):
    result = subprocess.run([str(binary or ROOT / 'build/katago'), 'sampleinitializations',
        '-config', str(config), '-model', str(model), '-num', str(count),
        *(['-evaluate'] if evaluate else [])], capture_output=True, text=True, timeout=90)
    config.with_suffix('.log').write_text(result.stdout + result.stderr)
    return result


def outputs(result, game, *, evaluate=True, count=2):
    assert result.returncode == 0, result.stdout + result.stderr
    text = result.stdout
    if game == 'xiangqi':
        assert len(re.findall(r'^Xiangqi FEN: ', text, re.M)) == count
    if game != 'go':
        assert len(re.findall(rf'^{game}$', text, re.M)) == count
    values = np.array([list(map(float, row.split())) for row in re.findall(r'^White W/D/L/NR: (.*)$', text, re.M)])
    if not evaluate:
        assert len(values) == 0 and 'Winloss:' not in text
        return values
    assert values.shape == (count, 4) and np.isfinite(values).all()
    assert (values >= 0).all() and (values <= 1).all()
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=2e-6)
    winloss = np.array(list(map(float, re.findall(r'^Winloss: (.*)$', text, re.M))))
    np.testing.assert_allclose(winloss, values[:, 0] - values[:, 2], atol=2e-6)
    if game != 'go':
        assert (values[:, 3] == 0).all()
    assert 'Lead:' not in text
    return values


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('evaluate', [False, True])
def test_native_random_initializations(tmp_path, game, evaluate):
    config = configure(tmp_path / 'sample.cfg', game)
    outputs(run(config, evaluate=evaluate), game, evaluate=evaluate)


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_real_models_and_search(tmp_path, game, muzero, gumbel):
    check_real_models_and_search(tmp_path, game, muzero, gumbel)


def check_real_models_and_search(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True):
    model = constant_model(tmp_path / 'model', game, muzero)
    config = configure(tmp_path / 'sample.cfg', game, muzero, gumbel, False, cudaUseFP16=str(fp16).lower())
    values = outputs(run(config, model=model, binary=binary), game)
    # Constant symmetric prediction has no view ambiguity. Go's positional-ko
    # rules and all non-Go games mask NR before normalizing W/D/L.
    np.testing.assert_allclose(values, np.tile([2/9, 5/9, 2/9, 0], (2, 1)), atol=1e-4)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_imported_xiangqi_history_and_policy_terminal(tmp_path, muzero):
    model = constant_model(tmp_path / 'model', 'xiangqi', muzero)
    positions = tmp_path / 'positions'; positions.mkdir()
    fen = '3aka3/9/9/9/4P4/9/9/9/9/R3K4 w - - 119 60'
    (positions / 'start.sgf').write_text(f'(;GM[7]SZ[9:10]XQFEN[{fen}]XQTURN[73])')
    for initialize in (False, True):
        config = configure(tmp_path / f'policy-{initialize}.cfg', 'xiangqi', muzero, True, False,
            startPosesFromSgfDir=positions, startPosesProb=1, startPosesLoadProb=1, startPosesTurnWeightLambda=0,
            initGamesWithPolicy=str(initialize).lower(), policyInitAreaProp=0,
            startPosesPolicyInitAreaProp=1, policyInitGammaShape=10)
        result = run(config, model=model)
        values = outputs(result, 'xiangqi')
        if initialize:
            # Native policy init plays to the 120-ply draw. No post-terminal
            # representation/search is allowed, even for a constant latent net.
            np.testing.assert_array_equal(values, [[0, 1, 0, 0]] * 2)
            assert result.stdout.count('Terminal position (rule result)') == 2
            assert len(re.findall(r'^Xiangqi FEN: .* b - - 120 60$', result.stdout, re.M)) == 2
        else:
            assert result.stdout.count(f'Xiangqi FEN: {fen}') == 2
            # Despite maxVisits=4 in the config, source diagnostic search uses
            # 20 visits: one root NN + 19 actual successor draws. MuZero keeps
            # latent values, with no successor-rule leakage.
            draw = 5/9 if muzero else (19 + 5/9) / 20
            np.testing.assert_allclose(values[:, 1], draw, atol=2e-6)
            assert 'Terminal position (rule result)' not in result.stdout


@cuda
@pytest.mark.parametrize('game', ['renju', 'hex'])
@pytest.mark.parametrize('muzero', [False, True])
def test_balanced_openings_still_use_game_runner(tmp_path, game, muzero):
    model = constant_model(tmp_path / 'model', game, muzero)
    settings = dict(balanceOpeningProb=1, logToStdout='true')
    if game == 'hex':
        settings['makeOpeningFairRate'] = 1
    config = configure(tmp_path / 'sample.cfg', game, muzero, True, False, **settings)
    result = run(config, model=model, evaluate=False)
    outputs(result, game, evaluate=False)
    lengths = list(map(int, re.findall(r'Balanced opening: \w+ moves=(\d+)', result.stdout)))
    assert len(lengths) == 2 and min(lengths) >= 1
    if game == 'hex':
        assert lengths == [1, 1]


def test_muzero_mismatch_is_rejected_before_sampling(tmp_path):
    model = constant_model(tmp_path / 'model', 'hex', True)
    for game, settings, error in [('renju', {}, 'game differs'), ('hex', {'bSizesXY': '7-7'}, 'canvas must contain')]:
        config = configure(tmp_path / f'{game}.cfg', game, True, False, False, **settings)
        result = run(config, model=model)
        assert result.returncode != 0 and error in result.stdout + result.stderr
        assert 'White W/D/L/NR:' not in result.stdout
