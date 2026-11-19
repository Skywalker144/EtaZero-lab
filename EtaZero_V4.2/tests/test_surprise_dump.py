"""Native per-turn statistics, four-outcome KL, reanalysis and model lifetimes."""
from collections import defaultdict
import csv
import os
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
from test_xiangqi_controller import selfplay_config


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA selfplay')


def configure(path, game, muzero=False, gumbel=False, random=True, **changes):
    values = dict(game=game, bSizesXY='9-10' if game == 'xiangqi' else '5-5', maxMovesPerGame=6,
        koRules='SIMPLE', scoringRules='TERRITORY', taxRules='SEKI', multiStoneSuicideLegals='false', hasButtons='false',
        numGameThreads=2, numSearchThreads=1, maxVisits=16, cheapSearchProb=1, cheapSearchVisits=4,
        cheapSearchTargetWeight=.3, policySurpriseDataWeight=0, valueSurpriseDataWeight=0,
        useReanalyze='true', reanalyzeProp=1, reanalyzePolicySurpriseWeight=1, reanalyzeValueSurpriseWeight=1,
        reanalyzeSurpriseExponent=1, reanalyzeUseOutcomeTargets='true',
        useMuZero=str(muzero).lower(), useGraphSearch=str(not (muzero or gumbel)).lower(),
        rootSearchAlgorithm='gumbel' if gumbel else 'puct', nonrootSearchAlgorithm='gumbel' if gumbel else 'puct',
        gumbelMaxConsideredActions=4, debugSkipNeuralNet=str(random).lower(),
        rootPolicyTemperature=1, rootPolicyTemperatureEarly=1, nnRandomize='false')
    if game == 'hex':
        # These diagnostics exercise played prefixes, not random setup boards.
        values.update(specialOpeningProb=0, completelyRandomOpeningProb=0)
    values.update(changes)
    return selfplay_config(path, 10 if game == 'xiangqi' else 5, **values)


def command(config, output, model='/dev/null', *, reanalyze=False, games=2, binary=None):
    return [str(binary or ROOT / 'build/katago'), 'selfplaysurprisedump', '-config', str(config),
            '-model', str(model), '-output', str(output), '-num-games', str(games),
            *(['-reanalyze-all'] if reanalyze else [])]


def read_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert not any('score' in field.lower() for field in rows[0])
    for row in rows:
        assert None not in row and None not in row.values()
        for key in row.keys() - {'gameHash', 'game'}:
            row[key] = float(row[key])
            assert np.isfinite(row[key]), (key, row)
    return rows


def run(config, output, model='/dev/null', **options):
    result = subprocess.run(command(config, output, model, **options), capture_output=True, text=True, timeout=90)
    output.with_suffix('.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return read_rows(output)


def grouped(rows, game, reanalyze, games=2):
    groups = defaultdict(list)
    for row in rows:
        assert row['game'] == game
        assert row['xSize'] == (9 if game == 'xiangqi' else 5)
        assert row['ySize'] == (10 if game == 'xiangqi' else 5)
        groups[row['gameIdx']].append(row)
        for prefix in ('outcome', 'mcts'):
            values = [row[f'{prefix}White{suffix}'] for suffix in ('Win', 'Draw', 'Loss', 'NoResult')]
            assert min(values) >= 0 and sum(values) == pytest.approx(1, abs=2e-6)
            if game != 'go':
                assert values[-1] == 0
        assert row['wasCheapSearch'] == 1 and row['wasReanalyzed'] == reanalyze
        assert row['unreducedNumVisits'] == (16 if reanalyze else 4)
        assert row['origNumVisits'] == (4 if reanalyze else 0)
        assert row['policySurprise'] >= 0 and 0 <= row['valueSurprise'] <= 1
        assert row['policyEntropy'] >= 0 and row['searchEntropy'] >= 0 and row['targetWeight'] >= 0
        if not reanalyze:
            assert row['origPolicySurprise'] == row['origValueSurprise'] == 0
    assert len(groups) == games
    for group in groups.values():
        assert len({r['gameHash'] for r in group}) == 1
        assert [r['turnAfterStart'] for r in group] == list(range(len(group)))
        assert all(r['gameNumMoves'] == len(group) for r in group)
    return list(groups.values())


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('reanalyze', [False, True])
def test_native_rows_and_reanalysis_overrides(tmp_path, game, reanalyze):
    config = configure(tmp_path / 'selfplay.cfg', game)
    output = tmp_path / 'stats.csv'
    rows = run(config, output, reanalyze=reanalyze)
    grouped(rows, game, reanalyze)
    if not reanalyze:
        assert 'forcing it to 0' in output.with_suffix('.log').read_text()


def constant_model(path, game, muzero):
    net, _ = make_model(path.with_name('initial'), game, muzero)
    head = net.prediction if muzero else net
    # Identical W/L removes the player-to-move dependence, yielding a known
    # white-oriented four-outcome reference at every initial/recurrent call.
    with torch.no_grad():
        head.value_head.linear_valuehead.weight.zero_()
        head.value_head.linear_valuehead.bias.copy_(torch.log(torch.tensor([.2, .5, .2, .1])))
    if muzero:
        export_components(net, path, 'surprise-constant-values')
    else:
        path = path.with_suffix('.bin')
        with path.open('wb') as stream:
            write_model_binary(net, stream, 'surprise-constant-values')
    return path


def check_kl(groups, game, search_values):
    raw = np.array([.2, .5, .2, .1] if game == 'go' else [2/9, 5/9, 2/9, 0.])
    for group in groups:
        mcts = np.array([[r[f'mctsWhite{s}'] for s in ('Win', 'Draw', 'Loss', 'NoResult')] for r in group])
        outcome = np.array([group[0][f'outcomeWhite{s}'] for s in ('Win', 'Draw', 'Loss', 'NoResult')])
        alpha = 1 / (1 + group[0]['xSize'] * group[0]['ySize'] * .016)
        for turn, row in enumerate(group):
            # Independent closed-form geometric mixture of future search values
            # and terminal/bootstrap, not the native backwards recurrence.
            if search_values:
                target = mcts[turn]
            else:
                weights = alpha * (1-alpha) ** np.arange(len(group)-turn)
                target = weights @ mcts[turn:] + (1-alpha) ** (len(group)-turn) * outcome
            positive = target > 0
            kl = np.sum(target[positive] * np.log(target[positive] / np.maximum(raw[positive], 1e-100)))
            assert row['valueSurprise'] == pytest.approx(np.clip(kl, 0, 1), abs=5e-5)
            assert row['rawNNWhiteWinLoss'] == pytest.approx(0, abs=2e-6)


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_real_models_four_outcome_kl_and_reanalysis(tmp_path, game, muzero, gumbel):
    check_four_outcome_kl_and_reanalysis(tmp_path, game, muzero, gumbel)


def check_four_outcome_kl_and_reanalysis(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True):
    torch.set_num_threads(1)
    model = constant_model(tmp_path / 'model', game, muzero)
    config = configure(tmp_path / 'selfplay.cfg', game, muzero, gumbel, False,
                        useSearchValueSurprise=str(gumbel).lower(), cudaUseFP16=str(fp16).lower())
    for reanalyze in (False, True):
        rows = run(config, tmp_path / f'stats-{reanalyze}.csv', model, reanalyze=reanalyze, binary=binary)
        groups = grouped(rows, game, reanalyze)
        check_kl(groups, game, gumbel)
        if game != 'go':
            assert all(r['hitTurnLimit'] == 1 for r in rows)
            np.testing.assert_allclose([rows[0][f'outcomeWhite{s}'] for s in ('Win', 'Draw', 'Loss', 'NoResult')],
                                       [2/9, 5/9, 2/9, 0], atol=1e-4)


@cuda
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (True, True)])
@pytest.mark.parametrize('search_values', [False, True])
def test_true_draw_and_imported_turn_offset(tmp_path, muzero, gumbel, search_values):
    model = constant_model(tmp_path / 'model', 'xiangqi', muzero)
    positions = tmp_path / 'positions'; positions.mkdir()
    fen = '3aka3/9/9/9/4P4/9/9/9/9/R3K4 w - - 119 60'
    (positions / 'start.sgf').write_text(f'(;GM[7]SZ[9:10]XQFEN[{fen}]XQTURN[73])')
    config = configure(tmp_path / 'selfplay.cfg', 'xiangqi', muzero, gumbel, False,
        startPosesFromSgfDir=positions, startPosesProb=1, startPosesLoadProb=1,
        startPosesTurnWeightLambda=0, useSearchValueSurprise=str(search_values).lower(), useUncertainty='false')
    rows = run(config, tmp_path / 'draw.csv', model, reanalyze=True)
    groups = grouped(rows, 'xiangqi', True)
    assert len(rows) == 2
    for row in rows:
        assert row['hitTurnLimit'] == 0 and row['turnIdx'] == 73
        assert row['outcomeWhiteDraw'] == 1
        assert row['outcomeWhiteWin'] == row['outcomeWhiteLoss'] == row['outcomeWhiteNoResult'] == 0
        # Ordinary search averages one NN root evaluation and 15 terminal draws.
        # MuZero sees constant latent predictions, never the real successor's
        # 120-ply rule adjudication. The environment outcome remains a true draw.
        expected_draw = 5/9 if muzero else (15 + 5/9) / 16
        assert row['mctsWhiteDraw'] == pytest.approx(expected_draw, abs=2e-6)
        if muzero and search_values:
            assert row['valueSurprise'] == pytest.approx(0, abs=2e-6)
        else:
            assert row['valueSurprise'] > .1
    check_kl(groups, 'xiangqi', search_values)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_signal_finishes_complete_csv_groups(tmp_path, muzero):
    model = constant_model(tmp_path / 'model', 'hex', muzero)
    config = configure(tmp_path / 'selfplay.cfg', 'hex', muzero, True, False)
    output = tmp_path / 'cancel.csv'
    with (tmp_path / 'native.log').open('w') as stream:
        process = subprocess.Popen(command(config, output, model, reanalyze=True, games=100000),
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: output.exists() and len(output.read_text().splitlines()) > 6, seconds=30)
            assert process.poll() is None
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=15) == 0
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    rows = read_rows(output)
    count = len({r['gameIdx'] for r in rows})
    assert 1 <= count < 100000
    grouped(rows, 'hex', True, games=count)


@cuda
def test_muzero_game_and_canvas_mismatch_before_output(tmp_path):
    model = constant_model(tmp_path / 'model', 'hex', True)
    for game, settings, message in [('renju', {}, 'game differs'), ('hex', {'bSizesXY': '7-7'}, 'canvas must contain')]:
        config = configure(tmp_path / f'{game}.cfg', game, True, False, False, **settings)
        output = tmp_path / f'{game}.csv'
        result = subprocess.run(command(config, output, model), capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and message in result.stdout + result.stderr
        assert not output.exists()
