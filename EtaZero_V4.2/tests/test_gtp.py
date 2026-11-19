"""Native GTP control flow and game boundaries, with source random evaluation."""
import json
import re
import subprocess

import pytest

from etazero import ROOT


def gtp_config(path, game='go', muzero=False, gumbel=False, random=True, **changes):
    values = dict(game=game, rules='tromp-taylor' if game == 'go' else game,
        numSearchThreads=2, maxVisits=32, defaultBoardSize=5,
        nnCacheSizePowerOfTwo=8, nnMutexPoolSizePowerOfTwo=4, nnRandomize='false',
        useMuZero=str(muzero).lower(), useGraphSearch=str(not (muzero or gumbel)).lower(),
        rootSymmetryPruning='false', subtreeValueBiasFactor=0 if muzero else .4,
        avoidRepeatedPatternUtility=0 if muzero else .03,
        rootSearchAlgorithm='gumbel' if gumbel else 'puct',
        nonrootSearchAlgorithm='gumbel' if gumbel else 'puct',
        analysisWideRootNoise=0, analysisIgnorePreRootHistory='false',
        logToStdout='false', logToStderr='true', logAllGTPCommunication='false', logSearchInfo='true',
        debugSkipNeuralNet=str(random).lower())
    values.update(changes)
    path.write_text(''.join(f'{k} = {v}\n' for k,v in values.items()))
    return path


def command(config, model='/dev/null', human=None):
    return [str(ROOT/'build/katago'), 'gtp', '-config', str(config), '-model', str(model),
            *(['-human-model', str(human)] if human else [])]


def run_gtp(config, commands, model='/dev/null', human=None):
    result = subprocess.run(command(config, model, human),
        input=''.join(f'{i} {c}\n' for i,c in enumerate(commands, 1)),
        capture_output=True, text=True, timeout=90)
    config.with_suffix('.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    replies = {}
    # The source benchmark uses carriage returns for its progress line; text
    # mode normalizes these to newlines, including just after the GTP header.
    headers = list(re.finditer(r'(?m)^([=?])(\d+)(?=[ \n]|$)', result.stdout))
    assert headers and not result.stdout[:headers[0].start()].strip(), result.stdout
    for i,match in enumerate(headers):
        end = headers[i+1].start() if i+1 < len(headers) else len(result.stdout)
        assert int(match[2]) not in replies, result.stdout
        replies[int(match[2])] = (match[1] == '?', result.stdout[match.end():end].strip())
    assert set(replies) == set(range(1, len(commands)+1)), result.stdout
    return [replies[i] for i in range(1, len(commands)+1)]


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex'])
def test_point_game_setup_raw_nn_and_undo(tmp_path, game):
    config = gtp_config(tmp_path/'gtp.cfg', game)
    replies = run_gtp(config, ['set_position W C3 B B3 B D3 B C2 B C4',
        'clear_board', 'play B A1', 'play W E5', 'printsgf', 'undo', 'printsgf',
        'kata-raw-nn 0', 'fixed_handicap 2', 'komi 1', 'kata-get-models', 'quit'])
    assert replies[0][0] == (game == 'go')  # Only Go forbids zero-liberty initial stones.
    assert all(not replies[i][0] for i in (1,2,3,4,5,6,7,10,11)), replies
    assert 'W[ea]' in replies[4][1] and 'W[ea]' not in replies[6][1]
    assert 'whiteWin ' in replies[7][1] and 'draw ' in replies[7][1]
    if game != 'go':
        assert replies[8][0] and replies[9][0]
    assert len(json.loads(replies[10][1])) == 1


def test_gumbel_runtime_rejection_does_not_poison_next_config(tmp_path):
    config = gtp_config(tmp_path/'gtp.cfg', gumbel=True)
    replies = run_gtp(config, ['kata-set-param useGraphSearch true', 'kata-set-param maxVisits 12',
        'kata-search B', 'kata-search_analyze B 0 focus A1', 'kata-search B', 'quit'])
    assert [r[0] for r in replies] == [True, False, False, True, False, False]


def test_rejected_placement_rectangle_keeps_the_previous_position(tmp_path):
    config = gtp_config(tmp_path/'gtp.cfg', 'renju')
    r = run_gtp(config, ['play B C3', 'boardsize 4 5', 'printsgf', 'kata-raw-nn 0', 'quit'])
    assert [err for err,_ in r] == [False, True, False, False, False]
    assert 'SZ[5]' in r[2][1] and 'B[cc]' in r[2][1]


def test_source_benchmark_preserves_engine_position(tmp_path):
    config = gtp_config(tmp_path/'gtp.cfg', defaultBoardSize=7, maxVisits=4)
    r = run_gtp(config, ['play B C3', 'kata-benchmark 4', 'printsgf', 'kata-search W', 'quit'])
    assert all(not err for err,_ in r), r
    assert '10 / 10 positions' in r[1][1] and 'nnEvals/s' in r[1][1]
    assert ';B[ce]' in r[2][1]


@pytest.mark.parametrize('change,error', [
    (dict(useGraphSearch='true', rootSearchAlgorithm='gumbel', nonrootSearchAlgorithm='gumbel'), 'Gumbel GTP'),
    (dict(game='xiangqi', rules='xiangqi'), 'Xiangqi requires a 9 x 10 board'),
    (dict(game='hex', rules='tromp-taylor'), 'game must match'),
    (dict(muzero=True, subtreeValueBiasFactor=.4), 'MuZero latent search'),
])
def test_invalid_startup_is_rejected_before_search(tmp_path, change, error):
    config = gtp_config(tmp_path/'bad.cfg', **change)
    result = subprocess.run(command(config), input='', capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and error in result.stdout + result.stderr
