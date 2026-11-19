"""Source position viewers/searchers with full game histories and model services."""
import json
import os
import re
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from test_evalsgf import GAMES, config_for, make_model
from test_random_initializations import constant_model
from test_xiangqi_gtp import CHECK_FEN, CHECKS


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA search')
COMMANDS = ['viewstartposes', 'trystartposes']
START_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'


def position(game, **changes):
    sample = dict(game=game, xSize=9 if game == 'xiangqi' else 5, ySize=10 if game == 'xiangqi' else 5,
        nextPla='B', moveLocs=['a3a4', 'a6a5'] if game == 'xiangqi' else ['A1', 'E5'],
        movePlas=['B', 'W'], initialTurnNumber=73, hintLoc='null', weight=.5, trainingWeight=.375)
    if game == 'xiangqi':
        sample.update(xiangqiFen=START_FEN, xiangqiHistory=[])
    else:
        sample['board'] = '...../' * 5
    sample.update(changes)
    return sample


def write_positions(path, samples):
    path.write_text(''.join(json.dumps(sample) + '\n' for sample in samples))
    return path


def configure(path, game, **options):
    values = dict(maxVisits=8, numSearchThreads=1, rootPolicyOptimism=0, policyOptimism=0,
                  rootNoiseEnabled='false', fpuReductionMax=0, rootFpuReductionMax=0,
                  useUncertainty='false', logToStdout='false', logToStderr='true')
    values.update(options)
    return config_for(path, game, **values)


def run(name, files, *, config=None, model='/dev/null', extra=(), binary=None):
    if not isinstance(files, (list, tuple)):
        files = [files]
    result = subprocess.run([str(binary or ROOT / 'build/katago'), name,
        *(['-config', str(config), '-model', str(model)] if config else []),
        *(arg for file in files for arg in ('-start-poses-file' if name == 'viewstartposes' else '-startposes', str(file))),
        *map(str, extra)], capture_output=True, text=True, timeout=90)
    index = len(list(files[0].parent.glob(f'{name}-*.log')))
    (files[0].parent / f'{name}-{index}.log').write_text(result.stdout + result.stderr)
    return result


def values(result, game, count=None):
    assert result.returncode == 0, result.stdout + result.stderr
    rows = np.array([list(map(float, line.split())) for line in re.findall(r'^White W/D/L/NR: (.*)$', result.stdout, re.M)])
    if count is not None:
        assert rows.shape == (count, 4), result.stdout
    assert rows.ndim == 2 and rows.shape[1] == 4
    assert np.isfinite(rows).all() and (rows >= 0).all() and (rows <= 1).all()
    np.testing.assert_allclose(rows.sum(axis=1), 1, atol=2e-6)
    if game != 'go':
        assert (rows[:, 3] == 0).all()
    return rows


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('gumbel', [False, True])
def test_native_search_and_game_inference(tmp_path, game, gumbel):
    config = configure(tmp_path / 'search.cfg', game, gumbel=gumbel)
    config.write_text(re.sub(r'^game = .*\n', '', config.read_text(), flags=re.M))
    files = [write_positions(tmp_path / f'{i}.txt', [position(game)]) for i in range(2)]
    for name in COMMANDS:
        result = run(name, files, config=config)
        values(result, game, 2)
        assert len(re.findall(r'^: .*N\s+8\s+--', result.stdout, re.M)) == 2
        assert 'TrainingWeight: 0.375' in result.stdout or 'Training Weight: 0.375' in result.stdout
        if game == 'xiangqi':
            assert result.stdout.count('Xiangqi FEN:') == 2
            assert re.search(r'[a-i][0-9][a-i][0-9]', result.stdout)


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_real_model_four_outcomes(tmp_path, game, muzero, gumbel):
    check_real_model(tmp_path, game, muzero, gumbel)


def check_real_model(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True):
    model = constant_model(tmp_path / 'model', game, muzero)
    config = configure(tmp_path / 'search.cfg', game, muzero=muzero, gumbel=gumbel,
                       random=False, cudaUseFP16=str(fp16).lower())
    path = write_positions(tmp_path / 'positions.txt', [position(game)])
    for name in COMMANDS:
        result = run(name, path, config=config, model=model, binary=binary)
        actual, = values(result, game, 1)
        # Constant symmetric W/L and uniform policy keep this shallow search
        # away from terminals. Only Go random SIMPLE rules can retain NR.
        expected = [.2, .5, .2, .1] if actual[3] > .05 else [2/9, 5/9, 2/9, 0]
        np.testing.assert_allclose(actual, expected, atol=1e-4)
        assert re.search(r'^: .*N\s+8\s+--', result.stdout, re.M)


def terminal_sample(game):
    if game == 'xiangqi':
        return position(game, xiangqiFen=CHECK_FEN, xiangqiHistory=CHECKS[:2],
                        moveLocs=CHECKS[2:], movePlas=['B', 'W'] * 3, initialTurnNumber=97), [1, 0, 0, 0]
    # Native rule setup, without a fabricated search target.
    board = 'X..../' * 5 if game == 'hex' else '...../' * 4 + 'XXXXX/'
    return position(game, board=board, nextPla='W', moveLocs=[], movePlas=[]), [0, 0, 1, 0]


@pytest.mark.parametrize('game', GAMES[1:])
def test_actual_terminal_rule_results(tmp_path, game):
    sample, expected = terminal_sample(game)
    path = write_positions(tmp_path / 'terminal.txt', [sample])
    config = configure(tmp_path / 'search.cfg', game, gumbel=True)
    for name in COMMANDS:
        np.testing.assert_array_equal(values(run(name, path, config=config), game, 1), [expected])


@cuda
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_xiangqi_full_history_and_hint_transition(tmp_path, muzero, gumbel):
    model = constant_model(tmp_path / 'model', 'xiangqi', muzero)
    config = configure(tmp_path / 'search.cfg', 'xiangqi', muzero=muzero, gumbel=gumbel, random=False)
    sample, expected = terminal_sample('xiangqi')
    path = write_positions(tmp_path / 'terminal.txt', [sample])
    for name in COMMANDS:
        np.testing.assert_array_equal(values(run(name, path, config=config, model=model), 'xiangqi', 1), [expected])

    # Every legal move advances rule60 from 119 to 120. A hint is played in the
    # real environment even when subsequent search uses a latent model.
    fen = '3aka3/9/9/9/4P4/9/9/9/9/R3K4 w - - 119 60'
    samples = [position('xiangqi', xiangqiFen=fen, moveLocs=[], movePlas=[], hintLoc=hint)
               for hint in ['a0a1', 'a0a2', 'a0a3']]
    path = write_positions(tmp_path / 'hints.txt', samples)
    result = run('trystartposes', path, config=config, model=model)
    groups = result.stdout.split('Searching startpos:')[1:]
    assert result.returncode == 0 and len(groups) == 3, result.stdout + result.stderr
    for group, sample in zip(groups, samples):
        assert f'There was a hintpos {sample["hintLoc"]}' in group
        rows = [list(map(float, row.split())) for row in re.findall(r'^White W/D/L/NR: (.*)$', group, re.M)]
        if 're-searching after playing it' in group:
            assert len(rows) == 2
            np.testing.assert_array_equal(rows[1], [0, 1, 0, 0])
            assert re.search(r'Xiangqi FEN: .* b - - 120 60', group)
        else:
            assert 'it was the chosen move' in group and len(rows) == 1
        assert rows[0][1] == pytest.approx(5/9 if muzero else (7 + 5/9)/8, abs=2e-6)


def test_source_weight_sampling_filters_and_raw_viewer(tmp_path):
    samples = [position('xiangqi', weight=w) for w in [0, 0, 2]]
    path = write_positions(tmp_path / 'weighted.txt', samples)
    config = configure(tmp_path / 'search.cfg', 'xiangqi')
    result = run('trystartposes', path, config=config, extra=['-random-sample'])
    values(result, 'xiangqi', 3)
    assert re.findall(r'^Weight: (.*)$', result.stdout, re.M) == ['2'] * 3
    result = run('trystartposes', path, config=config, extra=['-min-weight', '1'])
    values(result, 'xiangqi', 1)
    result = run('viewstartposes', path, config=config, extra=['-idx', '1'])
    values(result, 'xiangqi', 1)
    assert 'StartPos: 1/3' in result.stdout and 'Weight: 0' in result.stdout
    result = run('viewstartposes', path, config=config, extra=['-min-weight', '1'])
    values(result, 'xiangqi', 1)
    assert 'StartPos: 2/3' in result.stdout
    # Viewing or checking legality without a model may mix game families.
    path = write_positions(tmp_path / 'mixed.txt', [position(game) for game in GAMES])
    result = run('viewstartposes', path)
    assert result.returncode == 0 and result.stdout.count('StartPos:') == 6
    assert 'White W/D/L/NR' not in result.stdout
    checked = run('viewstartposes', path, extra=['-check-legality'])
    assert checked.returncode == 0 and 'Illegal' not in checked.stdout
    # An unselected row must not constrain game inference for model setup.
    result = run('viewstartposes', path, config=config, extra=['-idx', '5'])
    values(result, 'xiangqi', 1)


@pytest.mark.parametrize('game', ['hex', 'xiangqi'])
def test_illegal_hints_and_replay_keep_source_handling(tmp_path, game):
    hint = 'a3a4' if game == 'xiangqi' else 'A1'  # already played
    path = write_positions(tmp_path / 'hints.txt', [position(game, hintLoc=hint)])
    config = configure(tmp_path / 'search.cfg', game)
    checked = run('viewstartposes', path, config=config, extra=['-check-legality'])
    assert checked.returncode == 0 and 'Illegal hint in startpos' in checked.stdout
    assert 'White W/D/L/NR' not in checked.stdout
    tried = run('trystartposes', path, config=config)
    values(tried, game, 1)
    assert 'Hint move was illegal, skipping' in tried.stdout
    sample = position(game); sample['moveLocs'].append(hint); sample['movePlas'].append('B')
    path = write_positions(tmp_path / 'bad-move.txt', [sample])
    checked = run('viewstartposes', path, extra=['-check-legality'])
    assert checked.returncode == 0 and 'Illegal move in startpos' in checked.stdout
    for name in COMMANDS:
        result = run(name, path, config=config)
        assert result.returncode != 0 and 'Illegal move in startpos' in result.stdout + result.stderr


@cuda
def test_model_game_and_canvas_boundaries(tmp_path):
    model = constant_model(tmp_path / 'model', 'hex', True)
    for name in COMMANDS:
        config = configure(tmp_path / f'{name}.cfg', 'hex', muzero=True, random=False)
        mismatch = write_positions(tmp_path / 'other-game.txt', [position('renju')])
        result = run(name, mismatch, config=config, model=model)
        assert result.returncode != 0 and 'game must match' in result.stdout + result.stderr
        config.write_text(re.sub(r'^game = .*\n', '', config.read_text(), flags=re.M))
        result = run(name, mismatch, config=config, model=model)
        assert result.returncode != 0 and 'model game differs' in result.stdout + result.stderr
        oversized = position('hex', xSize=7, ySize=7, board='......./' * 7)
        path = write_positions(tmp_path / 'canvas.txt', [oversized, position('hex')])
        result = run(name, path, config=config, model=model)
        assert result.returncode != 0 and 'canvas must contain' in result.stdout + result.stderr
        if name == 'viewstartposes':
            values(run(name, path, config=config, model=model, extra=['-idx', '1']), 'hex', 1)

    _, go_model = make_model(tmp_path / 'go', 'go', False)
    config = configure(tmp_path / 'ordinary.cfg', 'hex', random=False)
    path = write_positions(tmp_path / 'hex.txt', [position('hex')])
    for name in COMMANDS:
        result = run(name, path, config=config, model=go_model)
        assert result.returncode != 0 and 'Model does not support' in result.stdout + result.stderr


@pytest.mark.parametrize('name', COMMANDS)
def test_parse_errors_and_empty_selection(tmp_path, name):
    config = configure(tmp_path / 'search.cfg', 'hex')
    path = write_positions(tmp_path / 'positions.txt', [dict(missing='fields'), position('hex')])
    result = run(name, path, config=config)
    values(result, 'hex', 1)
    assert 'ERROR parsing startpos' in result.stdout + result.stderr
    result = run(name, path, config=config, extra=['-min-weight', '1'])
    assert result.returncode == 0 and 'White W/D/L/NR' not in result.stdout


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_primary_metadata_model_uses_search_profile(tmp_path, muzero):
    from test_evalsgf import run as evalsgf
    net, model = make_model(tmp_path / 'metadata', 'go', muzero, metadata=True)
    config = configure(tmp_path / 'search.cfg', 'go', muzero=muzero, random=False,
                       maxVisits=1, rules='Japanese', humanSLProfile='rank_5d')
    path = write_positions(tmp_path / 'empty.txt', [position('go', moveLocs=[], movePlas=[], initialTurnNumber=0)])
    sgf = tmp_path / 'empty.sgf'; sgf.write_text('(;GM[1]SZ[5]RU[Japanese]KM[6.5])')
    inputs = tmp_path / 'input.npz'
    reference = evalsgf(config, sgf, '-dump-npz-input-to', inputs, model=model, turn=0)
    assert reference.returncode == 0, reference.stdout + reference.stderr
    with np.load(inputs) as data:
        spatial, globals_, metadata = [torch.from_numpy(data[key]) for key in ('binaryInputNCHW', 'globalInputNC', 'metadataInputNC')]
    assert metadata.any()
    head = net.prediction if muzero else net
    with torch.no_grad():
        output = net.initial(spatial, globals_, metadata)[1] if muzero else net(spatial, globals_, metadata)
        expected = head.postprocess_output(output)[0][1][0].softmax(-1).numpy()[[2, 1, 0, 3]]
    # viewstartposes uses the configured Go rules. trystartposes deliberately
    # randomizes them, so its input tensor is not this Japanese-rules oracle.
    actual, = values(run('viewstartposes', path, config=config, model=model), 'go', 1)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=3e-3)
