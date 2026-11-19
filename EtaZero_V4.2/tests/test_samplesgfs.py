"""Source value-fluctuation sampling with actual game observations and models."""
import json
import math
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from export_muzero import export_components
from test_datamine import XQ_MOVES, configure as selfplay_config
from test_evalsgf import GAMES, make_model, record
from test_xiangqi_controller import native_selfplay


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA sampling')


def configure(path, game, muzero=False, *, random=False, **changes):
    # Omit game: the first eligible SGF supplies it, as for other SGF diagnostics.
    options = dict(useMuZero=str(muzero).lower(), debugSkipNeuralNet=str(random).lower(),
        maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5, requireMaxBoardSize='false',
        nnRandomize='false', nnMaxBatchSize=4, nnCacheSizePowerOfTwo=8, nnMutexPoolSizePowerOfTwo=4,
        cudaUseFP16='true')
    options.update(changes)
    path.write_text(''.join(f'{key} = {value}\n' for key, value in options.items()))
    return path


def model_with_values(path, game, muzero, *, zero=False, metadata=False):
    net, _ = make_model(path.with_name('initial'), game, muzero, metadata=metadata)
    head = net.prediction if muzero else net
    with torch.no_grad():
        head.value_head.linear_valuehead.weight.zero_()
        head.value_head.linear_valuehead.bias.copy_(torch.log(torch.tensor([1 if zero else 3, 1, 1, 1.])))
    if muzero:
        export_components(net, path, 'value-fluctuation')
    else:
        path = path.with_suffix('.bin')
        with path.open('wb') as stream:
            write_model_binary(net, stream, 'value-fluctuation')
    return path


def command(files, output, *, config=None, model=None, threads=2, turn_lambda=0, extra=(), binary=None):
    if not isinstance(files, (list, tuple)):
        files = [files]
    return [str(binary or ROOT / 'build/katago'), 'samplesgfs',
        *(arg for file in files for arg in ('-sgf', str(file))), '-outdir', str(output),
        '-sample-prob', '1', '-turn-weight-lambda', str(turn_lambda), '-num-threads', str(threads), '-for-testing',
        *(['-config', str(config)] if config else []),
        *(['-value-fluctuation-model', str(model)] if model else []), *map(str, extra)]


def run(files, output, **options):
    result = subprocess.run(command(files, output, **options), capture_output=True, text=True, timeout=90)
    output.with_suffix('.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = []
    for path in sorted(output.glob('*.startposes.txt')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
        viewed = subprocess.run([str(options.get('binary') or ROOT / 'build/katago'), 'viewstartposes',
            '-start-poses-file', str(path), '-check-legality'], capture_output=True, text=True, timeout=30)
        assert viewed.returncode == 0, viewed.stdout + viewed.stderr
        assert not any(word in viewed.stdout for word in ['Illegal', 'Mismatching', 'ERROR'])
    return rows, result.stdout


def totals(text):
    return np.array([float(re.search(rf'^: totalWeightFrom{label} (\S+)$', text, re.M)[1])
                     for label in ['Count', 'Surprise', 'Uncertainty']])


@pytest.mark.parametrize('game', GAMES)
def test_model_free_and_random_evaluator(tmp_path, game):
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game))
    plain, text = run(sgf, tmp_path / 'plain')
    assert len(plain) == 3 and all(row['weight'] == 1 for row in plain)
    assert 'Loaded neural net' not in text
    cfg = configure(tmp_path / 'sampling.cfg', game, random=True)
    # Blur keeps either random adjacent fluctuation from falling below the
    # source 0.1 cutoff; the exact unsmoothed weights need not both survive.
    sampled, text = run(sgf, tmp_path / 'sampled', config=cfg, model='/dev/null', extra=[
        '-value-fluctuation-turn-scale', 2, '-value-fluctuation-forward-turn-scale', 2])
    assert len(sampled) == 2
    assert all(row.get('game', 'go') == game for row in sampled)
    assert sum(row['weight'] for row in sampled) == pytest.approx(2)
    assert text.count('Loaded neural net') == 1


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_exact_two_sided_blur_and_four_results(tmp_path, game, muzero, gumbel):
    check_exact_weights(tmp_path, game, muzero, gumbel)


def check_exact_weights(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True, go_rules='Tromp-Taylor'):
    model = model_with_values(tmp_path / 'model', game, muzero)
    cfg = configure(tmp_path / 'sampling.cfg', game, muzero, cudaUseFP16=str(fp16).lower(),
                    rootSearchAlgorithm='gumbel' if gumbel else 'puct', maxVisits=100000000)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game).replace('Tromp-Taylor', go_rules))
    rows, text = run(sgf, tmp_path / 'sampled', config=cfg, model=model, binary=binary, extra=[
        '-value-fluctuation-turn-scale', 2, '-value-fluctuation-forward-turn-scale', 2,
        '-value-fluctuation-weight-by-count', 2, '-value-fluctuation-weight-by-surprise', 3,
        '-value-fluctuation-weight-by-uncertainty', 4, '-training-weight', .625])
    assert len(rows) == 2
    rows.sort(key=lambda row: row['initialTurnNumber'] + len(row['moveLocs']))
    # W/D/L/NR logits log(3,1,1,1), with NR suppressed by these rules:
    # white W-L is -0.4,+0.4,-0.4. Both adjacent variances are 0.64.
    # Backward scale 2 gives [.48,.32]; forward scale 2 gives [.24,.28].
    # Hence normalized weights are exactly 6:7, independently of model precision.
    value = 1/3 if game == 'go' and go_rules == 'Japanese' else .4
    p = .001 + .998 * (.5 - .5 * value)
    expected_totals = np.array([4, -3 * math.log(p * (1-p)), 4 * 2 * (1-value*value)])
    np.testing.assert_allclose(totals(text), expected_totals, atol=8e-4)
    np.testing.assert_allclose([row['weight'] for row in rows], expected_totals.sum() * np.array([6, 7]) / 13, atol=8e-4)
    assert all(row['trainingWeight'] == .625 and row['hintLoc'] == 'null' for row in rows)
    if not muzero:
        # Source Eigen can have several inference server threads. Sum their
        # shutdown counters instead of requiring all calls on the same server.
        counts = [list(map(int, pair)) for pair in re.findall(r'finishing, processed (\d+) rows (\d+) batches', text)]
        assert np.array(counts).sum(axis=0).tolist() == [3, 3]  # skipCache, no search


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_go_rules_retain_no_result_probability(tmp_path, muzero):
    # SIMPLE ko + TERRITORY retains NR, so W-L is 2/6 instead of 2/5.
    check_exact_weights(tmp_path, 'go', muzero, False, go_rules='Japanese')


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_cap_turn_scaling_and_side_variations(tmp_path, muzero):
    model = model_with_values(tmp_path / 'model', 'renju', muzero)
    cfg = configure(tmp_path / 'sampling.cfg', 'renju', muzero)
    sgf = tmp_path / 'branch.sgf'
    sgf.write_text('(;GM[4]SZ[5]RU[renju];B[aa](;W[ee])(;W[dd]C[%SAMPLELIGHT%]))')
    rows, _ = run(sgf, tmp_path / 'sampled', config=cfg, model=model, turn_lambda=math.log(2),
        extra=['-value-fluctuation-max-weight', .75, '-sample-weight', 2, '-training-weight', .25,
               '-override-config', 'nnRandomize=true', '-force-sample-weight', 8])
    by_turn = {row['initialTurnNumber'] + len(row['moveLocs']): row for row in rows}
    assert set(by_turn) == {0, 1, 2}
    assert by_turn[0]['weight'] == 1.5 and by_turn[1]['weight'] == pytest.approx(.75)
    assert by_turn[2]['weight'] == 4 and by_turn[2]['moveLocs'][-1] == 'D2'
    # The main-line fluctuation pass intentionally receives empty comments;
    # forced comment weights apply in the source side-variation traversal.
    sgf.write_text(sgf.read_text().replace(';B[aa]', ';B[aa]C[%SAMPLE%]'))
    rows2, _ = run(sgf, tmp_path / 'comments', config=cfg, model=model, turn_lambda=math.log(2),
        extra=['-value-fluctuation-max-weight', .75, '-sample-weight', 2, '-force-sample-weight', 8])
    assert sorted(row['weight'] for row in rows2) == pytest.approx([.75, 1.5, 4])


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_xiangqi_full_history_and_original_turns(tmp_path, muzero):
    model = model_with_values(tmp_path / 'model', 'xiangqi', muzero)
    cfg = configure(tmp_path / 'sampling.cfg', 'xiangqi', muzero)
    sgf = tmp_path / 'history.sgf'
    text = '(;GM[7]SZ[9:10]RU[xiangqi]XQHIST[a3a4][a6a5]XQTURN[97]'
    text += ''.join(f';{"B" if i % 2 == 0 else "W"}[{move}]' for i, move in enumerate(XQ_MOVES[2:])) + ')'
    sgf.write_text(text)
    rows, text = run(sgf, tmp_path / 'sampled', config=cfg, model=model,
                     extra=['-value-fluctuation-max-weight', 100, '-flip-if-pass'])
    assert len(rows) == 10
    rows.sort(key=lambda row: row['initialTurnNumber'] + len(row['moveLocs']))
    # Source normalization counts the full turn-indexed vector, including its
    # zero prefix; do not silently redefine it as the number of SGF moves.
    assert totals(text)[0] == 107
    for i, row in enumerate(rows):
        assert row['xiangqiHistory'] + row['moveLocs'] == XQ_MOVES[:i+2]
        assert row['initialTurnNumber'] + len(row['moveLocs']) == 97 + i
        assert row['weight'] == pytest.approx(10.7, abs=1e-5)
    # Consume the last snapshot, whose eight visible moves follow three plies
    # of owned rule history, through the native selfplay training writer.
    consumed = tmp_path / 'consumed'; consumed.mkdir()
    (consumed / 'chosen.startposes.txt').write_text(json.dumps(rows[-1]) + '\n')
    models = tmp_path / 'models'; models.mkdir()
    if muzero:
        shutil.copytree(model, models / 'published')
    else:
        published = models / 'published'; published.mkdir()
        shutil.copyfile(model, published / 'model.bin')
    cfg = selfplay_config(tmp_path / 'consume.cfg', 'xiangqi', muzero, True, False,
        hintPosesDir=consumed, hintPosesProb=1, maxMovesPerGame=1, cheapSearchProb=0,
        forkSidePositionProb=0, earlyForkGameProb=0)
    result = native_selfplay(tmp_path, cfg)
    assert result.returncode == 0, result.stdout + result.stderr
    shards = list((tmp_path / 'output').glob('*/tdata/*.npz'))
    assert shards
    for path in shards:
        with np.load(path) as data:
            targets = data['globalTargetsNC']
            assert np.all(targets[:, 51] + targets[:, 56] == 106)
            assert np.all(targets[:, [2, 6, 10, 14, 18]] == 0)
            assert data['policyTargetsNCMove'].shape[-1] == 2086


@cuda
@pytest.mark.parametrize('game', ['freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('muzero', [False, True])
def test_actual_terminal_endpoint(tmp_path, game, muzero):
    model = model_with_values(tmp_path / 'model', game, muzero, zero=game != 'xiangqi')
    cfg = configure(tmp_path / 'sampling.cfg', game, muzero)
    sgf = tmp_path / 'terminal.sgf'
    if game == 'xiangqi':
        fen = '3aka3/9/9/9/4P4/9/9/9/9/R3K4 w - - 119 60'
        sgf.write_text(f'(;GM[7]SZ[9:10]XQFEN[{fen}]XQTURN[0];B[a0a1])')
        expected_count = 1
    else:
        gm = 11 if game == 'hex' else 4
        # Exactly one genuine winning move. The endpoint must be its rule
        # result, although the constant network has W-L=0 everywhere.
        sgf.write_text(f'(;GM[{gm}]SZ[5]RU[{game}]AB[aa][ab][ac][ad]AW[ea][ec]PL[B];B[ae])')
        expected_count = 7
    rows, text = run(sgf, tmp_path / 'sampled', config=cfg, model=model,
                     extra=['-value-fluctuation-max-weight', 100, '-allow-game-over'])
    assert len(rows) == 1 and rows[0]['weight'] == pytest.approx(expected_count)
    assert totals(text)[0] == expected_count
    if not muzero:
        assert 'processed 1 rows 1 batches' in text


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_zero_fluctuations_are_not_uniform_samples(tmp_path, muzero):
    model = model_with_values(tmp_path / 'model', 'hex', muzero, zero=True)
    cfg = configure(tmp_path / 'sampling.cfg', 'hex', muzero)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('hex'))
    rows, text = run(sgf, tmp_path / 'sampled', config=cfg, model=model)
    assert rows == [] and np.all(totals(text) == 0)


def test_parallel_multisgf_tolerates_bad_record_and_preserves_mainline_weights(tmp_path):
    cfg = configure(tmp_path / 'sampling.cfg', 'renju', random=True)
    directory = tmp_path / 'sgfs'; directory.mkdir()
    (directory / 'games.sgfs').write_text('\n'.join([record('renju'),
        '(;GM[4]SZ[5]RU[renju];B[zz])', record('renju')]))
    rows, text = run([], tmp_path / 'sampled', config=cfg, model='/dev/null', threads=3,
                     extra=['-sgfsdir', directory, '-tolerate-illegal-moves',
                            '-value-fluctuation-turn-scale', 2, '-value-fluctuation-forward-turn-scale', 2])
    assert len(rows) == 4 and sum(row['weight'] for row in rows) == pytest.approx(4)
    assert 'Bad sgf in SGFS' in text and 'Move out of bounds' in text
    assert text.count('Loaded neural net') == 1


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_metadata_and_fatal_model_errors_with_tolerant_input(tmp_path, muzero):
    model = model_with_values(tmp_path / 'model', 'go', muzero, metadata=True)
    cfg = configure(tmp_path / 'sampling.cfg', 'go', muzero, humanSLProfile='rank_5d')
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('go'))
    rows, _ = run(sgf, tmp_path / 'sampled', config=cfg, model=model)
    assert [row['weight'] for row in rows] == pytest.approx([1, 1], abs=1e-5)
    cfg.write_text(re.sub(r'^humanSLProfile = .*\n', '', cfg.read_text(), flags=re.M))
    result = subprocess.run(command([sgf, sgf], tmp_path / 'invalid', config=cfg, model=model,
        extra=['-tolerate-illegal-moves']), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'requires a humanSLProfile' in result.stderr
    assert not list((tmp_path / 'invalid').glob('*.startposes.txt'))


def test_filter_before_model_loading_and_game_mismatch(tmp_path):
    cfg = configure(tmp_path / 'sampling.cfg', 'renju', random=True)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('renju'))
    rows, text = run(sgf, tmp_path / 'filtered', config=cfg, model='nonexistent-model', extra=['-max-depth', 1])
    assert rows == [] and 'Loaded neural net' not in text
    cfg.write_text(cfg.read_text() + 'game = hex\n')
    rows, text = run(sgf, tmp_path / 'mismatch', config=cfg, model='/dev/null', extra=['-tolerate-illegal-moves'])
    assert rows == [] and 'SGF game differs' in text


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_parallel_nonuniform_model_matches_serial(tmp_path, muzero):
    _, model = make_model(tmp_path / 'model', 'xiangqi', muzero)
    cfg = configure(tmp_path / 'sampling.cfg', 'xiangqi', muzero)
    directory = tmp_path / 'sgfs'; directory.mkdir()
    (directory / 'games.sgfs').write_text('\n'.join(
        f'(;GM[7]SZ[9:10]RU[xiangqi];B[{move}];W[a6a5])'
        for move in ['a3a4', 'c3c4', 'e3e4', 'g3g4', 'i3i4', 'b0c2']))
    options = dict(config=cfg, model=model, extra=['-sgfsdir', directory,
        '-value-fluctuation-turn-scale', 2, '-value-fluctuation-forward-turn-scale', 2])
    serial, _ = run([], tmp_path / 'serial', threads=1, **options)
    parallel, text = run([], tmp_path / 'parallel', threads=4, **options)
    assert len(serial) == len(parallel) == 12 and text.count('Loaded neural net') == 1
    def ordered(rows):
        return sorted((json.dumps({key: value for key, value in row.items() if key != 'weight'}, sort_keys=True),
                       row['weight']) for row in rows)
    serial, parallel = ordered(serial), ordered(parallel)
    assert [key for key, _ in serial] == [key for key, _ in parallel]
    np.testing.assert_allclose([weight for _, weight in serial], [weight for _, weight in parallel], atol=5e-4)


@cuda
@pytest.mark.parametrize('game,size,message', [('renju', 5, 'model game differs'),
                                             ('hex', 7, 'canvas must contain')])
def test_bundle_boundaries_are_model_errors(tmp_path, game, size, message):
    model = model_with_values(tmp_path / 'model', 'hex', True)
    cfg = configure(tmp_path / 'sampling.cfg', game, True)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game).replace('SZ[5]', f'SZ[{size}]'))
    result = subprocess.run(command([sgf, sgf], tmp_path / 'invalid', config=cfg, model=model,
        extra=['-tolerate-illegal-moves']), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and message in result.stderr
    assert not list((tmp_path / 'invalid').glob('*.startposes.txt'))
