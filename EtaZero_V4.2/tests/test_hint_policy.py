"""Native SGF hint likelihood: source traversal, raw policy and game actions."""
import math
import os
import re
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from test_evalsgf import GAMES, config_for, make_model
from test_gtp import run_gtp
from test_search_entropy import uniform_model
from test_xiangqi_gtp import raw_policy


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA hint inference')


def configure(path, game, **options):
    values = dict(logToStdout='false', logToStderr='true', maxVisits=100000,
                  rootPolicyOptimism=0, policyOptimism=0)
    values.update(options)
    return config_for(path, game, **values)


def record(game, *, variations=False):
    if game == 'xiangqi':
        header, first, second, counts = 'GM[7]SZ[9:10]RU[xiangqi]', 'a3a4', 'a6a5', [44, 44]
    else:
        gm = 1 if game == 'go' else 11 if game == 'hex' else 4
        header, first, second = f'GM[{gm}]SZ[5]RU[{"Tromp-Taylor" if game == "go" else game}]', 'aa', 'ee'
        counts = [26, 25] if game == 'go' else [25, 24]
    if variations:
        # Same child board, parent and comment is deduplicated. A distinct
        # comment remains a distinct hint under the source hashComments=true.
        return f'(;{header}(;B[{first}]C[%HINT%])(;B[{first}]C[%HINT%])(;B[{first}]C[%HINT% another]))', [counts[0]] * 2
    return f'(;{header};B[{first}]C[%HINT%];W[{second}]C[%HINT%])', counts


def write_sgf(directory, text, name='game.sgf'):
    directory.mkdir(exist_ok=True)
    path = directory / name; path.write_text(text)
    return path


def run(config, directories, model='/dev/null', *, binary=None):
    if not isinstance(directories, (list, tuple)):
        directories = [directories]
    result = subprocess.run([str(binary or ROOT / 'build/katago'), 'checksgfhintpolicy',
        '-config', str(config), '-model', str(model),
        *(arg for directory in directories for arg in ('-sgfdir', str(directory)))],
        capture_output=True, text=True, timeout=90)
    index = len(list(config.parent.glob('hint-policy-*.log')))
    (config.parent / f'hint-policy-{index}.log').write_text(result.stdout + result.stderr)
    return result


def outputs(result, count):
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'Total number of hint positions: {count}\n' in result.stdout
    match = re.search(r'^Average log policy across all hints: (\S+)$', result.stdout, re.M)
    assert match, result.stdout
    value = float(match[1])
    if count:
        assert math.isfinite(value) and value <= 0
    else:
        # Preserve source zero-weight behavior, never turn absent hints into
        # a valid zero log-loss measurement.
        assert math.isnan(value)
    return value


@pytest.mark.parametrize('game', GAMES)
def test_native_hint_traversal_and_inferred_game(tmp_path, game):
    directory = tmp_path / 'sgfs'
    write_sgf(directory, record(game)[0])
    config = configure(tmp_path / 'hint.cfg', game)
    config.write_text(re.sub(r'^game = .*\n', '', config.read_text(), flags=re.M))
    outputs(run(config, directory), 2)


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_exact_uniform_hint_likelihood(tmp_path, game, muzero, gumbel):
    check_uniform_hint_likelihood(tmp_path, game, muzero, gumbel)


def check_uniform_hint_likelihood(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True):
    model = uniform_model(tmp_path / 'model', game, muzero)
    config = configure(tmp_path / 'hint.cfg', game, muzero=muzero, gumbel=gumbel, random=False,
                       cudaUseFP16=str(fp16).lower(), nnRandomize='true')
    directories, counts = [], []
    for index, variations in enumerate([False, True]):
        text, count = record(game, variations=variations)
        directory = tmp_path / f'sgfs-{index}'
        write_sgf(directory, text); directories.append(directory); counts.extend(count)
    result = run(config, directories, model, binary=binary)
    actual = outputs(result, len(counts))
    assert actual == pytest.approx(-np.log(counts).mean(), abs=2e-5)
    if not muzero:
        # One thread; skipCache=true must perform all eight evaluations per
        # ruleset, including repeated files/positions and random symmetries.
        count = len(counts) * 8 * (2 if game == 'go' else 1)
        assert f'processed {count} rows {count} batches' in result.stdout + result.stderr


@pytest.mark.parametrize('game', GAMES)
def test_no_hint_terminal_and_invalid_sgf_selection(tmp_path, game):
    directory = tmp_path / 'sgfs'
    base = record(game)[0].replace('C[%HINT%]', 'C[%HINTLIGHT%]')
    write_sgf(directory, base, 'light.sgf')
    write_sgf(directory, 'not an sgf', 'invalid.sgf')
    write_sgf(directory, base.replace(';B[', 'C[%HINT%];B[', 1), 'root-comment.sgf')
    if game in ['freestyle', 'standard', 'renju', 'hex']:
        gm = 11 if game == 'hex' else 4
        moves = ';B[aa];W[ea];B[ab];W[eb];B[ac];W[ec];B[ad];W[ed];B[ae]C[%HINT%]'
        write_sgf(directory, f'(;GM[{gm}]SZ[5]RU[{game}]{moves})', 'terminal.sgf')
    elif game == 'xiangqi':
        fen = '3aka3/9/9/9/4P4/9/9/9/9/R3K4 w - - 119 60'
        write_sgf(directory, f'(;GM[7]SZ[9:10]XQFEN[{fen}];B[a0a1]C[%HINT%])', 'terminal.sgf')
    else:
        write_sgf(directory, '(;GM[1]SZ[5];B[aa];W[];B[]C[%HINT%])', 'terminal.sgf')
    config = configure(tmp_path / 'hint.cfg', game)
    result = run(config, directory)
    outputs(result, 0)
    assert 'Invalid SGF' in result.stdout + result.stderr
    assert 'Loaded neural net' not in result.stdout + result.stderr


@cuda
@pytest.mark.parametrize('muzero', [False, True])
@pytest.mark.parametrize('symmetry', [0, 2])
def test_xiangqi_nonuniform_policy_and_full_history(tmp_path, muzero, symmetry):
    _, model = make_model(tmp_path / 'model', 'xiangqi', muzero)
    config = configure(tmp_path / 'hint.cfg', 'xiangqi', muzero=muzero, random=False,
                       nnForcedSymmetry=symmetry, defaultBoardXSize=9, defaultBoardYSize=10)
    fen = '5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1'
    # Four plies of rule history exist before the hinted fifth ply. FEN alone
    # would lose repetition state and the model's previous-board features.
    sgf = write_sgf(tmp_path / 'sgfs', f'(;GM[7]SZ[9:10]XQFEN[{fen}]XQHIST[e8f8][f9e9];B[f8e8];W[e9f9];B[e8f8]C[%HINT%])')
    # Source GTP's move number is one-based: 3 loads the first two SGF moves.
    replies = run_gtp(config, [f'loadsgf {sgf} 3', f'kata-raw-nn {symmetry}', 'quit'], model=model)
    assert not any(error for error, _ in replies), replies
    reference = float(raw_policy(replies[1][1])['e8f8'])
    assert reference > 0
    actual = outputs(run(config, sgf.parent, model), 1)
    assert actual == pytest.approx(math.log(reference), abs=2e-4)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_go_metadata_profile_and_fixed_symmetry(tmp_path, muzero):
    from test_evalsgf import run as evalsgf
    net, model = make_model(tmp_path / 'model', 'go', muzero, metadata=True)
    config = configure(tmp_path / 'hint.cfg', 'go', muzero=muzero, random=False,
                       nnForcedSymmetry=0, humanSLProfile='rank_5d')
    directory = tmp_path / 'sgfs'
    write_sgf(directory, '(;GM[1]SZ[5]RU[AGA]KM[123];B[aa]C[%HINT%])')
    first = outputs(run(config, directory, model), 1)
    second = outputs(run(config, directory, model), 1)
    assert first == second
    expected_logs = []
    for rule, komi in [('Chinese', 7.5), ('Japanese', 6.5)]:
        sgf = tmp_path / f'{rule}.sgf'; sgf.write_text(f'(;GM[1]SZ[5]RU[{rule}]KM[{komi}])')
        inputs = tmp_path / f'{rule}.npz'
        reference = evalsgf(config, sgf, '-visits', '1', '-dump-npz-input-to', inputs, model=model, turn=0)
        assert reference.returncode == 0, reference.stdout + reference.stderr
        with np.load(inputs) as data:
            spatial, globals_, metadata = [torch.from_numpy(data[key]) for key in ('binaryInputNCHW', 'globalInputNC', 'metadataInputNC')]
        assert metadata.any()
        head = net.prediction if muzero else net
        with torch.no_grad():
            output = net.initial(spatial, globals_, metadata)[1] if muzero else net(spatial, globals_, metadata)
            # Empty Go board: all 25 points and pass are legal; SGF aa is index 0.
            logits = head.postprocess_output(output)[0][0][0, 0]
            expected_logs.append(logits.log_softmax(-1)[0].item())
    assert first == pytest.approx(np.mean(expected_logs), abs=2e-5)
    config.write_text(re.sub(r'^humanSLProfile = .*\n', '', config.read_text(), flags=re.M))
    result = run(config, directory, model)
    assert result.returncode != 0 and 'requires a humanSLProfile' in result.stdout + result.stderr


@cuda
def test_game_and_representation_canvas_boundaries(tmp_path):
    model = uniform_model(tmp_path / 'model', 'hex', True)
    config = configure(tmp_path / 'hint.cfg', 'renju', muzero=True, random=False)
    directory = tmp_path / 'sgfs'; write_sgf(directory, record('renju')[0])
    result = run(config, directory, model)
    assert result.returncode != 0 and 'model game differs' in result.stdout + result.stderr
    config = configure(config, 'hex', muzero=True, random=False)
    result = run(config, directory, model)
    assert result.returncode != 0 and 'game must match' in result.stdout + result.stderr
    write_sgf(directory, record('hex')[0].replace('SZ[5]', 'SZ[7]'))
    result = run(config, directory, model)
    assert result.returncode != 0 and 'canvas must contain' in result.stdout + result.stderr


def test_mixed_games_cannot_silently_share_a_model(tmp_path):
    config = configure(tmp_path / 'hint.cfg', 'hex')
    config.write_text(re.sub(r'^game = .*\n', '', config.read_text(), flags=re.M))
    directory = tmp_path / 'sgfs'
    write_sgf(directory, record('hex')[0], 'hex.sgf')
    write_sgf(directory, record('renju')[0], 'renju.sgf')
    result = run(config, directory)
    assert result.returncode != 0 and 'Hint position game differs' in result.stdout + result.stderr
