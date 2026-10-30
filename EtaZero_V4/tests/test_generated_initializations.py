"""Native GameRunner initialization -> PosWriter -> imported hint positions."""
import json
import os
import re
import signal
import subprocess

import numpy as np
import pytest

from etazero import ROOT
from test_controller_recovery import wait_until
from test_evalsgf import GAMES
from test_initializations import configure
from test_surprise_dump import constant_model
from test_xiangqi_controller import native_selfplay


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA initialization')


def command(config, output, model='/dev/null', *, count=3, threads=2, binary=None):
    return [str(binary or ROOT / 'build/katago'), 'genposesfromselfplayinit',
            '-config', str(config), '-model', str(model), '-outdir', str(output),
            '-poses', str(count), '-threads', str(threads), '-max-poses-per-out-file', '2']


def read(output, *, count=None, binary=None):
    paths = sorted(output.glob('*.outposes.txt'))
    rows = []
    for path in paths:
        lines = path.read_text().splitlines()
        assert 1 <= len(lines) <= 2
        rows.extend(json.loads(line) for line in lines)
        result = subprocess.run([str(binary or ROOT / 'build/katago'), 'viewstartposes',
            '-start-poses-file', str(path), '-check-legality'], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert not any(word in result.stdout for word in ('Illegal', 'Mismatching', 'ERROR'))
    if count is not None:
        assert len(rows) == count
    return rows


def generate(config, output, model='/dev/null', **options):
    result = subprocess.run(command(config, output, model, **options), capture_output=True, text=True, timeout=90)
    output.with_suffix('.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return read(output, count=options.get('count', 3), binary=options.get('binary'))


@pytest.mark.parametrize('game', GAMES)
def test_source_writer_and_random_evaluator(tmp_path, game):
    config = configure(tmp_path / 'generate.cfg', game)
    rows = generate(config, tmp_path / 'poses')
    assert all(row.get('game', 'go') == game and row['hintLoc'] == 'null' and row['weight'] == 1 for row in rows)
    assert all(row['initialTurnNumber'] == 0 and row['moveLocs'] == [] for row in rows)


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_model_services_and_policy_prefix(tmp_path, game, muzero, gumbel):
    check_model_services(tmp_path, game, muzero, gumbel)


def check_model_services(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True):
    model = constant_model(tmp_path / 'model', game, muzero)
    config = configure(tmp_path / 'generate.cfg', game, muzero, gumbel, False,
        cudaUseFP16=str(fp16).lower(), initGamesWithPolicy='true', policyInitAreaProp=.1)
    rows = generate(config, tmp_path / 'poses', model, binary=binary)
    assert all(row.get('game', 'go') == game for row in rows)
    assert re.search(r'NN rows: [1-9][0-9]*', (tmp_path / 'poses/log.log').read_text())


def hint_options(tmp_path, sample):
    directory = tmp_path / 'hints'; directory.mkdir()
    (directory / 'input.hintposes.txt').write_text(json.dumps(sample) + '\n')
    return dict(hintPosesDir=directory, hintPosesProb=1)


def point_sample(game, width=5, height=5):
    # A1 has no Go liberty: it must remain present in placement games.
    rows = [['.'] * width for _ in range(height)]
    for x, y, color in [(0, 0, 'X'), (1, 0, 'O'), (0, 1, 'O'), (2, 2, 'X')]:
        rows[y][x] = color
    return dict(game=game, xSize=width, ySize=height, board='/'.join(''.join(row) for row in rows[::-1]) + '/',
                nextPla='B', moveLocs=['A4', 'D2'], movePlas=['B', 'W'],
                initialTurnNumber=73, hintLoc='null', weight=1)


def point_variants(sample, symmetries):
    # Transform explicit coordinates, independently of the C++ board layout.
    width, height = sample['xSize'], sample['ySize']
    board = [list(row) for row in sample['board'].strip('/').split('/')[::-1]]
    variants = set()
    for sym in symmetries:
        def xy(x, y):
            x, y = width - 1 - x if sym & 2 else x, height - 1 - y if sym & 1 else y
            return (y, x) if sym & 4 else (x, y)
        nx, ny = (height, width) if sym & 4 else (width, height)
        result = [['.'] * nx for _ in range(ny)]
        for y in range(height):
            for x in range(width):
                u, v = xy(x, y); result[v][u] = board[y][x]
        moves = []
        for move in sample['moveLocs']:
            x, y = xy(ord(move[0])-ord('A'), int(move[1:])-1)
            moves.append(chr(ord('A')+x) + str(y+1))
        variants.add((nx, ny, '/'.join(''.join(row) for row in result[::-1]) + '/', tuple(moves)))
    return variants


@pytest.mark.parametrize('game', ['freestyle', 'standard', 'renju', 'hex'])
def test_only_valid_symmetries_and_zero_liberty_stones(tmp_path, game):
    sample = point_sample(game)
    config = configure(tmp_path / 'generate.cfg', game, **hint_options(tmp_path, sample))
    rows = generate(config, tmp_path / 'poses', count=24)
    allowed = point_variants(sample, [0, 3] if game == 'hex' else range(8))
    for row in rows:
        assert (row['xSize'], row['ySize'], row['board'], tuple(row['moveLocs'])) in allowed
        assert row['movePlas'] == sample['movePlas'] and row['initialTurnNumber'] == 73


def test_rectangular_go_moves_use_original_dimensions(tmp_path):
    sample = point_sample('go', 5, 4)
    # Go setup must have liberties, unlike the placement fixture above.
    sample['board'] = '...../..X../O..../X.O../'
    config = configure(tmp_path / 'generate.cfg', 'go', bSizesXY='5-4', **hint_options(tmp_path, sample))
    rows = generate(config, tmp_path / 'poses', count=24)
    allowed = point_variants(sample, range(8))
    assert all((row['xSize'], row['ySize'], row['board'], tuple(row['moveLocs'])) in allowed for row in rows)


def test_xiangqi_complete_history_and_hint_consumption(tmp_path):
    # Seven plies of a long-check cycle; e9f9 next makes the checker lose.
    fen = '5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1'
    history = ['e8f8', 'f9e9', 'f8e8', 'e9f9', 'e8f8', 'f9e9', 'f8e8']
    sample = dict(game='xiangqi', xSize=9, ySize=10, xiangqiFen=fen, xiangqiHistory=history[:2],
        nextPla='B', moveLocs=history[2:], movePlas=['B','W','B','W','B'],
        initialTurnNumber=97, hintLoc='null', weight=1)
    config = configure(tmp_path / 'generate.cfg', 'xiangqi', **hint_options(tmp_path, sample))
    rows = generate(config, tmp_path / 'poses', count=24)
    reflected_fen = '3k5/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1'
    reflected_moves = ['e8d8', 'd9e9', 'd8e8', 'e9d9', 'e8d8', 'd9e9', 'd8e8']
    for row in rows:
        expected = reflected_moves if row['xiangqiFen'] == reflected_fen else history
        assert row['xiangqiFen'] in (fen, reflected_fen)
        assert row['xiangqiHistory'] + row['moveLocs'] == expected
        assert row['initialTurnNumber'] + len(row['moveLocs']) == 102
    # Native hint loading uses these source suffixes. Exported outposes files
    # remain unchanged; a separate copy is consumed as training input.
    consumed = tmp_path / 'consumed'; consumed.mkdir()
    (consumed / 'generated.startposes.txt').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    config = configure(tmp_path / 'consume.cfg', 'xiangqi', hintPosesDir=consumed, hintPosesProb=1,
                       maxMovesPerGame=1, cheapSearchProb=0, logToStdout='true')
    result = native_selfplay(tmp_path, config)
    assert result.returncode == 0, result.stdout + result.stderr
    shards = list((tmp_path / 'output').glob('*/tdata/*.npz'))
    assert shards
    for path in shards:
        with np.load(path) as data:
            # Imported history must reach the native writer without losing its
            # original turn offset or introducing an impossible NR outcome.
            assert np.all(data['globalTargetsNC'][:, 56] == 97)
            assert np.all(data['globalTargetsNC'][:, 51] + data['globalTargetsNC'][:, 56] == 102)
            assert np.all(data['globalTargetsNC'][:, [2, 6, 10, 14, 18]] == 0)


@cuda
def test_bundle_game_and_canvas_are_checked(tmp_path):
    model = constant_model(tmp_path / 'model', 'hex', True)
    for game, changes, error in [('renju', {}, 'game differs'), ('hex', {'bSizesXY': '7-7'}, 'canvas must contain')]:
        config = configure(tmp_path / f'{game}.cfg', game, muzero=True, random=False, **changes)
        output = tmp_path / game
        result = subprocess.run(command(config, output, model), capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and error in result.stdout + result.stderr
        assert not list(output.glob('*.outposes.txt'))


def check_shutdown(tmp_path, config, model='/dev/null'):
    output = tmp_path / 'poses'
    with (tmp_path / 'process.log').open('w') as log:
        process = subprocess.Popen(command(config, output, model, count=100000), stdout=log, stderr=log)
        try:
            wait_until(lambda: (output / '0.outposes.txt').exists() and (output / '0.outposes.txt').stat().st_size > 0, seconds=30)
            assert process.poll() is None
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=15) == 0
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    rows = read(output)
    assert 0 < len(rows) < 100000
    assert 'All done' in (output / 'log.log').read_text()


def test_cancel_in_flight_game_drains_writer(tmp_path):
    config = configure(tmp_path / 'generate.cfg', 'xiangqi', gumbel=True, maxMovesPerGame=30)
    check_shutdown(tmp_path, config)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_cancel_real_model_services(tmp_path, muzero):
    model = constant_model(tmp_path / 'model', 'xiangqi', muzero)
    config = configure(tmp_path / 'generate.cfg', 'xiangqi', muzero, True, False, maxMovesPerGame=30)
    check_shutdown(tmp_path, config, model)
