"""Native SGF mining, source weights and full-history position output."""
import json
import math
import os
import re
import signal
import shutil
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from export_muzero import export_components
from katago.train.model_pytorch import Model
from katago.train.muzero_model import MuZeroConfig, MuZeroModel
from test_controller_recovery import wait_until
from test_evalsgf import GAMES
from test_initializations import configure as initialization_config
from test_muzero_model import component_config
from test_xiangqi_controller import native_selfplay


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA mining')
MODES = ['tree', 'game', 'surprise']
XQ_MOVES = ['a3a4', 'a6a5', 'c3c4', 'c6c5', 'e3e4', 'e6e5', 'g3g4', 'g6g5',
            'i3i4', 'i6i5', 'b0c2', 'b9c7']


def configure(path, game, muzero=False, gumbel=False, random=True, **options):
    size = 19 if game == 'go' else 10 if game == 'xiangqi' else 5
    values = dict(bSizesXY='9-10' if game == 'xiangqi' else f'{size}-{size}',
        maxBoardSizeForNNBuffer=size, nnMaxBatchSize=8, maxVisits=8,
        rootPolicyOptimism=0, policyOptimism=0, fpuReductionMax=0, rootFpuReductionMax=0,
        nnRandomize='false', logToStderr='true', logToStdout='false')
    values.update(options)
    return initialization_config(path, game, muzero, gumbel, random, **values)


def record(game, *, comments=True):
    if game == 'xiangqi':
        # History precedes the SGF main line, and the eventual snapshot is also
        # beyond turn eight. Both parts must survive serialization.
        header = 'GM[7]SZ[9:10]RU[xiangqi]XQHIST[a3a4][a6a5]'
        moves = XQ_MOVES[2:]
    else:
        gm, size = (1, 19) if game == 'go' else (11 if game == 'hex' else 4, 5)
        header = f'GM[{gm}]SZ[{size}]RU[{"Chinese" if game == "go" else game}]'
        moves = ['aa', 'ee', 'ca', 'ce', 'ac', 'ec', 'cc', 'db', 'bd', 'bb']
    return '(;' + header + 'PB[black]PW[white]BR[5d]WR[3d]' + ''.join(
        f';{"B" if i % 2 == 0 else "W"}[{move}]' + ('C[%HINT%]' if comments and i == len(moves)-1 else '')
        for i, move in enumerate(moves)) + ')'


def uniform_model(path, game, muzero, *, metadata=False, value_logits=None):
    torch.manual_seed(193)
    canvas = 19 if game == 'go' else 10 if game == 'xiangqi' else 5
    config = component_config(game)
    if metadata:
        config['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    net = (MuZeroModel(MuZeroConfig(game, 6, config, component_config(game), component_config(game)), canvas)
           if muzero else Model(config, canvas))
    net.initialize(); net.eval()
    head = net.prediction if muzero else net
    with torch.no_grad():
        head.policy_head.conv2p.weight.zero_()
        if head.policy_head.has_pass:
            head.policy_head.linear_pass2.weight.zero_()
        head.value_head.linear_valuehead.weight.zero_()
        head.value_head.linear_valuehead.bias.zero_()
        if value_logits is not None:
            head.value_head.linear_valuehead.bias.copy_(torch.tensor(value_logits))
    if muzero:
        export_components(net, path, 'mining-uniform')
    else:
        path = path.with_suffix('.bin')
        with path.open('wb') as stream:
            write_model_binary(net, stream, 'mining-uniform')
    return path


def command(config, sgf, output, mode, model='/dev/null', *, threads=2, extra=(), binary=None):
    return [str(binary or ROOT / 'build/katago'), 'dataminesgfs', '-config', str(config),
        '-model', str(model), '-sgf', str(sgf), '-outdir', str(output), '-threads', str(threads),
        f'-{mode}-mode', '-for-testing', '-max-poses-per-out-file', '2', *map(str, extra)]


def read(output, *, binary=None):
    rows = []
    for path in sorted(output.glob('*.hintposes.txt')):
        lines = path.read_text().splitlines()
        assert 0 < len(lines) <= 2
        rows.extend(json.loads(line) for line in lines)
        checked = subprocess.run([str(binary or ROOT / 'build/katago'), 'viewstartposes',
            '-start-poses-file', str(path), '-check-legality'], capture_output=True, text=True, timeout=30)
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert not any(word in checked.stdout for word in ('Illegal', 'Mismatching', 'ERROR'))
    return rows


def run(config, sgf, output, mode, model='/dev/null', **options):
    result = subprocess.run(command(config, sgf, output, mode, model, **options),
                            capture_output=True, text=True, timeout=120)
    output.with_suffix('.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return read(output, binary=options.get('binary'))


@pytest.mark.parametrize('game', GAMES)
def test_native_manual_hint_selection(tmp_path, game):
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game))
    config = configure(tmp_path / 'mining.cfg', game)
    rows = run(config, sgf, tmp_path / 'poses', 'tree', extra=[
        '-manual-hint-only', '-min-hint-weight', 10, '-training-weight', .375, '-flip-if-pass'])
    assert len(rows) == 1
    row, = rows
    assert row.get('game', 'go') == game and row['weight'] == 10 and row['trainingWeight'] == .375
    assert len(row['moveLocs']) == 7 and row['initialTurnNumber'] == (4 if game == 'xiangqi' else 2)
    if game == 'xiangqi':
        assert row['xiangqiHistory'] == XQ_MOVES[:4]
        assert row['moveLocs'] == XQ_MOVES[4:-1] and row['hintLoc'] == XQ_MOVES[-1]


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_real_services_three_modes(tmp_path, game, muzero, gumbel):
    check_services(tmp_path, game, muzero, gumbel)


def check_services(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True):
    model = uniform_model(tmp_path / 'model', game, muzero)
    config = configure(tmp_path / 'mining.cfg', game, muzero, gumbel, False, cudaUseFP16=str(fp16).lower())
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game))
    tree = run(config, sgf, tmp_path / 'tree', 'tree', model, binary=binary,
               extra=['-manual-hint-only', '-min-hint-weight', 10, '-training-weight', .375])
    assert len(tree) == 1 and tree[0]['weight'] == 10
    game_rows = run(config, sgf, tmp_path / 'game', 'game', model, binary=binary,
        extra=['-game-mode-fast-threshold', -1, '-utility-threshold', -1,
               '-start-poses-before-hints', 2, '-training-weight', .375])
    assert any(row['hintLoc'] != 'null' for row in game_rows)
    assert any(row['hintLoc'] == 'null' for row in game_rows)
    assert all(row['trainingWeight'] == .375 for row in tree + game_rows)
    assert all(row.get('game', 'go') == game for row in tree + game_rows)
    if game == 'xiangqi':
        for row in game_rows:
            prefix = row['xiangqiHistory'] + row['moveLocs']
            assert prefix == XQ_MOVES[:len(prefix)] and len(prefix) >= 2
    # Uniform zero utility cannot produce either >0.2 surprise criterion.
    assert run(config, sgf, tmp_path / 'surprise', 'surprise', model, binary=binary,
               extra=['-manual-hint-only']) == []
    for mode in MODES:
        log = (tmp_path / mode / 'log.log').read_text()
        assert re.search(r'NN rows: [1-9][0-9]*', log)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_source_geometric_policy_weight(tmp_path, muzero):
    game = 'standard'
    model = uniform_model(tmp_path / 'model', game, muzero)
    config = configure(tmp_path / 'mining.cfg', game, muzero, False, False)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game))
    row, = run(config, sgf, tmp_path / 'tree', 'tree', model, extra=[
        '-manual-hint-only', '-turn-weight-lambda', .1, '-hint-scale', 2])
    # Nine occupied points, all remaining 16 legal; four equal uncached
    # probabilities yield the same geometric mean. Utility differences vanish.
    w = .12 / (1.1 / 16 + .02) - .5
    w = max(1, w)
    expected = [w * 1.5 + 1, w, w * .75 - .1, (w * .75 - .1) * .75 - .1]
    assert any(row['weight'] == pytest.approx(value * math.exp(-.2) * 2, abs=2e-5) for value in expected)


@pytest.mark.parametrize('option,value', [('-max-policy', 0), ('-min-turn', 100),
    ('-max-depth', 5), ('-required-player-name', 'absent'), ('-min-rank', 99)])
def test_source_filters(tmp_path, option, value):
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('renju'))
    config = configure(tmp_path / 'mining.cfg', 'renju')
    assert run(config, sgf, tmp_path / 'poses', 'tree', extra=[
        '-manual-hint-only', '-min-hint-weight', 10, option, value]) == []


@pytest.mark.parametrize('mode', MODES)
def test_full_queue_cancellation(tmp_path, mode):
    config = configure(tmp_path / 'mining.cfg', 'xiangqi', maxVisits=100000000)
    check_cancel(tmp_path, config, mode)


def check_cancel(tmp_path, config, mode, model='/dev/null'):
    # Distinct comments defeat source comment hashing; both bounded queues fill.
    sgf = tmp_path / 'many.sgfs'
    sgf.write_text('\n'.join(record('xiangqi').replace('%HINT%', f'%HINT% {i}') for i in range(1300)))
    output = tmp_path / 'poses'
    with (tmp_path / 'process.log').open('w+') as log:
        process = subprocess.Popen(command(config, sgf, output, mode, model, threads=1, extra=['-manual-hint-only']), stdout=log, stderr=log)
        try:
            wait_until(lambda: (output / 'log.log').exists() and
                (output / 'log.log').read_text().count('Starting ') >= (130 if mode == 'game' else 1026), seconds=40)
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=20)
            assert process.returncode == 0
            assert 'All done' in (output / 'log.log').read_text()
        finally:
            if process.poll() is None:
                process.kill(); process.wait()


@cuda
@pytest.mark.parametrize('mode', ['tree', 'game'])
def test_cancel_muzero_service(tmp_path, mode):
    model = uniform_model(tmp_path / 'model', 'xiangqi', True)
    config = configure(tmp_path / 'mining.cfg', 'xiangqi', True, True, False, maxVisits=100000000)
    check_cancel(tmp_path, config, mode, model)


@pytest.mark.parametrize('mode', ['tree', 'game'])
def test_invalid_sgf_drains_workers(tmp_path, mode):
    config = configure(tmp_path / 'mining.cfg', 'renju')
    sgf = tmp_path / 'invalid.sgf'; sgf.write_text('(;GM[4]SZ[5]RU[renju];B[zz])')
    result = subprocess.run(command(config, sgf, tmp_path / 'poses', mode), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Move out of bounds' in result.stderr
    assert 'Waiting for final writing and cleanup' in result.stderr
    assert 'NN rows:' in result.stderr and 'finishing, processed' in result.stderr


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_metadata_primary_model(tmp_path, muzero):
    model = uniform_model(tmp_path / 'model', 'go', muzero, metadata=True)
    config = configure(tmp_path / 'mining.cfg', 'go', muzero, False, False, humanSLProfile='rank_5d')
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('go'))
    row, = run(config, sgf, tmp_path / 'poses', 'tree', model,
               extra=['-manual-hint-only', '-min-hint-weight', 10])
    assert row['weight'] == 10


@cuda
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_surprise_output_and_previous_weights(tmp_path, muzero, gumbel):
    model = uniform_model(tmp_path / 'model', 'standard', muzero, value_logits=[3, 0, -3, 0])
    config = configure(tmp_path / 'mining.cfg', 'standard', muzero, gumbel, False, maxVisits=32)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('standard'))
    rows = run(config, sgf, tmp_path / 'poses', 'surprise', model,
               extra=['-manual-hint-only', '-training-weight', .375])
    assert len(rows) == 3
    by_turn = {row['initialTurnNumber'] + len(row['moveLocs']): row for row in rows}
    assert set(by_turn) == {7, 8, 9}
    weight = by_turn[9]['weight']
    assert weight > 0
    assert by_turn[8]['weight'] == pytest.approx(weight * .5)
    assert by_turn[7]['weight'] == pytest.approx(weight * .25)
    assert all(row['trainingWeight'] == .375 for row in rows)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_mined_xiangqi_hint_reaches_native_training_writer(tmp_path, muzero):
    model = uniform_model(tmp_path / 'model', 'xiangqi', muzero)
    config = configure(tmp_path / 'mining.cfg', 'xiangqi', muzero, True, False)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('xiangqi'))
    hints = tmp_path / 'hints'
    row, = run(config, sgf, hints, 'tree', model, extra=['-manual-hint-only', '-min-hint-weight', 10])
    models = tmp_path / 'models'; models.mkdir()
    if muzero:
        shutil.copytree(model, models / 'published')
    else:
        published = models / 'published'; published.mkdir()
        shutil.copyfile(model, published / 'model.bin')
    config = configure(tmp_path / 'consume.cfg', 'xiangqi', muzero, True, False,
        hintPosesDir=hints, hintPosesProb=1, maxMovesPerGame=1, cheapSearchProb=0,
        forkSidePositionProb=0, earlyForkGameProb=0)
    result = native_selfplay(tmp_path, config)
    assert result.returncode == 0, result.stdout + result.stderr
    shards = list((tmp_path / 'output').glob('*/tdata/*.npz'))
    assert shards
    for path in shards:
        with np.load(path) as data:
            targets = data['globalTargetsNC']
            assert np.all(targets[:, 51] + targets[:, 56] == row['initialTurnNumber'] + len(row['moveLocs']))
            assert np.all(targets[:, [2, 6, 10, 14, 18]] == 0)
            assert data['policyTargetsNCMove'].shape[-1] == 2086


def test_comment_deduplication_and_game_filter(tmp_path):
    config = configure(tmp_path / 'mining.cfg', 'renju')
    sgf = tmp_path / 'games.sgfs'
    base = record('renju')
    sgf.write_text('\n'.join([base, base, base.replace('%HINT%', '%HINT% %NOHINT%'),
        base.replace('%HINT%', '%HINTLIGHT%'), record('hex')]))
    rows = run(config, sgf, tmp_path / 'poses', 'tree',
               extra=['-manual-hint-only', '-min-hint-weight', 10])
    assert sorted(row['weight'] for row in rows) == [5, 10]
    assert 'SGF game does not match' in (tmp_path / 'poses/log.log').read_text()


@cuda
def test_bundle_game_and_canvas(tmp_path):
    model = uniform_model(tmp_path / 'model', 'hex', True)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('hex'))
    for game, options, error in [('renju', {}, 'model game differs'),
                                 ('hex', {'bSizesXY': '7-7'}, 'canvas must contain')]:
        config = configure(tmp_path / f'{game}.cfg', game, True, False, False, **options)
        result = subprocess.run(command(config, sgf, tmp_path / game, 'tree', model),
                                capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and error in result.stderr
