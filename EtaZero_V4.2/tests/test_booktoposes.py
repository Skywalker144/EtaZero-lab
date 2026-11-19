"""Source book sampling weights, actual model inference, and reusable game histories."""
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
from katago.train.model_pytorch import Model
from katago.train.muzero_model import MuZeroConfig, MuZeroModel
from test_book import GAMES
from test_datamine import configure as selfplay_config
from test_muzero_model import component_config
from test_xiangqi_controller import native_selfplay


BINARY = os.environ.get('ETAZERO_BOOK_BINARY', ROOT / 'build/katago')


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires actual CUDA book inference')


def create_books(directory, binary=None):
    directory.mkdir()
    result = subprocess.run([str(binary or BINARY), 'runtests', '-book-games', str(directory)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return directory


@pytest.fixture(scope='module')
def books(tmp_path_factory):
    return create_books(tmp_path_factory.mktemp('booktoposes') / 'books')


def configure(path, game=None, muzero=False, gumbel=False, random=True, fp16=True, **changes):
    options = dict(useMuZero=str(muzero).lower(), debugSkipNeuralNet=str(random).lower(),
        maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 7, requireMaxBoardSize='false',
        nnRandomize='true', nnMaxBatchSize=4, nnCacheSizePowerOfTwo=8, nnMutexPoolSizePowerOfTwo=4,
        cudaUseFP16=str(fp16).lower(), inputsUseNHWC='true',
        rootSearchAlgorithm='gumbel' if gumbel else 'puct')
    # Infer game from the book unless this is explicitly overridden below.
    options.update(changes)
    path.write_text(''.join(f'{key} = {value}\n' for key, value in options.items()))
    return path


def model_file(path, game, muzero, *, metadata=False, uniform=True, canvas=None):
    torch.set_num_threads(1)
    torch.manual_seed(685)
    canvas = canvas or (10 if game == 'xiangqi' else 7)
    config = component_config(game)
    if metadata:
        config['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    if muzero:
        net = MuZeroModel(MuZeroConfig(game, 6, config, component_config(game), component_config(game)), canvas)
    else:
        net = Model(config, canvas)
    net.initialize(); net.eval()
    head = net.prediction if muzero else net
    if uniform:
        with torch.no_grad():
            head.policy_head.conv2p.weight.zero_()
            if head.policy_head.has_pass:
                head.policy_head.linear_pass2.weight.zero_()
            head.value_head.linear_valuehead.weight.zero_()
            head.value_head.linear_valuehead.bias.copy_(torch.log(torch.tensor([3., 1., 1., 1.])))
    if muzero:
        export_components(net, path, 'booktoposes-check')
    else:
        path = path.with_suffix('.bin')
        with path.open('wb') as stream:
            write_model_binary(net, stream, 'booktoposes-check')
    return net, path


def run(book, output, config, model='/dev/null', *, binary=None, threads=2, extra=(), success=True):
    output.mkdir()
    result = subprocess.run([str(binary or BINARY), 'booktoposes',
        '-book-file', str(book), '-out-dir', str(output), '-config', str(config), '-model', str(model),
        '-include-depth', '100', '-num-threads', str(threads), *map(str, extra)],
        capture_output=True, text=True, timeout=90)
    text = result.stdout + result.stderr
    output.with_suffix('.log').write_text(text)
    assert (result.returncode == 0) == success, text
    rows = [json.loads(line) for path in sorted(output.glob('*.bookposes.txt')) for line in path.read_text().splitlines()]
    if success:
        for path in output.glob('*.bookposes.txt'):
            checked = subprocess.run([str(binary or BINARY), 'viewstartposes',
                '-start-poses-file', str(path), '-check-legality'], capture_output=True, text=True, timeout=30)
            assert checked.returncode == 0, checked.stdout + checked.stderr
            assert not any(word in checked.stdout for word in ['Illegal', 'Mismatching', 'ERROR'])
    return rows, text


def totals(text):
    labels = ['Constant', 'Depth', 'PolicySurprise', 'ValueSurpriseIrreducible', 'ValueSurpriseDivergence']
    return np.array([float(re.search(rf'totalWeightFrom{label} (\S+)', text)[1]) for label in labels])


@pytest.mark.parametrize('game', GAMES)
def test_native_sweep_depth_weights_and_startposes(books, tmp_path, game):
    cfg = configure(tmp_path / 'sample.cfg', game)
    rows, text = run(books / f'{game}.katabook', tmp_path / 'poses', cfg,
                     extra=['-constant-weight', 2, '-depth-weight', 3, '-depth-weight-scale', 2, '-enable-hints'])
    assert rows
    for row in rows:
        depth = row['initialTurnNumber'] + len(row['moveLocs']) - (0 if game == 'go' else 17)
        assert row.get('game', 'go') == game
        assert row['weight'] == pytest.approx(2 + 3 * math.exp(-depth / 2))
        assert 0 <= depth <= 6 and len(row['moveLocs']) == min(depth, 5)
        if game == 'xiangqi':
            assert len(row['xiangqiHistory']) + len(row['moveLocs']) == depth
    assert totals(text)[0] == 2 * len(rows)
    assert totals(text)[1] == pytest.approx(sum(r['weight'] - 2 for r in rows), rel=2e-6)
    assert np.all(totals(text)[2:] == 0)


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
@cuda
def test_exact_source_weights(books, tmp_path, game, muzero, gumbel):
    check_source_weights(tmp_path, game, muzero, gumbel, books=books)


def check_source_weights(tmp_path, game, muzero, gumbel, *, books=None, binary=None, fp16=True):
    books = books or create_books(tmp_path / 'books', binary)
    _, model = model_file(tmp_path / 'model', game, muzero)
    cfg = configure(tmp_path / 'sample.cfg', game, muzero, gumbel, random=False, fp16=fp16)
    depth_limit = 0 if game == 'xiangqi' else 1
    rows, text = run(books / f'{game}.katabook', tmp_path / 'poses', cfg, model, binary=binary,
        extra=['-max-depth', depth_limit, '-constant-weight', 2, '-depth-weight', 3, '-depth-weight-scale', 2,
               '-policy-surprise-weight', 4, '-value-surprise-weight', 5, '-enable-hints'])
    assert rows and any(r['hintLoc'] != 'null' for r in rows)
    # Each fixture book stores W-L=1/8. Source conversion gives a Bernoulli
    # target q=9/16; D and NR are not collapsed into a new four-way book loss.
    q = 9 / 16
    entropy = -q * math.log(q) - (1-q) * math.log(1-q)
    expected_parts = []
    for row in rows:
        depth = row['initialTurnNumber'] + len(row['moveLocs']) - (0 if game == 'go' else 17)
        assert 0 <= depth <= depth_limit
        # With NR masked, [3,1,1,1] gives mover W-L=2/5, hence white chance
        # 3/10 on Black's turn and 7/10 on White's turn. At depths 0/1 no Go
        # captures or Renju forbidden patterns exist in these fixtures.
        chance = .3 if depth % 2 == 0 else .7
        cross_entropy = -q * math.log(chance) - (1-q) * math.log(1-chance)
        legal = 44 if game == 'xiangqi' else (50 if game == 'go' else 49) - depth
        policy = math.log(legal) if row['hintLoc'] != 'null' else 0
        parts = [2, 3 * math.exp(-depth / 2), 4 * policy, 5 * entropy, 5 * (cross_entropy-entropy)]
        assert row['weight'] == pytest.approx(sum(parts), abs=2e-4)
        expected_parts.append(parts)
    np.testing.assert_allclose(totals(text), np.sum(expected_parts, axis=0), atol=2e-3, rtol=1e-5)
    if not muzero:
        # An uncached explicit sweep, not one evaluation reused for all views.
        row_counts = re.findall(r'finishing, processed (\d+) rows', text)
        assert sum(map(int, row_counts)) == len(rows) * (2 if game in ('hex', 'xiangqi') else 8)


@pytest.mark.parametrize('name,count', [('history', 4), ('check', 4), ('chase', 4), ('limit', 1)])
def test_terminal_filter_and_owned_xiangqi_history(books, tmp_path, name, count):
    cfg = configure(tmp_path / 'sample.cfg', 'xiangqi')
    rows, text = run(books / f'xiangqi-{name}.katabook', tmp_path / 'poses', cfg,
                     extra=['-constant-weight', 1, '-policy-surprise-weight', 1, '-enable-hints'])
    assert len(rows) == count
    assert all(row['hintLoc'] != 'null' for row in rows)
    for row in rows:
        assert len(row['xiangqiHistory']) == (0 if name == 'limit' else 4)
        if name == 'history':
            assert 97 <= row['initialTurnNumber'] + len(row['moveLocs']) < 101
    assert totals(text)[0] == count


@cuda
@pytest.mark.parametrize('game', ['freestyle', 'standard', 'renju', 'hex'])
@pytest.mark.parametrize('muzero', [False, True])
def test_placement_terminal_has_no_policy_sample(books, tmp_path, game, muzero):
    _, model = model_file(tmp_path / 'model', game, muzero)
    cfg = configure(tmp_path / 'sample.cfg', game, muzero, random=False)
    rows, text = run(books / f'{game}-terminal.katabook', tmp_path / 'poses', cfg, model,
        extra=['-constant-weight', 1, '-policy-surprise-weight', 1, '-enable-hints'])
    row, = rows
    assert row['initialTurnNumber'] == 41 and not row['moveLocs']
    assert row['weight'] == pytest.approx(1 + math.log(19), abs=2e-5)
    assert totals(text)[0] == 1


def test_source_visit_depth_and_minimum_weight_filters(books, tmp_path):
    cfg = configure(tmp_path / 'sample.cfg', 'hex')
    book = books / 'hex.katabook'
    # Leaves have ten visits. Requiring more excludes them, while depth and
    # the final weight threshold operate at their separate source boundaries.
    filtered, _ = run(book, tmp_path / 'filtered', cfg, extra=[
        '-min-visits', 11, '-max-depth', 3, '-constant-weight', 2, '-depth-weight', 3,
        '-depth-weight-scale', 2, '-min-weight', 3, '-enable-hints'])
    assert filtered
    assert all(row['hintLoc'] != 'null' and row['weight'] >= 3 and
               row['initialTurnNumber'] + len(row['moveLocs']) <= 19 for row in filtered)
    empty, text = run(book, tmp_path / 'empty', cfg, extra=['-min-visits', 1e9, '-constant-weight', 1])
    assert not empty and np.all(totals(text) == 0)


@pytest.mark.parametrize('changes,threads,error', [
    ({'game': 'go'}, 2, 'configured game differs from the book'),
    ({}, 0, 'at least one processing thread'),
])
def test_invalid_inputs(books, tmp_path, changes, threads, error):
    cfg = configure(tmp_path / 'sample.cfg', 'xiangqi')
    cfg.write_text(cfg.read_text() + ''.join(f'{key} = {value}\n' for key, value in changes.items()))
    rows, text = run(books / 'xiangqi.katabook', tmp_path / 'poses', cfg, threads=threads, success=False)
    assert not rows and error in text


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_xiangqi_history_consumed_by_native_selfplay(books, tmp_path, muzero):
    _, model = model_file(tmp_path / 'model', 'xiangqi', muzero)
    cfg = configure(tmp_path / 'sample.cfg', 'xiangqi', muzero, random=False)
    rows, _ = run(books / 'xiangqi-history.katabook', tmp_path / 'poses', cfg, model,
                  extra=['-constant-weight', 1, '-policy-surprise-weight', 1, '-enable-hints'])
    last = max(rows, key=lambda r: r['initialTurnNumber'] + len(r['moveLocs']))
    assert len(last['xiangqiHistory']) + len(last['moveLocs']) == 7
    assert last['initialTurnNumber'] + len(last['moveLocs']) == 100
    consumed = tmp_path / 'consumed'; consumed.mkdir()
    (consumed / 'chosen.startposes.txt').write_text(json.dumps(last) + '\n')
    published = tmp_path / 'models/published'; published.parent.mkdir()
    if muzero:
        shutil.copytree(model, published)
    else:
        published.mkdir(); shutil.copyfile(model, published / 'model.bin')
    cfg = selfplay_config(tmp_path / 'consume.cfg', 'xiangqi', muzero, True, False,
        hintPosesDir=consumed, hintPosesProb=1, maxMovesPerGame=1, cheapSearchProb=0,
        forkSidePositionProb=0, earlyForkGameProb=0)
    result = native_selfplay(tmp_path, cfg)
    assert result.returncode == 0, result.stdout + result.stderr
    shards = list((tmp_path / 'output').glob('*/tdata/*.npz'))
    assert shards
    for shard in shards:
        with np.load(shard) as data:
            assert np.all(data['globalTargetsNC'][:, 51] + data['globalTargetsNC'][:, 56] == 100)
            assert data['policyTargetsNCMove'].shape[-1] == 2086
            assert np.all(data['globalTargetsNC'][:, [2, 6, 10, 14, 18]] == 0)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_nonuniform_xiangqi_sweep_matches_full_history_gtp(books, tmp_path, muzero):
    from test_gtp import run_gtp
    from test_xiangqi_gtp import config_for_gtp, raw_policy
    _, model = model_file(tmp_path / 'model', 'xiangqi', muzero, uniform=False)
    cfg = configure(tmp_path / 'sample.cfg', 'xiangqi', muzero, random=False, fp16=False)
    rows, _ = run(books / 'xiangqi-history.katabook', tmp_path / 'poses', cfg, model,
        extra=['-constant-weight', 1, '-policy-surprise-weight', 2, '-value-surprise-weight', 3, '-enable-hints'])
    last = max(rows, key=lambda r: r['initialTurnNumber'] + len(r['moveLocs']))
    full_history = last['xiangqiHistory'] + last['moveLocs']
    assert len(full_history) == 7
    sgf = tmp_path / 'current.sgf'
    sgf.write_text(f'(;GM[7]SZ[9:10]RU[xiangqi]XQFEN[{last["xiangqiFen"]}]XQHIST' +
                   ''.join(f'[{m}]' for m in full_history) + 'XQTURN[100])')
    gtp = config_for_gtp(tmp_path / 'gtp.cfg', muzero=muzero, random=False, cudaUseFP16='false')
    replies = run_gtp(gtp, [f'loadsgf {sgf} 1', 'kata-raw-nn 0 0', 'kata-raw-nn 2 0', 'quit'], model=model)
    assert not any(error for error, _ in replies), replies
    expected = []
    q = 9 / 16
    for _, text in replies[1:3]:
        policy = float(raw_policy(text)[last['hintLoc']])
        values = dict(re.findall(r'(?m)^(whiteWin|whiteLoss) ([^\n]+)', text))
        chance = .5 * (1 + float(values['whiteWin']) - float(values['whiteLoss']))
        expected.append(1 - 2 * math.log(policy) - 3 * (q * math.log(chance) + (1-q) * math.log(1-chance)))
    assert last['weight'] == pytest.approx(np.mean(expected), abs=5e-4)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
@pytest.mark.parametrize('profile_key', ['humanSLProfile', 'humanSLProfile0'])
def test_go_metadata_and_exact_mean_log_policy(books, tmp_path, muzero, profile_key):
    from test_evalsgf import run as evalsgf, config_for
    net, model = model_file(tmp_path / 'model', 'go', muzero, metadata=True, uniform=False)
    cfg = configure(tmp_path / 'sample.cfg', 'go', muzero, random=False, fp16=False,
                    numSearchThreads=1, maxVisits=1, rootPolicyOptimism=0, policyOptimism=0)
    _, text = run(books / 'go.katabook', tmp_path / 'missing-profile', cfg, model, success=False)
    assert 'requires a humanSLProfile' in text
    cfg.write_text(cfg.read_text() + f'{profile_key} = rank_5d\n')
    rows, _ = run(books / 'go.katabook', tmp_path / 'poses', cfg, model,
                  extra=['-max-depth', 0, '-constant-weight', 1, '-policy-surprise-weight', 2,
                         '-value-surprise-weight', 3, '-enable-hints'])
    row, = rows
    sgf = tmp_path / 'empty.sgf'; sgf.write_text('(;GM[1]SZ[7]RU[Tromp-Taylor]KM[7.5])')
    inputs = tmp_path / 'input.npz'
    # The independent search diagnostic has source SearchParams requirements;
    # it is a separate consumer from the raw-inference-only sampling command.
    reference_cfg = config_for(tmp_path / 'reference.cfg', 'go', muzero=muzero, random=False,
        maxBoardSizeForNNBuffer=7, numSearchThreads=1, maxVisits=1, humanSLProfile='rank_5d',
        rootPolicyOptimism=0, policyOptimism=0)
    result = evalsgf(reference_cfg, sgf, '-visits', 1, '-dump-npz-input-to', inputs, model=model, turn=0)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(inputs) as data:
        spatial, globals_, meta = [torch.from_numpy(data[k]) for k in ('binaryInputNCHW', 'globalInputNC', 'metadataInputNC')]
    assert meta.any()
    with torch.no_grad():
        output = net.initial(spatial, globals_, meta)[1] if muzero else net(spatial, globals_, meta)
        head = net.prediction if muzero else net
        output = head.postprocess_output(output)[0]
        log_policy = output[0][0, 0].log_softmax(-1).numpy()
        values = output[1][0].clone(); values[3] = -torch.inf
        values = values.softmax(-1).numpy()
    # Empty-board features are invariant. Average the log probabilities of
    # the full eight-position orbit; averaging probabilities then taking log
    # would be a different weighting rule.
    grid = log_policy[:-1].reshape(7, 7)
    x = 'ABCDEFG'.index(row['hintLoc'][0]); y = 7 - int(row['hintLoc'][1:])
    mean_log_policy = np.mean([view[y, x] for k in range(4) for view in (np.rot90(grid, k), np.fliplr(np.rot90(grid, k)))])
    chance = .5 * (1 + values[2] - values[0])  # Black to play, white perspective.
    q = 9 / 16
    expected = 1 - 2 * mean_log_policy - 3 * (q * math.log(chance) + (1-q) * math.log(1-chance))
    assert row['weight'] == pytest.approx(expected, abs=2e-4)


@cuda
def test_muzero_game_canvas_and_indexed_selection(books, tmp_path):
    _, model = model_file(tmp_path / 'model', 'hex', True)
    cfg = configure(tmp_path / 'sample.cfg', 'hex', random=False, useMuZero0='true')
    rows, _ = run(books / 'hex.katabook', tmp_path / 'indexed', cfg, model,
                  extra=['-max-depth', 0, '-constant-weight', 1, '-policy-surprise-weight', 1])
    assert len(rows) == 1 and rows[0]['weight'] == pytest.approx(1 + math.log(49), abs=2e-5)
    _, text = run(books / 'renju.katabook', tmp_path / 'wrong-game', cfg, model, success=False)
    assert 'model game differs from the book' in text
    _, small = model_file(tmp_path / 'small', 'hex', True, canvas=5)
    _, text = run(books / 'hex.katabook', tmp_path / 'small-canvas', cfg, small, success=False)
    assert 'canvas must contain the book board' in text
