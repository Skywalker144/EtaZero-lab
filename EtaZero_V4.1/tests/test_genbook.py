"""Native opening-book expansion, real-state extraction, and source policy costs."""
import json
import math
import os
from pathlib import Path
import re
import subprocess

import pytest

from etazero import ROOT
from test_book import GAMES
from test_booktoposes import create_books, model_file
from test_evalsgf import config_for, record


BINARY = Path(os.environ.get('ETAZERO_BOOK_BINARY', ROOT / 'build/katago'))
cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires actual CUDA book search')


def configure(path, game, muzero=False, gumbel=False, random=True, **changes):
    options = dict(boardSizeX=9 if game == 'xiangqi' else 5, boardSizeY=10 if game == 'xiangqi' else 5,
        komi=0, repBound=9, numGameThreads=2, numToExpandPerIteration=3,
        rulesLabel=game, rulesLink='https://example.invalid/rules', logSearchInfo='false',
        maxVisits=32, maxVisitsForLeaves=4, minTreeVisitsToRecord=2, maxDepthToRecord=4,
        rootSymmetryPruning='true', cudaUseFP16='true', inputsUseNHWC='true',
        maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 7,
        errorFactor=1.5, costPerMove=.45, costPerUCBWinLossLoss=4.5,
        costPerUCBWinLossLossPow3=.75, costPerUCBWinLossLossPow7=.75, costPerLogPolicy=.75,
        costPerMovesExpanded=.25, costPerSquaredMovesExpanded=.05, bonusPerWinLossError=1,
        bonusPerExcessUnexpandedPolicy=2, bonusPerUnexpandedBestWinLoss=1.5,
        policyBoostSoftUtilityScale=.04, utilityPerPolicyForSorting=.02)
    options.update(changes)
    return config_for(path, game, muzero=muzero, gumbel=gumbel, random=random, **options)


def run(config, book, *, model='/dev/null', human=None, iterations=2, extra=(), success=True, binary=None):
    result = subprocess.run([str(binary or BINARY), 'genbook', '-config', str(config),
        '-model', str(model), '-book-file', str(book), '-log-file', str(book.with_suffix('.log')),
        '-num-iters', str(iterations), '-save-every', '1', '-num-book-threads', '2',
        *(['-human-model', str(human)] if human else []), *map(str, extra)],
        capture_output=True, text=True, timeout=90)
    text = result.stdout + result.stderr
    assert (result.returncode == 0) == success, text
    if success:
        assert 'DONE' in text and 'WARNING: Illegal move' not in text
        checked = subprocess.run([str(binary or BINARY), 'checkbook', '-book-file', str(book)],
                                 capture_output=True, text=True, timeout=30)
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert 'WARNING' not in checked.stdout and 'failed integrity' not in checked.stdout
    return text


def read_book(book):
    lines = book.read_text().splitlines()
    header = json.loads(lines[0])
    count = int(lines[1])
    nodes = [json.loads(line) for line in lines[count+2:]]
    assert len(nodes) == count
    for node in nodes:
        assert all(math.isfinite(node[key]) for key in ('wl', 'wlE', 'maxP', 'w', 'v'))
        for move in node['mvs']:
            assert 0 <= move['id'] < count and 0 <= move['rP'] <= 1
    return header, nodes


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('gumbel', [False, True])
def test_expansion_resume_and_trace_book(tmp_path, game, gumbel):
    cfg = configure(tmp_path / 'book.cfg', game, gumbel=gumbel)
    book = tmp_path / 'book.katabook'
    run(cfg, book)
    header, nodes = read_book(book)
    assert len(nodes) > 2
    assert any(node['par'] and nodes[node['par'][0]['id']]['par'] for node in nodes)
    symmetries = {0, 2} if game == 'xiangqi' else {0, 3} if game == 'hex' else set(range(8))
    for node in nodes:
        assert set(node['syms']) <= symmetries
        for move in node['mvs']:
            assert move['sym'] in symmetries
            assert re.fullmatch(r'[a-i][0-9][a-i][0-9]' if game == 'xiangqi' else r'(?:[A-HJ-Z][0-9]+|pass)', move['m'])
    if game == 'xiangqi':
        assert header['xiangqiHistory'] == [] and header['xiangqiFen'].endswith('0 1')
    run(cfg, book, iterations=1)
    _, continued = read_book(book)
    assert len(continued) > len(nodes)
    traced = tmp_path / 'traced.katabook'
    run(cfg, traced, iterations=0, extra=['-trace-book-file', book])
    # Source tracing replays a path to each leaf, not every alternate edge of
    # the transposition graph. All leaf identities must survive this operation.
    hashes = book.read_text().splitlines()[2:len(continued)+2]
    traced_lines = traced.read_text().splitlines()
    traced_hashes = set(traced_lines[2:int(traced_lines[1])+2])
    assert {hashes[n['id']] for n in continued if not n['mvs']} <= traced_hashes


def check_real_models(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True):
    _, model = model_file(tmp_path / 'model', game, muzero)
    cfg = configure(tmp_path / 'book.cfg', game, muzero, gumbel, random=False,
                    cudaUseFP16=str(fp16).lower())
    book = tmp_path / 'book.katabook'
    run(cfg, book, model=model, binary=binary)
    _, nodes = read_book(book)
    assert len(nodes) > 2
    # Uniform logits give independent exact root priors: no symmetry pruning
    # multiplicity or improved Gumbel target should enter raw book policy costs.
    expected = 1 / (44 if game == 'xiangqi' else 26 if game == 'go' else 25)
    assert nodes[0]['mvs']
    for move in nodes[0]['mvs']:
        assert move['rP'] == pytest.approx(expected, abs=3e-6)
    run(cfg, book, model=model, binary=binary, iterations=1)
    assert len(read_book(book)[1]) > len(nodes)


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
@cuda
def test_actual_model_search_and_raw_policy(tmp_path, game, muzero, gumbel):
    check_real_models(tmp_path, game, muzero, gumbel)


@pytest.mark.parametrize('game', ['hex', 'xiangqi'])
@pytest.mark.parametrize('gumbel', [False, True])
@cuda
def test_deep_latent_tree_extracts_only_real_legal_paths(tmp_path, game, gumbel):
    _, model = model_file(tmp_path / 'model', game, True)
    cfg = configure(tmp_path / 'book.cfg', game, True, gumbel, random=False,
                    maxVisits=512, maxVisitsForLeaves=8, maxDepthToRecord=5,
                    gumbelMaxConsideredActions=2)
    book = tmp_path / 'book.katabook'
    run(cfg, book, model=model, iterations=1)
    _, nodes = read_book(book)
    depths = {0: 0}
    for node in nodes:
        for move in node['mvs']:
            depths[move['id']] = min(depths.get(move['id'], 100), depths[node['id']]+1)
    # With 2086 static joint actions, this uniform Xiangqi latent model can
    # spend every deeper visit on real-illegal moves. They must be omitted,
    # rather than forcing the book to contain a path of a prescribed depth.
    assert max(depths.values()) >= (1 if game == 'xiangqi' else 3)


@pytest.mark.parametrize('game', GAMES)
def test_bonus_and_sgf_trace(tmp_path, game):
    cfg = configure(tmp_path / 'book.cfg', game)
    sgf = tmp_path / 'bonus.sgf'
    sgf.write_text(record(game).replace(';B[', 'C[BONUS 2 EXPAND 3 VISITS 8 BRANCH 2];B[', 1))
    book = tmp_path / 'book.katabook'
    text = run(cfg, book, iterations=0, extra=['-bonus-file', sgf, '-trace-sgf-file', sgf])
    assert 'Adding bonus 2' in text and 'Adding expand bonus 3' in text
    assert 'Adding required visits 8' in text and 'Adding required branching factor 2' in text
    assert len(read_book(book)[1]) >= 3


def test_trace_nonempty_go_initial_board(tmp_path):
    cfg = configure(tmp_path / 'book.cfg', 'go')
    sgf = tmp_path / 'initial.sgf'
    sgf.write_text('(;GM[1]SZ[5]RU[Tromp-Taylor]KM[0]PL[B]AB[ab][bc]AW[de])')
    book, traced = tmp_path / 'book.katabook', tmp_path / 'traced.katabook'
    run(cfg, book, iterations=1, extra=['-bonus-file', sgf])
    run(cfg, traced, iterations=0, extra=['-bonus-file', sgf, '-trace-book-file', book])
    _, nodes = read_book(book)
    hashes = book.read_text().splitlines()[2:len(nodes)+2]
    lines = traced.read_text().splitlines()
    assert {hashes[n['id']] for n in nodes if not n['mvs']} <= set(lines[2:int(lines[1])+2])


@pytest.mark.parametrize('game', GAMES[1:5])
def test_zero_liberty_bonus_setup_and_source_terminal_filter(tmp_path, game):
    cfg = configure(tmp_path / 'book.cfg', game)
    gm = 11 if game == 'hex' else 4
    sgf = tmp_path / 'terminal.sgf'
    # The white corner has no Go liberties. Source SGF tracing excludes terminal
    # samples, leaving just the initial position for this one-move ending.
    sgf.write_text(f'(;GM[{gm}]SZ[5]RU[{game}]PL[B]AB[ba][bb][bc][bd][ab]AW[aa]C[BONUS 1];B[be])')
    book = tmp_path / 'book.katabook'
    run(cfg, book, iterations=0, extra=['-bonus-file', sgf, '-trace-sgf-file', sgf])
    header, nodes = read_book(book)
    assert header['initialTurnNumber'] == 6
    assert len(nodes) == 1 and not nodes[0]['mvs']


@pytest.fixture(scope='module')
def books(tmp_path_factory):
    return create_books(tmp_path_factory.mktemp('genbook-fixtures') / 'books')


def root_only(source, target):
    """Keep the native fixture's initial state/hash, removing all book edges."""
    lines = source.read_text().splitlines()
    root = json.loads(lines[int(lines[1])+2])
    root.update(mvs=[], par=[], cEx=True)
    target.write_text('\n'.join([lines[0], '1', lines[2], json.dumps(root)]) + '\n')


@pytest.mark.parametrize('game', (*GAMES[1:5], 'xiangqi'))
@pytest.mark.parametrize('muzero', [False, True])
@cuda
def test_trace_terminal_book_uses_real_adjudication(books, tmp_path, game, muzero):
    _, model = model_file(tmp_path / 'model', game, muzero)
    cfg = configure(tmp_path / 'book.cfg', game, muzero, True, random=False)
    source = books / ('xiangqi-history.katabook' if game == 'xiangqi' else f'{game}-terminal.katabook')
    book = tmp_path / 'book.katabook'
    root_only(source, book)
    run(cfg, book, model=model, iterations=0,
        extra=['-allow-changing-book-params', '-trace-book-file', source])
    header, nodes = read_book(book)
    terminal = [node for node in nodes if not node['cEx']]
    assert terminal and all(node['wl'] == (0 if game == 'xiangqi' else -1) and
                           node['maxP'] == 1 and node['v'] == 4 for node in terminal)
    if game == 'xiangqi':
        assert header['xiangqiHistory'] == ['b0c2', 'b9c7', 'c2b0', 'c7b9']
        assert header['initialTurnNumber'] == 97


@pytest.mark.parametrize('muzero', [False, True])
@cuda
def test_xiangqi_full_history_bonus_and_resume_boundary(tmp_path, muzero):
    _, model = model_file(tmp_path / 'model', 'xiangqi', muzero)
    cfg = configure(tmp_path / 'book.cfg', 'xiangqi', muzero, True, random=False)
    fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
    prior = ['b0c2', 'b9c7', 'c2b0', 'c7b9', 'b0c2', 'b9c7', 'c2b0']
    sgf = tmp_path / 'cycle.sgf'
    sgf.write_text(f'(;GM[7]SZ[9:10]RU[xiangqi]XQFEN[{fen}]XQHIST' +
                   ''.join(f'[{move}]' for move in prior) + 'XQTURN[97]C[BONUS 2];W[a6a5])')
    book = tmp_path / 'book.katabook'
    run(cfg, book, model=model, iterations=0, extra=['-bonus-file', sgf, '-trace-sgf-file', sgf])
    header, nodes = read_book(book)
    assert header['xiangqiHistory'] == prior and header['initialTurnNumber'] == 97
    assert len(nodes) >= 2 and nodes[0]['mvs']
    # Same occupancy/player at a different point of the cycle is not the same
    # initial state. Reject it before mutating an existing book.
    original = book.read_bytes()
    sgf.write_text(sgf.read_text().replace(''.join(f'[{move}]' for move in prior),
                                         ''.join(f'[{move}]' for move in prior[:3])))
    text = run(cfg, book, model=model, iterations=0, extra=['-bonus-file', sgf], success=False)
    assert 'different complete Xiangqi histories' in text
    assert book.read_bytes() == original


@pytest.mark.parametrize('primary_mu,policy_mu', [(False, False), (False, True), (True, False), (True, True)])
@cuda
def test_secondary_model_supplies_source_policy_costs(tmp_path, primary_mu, policy_mu):
    _, primary = model_file(tmp_path / 'primary', 'hex', primary_mu, uniform=False)
    _, policy = model_file(tmp_path / 'policy', 'hex', policy_mu)
    cfg = configure(tmp_path / 'book.cfg', 'hex', primary_mu, random=False)
    # Let the ordinary evaluator use the actual 5x5 book and MuZero its fixed
    # 7x7 canvas. Raw policies must be indexed using the model that produced them.
    cfg.write_text(re.sub(r'^maxBoardSizeForNNBuffer = .*\n', '', cfg.read_text(), flags=re.M))
    book = tmp_path / 'book.katabook'
    run(cfg, book, model=primary, human=policy, iterations=1)
    _, nodes = read_book(book)
    assert nodes[0]['mvs']
    assert all(move['rP'] == pytest.approx(1/25, abs=3e-6) for move in nodes[0]['mvs'])


@pytest.mark.parametrize('muzero', [False, True])
@cuda
def test_metadata_boundary(tmp_path, muzero):
    _, model = model_file(tmp_path / 'model', 'go', muzero, metadata=True)
    cfg = configure(tmp_path / 'book.cfg', 'go', muzero, random=False)
    book = tmp_path / 'book.katabook'
    text = run(cfg, book, model=model, success=False)
    assert 'metadata model requires humanSLProfile' in text and not book.exists()
    configure(cfg, 'go', muzero, random=False, humanSLProfile='rank_1d')
    run(cfg, book, model=model, iterations=1)


@cuda
def test_muzero_model_and_search_boundaries(tmp_path):
    _, model = model_file(tmp_path / 'model', 'hex', True)
    book = tmp_path / 'book.katabook'
    for game, changes, expected in [
        ('renju', {}, 'model and book game differ'),
        ('hex', dict(boardSizeX=9, boardSizeY=9), 'canvas is smaller'),
        ('hex', dict(useGraphSearch='true'), 'graph'),
        ('hex', dict(subtreeValueBiasFactor=.4), 'subtreeValueBias'),
    ]:
        cfg = configure(tmp_path / 'book.cfg', game, True, random=False, **changes)
        text = run(cfg, book, model=model, success=False)
        assert expected.lower() in text.lower() and not book.exists()
    cfg = configure(tmp_path / 'book.cfg', 'hex', True, random=False, useMuZero='false', useMuZero0='true')
    run(cfg, book, model=model, iterations=1)


def test_trace_game_mismatch_preserves_book(tmp_path):
    cfg = configure(tmp_path / 'book.cfg', 'hex')
    book = tmp_path / 'book.katabook'
    run(cfg, book)
    original = book.read_bytes()
    sgf = tmp_path / 'wrong.sgf'; sgf.write_text(record('renju'))
    text = run(cfg, book, iterations=0, extra=['-trace-sgf-file', sgf], success=False)
    assert 'game does not match book rules' in text and book.read_bytes() == original


@pytest.mark.parametrize('game', GAMES)
def test_generation_exports_native_html(tmp_path, game):
    cfg = configure(tmp_path / 'book.cfg', game)
    book = tmp_path / 'book.katabook'
    text = run(cfg, book, iterations=1, extra=['-html-dir', tmp_path / 'html'])
    assert 'Done exporting' in text
    assert (tmp_path / 'html/root/root.html').is_file()
