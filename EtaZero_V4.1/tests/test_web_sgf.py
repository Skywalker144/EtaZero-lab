"""SGF import delegates parsing and history to native GTP, including Xiangqi prehistory."""
import json
from threading import Thread

import pytest
from playwright.sync_api import sync_playwright

from etazero.gtp import GtpProcess
from etazero.web_app import App
from etazero.web_session import Session, position
from web.native_server import make_server
from test_benchmarknn import cuda, model_for
from test_web_browser import ready, open_game
from test_web_native import profile
from test_web_session import GAMES, catalog_for, submit, idle, wait_for
from test_xiangqi_gtp import CHECK_FEN, CHECKS


def record(game):
    if game == 'xiangqi':
        return (f'(;GM[7]SZ[9:10]XQFEN[{CHECK_FEN}]XQHIST[e8f8][f9e9]XQTURN[97]'
                + ''.join(f';{"B" if i % 2 == 0 else "W"}[{move}]' for i, move in enumerate(CHECKS[2:7])) + ')')
    gm = 1 if game == 'go' else 11 if game == 'hex' else 4
    rules = 'Tromp-Taylor' if game == 'go' else game
    # Last B captures the central W in Go, but retains it in placement games.
    return f'(;GM[{gm}]SZ[5]RU[{rules}]KM[0];B[bc];W[cc];B[dc];W[ae];B[cd];W[ad];B[cb])'


@pytest.mark.parametrize('game', GAMES)
def test_imported_history_matches_native_prefixes_and_undo(tmp_path, game):
    config = profile(tmp_path / 'gtp.cfg', game)
    sgf = record(game)
    source = tmp_path / 'source.sgf'
    source.write_text(sgf)
    with Session(config, '/dev/null', sgf=sgf) as session, GtpProcess(config, '/dev/null') as reference:
        state = session.snapshot()
        count = len(state['game']['moves'])
        assert len(state['history']) == count + 1
        for turn, expected in enumerate(state['history']):
            reference.command(f'loadsgf {source} {turn + 1}')
            assert expected == position(reference.state())
        assert state['game'] == reference.state()
        assert state['history'][-1] == position(state['game'])
        if game == 'xiangqi':
            assert state['history'][0]['turnNumber'] == 97
            session.play(CHECKS[-1])
            assert session.snapshot()['game']['result'] == 'W+'
            session.undo()
            assert session.snapshot()['game'] == state['game']
        else:
            assert state['game']['board'][12] == (0 if game == 'go' else -1)
        session.undo(count)
        assert position(session.snapshot()['game']) == state['history'][0]
        assert len(session.snapshot()['history']) == 1


def test_history_query_preserves_current_search_and_handles_tolerant_sgf(tmp_path):
    config = profile(tmp_path / 'gtp.cfg', 'go')
    # Setup stones and consecutive black moves exercise the source SGF reader,
    # rather than Python assuming empty, alternating play.
    sgf = '(;GM[1]SZ[5]RU[Tromp-Taylor]AB[aa][bb]PL[B];B[cc];B[dd];W[ee])'
    source = tmp_path / 'setup.sgf'
    source.write_text(sgf)
    with Session(config, '/dev/null', sgf=sgf) as session, GtpProcess(config, '/dev/null') as reference:
        session.search()
        current, report = session.engine.state(), session.engine.analysis()
        history = json.loads(session.engine.command('kata-get-history').body)
        assert session.engine.state() == current and session.engine.analysis() == report
        assert history == session.snapshot()['history']
        for turn, expected in enumerate(history):
            reference.command(f'loadsgf {source} {turn + 1}')
            assert expected == position(reference.state())
        session.undo()
        reference.command('undo')
        undone = session.snapshot()
        assert undone['game'] == reference.state()
        assert undone['history'][-1] == position(undone['game'])


def test_imported_native_terminal_is_not_searched_and_can_be_undone(tmp_path):
    config = profile(tmp_path / 'gtp.cfg', 'go')
    with App(lambda: catalog_for(config, 'go')) as app:
        submit(app, 'new', model='model', mode='play', human='W',
               sgf='(;GM[1]SZ[5]RU[Tromp-Taylor]KM[0];B[];W[])')
        end = idle(app)
        assert end['game']['isGameFinished'] and end['analysis'] is None
        assert end['history'][-1] == position(end['game'])
        submit(app, 'undo')
        undone = idle(app)
        assert not undone['game']['isGameFinished'] and len(undone['game']['moves']) == 1


@pytest.mark.parametrize('change', [dict(sgf='not SGF'), dict(sgf=record('xiangqi')),
                                  dict(sgf=record('go'), fen=CHECK_FEN), dict(sgf_turn=2)])
def test_failed_import_preserves_existing_session(tmp_path, change):
    config = profile(tmp_path / 'gtp.cfg', 'go')
    with App(lambda: catalog_for(config, 'go')) as app:
        submit(app, 'new', model='model', mode='manual')
        idle(app)
        submit(app, 'play', move='C3')
        before = idle(app)
        old = app.session
        submit(app, 'new', model='model', **change)
        failed = wait_for(app, lambda s: not s['busy'] and s['phase'] == 'error')
        assert failed['error'] and app.session is old and failed['game_id'] == before['game_id']
        assert failed['game'] == before['game'] and failed['history'] == before['history']
        submit(app, 'undo')
        assert idle(app)['game']['moves'] == []


@cuda
@pytest.mark.parametrize('game', ['go', 'xiangqi'])
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_cuda_sgf_search_retains_native_root_and_history(tmp_path, game, muzero, gumbel):
    model = model_for(tmp_path / 'model', game, muzero)
    config = profile(tmp_path / 'gtp.cfg', game, random=False, muzero=muzero, gumbel=gumbel,
                     cudaUseFP16='true', maxVisits=8)
    with Session(config, model, sgf=record(game), sgf_turn=3) as session:
        before = session.snapshot()
        session.search(play=True)
        searched = session.snapshot()
        assert len(searched['game']['moves']) == 4
        assert searched['analysis']['ply'] == 3
        assert searched['analysis']['position'] == position(before['game'])
        session.undo()
        assert session.snapshot()['game'] == before['game']
        assert session.snapshot()['history'] == before['history']
