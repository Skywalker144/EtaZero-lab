"""Workbench actions exercise native games, search, history and cancellation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Event

import pytest

from etazero.web_app import App, Conflict
from etazero.web_session import Session
from test_web_native import profile
from test_xiangqi_gtp import CHECK_FEN, CHECKS
from test_benchmarknn import cuda, model_for


GAMES = ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi']


def catalog_for(config, game, model='/dev/null', *, key='model'):
    return dict(runs=[dict(id='run', label='Run', config=str(config), game=game,
                          models=[dict(id=key, path=str(model), game=game)])], errors=[])


def submit(app, operation, **payload):
    state = app.snapshot()
    return app.submit(operation, dict(version=state['version'], game_id=state['game_id'], **payload))


def wait_for(app, predicate, timeout=20):
    with app.condition:
        assert app.condition.wait_for(lambda: predicate(app.state), timeout=timeout), app.state
        return deepcopy(app.state)


def idle(app):
    state = wait_for(app, lambda s: not s['busy'] and s['phase'] in ('idle', 'error'))
    assert state['error'] is None, state['error']
    return state


def analyzed(app):
    state = wait_for(app, lambda s: s['phase'] == 'error' or
                     (s['phase'] == 'analyzing' and s['analysis'] is not None and
                      s['analysis']['status'] == 'running' and s['analysis']['report'] is not None and
                      s['analysis']['position']['turnNumber'] == s['game']['turnNumber']))
    assert state['error'] is None, state['error']
    return state


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('gumbel', [False, True])
def test_session_native_reports_play_and_undo(tmp_path, game, gumbel):
    config = profile(tmp_path / 'gtp.cfg', game, gumbel=gumbel)
    with Session(config, '/dev/null') as session:
        initial = session.snapshot()
        assert initial['parameters'] == json.loads(session.engine.command('kata-get-params').body)
        assert initial['game'] == session.engine.state()
        assert len(initial['history']) == 1 and initial['connected']
        move = 'b0c2' if game == 'xiangqi' else 'C3'
        session.play(move)
        before_search = session.snapshot()['game']
        reports = []
        result = session.search(on_report=reports.append)
        searched = session.snapshot()
        assert searched['game'] == before_search == session.engine.state()
        assert searched['analysis']['report'] == session.engine.analysis() == reports[-1]
        assert searched['analysis']['status'] == 'complete' and result['move'] in before_search['legalMoves']
        assert not result['stopped']
        session.search(play=True)
        played = session.snapshot()
        assert len(played['game']['moves']) == 2 and len(played['history']) == 3
        assert played['analysis']['ply'] == 1  # The report belongs to the pre-move board.
        assert played['analysis']['position']['board'] == before_search['board']
        assert all('moves' not in p and 'legalMoves' not in p for p in played['history'])
        session.undo(2)
        undone = session.snapshot()
        assert undone['game'] == initial['game'] and undone['history'] == initial['history']
        assert undone['analysis'] is None
        # Rejected input cannot alter the native board or its displayed history.
        with pytest.raises(ValueError):
            session.play('C3\nquit')
        with pytest.raises(ValueError):
            session.undo(1)
        assert session.snapshot() == undone


def test_captured_boards_and_full_xiangqi_rule_history(tmp_path):
    config = profile(tmp_path / 'go.cfg', 'go')
    with Session(config, '/dev/null') as session:
        for move in ['B3', 'C3', 'D3', 'A1', 'C2', 'A2', 'C4']:
            session.play(move)
        capture = session.snapshot()
        assert capture['game']['board'][12] == 0 and capture['history'][-2]['board'][12] == -1
        session.undo()
        assert session.snapshot()['game']['board'][12] == -1
    config = profile(tmp_path / 'xiangqi.cfg', 'xiangqi')
    with Session(config, '/dev/null', fen=CHECK_FEN) as session:
        for move in CHECKS:
            session.play(move)
        end = session.snapshot()
        assert end['game']['result'] == 'W+' and end['game']['isGameFinished']
        with pytest.raises(ValueError, match='finished'):
            session.search()
        session.undo()
        assert not session.snapshot()['game']['isGameFinished']
        session.play(CHECKS[-1])
        assert session.snapshot()['game'] == end['game']


@pytest.mark.parametrize('perspective,expected', [('BLACK', 'B'), ('WHITE', 'W'), ('SIDETOMOVE', 'W')])
def test_analysis_perspective_comes_from_native_config(tmp_path, perspective, expected):
    config = profile(tmp_path / 'gtp.cfg', 'hex', reportAnalysisWinratesAs=perspective)
    with Session(config, '/dev/null') as session:
        session.play('C3')
        session.search()
        state = session.snapshot()
        assert state['game']['currentPlayer'] == 'W'
        assert state['analysis']['position']['analysisPerspective'] == expected
        assert state['analysis']['report'] == session.engine.analysis()


@pytest.mark.parametrize('continuous', [False, True])
@pytest.mark.parametrize('gumbel', [False, True])
def test_session_stop_prevents_partial_move_and_keeps_process_usable(tmp_path, continuous, gumbel):
    config = profile(tmp_path / 'gtp.cfg', 'hex', gumbel=gumbel, maxVisits=1000000)
    with Session(config, '/dev/null') as session, ThreadPoolExecutor(1) as executor:
        before = session.snapshot()['game']
        ready = Event()
        task = executor.submit(session.search, continuous=continuous, play=not continuous,
                               on_report=lambda report: ready.set())
        assert ready.wait(15) and not task.done()
        session.stop()
        assert task.result(timeout=15) == dict(move=None, stopped=True)
        assert session.snapshot()['game'] == before
        assert session.snapshot()['analysis']['status'] == 'stopped'
        assert session.snapshot()['analysis']['report'] == session.engine.analysis()
        session.play('A1')
        session.undo()
        assert session.snapshot()['game'] == before
        # An action cancelled while queued must not miss the interruption just
        # because GTP had no command active when stop() was first requested.
        cancel = Event()
        cancel.set()
        assert session.search(play=True, cancelled=cancel)['stopped']
        assert session.snapshot()['game'] == before


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('gumbel', [False, True])
def test_controller_manual_human_branch_config_and_model_switch(tmp_path, game, gumbel):
    config = profile(tmp_path / 'gtp.cfg', game, gumbel=gumbel, maxVisits=8)
    catalog = catalog_for(config, game)
    with App(lambda: deepcopy(catalog)) as app:
        submit(app, 'new', model='model', mode='manual')
        start = idle(app)
        old = app.session
        assert start['game'] == old.engine.state()
        first = 'b0c2' if game == 'xiangqi' else 'C3'
        submit(app, 'play', move=first)
        assert len(idle(app)['history']) == 2
        submit(app, 'step')
        assert len(idle(app)['game']['moves']) == 2
        submit(app, 'branch', turn=1)
        state = idle(app)
        assert state['game']['moves'] == [['B', first]] and state['mode'] == 'manual'
        submit(app, 'configure', parameters={'maxVisits': 12}, mode='play', human='B')
        configured = idle(app)
        assert int(configured['parameters']['maxVisits']) == 12
        assert configured['game']['moves'] == [['B', first]]  # Configure never autoplays.
        submit(app, 'retry')
        assert len(idle(app)['game']['moves']) == 2
        submit(app, 'undo')
        assert idle(app)['game'] == start['game']  # Undo human move and its AI reply.
        with pytest.raises(Conflict):
            app.submit('play', dict(version=start['version'], game_id=start['game_id'], move=first))
        # Same model can create a fresh game without clearing/replaying the old
        # process before the replacement has loaded and accepted its setup.
        submit(app, 'new', model='model', mode='manual')
        fresh = idle(app)
        assert fresh['game_id'] != start['game_id'] and old.engine.process.poll() is not None
        assert fresh['game']['moves'] == []
        with pytest.raises(Conflict):
            app.submit('play', dict(version=fresh['version'], game_id=start['game_id'], move=first))
    assert app.session.engine.process.poll() is not None


@pytest.mark.parametrize('gumbel', [False, True])
def test_controller_mutation_during_analysis_cancels_and_restarts_on_new_root(tmp_path, gumbel):
    config = profile(tmp_path / 'gtp.cfg', 'hex', gumbel=gumbel, maxVisits=1000000)
    catalog = catalog_for(config, 'hex')
    with App(lambda: catalog) as app:
        submit(app, 'new', model='model', mode='manual')
        idle(app)
        submit(app, 'analyze')
        first = analyzed(app)
        # Report-only versions do not make a move stale.
        newer = wait_for(app, lambda s: s['version'] > first['version'])
        assert newer['phase'] == 'analyzing'
        app.submit('play', dict(version=first['version'], game_id=first['game_id'], move='C3'))
        second = analyzed(app)
        assert second['game']['moves'] == [['B', 'C3']]
        assert second['analysis']['ply'] == 1 and second['analysis']['position']['board'][12] == 1
        submit(app, 'undo')
        third = analyzed(app)
        assert third['game']['moves'] == [] and third['analysis']['ply'] == 0
        submit(app, 'stop')
        stopped = idle(app)
        assert not stopped['analysis_enabled'] and stopped['analysis']['status'] == 'stopped'
        assert stopped['game']['moves'] == []
        submit(app, 'configure', parameters={'maxVisits': 8})
        idle(app)
        submit(app, 'step')
        assert len(idle(app)['game']['moves']) == 1


def test_failed_setup_and_params_keep_existing_game_refresh_does_not_reload(tmp_path):
    config = profile(tmp_path / 'gtp.cfg', 'renju', gumbel=True)
    catalog = catalog_for(config, 'renju')
    with App(lambda: deepcopy(catalog)) as app:
        submit(app, 'new', model='model', mode='manual')
        idle(app)
        submit(app, 'play', move='C3')
        before = idle(app)
        old = app.session
        with pytest.raises(ValueError, match='opening must be a native opening options object'):
            submit(app, 'new', model='model', opening='balanced')
        assert app.snapshot() == before
        submit(app, 'new', model='model', size=[4, 5])
        failed = wait_for(app, lambda s: s['phase'] == 'error' and not s['busy'])
        assert failed['game'] == before['game'] and failed['game_id'] == before['game_id']
        assert app.session is old and failed['connected']
        submit(app, 'configure', parameters={'useGraphSearch': True})
        rejected = wait_for(app, lambda s: s['phase'] == 'error' and not s['busy'])
        assert rejected['game'] == before['game'] and rejected['parameters'] == before['parameters']
        catalog['runs'] = []
        catalog['errors'] = [dict(run='run', error='missing committed model')]
        submit(app, 'refresh')
        refreshed = idle(app)
        assert refreshed['game'] == before['game'] and app.session is old
        assert refreshed['catalog_revision'] == 1
        assert {key: app.catalog()[key] for key in catalog} == catalog
        submit(app, 'undo')
        assert idle(app)['game']['moves'] == []


@pytest.mark.parametrize('gumbel', [False, True])
def test_controller_cancelled_ai_turn_can_be_retried(tmp_path, gumbel):
    config = profile(tmp_path / 'gtp.cfg', 'hex', gumbel=gumbel, maxVisits=1000000)
    with App(lambda: catalog_for(config, 'hex')) as app:
        submit(app, 'new', model='model', human='W')
        running = wait_for(app, lambda s: s['phase'] == 'thinking' and s['analysis'] is not None and
                           s['analysis']['report'] is not None)
        assert running['game']['moves'] == []
        submit(app, 'stop')
        stopped = idle(app)
        assert stopped['game']['moves'] == [] and stopped['analysis']['status'] == 'stopped'
        submit(app, 'configure', parameters={'maxVisits': 8})
        assert idle(app)['game']['moves'] == []
        submit(app, 'retry')
        assert len(idle(app)['game']['moves']) == 1


def test_native_disconnect_is_visible_and_new_session_recovers(tmp_path):
    config = profile(tmp_path / 'gtp.cfg', 'renju', maxVisits=1000000)
    with App(lambda: catalog_for(config, 'renju')) as app:
        submit(app, 'new', model='model', mode='manual')
        before = idle(app)
        submit(app, 'analyze')
        analyzed(app)
        app.session.engine.process.kill()
        failed = wait_for(app, lambda s: not s['busy'] and s['phase'] == 'error')
        assert not failed['connected'] and not failed['analysis_enabled']
        assert failed['game'] == before['game'] and failed['analysis']['status'] == 'error'
        with pytest.raises(Conflict, match='disconnected'):
            submit(app, 'step')
        submit(app, 'new', model='model', parameters={'maxVisits': 8})
        restored = idle(app)
        assert restored['connected'] and restored['game_id'] != before['game_id']
        submit(app, 'step')
        assert len(idle(app)['game']['moves']) == 1


def test_controller_close_cancels_live_search_and_wakes_poll(tmp_path):
    config = profile(tmp_path / 'gtp.cfg', 'renju', maxVisits=1000000)
    app = App(lambda: catalog_for(config, 'renju'))
    try:
        submit(app, 'new', model='model', mode='manual')
        idle(app)
        submit(app, 'analyze')
        state = analyzed(app)
        assert app.session.engine.process.poll() is None
        with ThreadPoolExecutor(1) as executor:
            poll = executor.submit(app.snapshot, state['version'])
            app.close()
            poll.result(timeout=10)
        assert app.session.engine.process.poll() is not None
        with pytest.raises(Conflict):
            submit(app, 'new', model='model')
    finally:
        app.close()


@cuda
@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_real_cuda_models_through_controller(tmp_path, game, muzero, gumbel):
    model = model_for(tmp_path / 'model', game, muzero)
    config = profile(tmp_path / 'gtp.cfg', game, muzero=muzero, random=False, gumbel=gumbel,
                     cudaUseFP16='true', maxVisits=8, maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5)
    with App(lambda: catalog_for(config, game, model)) as app:
        submit(app, 'new', model='model', human='W', mode='play')
        first = idle(app)
        assert first['connected'] and first['models']
        assert first['game']['moves'][0][0] == 'B'
        assert first['game']['currentPlayer'] == 'W'
        report = first['analysis']['report']
        assert report['rootInfo']['visits'] > 0
        assert sum(report['rootInfo'][k] for k in ('winValue', 'drawValue', 'lossValue', 'noResultValue')) == pytest.approx(1)
        submit(app, 'play', move=first['game']['legalMoves'][0])
        played = idle(app)
        assert len(played['game']['moves']) == 3 and played['analysis']['ply'] == 2
        submit(app, 'undo')
        undone = idle(app)
        assert undone['game'] == first['game'] and undone['history'] == first['history']


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_native_resignation_is_separate_from_rule_terminal_and_undoable(tmp_path, muzero):
    import torch
    from export_model_pytorch import write_model_binary
    from export_muzero import export_components
    from katago.train.model_pytorch import Model
    from test_muzero_model import make_model, component_config
    model = make_model('renju') if muzero else Model(component_config('renju'), 5)
    if not muzero:
        model.initialize()
    value = model.prediction.value_head if muzero else model.value_head
    with torch.no_grad():
        value.linear_valuehead.weight.zero_()
        value.linear_valuehead.bias.copy_(torch.tensor([-8., -8., 8., -8.]))
    model.eval()
    path = tmp_path / ('bundle' if muzero else 'model.bin')
    if muzero:
        export_components(model, path, 'web-resign')
    else:
        with path.open('wb') as stream:
            write_model_binary(model, stream, 'web-resign')
    config = profile(tmp_path / 'gtp.cfg', 'renju', muzero=muzero, random=False,
                     maxVisits=1, numSearchThreads=1, allowResignation='true', resignThreshold=-.5,
                     resignConsecTurns=1, resignMinMovesPerBoardArea=0, cudaUseFP16='true')
    with App(lambda: catalog_for(config, 'renju', path)) as app:
        submit(app, 'new', model='model', human='W')
        resigned = idle(app)
        assert resigned['resignation'] == dict(player='B', result='W+R', source='native GTP search response')
        assert not resigned['game']['isGameFinished'] and resigned['game']['result'] is None
        assert resigned['game']['moves'] == [] and len(resigned['history']) == 1
        with pytest.raises(Conflict):
            submit(app, 'step')
        submit(app, 'undo')
        undone = idle(app)
        assert undone['resignation'] is None and undone['game'] == resigned['game']
        submit(app, 'configure', mode='manual', parameters={'allowResignation': False})
        idle(app)
        submit(app, 'step')
        assert len(idle(app)['game']['moves']) == 1


@cuda
@pytest.mark.parametrize('game', ['renju', 'xiangqi'])
def test_live_model_switch_ordinary_to_muzero(tmp_path, game):
    ordinary = model_for(tmp_path / 'ordinary', game, False)
    muzero = model_for(tmp_path / 'muzero', game, True)
    first = profile(tmp_path / 'first.cfg', game, random=False, maxVisits=8)
    second = profile(tmp_path / 'second.cfg', game, random=False, muzero=True, gumbel=True,
                     maxVisits=8, cudaUseFP16='true')
    catalog = catalog_for(first, game, ordinary, key='ordinary')
    catalog['runs'] += catalog_for(second, game, muzero, key='muzero')['runs']
    with App(lambda: catalog) as app:
        submit(app, 'new', model='ordinary', human='W')
        old_game = idle(app)
        previous = app.session
        submit(app, 'new', model='muzero')
        new_game = idle(app)
        assert new_game['game_id'] != old_game['game_id']
        assert new_game['selected_model']['id'] == 'muzero' and new_game['config'] == str(second)
        assert new_game['parameters']['useGumbel'] is True
        assert len(new_game['game']['moves']) == 1 and new_game['analysis']['report']['rootInfo']['visits'] > 0
        assert previous.engine.process.poll() is not None
