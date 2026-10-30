"""Native opening suggestions and atomic, cancellable Web session creation."""
from concurrent.futures import ThreadPoolExecutor
import json
import time
from threading import Event, Thread

import pytest
from playwright.sync_api import sync_playwright

from etazero.gtp import GtpProcess, GtpError
from etazero.web_session import Session, position
from etazero.web_app import App, Conflict
from web.native_server import make_server
from test_benchmarknn import cuda, model_for
from test_controller_recovery import wait_until
from test_web_native import profile, report_command
from test_web_session import catalog_for, submit, idle, wait_for
from test_web_browser import ready, open_game


@pytest.mark.parametrize('game', ['go', 'xiangqi', 'renju', 'hex'])
def test_invalid_opening_request_keeps_gtp_usable(tmp_path, game):
    config = profile(tmp_path / 'gtp.cfg', game)
    with GtpProcess(config, '/dev/null') as engine:
        before = engine.state()
        assert engine.command('known_command kata-suggest-opening').body == 'true'
        for options in ('[]', '{"unknown":1}', '{"balanceOpeningDropPow":true}',
                        '{"minAcceptRate":0}', '{"balanceOpeningDropPow":-1}', '{}'):
            with pytest.raises(GtpError):
                engine.command('kata-suggest-opening ' + options)
            assert engine.state() == before


def test_invalid_opening_creation_retains_current_game(tmp_path):
    config = profile(tmp_path / 'gtp.cfg', 'hex')
    with App(lambda: catalog_for(config, 'hex')) as app:
        submit(app, 'new', model='model', mode='manual')
        idle(app)
        submit(app, 'play', move='C3')
        before = idle(app)
        original = app.session
        for options in ({'minAcceptRate': 0}, {'unknown': 1}, {}):
            submit(app, 'new', model='model', opening=options)
            failed = wait_for(app, lambda s: not s['busy'] and s['error'] is not None)
            assert app.session is original and app.preparing is None
            assert failed['game_id'] == before['game_id'] and failed['game'] == before['game']
            assert failed['history'] == before['history'] and failed['connected']
        for options in (dict(sgf='(;GM[11]SZ[5])'), dict(fen='anything'), dict(sgf_turn=0)):
            with pytest.raises(ValueError, match='cannot be combined'):
                submit(app, 'new', model='model', opening={}, **options)
        submit(app, 'undo')
        assert idle(app)['game']['moves'] == []


@cuda
@pytest.mark.parametrize('game,muzero,gumbel', [
    (game, muzero, gumbel) for game in ('renju', 'hex')
    for muzero, gumbel in ((False, False), (False, True), (True, False), (True, True))
] + [('freestyle', False, False), ('standard', False, False)])
def test_native_suggestion_preserves_search_and_session_replays_prefix(tmp_path, game, muzero, gumbel):
    model = model_for(tmp_path / 'model', game, muzero)
    config = profile(tmp_path / 'gtp.cfg', game, random=False, muzero=muzero, gumbel=gumbel,
                     cudaUseFP16='true', maxVisits=8, balanceOpeningDropPow=3)
    with Session(config, model) as session:
        before = session.snapshot()
        session.engine.command(report_command())
        search = session.engine.analysis()
        suggestion = json.loads(session.engine.command('kata-suggest-opening').body)
        assert suggestion['status'] == 'complete' and suggestion['game'] == game
        assert suggestion['parameters']['balanceOpeningDropPow'] == 3
        assert session.engine.state() == before['game']
        assert session.engine.analysis() == search  # NN requests did not replace the searched tree.
        assert session.generate_opening({'balanceOpeningDropPow': 2})
        result = session.snapshot()
        assert result['opening']['parameters']['balanceOpeningDropPow'] == 2
        assert result['opening']['moves'] == result['game']['moves']
        assert result['history'][0] == position(before['game'])
        assert len(result['history']) == len(result['game']['moves']) + 1
        assert not result['game']['isGameFinished']
        if game == 'hex': assert len(result['game']['moves']) == 1
        with pytest.raises(GtpError, match='empty ongoing board'):
            session.engine.command('kata-suggest-opening')
        assert session.engine.state() == result['game']
        session.search(play=True)
        session.undo()
        assert session.snapshot()['game'] == result['game']
        for expected in result['history'][-2::-1]:
            session.undo()
            assert position(session.snapshot()['game']) == expected
        assert session.snapshot()['game'] == before['game']


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_native_cancel_drains_response_without_modifying_board(tmp_path, muzero):
    model = model_for(tmp_path / 'model', 'hex', muzero)
    config = profile(tmp_path / 'gtp.cfg', 'hex', random=False, muzero=muzero)
    with GtpProcess(config, model) as engine, ThreadPoolExecutor(1) as executor:
        before = engine.state()
        engine.timeout = .05
        task = executor.submit(engine.command,
            'kata-suggest-opening {"balanceOpeningDropPow":1e300,"minAcceptRate":1e-300}', unlimited_wait=True)
        wait_until(lambda: engine.active_id is not None)
        time.sleep(.2)
        assert not task.done() and engine.process.poll() is None
        engine.stop()
        result = json.loads(task.result(timeout=15).body)
        assert result['status'] == 'cancelled' and result['moves'] == []
        engine.timeout = 120
        assert engine.state() == before
        cancelled = Event(); cancelled.set()
        queued = engine.command('kata-suggest-opening {"balanceOpeningDropPow":1e300,"minAcceptRate":1e-300}',
                                should_stop=cancelled.is_set, unlimited_wait=True)
        assert json.loads(queued.body)['status'] == 'cancelled'
        assert json.loads(engine.command('kata-suggest-opening {"minAcceptRate":1}').body)['status'] == 'complete'
        # A client disconnect must join the same native task before destroying
        # its model service, even without an explicit stop/newline command.
        disconnected = executor.submit(engine.command,
            'kata-suggest-opening {"balanceOpeningDropPow":1e300,"minAcceptRate":1e-300}', unlimited_wait=True)
        wait_until(lambda: engine.active_id is not None)
        with engine.write_lock:
            engine.process.stdin.close()
        assert json.loads(disconnected.result(timeout=15).body)['status'] == 'cancelled'
        assert engine.process.wait(timeout=15) == 0


@cuda
@pytest.mark.parametrize('existing,close', [(False, False), (True, False), (True, True)])
def test_cancel_prepared_session_never_replaces_existing_game(tmp_path, existing, close):
    model = model_for(tmp_path / 'model', 'hex', True)
    config = profile(tmp_path / 'gtp.cfg', 'hex', random=False, muzero=True)
    with App(lambda: catalog_for(config, 'hex', model)) as app:
        if existing:
            submit(app, 'new', model='model', mode='manual')
            idle(app)
            submit(app, 'play', move='C3')
            idle(app)
        original = app.session
        before = app.snapshot()
        submit(app, 'new', model='model', mode='manual',
               opening=dict(balanceOpeningDropPow=1e300, minAcceptRate=1e-300))
        wait_for(app, lambda s: s['phase'] == 'opening')
        candidate = app.preparing
        wait_until(lambda: candidate.engine.active_id is not None)
        assert candidate.engine.process.poll() is None
        with pytest.raises(Conflict): submit(app, 'new', model='model')
        if close:
            app.close()
            assert original.engine.process.poll() is not None
        else:
            submit(app, 'stop')
            after = idle(app)
            assert app.session is original and app.preparing is None
            assert after['game_id'] == before['game_id'] and after['game'] == before['game']
            assert after['history'] == before['history'] and after['selected_model'] == before['selected_model']
            # A new successful request still replaces the old game and accepts
            # ordinary manual/search/history operations after cancellation.
            submit(app, 'new', model='model', mode='manual', opening={'minAcceptRate': 1})
            done = idle(app)
            assert done['opening']['moves'] == done['game']['moves'] and len(done['game']['moves']) == 1
            assert done['game_id'] != before['game_id']
        assert candidate.engine.process.poll() == 0


@cuda
@pytest.mark.parametrize('game,muzero', [('renju', False), ('hex', True)])
def test_browser_native_opening_history_failure_and_cancel(tmp_path, game, muzero):
    model = model_for(tmp_path / 'model', game, muzero)
    config = profile(tmp_path / 'gtp.cfg', game, random=False, muzero=muzero, gumbel=True, cudaUseFP16='true')
    with App(lambda: catalog_for(config, game, model)) as app:
        server = make_server(app, '127.0.0.1', 0)
        thread = Thread(target=server.serve_forever); thread.start()
        try:
            with sync_playwright() as api:
                browser = api.chromium.launch(headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 1100})
                errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
                open_game(page, f'http://127.0.0.1:{server.server_port}')
                page.locator('#opening-panel summary').click()
                page.check('#balanced-opening')
                page.fill('#opening-options', '{"balanceOpeningDropPow":2}')
                old = app.snapshot()['game_id']
                page.click('#new-game')
                ready(page, f'state.game_id !== {json.dumps(old)} && !state.busy')
                current = app.snapshot()
                assert page.locator('#opening-info').is_visible()
                assert page.evaluate('state.opening.moves') == current['game']['moves']
                assert page.evaluate('displayedGame().board') == current['game']['board']
                page.screenshot(path=str(tmp_path / 'opening.png'), full_page=True)
                page.click('#first')
                assert page.evaluate('displayedGame().board') == current['history'][0]['board']
                page.click('#branch'); ready(page, 'state.game.moves.length === 0 && !state.busy')
                if game == 'hex':
                    page.fill('#opening-options', '{"balanceOpeningDropPow":1e300,"minAcceptRate":1e-300}')
                    page.click('#new-game'); ready(page, 'state.phase === "opening"')
                    page.click('#stop'); ready(page)
                    assert app.snapshot()['game_id'] == current['game_id']
                    assert page.evaluate('state.game.moves.length') == 0
                page.fill('#opening-options', '{"unknown":1}')
                page.click('#new-game')
                page.wait_for_function('state && !state.busy && state.error')
                assert page.locator('#error').is_visible()
                assert app.snapshot()['game_id'] == current['game_id']
                assert not errors, errors
                browser.close()
        finally:
            app.close(); server.shutdown(); server.server_close(); thread.join(timeout=5)
