"""Outer raw-NN diagnostics use native replies and never perform a search."""
import json
from threading import Thread

import pytest
from playwright.sync_api import sync_playwright

from etazero.gtp import GtpError
from etazero.web_app import App
from etazero.web_session import Session
from etazero.web_raw_nn import parse_raw_nn
from web.native_server import make_server
from test_benchmarknn import cuda, model_for
from test_web_browser import open_game, ready
from test_web_native import profile
from test_web_session import GAMES, catalog_for, submit, idle, analyzed, wait_for


def check_report(session, result):
    state = session.snapshot()['game']
    assert result['position']['board'] == state['board']
    assert result['position']['currentPlayer'] == state['currentPlayer']
    assert result['ply'] == len(state['moves'])
    policy = result['report']['policy']
    assert {row['move'] for row in policy if row['probability'] is not None} == set(state['legalMoves'])
    assert sum(row['probability'] or 0 for row in policy) == pytest.approx(1, abs=0.001)
    assert sum(result['report']['values'][key] for key in ('whiteWin', 'draw', 'whiteLoss', 'noResult')) == pytest.approx(1, abs=2e-6)
    assert len(policy) == (2086 if state['game'] == 'xiangqi' else state['boardXSize'] * state['boardYSize'] + 1)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('gumbel', [False, True])
def test_native_diagnosis_preserves_position_parameters_and_search_tree(tmp_path, game, gumbel):
    config = profile(tmp_path / 'gtp.cfg', game, gumbel=gumbel)
    with Session(config, '/dev/null') as session:
        session.play('b0c2' if game == 'xiangqi' else 'C3')
        session.search()
        before = session.snapshot()
        tree = session.engine.analysis()
        symmetry = 2 if game == 'xiangqi' else 3
        session.diagnose(symmetry=symmetry, optimism=.7)
        result = session.snapshot()['raw_nn']
        assert result['command'] == f'kata-raw-nn {symmetry} 0.7'
        assert result['report']['symmetry'] == symmetry
        check_report(session, result)
        assert session.engine.analysis() == tree
        after = session.snapshot()
        assert {k: v for k, v in after.items() if k != 'raw_nn'} == {k: v for k, v in before.items() if k != 'raw_nn'}
        assert session.engine.state() == before['game']
        session.play(after['game']['legalMoves'][0])
        assert session.snapshot()['raw_nn'] == result  # Explicitly tied to the previous position.
        session.undo()
        assert session.snapshot()['raw_nn'] is None
        session.diagnose()
        session.configure({'maxVisits': 12})
        assert session.snapshot()['raw_nn'] is None


def test_rectangular_policy_coordinates_and_masked_pass(tmp_path):
    config = profile(tmp_path / 'gtp.cfg', 'go')
    with Session(config, '/dev/null', size=[4, 5]) as session:
        session.play('C3')
        session.diagnose()
        report = session.snapshot()['raw_nn']['report']
        check_report(session, session.snapshot()['raw_nn'])
        assert [row['move'] for row in report['policy'][:4]] == ['A5', 'B5', 'C5', 'D5']
        assert report['policy'][10] == dict(move='C3', probability=None)
        assert report['policy'][-1]['move'] == 'pass'
        assert report['policy'][-1]['probability'] is not None


def test_invalid_requests_and_missing_human_model_keep_session_usable(tmp_path):
    config = profile(tmp_path / 'gtp.cfg', 'hex')
    with Session(config, '/dev/null') as session:
        session.diagnose()
        before = session.snapshot()
        for options in [dict(source='bad'), dict(symmetry=True), dict(symmetry='0\nquit'),
                        dict(optimism=float('nan')), dict(optimism=2), dict(source='human', optimism=0)]:
            with pytest.raises(ValueError):
                session.diagnose(**options)
            assert session.snapshot() == before
        for options, message in [(dict(source='human'), 'not provided'), (dict(symmetry=1), 'invalid for the game')]:
            with pytest.raises(GtpError, match=message):
                session.diagnose(**options)
            assert session.snapshot() == before
        session.play('C3')
        session.diagnose()
        check_report(session, session.snapshot()['raw_nn'])


def test_malformed_native_response_is_not_rendered_as_zero_or_logits():
    state = dict(game='go', boardXSize=2, boardYSize=2)
    text = 'symmetry 0\nwhiteWin 0.2\ndraw 0.1\nwhiteLoss 0.6\nnoResult 0.1\npolicy\n0.1 NAN\n0.3 0.2\npolicyPass 0.4'
    assert parse_raw_nn(text, state)['policy'][1] == dict(move='B2', probability=None)
    for invalid in [text.replace('0.1 NAN', '0.1'), text.replace('whiteWin 0.2', 'whiteWin inf'),
                    text.replace('0.1 NAN', '0.1 nan'), text + '\nsymmetry 1']:
        with pytest.raises(ValueError):
            parse_raw_nn(invalid, state)


def test_wide_native_coordinate_labels():
    text = 'symmetry 0\nwhiteWin 1\ndraw 0\nwhiteLoss 0\nnoResult 0\npolicy\n'
    text += '\n'.join([' '.join(['NAN'] * 27)] * 2) + '\npolicyPass 1'
    rows = parse_raw_nn(text, dict(game='go', boardXSize=27, boardYSize=2))['policy']
    assert [row['move'] for row in rows[24:27]] == ['Z2', 'AA2', 'AB2']


@pytest.mark.parametrize('game', ['go', 'hex', 'xiangqi'])
def test_terminal_diagnosis_has_no_fabricated_policy(tmp_path, game):
    from test_xiangqi_gtp import CHECK_FEN, CHECKS
    config = profile(tmp_path / 'gtp.cfg', game)
    with Session(config, '/dev/null', **(dict(fen=CHECK_FEN) if game == 'xiangqi' else {})) as session:
        session.diagnose()
        moves = (CHECKS if game == 'xiangqi' else ['pass', 'pass'] if game == 'go' else
                 ['A1', 'B1', 'A2', 'B2', 'A3', 'B3', 'A4', 'B4', 'A5'])
        for move in moves:
            session.play(move)
        before = session.snapshot()
        assert before['game']['isGameFinished']
        with pytest.raises(ValueError, match='finished'):
            session.diagnose()
        assert session.snapshot() == before


def test_diagnostic_during_analysis_serializes_and_resumes_native_search(tmp_path):
    config = profile(tmp_path / 'gtp.cfg', 'hex', maxVisits=1000000)
    with App(lambda: catalog_for(config, 'hex')) as app:
        submit(app, 'new', model='model', mode='manual'); idle(app)
        submit(app, 'analyze'); analyzed(app)
        submit(app, 'diagnose', symmetry=3, optimism=1)
        state = wait_for(app, lambda s: s['raw_nn'] is not None and s['phase'] == 'analyzing')
        assert state['analysis_enabled'] and not state['game']['moves']
        assert state['raw_nn']['command'] == 'kata-raw-nn 3 1'
        submit(app, 'stop'); state = idle(app)
        check_report(app.session, state['raw_nn'])
        submit(app, 'diagnose', source='human')
        failed = wait_for(app, lambda s: not s['busy'] and s['phase'] == 'error')
        assert 'not provided' in failed['error'] and failed['connected']
        assert failed['raw_nn'] == state['raw_nn']
        submit(app, 'play', move='C3'); idle(app)
        submit(app, 'diagnose'); final = idle(app)
        assert final['raw_nn']['ply'] == 1 and final['analysis']['ply'] == 0


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero', [False, True])
def test_cuda_main_and_human_diagnostics_match_native_text(tmp_path, game, muzero):
    from test_humansl_protocols import model_for as human_model_for
    model = model_for(tmp_path / 'model', game, muzero)
    human = human_model_for(tmp_path / 'human', game, not muzero, metadata=game == 'go')
    config = profile(tmp_path / 'gtp.cfg', game, random=False, muzero=muzero, cudaUseFP16='true',
                     **(dict(humanSLProfile='rank_5d') if game == 'go' else {}))
    with Session(config, model, human_model=human) as session:
        session.play('b0c2' if game == 'xiangqi' else 'C3')
        for options in [dict(optimism=0), dict(optimism=1), dict(source='human')]:
            session.diagnose(**options)
            data = session.snapshot()['raw_nn']
            assert data['command'] == ('kata-raw-human-nn 0' if options.get('source') == 'human' else
                                      f'kata-raw-nn 0 {options["optimism"]}')
            check_report(session, data)
            assert data['text'] == session.engine.command(data['command']).body
            # Independent text oracle: the UI probabilities retain source precision.
            source = data['text'].split('\npolicy\n')[1].replace('policyPass', '').split()
            if game == 'xiangqi':
                assert [r['move'] for r in data['report']['policy']] == source[::2]
                source = source[1::2]
            assert [r['probability'] for r in data['report']['policy']] == [None if v == 'NAN' else float(v) for v in source]


@pytest.mark.parametrize('game', GAMES)
def test_browser_raw_policy_snapshot_and_missing_source(tmp_path, game):
    config = profile(tmp_path / 'gtp.cfg', game)
    with App(lambda: catalog_for(config, game)) as app:
        server = make_server(app, '127.0.0.1', 0)
        thread = Thread(target=server.serve_forever); thread.start()
        try:
            with sync_playwright() as api:
                browser = api.chromium.launch(headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 1100})
                errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
                open_game(page, f'http://127.0.0.1:{server.server_port}')
                page.locator('#nn-panel > summary').click()
                page.click('#diagnose'); ready(page, 'state.raw_nn && !state.busy')
                data = app.snapshot()['raw_nn']
                assert app.snapshot()['analysis'] is None  # No search needed for diagnostics.
                assert page.locator('#nn-text').text_content() == data['text']
                assert 'kata-raw-nn 0 0' in page.locator('#nn-command').inner_text()
                if game == 'xiangqi':
                    page.select_option('[aria-label="网络诊断起点"]', 'b0')
                    selector = '#nn-list .nn-entry'
                    expected = [r for r in data['report']['policy'] if r['move'].startswith('b0')]
                    assert page.locator('#nn-symmetry option').all_text_contents() == ['0', '2']
                else:
                    selector = '#nn-map .heat-cell'
                    expected = data['report']['policy'][:25]
                    assert ('已屏蔽' in page.locator('#nn-list').inner_text()) == (game != 'go')
                assert page.locator(selector).evaluate_all('cells => cells.map(c => c.dataset.move)') == [r['move'] for r in expected]
                titles = page.locator(selector).evaluate_all('cells => cells.map(c => c.title.split(": ")[1])')
                assert [None if v == 'NAN' else float(v) for v in titles] == [r['probability'] for r in expected]
                page.fill('#nn-optimism', '1')
                assert app.snapshot()['raw_nn'] == data  # Editing options does not run inference.
                page.click('#step'); ready(page, 'state.game.moves.length === 1 && !state.busy')
                assert '与当前展示局面不同' in page.locator('#nn-context').inner_text()
                page.click('#show-nn')
                assert page.locator('#diagnose').is_disabled()
                assert '与当前展示局面不同' not in page.locator('#nn-context').inner_text()
                page.click('#live')
                page.select_option('#nn-source', 'human')
                assert page.locator('#nn-optimism').is_disabled()
                page.click('#diagnose')
                page.wait_for_function('!state.busy && state.phase === "error"')
                assert 'not provided' in page.locator('#error').inner_text()
                assert app.snapshot()['raw_nn'] == data
                page.select_option('#nn-source', 'main'); page.click('#diagnose'); ready(page)
                assert app.snapshot()['raw_nn']['ply'] == 1
                page.set_viewport_size({'width': 390, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
                if game in ('hex', 'xiangqi'):
                    page.screenshot(path=f'/tmp/etazero-v4-raw-nn-{game}.png', full_page=True)
                assert not errors, errors
                browser.close()
        finally:
            app.close(); server.shutdown(); server.server_close(); thread.join(timeout=5)
