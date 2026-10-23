"""Xiangqi movement and redraw behavior in a real browser, without model inference."""
from copy import deepcopy
import json
from threading import Event, Thread
from unittest.mock import Mock

from etazero.config import ROOT
from etazero.games import XIANGQI_INDEX
from playwright.sync_api import sync_playwright

from web.app import App
from web.server import make_server


def test_xiangqi_move_animation_survives_redraws_and_history(tmp_path):
    model = tmp_path / 'fixture/models/iteration_000001/model.pt'
    model.parent.mkdir(parents=True)
    model.touch()
    (model.parent / 'manifest.json').write_text(json.dumps(dict(
        id='xiangqi-animation', canvas=10, game='xiangqi', algorithm='alphazero',
        checkpoint=dict(iteration=1))))
    profile = dict(visits=2, device='cpu', search_threads=1, virtual_loss=1,
                   board_size=10, rule='xiangqi', inference_precision='float32')
    app = App(ROOT / 'build/etazero', {'fixture': model}, dict(analysis=profile), 10)
    first = XIANGQI_INDEX[0, 0, 0, 2]
    capture = XIANGQI_INDEX[0, 3, 0, 2]
    board = [0] * 100
    board[0], board[30], board[4], board[94] = 1, -1, 7, -7
    initial = dict(board_size=10, board_width=9, board_height=10, board_stride=10,
                   rule=5, board=board, boards=[board], players=[1], moves=[], turn=0,
                   player=1, finished=False, winner=0, legal_actions=[first])
    moved = deepcopy(initial)
    moved['board'][0], moved['board'][20] = 0, 1
    moved.update(moves=[first], turn=1, player=-1, legal_actions=[capture],
                 boards=[board, moved['board']], players=[1, -1])
    captured = deepcopy(moved)
    captured['board'][30], captured['board'][20] = 0, -1
    captured.update(moves=[first, capture], turn=2, player=1, legal_actions=[],
                    boards=[board, moved['board'], captured['board']], players=[1, -1, 1])
    app.engine = Mock()
    app.engine.command.return_value = dict(state=deepcopy(moved))
    app.publish(game=deepcopy(initial), game_id='fixture', model='fixture', rule='xiangqi',
                mode='manual', analysis_config=profile)
    server = make_server(app, '127.0.0.1', 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport=dict(width=1440, height=1000), reduced_motion='no-preference')
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            # Pause only piece animations so geometry assertions do not race the clock.
            page.add_init_script("""const animate = Element.prototype.animate;
                Element.prototype.animate = function(...args) {
                    const animation = animate.apply(this, args);
                    if (this.closest('.xiangqi-pieces')) {
                        animation.pause(); animation.currentTime = 0;
                        animation.play = () => animation.pause();
                    }
                    return animation;
                };""")
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_selector('#board.xiangqi-board')

            def show(game, **values):
                app.publish(game=deepcopy(game), **values)
                page.wait_for_function('version => state.version >= version', arg=app.snapshot()['version'])

            def center(selector):
                return page.locator(selector).evaluate('node => {const r=node.getBoundingClientRect(); return [r.x+r.width/2,r.y+r.height/2];}')

            def finish():
                while page.locator('#board .moving-piece').count():
                    page.evaluate("window.finishingPiece = document.querySelector('.moving-piece'); finishingPiece.getAnimations()[0].finish()")
                    page.wait_for_function("!finishingPiece.isConnected || !finishingPiece.classList.contains('moving-piece')")
                assert page.locator('#board .captured-piece').count() == 0

            assert page.locator('#board .xiangqi-piece').count() == 4
            assert page.locator('#board').evaluate('node => node.getAnimations({subtree:true}).length') == 0
            page.locator('#board [data-square="0"]').click()
            assert 'legal-destination' in page.locator('#board [data-square="20"]').get_attribute('class')
            page.locator('#board [data-square="20"]').click()
            page.wait_for_function('state.game.turn === 1 && !state.busy && !pending')
            app.engine.command.assert_called_once_with(f'play {first}')
            source = center('#board [data-square="0"]')
            target = center('#board [data-square="20"]')
            assert abs(center('#board .moving-piece')[1] - source[1]) < 1
            page.evaluate("window.pieceAnimation = document.querySelector('.moving-piece').getAnimations()[0]; pieceAnimation.currentTime = 120")
            midway = center('#board .moving-piece')[1]
            assert target[1] < midway < source[1]
            # Changing disabled buttons and analysis overlays must preserve the live animation.
            page.evaluate("highlight = 0; render();")
            show(moved, busy=True)
            show(moved, busy=False)
            assert page.evaluate("document.querySelector('.moving-piece').getAnimations()[0] === pieceAnimation")
            finish()
            assert page.locator('#board .xiangqi-piece').count() == 4

            show(captured)  # AI reply: capture stays visible until the moving rook arrives.
            assert page.locator('#board .captured-piece').count() == 1
            assert page.locator('#board .moving-piece .black-piece').count() == 1
            finish()
            assert page.locator('#board .xiangqi-piece').count() == 3
            page.click('#prev')
            assert page.locator('#board .moving-piece').count() == 1
            finish()
            assert page.locator('#board .xiangqi-piece').count() == 4
            page.click('#live')
            finish()
            page.click('#first')  # Multi-move jumps display the target immediately.
            assert page.locator('#board .moving-piece').count() == 0
            page.click('#live')
            assert page.locator('#board .moving-piece').count() == 0

            show(moved)  # Single-move undo animates in reverse.
            assert page.locator('#board .moving-piece').count() == 1
            show(captured)  # A second move before completion must leave no old captures.
            finish()
            assert page.locator('#board .xiangqi-piece').count() == 3
            show(moved)
            show(initial, game_id='replacement')  # Reopening clears unfinished effects.
            assert page.locator('#board .moving-piece,.captured-piece').count() == 0
            page.emulate_media(reduced_motion='reduce')
            show(moved)
            assert page.locator('#board .moving-piece').count() == 0
            page.set_viewport_size(dict(width=390, height=844))
            show(captured)
            assert page.locator('#board .xiangqi-piece').count() == 3
            page.screenshot(path=str(tmp_path / 'xiangqi-mobile.png'), full_page=True)

            # A single poll can contain both the human move and the AI reply.
            page.emulate_media(reduced_motion='no-preference')
            show(initial, game_id='coalesced', mode='play', human=1)
            with app.condition:
                app.publish(game=deepcopy(moved), busy=True)
                app.publish(game=deepcopy(captured), busy=False)
            page.wait_for_function('state.game.turn === 2')
            assert page.locator('#board .moving-piece .red-piece').count() == 1
            page.evaluate("document.querySelector('.moving-piece').getAnimations()[0].finish()")
            page.wait_for_selector('#board .moving-piece .black-piece')
            black_source = center('#board [data-square="30"]')
            assert abs(center('#board .moving-piece')[1] - black_source[1]) < 1
            finish()

            # Exercise the actual automatic-reply path while the human animation is paused.
            show(initial, game_id='automatic')
            release = Event()

            def engine_reply(request, **kwargs):
                if isinstance(request, str):
                    assert request == f'play {first}'
                    return dict(state=deepcopy(moved))
                assert request['command'] == 'genmove'
                assert release.wait(5)
                return dict(state=deepcopy(captured), analysis=dict(completed_visits=0))

            app.engine.command.side_effect = engine_reply
            page.click('#board [data-square="0"]')
            page.click('#board [data-square="20"]')
            page.wait_for_selector('#board .moving-piece .red-piece')
            page.evaluate("window.humanAnimation = document.querySelector('.moving-piece').getAnimations()[0]")
            release.set()
            page.wait_for_function('state.game.turn === 2 && !state.busy && !pending')
            assert page.evaluate("document.querySelector('.moving-piece').getAnimations()[0] === humanAnimation")
            page.evaluate('humanAnimation.finish()')
            page.wait_for_selector('#board .moving-piece .black-piece')
            finish()
            assert errors == []
            browser.close()
    finally:
        app.close()
        server.shutdown()
        server.server_close()
        thread.join()
