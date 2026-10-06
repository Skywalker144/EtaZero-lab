"""Exercise analysis updates through the real server with a controllable search."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
import unittest

from web.app import App
from web.server import make_server


def position(moves):
    board = [0] * 225
    for index, action in enumerate(moves):
        board[action] = 1 if index % 2 == 0 else -1
    return dict(board_size=15, board=board, moves=moves, turn=len(moves),
                player=1 if len(moves) % 2 == 0 else -1, finished=False, winner=0)


def analysis(game):
    actions = [i for i, value in enumerate(game['board']) if not value]
    weights = [1 / (1 + abs(i - 112)) for i in actions]
    total = sum(weights)
    candidates = [dict(action=i, network_prior=w / total, visits=1,
                       visit_policy=1 / len(actions), selection_weight=w / total)
                  for i, w in zip(actions, weights)]
    probabilities = [0.0] * 225
    for candidate in candidates:
        probabilities[candidate['action']] = candidate['network_prior']
    return dict(board_size=15, board=game['board'], turn=game['turn'], player=game['player'],
                action=actions[0], candidates=candidates, seconds=.125,
                completed_visits=100, root_value=.12, wdl=[.5, .12, .38],
                network_wdl=[.45, .2, .35], requests=100, batches=25,
                network_planes=dict(precision='float32', heads=[
                    dict(name=name, probabilities=probabilities, logits=[0.0] * 225)
                    for name in ('policy', 'opponent_policy', 'long_optimistic_policy', 'short_optimistic_policy')]))


class ControlledEngine:
    def __init__(self):
        self.game = position([112, 113])
        self.searching, self.release = Event(), Event()
        self.fail = False

    def command(self, command):
        operation, value = command.split()
        if operation == 'play':
            self.game = position(self.game['moves'] + [int(value)])
        elif operation == 'undo':
            self.game = position(self.game['moves'][:-int(value)])
        elif operation == 'genmove':
            self.searching.set()
            if not self.release.wait(15):
                raise TimeoutError('Test did not release the search')
            if self.fail:
                raise RuntimeError('Test search failed')
            result = analysis(self.game)
            self.game = position(self.game['moves'] + [result['action']])
            return dict(state=deepcopy(self.game), analysis=result)
        return dict(state=deepcopy(self.game))

    def close(self):
        self.release.set()


class AnalysisUpdateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        model = Path(directory.name) / 'ui_fixture/models/iteration_000012/model.pt'
        model.parent.mkdir(parents=True)
        model.touch()
        (model.parent / 'manifest.json').write_text(json.dumps(dict(
            id='ui-fixture', canvas=15, algorithm='alphazero', weights='SWA',
            checkpoint=dict(iteration=12, total_steps=10000))))
        config = dict(evaluation=dict(visits=100, device='cpu', search_threads=1,
                                     virtual_loss=1, board_size=15, rule='freestyle',
                                     inference_precision='float32'))
        self.app = App(Path('EtaZero_V1.2/build/etazero'), {'fixture': model}, config, 15)
        self.engine = ControlledEngine()
        self.app.engine = self.engine
        self.app.publish(game=self.engine.game, game_id='fixture', model='fixture',
                         evaluation=config['evaluation'], analysis=analysis(position([112])))
        self.previous = self.app.snapshot()['analysis']
        self.addCleanup(self.app.close)
        self.addCleanup(self.engine.release.set)

    def wait_idle(self):
        with self.app.condition:
            self.assertTrue(self.app.condition.wait_for(lambda: not self.app.state['busy'], timeout=5))
        return self.app.snapshot()

    def test_keep_previous_until_search_finishes_and_clear_on_undo(self):
        self.app.submit('play', dict(version=self.app.snapshot()['version'], action=114))
        self.assertTrue(self.engine.searching.wait(5))
        thinking = self.app.snapshot()
        self.assertEqual(thinking['game']['turn'], 3)
        self.assertEqual(thinking['analysis'], self.previous)
        self.engine.release.set()
        completed = self.wait_idle()
        self.assertEqual(completed['game']['turn'], 4)
        self.assertEqual(completed['analysis']['turn'], 3)
        self.app.submit('undo', dict(version=completed['version']))
        self.assertIsNone(self.wait_idle()['analysis'])

    def test_failed_search_and_manual_move_keep_previous(self):
        self.engine.fail = True
        self.engine.release.set()
        self.app.submit('play', dict(version=self.app.snapshot()['version'], action=114))
        failed = self.wait_idle()
        self.assertEqual(failed['analysis'], self.previous)
        self.assertIn('Test search failed', failed['error'])
        self.app.publish(mode='manual')
        self.app.submit('play', dict(version=self.app.snapshot()['version'], action=115))
        self.assertEqual(self.wait_idle()['analysis'], self.previous)

    def test_browser_retains_heatmaps_and_responsive_layout(self):
        from playwright.sync_api import sync_playwright

        server = make_server(self.app, '127.0.0.1', 0)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(self.app.close)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport=dict(width=1440, height=900))
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function("document.querySelector('#status').textContent === '轮到你落子'")
            page.locator('#tab-network').click()
            page.locator('#network-kind').select_option('logits')
            old_map = page.locator('#network-maps').inner_html()
            page.locator('#board [data-action="114"]').click()
            page.wait_for_function("document.querySelector('#analysis-context').textContent.includes('正在更新')")
            self.assertTrue(page.locator('#network-data').is_visible())
            self.assertEqual(page.locator('#network-maps').inner_html(), old_map)
            self.assertEqual(page.locator('#board .stone').count(), 3)
            page.evaluate("document.querySelector('#overlay').value = 'network_prior'; render()")
            self.assertEqual(page.locator('#board .overlay-value').count(), 0)
            page.locator('#tab-heatmaps').click()
            self.assertTrue(page.locator('#heatmap-data').is_visible())
            self.assertEqual(page.locator('#prior-map .heat-stone').count(), 1)
            page.screenshot(path='/tmp/etazero-ui-thinking.png', full_page=True)
            self.engine.release.set()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 4 手' && !document.querySelector('#step').disabled")
            self.assertEqual(page.locator('#tab-heatmaps').get_attribute('aria-selected'), 'true')
            self.assertEqual(page.locator('#network-kind').input_value(), 'logits')
            self.assertEqual(page.locator('#prior-map .heat-stone').count(), 3)
            self.assertEqual(page.locator('#network-maps .heat-stone').count(), 12)
            for width in (1920, 1440, 1280, 1151, 1150, 900, 680, 390):
                page.set_viewport_size(dict(width=width, height=900))
                self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width)
                board = page.locator('#board').bounding_box()
                self.assertAlmostEqual(board['x'] + board['width'] / 2, width / 2, delta=1)
                cell = page.locator('#prior-map .heat-cell').first.bounding_box()
                self.assertAlmostEqual(cell['width'], cell['height'], delta=1)
                if width > 1150:
                    self.assertAlmostEqual(page.locator('.setup').bounding_box()['width'],
                                           page.locator('.analysis').bounding_box()['width'], delta=1)
                    self.assertTrue(page.locator('#setup-details').evaluate('(node) => node.open'))
                else:
                    self.assertFalse(page.locator('#setup-details').evaluate('(node) => node.open'))
            page.locator('.setup-toggle').click()
            self.assertTrue(page.locator('#settings').is_visible())
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            page.locator('.setup-toggle').click()
            page.locator('#overlay').select_option('none')
            page.screenshot(path='/tmp/etazero-ui-mobile.png', full_page=True)
            page.set_viewport_size(dict(width=1440, height=900))
            page.screenshot(path='/tmp/etazero-ui-desktop.png', full_page=True)
            self.assertEqual(errors, [])
            browser.close()
