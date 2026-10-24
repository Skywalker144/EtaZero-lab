"""Exercise analysis updates through the real server with a controllable search."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from threading import Event, Thread
import unittest

from web.app import App, Conflict
from web.server import make_server


def position(moves):
    board = [0] * 225
    for index, action in enumerate(moves):
        board[action] = 1 if index % 2 == 0 else -1
    return dict(board_size=15, board_width=15, board_height=15, board_stride=15, rule='freestyle', board=board, moves=moves, turn=len(moves),
                player=1 if len(moves) % 2 == 0 else -1, finished=False, winner=0)


def analysis(game):
    actions = [i for i, value in enumerate(game['board']) if not value]
    weights = [1 / (1 + abs(i - 112)) for i in actions]
    total = sum(weights)
    candidates = [dict(action=i, rank=rank, network_prior=w / total, visits=max(1, round(w * 100)),
                       visit_policy=w / total, selection_weight=w / total, winrate=.56,
                       draw_rate=.12, wdl=[.5, .12, .38], utility=.12, lcb=.1, radius=.02,
                       raw_selection_weight=w, improved_policy=None, pv=[i])
                  for rank, (i, w) in enumerate(zip(actions, weights), 1)]
    probabilities = [0.0] * 225
    for candidate in candidates:
        probabilities[candidate['action']] = candidate['network_prior']
    return dict(board_size=15, board_width=15, board_height=15, board_stride=15, rule=game['rule'], board=game['board'], turn=game['turn'], player=game['player'],
                action=actions[0], candidates=candidates, seconds=.125,
                completed_visits=100, root_value=.12, wdl=[.5, .12, .38],
                network_wdl=[.45, .2, .35], requests=100, batches=25, root_search_algo='puct', final=True,
                network_planes=dict(precision='float32', heads=[
                    dict(name=name, probabilities=probabilities, logits=[0.0] * 225)
                    for name in ('policy', 'opponent_policy', 'long_optimistic_policy', 'short_optimistic_policy')]))


class ControlledEngine:
    def __init__(self):
        self.game = position([112, 113])
        self.searching, self.release = Event(), Event()
        self.fail = False
        self.analyses = []

    def command(self, command, on_report=None, should_stop=None):
        if isinstance(command, dict):
            operation, value = command['command'], command.get('visits')
        else:
            operation, *values = command.split()
            value = values[0] if values else None
        if operation == 'raw_nn':
            return dict(state=deepcopy(self.game), network_planes=analysis(self.game)['network_planes'])
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
        elif operation == 'analyze':
            self.analyses.append(deepcopy(self.game))
            count = 0
            while not should_stop():
                result = analysis(self.game)
                count += 100
                result.update(completed_visits=count, final=False)
                result.pop('network_planes')
                on_report(result)
                time.sleep(.02)
            result = analysis(self.game)
            result.update(completed_visits=count, final=True)
            result.pop('network_planes')
            return dict(state=deepcopy(self.game), analysis=result, cancelled=True)
        return dict(state=deepcopy(self.game))

    def stop(self):
        pass  # The task's cancellation callback is also checked before search starts.

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
        config = dict(analysis=dict(visits=100, device='cpu', search_threads=1,
                                     virtual_loss=1, board_size=15, rule='freestyle',
                                     inference_precision='float32'))
        self.app = App(Path('EtaZero_V2.7/build/etazero'), {'fixture': model}, config, 15)
        self.engine = ControlledEngine()
        self.app.engine = self.engine
        self.app.publish(game=self.engine.game, game_id='fixture', model='fixture',
                         analysis_config=config['analysis'], analysis=analysis(position([112])))
        self.previous = self.app.snapshot()['analysis']
        self.addCleanup(self.app.close)
        self.addCleanup(self.engine.release.set)

    def wait_idle(self):
        with self.app.condition:
            self.assertTrue(self.app.condition.wait_for(lambda: not self.app.state['busy'], timeout=5))
        return self.app.snapshot()

    def wait_analysis(self, moves):
        with self.app.condition:
            self.assertTrue(self.app.condition.wait_for(
                lambda: self.app.state['analysis_progress'] is not None and
                self.app.state['analysis_progress']['board'] == position(moves)['board'], timeout=5))
        return self.app.snapshot()

    def submit(self, operation, **values):
        state = self.app.snapshot()
        return self.app.submit(operation, dict(version=state['version'], game_id=state['game_id'], **values))

    def test_balanced_opening_rejected_for_go_and_xiangqi_before_engine_work(self):
        for rule in ('go', 'xiangqi'):
            with self.subTest(rule=rule):
                self.app.metadata['fixture']['game'] = rule
                with self.assertRaisesRegex(ValueError, '不支持平衡开局'):
                    self.submit('new', model='fixture', size=10, rule=rule, visits=2,
                                human=1, mode='manual', opening='balanced')
                self.assertFalse(self.app.snapshot()['busy'])
                self.assertIs(self.app.engine, self.engine)

    def test_browser_xiangqi_network_selection_and_hover_outside_top_candidates(self):
        from etazero.games import XIANGQI_MOVES
        from playwright.sync_api import sync_playwright

        game = dict(board_size=10, board_width=9, board_height=10, board_stride=10,
                    rule=5, board=[0]*100, moves=[], turn=0, player=1, finished=False,
                    winner=0, legal_actions=[])
        report = analysis(position([]))
        report.update(game)
        report['candidates'] = report['candidates'][:2]
        for i, candidate in enumerate(report['candidates']):
            candidate.update(action=i, visits=20-15*i, visit_policy=.8-.6*i)
        for head in report['network_planes']['heads']:
            head.update(probabilities=[1/2086]*2086, logits=[0.0]*2086)
        self.app.metadata['fixture'].update(game='xiangqi', canvas=10)
        self.app.config['analysis'].update(rule='xiangqi', board_size=10)
        for run in self.app.runs.values():
            run['analysis_config'].update(rule='xiangqi', board_size=10)
        self.app.publish(game=game, rule='xiangqi', mode='manual', analysis=report)
        server = make_server(self.app, '127.0.0.1', 0)
        Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_selector('#board.xiangqi-board')
            page.locator('#candidate-top').fill('1')
            self.assertEqual(page.locator('#board g[data-candidate="1"]').count(), 0)
            page.locator('#tab-network').click()
            choice = page.locator('#network-maps button[data-action="1"]').first
            choice.hover()
            x,y,u,v = XIANGQI_MOVES[1]
            for square in (y*10+x, v*10+u):
                self.assertIn('preview-point', page.locator(f'#board [data-square="{square}"]').get_attribute('class'))
            choice.click()
            arrow = page.locator('#board g[data-candidate="1"]')
            self.assertEqual(arrow.get_attribute('stroke'), '#d76025')
            page.locator('#candidate-top').focus()
            page.locator('#candidate-top').hover()
            self.assertEqual(page.locator('#board .preview-point').count(), 0)
            page.select_option('#overlay', 'none')
            self.assertEqual(arrow.get_attribute('stroke'), '#d76025')
            self.assertEqual(page.locator('#board g[data-candidate]').count(), 1)
            self.assertEqual(errors, [])
            browser.close()

    def test_continuous_analysis_follows_moves_and_undo_and_stays_paused(self):
        self.app.publish(mode='manual', visits=2, analysis=None)
        self.submit('analyze')
        state = self.wait_analysis([112, 113])
        self.assertTrue(state['analysis_enabled'])
        self.assertFalse(state['busy'])
        self.assertGreater(state['analysis_progress']['completed_visits'], state['visits'])
        self.assertEqual(len(state['analysis_progress']['network_planes']['heads']), 4)
        self.submit('play', action=114)
        state = self.wait_analysis([112, 113, 114])
        self.assertEqual(state['analysis_progress']['player'], -1)
        self.submit('undo')
        self.wait_analysis([112, 113])
        self.submit('stop')
        stopped = self.wait_idle()
        self.assertFalse(stopped['analysis_enabled'])
        self.assertTrue(stopped['analysis']['final'])
        self.assertIn('network_planes', stopped['analysis'])
        starts = len(self.engine.analyses)
        for operation, values in [('play', dict(action=115)), ('undo', {}),
                                  ('configure', dict(human=1, visits=32, mode='manual'))]:
            self.submit(operation, **values)
            self.assertFalse(self.wait_idle()['analysis_enabled'])
        self.assertEqual(len(self.engine.analyses), starts)
        self.submit('analyze')
        self.wait_analysis([112, 113])
        self.assertEqual(len(self.engine.analyses), starts + 1)

    def test_reports_do_not_invalidate_moves_but_board_changes_do(self):
        self.app.publish(mode='manual')
        self.submit('analyze')
        state = self.wait_analysis([112, 113])
        with self.app.condition:
            self.assertTrue(self.app.condition.wait_for(lambda: self.app.state['version'] > state['version'], timeout=5))
        self.app.submit('play', dict(version=state['version'], action=115))
        self.wait_analysis([112, 113, 115])
        with self.assertRaises(Conflict):
            self.app.submit('undo', dict(version=state['version']))

    def test_pause_before_continuous_search_starts(self):
        self.app.publish(mode='manual')
        release = Event()
        self.addCleanup(release.set)
        self.app.executor.submit(lambda: release.wait(5))
        self.submit('analyze')
        self.submit('stop')
        release.set()
        state = self.wait_idle()
        self.assertFalse(state['analysis_enabled'])
        self.assertEqual(self.engine.analyses, [])
        self.assertEqual(state['game']['moves'], [112, 113])

    def test_pause_at_terminal_position_stays_paused_after_undo(self):
        self.engine.game.update(finished=True, winner=1)
        self.app.publish(mode='manual', analysis_enabled=True, game=deepcopy(self.engine.game))
        self.submit('stop')
        self.assertFalse(self.app.snapshot()['analysis_enabled'])
        self.submit('undo')
        self.assertFalse(self.wait_idle()['analysis_enabled'])
        self.assertEqual(self.engine.analyses, [])

    def test_browser_continuous_analysis_and_pause_controls(self):
        from playwright.sync_api import sync_playwright

        self.app.publish(mode='manual', visits=2, analysis=None)
        server = make_server(self.app, '127.0.0.1', 0)
        Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport=dict(width=1440, height=900))
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function("!document.querySelector('#analyze').disabled")
            controls = page.locator('.actions').bounding_box()
            board = page.locator('#board').bounding_box()
            self.assertLess(controls['y'] + controls['height'], board['y'])
            page.locator('#analyze').click()
            page.wait_for_function("document.querySelector('#analysis-context').textContent.includes('实时分析')")
            self.assertEqual(page.locator('#analyze').get_attribute('aria-pressed'), 'true')
            page.keyboard.press('a')
            page.wait_for_function("document.querySelector('#analyze').getAttribute('aria-pressed') === 'false' && !document.querySelector('#analyze').disabled")
            page.keyboard.press('a')
            page.wait_for_function("document.querySelector('#analysis-context').textContent.includes('实时分析')")
            self.assertFalse(page.locator('#board [data-action="114"]').is_disabled())
            page.locator('#tab-network').click()
            self.assertEqual(page.locator('#network-maps > section').count(), 4)
            page.evaluate("window.savedNetworkNode = document.querySelector('#network-maps > section')")
            page.wait_for_function("Number(document.querySelector('#completed').textContent.replaceAll(',', '')) >= 500")
            self.assertTrue(page.evaluate('window.savedNetworkNode.isConnected'))
            page.locator('#board [data-action="114"]').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 3 手' && document.querySelector('#status').textContent.includes('持续分析')")
            self.wait_analysis([112, 113, 114])
            page.locator('#undo').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 2 手' && document.querySelector('#status').textContent.includes('持续分析')")
            page.locator('#analyze').click()
            page.wait_for_function("document.querySelector('#analyze').getAttribute('aria-pressed') === 'false' && !document.querySelector('#undo').disabled")
            starts = len(self.engine.analyses)
            page.locator('#board [data-action="115"]').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 3 手' && !document.querySelector('#undo').disabled")
            page.locator('#undo').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 2 手' && !document.querySelector('#undo').disabled")
            self.assertEqual(len(self.engine.analyses), starts)
            page.locator('body').click(position=dict(x=5, y=5))
            page.keyboard.press('a')
            page.wait_for_function("document.querySelector('#status').textContent.includes('持续分析')")
            page.keyboard.press('Escape')
            page.wait_for_function("document.querySelector('#status').textContent.includes('分析已暂停')")
            for theme in ('light', 'dark'):
                page.locator('#theme').select_option(theme)
                for width in (1440, 390):
                    page.set_viewport_size(dict(width=width, height=900))
                    readable = page.evaluate('''() => {
                      const rgb = color => {
                        const ctx = document.createElement('canvas').getContext('2d');
                        ctx.fillStyle = color; ctx.fillRect(0, 0, 1, 1);
                        return [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3);
                      };
                      const luminance = color => rgb(color).map(x => x / 255)
                        .map(x => x <= .04045 ? x / 12.92 : ((x + .055) / 1.055) ** 2.4)
                        .reduce((sum, x, i) => sum + x * [.2126, .7152, .0722][i], 0);
                      const contrast = (a, b) => (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
                      return [...document.querySelectorAll('.candidate-marker')].every(marker => {
                        const bg = luminance(getComputedStyle(marker).backgroundColor);
                        return [...marker.children].every(child => {
                          const css = getComputedStyle(child);
                          if (css.display === 'none') return true;
                          const childBg = child.tagName === 'SUP' ? luminance(css.backgroundColor) : bg;
                          const minimum = child.tagName === 'STRONG' ? 12 : child.tagName === 'SUP' ? 9 : 10;
                          return parseFloat(css.fontSize) >= minimum && contrast(luminance(css.color), childBg) >= 4.5;
                        });
                      });
                    }''')
                    self.assertTrue(readable, (theme, width))
                    self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                    page.screenshot(path=f'/tmp/etazero-continuous-{theme}-{width}.png', full_page=True)
            self.assertEqual(errors, [])
            browser.close()

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
            # Old three-class models stay readable; V3.2 exposes a distinct NR.
            self.assertTrue(page.locator('#wdl-no-result').is_hidden())
            page.evaluate("renderWdl('wdl', '搜索', [.4, .2, .3, .1])")
            self.assertIn('D 20.0%', page.locator('#wdl').inner_text())
            self.assertIn('NR 10.0%', page.locator('#wdl').inner_text())
            page.evaluate("renderWdl('wdl', '搜索', [.5, .12, .38])")
            self.assertNotIn('NR', page.locator('#wdl').inner_text())
            candidate = page.locator('[data-candidate="114"]')
            candidate.hover()
            self.assertEqual(page.locator('#board .preview-point').count(), 0)
            page.locator('#show-analysis').click()
            candidate.hover()
            self.assertEqual(page.locator('#board .preview-point').get_attribute('data-action'), '114')
            self.assertEqual(page.locator('#history-count').inner_text(), '1 / 2 手')
            page.locator('#status').hover()
            self.assertEqual(page.locator('#board .preview-point').count(), 0)
            candidate.focus()
            self.assertEqual(page.locator('#board .preview-point').get_attribute('data-action'), '114')
            page.locator('#tab-network').focus()
            self.assertEqual(page.locator('#board .preview-point').count(), 0)
            page.locator('#live').click()
            page.locator('#tab-network').click()
            page.locator('#network-kind').select_option('logits')
            old_map = page.locator('#network-maps').inner_html()
            page.locator('#theme').select_option('light')
            light = page.locator('#network-maps .heat-cell').first.evaluate('(el) => getComputedStyle(el).backgroundColor')
            page.locator('#theme').select_option('dark')
            dark = page.locator('#network-maps .heat-cell').first.evaluate('(el) => getComputedStyle(el).backgroundColor')
            self.assertNotEqual(light, dark)
            self.assertEqual(page.locator('#network-maps').inner_html(), old_map)
            page.locator('#theme').select_option('light')
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
