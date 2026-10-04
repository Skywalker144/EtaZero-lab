import os
import unittest
from pathlib import Path
from urllib.parse import urlsplit


class ThemeTests(unittest.TestCase):
    def test_system_theme_and_saved_override(self):
        from playwright.sync_api import sync_playwright

        static = Path(__file__).resolve().parents[1] / 'static'
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(color_scheme='dark')
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))

            def serve(route):
                path = urlsplit(route.request.url).path
                files = {'/': 'index.html', '/app.js': 'app.js', '/theme.js': 'theme.js', '/styles.css': 'styles.css'}
                if path in files:
                    route.fulfill(path=str(static / files[path]))
                else:
                    route.fulfill(status=503, json={'error': '测试未连接模型服务'})

            page.route('http://etazero.test/**', serve)
            page.goto('http://etazero.test/')
            self.assertEqual(page.locator('#theme').input_value(), 'system')
            self.assertEqual(page.locator('html').get_attribute('data-theme'), 'dark')
            self.assertEqual(page.evaluate("getComputedStyle(document.documentElement).colorScheme"), 'dark')
            page.locator('#numbers').uncheck()
            page.reload()
            self.assertFalse(page.locator('#numbers').is_checked())
            self.assertEqual(page.locator('#theme').input_value(), 'system')
            page.emulate_media(color_scheme='light')
            page.wait_for_function("document.documentElement.dataset.theme === 'light'")
            page.locator('#theme').select_option('dark')
            page.reload()
            self.assertEqual(page.locator('#theme').input_value(), 'dark')
            self.assertEqual(page.locator('html').get_attribute('data-theme'), 'dark')
            page.emulate_media(color_scheme='dark')
            page.locator('#theme').select_option('light')
            self.assertEqual(page.locator('html').get_attribute('data-theme'), 'light')
            page.locator('#theme').select_option('system')
            self.assertEqual(page.locator('html').get_attribute('data-theme'), 'dark')
            page.reload()
            page.emulate_media(color_scheme='light')
            page.wait_for_function("document.documentElement.dataset.theme === 'light'")
            page.set_viewport_size({'width': 390, 'height': 844})
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
            page.evaluate("localStorage.setItem('etazero-ui', 'invalid json')")
            page.emulate_media(color_scheme='dark')
            page.reload()
            self.assertEqual(page.locator('#theme').input_value(), 'system')
            self.assertEqual(page.locator('html').get_attribute('data-theme'), 'dark')
            self.assertEqual(errors, [])
            browser.close()


@unittest.skipUnless(os.environ.get('ETAZERO_WEB_TEST_URL'), 'Set ETAZERO_WEB_TEST_URL')
class BrowserTests(unittest.TestCase):
    def test_real_game_in_browser(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1440, 'height': 900}, device_scale_factor=1)
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(os.environ['ETAZERO_WEB_TEST_URL'])
            page.wait_for_function("document.querySelector('#model').options.length > 0")
            page.locator('#size').select_option('7')
            page.locator('#rule').select_option('standard')
            page.locator('#mode').select_option('play')
            page.locator('input[name="human"][value="1"]').check()
            page.locator('#visits').fill('32')
            page.locator('#new-game').click()
            page.wait_for_function("document.querySelector('#status').textContent === '轮到你落子'")
            page.locator('[data-action="24"]').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 2 手'")
            page.wait_for_function("!document.querySelector('#undo').disabled")
            page.locator('#tab-heatmaps').click()
            self.assertTrue(page.locator('#heatmap-data').is_visible())
            self.assertEqual(page.locator('#prior-map .heat-cell').count(), 49)
            self.assertEqual(page.locator('#prior-map .heat-stone').count(), 1)
            self.assertEqual(page.locator('#board .stone').count(), 2)
            snapshot = page.evaluate("async () => (await (await fetch('/api/state')).json()).analysis")
            for field, selector in [('network_prior', '#prior-map'), ('visit_policy', '#search-map')]:
                values = page.locator(selector + ' .heat-cell').evaluate_all("cells => cells.map(cell => Number(cell.dataset.value))")
                self.assertAlmostEqual(sum(values), 1)
                for candidate in snapshot['candidates']:
                    self.assertAlmostEqual(values[candidate['action']], candidate[field])
            page.locator('#search-map-kind').select_option('selection_weight')
            selected = page.locator('#search-map .heat-cell').evaluate_all("cells => cells.map(cell => Number(cell.dataset.value))")
            for candidate in snapshot['candidates']:
                self.assertAlmostEqual(selected[candidate['action']], candidate['selection_weight'])
            page.locator('#prior-map .heat-cell').first.hover()
            self.assertIn('网络先验', page.locator('#heatmap-detail').inner_text())
            page.locator('#heatmaps').screenshot(path='/tmp/etazero-heatmaps.png')
            page.locator('#tab-candidates').click()
            page.locator('#undo').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 0 手'")
            self.assertTrue(page.locator('#heatmap-data').is_hidden())
            self.assertEqual(page.locator('#prior-map .heat-cell').count(), 0)
            for _ in range(4):
                page.wait_for_function("document.querySelector('#status').textContent === '轮到你落子'")
                turn = int(page.locator('#move-count').inner_text().split()[1])
                page.locator('.board-point:not([disabled])').first.click()
                page.wait_for_function("n => Number(document.querySelector('#move-count').textContent.split(' ')[1]) >= n", arg=turn + 2)
            page.wait_for_function("!document.querySelector('#undo').disabled")
            page.screenshot(path='/tmp/etazero-web-desktop.png', full_page=True)
            count = page.locator('#move-count').inner_text()
            page.reload()
            page.wait_for_function("text => document.querySelector('#move-count').textContent === text", arg=count)
            page.locator('#size').select_option('5')
            page.locator('#rule').select_option('renju')
            page.locator('input[name="human"][value="-1"]').check()
            page.locator('#new-game').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 1 手'")
            page.wait_for_function("document.querySelector('#status').textContent === '轮到你落子'")
            self.assertTrue(page.locator('#undo').is_disabled())
            page.locator('.board-point:not([disabled])').first.click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 3 手'")
            page.wait_for_function("!document.querySelector('#undo').disabled")
            page.locator('#undo').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 1 手'")
            for _ in range(25):
                page.wait_for_function("!document.querySelector('#new-game').disabled")
                status = page.locator('#status').inner_text()
                if '获胜' in status or '和棋' in status:
                    break
                page.locator('.board-point:not([disabled])').first.click()
                page.wait_for_function("!document.querySelector('#new-game').disabled")
            else:
                self.fail('Game did not finish')
            self.assertEqual(page.locator('.board-point:not([disabled])').count(), 0)
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path='/tmp/etazero-web-mobile.png', full_page=True)
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
            self.assertEqual(errors, [])
            browser.close()

    def test_developer_workflow(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1440, 'height': 900})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(os.environ['ETAZERO_WEB_TEST_URL'])
            page.wait_for_function("!document.querySelector('#new-game').disabled && document.querySelector('#state-version').textContent.includes('v')")
            page.locator('#size').select_option('7')
            page.locator('#mode').select_option('manual')
            page.locator('#visits').fill('32')
            page.locator('#new-game').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 0 手' && !document.querySelector('#analyze').disabled")
            page.locator('#board [data-action="24"]').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 1 手' && !document.querySelector('#analyze').disabled")
            page.locator('#board [data-action="0"]').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 2 手' && !document.querySelector('#analyze').disabled")
            page.locator('#analyze').click()
            page.wait_for_function("document.querySelector('#completed').textContent === '32' && !document.querySelector('#analyze').disabled")
            self.assertEqual(page.locator('#move-count').inner_text(), '第 2 手')
            self.assertEqual(page.locator('#candidates tr').count(), 47)
            page.locator('[data-sort="network_prior"]').click()
            priors = page.locator('#candidates tr td:nth-child(2)').all_text_contents()
            self.assertEqual([float(p) for p in priors], sorted([float(p) for p in priors], reverse=True))
            page.locator('#visited-only').check()
            self.assertLess(page.locator('#candidates tr').count(), 47)
            page.locator('#visited-only').uncheck()
            page.locator('#candidates [data-candidate]').first.click()
            self.assertEqual(page.locator('#board .highlight').count(), 1)
            page.locator('#overlay').select_option('network_prior')
            self.assertGreater(page.locator('#board .overlay-value').count(), 0)
            page.locator('#tab-heatmaps').click()
            self.assertEqual(page.locator('#prior-map .heat-stone').count(), 2)
            page.locator('#tab-raw').click()
            self.assertIn('network_wdl', page.locator('#raw-analysis').inner_text())
            page.locator('#tab-candidates').click()
            page.locator('#step').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 3 手' && !document.querySelector('#step').disabled")
            self.assertIn('历史分析', page.locator('#analysis-context').inner_text())
            self.assertEqual(page.locator('#board .overlay-value').count(), 0)
            page.locator('#show-analysis').click()
            self.assertEqual(page.locator('#board .stone').count(), 2)
            self.assertTrue(page.locator('#analyze').is_disabled())
            self.assertTrue(page.locator('#branch').is_visible())
            self.assertEqual(page.locator('#board .board-point:not([disabled])').count(), 0)
            page.locator('#live').click()
            self.assertEqual(page.locator('#board .stone').count(), 3)
            page.locator('#history [data-turn="1"]').click()
            page.locator('#branch').click()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 1 手' && !document.querySelector('#step').disabled")
            self.assertTrue(page.locator('#analysis-data').is_hidden())
            page.locator('#visits').fill('16')
            page.locator('#apply-settings').click()
            page.wait_for_function("document.querySelector('#session-info').textContent.includes('16v') && !document.querySelector('#analyze').disabled")
            page.locator('h1').click()
            page.keyboard.press('a')
            page.wait_for_function("document.querySelector('#completed').textContent === '16' && !document.querySelector('#analyze').disabled")
            selected_run = page.locator('#run').input_value()
            self.assertTrue(selected_run)
            page.locator('#refresh-models').click()
            page.wait_for_function("!document.querySelector('#refresh-models').disabled")
            self.assertEqual(page.locator('#run').input_value(), selected_run)
            self.assertEqual(page.locator('#move-count').inner_text(), '第 1 手')
            # Failed requests must release the lock, then resync without losing the game.
            page.context.set_offline(True)
            page.locator('#analyze').click()
            page.wait_for_function("document.querySelector('#connection').textContent.includes('断线')")
            self.assertTrue(page.locator('#step').is_disabled())
            page.context.set_offline(False)
            page.wait_for_function("document.querySelector('#connection').textContent === '引擎已连接' && !document.querySelector('#analyze').disabled")
            self.assertEqual(page.locator('#move-count').inner_text(), '第 1 手')
            page.locator('#analyze').click()
            page.wait_for_function("!document.querySelector('#analyze').disabled && document.querySelector('#error').hidden")
            page.screenshot(path='/tmp/etazero-workbench-light.png', full_page=True)
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
            page.locator('#theme').select_option('dark')
            page.locator('#numbers').uncheck()
            page.reload()
            page.wait_for_function("document.querySelector('#move-count').textContent === '第 1 手'")
            self.assertEqual(page.locator('html').get_attribute('data-theme'), 'dark')
            self.assertFalse(page.locator('#numbers').is_checked())
            page.screenshot(path='/tmp/etazero-workbench-dark.png', full_page=True)
            for width in (1024, 768, 390):
                page.set_viewport_size({'width': width, 'height': 844})
                self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), width)
            page.screenshot(path='/tmp/etazero-workbench-mobile.png', full_page=True)
            self.assertEqual(errors, [])
            browser.close()

    def test_directory_and_generation_selection(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1440, 'height': 900})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(os.environ['ETAZERO_WEB_TEST_URL'])
            page.wait_for_function("document.querySelector('#connection').textContent === '引擎已连接'")
            catalog = page.request.get(os.environ['ETAZERO_WEB_TEST_URL'] + '/api/catalog').json()
            before = page.request.get(os.environ['ETAZERO_WEB_TEST_URL'] + '/api/state').json()
            self.assertEqual(page.locator('#run option').count(), len(catalog['runs']))
            for run in catalog['runs']:
                page.locator('#run').select_option(run['id'])
                if not run['models']:
                    self.assertTrue(page.locator('#new-game').is_disabled())
                    self.assertIn('暂无', page.locator('#model').inner_text())
                    continue
                ids = page.locator('#model option').evaluate_all('options => options.map(option => option.value)')
                self.assertEqual(ids, run['models'])
                self.assertEqual(page.locator('#model').input_value(), run['default_model'])
                if len(ids) > 1:
                    page.locator('#model').select_option(ids[-1])
                    manifest = next(m['manifest'] for m in catalog['models'] if m['id'] == ids[-1])
                    self.assertIn(manifest['id'], page.locator('#model-info').text_content())
                    page.locator('#refresh-models').click()
                    page.wait_for_function("!document.querySelector('#refresh-models').disabled")
                    self.assertEqual(page.locator('#model').input_value(), ids[-1])
                    self.assertEqual(page.locator('#run').input_value(), run['id'])
            after = page.request.get(os.environ['ETAZERO_WEB_TEST_URL'] + '/api/state').json()
            self.assertEqual(after['game'], before['game'])
            self.assertEqual(after['model'], before['model'])
            self.assertEqual(errors, [])
            browser.close()


if __name__ == '__main__':
    unittest.main()
