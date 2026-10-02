import json
import os
import subprocess
import tempfile
from pathlib import Path
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from threading import Thread

from etazero.eval_config import load_evaluation_config
from etazero.config import write_native
from etazero.evaluation import model_info
from web.engine import Engine
from web.app import App, Conflict
from web.server import make_server


MODEL = os.environ.get('ETAZERO_WEB_TEST_MODEL')
BINARY = Path(__file__).resolve().parents[2] / 'EtaZero_V0/build/etazero'


@unittest.skipUnless(MODEL, 'Set ETAZERO_WEB_TEST_MODEL')
class WebTests(unittest.TestCase):
    def setUp(self):
        self.config = load_evaluation_config('configs/minimal_test', environ={
            'EVAL_DEVICE': os.environ.get('ETAZERO_TEST_DEVICE', 'cuda:0'),
            'EVAL_VISITS': '16',
        })

    def test_persistent_engine_and_rules(self):
        with Engine(BINARY, Path(MODEL), self.config) as engine:
            pid = engine.process.pid
            engine.command('new 5 standard')
            engine.command('play 12')
            analyzed = engine.command('analyze 16')
            self.assertEqual(analyzed['state']['turn'], 1)
            self.assertEqual(analyzed['analysis']['board'], analyzed['state']['board'])
            reply = engine.command('genmove 16')
            self.assertEqual(reply['state']['turn'], 2)
            self.assertEqual(reply['analysis']['turn'], 1)
            self.assertEqual(reply['analysis']['board'][12], 1)
            self.assertEqual(sum(value != 0 for value in reply['analysis']['board']), 1)
            self.assertAlmostEqual(sum(c['network_prior'] for c in reply['analysis']['candidates']), 1)
            self.assertAlmostEqual(sum(c['visit_policy'] for c in reply['analysis']['candidates']), 1)
            self.assertAlmostEqual(sum(c['selection_weight'] for c in reply['analysis']['candidates']), 1)
            self.assertEqual(sum(c['visits'] for c in reply['analysis']['candidates']), 15)
            self.assertEqual(reply['analysis']['completed_visits'], 16)
            self.assertAlmostEqual(sum(reply['analysis']['wdl']), 1)
            self.assertAlmostEqual(reply['analysis']['root_value'], reply['analysis']['wdl'][0] - reply['analysis']['wdl'][2])
            self.assertEqual(reply['analysis']['action'], reply['analysis']['candidates'][0]['action'])
            self.assertEqual(engine.command('undo 2')['state']['turn'], 0)
            with self.assertRaises(RuntimeError):
                engine.command('play 25')
            with self.assertRaises(RuntimeError):
                engine.command('new 25 standard')
            with self.assertRaises(RuntimeError):
                engine.command('play 2garbage')
            with self.assertRaises(RuntimeError):
                engine.command('genmove 1')
            self.assertEqual(engine.command('state')['state']['turn'], 0)
            for move in [0, 5, 1, 6, 2, 7, 3, 8, 4]:
                state = engine.command(f'play {move}')['state']
            self.assertTrue(state['finished'])
            self.assertEqual(state['winner'], 1)
            self.assertFalse(engine.command('undo 1')['state']['finished'])
            engine.command('new 7 renju')
            for move in [21, 0, 22, 2, 24, 4, 25, 6, 26, 8, 23]:
                state = engine.command(f'play {move}')['state']
            self.assertTrue(state['finished'])
            self.assertEqual(state['winner'], -1)
            self.assertEqual(engine.process.pid, pid)
        self.assertIsNotNone(engine.process.poll())

    def test_http_game_undo_white_restart_and_stale_requests(self):
        app = App(BINARY, {'test': Path(MODEL)}, self.config, 7)
        server = make_server(app, '127.0.0.1', 0)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'

        def request(path, payload=None):
            data = None if payload is None else json.dumps(payload).encode()
            with urlopen(Request(base + path, data=data, headers={'Content-Type': 'application/json'}), timeout=15) as response:
                return json.load(response)

        def ready():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                state = request('/api/state')
                if not state['busy']:
                    self.assertIsNone(state['error'])
                    return state
                time.sleep(0.05)
            self.fail('Timed out waiting for AI')

        try:
            with urlopen(base) as response:
                self.assertIn('EtaZero', response.read().decode())
            self.assertEqual(len(request('/api/catalog')['models']), 1)
            request('/api/new', {'version': 0, 'model': 'test', 'size': 7, 'rule': 'standard', 'human': 1, 'visits': 16})
            state = ready()
            pid = app.engine.process.pid
            old_version = state['version']
            request('/api/play', {'version': old_version, 'action': 24})
            with self.assertRaises(HTTPError) as stale:
                request('/api/play', {'version': old_version, 'action': 25})
            self.assertEqual(stale.exception.code, 409)
            state = ready()
            self.assertEqual(state['game']['turn'], 2)
            request('/api/undo', {'version': state['version']})
            state = ready()
            self.assertEqual(state['game']['turn'], 0)
            request('/api/new', {'version': state['version'], 'model': 'test', 'size': 7, 'rule': 'renju', 'human': -1, 'visits': 16})
            state = ready()
            self.assertEqual(state['game']['turn'], 1)
            self.assertEqual(state['game']['player'], -1)
            self.assertEqual(app.engine.process.pid, pid)
            with self.assertRaises(HTTPError) as invalid:
                request('/api/play', {'version': state['version'], 'action': True})
            self.assertEqual(invalid.exception.code, 400)
            with self.assertRaises(HTTPError) as invalid_size:
                request('/api/new', {'version': state['version'], 'model': 'test', 'size': 25, 'rule': 'standard', 'human': 1, 'visits': 16})
            self.assertEqual(invalid_size.exception.code, 400)
            state = request('/api/state')
            self.assertIsNone(state['error'])
            self.assertEqual(state['human'], -1)
            self.assertEqual(state['game']['turn'], 1)
            app.engine.process.kill()
            app.engine.process.wait()
            request('/api/new', {'version': state['version'], 'model': 'test', 'size': 7, 'rule': 'standard', 'human': 1, 'visits': 16})
            state = ready()
            self.assertEqual(state['game']['turn'], 0)
            self.assertNotEqual(app.engine.process.pid, pid)
        finally:
            server.shutdown()
            server.server_close()
            app.close()
            thread.join()

    def test_analysis_matches_existing_evaluation(self):
        model, info = model_info(MODEL)
        with Engine(BINARY, model, self.config) as engine, tempfile.TemporaryDirectory() as directory:
            # A smaller board exercises local-to-canvas action mapping.
            engine.command('new 7 renju')
            engine.command('play 24')
            analysis = engine.command('analyze 16')['analysis']
            resolved = Path(directory) / 'eval.cfg'
            write_native({**self.config, 'network': {'canvas': info['canvas']}}, resolved)
            native_move = 3 * info['canvas'] + 3
            reply = json.loads(subprocess.run(
                [str(BINARY), 'evaluate', '--config', str(resolved), '--model', str(model),
                 '--model-id', info['id'], '--device', self.config['evaluation']['device'],
                 '--size', '7', '--rule', 'renju', '--moves', str(native_move), '--seed', '1'],
                check=True, capture_output=True, text=True).stdout)
            self.assertEqual(analysis['action'], reply['action'] // info['canvas'] * 7 + reply['action'] % info['canvas'])
            self.assertAlmostEqual(analysis['root_value'], reply['value'], places=6)
            for candidate in analysis['candidates']:
                action = candidate['action'] // 7 * info['canvas'] + candidate['action'] % 7
                self.assertEqual(candidate['visits'], reply['visits'][action])
                self.assertAlmostEqual(candidate['network_prior'], reply['network_policy'][action], places=7)
                self.assertAlmostEqual(candidate['selection_weight'], reply['policy'][action], places=7)

    def test_manual_analysis_branch_configuration_and_catalog_refresh(self):
        discovered = {'test': Path(MODEL)}
        app = App(BINARY, discovered.copy(), self.config, 7, discover_models=lambda: discovered.copy())

        def perform(operation, **payload):
            app.submit(operation, dict(version=app.snapshot()['version'], **payload))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                state = app.snapshot()
                if not state['busy']:
                    self.assertIsNone(state['error'])
                    return state
                time.sleep(.02)
            self.fail('Timed out waiting for operation')

        try:
            state = perform('new', model='test', size=7, rule='renju', human=-1, visits=16, mode='manual')
            self.assertEqual(state['game']['turn'], 0)  # Manual mode never auto-plays black.
            game_id = state['game_id']
            state = perform('play', action=24)
            self.assertEqual(state['game']['moves'], [24])
            before = state['game']
            state = perform('analyze')
            self.assertEqual(state['game'], before)  # Analysis must not advance the position.
            self.assertEqual(state['analysis']['turn'], 1)
            self.assertEqual(state['analysis']['player'], -1)
            self.assertEqual(state['analysis']['board'], before['board'])
            state = perform('configure', human=1, visits=32, mode='manual')
            self.assertEqual(state['game'], before)
            state = perform('step')
            self.assertEqual(state['game']['turn'], 2)
            self.assertEqual(state['analysis']['completed_visits'], 32)
            state = perform('undo')
            self.assertEqual(state['game']['moves'], [24])
            self.assertIsNone(state['analysis'])
            state = perform('play', action=0)
            state = perform('configure', human=1, visits=16, mode='play')
            state = perform('branch', turn=1)
            self.assertEqual(state['mode'], 'manual')
            self.assertEqual(state['game']['moves'], [24])
            self.assertEqual(state['game_id'], game_id)
            state = perform('play', action=1)
            self.assertEqual(state['game']['moves'], [24, 1])
            for turn in (-1, 2, True):
                with self.assertRaises(ValueError):
                    app.submit('branch', dict(version=state['version'], turn=turn))
            with self.assertRaises(Conflict):
                app.submit('step', dict(version=state['version'] - 1))
            discovered['newly-published'] = Path(MODEL)
            before = state['game']
            state = perform('refresh')
            self.assertEqual(state['game'], before)
            self.assertEqual(state['catalog_revision'], 1)
            catalog = app.catalog()
            self.assertEqual(len(catalog['models']), 2)
            self.assertIn('manifest', catalog['models'][0])
            self.assertEqual(catalog['evaluation']['device'], self.config['evaluation']['device'])
            # Finish a manual game, then recover a nonterminal position by branching.
            perform('new', model='test', size=5, rule='freestyle', human=1, visits=16, mode='manual')
            for action in [0, 5, 1, 6, 2, 7, 3, 8, 4]:
                state = perform('play', action=action)
            self.assertTrue(state['game']['finished'])
            with self.assertRaises(Conflict):
                app.submit('analyze', dict(version=state['version']))
            state = perform('branch', turn=8)
            self.assertFalse(state['game']['finished'])
            self.assertEqual(state['game']['turn'], 8)
        finally:
            app.close()


if __name__ == '__main__':
    unittest.main()
