"""The shared HTTP boundary serves V4's native schema without V3 imports."""
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from etazero.web_app import App, Conflict
from web.http import make_server
from test_web_session import GAMES, catalog_for, idle, analyzed
from test_web_native import profile


def request(server, path, payload=None, **headers):
    data = None if payload is None else json.dumps(payload).encode()
    try:
        response = urlopen(Request(f'http://127.0.0.1:{server.server_port}{path}', data=data,
                                   headers={'Content-Type': 'application/json', **headers}), timeout=30)
    except HTTPError as error:
        response = error
    with response:
        body = response.read()
        return response.status, (json.loads(body) if response.headers['Content-Type'].startswith('application/json')
                                  else body), response.headers


def command(server, operation, **payload):
    status, state, _ = request(server, '/api/state')
    assert status == 200
    return request(server, '/api/' + operation, dict(version=state['version'], game_id=state['game_id'], **payload))


@pytest.mark.parametrize('game', GAMES)
def test_http_native_game_analysis_cancel_and_recovery(tmp_path, game):
    config = profile(tmp_path / 'gtp.cfg', game, gumbel=True, maxVisits=8, reportAnalysisWinratesAs='WHITE')
    catalog = catalog_for(config, game)
    with App(lambda: catalog) as app:
        server = make_server(app, '127.0.0.1', 0, files={}, conflict=Conflict)
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            status, discovered, _ = request(server, '/api/catalog')
            assert status == 200 and {key: discovered[key] for key in catalog} == catalog
            assert len(discovered['xiangqi_moves']) == 2086
            assert command(server, 'new', model='model', mode='manual')[0] == 202
            start = idle(app)
            status, state, headers = request(server, '/api/state')
            assert status == 200 and state == start
            assert state['game'] == app.session.engine.state()
            assert state['game']['analysisPerspective'] == 'W'
            assert headers['Cache-Control'] == 'no-store' and headers['X-Content-Type-Options'] == 'nosniff'
            first = 'b0c2' if game == 'xiangqi' else 'C3'
            assert command(server, 'play', move=first)[0] == 202
            idle(app)
            stale = dict(version=start['version'], game_id=start['game_id'], move=first)
            assert request(server, '/api/play', stale)[0] == 409
            assert command(server, 'analyze')[0] == 202
            before = analyzed(app)
            status, progress, _ = request(server, f'/api/state?since={before["version"]}')
            assert status == 200 and progress['version'] > before['version']
            assert 'rootInfo' in progress['analysis']['report'] and 'policy' in progress['analysis']['report']
            assert 'requests' not in progress['analysis']['report']  # No fabricated legacy metrics.
            assert command(server, 'stop')[0] == 202
            stopped = idle(app)
            assert stopped['game']['moves'] == [['B', first]]
            assert command(server, 'undo')[0] == 202
            assert idle(app)['game'] == start['game']
            assert command(server, 'new', model='model', opening='balanced')[0] == 400
            assert app.snapshot()['game'] == start['game']
        finally:
            app.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        assert not thread.is_alive()


def test_shared_http_errors_and_static_assets(tmp_path):
    asset = tmp_path / 'page.html'
    asset.write_text('<!doctype html><title>Workbench</title>')
    with App(lambda: dict(runs=[], errors=[])) as app:
        server = make_server(app, '127.0.0.1', 0, files={'/': asset}, conflict=Conflict)
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            assert request(server, '/')[1] == asset.read_bytes()
            assert request(server, '/missing')[0] == 404
            assert request(server, '/api/state?since=bad')[0] == 400
            assert request(server, '/api/new', [1, 2])[0] == 400
            assert request(server, '/api/new', {'version': 0}, Origin='https://other.invalid')[0] == 403
            assert request(server, '/api/unsupported', {'version': 0})[0] == 404
            assert not app.snapshot()['busy'] and app.snapshot()['game'] is None
        finally:
            app.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
