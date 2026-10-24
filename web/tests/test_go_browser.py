"""Go profile defaults and session overrides through the browser and native rules."""
from copy import deepcopy
import json
from threading import Thread

import pytest

from etazero.config import ROOT, load_config
from etazero.engine_config import load_engine_config
from web.app import App
from web.server import make_server


def test_go_profile_defaults_overrides_and_page_reload(tmp_path):
    if not (ROOT / 'build/etazero').is_file():
        pytest.skip('Build native engine first')
    import torch
    from playwright.sync_api import sync_playwright
    from etazero.models.network import make_network, inference_network

    # A small real Go network keeps this test independent of published runs.
    torch.set_num_threads(1)
    config = load_config(ROOT / 'tests/fixtures/configs/smoke_test', run_dir=tmp_path / 'network')
    config['environment'].update(sizes='5', size_weights='1', rules='go', rule_weights='1')
    config['network'].update(canvas=5, channels=24, blocks=1)
    model = tmp_path / 'japanese/models/iteration_000001/model.pt'
    model.parent.mkdir(parents=True)
    torch.jit.script(inference_network(make_network(config).eval())).save(str(model))
    (model.parent / 'manifest.json').write_text(json.dumps(dict(
        id='go-browser', canvas=5, game='go', algorithm='alphazero', bytes=model.stat().st_size,
        checkpoint=dict(iteration=1))))
    other = tmp_path / 'aga/models/iteration_000001/model.pt'
    other.parent.mkdir(parents=True)
    other.write_bytes(model.read_bytes())
    (other.parent / 'manifest.json').write_bytes((model.parent / 'manifest.json').read_bytes())

    profile = load_engine_config(ROOT / 'tests/fixtures/configs/minimal_test', environ={})
    profile['analysis'].update(rule='go', board_size=5, komi=.5, go_rules='japanese',
                               device='cpu', inference_precision='float32', visits=2,
                               nn_symmetry=0, root_num_symmetries_to_sample=1)
    app = App(ROOT / 'build/etazero', {'japanese': model, 'aga': other}, profile, 5)
    other_run = str(other.parent.parent.parent)
    app.runs[other_run]['analysis_config'].update(go_rules='aga', komi=6.5)
    server = make_server(app, '127.0.0.1', 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport=dict(width=1440, height=900))
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function("!document.querySelector('#new-game').disabled")
            assert page.input_value('#go-rules') == 'japanese'
            assert float(page.input_value('#komi')) == .5
            page.select_option('#run', other_run)
            # Presets not hardcoded into the HTML must also survive selection.
            assert page.input_value('#go-rules') == 'aga'
            assert float(page.input_value('#komi')) == 6.5
            page.select_option('#mode', 'manual')

            def create():
                previous = app.snapshot()['game_id']
                page.click('#new-game')
                page.wait_for_function("id => !state.busy && (state.game_id !== id || state.error)", arg=previous)
                state = app.snapshot()
                assert state['error'] is None
                return state['game']

            game = create()
            assert (game['go_rules'], game['komi']) == ('aga', 6.5)
            page.select_option('#go-rules', 'tromp-taylor')
            page.fill('#komi', '-.5')
            game = create()
            assert (game['go_rules'], game['komi']) == ('tromp-taylor', -.5)
            effective = json.loads(page.locator('#engine-config').text_content())
            assert (effective['go_rules'], effective['komi']) == ('tromp-taylor', -.5)
            before = deepcopy(app.snapshot()['game'])
            page.reload()
            page.wait_for_function("state?.game && !document.querySelector('#new-game').disabled")
            assert page.input_value('#run') == other_run
            assert page.input_value('#go-rules') == 'tromp-taylor'
            assert float(page.input_value('#komi')) == -.5
            assert app.snapshot()['game'] == before
            # Live reports must not erase edits intended for the next game.
            page.fill('#komi', '2.5')
            page.click('#pass')
            page.wait_for_function("state.game.turn === 1 && !state.busy")
            assert float(page.input_value('#komi')) == 2.5
            assert app.snapshot()['game']['komi'] == -.5
            page.select_option('#run', str(model.parent.parent.parent))
            assert page.input_value('#go-rules') == 'japanese'
            assert float(page.input_value('#komi')) == .5
            assert errors == []
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        app.close()
