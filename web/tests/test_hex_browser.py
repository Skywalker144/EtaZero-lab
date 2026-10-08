"""Real Hex board geometry and White analysis through the local web application."""
import os
from pathlib import Path
from threading import Thread
import pytest
from etazero.config import ROOT
from etazero.engine_config import load_engine_config
from web.app import App
from web.server import make_server


@pytest.mark.skipif(not os.environ.get('ETAZERO_WEB_TEST_MODEL'),reason='Requires a published model and host browser/CUDA')
def test_hex_cells_goals_moves_and_white_analysis(tmp_path):
    from playwright.sync_api import sync_playwright
    model=Path(os.environ['ETAZERO_WEB_TEST_MODEL'])
    profile=ROOT/'tests/fixtures/configs/smoke_test'
    config=load_engine_config(profile)
    config['analysis'].update(rule='hex',board_size=5,visits=9,inference_precision='float16')
    match=load_engine_config(profile,True)
    config.update(opening=match['opening'],hex_opening=match['hex_opening'])
    app=App(ROOT/'build/etazero',{'hex-model':model},config,5)
    server=make_server(app,'127.0.0.1',0)
    Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch()
            page=browser.new_page(viewport={'width':1440,'height':1000})
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/')
            page.wait_for_function("document.querySelector('#model').options.length === 1")
            page.locator('#size').select_option('5')
            page.locator('#rule').select_option('hex')
            page.locator('#mode').select_option('manual')
            page.locator('#visits').fill('9')
            page.locator('#new-game').click()
            page.wait_for_function("document.querySelectorAll('#board .board-point').length === 25")
            assert page.locator('#board polygon').count()==25
            assert page.locator('#board .hex-goal').count()==4
            assert '"balance_exponent": 20' in page.locator('#opening-config').text_content()
            assert 'avg_dist_factor' not in page.locator('#opening-config').text_content()
            positions=page.locator('#board .board-point').evaluate_all("cells=>cells.map(c=>[parseFloat(c.style.left),parseFloat(c.style.top)])")
            assert positions[5][0]>positions[0][0] and positions[5][1]>positions[0][1]
            page.locator('#board [data-action="1"]').click()
            page.wait_for_function("document.querySelectorAll('#board .stone').length === 1")
            page.locator('#analyze').click()
            page.wait_for_function("document.querySelectorAll('#candidates tr').length > 0")
            # A screenshot is reviewable evidence of the six-sided cells and goals.
            page.screenshot(path=str(tmp_path/'hex-board.png'),full_page=True)
            assert not errors
            browser.close()
    finally:
        app.close();server.shutdown();server.server_close()
