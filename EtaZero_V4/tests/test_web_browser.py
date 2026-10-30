"""Browser interactions use the real V4 GTP controller and native reports."""
import json
from pathlib import Path
import subprocess
from threading import Thread

import pytest
from playwright.sync_api import sync_playwright

from etazero import ROOT
from etazero.web_app import App
from web.native_server import make_server, catalog_for as server_catalog
from test_web_session import catalog_for, GAMES
from test_web_native import profile
from test_benchmarknn import cuda, model_for


def ready(page, expression='!state.busy && state.phase === "idle"'):
    page.wait_for_function('state && !pending && (' + expression + ')', timeout=30000)
    error = page.locator('#error')
    assert not error.is_visible(), error.inner_text()


def open_game(page, url):
    page.goto(url)
    page.locator('#new-game').wait_for()
    page.wait_for_function('!document.getElementById("new-game").disabled')
    page.select_option('#mode', 'manual')
    page.click('#new-game')
    ready(page, 'state.game && !state.busy')


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('gumbel', [False, True])
def test_native_browser_board_history_analysis_and_parameters(tmp_path, game, gumbel):
    config = profile(tmp_path / 'gtp.cfg', game, gumbel=gumbel, maxVisits=8, reportAnalysisWinratesAs='WHITE')
    with App(lambda: catalog_for(config, game)) as app:
        server=make_server(app,'127.0.0.1',0)
        thread=Thread(target=server.serve_forever);thread.start()
        try:
            with sync_playwright() as browser_api:
                browser=browser_api.chromium.launch(headless=True)
                page=browser.new_page(viewport={'width':1440,'height':1100})
                errors=[];page.on('pageerror',lambda error: errors.append(str(error)))
                open_game(page,f'http://127.0.0.1:{server.server_port}')
                assert page.evaluate('state.game.game') == game
                assert page.locator('#board [data-point]').count() == (90 if game=='xiangqi' else 25)
                before=app.snapshot()['game']
                if game=='xiangqi':
                    assert page.locator('.xiangqi-piece').count()==32
                    page.click('[data-point="b0"]');page.click('[data-point="c2"]')
                    first='b0c2'
                else:
                    page.click('[data-point="C3"]');first='C3'
                ready(page, 'state.game.moves.length === 1 && !state.busy')
                assert app.snapshot()['game']['moves']==[['B',first]]
                page.click('#step');ready(page, 'state.game.moves.length === 2 && !state.busy')
                page.locator('#timeline').evaluate("el => {el.value='1';el.dispatchEvent(new Event('input',{bubbles:true}));}")
                assert page.locator('#view-note').is_visible()
                assert page.evaluate('displayedGame().board')==app.snapshot()['history'][1]['board']
                page.click('#branch');ready(page,'state.game.moves.length === 1 && !state.busy')
                page.click('#undo');ready(page,'state.game.moves.length === 0 && !state.busy')
                assert app.snapshot()['game']==before
                page.click('#analyze');ready(page,'state.analysis?.report && state.phase === "analyzing"')
                assert page.locator('#perspective').inner_text().startswith('报告视角：'+('黑方' if game=='xiangqi' else '白方'))
                page.click('#stop');ready(page)
                report=app.snapshot()['analysis']['report']
                assert json.loads(page.locator('#raw-analysis').text_content())==report==app.session.engine.analysis()
                assert page.locator('#candidates tr').count()==len(report['moveInfos'])
                page.select_option('#sort','visits')
                visits=page.locator('#candidates tr td:nth-child(3)').all_text_contents()
                assert list(map(int,visits))==sorted((row['visits'] for row in report['moveInfos']),reverse=True)
                page.locator('#candidates [data-candidate]').first.click()
                assert page.locator('#candidate-detail').inner_text().find('PV ')>=0
                page.locator('#policy-panel summary').click()
                page.select_option('#policy-source', 'humanPolicy')
                assert '没有 humanPolicy' in page.locator('#policy-note').inner_text()
                assert page.locator('#policy-map .heat-cell').count() == 0
                page.select_option('#policy-source', 'policy')
                assert not errors, errors
                if game=='xiangqi':
                    page.locator('[aria-label="策略起点"]').wait_for()
                    page.select_option('[aria-label="策略起点"]','1,0')
                    assert 'b0' in page.locator('#policy-list').inner_text()
                else:
                    page.locator('#policy-map .heat-cell').first.wait_for()
                    assert page.locator('#policy-map .heat-cell').count()==25
                if not gumbel:
                    page.screenshot(path=f'/tmp/etazero-v4-web-{game}.png',full_page=True)
                page.fill('#visits','12');page.click('#apply-settings');ready(page)
                assert int(app.snapshot()['parameters']['maxVisits'])==12
                assert app.snapshot()['game']['moves']==[]
                page.reload();ready(page,'state.game && !state.busy')
                assert page.locator('#mode').input_value()=='manual'
                page.select_option('#theme','dark')
                assert page.locator('html').get_attribute('data-theme')=='dark'
                page.set_viewport_size({'width':390,'height':844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
                if game=='xiangqi' and not gumbel:
                    page.screenshot(path='/tmp/etazero-v4-web-xiangqi-mobile.png',full_page=True)
                assert not errors, errors
                browser.close()
        finally:
            app.close();server.shutdown();server.server_close();thread.join(timeout=5)


@cuda
@pytest.mark.parametrize('game', ['go', 'xiangqi'])
@pytest.mark.parametrize('muzero', [False, True])
def test_cuda_browser_humansl_policy_uses_native_values(tmp_path, game, muzero):
    import torch
    from test_humansl_protocols import model_for as human_model_for
    from katago.game.xiangqi import MOVES

    torch.manual_seed(711)
    primary = human_model_for(tmp_path / 'primary', game, muzero)
    human = human_model_for(tmp_path / 'human', game, muzero, metadata=game == 'go', width=8)
    config = profile(tmp_path / 'gtp.cfg', game, random=False, muzero=muzero, gumbel=muzero,
                     cudaUseFP16='true', maxVisits=8, humanSLChosenMoveProp=.5,
                     **({'humanSLProfile': 'rank_5d'} if game == 'go' else {}))
    with App(lambda: catalog_for(config, game, primary), human_model=human) as app:
        server = make_server(app, '127.0.0.1', 0)
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            with sync_playwright() as browser_api:
                browser = browser_api.chromium.launch(headless=True)
                page = browser.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                open_game(page, f'http://127.0.0.1:{server.server_port}')
                page.click('#step')
                ready(page, 'state.game.moves.length === 1 && !state.busy')
                report = app.snapshot()['analysis']['report']
                assert report['humanPolicy'] != report['policy']
                page.locator('#policy-panel summary').click()
                for source in ['humanPolicy', 'policy']:
                    page.select_option('#policy-source', source)
                    assert f'来源 {source}' in page.locator('#policy-note').inner_text()
                    if game == 'xiangqi':
                        page.select_option('[aria-label="策略起点"]', '1,0')
                        titles = page.locator('#policy-list button').evaluate_all('cells => cells.map(c => c.title)')
                        expected = [report[source][i] for i, m in enumerate(MOVES) if m[:2] == (1, 0)]
                    else:
                        titles = page.locator('#policy-map .heat-cell').evaluate_all('cells => cells.map(c => c.title)')
                        expected = report[source][:25]
                    assert [float(title.split(': ')[1]) for title in titles] == expected
                assert json.loads(page.locator('#raw-analysis').text_content()) == report
                assert not errors, errors
                browser.close()
        finally:
            app.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_native_cli_help_and_external_model_catalog(tmp_path):
    command=['bash',str(ROOT.parent/'web/webui.sh'),'--version',ROOT.name,'--help']
    result=subprocess.run(command,cwd=ROOT.parent,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'KataGo 原生工作台' in result.stdout and '--human-model' in result.stdout
    assert '正在构建' not in result.stdout
    model=tmp_path/'model.bin';model.touch()
    with pytest.raises(ValueError,match='requires an explicit native'):
        server_catalog(tmp_path,model=model)
    config=profile(tmp_path/'gtp.cfg','renju')
    catalog=server_catalog(tmp_path,config,model)
    row=catalog['runs'][0]['models'][0]
    assert row==dict(id=str(model),path=str(model),run='direct-model')
    assert catalog['default_model']==str(model)


@cuda
@pytest.mark.parametrize('game',['go','renju','hex','xiangqi'])
@pytest.mark.parametrize('muzero,gumbel',[(False,False),(False,True),(True,False),(True,True)])
def test_cuda_model_browser_move_and_native_analysis(tmp_path,game,muzero,gumbel):
    model=model_for(tmp_path/'model',game,muzero)
    config=profile(tmp_path/'gtp.cfg',game,random=False,muzero=muzero,gumbel=gumbel,
                   cudaUseFP16='true',maxVisits=8)
    with App(lambda:catalog_for(config,game,model)) as app:
        server=make_server(app,'127.0.0.1',0)
        thread=Thread(target=server.serve_forever);thread.start()
        try:
            with sync_playwright() as browser_api:
                browser=browser_api.chromium.launch(headless=True)
                page=browser.new_page(viewport={'width':1440,'height':1000})
                errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                open_game(page,f'http://127.0.0.1:{server.server_port}')
                page.click('#step');ready(page,'state.game.moves.length === 1 && !state.busy')
                state=app.snapshot()
                assert state['analysis']['move']==state['game']['moves'][-1][1]
                assert json.loads(page.locator('#raw-analysis').text_content())==state['analysis']['report']
                assert state['analysis']['report']['rootInfo']['visits']>0
                assert '★' in page.locator('#candidates').inner_text()
                page.click('#undo');ready(page,'state.game.moves.length === 0 && !state.busy')
                assert app.snapshot()['game']['result'] is None and not errors
                browser.close()
        finally:
            app.close();server.shutdown();server.server_close();thread.join(timeout=5)
