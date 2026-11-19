"""Real CUDA MuZero through source GTP commands, clocks, raw NN and latent PVs."""
import json
import os
from queue import Queue
import re
import subprocess
from threading import Thread

import pytest
import torch

from export_muzero import export_components
from test_gtp import gtp_config, command, run_gtp
from test_analysis import request, run as run_analysis
from test_muzero_model import make_model
from test_muzero_analysis import bundle_for, config_for

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA GTP')


@pytest.mark.parametrize('game,gumbel,fp16', [('go', False, False), ('go', True, True),
    ('freestyle', False, True), ('standard', True, False), ('renju', True, True), ('hex', False, True)])
def test_commands_model_lifetime_and_latent_pv(tmp_path, game, gumbel, fp16):
    model = bundle_for(tmp_path/'model', game, kind='transformer' if game == 'hex' else 'regular')
    config = gtp_config(tmp_path/'gtp.cfg', game, muzero=True, gumbel=gumbel, random=False,
                        cudaUseFP16=str(fp16).lower(), numSearchThreads=1, analysisPVLen=12)
    cmds = ['kata-get-models', 'play B C3', 'play W E5', 'printsgf', 'undo', 'printsgf',
        'kata-raw-nn 0', 'clear_cache', 'clear_board',
        'kata-search_analyze B 0 allow B A1 100 allow W A1 100 pvVisits true pvEdgeVisits true',
        'debug_moves A1 A1 A1 rawstats', 'kata-set-param useGraphSearch true', 'kata-set-param maxVisits 8',
        'genmove B', 'printsgf', 'undo', 'printsgf', 'boardsize 6', 'boardsize 4 5' if game == 'go' else 'boardsize 4',
        'kata-raw-nn 0', 'boardsize 5', 'kata-search W', 'kata-raw-human-nn 0',
        'kata-set-param maxVisits 1', 'kata-search W', 'kata-search W', 'quit']
    r = run_gtp(config, cmds, model)
    assert {i for i,(err,_) in enumerate(r) if err} == {11,17,22}, r
    info = json.loads(r[0][1])[0]
    assert info['version'] == (17 if game == 'go' else 18)
    assert info['internalName'] == 'analysis-test-prediction'
    assert info['maxBatchSize'] == 8  # Native GTP derives this when nnMaxBatchSize is absent.
    assert info['usingFP16'] == str(fp16).lower()
    assert 'W[ea]' in r[3][1] and 'W[ea]' not in r[5][1]
    assert 'whiteWin ' in r[6][1] and 'whiteLoss ' in r[6][1] and 'draw ' in r[6][1]
    # Independently request the same true position through the already-connected
    # JSON endpoint. GTP undo/raw-NN must preserve both the player and full input.
    analysis = config_for(tmp_path/'analysis.cfg', game, cudaUseFP16=str(fp16).lower(),
                          numSearchThreads=1, maxVisits=1)
    report = run_analysis(analysis, [request('same', rules='tromp-taylor' if game == 'go' else game,
        boardXSize=5, boardYSize=5, moves=[['B', 'C3']])], model)[0]
    scalars = dict(re.findall(r'(?m)^(whiteWin|whiteLoss|draw|noResult) ([^\n]+)', r[6][1]))
    tol = 1e-3 if fp16 else 5e-5
    assert .5*(1+float(scalars['whiteWin'])-float(scalars['whiteLoss'])) == pytest.approx(report['rootInfo']['rawWinrate'], abs=tol)
    assert float(scalars['draw']) == pytest.approx(report['rootInfo']['rawDrawProb'], abs=tol)
    raw_policy = r[6][1].split('\npolicy\n')[1].replace('policyPass', '').split()
    assert len(raw_policy) == len(report['policy'])
    for a,b in zip(raw_policy,report['policy']):
        assert a == 'NAN' if b < 0 else float(a) == pytest.approx(b, abs=tol)
    assert 'pv A1 A1 A1 A1' in r[9][1], r[9]
    assert 'pvVisits ' in r[9][1] and 'pvEdgeVisits ' in r[9][1]
    assert 'White winloss:' in r[10][1] and 'illegal move sequence' not in r[10][1]
    assert re.fullmatch(r'[A-HJ-Z][1-9][0-9]*|pass', r[13][1])
    assert ';B[' in r[14][1] and ';B[' not in r[16][1]
    log = config.with_suffix('.log').read_text()
    assert 'All cleaned up, quitting' in log and 'Unused key' not in log
    assert log.count('Loaded neural net with nnXLen') == 1  # Fixed bundle survives board-size changes.
    assert 'NN rows:' in log and 'NN batches:' in log
    assert 'NN rows: 0' in log and re.search(r'NN avg batch size: -?nan', log)


def test_pass_latent_pv_is_not_replayed_as_go_encore(tmp_path):
    model = bundle_for(tmp_path/'model', 'go')
    config = gtp_config(tmp_path/'gtp.cfg', muzero=True, random=False, numSearchThreads=1, analysisPVLen=12,
                        rules='japanese', preventCleanupPhase='true')
    r = run_gtp(config, ['kata-search_analyze B 0 allow B pass 100 allow W pass 100 pvVisits true pvEdgeVisits true',
                         'lz-genmove_analyze B 0 allow B pass 100 allow W pass 100', 'quit'], model)
    assert all(not err for err,_ in r)
    assert 'pv pass pass pass pass' in r[0][1] and 'pv pass pass pass pass' in r[1][1]


def test_metadata_rejected_settings_and_sgf_boundaries(tmp_path):
    model = bundle_for(tmp_path/'model', 'go', metadata=True)
    config = gtp_config(tmp_path/'gtp.cfg', muzero=True, random=False, requireMaxBoardSize='true')
    bad = tmp_path/'bad.sgf'; bad.write_text('(;GM[1]FF[4]SZ[6]RU[Tromp-Taylor];B[aa])')
    other = tmp_path/'other.sgf'; other.write_text('(;GM[4]FF[4]SZ[5]RU[renju];B[aa])')
    good = tmp_path/'good.sgf'; good.write_text('(;GM[1]FF[4]SZ[5]RU[Tromp-Taylor];B[aa];W[ee])')
    r = run_gtp(config, ['kata-search B', 'kata-raw-nn 0', 'kata-set-param ponderingEnabled true',
        'kata-set-param humanSLProfile rank_5d', 'kata-raw-nn 0', 'kata-set-param useGraphSearch true',
        'kata-set-param maxVisits 8', 'kata-search B', 'boardsize 4',
        f'loadsgf {bad}', f'loadsgf {other}', f'loadsgf {good}', 'printsgf', 'undo', 'printsgf',
        'kata-set-param alwaysComputePassAliveUnderSuicideRules true', 'kata-set-param excludeTerritoryAdjacentToAtari true',
        'kata-search W', 'kata-get-models', 'quit'], model)
    assert {i for i,(err,_) in enumerate(r) if err} == {0,1,2,5,8,9,10}, r
    assert 'SGF game is not supported' in r[10][1]
    assert 'W[ee]' in r[12][1] and 'W[ee]' not in r[14][1]
    assert json.loads(r[18][1])[0]['usesHumanSLProfile'] is True


def test_go_handicap_and_source_benchmark(tmp_path):
    torch.manual_seed(513)
    model = make_model('go', pos_len=7); model.eval()
    bundle = tmp_path/'model'; export_components(model, bundle, 'gtp-benchmark')
    config = gtp_config(tmp_path/'gtp.cfg', muzero=True, random=False, defaultBoardSize=7, maxVisits=4)
    r = run_gtp(config, ['fixed_handicap 2', 'kata-search W', 'clear_board', 'place_free_handicap 2',
        'kata-search W', 'clear_board', 'kata-benchmark 4', 'kata-search B', 'quit'], bundle)
    assert all(not err for err,_ in r), r
    assert 'visits' in r[6][1].lower() and 'nnEvals' in r[6][1]


@pytest.mark.parametrize('game', ['go', 'renju'])
def test_go_handicap_resignation_delay_does_not_apply_to_placement(tmp_path, game):
    model = make_model(game)
    with torch.no_grad():
        model.prediction.value_head.linear_valuehead.weight.zero_()
        model.prediction.value_head.linear_valuehead.bias.copy_(torch.tensor([-8., -8., 8., -8.]))
    model.eval()
    bundle = tmp_path/'model'; export_components(model, bundle, 'resign-test')
    config = gtp_config(tmp_path/'gtp.cfg', game, muzero=True, random=False,
        maxVisits=1, numSearchThreads=1, allowResignation='true', resignThreshold=-.5,
        resignConsecTurns=1, resignMinMovesPerBoardArea=0)
    r = run_gtp(config, ['komi 0', 'play B C3', 'genmove W', 'quit'], bundle)
    assert all(not err for err,_ in r), r
    assert (r[2][1] == 'resign') == (game != 'go')


@pytest.mark.parametrize('gumbel', [False, True])
def test_live_ponder_analyze_cancel_and_native_clock(tmp_path, gumbel):
    model = bundle_for(tmp_path/'model', 'go')
    config = gtp_config(tmp_path/'gtp.cfg', muzero=True, gumbel=gumbel, random=False,
        maxVisits=1000000000, maxTime=100, lagBuffer=0, ponderingEnabled='true')
    check_live_gtp(config, model, tmp_path/'live.log')


def check_live_gtp(config, model, log, human=None):
    messages = Queue()
    with log.open('w') as stderr:
        proc = subprocess.Popen(command(config, model, human), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=stderr, text=True, bufsize=1)
        def read():
            for line in proc.stdout: messages.put(line.strip())
            messages.put(None)
        reader = Thread(target=read, daemon=True); reader.start()
        def send(id, text):
            proc.stdin.write(f'{id} {text}\n'); proc.stdin.flush()
        def until(prefix):
            for _ in range(1000):
                value = messages.get(timeout=20)
                assert value is not None, log.read_text()
                if value.startswith(prefix): return value
            pytest.fail(prefix)
        try:
            send(1, 'time_settings 0 0.1 1'); until('=1')
            send(2, 'time_left B 0.1 1'); until('=2')
            send(3, 'genmove B'); assert 'resign' not in until('=3')
            send(4, 'kata-debug-print-tc'); until('=4')
            send(5, 'kata-analyze W 1 rootInfo true pvVisits true'); until('=5'); until('info ')
            send(6, 'stop'); until('=6')
            send(7, 'kata-set-param maxTime 100'); until('=7')
            send(8, 'kata-search_analyze_cancellable W 1'); until('=8'); until('info ')
            send(9, 'stop'); until('play cancelled'); until('=9')
            send(10, 'clear_cache'); until('=10')
            send(11, 'quit'); until('=11')
            proc.stdin.close(); assert proc.wait(timeout=20) == 0, log.read_text()
            reader.join(timeout=2)
        finally:
            if proc.poll() is None: proc.kill(); proc.wait(timeout=10)
    assert 'Async bot thread failed' not in log.read_text()
