"""Joint-action GTP preserves full Xiangqi histories and native search controls."""
import json
import re
import subprocess
from queue import Queue
from threading import Thread

import pytest
import torch

from export_model_pytorch import write_model_binary
from katago.game import xiangqi as xq
from katago.train.model_pytorch import Model
from test_analysis import analysis_config, request, run as run_analysis
from test_gtp import command, gtp_config, run_gtp
from test_muzero_analysis import bundle_for, config_for, cuda
from test_muzero_model import component_config


CHECK_FEN = '5k3/3RR4/9/9/4P4/9/9/9/9/4K4 w - - 0 1'
CHECKS = ['e8f8', 'f9e9', 'f8e8', 'e9f9', 'e8f8', 'f9e9', 'f8e8', 'e9f9']
MOVES = [f'{chr(97+x)}{y}{chr(97+u)}{v}' for x,y,u,v in xq.MOVES]


def config_for_gtp(path, **changes):
    values = dict(defaultBoardXSize=9, defaultBoardYSize=10)
    values.update(changes)
    result = gtp_config(path, 'xiangqi', **values)
    # Exercise the native Xiangqi defaults, including its rectangular physical
    # board and padded NN canvas, whenever no explicit dimensions were requested.
    if not any(k.startswith('defaultBoard') for k in changes):
        result.write_text(''.join(line for line in result.read_text().splitlines(True)
                                  if not line.startswith('defaultBoard')))
    return result


def raw_policy(text):
    values = dict(line.split() for line in text.split('\npolicy\n')[1].splitlines())
    assert list(values) == MOVES
    assert 'policyPass' not in text
    assert sum(float(p) for p in values.values() if p != 'NAN') == pytest.approx(1, abs=2e-4)
    return values


@pytest.mark.parametrize('gumbel', [False, True])
def test_joint_moves_state_boundaries_and_analysis(tmp_path, gumbel):
    config = config_for_gtp(tmp_path/'gtp.cfg', gumbel=gumbel)
    other = tmp_path/'go.sgf'
    other.write_text('(;GM[1]SZ[9]RU[Tromp-Taylor];B[aa])')
    commands = ['printsgf', 'play W b0c2', 'play B pass', 'play B b0c2', 'printsgf',
        'boardsize 10', 'set_position B b0c2', 'set_position fen invalid', 'loadsgf '+str(other),
        'kata-set-rules tromp-taylor', 'genmove B', 'kata-search_analyze B 0', 'kata-analyze B 0',
        'kata-raw-nn B 0', 'printsgf', 'undo', 'printsgf', 'kata-raw-nn 0', 'kata-raw-nn 1',
        'kata-search_analyze B 0 allow B b0c2 1 pvVisits true pvEdgeVisits true rootInfo true',
        'debug_moves b0c2 rawstats', 'kata-set-rules xiangqi', 'kata-set-param maxVisits 8',
        'genmove B', 'printsgf', 'undo', 'printsgf', 'clear_board', 'printsgf', 'quit']
    r = run_gtp(config, commands)
    assert {i for i,(err,_) in enumerate(r) if err} == {1,2,5,6,7,8,9,10,11,12,13,18}, r
    assert 'GM[7]SZ[9:10]' in r[0][1]
    assert r[4][1] == r[14][1]
    assert r[0][1] == r[16][1] == r[26][1] == r[28][1]
    policy = raw_policy(r[17][1])
    assert policy['b0c2'] != 'NAN' and policy['h9g7'] == 'NAN'
    assert 'info move b0c2 ' in r[19][1] and 'pv b0c2' in r[19][1]
    assert 'pvVisits ' in r[19][1] and 'pvEdgeVisits ' in r[19][1]
    assert 'Xiangqi FEN:' in r[20][1] and 'White winloss:' in r[20][1]
    assert r[23][1] in MOVES


def check_imported_history(tmp_path, config, model='/dev/null', human=None):
    # Import an origin that already owns two plies, then five SGF moves.
    # Rebuilding from its FEN would lose the first repeated checking cycle.
    sgf = tmp_path/'checking.sgf'
    sgf.write_text(f'(;GM[7]SZ[9:10]XQFEN[{CHECK_FEN}]XQHIST[e8f8][f9e9]XQTURN[97]'
                   + ''.join(f';{"B" if i%2 == 0 else "W"}[{m}]' for i,m in enumerate(CHECKS[2:7])) + ')')
    saved = tmp_path/'saved.sgf'
    commands = ['loadsgf '+str(sgf), 'printsgf', 'kata-search_analyze W 0 allow W e9f9 1',
        'play W e9f9', 'final_score', 'kata-raw-nn 0', 'genmove B', 'undo',
        'kata-set-param ignorePreRootHistory true', 'kata-set-rules xiangqi', 'printsgf',
        'printsgf '+str(saved), 'clear_board', 'loadsgf '+str(saved), 'play W e9f9', 'final_score',
        'final_status_list alive', 'undo', 'showboard',
        'set_position fen 5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1', 'final_score', 'clear_board', 'quit']
    r = run_gtp(config, commands, model, human)
    assert {i for i,(err,_) in enumerate(r) if err} == {5,6,16}, r
    assert r[4][1] == r[15][1] == 'W+'
    assert r[1][1] == r[10][1]
    assert 'XQHIST[e8f8][f9e9]' in r[10][1] and 'XQTURN[97]' in r[10][1]
    assert 'info move e9f9 ' in r[2][1] and r[2][1].endswith('play e9f9')
    assert r[20][1] == '0'


@pytest.mark.parametrize('gumbel', [False, True])
def test_sgf_undo_and_rule_changes_keep_preexisting_check_history(tmp_path, gumbel):
    config = config_for_gtp(tmp_path/'history.cfg', gumbel=gumbel)
    check_imported_history(tmp_path, config)


def ordinary_model(path, kind='regular'):
    net = Model(component_config('xiangqi', kind), 10)
    net.initialize(); net.eval()
    with path.open('wb') as stream:
        write_model_binary(net, stream, 'gtp-xiangqi')
    return path


@cuda
@pytest.mark.parametrize('muzero,gumbel,fp16,human', [
    (False,False,False,False), (False,True,True,False), (False,False,True,True),
    (True,False,False,False), (True,True,True,False)])
def test_real_models_joint_policy_and_full_history(tmp_path, muzero, gumbel, fp16, human):
    torch.manual_seed(2048)
    kind = 'transformer' if gumbel else 'regular'
    model = bundle_for(tmp_path/'bundle', kind=kind) if muzero else ordinary_model(tmp_path/'model.bin', kind)
    secondary = ordinary_model(tmp_path/'human.bin') if human else None
    changes = dict(cudaUseFP16=str(fp16).lower(), inputsUseNHWC=str(fp16).lower())
    if human:
        changes.update(humanSLChosenMoveProp=.5, humanSLRootExploreProbWeightful=.4,
                       humanSLPlaExploreProbWeightless=.3, humanSLOppExploreProbWeightful=.3)
    config = config_for_gtp(tmp_path/'cuda.cfg', muzero=muzero, gumbel=gumbel, random=False, **changes)
    check_imported_history(tmp_path, config, model, secondary)
    moves = [['B','b0c2'], ['W','h9g7'], ['B','c2b0'], ['W','g7h9']]
    r = run_gtp(config, [*(f'play {p} {m}' for p,m in moves), 'play B b0c2', 'undo',
        'kata-raw-nn 0', 'kata-raw-human-nn 0', 'kata-get-models',
        'kata-search_analyze B 0 allow B b0c2 100 allow W b0c2 100 pvVisits true pvEdgeVisits true'
            if muzero else 'kata-search_analyze B 0 allow B b0c2 1 pvVisits true pvEdgeVisits true',
        'debug_moves b0c2 b0c2 b0c2 rawstats' if muzero else 'debug_moves b0c2 rawstats', 'quit'], model, secondary)
    assert {i for i,(err,_) in enumerate(r) if err} == (set() if human else {7}), r
    info = json.loads(r[8][1])
    assert len(info) == 1 + human and all(m['version'] == 19 for m in info)
    acfg = (config_for if muzero else analysis_config)(tmp_path/'analysis.cfg',
        **({} if muzero else dict(random=False)), numSearchThreads=1, maxVisits=1,
        cudaUseFP16=str(fp16).lower(), inputsUseNHWC=str(fp16).lower())
    report = run_analysis(acfg, [request(moves=moves)], model, secondary)[0]
    tol = 1e-3 if fp16 else 5e-5
    for idx, key in [(6,'policy'), *([(7,'humanPolicy')] if human else [])]:
        policy = raw_policy(r[idx][1])
        for m,p in zip(MOVES, report[key]):
            assert policy[m] == 'NAN' if p < 0 else float(policy[m]) == pytest.approx(p, abs=tol)
    scalars = dict(re.findall(r'(?m)^(whiteWin|whiteLoss|draw|noResult) ([^\n]+)', r[6][1]))
    assert .5*(1+float(scalars['whiteWin'])-float(scalars['whiteLoss'])) == pytest.approx(report['rootInfo']['rawWinrate'], abs=tol)
    assert float(scalars['draw']) == pytest.approx(report['rootInfo']['rawDrawProb'], abs=tol)
    assert scalars['noResult'] == '0.000000'
    if muzero:
        assert 'pv b0c2 b0c2 b0c2 b0c2' in r[9][1]  # Latent branches don't replay real rules.
        assert 'illegal move sequence' not in r[10][1]
    assert 'White winloss:' in r[10][1]
    log = config.with_suffix('.log').read_text()
    assert 'All cleaned up, quitting' in log and 'Unused key' not in log
    assert 'Xiangqi FEN:' in log and 'NN rows:' in log and 'NN batches:' in log


def test_auto_patterns_save_complete_joint_action_positions(tmp_path):
    poses = tmp_path/'poses'; poses.mkdir()
    config = config_for_gtp(tmp_path/'patterns.cfg', autoAvoidRepeatDir=poses,
        autoAvoidRepeatMinTurnNumber=0, autoAvoidRepeatMaxTurnNumber=100000,
        autoAvoidRepeatUtility=.1, autoAvoidRepeatLambda=.9, autoAvoidRepeatMaxPoses=100)
    r = run_gtp(config, ['play B b0c2', 'genmove W', 'quit'])
    assert all(not err for err,_ in r), r
    samples = [json.loads(line) for p in poses.glob('9x10/*') for line in p.read_text().splitlines()]
    assert len(samples) == 1
    assert samples[0]['game'] == 'xiangqi' and samples[0]['xiangqiHistory'] == ['b0c2']
    assert samples[0]['hintLoc'] == r[1][1] and samples[0]['initialTurnNumber'] == 1
    r = run_gtp(config, ['play B b0c2', 'kata-search W', 'quit'])
    assert all(not err for err,_ in r), r


def check_live_controls(tmp_path, config, model='/dev/null'):
    messages = Queue(); log = tmp_path/'live.log'
    with log.open('w') as stderr:
        proc = subprocess.Popen(command(config, model), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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
            send(3, 'genmove B'); assert until('=3').split()[1] in MOVES
            send(4, 'kata-debug-print-tc'); until('=4')
            send(5, 'kata-analyze W 1 rootInfo true pvVisits true'); until('=5')
            assert until('info ').split()[2] in MOVES
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


@pytest.mark.parametrize('gumbel', [False, True])
def test_native_clock_ponder_analysis_and_cancellation(tmp_path, gumbel):
    config = config_for_gtp(tmp_path/'live.cfg', gumbel=gumbel,
        maxVisits=1000000000, maxTime=100, lagBuffer=0, ponderingEnabled='true')
    check_live_controls(tmp_path, config)


@cuda
@pytest.mark.parametrize('gumbel', [False, True])
def test_latent_clock_ponder_analysis_and_cancellation(tmp_path, gumbel):
    model = bundle_for(tmp_path/'model')
    config = config_for_gtp(tmp_path/'live.cfg', muzero=True, gumbel=gumbel, random=False,
        cudaUseFP16='true', maxVisits=1000000000, maxTime=100, lagBuffer=0, ponderingEnabled='true')
    check_live_controls(tmp_path, config, model)
