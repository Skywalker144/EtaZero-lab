"""Public KataGo requests exercise the real CPU model and asynchronous server."""
import json
import subprocess

import pytest
from etazero.config import write_native
from etazero.version import VERSION
from test_multigame_diagnostics import cpu_model
from test_multigame import BINARY


def source_process(tmp_path,algorithm='alphazero',human=False,prune=False,pda_player=None):
    _,model,config,profile=cpu_model(tmp_path,'go',algorithm,5,6)
    profile['analysis'].update(reuse_tree=False,num_analysis_threads=2,search_threads=3,
                               min_playouts_per_thread=0,root_symmetry_pruning=prune)
    if pda_player is not None:profile['analysis']['playout_doubling_advantage_player']=pda_player
    if human:profile['analysis'].update(human_model=str(model),human_sl_profile='',search_threads=1)
    # Keep the model configuration, replacing only the engine fields.
    text=config.read_text();start=text.index('[analysis]')
    native=tmp_path/'engine.cfg';write_native(profile,native)
    config.write_text(text[:start]+native.read_text())
    return subprocess.Popen([str(BINARY),'analysis','--stream','true','--protocol','katago',
        '--config',str(config),'--model',str(model),'--model-id','test','--device','cpu'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_source_requests_setup_turns_and_control(tmp_path,algorithm):
    p=source_process(tmp_path,algorithm)
    query=dict(id='board',boardXSize=5,boardYSize=5,rules='chinese',komi=0.5,
               initialStones=[['B','A1'],['W','B2']],initialPlayer='W',
               moves=[['W','C3'],['B','pass']],analyzeTurns=[0,2],priorities=[1,2],
               maxVisits=20,includePolicy=True,includePVVisits=True,includeNoResultValue=True,
               analysisPVLen=4,overrideSettings={'cpuctExploration':1.2,'reportAnalysisWinratesAs':'BLACK'})
    requests=[dict(id='v',action='query_version'),dict(id='m',action='query_models'),query,
              dict(id='cache',action='clear_cache')]
    out,err=p.communicate(''.join(json.dumps(x)+'\n' for x in requests),timeout=45)
    assert p.returncode==0,err
    replies=[json.loads(x) for x in out.splitlines()]
    assert len(replies)==5 and not any('error' in x for x in replies),replies
    assert next(x for x in replies if x['id']=='v')['version']=='EtaZero_V'+VERSION
    reports=[x for x in replies if x['id']=='board']
    assert {x['turnNumber'] for x in reports}=={0,2}
    for report in reports:
        assert report['rootInfo']['visits']>=20 and report['rootInfo']['currentPlayer']=='W'
        assert len(report['policy'])==26 and report['policy'][20]==-1 # A1 in source row-major coordinates.
        assert report['moveInfos']
        for move in report['moveInfos']:
            assert 0<=move['winrate']<=1 and move['visits']>=move['edgeVisits']
            assert len(move['pv'])==len(move['pvVisits'])==len(move['pvEdgeVisits'])<=4
    info=next(x for x in replies if x['id']=='m')['models'][0]
    assert info['version'] is None and info['usesHumanSLProfile'] is False


def test_source_cancel_queued_jobs_and_reject_unsupported_options(tmp_path):
    p=source_process(tmp_path)
    base=dict(boardXSize=5,boardYSize=5,rules='chinese',moves=[],maxVisits=1000000)
    queries=[dict(base,id='long'+str(i)) for i in range(4)]
    queries += [dict(id='stop',action='terminate_all'),dict(base,id='bad',includeOwnership=True)]
    out,err=p.communicate(''.join(json.dumps(x)+'\n' for x in queries),timeout=45)
    assert p.returncode==0,err
    replies=[json.loads(x) for x in out.splitlines()]
    assert len(replies)==6,replies
    assert 'error' in next(x for x in replies if x['id']=='bad')
    stopped=[x for x in replies if x['id'].startswith('long')]
    assert len(stopped)==4 and any(x.get('noResults') for x in stopped)
    assert all(x.get('noResults') or x['rootInfo']['visits']<1000000 for x in stopped)

@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
@pytest.mark.parametrize('width,height',[(3,5),(5,3)])
def test_rectangular_json_analysis(tmp_path,algorithm,width,height):
    p=source_process(tmp_path,algorithm)
    query=dict(id='rect',boardXSize=width,boardYSize=height,rules='chinese',moves=[['B','A1']],
               maxVisits=12,includePolicy=True,overrideSettings={'rootSymmetryPruning':True})
    out,err=p.communicate(json.dumps(query)+'\n',timeout=45)
    assert p.returncode==0,err
    report=json.loads(out)
    assert 'error' not in report,report
    assert len(report['policy'])==width*height+1
    assert report['policy'][(height-1)*width]==-1
    assert report['rootInfo']['visits']>=12
    assert report['moveInfos']


def gtp_process(tmp_path,algorithm='alphazero'):
    _,model,config,profile=cpu_model(tmp_path,'go',algorithm,5,9)
    profile['analysis'].update(visits=12,reuse_tree=algorithm=='alphazero',search_threads=2,min_playouts_per_thread=0)
    text=config.read_text();native=tmp_path/'engine.cfg';write_native(profile,native)
    config.write_text(text[:text.index('[analysis]')]+native.read_text())
    return subprocess.Popen([str(BINARY),'gtp','--config',str(config),'--model',str(model),
        '--model-id','test','--device','cpu'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)


def gtp_command(p,command):
    p.stdin.write(command+'\n');p.stdin.flush()
    response=[]
    while True:
        line=p.stdout.readline()
        assert line, (command,p.poll())
        if line=='\n':break
        response.append(line.rstrip('\n'))
    return '\n'.join(response)


def test_pipelined_state_changes_preserve_command_order(tmp_path):
    _,model,config,_=cpu_model(tmp_path,'go','alphazero',5,6)
    requests=[]
    for i in range(100):
        requests += [dict(id=f'{i}-new',command='new',size=5,rule='go'),
                     dict(id=f'{i}-first',command='play',action=0),
                     dict(id=f'{i}-second',command='play',action=1),
                     dict(id=f'{i}-undo',command='undo'),dict(id=f'{i}-state',command='state')]
    run=subprocess.run([str(BINARY),'analysis','--stream','true','--protocol','json',
        '--config',str(config),'--model',str(model),'--model-id','test','--device','cpu'],
        input=''.join(json.dumps(q)+'\n' for q in requests),text=True,capture_output=True,timeout=30)
    assert run.returncode==0,run.stderr
    replies=[json.loads(line) for line in run.stdout.splitlines() if json.loads(line).get('final')]
    assert len(replies)==len(requests)
    for request,response in zip(requests,replies):
        assert response['id']==request['id'] and response['ok'] and not response.get('cancelled'),response
        if request['command']=='state':assert response['state']['moves']==[0]

    p=gtp_process(tmp_path)
    commands=['1 boardsize 5']
    for i in range(20):commands += [f'{2*i+2} genmove b',f'{2*i+3} clear_board']
    out,err=p.communicate('\n'.join(commands)+'\n',timeout=45)
    assert p.returncode==0,err
    responses=[part.strip() for part in out.split('\n\n') if part.strip()]
    assert len(responses)==len(commands),out
    for command,response in zip(commands,responses):assert response.startswith('='+command.split()[0]),response


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_gtp_rectangular_sgf_rules_parameters_and_search(tmp_path,algorithm):
    p=gtp_process(tmp_path,algorithm)
    def ok(command):
        reply=gtp_command(p,command)
        assert reply.startswith('='),(command,reply)
        return reply[1:].strip()
    try:
        ok('rectangular_boardsize 3 5')
        ok('set_position B A1 W C5')
        ok('play w B3');ok('play w B4') # tolerant player change, with full undo record
        ok('undo')
        board=ok('showboard');assert 'X' in board and 'O' in board
        assert ok('get_komi')=='7.5'
        ok('komi 0.5');ok('kata-set-rule ko SIMPLE')
        stones=ok('showboard')
        ok('kata-set-params {"alwaysComputePassAliveUnderSuicideRules":false,"excludeTerritoryAdjacentToAtari":false}')
        assert ok('kata-get-param excludeTerritoryAdjacentToAtari')=='false'
        assert ok('showboard')==stones
        ok('kata-set-param excludeTerritoryAdjacentToAtari auto')
        ok('kata-set-param visitCapContemptPla W')
        assert ok('kata-get-param visitCapContemptPla')=='-1'
        assert gtp_command(p,'kata-set-param playoutDoublingAdvantagePla invalid').startswith('?')
        assert json.loads(ok('kata-get-rules'))['ko']=='SIMPLE'
        ok('kata-set-params {"maxVisits": 18, "numSearchThreads": 1, "cpuctExploration": 1.3}')
        assert int(ok('kata-get-param maxVisits'))==18
        assert float(ok('kata-get-param cpuctExploration'))==1.3
        assert gtp_command(p,'kata-set-param cpuctExploration 20').startswith('?')
        assert float(ok('kata-get-param cpuctExploration'))==1.3
        before=ok('printsgf')
        move=ok('kata-search b');assert move and move!='resign'
        assert ok('printsgf')==before
        ok('debug_moves')
        ok('kata-get-models');ok('kata-debug-print-tc');assert float(ok('cputime'))>0
        path=tmp_path/'position.sgf';ok(f'printsgf {path}')
        ok('clear_board');ok(f'loadsgf {path}')
        assert ok('printsgf')==before
        assert 'SZ[3:5]' in before
        assert gtp_command(p,'final_score').startswith('?')
        ok('clear_cache');ok('clear_board')
        ok('set_free_handicap A1 C5');assert 'AB[' in ok('printsgf')
        ok('clear_board');placed=ok('place_free_handicap 2');assert len(placed.split())==2
        ok('boardsize 7:9');placed=ok('fixed_handicap 4');assert len(placed.split())==4
        ok('clear_board');ok('kata-set-rules tromp-taylor');ok('play b pass');ok('play w pass')
        assert ok('final_score')=='W+0.5'
        assert ok('final_status_list dead')==''
        ok('quit');p.wait(timeout=10);assert p.returncode==0
    finally:
        if p.poll() is None:p.kill();p.wait()


def test_gtp_streaming_controls_and_benchmark(tmp_path):
    import re
    import signal
    def expired(*_):raise TimeoutError("GTP streaming command did not finish")
    previous=signal.signal(signal.SIGALRM,expired);signal.alarm(30)
    p=gtp_process(tmp_path)
    try:
        assert gtp_command(p,'kata-set-params {"analysisWideRootNoise":0.2,"analysisIgnorePreRootHistory":true,"wideRootNoise":0,"useLcbForSelection":false}').startswith('=')
        assert gtp_command(p,'kata-get-param wideRootNoise')=='= 0.0'
        assert gtp_command(p,'kata-get-param analysisWideRootNoise')=='= 0.2'
        result=gtp_command(p,'lz-analyze b 0.5 visits 2 minmoves 5 maxmoves 5 focus A1:2,B2 0.3 avoid B C3 1 allow W D4 1')
        assert result.startswith('=') and all(line.count('info move ')==5 for line in result.splitlines()[1:]),result
        assert all(0<=int(x)<=10000 for x in re.findall(r'winrate (\d+)',result))
        # Blank input must stop continuous analysis without emitting a new response.
        p.stdin.write('kata-analyze b 0.1\n');p.stdin.flush()
        assert p.stdout.readline().startswith('=')
        assert 'info move' in p.stdout.readline()
        p.stdin.write('\n');p.stdin.flush()
        while True:
            line=p.stdout.readline()
            assert line
            if line=='\n':break
        assert gtp_command(p,'kata-set-param maxVisits 1000000').startswith('=')
        before=gtp_command(p,'printsgf')
        p.stdin.write('kata-search_analyze_cancellable b 0.1\n');p.stdin.flush()
        assert p.stdout.readline().startswith('=')
        assert 'info move' in p.stdout.readline()
        # A subsequent command cancels the search and remains queued.
        p.stdin.write('printsgf\n');p.stdin.flush()
        while True:
            line=p.stdout.readline()
            assert line
            if line=='\n':break
        response=[]
        while True:
            line=p.stdout.readline()
            assert line
            if line=='\n':break
            response.append(line.rstrip('\n'))
        assert '\n'.join(response)==before
        assert gtp_command(p,'boardsize 9').startswith('=')
        result=gtp_command(p,'kata-benchmark 2');assert result.startswith('=') and 'visits' in result,result
        assert gtp_command(p,'kata-benchmark 2bad').startswith('?')
        assert gtp_command(p,'kata-set-param maxVisits 4').startswith('=')
        gtp_command(p,'clear_board');gtp_command(p,'kata-set-rules tromp-taylor')
        gtp_command(p,'play b pass');gtp_command(p,'play w pass')
        score=gtp_command(p,'final_score');assert score.startswith('=')
        assert gtp_command(p,'kata-search b').startswith('=')
        assert gtp_command(p,'final_score')==score
        assert gtp_command(p,'genmove b').startswith('=')
        assert gtp_command(p,'quit').startswith('=');p.wait(timeout=10)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if p.poll() is None:p.kill();p.wait()


def test_gtp_reports_current_version(tmp_path):
    p=gtp_process(tmp_path)
    try:
        assert gtp_command(p,'version')=='= '+VERSION
        assert gtp_command(p,'quit')=='='
        assert p.wait(timeout=10)==0
    finally:
        if p.poll() is None:p.kill();p.wait()


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
@pytest.mark.parametrize('prune',[False,True])
def test_human_prior_is_independent_of_include_policy_and_uses_reported_move(tmp_path,algorithm,prune):
    p=source_process(tmp_path,algorithm,human=True,prune=prune)
    request=dict(boardXSize=5,boardYSize=5,rules='chinese',moves=[],maxVisits=24)
    queries=[dict(request,id=str(flag),includePolicy=flag) for flag in (True,False)]
    out,err=p.communicate(''.join(json.dumps(q)+'\n' for q in queries),timeout=45)
    assert p.returncode==0,err
    reports={row['id']:row for row in map(json.loads,out.splitlines())}
    policy=reports['True']['humanPolicy']
    assert 'humanPolicy' not in reports['False']
    for report in reports.values():
        assert report['moveInfos']
        for move in report['moveInfos']:
            vertex=move['move']
            index=25 if vertex=='pass' else (5-int(vertex[1:]))*5+'ABCDEFGHJKLMNOPQRST'.index(vertex[0])
            assert move['humanPrior']==pytest.approx(max(0,policy[index]))
    if prune:assert any('isSymmetryOf' in move for move in reports['True']['moveInfos'])


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_finished_go_root_remains_analyzable(tmp_path,algorithm):
    p=source_process(tmp_path,algorithm)
    query=dict(id='ended',boardXSize=5,boardYSize=5,rules='chinese',komi=.5,
               moves=[['B','pass'],['W','pass']],maxVisits=20,includePolicy=True)
    out,err=p.communicate(json.dumps(query)+'\n',timeout=45)
    assert p.returncode==0,err
    report=json.loads(out)
    assert 'error' not in report and not report.get('noResults'),report
    assert report['turnNumber']==2 and report['rootInfo']['visits']>=20
    assert report['moveInfos'] and all(p>=0 for p in report['policy'])


def test_torch_primary_metadata_profiles_are_request_local(tmp_path):
    import numpy as np
    import torch
    from etazero.models.katago_network import KataGoNet, inference_network
    _,path,config,profile=cpu_model(tmp_path,'go','alphazero',5,6)
    torch.jit.script(inference_network(KataGoNet('b2c16-fson-mish-meta',6,game='go').eval())).save(str(path))
    profile['analysis'].update(num_analysis_threads=2,search_threads=1,human_sl_profile='',human_model='')
    text=config.read_text();native=tmp_path/'engine.cfg';write_native(profile,native)
    config.write_text(text[:text.index('[analysis]')]+native.read_text())
    base=dict(boardXSize=5,boardYSize=5,rules='chinese',moves=[],maxVisits=12,includePolicy=True)
    requests=[dict(base,id=str(i),overrideSettings={'humanSLProfile':name})
              for i,name in enumerate(['rank_1d','rank_20k','rank_1d','rank_20k'])]
    requests.append(dict(base,id='missing'))
    result=subprocess.run([str(BINARY),'analysis','--stream','true','--protocol','katago',
        '--config',str(config),'--model',str(path),'--model-id','metadata','--device','cpu'],
        input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,timeout=45)
    assert result.returncode==0,result.stderr
    reports={r['id']:r for r in map(json.loads,result.stdout.splitlines())}
    assert all('error' not in reports[str(i)] for i in range(4)),reports
    assert 'error' in reports['missing'] # Never invent a profile for an external model.
    np.testing.assert_allclose(reports['0']['policy'],reports['2']['policy'],atol=1e-6)
    np.testing.assert_allclose(reports['1']['policy'],reports['3']['policy'],atol=1e-6)
    assert not np.allclose(reports['0']['policy'],reports['1']['policy'],atol=1e-6)


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_sgf_roundtrip_preserves_history_modes_and_winner(tmp_path,algorithm):
    # Pinned KataGo testpassalivesuicide.cpp: these stones distinguish the two modes.
    p=gtp_process(tmp_path,algorithm)
    def ok(command):
        result=gtp_command(p,command)
        assert result.startswith('='),(command,result)
        return result[1:].strip()
    try:
        ok('rectangular_boardsize 4 5');ok('kata-set-rules chinese');ok('komi 16.5')
        ok('kata-set-params {"alwaysComputePassAliveUnderSuicideRules":false,"excludeTerritoryAdjacentToAtari":false}')
        rows=['...x','xx..','oxxx','oox.','.x.x']
        stones=[f'{"B" if stone=="x" else "W"} {"ABCD"[x]}{5-y}'
                for y,row in enumerate(rows) for x,stone in enumerate(row) if stone!='.']
        ok('set_position '+' '.join(stones))
        path=tmp_path/'position.sgf';ok(f'printsgf {path}')
        raw=ok('kata-raw-nn 0')
        ok('play b pass');ok('play w pass');assert ok('final_score')=='B+3.5'
        ok('clear_board');ok(f'loadsgf {path}')
        assert ok('kata-get-param alwaysComputePassAliveUnderSuicideRules')=='false'
        assert ok('kata-raw-nn 0')==raw
        ok('play b pass');ok('play w pass');assert ok('final_score')=='B+3.5'
        ok('undo');ok('play w pass');assert ok('final_score')=='B+3.5'
        ok('quit');p.wait(timeout=5)
    finally:
        if p.poll() is None:p.kill();p.wait()


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_source_pda_uses_search_player_and_reports_variance(tmp_path,algorithm):
    p=source_process(tmp_path,algorithm,pda_player='unspecified')
    base=dict(boardXSize=5,boardYSize=5,rules='chinese',moves=[],initialPlayer='W',maxVisits=1)
    queries=[dict(base,id=str(i),overrideSettings=settings) for i,settings in enumerate([
        {'playoutDoublingAdvantage':-1},
        {'playoutDoublingAdvantage':-1,'playoutDoublingAdvantagePla':'W'},
        {'playoutDoublingAdvantage':1,'playoutDoublingAdvantagePla':'B'},
        {'playoutDoublingAdvantage':4},
        {'playoutDoublingAdvantage':-4},
    ])]
    out,err=p.communicate(''.join(json.dumps(q)+'\n' for q in queries),timeout=45)
    assert p.returncode==0,err
    reports={x['id']:x for x in map(json.loads,out.splitlines())}
    for i in ('0','1','2'):
        assert 'error' not in reports[i],reports[i]
        assert reports[i]['rootInfo']['rawVarTimeLeft']>0
    for key in ('rawWinrate','rawStWrError','rawVarTimeLeft'):
        assert reports['0']['rootInfo'][key]==reports['1']['rootInfo'][key]==reports['2']['rootInfo'][key]
    assert all('error' in reports[i] for i in ('3','4'))


def test_gtp_parent_overrides_recompute_only_inherited_defaults(tmp_path):
    from etazero.engine_config import load_engine_config
    _,model,config,_=cpu_model(tmp_path,'go','alphazero',5,6)
    source=tmp_path/'source.cfg';source.write_text('numSearchThreads=1\nmaxVisits=8\ndevice=cpu\nboard_size=5\n')
    profile=load_engine_config(source,environ={},gtp=True)
    text=config.read_text();native=tmp_path/'native.cfg';write_native(profile,native)
    config.write_text(text[:text.index('[analysis]')]+native.read_text())
    p=subprocess.Popen([str(BINARY),'gtp','--config',str(config),'--model',str(model),'--model-id','defaults','--device','cpu'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def ok(command):
        value=gtp_command(p,command);assert value.startswith('='),(command,value)
        return value[1:].strip()
    try:
        assert float(ok('kata-get-param rootFpuReductionMax'))==.1
        ok('kata-set-param rootNoiseEnabled true')
        assert float(ok('kata-get-param rootFpuReductionMax'))==0
        ok('kata-set-param fpuLossProp 0.4')
        assert float(ok('kata-get-param rootFpuLossProp'))==.4
        ok('kata-set-param policyOptimism 0.1')
        assert float(ok('kata-get-param rootPolicyOptimism'))==.1
        ok('kata-set-param rootPolicyTemperature 1.3')
        assert float(ok('kata-get-param rootPolicyTemperatureEarly'))==1.3
        ok('kata-set-param rootFpuReductionMax 0.7')
        ok('kata-set-param rootNoiseEnabled false')
        assert float(ok('kata-get-param rootFpuReductionMax'))==.7
        ok('kata-set-param playoutDoublingAdvantage -1')
        assert float(ok('kata-get-param playoutDoublingAdvantage'))==-1
        assert gtp_command(p,'kata-set-param playoutDoublingAdvantage 4').startswith('?')
        assert float(ok('kata-get-param playoutDoublingAdvantage'))==-1
        ok('quit');p.wait(timeout=5)
    finally:
        if p.poll() is None:p.kill();p.wait()
