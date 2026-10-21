"""Go/Xiangqi rule state, action coordinates, replay and CUDA lifecycle acceptance."""
import json
import os
from pathlib import Path
import random
import subprocess
import numpy as np
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import load_config, validate
from etazero.runtime import run_training
from etazero.games import XIANGQI_MOVES, XIANGQI_INDEX, spec_for_config
from etazero.schema import unpack_observations
from etazero.data import read_raw, training_view, metadata
from etazero.symmetry import augment_batch

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'build/etazero'


def small_config(rule,algorithm):
    c=load_config(CONFIGS/'smoke_test')
    size=5 if rule=='go' else 10
    c['environment'].update(sizes=str(size),size_weights='1',rules=rule,rule_weights='1')
    c['network'].update(canvas=size,channels=24,blocks=1)
    c['agent']['algorithm']=algorithm
    c['symmetry'].update(root_num_symmetries_to_sample=1,nn_symmetry=0)
    c['search'].update(full_search_visits=4,cheap_search_visits=2,reuse_tree=False)
    c['graph_search']['use_graph_search']=rule=='go' and algorithm=='alphazero'
    c['parallelism'].update(game_threads=2,search_threads=1)
    c['inference'].update(server_threads=1,max_batch=4,cache_entries=64,inference_precision='float16')
    c['selfplay']['bootstrap_games']=2
    c['training'].update(train_steps=3,batch_size=4,checkpoint_every=1,skip_validation=True,amp='float16',compile=False,replay_ratio=.5)
    c['replay'].update(min_rows=8,keep_target_rows=64)
    c['shuffle'].update(workers=1,group_rows=64,bucket_rows=64,training_shard_rows=32)
    c['opening']['probability']=0
    c['policy_init']['policy_init']=False
    c['side_positions']['side_position_prob']=.02
    if algorithm=='muzero':
        c['inference']['muzero_queue_capacity']=64
        c['muzero']=dict(latent_channels=16,dynamics_channels=24,dynamics_blocks=1,prediction_channels=24,prediction_blocks=0)
        c['unroll']=dict(steps=3,hidden_gradient_scale=.5)
    validate(c)
    return c


def test_action_vocabulary_and_fixed_root_coordinates():
    assert len(XIANGQI_MOVES)==len(XIANGQI_INDEX)==2086
    for rule in ('go','xiangqi'):
        c=small_config(rule,'muzero');spec=spec_for_config(c);n=c['network']['canvas'];a=spec.actions(n)
        obs=torch.zeros(2,spec.planes,n,n);obs[:,0,:,:9 if rule=='xiangqi' else n]=1
        g=torch.zeros(2,spec.globals)
        if rule=='xiangqi':g[0,0]=1
        action=XIANGQI_INDEX[(1,0,2,2)] if rule=='xiangqi' else n*n
        batch=dict(obs=obs,globals=g,actions=torch.tensor([[action,action],[action,action]]),policy=torch.arange(a).expand(2,4,a).clone())
        for symmetry in range(8):
            transformed=augment_batch(batch,symmetry,spec)
            for row in range(2):
                assert transformed['policy'][row,0,transformed['actions'][row,0]]==action
            if rule=='go':assert (transformed['actions']==n*n).all()
        if rule=='xiangqi':
            transformed=augment_batch(batch,0,spec)
            assert transformed['actions'][1,0]==XIANGQI_INDEX[(7,9,6,7)]


def test_xiangqi_svb_available_for_real_board_search():
    from etazero.engine_config import load_engine_config, validate_engine_config
    c=small_config('xiangqi','alphazero')
    c['subtree_value_bias']['subtree_value_bias_factor']=.25
    validate(c)
    assert spec_for_config(c).subtree_value_bias
    latent=small_config('xiangqi','muzero')
    latent['subtree_value_bias']['subtree_value_bias_factor']=.25
    with pytest.raises(ValueError,match='MuZero latent search requires subtree_value_bias_factor=0'):
        validate(latent)
    for match in (False,True):
        profile=load_engine_config(CONFIGS/'minimal_test',match=match,environ={})
        mode=profile['match' if match else 'analysis']
        mode.update(rule='xiangqi',board_size=10,nn_symmetry=0,root_num_symmetries_to_sample=1,
                    subtree_value_bias_factor=0)
        validate_engine_config(profile,match=match)
        mode['subtree_value_bias_factor']=.25
        validate_engine_config(profile,match=match)


def test_game_layout_boundary_and_unknown_rule():
    from etazero.games import spec_for_rule
    mixed={'environment':{'rules':'renju,hex'}}
    assert spec_for_config(mixed).layout==spec_for_rule('hex').layout
    mixed['environment']['rules']='renju,go'
    with pytest.raises(ValueError,match='one game input/action layout'):spec_for_config(mixed)
    with pytest.raises(KeyError):spec_for_rule('unknown')


def test_terminal_outcomes_and_rectangular_board_geometry():
    from etazero.outcomes import Reason, has_result, terminal_wdl
    from etazero.games import spec_for_rule
    reasons=np.array([Reason.SCORE,Reason.NO_RESULT,Reason.MATE,Reason.INSUFFICIENT_MATERIAL])
    np.testing.assert_array_equal(has_result(reasons),[True,False,True,True])
    np.testing.assert_array_equal(terminal_wdl(np.array([0,0,1,0]),reasons,player=-1),
                                  [[0,1,0],[.5,0,.5],[0,0,1],[0,1,0]])
    xq=spec_for_rule('xiangqi')
    assert (xq.width(10),xq.height(10),xq.board_stride(10),xq.area(10),xq.actions(10))==(9,10,10,90,2086)


@pytest.fixture(scope='module')
def records(tmp_path_factory):
    probe=ROOT/'build/multigame_test'
    if not probe.exists():pytest.skip('Build native tests first')
    root=tmp_path_factory.mktemp('multigame_records')
    subprocess.run([str(probe),str(root)],check=True,capture_output=True,text=True,timeout=60)
    return root


@pytest.mark.parametrize('rule',['go','xiangqi'])
def test_side_globals_audit_rejects_pda_and_nonfinite(records,rule):
    from etazero.data import validate_raw
    a=read_raw(next((records/(rule+'-az')).glob('*.npz')))
    assert len(a['side_globals']) > 0
    original=a['side_globals'].copy()
    pda=15 if rule=='go' else 3
    for column,value,message in ((pda,1,'clear PDA'),(pda+1,.25,'clear PDA'),(0,np.nan,'nonfinite side_globals')):
        a['side_globals']=original.copy()
        a['side_globals'][0,column]=value
        with pytest.raises(ValueError,match=message):validate_raw(a,deep=True)


@pytest.mark.parametrize('rule',['go','xiangqi'])
@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_native_rules_records_unroll_and_loss(records,rule,algorithm):
    from etazero.network import make_network
    from etazero.training import training_forward,forward_batch
    from etazero.muzero.data import prepare_batch
    if algorithm=="muzero":
        from etazero.muzero.optimization import parameter_groups
    else:
        from etazero.optimization import parameter_groups
    c=small_config(rule,algorithm);spec=spec_for_config(c);canvas=c['network']['canvas']
    paths=list((records/(rule+('-mz' if algorithm=='muzero' else '-az'))).glob('*.npz'))
    a=read_raw(paths[0]);v=training_view(a)
    assert v['policy'].shape[-1]==spec.actions(canvas)
    v['obs']=unpack_observations(v['obs'],canvas)
    if algorithm=='muzero':prepare_batch(v,1.,random.Random(4),spec)
    batch={k:torch.from_numpy(value[:4].copy()).float() for k,value in v.items() if k!='recurrent_indices'}
    if algorithm=='muzero':batch['recurrent_indices']=torch.arange(4)
    batch=augment_batch(batch,1,spec)
    model=make_network(c);parameter_groups(model)
    output=forward_batch(training_forward(model,c),batch)
    losses=output[0]
    assert torch.isfinite(losses[0]);losses[0].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


@pytest.mark.parametrize('rule',['go','xiangqi'])
@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_evicted_replay_restores_identical_arrays(records,tmp_path,rule,algorithm):
    import shutil
    from etazero.shuffle import build_snapshot,prune_derived,restore_snapshot
    from etazero.storage import sha256
    from etazero.data import game_weight_statistics
    source=records/(rule+('-mz' if algorithm=='muzero' else '-az'))
    shutil.copytree(source,tmp_path/'selfplay')
    entries=[]
    for p in sorted((tmp_path/'selfplay').glob('*.npz')):
        raw=read_raw(p)
        entries.append(dict(path=str(p.relative_to(tmp_path)),metadata=metadata(raw),weight_stats=game_weight_statistics(raw)))
    c=small_config(rule,algorithm);c['replay'].update(min_rows=1,keep_target_rows='all',taper_exponent=1,expand_per_row=1)
    snapshot=tmp_path/'snapshots'/build_snapshot(tmp_path,1,entries,c)
    hashes={str(p.relative_to(snapshot)):sha256(p) for p in (snapshot/'data').rglob('*.npz')}
    prune_derived(tmp_path,[snapshot.name],set());restore_snapshot(snapshot)
    assert hashes=={str(p.relative_to(snapshot)):sha256(p) for p in (snapshot/'data').rglob('*.npz')}


@pytest.mark.parametrize('rule',['go','xiangqi'])
@pytest.mark.parametrize('algorithm,architecture,channels,blocks',[
    ('alphazero','plain',128,10),('alphazero','transformer',192,5),
    ('alphazero','b1c6nbt-ih',6,1),('muzero','resnet',24,1)])
def test_backbone_export_preserves_game_actions(rule,algorithm,architecture,channels,blocks):
    from etazero.network import make_network,inference_network
    torch.set_num_threads(1)
    c=small_config(rule,algorithm);c['network'].update(architecture=architecture,channels=channels,blocks=blocks)
    validate(c);spec=spec_for_config(c);canvas=c['network']['canvas']
    obs=torch.zeros(2,spec.planes,canvas,canvas);obs[:,0,:,:9 if rule=='xiangqi' else canvas]=1
    g=torch.zeros(2,spec.globals)
    model=make_network(c).eval();inference=inference_network(model)
    scripted=torch.jit.script(inference)
    assert scripted.game_name()==spec.name
    with torch.no_grad():
        expected=inference(obs,g);actual=scripted(obs,g)
        torch.testing.assert_close(actual,expected)
        assert actual[1 if algorithm=='muzero' else 0].shape==(2,spec.actions(canvas))
        if algorithm=='muzero':
            actions=torch.tensor([XIANGQI_INDEX[(1,0,2,2)],XIANGQI_INDEX[(7,9,6,7)]]) if rule=='xiangqi' else torch.tensor([0,canvas*canvas])
            torch.testing.assert_close(scripted.recurrent(actual[0],actions),inference.recurrent(expected[0],actions))


@pytest.mark.parametrize('rule',['go','xiangqi'])
@pytest.mark.parametrize('reanalyze',[False,True])
def test_cpu_training_reanalysis_and_history(tmp_path,rule,reanalyze):
    if not BINARY.exists():pytest.skip('Build native engine first')
    from etazero.training import load_checkpoint
    torch.set_num_threads(1)
    c=small_config(rule,'alphazero')
    c['devices'].update(train='cpu',selfplay='cpu')
    c['training'].update(amp='off',train_steps=1)
    c['inference']['inference_precision']='float32'
    c['search'].update(cheap_search_probs=1,cheap_search_target_weight=.5)
    c['subtree_value_bias'].update(subtree_value_bias_factor=.45,
                                  subtree_value_bias_table_num_shards=4)
    c['reanalysis'].update(use_reanalyze=reanalyze,reanalyze_prop=1,
                          reanalyze_policy_surprise_weight=1,reanalyze_value_surprise_weight=1,
                          reanalyze_surprise_exponent=1,reanalyze_use_outcome_targets=True)
    validate(c)
    root=tmp_path/'run';state=run_training(root,c,BINARY,max_iteration=1)
    saved=load_checkpoint(root,state['checkpoint'],c)
    assert saved['optimizer_steps']>0 and saved['game_layout']['game']==rule
    assert run_training(root,c,BINARY,resume=True,max_iteration=1)==state
    rows=0
    for path in (root/'selfplay').rglob('*.npz'):
        raw=read_raw(path,deep=True)  # Replay native rules, historical NN input and legal targets.
        rows+=int(raw['reanalyzed'].sum())
    assert (rows>0)==reanalyze


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA acceptance')
@pytest.mark.parametrize('rule',['go','xiangqi'])
@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
@pytest.mark.parametrize('compiled',[False,True])
def test_cuda_training_resume_evaluation_and_web(tmp_path,rule,algorithm,compiled):
    from etazero.runtime import run_training
    from etazero.training import load_checkpoint
    from etazero.storage import load_json
    from etazero.engine_config import load_engine_config
    from etazero.analysis import analyze
    from web.engine import Engine
    from test_gpu import assert_compiled_partial_resume
    torch.set_num_threads(1)
    c=small_config(rule,algorithm);c['training'].update(compile=compiled,train_steps=8)
    c['replay'].update(min_rows=32,keep_target_rows=128)
    root=tmp_path/'run'
    state=run_training(root,c,BINARY,max_iteration=2)
    saved=load_checkpoint(root,state['checkpoint'],c)
    assert saved['optimizer_steps']>0 and saved['game_layout']['game']==rule
    assert run_training(root,c,BINARY,resume=True,max_iteration=2)==state
    assert_compiled_partial_resume(tmp_path,root,c)
    for path in (root/'selfplay').rglob('*.npz'):read_raw(path)
    profile=load_engine_config(CONFIGS/'minimal_test',environ={})
    profile['analysis'].update(rule=rule,board_size=c['network']['canvas'],root_num_symmetries_to_sample=1,
                               nn_symmetry=0,reuse_tree=False,use_graph_search=rule=='go' and algorithm=='alphazero',visits=4,device='cuda:0',inference_precision='float16')
    _,result=analyze(profile,BINARY,root)
    assert result['result']['action']>=0
    engine=Engine(BINARY,root/state['model']['path'],profile)
    try:
        initial=engine.command(dict(command='new',size=c['network']['canvas'],rule=rule))['state']
        action=initial['legal_actions'][0]
        played=engine.command(dict(command='play',action=action))['state']
        assert played['turn']==1
        undo=engine.command(dict(command='undo',count=1))['state']
        assert undo['board']==initial['board'] and undo['legal_actions']==initial['legal_actions']
        report=engine.command(dict(command='analyze',visits=4))['analysis']
        assert report['action'] in initial['legal_actions']
        raw=engine.command(dict(command='raw_nn'))
        assert raw
    finally:engine.close()
    if not compiled and os.environ.get('ETAZERO_BROWSER_TESTS')=='1':
        browser_moves(BINARY,root/state['model']['path'],profile,rule,c['network']['canvas'])


def browser_moves(binary, model, profile, rule, size, artifact_dir=None):
    from threading import Thread
    from playwright.sync_api import sync_playwright
    from web.app import App
    from web.server import make_server
    app=App(binary,{'test':model},profile,size)
    server=make_server(app,'127.0.0.1',0);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch();page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function("document.querySelector('#new-game') && !document.querySelector('#new-game').disabled")
            page.select_option('#mode','manual');page.click('#new-game')
            page.wait_for_function("document.querySelector('#move-count').textContent==='第 0 手' && !document.querySelector('#new-game').disabled")
            if rule=='go':
                page.click('#pass');page.wait_for_function("document.querySelector('#move-count').textContent==='第 1 手'")
            else:
                page.locator('[data-square="1"]').click();page.locator('[data-square="22"]').click()
                page.wait_for_function("document.querySelector('#move-count').textContent==='第 1 手'")
                assert page.locator('[data-square="22"] .xiangqi-piece').inner_text()=='馬'
            page.click('#undo');page.wait_for_function("document.querySelector('#move-count').textContent==='第 0 手'")
            page.click('#analyze')
            page.wait_for_function("document.querySelectorAll('#network-maps section').length > 0 && document.querySelectorAll('#candidates tr').length > 0")
            page.click('#analyze');page.wait_for_function("document.querySelector('#analyze').getAttribute('aria-pressed')==='false'")
            page.click('#tab-network');page.select_option('#network-kind','logits')
            if rule=='xiangqi':
                assert '%' not in page.locator('.xiangqi-policy-list button').first.inner_text()
                assert '红方' in page.locator('#value-label').inner_text()
            else:
                assert '停一手' in page.locator('#network-maps').inner_text()
            if artifact_dir:
                page.screenshot(path=str(Path(artifact_dir)/(rule+'-web.png')),full_page=True)
                page.set_viewport_size(dict(width=390,height=844))
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            assert not errors
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join();app.close()


@pytest.mark.parametrize('rule',['go','xiangqi'])
@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_native_match_and_completed_resume(tmp_path,rule,algorithm):
    if not BINARY.exists():pytest.skip('Build native engine first')
    from etazero.training import initialize
    from etazero.export import export_model
    from etazero.engine_config import load_engine_config
    from etazero.arena import single_match
    torch.set_num_threads(1)
    c=small_config(rule,algorithm);c['devices'].update(train='cpu',selfplay='cpu')
    c['training']['amp']='off';c['inference']['inference_precision']='float32'
    root=tmp_path/'model';root.mkdir();base=initialize(root,c);info=export_model(root,c,base)
    profile=load_engine_config(CONFIGS/'minimal_test',match=True,environ={})
    profile['match'].update(rule=rule,board_size=c['network']['canvas'],device='cpu',inference_precision='float32',
                            visits=2,game_threads=2,search_threads=1,root_num_symmetries_to_sample=1,
                            nn_randomize=False,nn_symmetry=0,reuse_tree=False,use_graph_search=False)
    profile['opening']['policy_init']=False
    path=root/info['path'];output=tmp_path/'match'
    _,games=single_match(profile,BINARY,path,path,output,games=4)
    assert len(games)==4 and all(g['reason'] in (4,5,6,7,8,9) for g in games)
    assert single_match(profile,BINARY,path,path,output,games=4)[1]==games
    from web.engine import Engine
    analysis={'analysis':{k:v for k,v in profile['match'].items() if k not in ('games','game_threads')}}
    engine=Engine(BINARY,path,analysis)
    try:
        before=engine.command(dict(command='new',rule=rule,size=c['network']['canvas']))['state']
        for opening in ('empty','balanced'):
            with pytest.raises(RuntimeError,match='layout does not match'):
                engine.command(dict(command='new',rule='freestyle',size=5,opening=opening))
            assert engine.command('state')['state']==before
        assert engine.command(dict(command='analyze',visits=2))['analysis']['action']>=0
        if rule=='go':
            engine.command(dict(command='new',rule=rule,size=5,komi=0))
            engine.command(dict(command='play',action=25))
            end=engine.command(dict(command='play',action=25))['state']
            assert end['finished'] and end['winner']==0 and end['reason']==4
            state=engine.command(dict(command='undo',count=1))['state']
            assert not state['finished'] and state['komi']==0 and len(state['boards'])==2
        else:
            fen='rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1'
            initial=engine.command(dict(command='new',rule=rule,size=10,fen=fen))['state']
            assert initial['player']==-1 and len(initial['legal_actions'])==44
            assert (initial['board_width'],initial['board_height'],initial['board_stride'],len(initial['board']))==(9,10,10,100)
            state=engine.command(dict(command='play',action=XIANGQI_INDEX[(1,9,2,7)]))['state']
            assert state['players']==[-1,1] and state['board'][72]==-5
            assert engine.command(dict(command='undo',count=1))['state']['fen']==initial['fen']
        if rule=='go' and algorithm=='alphazero':
            result=subprocess.run([str(BINARY),'analysis','--stream','true','--protocol','gtp',
                '--config',str(Path(engine.directory.name)/'analysis.cfg'),'--model',str(path),
                '--model-id',info['id'],'--device','cpu','--seed','1'],
                input='1 komi 0\n2 play b pass\n3 play w pass\n4 showboard\n5 clear_board\n6 showboard\n7 quit\n',
                check=True,capture_output=True,text=True,timeout=30)
            rows=[json.loads(line.split(' ',1)[1]) for line in result.stdout.splitlines() if line.startswith(('=4 {','=6 {'))]
            assert len(rows)==2 and rows[0]['finished'] and not rows[1]['finished']
            assert all(row['komi']==0 for row in rows) and '?' not in result.stdout
    finally:engine.close()
