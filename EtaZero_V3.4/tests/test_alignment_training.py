"""KataGo generic training targets, clocks, and optional optimizer controls."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import numpy as np
import pytest
import torch

from etazero.run.runtime import run_training
from etazero.train.brenorm import BoundsSchedule
from etazero.data.targets import full_trajectory_auxiliary, training_view
from etazero.games import SPECS
from etazero.train.lr_schedule import effective_scale
from etazero.models.network import continuation_losses, make_network
from etazero.schema import unpack_observations
from etazero.symmetry import mask_history
from etazero.train.learner import training_forward, forward_batch
from test_python import winning_record
from etazero.config import load_config
from config_samples import CONFIGS


def test_history_prefix_and_ladder_fallback():
    obs = torch.arange(22.).reshape(1,22,1,1).expand(6,-1,2,2).clone()
    globals = torch.ones(6,19)
    include = torch.arange(5)[None] < torch.arange(6)[:,None]
    result = mask_history(dict(obs=obs,globals=globals), SPECS['go'], include.float())
    for depth in range(6):
        assert result['obs'][depth,9:14,0,0].tolist() == [float(9+i) if i < depth else 0 for i in range(5)]
        assert result['obs'][depth,15,0,0] == (15 if depth else 14)
        assert result['obs'][depth,16,0,0] == (16 if depth >= 2 else 15 if depth else 14)
    torch.testing.assert_close(result['globals'][:,:5], include.float())
    assert obs[0,15,0,0] == 15  # Never mutate raw batch or current board.


def test_history_matches_pinned_source():
    source = Path('/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/data_processing_pytorch.py')
    if not source.exists(): pytest.skip('Pinned source checkout unavailable')
    import sys
    sys.path.insert(0, str(source.parents[2]))
    try:
        from katago.train.data_processing_pytorch import build_history_matrices, apply_history_matrices
        model_config = {'version': 15}
        obs, globals = torch.randn(10000,22,2,2), torch.randn(10000,19)
        base, builder = build_history_matrices(model_config, 'cpu')
        torch.manual_seed(853)
        expected = apply_history_matrices(model_config, obs, globals, torch.zeros(10000,80), base, builder)
        torch.manual_seed(853)
        actual = mask_history(dict(obs=obs,globals=globals), SPECS['go'])
        torch.testing.assert_close(actual['obs'], expected[0])
        torch.testing.assert_close(actual['globals'], expected[1])
    finally:
        sys.path.pop(0)


def test_brenorm_updates_are_piecewise_and_resume_exactly():
    class Model:
        def set_brenorm_bounds(self, *args): self.bounds = args
    options = dict(avg_momentum=.003,target_rmax=3.,target_dmax=5.,adjustment_scale=1e6)
    model = Model(); schedule = BoundsSchedule(model, options)
    schedule.update(0)
    assert model.bounds == (.003,1.,0.)
    schedule.update(500*128)
    expected = -math.expm1(-64000/1e6)
    assert model.bounds == pytest.approx((.003,1+2*expected,5*expected))
    resumed = BoundsSchedule(Model(), options, 64000, schedule.state_dict())
    for count in (70000,134000,198000):
        schedule.update(count); resumed.update(count)
        assert schedule.state_dict() == resumed.state_dict()


def test_future_boards_and_variance_use_full_real_trajectory():
    a, m = winning_record(2)
    future, variance, signed = full_trajectory_auxiliary(a)
    obs = unpack_observations(a['observations'], m['canvas']).astype(np.float32)
    for g in range(len(a['sizes'])):
        lo,hi = a['game_offsets'][g:g+2]; ol = a['observation_offsets'][g]
        values = (a['search_wdl'][lo:hi,0]-a['search_wdl'][lo:hi,2])*a['players'][ol:ol+hi-lo]
        values = np.r_[values, a['winners'][g]]
        for t in range(hi-lo):
            for head, offset in enumerate((8,32)):
                ix = ol+min(t+offset,hi-lo)
                expected = (obs[ix,1]-obs[ix,2])*a['players'][ix]*a['players'][ol+t]
                np.testing.assert_array_equal(future[lo+t,head], expected)
            expected = sum((i-t)*(values[i]-values[i-1])**2 for i in range(t+1,len(values)))
            assert variance[lo+t] == pytest.approx(expected,abs=1e-6)
    assert signed.min() >= -1 and signed.max() <= 1


def test_continuation_loss_coefficients_gates_and_gradient_floor():
    obs=torch.zeros(2,5,2,2);obs[:,0]=1
    future=torch.zeros(2,2,2,2,requires_grad=True)
    raw=torch.tensor([-100.,0.],requires_grad=True)
    targets=torch.ones_like(future);gate=torch.tensor([1.,0.])
    fl,vl=continuation_losses(future,raw,obs,targets,torch.tensor([100.,0.]),gate,gate)
    assert fl.item() == pytest.approx(.25*(4+1)/2/2)
    difference=100+1e-5
    assert vl.item() == pytest.approx(.0003*50*(difference-25)/2,rel=1e-6)
    (fl+vl).backward()
    assert raw.grad[0].item() == pytest.approx(-.0003*50*40*.05/2)
    assert raw.grad[1] == 0 and not future.grad[1].any()


def test_continuation_losses_and_gradients_match_source_functions():
    import ast
    from types import SimpleNamespace
    source = Path('/home/sky/RL/SkyZero-lab/KataGo/python/katago/train')
    if not source.exists(): pytest.skip('Pinned source checkout unavailable')
    env = {'torch': torch}
    for filename, names in [('metrics_pytorch.py', {'huber_loss','constant_like','loss_futurepos_samplewise','loss_variance_time_samplewise'}),
                            ('model_pytorch.py', {'SoftPlusWithGradientFloorFunction'})]:
        nodes = [n for n in ast.walk(ast.parse((source/filename).read_text()))
                 if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source/filename),'exec'),env)
    torch.manual_seed(573)
    obs = torch.zeros(6,5,5,5);obs[:,0,:3,:3] = 1;obs[3:,0] = 1
    future = torch.randn(6,2,5,5,requires_grad=True)
    raw = torch.tensor([-100.,-3.,0.,3.,10.,100.],requires_grad=True)
    ref_future, ref_raw = future.detach().clone().requires_grad_(), raw.detach().clone().requires_grad_()
    target = torch.randint(-1,2,future.shape).float();time_target = torch.rand(6)*500
    gate = torch.tensor([1.,0.,1.,.5,1.,1.]);row_weight = torch.tensor([1.,2.,0.,.25,1.,3.])
    # NR retains future boards but has no variance-time target weight.
    value_gate = gate.clone();value_gate[2]=0
    local = continuation_losses(future,raw,obs,target,time_target,gate,value_gate,row_weight)
    metrics = SimpleNamespace(num_futurepos_values=2,pos_len=5)
    source_future = env['loss_futurepos_samplewise'](metrics,ref_future,target,gate,obs[:,0],obs[:,0].sum((1,2)),row_weight).mean()
    prediction = env['SoftPlusWithGradientFloorFunction'].apply(ref_raw,.05,False)*40
    source_time = env['loss_variance_time_samplewise'](metrics,prediction,time_target,value_gate,row_weight).mean()
    torch.testing.assert_close(torch.stack(local),torch.stack((source_future,source_time)))
    sum(local).backward();(source_future+source_time).backward()
    torch.testing.assert_close(future.grad,ref_future.grad)
    torch.testing.assert_close(raw.grad,ref_raw.grad)


@pytest.mark.parametrize('algorithm', ['alphazero','muzero'])
@pytest.mark.parametrize('no_result', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_targets_reach_heads_and_backpropagate(algorithm,no_result,device):
    if device=='cuda':
        if os.environ.get('ETAZERO_GPU_TESTS')!='1':pytest.skip('Enable ETAZERO_GPU_TESTS=1 with host CUDA access')
        assert torch.cuda.is_available()
    torch.set_num_threads(1)
    c=load_config(CONFIGS/'smoke_test')
    a,m=winning_record(2)
    if no_result:
        from etazero.outcomes import Reason
        a['winners'][:]=0;a['reasons'][:]=Reason.NO_RESULT;a['search_wdl'][:]=[0,0,0,1]
    if algorithm == 'muzero':
        from test_muzero_pipeline import small_config
        from etazero.data.muzero import prepare_batch
        import random
        c=small_config()
        m.update(algorithm='muzero',unroll_steps=3)
        a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
        for target,source in [('trajectory_policy','policies'),('trajectory_q_values','q_values'),('trajectory_q_visits','q_visits')]:
            a[target]=a[source].copy()
    batch=training_view(a)
    batch['obs']=unpack_observations(batch['obs'],m['canvas']).astype(np.float32)
    if no_result:
        # Synthetic complete NR targets need a Go head with an active NR logit.
        # Preserve the board-mask/future tensors while adapting the input layout.
        c['environment'].update(rules='go',rule_weights='1')
        obs=np.zeros((len(batch['obs']),SPECS['go'].planes,m['canvas'],m['canvas']),np.float32)
        obs[:,:3]=batch['obs'][:,:3];batch['obs']=obs
        batch['globals']=np.zeros((len(obs),SPECS['go'].globals),np.float32)
        for key in ('policy','opponent_policy','q_values','q_visits'):
            batch[key]=np.pad(batch[key],[(0,0)]*(batch[key].ndim-1)+[(0,1)])
    if algorithm == 'muzero': batch=prepare_batch(batch,1.,random.Random(13),SPECS['go' if no_result else 'gomoku'])
    tensors={k:torch.from_numpy(v).to(device=device,dtype=torch.int64 if k in ('actions','recurrent_indices') else torch.float32) for k,v in batch.items()}
    model=make_network(c).to(device)
    forward=training_forward(model,c)
    values,_=forward_batch(forward,tensors)
    assert len(values) == 14
    torch.testing.assert_close(values[0],sum(values[1:]))
    values[0].backward()
    head=model.network.value_head if algorithm=='alphazero' else model.prediction.value_head
    assert head.conv_futurepos.weight.grad.abs().sum() > 0
    if no_result:
        assert values[11].item()==0 and values[13].item()==0
        assert not head.linear_miscvaluehead.weight.grad[3].any()
    else:
        assert head.linear_miscvaluehead.weight.grad[3].abs().sum() > 0


def test_learning_rate_source_boundaries_and_sqrt_cycle():
    assert effective_scale(1,dict(mode='auto'),550000000)==6
    assert effective_scale(1,dict(mode='auto2',auto2_factor=.5),40000000)==3
    schedule=dict(mode='piecewise',points='(0,12),(20M,9),(40M,6)')
    assert [effective_scale(1,schedule,n) for n in (0,19999999,20000000,40000000)] == [12,12,9,6]
    cycle=dict(mode='cycle',cycle_rounds=6,cycle_up_rounds=2,cycle_down_rounds=2,cycle_low=1,cycle_high=9)
    assert [effective_scale(1,cycle,0,i) for i in range(7)] == [4,9,9,9,4,1,4]

@pytest.mark.parametrize('kind', ['normuon','aurora'])
@pytest.mark.parametrize('polar', [False,True])
def test_optional_muon_kernels_restore_next_update(kind,polar,monkeypatch):
    from etazero.train.muon import Muon
    monkeypatch.setenv('KATAGO_MUON_BATCHED_NS','0')
    torch.set_num_threads(1)
    p=torch.nn.Parameter(torch.randn(4,3,2,2))
    options=dict(kind=kind,ns_steps=3,use_polar_express=polar,aurora_pp_iterations=2,aurora_pp_beta=.5)
    def build(x): return Muon([dict(group_name='normal',params=[x],lr=.001,weight_decay=.02)],options)
    optimizer=build(p)
    for _ in range(2):p.grad=torch.randn_like(p);optimizer.step()
    saved=copy.deepcopy(optimizer.state_dict());replica=torch.nn.Parameter(p.detach().clone());restored=build(replica);restored.load_state_dict(saved)
    p.grad=torch.randn_like(p);replica.grad=p.grad.clone();optimizer.step();restored.step()
    torch.testing.assert_close(p,replica,rtol=0,atol=0)
    assert torch.isfinite(p).all()
    if kind=='normuon': assert optimizer.state[p]['normuon_v'].shape==(4,)


def test_attention_bound_hinge_and_gradient_matches_norm_bound():
    from etazero.train.attention_penalty import configure, metrics
    from etazero.models.transformer import Attention
    torch.set_num_threads(1)
    model=Attention(2)
    x=torch.randn(4,96,2,2);mask=torch.ones(4,1,2,2)
    configure(model,dict(cap=0,batch_fraction=.5))
    model(x,mask)
    # RoPE is orthogonal; its FP32 norm agrees with the pre-rotation reference.
    tokens=model.norm(x.flatten(2).transpose(1,2))
    q=model.q_proj(tokens).reshape(4,4,3,32).transpose(1,2)[:2]
    k=model.k_proj(tokens).reshape(4,4,3,32).transpose(1,2)[:2]
    expected=(q.square().sum(-1).amax(-1)*k.square().sum(-1).amax(-1)).clamp(min=1).sqrt()/math.sqrt(32)
    penalty,bound=metrics(model,0)
    torch.testing.assert_close(penalty,expected.mean())
    torch.testing.assert_close(bound,expected.max())
    penalty.backward()
    assert model.q_proj.weight.grad.norm()>0 and model.k_proj.weight.grad.norm()>0
    assert model.v_proj.weight.grad is None or not model.v_proj.weight.grad.any()
    copy.deepcopy(model) # No retained autograd graph after collecting the penalty.


def test_attention_penalty_requires_an_attention_trunk():
    from etazero.train.attention_penalty import configure
    with pytest.raises(ValueError,match='attention layers'):
        configure(torch.nn.Linear(3,3),dict(cap=3,batch_fraction=1))


def test_muon_auxiliary_lr_and_decay_floor_resume():
    from etazero.train.optimization import optimizer_for, optimization_for, group_settings
    from test_muon import muon_config
    c=muon_config();c['optimizer'].update(kind='normuon',wd_floor_frac=.5,muon_adam_lr_factor=.3)
    model=make_network(c);optimizer=optimizer_for(model,c);controller=optimization_for(model,c,optimizer)
    assert controller.initial_rms
    state=controller.state_dict();other=make_network(c);restored=optimization_for(other,c,optimizer_for(other,c),state)
    assert restored.initial_rms==controller.initial_rms
    options=c['optimizer'];base=copy.deepcopy(options);base['muon_adam_lr_factor']=1
    a=group_settings('input',options,64,0,{},{});b=group_settings('input',base,64,0,{},{})
    assert a[0]==pytest.approx(b[0]*.3) and a[1]==b[1]
    assert group_settings('normal',options,64,0,{}, {})==group_settings('normal',base,64,0,{}, {})

@pytest.mark.parametrize('architecture', ['transformer','b7c96h6kv3qk32v16tflrs-fson-silu'])
def test_attention_training_metrics_and_export(architecture):
    from etazero.models.export import export_model, example_inputs
    from etazero.models.network import inference_network
    c=load_config(CONFIGS/'smoke_test');c['network'].update(architecture=architecture,channels=192 if architecture=='transformer' else 96,blocks=5 if architecture=='transformer' else 14)
    c['attention_penalty'].update(cap=0,batch_fraction=.5)
    a,m=winning_record(1);batch=training_view(a);batch['obs']=unpack_observations(batch['obs'],m['canvas']).astype(np.float32)
    tensors={k:torch.from_numpy(v[:2]).float() for k,v in batch.items()}
    model=make_network(c);model.network.use_flex_attention=False;forward=training_forward(model,c);values,_=forward_batch(forward,tensors)
    assert len(values)==16 and values[-2]>0 and values[-1]>0
    torch.testing.assert_close(values[0],sum(values[1:-2]))
    (values[0]+.001*values[-2]).backward()
    inference=inference_network(model.eval()).eval()
    with torch.no_grad():
        assert all(torch.isfinite(output).all() for output in inference(tensors['obs'],tensors['globals']))


def test_external_start_and_per_game_loss_weight_reach_reader(tmp_path,monkeypatch):
    import test_resume_sampling_config as sampling_tests
    monkeypatch.setattr(sampling_tests,"ROOT",Path(__file__).resolve().parents[1])
    from test_resume_sampling_config import sampling_config,native_selfplay
    from etazero.data.targets import read_raw
    c=sampling_config(False);c['environment'].update(sizes='5',size_weights='1',rules='freestyle',rule_weights='1')
    c['parallelism'].update(game_threads=1,search_threads=1)
    path=tmp_path/'starts.startpos';path.write_text('opening 5 freestyle 1 0.3 2 0 1\n')
    c['start_positions'].update(probability=1,files=str(path),policy_init=False)
    c['data_weight']['scale_data_weight']=.5
    result=native_selfplay(tmp_path,c);assert result.returncode==0,result.stderr
    files=list((tmp_path/'raw').glob('*.npz'));assert files
    raw=read_raw(files[0]);assert (raw['initial_position_kind']==5).all() and (raw['initial_position_moves']==2).all()
    np.testing.assert_allclose(raw['training_weights'],.3)
    view=training_view(raw);np.testing.assert_allclose(view['training_weight'],.3)
    path.write_text('zero 5 freestyle 0 1 2 0 1\n')
    other=tmp_path/'invalid';other.mkdir();result=native_selfplay(other,c)
    assert result.returncode!=0 and 'zero sampling weight' in result.stderr

@pytest.mark.skipif(__import__('os').environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA required')
@pytest.mark.parametrize('kind,architecture', [('normuon','b2c16-brn'),('aurora','transformer')])
def test_cuda_optional_training_controls_and_resume(tmp_path,kind,architecture):
    from etazero.run.runtime import run_training
    from etazero.train.learner import load_checkpoint
    from etazero.storage import load_json
    from etazero.config import validate
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    c=load_config(CONFIGS/'smoke_test');c['training'].update(compile=False,amp='bfloat16',train_steps=2,skip_validation=True)
    c['network'].update(architecture=architecture,channels=16 if kind=='normuon' else 192,blocks=2 if kind=='normuon' else 5)
    c['optimizer'].update(kind=kind,wd_floor_frac=.25,ns_steps=3,use_polar_express=True,lr_scale=1)
    c['lr_schedule'].update(mode='piecewise',points='(0,1),(16,0.5)')
    if kind=='aurora':c['attention_penalty'].update(cap=1,batch_fraction=.5)
    validate(c)
    root=tmp_path/'run';binary=Path(__file__).resolve().parents[1]/'build/etazero'
    state=run_training(root,c,binary,max_iteration=2)
    checkpoint=load_checkpoint(root,state['checkpoint'],c)
    events=[json.loads(line) for line in (root/'logs/events.jsonl').read_text().splitlines()]
    updates=[e for e in events if e['event']=='update'];assert updates and all(math.isfinite(e['loss']) for e in updates)
    assert any(e['future_position_loss']>0 and e['variance_time_loss']>0 for e in updates)
    if kind=='aurora':assert any(e['attention_logit_penalty']>0 for e in updates)
    assert run_training(root,c,binary,resume=True,max_iteration=2)==state
    assert checkpoint['optimization']['initial_rms']

@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda:0',marks=pytest.mark.skipif(__import__('os').environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA required'))])
def test_native_secondary_model_clock_and_constraints(tmp_path,algorithm,device):
    import subprocess
    from etazero.models.network import inference_network
    from etazero.engine_config import load_engine_config
    from etazero.config import write_native
    from test_muzero_pipeline import small_config
    torch.set_num_threads(1)
    c=small_config() if algorithm=='muzero' else load_config(CONFIGS/'smoke_test')
    model=make_network(c).to(device).eval();path=tmp_path/'policy.pt';torch.jit.script(inference_network(model)).save(str(path))
    profile=load_engine_config(CONFIGS/('muzero' if algorithm=='muzero' else 'baseline'),environ={},overrides={'analysis':dict(
        device=device,board_size=5,rule='freestyle',visits=16,search_threads=2,min_playouts_per_thread=0,
        nn_randomize=False,root_num_symmetries_to_sample=1,cache_entries=8,nn_mutex_pool_power_of_two=2,
        human_model=str(path),human_sl_root_weightless=1.,human_sl_opponent_weightful=.5,human_sl_chosen_move_proportion=1.,
        inference_precision='float16' if device.startswith('cuda') else 'float32',max_visits_pondering=8,max_playouts_pondering=8)})
    cfg=tmp_path/'engine.cfg';write_native({**profile,'network':{'canvas':6}},cfg)
    jobs=[dict(id='new',command='new',size=5,rule='freestyle'),dict(id='clock',command='time_settings',clock_type='fischer',clock_arguments=[30,1]),
          dict(id='left',command='time_left',player='b',time=20,stones=0),dict(id='invalid',command='time_left',player=0,time=20,stones=0),
          dict(id='human',command='analyze',visits=16,report_interval_ms=0,allow_moves=[dict(player='b',moves=[2],until_depth=1)]),
          dict(id='ponder',command='ponder'),dict(id='move',command='genmove',visits=16,report_interval_ms=0)]
    binary=Path(__file__).resolve().parents[1]/'build/etazero'
    args=['--config',str(cfg),'--model',str(path),'--model-id','secondary-test','--device',device]
    result=subprocess.run([str(binary),'analysis','--stream','true','--protocol','json',*args],input=''.join(json.dumps(j)+'\n' for j in jobs),text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stderr
    replies={r['id']:r for r in map(json.loads,result.stdout.splitlines()) if r.get('final')}
    assert not replies['invalid']['ok']
    assert all(replies[j['id']]['ok'] for j in jobs if j['id']!='invalid'),replies
    assert replies['human']['analysis']['action']==2 and replies['human']['analysis']['completed_visits']==1
    assert replies['move']['played']
    gtp=subprocess.run([str(binary),'gtp',*args],input='1 time_settings 30 10 5\n2 time_left b 8 3\n3 kata-time_settings fischer-capped 30 1 60 10\n4 kgs-time_settings byoyomi 0 10 3\n5 kata-list_time_settings\n6 quit\n',text=True,capture_output=True,timeout=90)
    assert gtp.returncode==0 and '?' not in gtp.stdout,gtp.stdout+gtp.stderr
