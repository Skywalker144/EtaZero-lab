from etazero.schema import GLOBALS
from config_samples import CONFIGS
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
import torch
from etazero.config import ROOT, load_config, validate
from etazero.data import read_raw, training_view, validate_raw, Catalog
from etazero.export import example_inputs
from etazero.network import MaskedBatchNorm, make_network, base_losses as losses, inference_network
from etazero.reader import BatchReader
from etazero.schema import CONTRACT_ID, RAW_DTYPES, pack_observations, unpack_observations
from etazero.shuffle import view_key, build_snapshot, desired_window, resource_plan, partition_rows, prune_derived, restore_snapshot
from etazero.storage import save_npz, save_json, sha256


@pytest.fixture
def config():
    return load_config(CONFIGS/"smoke_test")


def winning_record(rule=0):
    canvas,size=6,5
    actions=[0,6,1,7,2,8,3,9,4]
    t=len(actions)
    a={k:np.zeros(0,dtype=dtype) for k,dtype in RAW_DTYPES.items()}
    observations=[]; globals=[]
    board=np.zeros((canvas,canvas),np.int8)
    for i in range(t+1):
        player=1 if i%2==0 else -1
        obs=np.zeros((5,canvas,canvas),np.uint8);obs[0,:size,:size]=1
        obs[1]=board==player;obs[2]=board==-player
        # This two-row five-in-a-row fixture has no forbidden points.
        observations.append(obs);globals.append([rule==1,rule==2,-player if rule==2 else 0,rule==2,0,0,0,0])
        if i<t: board.flat[actions[i]]=player
    a["observations"]=pack_observations(np.stack(observations))
    a["globals"]=np.array(globals,np.float32)
    a["players"]=np.array([1,-1]*5,np.int8)
    a["actions"]=np.array(actions,np.int32)
    a["visits"]=np.zeros((t,canvas*canvas),np.int64);a["visits"][np.arange(t),actions]=1
    a["policies"]=a["visits"].astype(np.int16)*10
    # Hand-built child estimates: node count includes its own NN visit.
    a['q_visits']=a['visits'].astype(np.int16)*2
    a['q_values']=(a['visits']>0).astype(np.int16)*8000
    a['opponent_policies']=np.concatenate((a['policies'][1:],np.ones((1,canvas*canvas),np.int16)))
    a['opponent_policy_weights']=np.r_[np.ones(t-1,np.float32),np.zeros(1,np.float32)]
    a["simulations"]=np.ones(t,np.int32);a["temperatures"]=np.ones(t,np.float32)
    a["rewards"]=np.zeros(t,np.float32);a["rewards"][-1]=1
    a["game_offsets"]=np.array([0,t],np.int64);a["observation_offsets"]=np.array([0,t+1],np.int64)
    for key,value in (("game_ids",0),("seeds",7),("sizes",size),("rules",rule),("winners",1),("reasons",1)):
        a[key]=np.array([value],dtype=RAW_DTYPES[key])
    a["train_mask"]=np.ones(t,np.uint8)
    a['row_repeats']=np.ones(t,np.int32);a['target_weights']=np.ones(t,np.float32)
    a['sample_indices']=np.arange(t,dtype=np.int64)
    for key in ('policy_surprises','value_surprises'):
        a[key]=np.zeros(t,np.float32)
    a['cheap_search']=np.zeros(t,np.uint8)
    for key in ('network_wdl','search_wdl'):
        a[key]=np.tile(np.array([0.5,0,0.5],np.float32),(t,1))
    for key in ("opening_moves","balanced_moves","policy_moves","opening_attempts","opening_status","start_values","initial_position_moves","initial_position_kind"):
        a[key]=np.zeros(1,dtype=RAW_DTYPES[key])
    a["hint_actions"]=np.array([-1],dtype=RAW_DTYPES["hint_actions"])
    m={"contract":CONTRACT_ID,"canvas":canvas,"rows":t,"row_begin":0,"plies":t,"games":1,"opening_failures":[""],"run_id":"test","attempt_id":str(rule),
       "worker_id":0,"iteration_id":1,"model_id":"model","config_id":"config","shard_id":str(rule),"created_ns":rule}
    a["metadata"]=np.frombuffer(json.dumps(m).encode(),np.uint8)
    for key, shape in {'side_observations': (0,5,5), 'side_globals': (0,len(GLOBALS)),
                       'side_policies': (0,36), 'side_visits': (0,36), 'side_wdl': (0,3),
                       'side_q_values':(0,36),'side_q_visits':(0,36)}.items():
        a[key] = np.zeros(shape,dtype=RAW_DTYPES[key])
    for key in ("reanalyzed","reanalysis_used_outcome","reanalysis_original_visits","reanalysis_policy_surprise","reanalysis_value_surprise"):
        a[key]=np.zeros(t,dtype=RAW_DTYPES[key])
    a["forbidden_input"] = np.full(t, rule==2, np.uint8)
    return a,m


def compact_search(a):
    indices = np.flatnonzero(a['row_repeats'])
    rows = np.searchsorted(a['sample_indices'], indices)
    for key in ('policies','visits','opponent_policies','opponent_policy_weights','q_visits'):
        a[key] = a[key][rows]
    a['q_values']=np.repeat((a['q_visits']>0).astype(np.int16)*8000,a['row_repeats'][indices],axis=0)
    a['sample_indices'] = indices
    a['forbidden_input'] = np.full(int(a['row_repeats'].sum()), a['rules'][0]==2, np.uint8)


def test_configuration_inheritance_and_fail_fast(tmp_path,config,monkeypatch):
    assert config['network']['canvas']==6 and config['optimizer']['kind']=='sgd'
    assert config['training']['d4_augmentation']
    shutil.copytree(CONFIGS,tmp_path/"configs")
    monkeypatch.setattr('etazero.config.ROOT',tmp_path)
    current=tmp_path/"configs"/"smoke_test"
    (current/"train.cfg.local").write_text("[training]\ntrain_steps=7\n")
    assert load_config(current)["training"]["train_steps"]==7
    (current/"net.cfg.local").write_text("[training]\ntrain_steps=7\n")
    with pytest.raises(ValueError,match="ownership"):
        load_config(current)
    (current/"net.cfg.local").unlink()
    (tmp_path/"configs"/"baseline"/"run.cfg").write_text("[run]\nextends=smoke_test\n")
    with pytest.raises(ValueError,match="cycle"):
        load_config(current)


def test_script_config_environment_and_explicit_override(tmp_path):
    smoke = CONFIGS / 'smoke_test'
    baseline = CONFIGS / 'baseline'
    output = tmp_path / 'run'
    env = {**os.environ, 'CONFIG_DIR': str(smoke)}
    command = ['bash', str(ROOT / 'scripts/run.sh'), 'check-config', '--run-dir', str(output)]
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    selected = json.loads(result.stdout)['config']
    assert selected['network']['canvas'] == 6
    assert selected['run']['run_dir'] == str(output)
    result = subprocess.run(command + ['--config-dir', str(baseline)], cwd=tmp_path,
                            env=env, text=True, capture_output=True, check=True)
    selected = json.loads(result.stdout)['config']
    assert selected['network']['canvas'] == 15
    assert selected['run']['run_dir'] == str(output)
    assert selected['run']['max_iteration'] == 0


def test_auto_resume_rejects_unknown_data_and_weights(tmp_path,config):
    from etazero.runtime import run_training
    unrelated=tmp_path/"game.npz";unrelated.write_bytes(b"preserve existing data")
    with pytest.raises(ValueError,match="empty output directory"):
        run_training(tmp_path,config,tmp_path/"missing_binary")
    assert unrelated.read_bytes()==b"preserve existing data"
    with pytest.raises(ValueError,match="existing run.json"):
        run_training(tmp_path,config,tmp_path/"missing_binary",resume=True)
    save_json(tmp_path/".internal/run.json",{})
    with pytest.raises(ValueError,match="Weights initialization and resume"):
        run_training(tmp_path,config,tmp_path/"missing_binary",weights=tmp_path/"weights.pt")
    assert unrelated.read_bytes()==b"preserve existing data"


def test_source_identity_excludes_sibling_run_data(tmp_path,monkeypatch):
    from etazero import runtime
    source=tmp_path/"version";source.mkdir()
    (source/"code.py").write_text('print("source")\n')
    for name in ("AGENTS.md","RULES.md"):
        (tmp_path/name).write_text("rules\n")
    sibling=source/"data"/"other_run";sibling.mkdir(parents=True)
    output=source/"data"/"current_run";output.mkdir()
    artifact=sibling/"checkpoint.pt";artifact.write_bytes(b"old checkpoint")
    monkeypatch.setattr(runtime,"ROOT",source)
    identity=runtime.source_snapshot(output)
    artifact.write_bytes(b"new checkpoint")
    assert runtime.source_snapshot(output)==identity
    manifest=json.loads((output/".internal/source"/(identity+".tar.json")).read_text())
    assert set(manifest["files"])=={"version/code.py","AGENTS.md","RULES.md"}
    (source/"code.py").write_text('print("changed source")\n')
    assert runtime.source_snapshot(output)!=identity


@pytest.mark.parametrize("algorithm,root,nonroot,message",[
    ("alphazero","puct","gumbel","not an allowed"),
    ("muzero","puct","puct","explicit muzero and unroll"),
    ("alphazero","gumbel","puct","not implemented"),
    ("alphazero","gumbel","gumbel","not implemented"),
])
def test_combination_errors(config,algorithm,root,nonroot,message):
    config["agent"]={"algorithm":algorithm,"root_search_algo":root,"nonroot_search_algo":nonroot}
    with pytest.raises(ValueError,match=message):
        validate(config)


def test_masked_statistics_and_training_policy_domain(config):
    bn=MaskedBatchNorm(1)
    x=torch.tensor([[[[1.,3.],[1000.,1000.]]]])
    mask=torch.tensor([[[[1.,1.],[0.,0.]]]])
    result=bn(x,mask)
    torch.testing.assert_close(result[0,0,0],torch.tensor([-1.,1.])/math.sqrt(1.0001))
    assert result[0,0,1].eq(0).all()
    torch.testing.assert_close(bn.running_mean,torch.tensor([0.002]))
    torch.testing.assert_close(bn.running_std,torch.tensor([1+0.001*(math.sqrt(1.0001)-1)]))
    obs=torch.from_numpy(example_inputs(6,5,"standard",(0,))[0]).unsqueeze(0)
    policy=torch.zeros(1,36);policy[0,1]=1
    model=make_network(config)
    loss,pl,opl,spl,sopl,vl=losses(torch.zeros(1,4,36),torch.zeros(1,3),obs,policy,
                                 policy,torch.zeros(1),torch.tensor([[1.,0.,0.]]),8.0)
    assert float(pl)==pytest.approx(.930*math.log(25),abs=1e-6) # includes the occupied point, excludes padding
    assert float(vl.detach())==pytest.approx(.72*math.log(3),abs=1e-6)
    assert float(opl)==float(sopl)==0
    assert float(spl)==pytest.approx(8*math.log(25),abs=1e-5)
    assert float(loss.detach())==pytest.approx(8.930*math.log(25)+.72*math.log(3),abs=1e-5)


def test_full_trajectory_targets_and_corruption(tmp_path):
    a,_=winning_record()
    path=tmp_path/"game.npz";save_npz(path,a)
    loaded=read_raw(path);view=training_view(loaded)
    np.testing.assert_array_equal(view["value"].argmax(1),[0,2,0,2,0,2,0,2,0])
    assert len(loaded["observations"])==len(view["obs"])+1
    bad=copy.deepcopy(a);bad["policies"][0]=0
    with pytest.raises(ValueError,match="policy target"):
        validate_raw(bad)
    bad=copy.deepcopy(a);obs=unpack_observations(bad["observations"],6);obs[1,1,0,0]=1
    bad["observations"]=pack_observations(obs)
    with pytest.raises(ValueError,match="transition mismatch"):
        validate_raw(bad)


def test_four_policy_losses_and_gradients_against_independent_probabilities():
    # Three on-board cells; one occupied cell and one padding cell.
    obs=torch.zeros(2,5,2,2);obs[:,0].flatten(1)[:,:3]=1;obs[:,1,0,0]=1
    primary=torch.tensor([[0.,1.,0.,0.],[0.,0.,1.,0.]])
    opponent=torch.tensor([[0.,0.,1.,0.],[.25,.25,.25,.25]])
    enabled=torch.tensor([1.,0.])
    logits=torch.tensor([[[1.,2.,3.,99.],[3.,1.,2.,99.],[2.,3.,1.,99.],[1.,3.,2.,99.]],
                         [[3.,2.,1.,99.],[2.,1.,3.,99.],[3.,1.,2.,99.],[2.,3.,1.,99.]]],requires_grad=True)
    result=losses(logits,torch.zeros(2,3),obs,primary,opponent,enabled,
                  torch.tensor([[1.,0.,0.],[0.,0.,1.]]),8.0)
    # Independent scalar construction of softened target ratios and analytic CE gradient.
    expected=np.zeros(4);gradient=np.zeros((2,4,4))
    for sample in range(2):
        for head in range(4):
            hard=(primary if head%2==0 else opponent)[sample,:3].numpy().astype(float)
            target=hard if head<2 else np.array([math.sqrt(math.sqrt(float(p)+1e-7)) for p in hard])
            if head>=2: target/=sum(target)
            weight=(1 if head%2==0 else .15*float(enabled[sample]))*(1 if head<2 else 8)
            if head==0: weight*=.930
            unnormalized=np.exp(logits.detach().numpy()[sample,head,:3].astype(float))
            probs=unnormalized/sum(unnormalized)
            expected[head]+=weight*sum(-p*math.log(q) for p,q in zip(target,probs))/2
            gradient[sample,head,:3]=weight*(probs-target)/2
    np.testing.assert_allclose([float(x.detach()) for x in result[1:5]],expected,rtol=2e-6)
    assert float(result[0].detach())==pytest.approx(sum(expected)+.72*math.log(3),rel=2e-6)
    result[0].backward()
    np.testing.assert_allclose(logits.grad.numpy(),gradient,atol=2e-7,rtol=2e-6)
    # Soft primary explicitly supervises occupied cell 0; padded cell 3 has zero gradient.
    assert gradient[0,2,0] != 0 and not logits.grad[:,:,3].any()
    assert not logits.grad[1,[1,3]].any()


def test_network_preserves_independent_policy_and_value_features(config):
    # Six targets must be able to express six spatial patterns. Centering
    # removes the per-head constant, which has no effect on policy softmax.
    # Also retain independent pooled board features before the value MLP.
    with torch.random.fork_rng():
        torch.manual_seed(23)
        model=make_network(config).eval()
        obs=torch.zeros(16,5,6,6);obs[:,0]=1
        for row in range(16):
            cells=torch.randperm(36)[:row+2]
            obs[row,1].flatten()[cells[::2]]=1
            obs[row,2].flatten()[cells[1::2]]=1
        pooled=[]
        hook=model.value_head.hidden.register_forward_pre_hook(lambda module,args:pooled.append(args[0]))
        try:
            with torch.no_grad():
                policy,_=model(obs,torch.zeros(16,len(GLOBALS)))
        finally:
            hook.remove()
    centered=policy-policy.mean(2,keepdim=True)
    assert (torch.linalg.matrix_rank(centered,atol=1e-5)==6).all()
    features=pooled[0]-pooled[0].mean(0,keepdim=True)
    assert torch.linalg.matrix_rank(features,atol=1e-5)>=4


def test_opponent_targets_survive_missing_successor_and_terminal_boundary():
    a,m=winning_record()
    # Keep only turn 0 (whose successor is cheap/unselected) and the terminal turn.
    a['row_repeats'][:]=0;a['row_repeats'][[0,8]]=[2,1]
    a['target_weights']=a['row_repeats'].astype(np.float32);a['cheap_search'][1]=1
    compact_search(a);m['rows']=3;a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    validate_raw(a);view=training_view(a)
    assert view['opponent_policy'].argmax(1)[:2].tolist()==[6,6]
    assert view['opponent_policy_weight'].tolist()==[1,1,0]
    assert view['policy'].argmax(1).tolist()==[0,0,4]
    bad=copy.deepcopy(a);bad['opponent_policy_weights'][-1]=1
    with pytest.raises(ValueError,match='terminal weight'):validate_raw(bad)
    bad=copy.deepcopy(a);bad['opponent_policies'][0]=bad['policies'][0]
    with pytest.raises(ValueError,match='successor action'):validate_raw(bad)


@pytest.mark.parametrize("waves",[1,3])
def test_shuffle_conservation_and_reader_restoration(tmp_path,config,waves):
    config["replay"].update(min_rows=9,taper_exponent=1,expand_per_row=1,keep_target_rows='all')
    config["shuffle"].update(bucket_rows=2,training_shard_rows=2,waves=waves,temp_dir=str(tmp_path/"scratch"))
    entries=[];expected=Counter()
    for rule in range(3):
        a,m=winning_record(rule);path=tmp_path/"selfplay"/f"{rule}.npz";save_npz(path,a)
        entries.append({"path":str(path.relative_to(tmp_path)),"sha256":sha256(path),"metadata":m})
        view=training_view(a)
        expected.update((obs.tobytes(),p.tobytes(),op.tobytes(),float(w),v.tobytes())
                        for obs,p,op,w,v in zip(view["obs"],view["policy"],view['opponent_policy'],
                                               view['opponent_policy_weight'],view["value"]))
    identity=build_snapshot(tmp_path,1,entries,config)
    snapshot=tmp_path/"snapshots"/identity
    manifest=json.loads((snapshot/"manifest.json").read_text());actual=Counter()
    for item in manifest["files"]:
        with np.load(snapshot/"data"/item["path"]) as a:
            actual.update((obs.tobytes(),p.tobytes(),op.tobytes(),float(w),v.tobytes())
                          for obs,p,op,w,v in zip(a["obs"],a["policy"],a['opponent_policy'],
                                                 a['opponent_policy_weight'],a["value"]))
    assert expected==actual and manifest["rows"]==27
    assert manifest["resource_plan"]["waves"]==waves
    assert not list((tmp_path/"scratch").iterdir())
    first=BatchReader(snapshot,1,2,1);first.next();state=first.state()
    second=BatchReader(snapshot,1,2,999,state)
    try:
        for _ in range(5):
            a,b=first.next(),second.next()
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])
            assert len(a["value"])==1
    finally:
        first.close();second.close()


def test_bit_order_and_non_byte_aligned_planes():
    obs=np.zeros((1,5,5,5),np.uint8)
    obs[0,0].flat[[0,7,8,24]]=1
    packed=pack_observations(obs)
    np.testing.assert_array_equal(packed[0,0],[129,128,0,128]) # independent MSB-first byte values
    np.testing.assert_array_equal(unpack_observations(packed,5),obs)
    a,_=winning_record();a["observations"][0,0,-1]|=1
    with pytest.raises(ValueError,match="tail bits"):
        validate_raw(a)


def test_shuffle_resource_limits_and_failure_cleanup(tmp_path,config):
    a,m=winning_record();path=tmp_path/"selfplay"/"a.npz";save_npz(path,a)
    entries=[{"path":str(path.relative_to(tmp_path)),"sha256":sha256(path),"metadata":m}]
    config["replay"]["min_rows"]=9
    config["shuffle"].update(memory_mb=1,workers=4,training_shard_rows=4096)
    with pytest.raises(ValueError,match="memory"):
        resource_plan(10000,5000,3,config)
    config["shuffle"].update(memory_mb=2048,training_shard_rows=2,waves=3,temp_dir=str(tmp_path/"scratch"))
    a['globals']=a['globals'].astype(np.float64)
    bad=path.with_name('bad.npz');save_npz(bad,a);os.replace(bad,path)
    with pytest.raises(ValueError,match="dtype mismatch"):
        build_snapshot(tmp_path,1,entries,config)
    assert not list((tmp_path/"scratch").iterdir())
    assert not list((tmp_path/"snapshots").iterdir())


def test_katago_window_hand_values(config):
    replay=config["replay"]
    replay.update(min_rows=100,taper_exponent=0.5,expand_per_row=1)
    assert desired_window(100,replay)==100
    assert desired_window(400,replay)==300 # sqrt(400)-sqrt(100), divided by 0.5/sqrt(100), plus 100
    validate(config)
    assert desired_window(10000,replay)==1900 # No cap: 100 + (sqrt(10000)-10)/0.05.
    assert desired_window(0,replay)==100
    replay['keep_target_rows']=0
    with pytest.raises(ValueError,match='positive'):
        validate(config)
    replay['keep_target_rows']=-1
    with pytest.raises(ValueError,match='positive'):
        validate(config)


def test_replay_configuration_has_source_window_extensions(tmp_path,monkeypatch):
    baseline=load_config(CONFIGS/'baseline')
    assert set(baseline['replay'])=={'min_rows','taper_exponent','expand_per_row','keep_target_rows','taper_scale','add_to_data_rows','max_rows'}
    shutil.copytree(CONFIGS,tmp_path/'configs')
    monkeypatch.setattr('etazero.config.ROOT',tmp_path)
    selected=tmp_path/'configs'/'smoke_test'
    local=selected/'train.cfg.local'
    local.write_text('[replay]\nkeep_target_rows=all\n')
    assert load_config(selected)['replay']['keep_target_rows']=='all'
    local.write_text('[replay]\ntaper_scale=32\n')
    assert load_config(selected)['replay']['taper_scale']==32


@pytest.mark.parametrize('files,target,group_rows,expected_rows',[(3,12,128,12),(11,5,52,5)])
def test_katago_sampled_snapshot_waves_and_rebuild(tmp_path,config,files,target,group_rows,expected_rows):
    config['replay'].update(min_rows=9,taper_exponent=1,expand_per_row=1,keep_target_rows=target)
    config['shuffle'].update(group_rows=group_rows,bucket_rows=2,training_shard_rows=2)
    config['writer']['shard_rows']=16
    validate(config)
    entries=[];original=Counter()
    for i in range(files):
        a,m=winning_record(i%3);m.update(attempt_id=str(i),shard_id=str(i))
        a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
        path=tmp_path/'selfplay'/f'{i}.npz';save_npz(path,a)
        entries.append({'path':str(path.relative_to(tmp_path)),'sha256':sha256(path),'metadata':m})
        view=training_view(a)
        original.update((o.tobytes(),p.tobytes(),v.tobytes()) for o,p,v in zip(view['obs'],view['policy'],view['value']))
    def contents(snapshot,manifest):
        result=Counter()
        for item in manifest['files']:
            with np.load(snapshot/'data'/item['path']) as a:
                result.update((o.tobytes(),p.tobytes(),v.tobytes()) for o,p,v in zip(a['obs'],a['policy'],a['value']))
        return result
    sampled=[]
    for waves in (1,3):
        config['shuffle']['waves']=waves
        identity=build_snapshot(tmp_path,1,entries,config)
        snapshot=tmp_path/'snapshots'/identity;manifest=json.loads((snapshot/'manifest.json').read_text())
        assert manifest['window_rows']==files*9 and manifest['rows']==expected_rows
        assert manifest['keep_prob']==pytest.approx(target/(files*9))
        actual=contents(snapshot,manifest)
        assert sum(actual.values())==expected_rows and not (actual-original)
        sampled.append(actual)
        prune_derived(tmp_path,[identity],set())
        restore_snapshot(snapshot)
        assert contents(snapshot,manifest)==actual
    assert sampled[0]==sampled[1] # Wave assignment must not apply sampling again.
    assert all(sha256(tmp_path/e['path'])==e['sha256'] for e in entries)
    if files==11:
        config['replay']['keep_target_rows']=1
        config['shuffle']['group_rows']=9
        with pytest.raises(ValueError,match='empty sample'):
            build_snapshot(tmp_path,2,entries,config)
    config['replay']['min_rows']=files*9+1
    with pytest.raises(ValueError,match='min_rows'):
        build_snapshot(tmp_path,2,entries,config)


def test_unique_catalog_is_idempotent(tmp_path):
    a,_=winning_record();save_npz(tmp_path/"selfplay"/"a.npz",a)
    catalog=Catalog(tmp_path,"test","config")
    try:
        catalog.scan({1:"model"});catalog.scan({1:"model"})
        assert catalog.counts()==(9,1)
        assert catalog.previous_rows_per_game(2)==9.
        save_npz(tmp_path/"selfplay"/"duplicate.npz",a)
        with pytest.raises(Exception):
            catalog.scan({1:"model"})
        assert catalog.counts()==(9,1)
    finally:
        catalog.close()


def test_partition_distribution_and_conservation():
    rng=np.random.default_rng(12)
    pairs=Counter()
    for _ in range(12000):
        order,offsets=partition_rows(2,3,rng)
        labels=np.empty(2,np.int64)
        for bucket in range(3):
            labels[order[offsets[bucket]:offsets[bucket+1]]]=bucket
        pairs[tuple(labels)]+=1
    # Independent labels have nine equally likely outcomes, including collisions.
    assert len(pairs)==9 and all(abs(n/12000-1/9)<0.015 for n in pairs.values())
    order,offsets=partition_rows(10000,256,rng)
    np.testing.assert_array_equal(np.sort(order),np.arange(10000))
    assert offsets[0]==0 and offsets[-1]==10000 and (np.diff(offsets)>=0).all()


def test_cached_views_eviction_and_exact_rebuild(tmp_path,config,monkeypatch):
    import etazero.shuffle as shuffle
    from etazero.data import training_targets
    targets=training_targets(config)
    a,m=winning_record();path=tmp_path/'selfplay'/'a.npz';save_npz(path,a)
    entries=[{'path':str(path.relative_to(tmp_path)),'sha256':sha256(path),'metadata':m}]
    config['replay']['min_rows']=9
    identity=build_snapshot(tmp_path,1,entries,config)
    snapshot=tmp_path/'snapshots'/identity
    manifest_before=(snapshot/'manifest.json').read_bytes()
    hashes={p.name:sha256(p) for p in (snapshot/'data').glob('*.npz')}
    # Cache hits must not invoke trajectory decoding/validation again.
    with monkeypatch.context() as patch:
        patch.setattr(shuffle,'read_raw',lambda *a:(_ for _ in ()).throw(AssertionError('revalidated')))
        view=shuffle._source_view(path,view_key(entries[0],targets),tmp_path/'.internal/training_views',targets)
        assert len(view['value'])==9
    prune_derived(tmp_path,[identity],set())
    assert path.exists() and (snapshot/'manifest.json').read_bytes()==manifest_before
    assert not (snapshot/'data').exists() and not list((tmp_path/'.internal/training_views').glob('*'))
    restore_snapshot(snapshot)
    assert hashes=={p.name:sha256(p) for p in (snapshot/'data').glob('*.npz')}
    assert (snapshot/'manifest.json').read_bytes()==manifest_before
    cache=tmp_path/'.internal/training_views'/(view_key(entries[0],targets)+'.npz')
    cache.write_bytes(b'corrupt cache')
    with pytest.raises(ValueError,match='pickled'):
        shuffle._source_view(path,view_key(entries[0],targets),tmp_path/'.internal/training_views',targets)


def test_catalog_indexed_window_and_incremental_scan(tmp_path):
    catalog=Catalog(tmp_path,'test','config')
    try:
        for iteration in range(1,5):
            a,m=winning_record();m.update(iteration_id=iteration,attempt_id=str(iteration),shard_id=str(iteration),created_ns=iteration)
            a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
            directory=tmp_path/'selfplay'/f'iteration_{iteration:06d}';save_npz(directory/'a.npz',a)
            catalog.scan({iteration:'model'},[directory])
        assert catalog.counts()==(36,4)
        assert catalog.previous_rows_per_game(5)==9.
        assert [e['metadata']['iteration_id'] for e in catalog.entries(10)]==[3,4]
        # Live scans need not revisit committed historical directories; recovery does.
        (tmp_path/'selfplay'/'iteration_000001'/'a.npz').unlink()
        catalog.scan({4:'model'},[tmp_path/'selfplay'/'iteration_000004'])
        with pytest.raises(ValueError,match='missing'):
            catalog.scan({i:'model' for i in range(1,5)})
    finally:
        catalog.close()


def test_journal_barrier_and_failure(tmp_path,monkeypatch):
    from etazero.runtime import Journal
    journal=Journal(tmp_path)
    for step in range(100):
        journal('update',step=step)
    journal.flush()
    records=[json.loads(line) for line in (tmp_path/'logs/events.jsonl').read_text().splitlines()]
    assert [e['step'] for e in records if e['event']=='update']==list(range(100))
    journal.close()
    with (tmp_path/'logs/events.jsonl').open('ab') as file:
        file.write(b'{"truncated":')
    recovered=Journal(tmp_path);recovered('after_recovery');recovered.close()
    assert json.loads((tmp_path/'logs/events.jsonl').read_text().splitlines()[-2])['event']=='after_recovery'
    broken_dir=tmp_path/'broken';broken_dir.mkdir()
    broken=Journal(broken_dir)
    with monkeypatch.context() as patch:
        patch.setattr('etazero.runtime.os.fsync',lambda *a:(_ for _ in ()).throw(OSError('disk failure')))
        broken('update')
        with pytest.raises(RuntimeError,match='writer failed'):
            broken.flush()
        with pytest.raises(RuntimeError,match='writer failed'):
            broken.close()


def test_inference_normalization_keeps_mask_and_model_state(config):
    model=make_network(config)
    probes=[example_inputs(6,size,rule,(0,6,1)) for size in (5,6) for rule in ('freestyle','standard','renju')]
    obs=tuple(torch.from_numpy(np.stack(x)) for x in zip(*probes))
    for _ in range(3):
        model(*obs)
    model.eval();original={k:v.clone() for k,v in model.state_dict().items()}
    folded=inference_network(model)
    with torch.inference_mode():
        p,v=model(*obs)
        assert p.shape==(len(probes),6,36)
        assert folded(*obs)[0].shape==(len(probes),36)
        for a,b in zip((p[:,0],v),folded(*obs)):
            torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,original[key],rtol=0,atol=0)
    assert not any(isinstance(layer,MaskedBatchNorm) for layer in folded.modules())


def test_prefix_excluded_from_targets_catalog_and_shuffle(tmp_path,config):
    a,m=winning_record()
    prefix=4
    a['opening_moves'][0]=prefix;a['balanced_moves'][0]=prefix
    a['opening_status'][0]=1;a['opening_attempts'][0]=1
    a['train_mask'][:prefix]=0;a['row_repeats'][:prefix]=0;a['target_weights'][:prefix]=0
    for key in ('policies','visits','simulations','temperatures'):
        a[key][:prefix]=0
    m['rows']=5;a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    compact_search(a)
    validate_raw(a)
    view=training_view(a)
    assert len(view['value'])==5
    np.testing.assert_array_equal(view['value'].argmax(1),[0,2,0,2,0])
    np.testing.assert_array_equal(view['policy'],a['policies'])
    np.testing.assert_array_equal(view['obs'][0],a['observations'][4])
    path=tmp_path/'selfplay'/'game.npz';save_npz(path,a)
    catalog=Catalog(tmp_path,'test','config')
    try:
        catalog.scan({1:'model'})
        assert catalog.counts()==(5,1)
        assert catalog.previous_rows_per_game(2)==5.
        assert catalog.statistics(1)['avg_game_length']==9
        config['replay'].update(min_rows=5,keep_target_rows='all')
        snapshot=build_snapshot(tmp_path,1,catalog.entries(),config)
        assert json.loads((tmp_path/'snapshots'/snapshot/'manifest.json').read_text())['rows']==5
    finally:
        catalog.close()
    corrupt=copy.deepcopy(a);corrupt['train_mask'][1]=1;corrupt['row_repeats'][1]=1;corrupt['target_weights'][1]=1
    m['rows']=6;corrupt['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    with pytest.raises(ValueError,match='sample indices|temperature/budget|prefix mask|completed visits'):
        validate_raw(corrupt)
    # Policy init may end a game before search: retained evidence, zero training rows.
    a['train_mask'][:]=0;a['row_repeats'][:]=0;a['target_weights'][:]=0;a['opening_moves'][0]=9;a['policy_moves'][0]=5
    for key in ('policies','visits','simulations','temperatures'):
        a[key][:]=0
    m['rows']=0;a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    compact_search(a)
    validate_raw(a)
    assert len(training_view(a)['value'])==0


def test_cold_start_quota_hand_values(config):
    from etazero.runtime import iteration_plan
    config['training'].update(train_steps=1000,batch_size=128,replay_ratio=8)
    config['replay']['min_rows']=10000
    state={'iteration':0,'model':None,'checkpoint':{'id':'initial'},'target_rows':0,'replay_origin_rows':None,'replay_rows':0,'train_credit':0}
    assert iteration_plan(state,config)['train_steps']==0
    assert iteration_plan(state,config)['input_model']['evaluator']=='random'
    state['iteration']=1
    assert iteration_plan(state,config)['target_rows']==10000
    # 90000 excess bootstrap rows cannot pay for the next 16000 steady-state rows.
    state.update(iteration=2,model={'id':'trained'},target_rows=100000,replay_origin_rows=100000,replay_rows=100000)
    plan=iteration_plan(state,config)
    assert plan['target_rows']==116000
    assert math.ceil((plan['target_rows']-100000)/80)==200
    assert state['target_rows']==100000 # planning does not mutate committed state
    assert iteration_plan(state,config)==plan


@pytest.mark.parametrize('section,key,value',[('opening','probability',1.1),('opening','max_tries',0),
                                              ('policy_init','policy_temperature',0.01)])
def test_opening_config_validation(config,section,key,value):
    config[section][key]=value
    with pytest.raises(ValueError):
        validate(config)


def test_iteration_one_backfills_actual_shortfall(config):
    from etazero.runtime import Controller
    config['replay']['min_rows']=50
    class VariableLengthCatalog:
        rows=10;games=1
        def counts(self):
            return self.rows,self.games
        def previous_rows_per_game(self,iteration):
            assert iteration==1
            return 10.
    controller=object.__new__(Controller)
    controller.config=config;controller.catalog=VariableLengthCatalog()
    controller.stop=False;controller.services={}
    controller.stopping=lambda:False
    controller.journal=lambda *a,**k:None
    controller.scan=lambda:None
    quotas=[]
    def launch(plan,games):
        quotas.append(games)
        # The bootstrap game had ten effective rows; subsequent games only three.
        controller.catalog.rows+=games*3;controller.catalog.games+=games
    controller.launch=launch
    controller.produce({'iteration':1,'target_rows':50})
    assert quotas==[4,3,2,2,1,1,1]
    assert controller.catalog.rows==52


def test_adamw_decoupled_decay_and_config(config):
    from etazero.training import optimizer_for
    c=copy.deepcopy(config);c['optimizer']['kind']='adamw'
    model=make_network(c)
    with torch.no_grad(): model.stem.weight.fill_(2.)
    optimizer=optimizer_for(model,c)
    assert isinstance(optimizer,torch.optim.AdamW)
    group=next(g for g in optimizer.param_groups if g['group_name']=='input')
    model.stem.weight.grad=torch.zeros_like(model.stem.weight)
    optimizer.step()
    torch.testing.assert_close(model.stem.weight,torch.full_like(model.stem.weight,2*(1-group['lr']*group['weight_decay'])))
    c=copy.deepcopy(config);c['environment'].update(rules='renju,freestyle,standard,',rule_weights='1,0,0')
    validate(c)
    for weights in ('0,0,0','1,-1,0','1,nan,0'):
        c['environment']['rule_weights']=weights
        with pytest.raises(ValueError,match='weights'): validate(c)
    for probability in (-.1,1.1):
        c=copy.deepcopy(config);c['environment']['forbidden_feature_dropout_prob']=probability
        with pytest.raises(ValueError): validate(c)


def test_global_feature_corruption():
    a,_=winning_record(2)
    validate_raw(a)
    a['globals'][0,2]=1
    with pytest.raises(ValueError,match='rule/color'): validate_raw(a)
    a,_=winning_record(2)
    obs=unpack_observations(a['observations'],6);obs[2,3,2,2]=1
    a['observations']=pack_observations(obs);a['globals'][2,3]=0
    with pytest.raises(ValueError,match='dropped forbidden'): validate_raw(a)


def test_wdl_loss_distinguishes_draw_from_equal_win_loss():
    obs=torch.ones(2,5,5,5);policy=torch.zeros(2,25);policy[:,0]=1
    logits=torch.zeros(2,4,25)
    wdl_logits=torch.tensor([[0.,4.,0.],[0.,4.,0.]],requires_grad=True)
    target=torch.tensor([[0.,1.,0.],[.5,0.,.5]])
    *_,vl=losses(logits,wdl_logits,obs,policy,policy,torch.ones(2),target,8.0)
    # Same scalar Q=0 for both distributions, but the second target must penalize draw confidence.
    denominator=math.exp(4)+2
    assert float(vl.detach())==pytest.approx(.72*(math.log(denominator)-2),rel=1e-6)
    vl.backward()
    probabilities=np.array([1,math.exp(4),1])/denominator
    np.testing.assert_allclose(wdl_logits.grad.numpy(),.72*(probabilities[None]-target.numpy())/2,
                               rtol=2e-6,atol=2e-8)


def test_sample_repeats_survive_catalog_shuffle_and_reader(tmp_path,config):
    a,m=winning_record()
    repeats=np.array([2,0,3,0,1,0,1,0,2],np.int32)
    a['row_repeats']=repeats;a['target_weights']=repeats.astype(np.float32)
    # Targets may differ from normalized visits after pruning/LCB.
    a['visits'][0,1]=4;a['policies'][0,0]=8;a['policies'][0,1]=2
    m['rows']=int(repeats.sum());a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    compact_search(a)
    validate_raw(a);view=training_view(a)
    np.testing.assert_array_equal(view['value'],np.tile([1,0,0],(m['rows'],1)))
    assert np.array_equal(view['obs'][0],view['obs'][1])
    path=tmp_path/'selfplay/game.npz';save_npz(path,a)
    catalog=Catalog(tmp_path,'test','config')
    try:
        catalog.scan({1:'model'});assert catalog.counts()==(9,1)
        assert catalog.previous_rows_per_game(2)==9.
        config['replay'].update(min_rows=9,keep_target_rows='all')
        config['shuffle'].update(bucket_rows=9,training_shard_rows=9)
        identity=build_snapshot(tmp_path,1,catalog.entries(),config)
        reader=BatchReader(tmp_path/'snapshots'/identity,9,1,1)
        try:
            batch=reader.next();np.testing.assert_array_equal(batch['value'],view['value'])
            assert sorted(map(bytes,batch['policy']))==sorted(map(bytes,view['policy']))
        finally:reader.close()
    finally:catalog.close()
    a['forbidden_input']=np.zeros(13,np.uint8)
    a['q_values']=np.concatenate((np.repeat(a['q_values'][:1],4,axis=0),a['q_values']))
    a['row_repeats'][0]=6;m['rows']+=4;a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    with pytest.raises(ValueError,match='rounding'):validate_raw(a)


def test_sampled_search_preserves_full_trajectory_and_exact_targets(tmp_path):
    a,m=winning_record()
    original={k:v.copy() for k,v in a.items()}
    a['row_repeats']=np.array([2,0,1,0,0,1,0,0,3],np.int32)
    a['target_weights']=a['row_repeats'].astype(np.float32)
    # A cheap position can regain weight; an unselected full position is omitted too.
    a['cheap_search'][5]=1
    compact_search(a)
    m['rows']=7;a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    path=tmp_path/'sampled.npz';save_npz(path,a)
    loaded=read_raw(path)
    np.testing.assert_array_equal(loaded['sample_indices'],[0,2,5,8])
    assert loaded['visits'].shape==(4,36)
    for key in ('observations','globals','actions','game_offsets','observation_offsets'):
        np.testing.assert_array_equal(loaded[key],original[key])
    view=training_view(loaded)
    for key,source in [('obs','observations'),('globals','globals'),('policy','policies')]:
        np.testing.assert_array_equal(view[key],original[source][[0,0,2,5,8,8,8]])
    np.testing.assert_array_equal(view['value'].argmax(1),[0,0,0,2,0,0,0])
    corrupted=copy.deepcopy(loaded);corrupted['sample_indices'][1]=3
    with pytest.raises(ValueError,match='sample indices'):
        validate_raw(corrupted)


@pytest.mark.parametrize('waves',[1,3])
def test_temp_compression_preserves_snapshot_bytes(tmp_path,config,waves):
    import zipfile
    from etazero.storage import write_npz
    a,m=winning_record()
    raw=tmp_path/'selfplay/game.npz';save_npz(raw,a)
    entry={'path':str(raw.relative_to(tmp_path)),'sha256':sha256(raw),'metadata':m}
    config['replay'].update(min_rows=9,keep_target_rows='all')
    config['shuffle'].update(waves=waves,bucket_rows=2,training_shard_rows=2)
    results=[]
    for compressed in (False,True):
        config['shuffle']['compress_temp']=compressed
        identity=build_snapshot(tmp_path,1,[entry],config)
        manifest=json.loads((tmp_path/'snapshots'/identity/'manifest.json').read_text())
        results.append(manifest['files'])
    assert results[0]==results[1]
    from etazero.data import training_targets
    cache=tmp_path/'.internal/training_views'/(view_key(entry,training_targets(config))+'.npz')
    with zipfile.ZipFile(cache) as archive:
        assert all(item.compress_type==zipfile.ZIP_DEFLATED for item in archive.infolist())
    arrays={'zeros':np.zeros((4096,128),np.float32)}
    for compressed in (False,True):
        path=tmp_path/f'{compressed}.npz';write_npz(path,arrays,compressed)
        with np.load(path,allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored['zeros'],arrays['zeros'])
    assert (tmp_path/'True.npz').stat().st_size<(tmp_path/'False.npz').stat().st_size/20


def test_shuffle_fixed_output_file_count_even_when_empty(tmp_path,config):
    a,m=winning_record()
    raw=tmp_path/'selfplay/game.npz';save_npz(raw,a)
    entry={'path':str(raw.relative_to(tmp_path)),'sha256':sha256(raw),'metadata':m}
    config['replay'].update(min_rows=9,keep_target_rows='all')
    config['shuffle'].update(bucket_rows=96,training_shard_rows=8,waves=1)
    identity=build_snapshot(tmp_path,1,[entry],config)
    manifest=json.loads((tmp_path/'snapshots'/identity/'manifest.json').read_text())
    assert sorted(item['rows'] for item in manifest['files'])==[0]*3+[1]*9
    policies=[]
    for item in manifest['files']:
        with np.load(tmp_path/'snapshots'/identity/'data'/item['path']) as rows:
            policies.extend(map(bytes,rows['policy']))
    assert Counter(policies)==Counter(map(bytes,a['policies'].astype(np.float32)))


@pytest.mark.parametrize('section,key,value',[
    ('search','full_search_visits',1),('search','full_search_visits',2**31),
    ('search','cheap_search_visits',1000),('search','cheap_search_probs',1),
    ('lcb','min_visit_prop_for_lcb',1.1),('fpu','fpu_parent_weight_by_visited_policy_pow',-1),
    ('surprise_weighting','policy_surprise_data_weight',1),('dirichlet_noise','dirichlet_total_concentration',0),
    ('symmetry','root_num_symmetries_to_sample',0),('symmetry','root_num_symmetries_to_sample',9),
    ('temperature','nn_policy_temperature',0),('temperature','root_policy_temperature',0),
    ('temperature','root_policy_temperature_early',0),('temperature','temperature_halflife',0),
    ('temperature','temperature_only_below_prob',1.1),('value_weighting','value_weight_exponent',-0.1),
    ('fpu','fpu_loss_prop',1.1),('puct','c_puct_stdev_scale',1.1),
    ('reduce_visits','reduce_visits_threshold',1),('reduce_visits','reduce_visits_threshold',-0.1),
    ('reduce_visits','reduce_visits_threshold_lookback',0),('reduce_visits','reduce_visits_threshold_lookback',1001),
    ('reduce_visits','reduced_visits_min',1),('reduce_visits','reduced_visits_min',10),
    ('reduce_visits','reduced_visits_weight',1.1)])
def test_search_enhancement_validation(config,section,key,value):
    config[section][key]=value
    with pytest.raises(ValueError):validate(config)


def test_reduce_visits_config_inheritance_and_independent_pcr(tmp_path, config):
    selected = tmp_path / 'child'
    selected.mkdir()
    (selected / 'run.cfg').write_text(f'[run]\nextends=smoke_test\nrun_dir={tmp_path / "run"}\n')
    (selected / 'selfplay.cfg').write_text('[reduce_visits]\nreduce_visits_threshold=.7\nreduced_visits_min=3\n[search]\nfull_search_visits=8\ncheap_search_visits=4\n')
    inherited = load_config(selected)
    assert inherited['reduce_visits']['reduce_visits_threshold'] == .7
    assert inherited['reduce_visits']['reduced_visits_min'] == 3
    assert inherited['search']['cheap_search_visits'] == 4
    # These cases are valid; the two cap paths are alternatives, not stacked limits.
    config['search']['cheap_search_visits']=4
    config['reduce_visits'].update(reduced_visits_min=8,reduce_visits_threshold=0,reduced_visits_weight=0)
    validate(config)
    from etazero.engine_config import load_engine_config
    assert 'reduce_visits' not in load_engine_config(CONFIGS / 'smoke_test')['analysis']
    assert 'reduce_visits' not in load_engine_config(CONFIGS / 'smoke_test',True)['match']


def test_explicit_search_samples_and_independent_precision(config):
    from etazero.engine_config import load_engine_config, validate_engine_config
    baseline=load_config(CONFIGS / 'baseline')
    assert (baseline['search']['full_search_visits'],baseline['search']['cheap_search_visits'])==(400,70)
    assert baseline['puct']==dict(c_puct=1.05,c_puct_log=0.28,c_puct_base=500,
                                 c_puct_stdev_prior=0.4,c_puct_stdev_prior_weight=2,
                                 c_puct_stdev_scale=0,virtual_loss=1)
    assert baseline['temperature']==dict(nn_policy_temperature=1,root_policy_temperature_early=1.5,
                                        root_policy_temperature=1.1,temperature=0.75,final_temperature=0.15,
                                        temperature_halflife=15,temperature_only_below_prob=1)
    assert baseline['dirichlet_noise']['dirichlet_total_concentration']==10.83
    assert baseline['inference']['inference_precision']=='float16' and baseline['training']['amp']=='off'
    assert baseline['symmetry']['nn_randomize'] and baseline['symmetry']['root_num_symmetries_to_sample']==4
    for match in (False,True):
        profile=load_engine_config(CONFIGS / 'baseline',match)
        c=profile['match' if match else 'analysis']
        assert (c['visits'],c['c_puct'],c['c_puct_log'],c['c_puct_stdev_scale'])==(500,1,0.45,0.85)
        assert (c['c_puct_stdev_prior'],c['c_puct_stdev_prior_weight'],c['value_weight_exponent'])==(0.4,2,0.25)
        assert c['nn_randomize'] and c['root_num_symmetries_to_sample']==1
        assert (c['nn_policy_temperature'],c['root_policy_temperature'],c['temperature_early'],c['temperature'])==(1,1,0.6,0.2)
        assert c['reuse_tree'] is match and c['inference_precision']=='auto'
        c.update(device='cpu',visits=1,max_playouts=0,max_time=0,nn_randomize=False,nn_symmetry=7,
                 fpu_parent_weight_by_visited_policy=False,fpu_parent_weight=0.75,
                 fpu_parent_weight_by_visited_policy_pow=0)
        validate_engine_config(profile,match)
    config['search'].update(max_playouts=0,max_time=0)
    config['symmetry'].update(nn_randomize=False,nn_symmetry=7)
    config['fpu'].update(fpu_parent_weight_by_visited_policy=False,fpu_parent_weight=0.75,
                         fpu_parent_weight_by_visited_policy_pow=0)
    validate(config)


def test_output_row_forbidden_dropout_repeats_and_domains(config):
    a,m=winning_record(2)
    a['row_repeats'][2]=4;a['target_weights'][2]=4
    compact_search(a);m['rows']=12;a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    # Same position, four final rows, independently persisted augmentation decisions.
    a['forbidden_input'][2:6]=[1,0,1,0]
    obs=unpack_observations(a['observations'],6)
    obs[2,3,2,2]=1;a['observations']=pack_observations(obs)
    validate_raw(a);view=training_view(a)
    final=unpack_observations(view['obs'],6)
    np.testing.assert_array_equal(view['globals'][2:6,3],[1,0,1,0])
    np.testing.assert_array_equal(final[2:6,3,2,2],[1,0,1,0])
    assert not final[[3,5],3:5].any()
    assert a['globals'][2,3]==1 and obs[2,3,2,2]==1
    for key in ('policy','value','opponent_policy'):
        np.testing.assert_array_equal(view[key][2:6],np.repeat(view[key][2:3],4,axis=0))
    # Learner softmax includes every on-board point, including occupied/forbidden.
    canvas=6;logits=torch.zeros(1,4,canvas*canvas,requires_grad=True)
    input=torch.tensor(final[2:3],dtype=torch.float32)
    policy=torch.tensor(view['policy'][2:3]);opponent=torch.tensor(view['opponent_policy'][2:3])
    value=torch.tensor(view['value'][2:3])
    total,*_=losses(logits,torch.zeros(1,3,requires_grad=True),input,policy,opponent,torch.ones(1),value,8.)
    total.backward();grad=logits.grad[0,0].reshape(canvas,canvas)
    assert grad[0,0]>0 and grad[2,2]>0 and torch.count_nonzero(grad[5])==0 and torch.count_nonzero(grad[:,5])==0


def test_policy_init_loaded_defaults_and_explicit_match_mean(tmp_path,monkeypatch):
    from etazero.engine_config import load_engine_config
    import shutil
    shutil.copytree(CONFIGS,tmp_path/'configs')
    monkeypatch.setattr('etazero.config.ROOT',tmp_path)
    base=tmp_path/'configs/baseline'
    selfplay=base/'selfplay.cfg'
    selfplay.write_text(selfplay.read_text().replace('\npolicy_init_mean = 6\n','\n').replace('\npolicy_temperature = 1.6\n','\n'))
    c=load_config(base)
    assert c['policy_init']['policy_init'] and c['policy_init']['policy_init_mean']==12 and c['policy_init']['policy_temperature']==1
    match=base/'match.cfg'
    match.write_text(match.read_text().replace('\npolicy_init = false\n','\n').replace('\npolicy_temperature = 1\n','\n'))
    c=load_engine_config(base,True,environ={})
    assert not c['opening']['policy_init'] and c['opening']['policy_init_mean']==0 and c['opening']['policy_temperature']==1
    with pytest.raises(ValueError,match='policy_init_mean'):
        load_engine_config(base,True,environ={'MATCH_OPENING_POLICY_INIT':'true'})
    c=load_engine_config(base,True,environ={'MATCH_OPENING_POLICY_INIT':'true','MATCH_OPENING_POLICY_INIT_MEAN':'12','MATCH_OPENING_POLICY_TEMPERATURE':'1.6'})
    assert c['opening']['policy_init_mean']==12 and c['opening']['policy_temperature']==1.6
    for section,text in [('environment','[environment]\nsizes=15x14\n'),('environment','[environment]\nrules=vc1_b\n')]:
        (base/'env.cfg.local').write_text(text)
        with pytest.raises(ValueError):load_config(base)


def test_graph_configuration_bounds_and_environment_override(config):
    from etazero.engine_config import load_engine_config,validate_engine_config
    assert load_config(CONFIGS / 'baseline')['graph_search']==dict(use_graph_search=True,graph_search_catch_up_leak_prob=0)
    for match in (False,True):
        group='match' if match else 'analysis';prefix='MATCH_' if match else 'ANALYSIS_'
        for leak in (0,0.5,1):
            c=load_engine_config(CONFIGS / 'smoke_test',match,environ={prefix+'USE_GRAPH_SEARCH':'false',prefix+'GRAPH_SEARCH_CATCH_UP_LEAK_PROB':str(leak)})
            assert c[group]['use_graph_search'] is False and c[group]['graph_search_catch_up_leak_prob']==leak
        for leak in (-0.1,1.1,float('nan'),float('inf')):
            c=load_engine_config(CONFIGS / 'smoke_test',match)
            c[group]['graph_search_catch_up_leak_prob']=leak
            with pytest.raises(ValueError):validate_engine_config(c,match)
    for leak in (0,0.5,1):
        config['graph_search'].update(use_graph_search=False,graph_search_catch_up_leak_prob=leak);validate(config)
    for leak in (-0.1,1.1,float('nan'),float('inf')):
        config['graph_search']['graph_search_catch_up_leak_prob']=leak
        with pytest.raises(ValueError):validate(config)


def test_search_correction_samples_and_bounds(config):
    from etazero.engine_config import load_engine_config,validate_engine_config
    assert config['uncertainty']==dict(use_uncertainty=False,uncertainty_coeff=.25,uncertainty_exponent=1,uncertainty_max_weight=8)
    assert config['optimistic_policy']==dict(policy_optimism=0,root_policy_optimism=0)
    assert config['noise_pruning']==dict(use_noise_pruning=False,noise_prune_utility_scale=.15,noise_pruning_cap=1e50)
    bounds={'uncertainty_coeff':(.0001,1),'uncertainty_exponent':(0,2),'uncertainty_max_weight':(1,100),
            'policy_optimism':(0,1),'root_policy_optimism':(0,1),'noise_prune_utility_scale':(.001,10),'noise_pruning_cap':(0,1e50)}
    for match in (False,True):
        group='match' if match else 'analysis';prefix='MATCH_' if match else 'ANALYSIS_'
        c=load_engine_config(CONFIGS / 'smoke_test',match)
        assert c[group]['use_uncertainty'] and c[group]['use_noise_pruning']
        assert c[group]['root_policy_optimism']==.2 and c[group]['policy_optimism']==1
        overridden=load_engine_config(CONFIGS / 'smoke_test',match,environ={prefix+'USE_UNCERTAINTY':'false',prefix+'USE_NOISE_PRUNING':'false',prefix+'ROOT_POLICY_OPTIMISM':'0'})
        assert not overridden[group]['use_uncertainty'] and not overridden[group]['use_noise_pruning'] and overridden[group]['root_policy_optimism']==0
        for key,(low,high) in bounds.items():
            original=c[group][key]
            for value in (low,high):
                c[group][key]=value;validate_engine_config(c,match)
            for value in (low-.01,high*2+1,float('inf'),float('nan')):
                c[group][key]=value
                with pytest.raises(ValueError):validate_engine_config(c,match)
            c[group][key]=original
    for section,items in (('uncertainty',('uncertainty_coeff','uncertainty_exponent','uncertainty_max_weight')),
                          ('optimistic_policy',('policy_optimism','root_policy_optimism')),
                          ('noise_pruning',('noise_prune_utility_scale','noise_pruning_cap'))):
        if section == 'uncertainty':config[section]['use_uncertainty']=True
        if section == 'noise_pruning':config[section]['use_noise_pruning']=True
        for key in items:
            low,high=bounds[key];original=config[section][key]
            for value in (low,high):config[section][key]=value;validate(config)
            for value in (low-.01,high*2+1,float('inf'),float('nan')):
                config[section][key]=value
                with pytest.raises(ValueError):validate(config)
            config[section][key]=original
