"""Real-CUDA integration checks. Enable only in an execution context with host device access."""
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from etazero.config import ROOT, load_config, write_native
from etazero.data import read_raw
from etazero.evaluation import evaluate
from etazero.eval_config import load_evaluation_config
from etazero.export import example_inputs, export_model
from etazero.network import make_network, inference_network, MaskedBatchNorm
from etazero.optimization import Optimization, inference_weights
from etazero.runtime import run_training
from etazero.storage import load_json, sha256
from etazero.training import initialize, load_checkpoint, train_iteration, optimizer_for, commit_checkpoint, restore_rng

pytestmark=pytest.mark.skipif(os.environ.get("ETAZERO_GPU_TESTS")!="1",reason="Enable ETAZERO_GPU_TESTS=1 with host CUDA access")
BINARY=ROOT/"build"/"etazero"


def assert_preserved(root, files):
    for path, digest in files.items():
        if path.exists():
            assert sha256(path)==digest
        else:
            archived=list((root/'.internal/discarded').glob('*/'+str(path.relative_to(root))))
            assert len(archived)==1 and sha256(archived[0])==digest


@pytest.fixture(scope="module")
def gpu_config():
    assert torch.cuda.is_available(),"GPU tests require actual host CUDA access"
    torch.set_num_threads(1)
    return load_config(ROOT/"configs"/"smoke_test")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory,gpu_config):
    root=tmp_path_factory.mktemp("pipeline")
    state=run_training(root,gpu_config,BINARY,max_iteration=2)
    return root,state


def test_script_starts_and_auto_resumes_training(tmp_path):
    configs=tmp_path/"configs";shutil.copytree(ROOT/"configs",configs)
    root=tmp_path/"selfplay"/"smoke_test"
    selected=configs/"smoke_test"
    (selected/"run.cfg").write_text(f"[run]\nextends = baseline\nrun_dir = {root}\nmax_iteration = 1\ncpu_threads = 1\n")
    env={**os.environ,"CONFIG_DIR":str(selected)}
    command=["bash",str(ROOT/"scripts"/"run.sh")]
    def launch(*args):
        result=subprocess.run(command+list(args),cwd=tmp_path,env=env,text=True,capture_output=True,timeout=180)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        return load_json(root/".internal/state.json")
    first=launch()
    assert first["iteration"]==2 and first["checkpoint"]["total_steps"]==4
    run_id=first["run_id"]
    shards={p:sha256(p) for p in (root/"selfplay").rglob("*.npz")}
    assert shards
    second=launch("--iterations","2")
    assert second["run_id"]==run_id and second["iteration"]==3
    assert second["checkpoint"]["total_steps"]==8
    assert all(sha256(path)==checksum for path,checksum in shards.items())
    assert launch("--plot")==second
    assert (root/"training.png").is_file()


def test_unlimited_iterations_stop_and_auto_resume(tmp_path):
    configs=tmp_path/"configs";shutil.copytree(ROOT/"configs",configs)
    root=tmp_path/"run"
    selected=configs/"smoke_test"
    (selected/"run.cfg").write_text(f"[run]\nextends = baseline\nrun_dir = {root}\nmax_iteration = 0\ncpu_threads = 1\n")
    env={**os.environ,"CONFIG_DIR":str(selected)}
    command=["bash",str(ROOT/"scripts"/"run.sh")]
    stdout=tmp_path/"stdout";stderr=tmp_path/"stderr"
    with stdout.open('w') as out,stderr.open('w') as err:
        process=subprocess.Popen(command,cwd=tmp_path,env=env,stdout=out,stderr=err,start_new_session=True)
        deadline=time.monotonic()+90
        try:
            while time.monotonic()<deadline:
                state_path=root/".internal/state.json"
                if state_path.exists() and load_json(state_path)["iteration"]>=3:
                    process.send_signal(signal.SIGINT)
                    break
                if process.poll() is not None:
                    pytest.fail("Unlimited training exited before two iterations: "+stderr.read_text())
                time.sleep(0.01)
            else:
                pytest.fail("Timed out waiting for unlimited training to complete two iterations")
            assert process.wait(timeout=30)==0,stdout.read_text()+'\n'+stderr.read_text()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
    first=load_json(root/".internal/state.json")
    assert first["iteration"]>=3
    assert load_json(root/"config/effective.json")["run"]["max_iteration"]==0
    events=[json.loads(line) for line in (root/"logs/events.jsonl").read_text().splitlines()]
    assert any(e["event"]=="run_stopped" and e["completed_iterations"]>=2 for e in events)
    shards={p:sha256(p) for p in (root/"selfplay").rglob("*.npz")}
    target=first["iteration"]
    result=subprocess.run(command+["--iterations",str(target)],cwd=tmp_path,env=env,
                          text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    resumed=load_json(root/".internal/state.json")
    assert resumed["run_id"]==first["run_id"] and resumed["iteration"]==target+1
    assert resumed["checkpoint"]["total_steps"]==target*4
    assert all(sha256(path)==checksum for path,checksum in shards.items())


def test_gpu_pipeline_and_idempotent_completed_resume(pipeline,gpu_config):
    root,state=pipeline
    assert state["checkpoint"]["total_steps"]==8
    assert state["checkpoint"]["total_samples"]==64
    shards={p:sha256(p) for p in (root/"selfplay").rglob("*.npz")}
    assert shards
    for path in shards:
        read_raw(path)
    events=[json.loads(line) for line in (root/"logs/events.jsonl").read_text().splitlines()]
    assert max(e["max_batch"] for e in events if e["event"]=="inference")>1
    starts=[e for e in events if e['event']=='worker_start']
    assert len(starts)>1 and len({e['pid'] for e in starts})==1
    for entry in starts:
        attempt=load_json(root/'.internal/iterations'/f"{entry['iteration']:06d}"/f"attempt_{entry['attempt']}.json")
        assert attempt['model_id']==load_json(root/'.internal/iterations'/f"{entry['iteration']:06d}"/'plan.json')['input_model']['id']
    before=len(list((root/"checkpoints").glob("*.pt")))
    resumed=run_training(root,gpu_config,BINARY,resume=True,max_iteration=2)
    assert state==resumed and len(list((root/"checkpoints").glob("*.pt")))==before
    assert shards=={p:sha256(p) for p in shards}


def test_unbounded_replay_window_pipeline(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config)
    c['replay'].update(keep_target_rows='all',taper_exponent=1,expand_per_row=1)
    state=run_training(tmp_path,c,BINARY,max_iteration=2)
    status=load_json(tmp_path/'.internal/iterations'/'000002'/'status.json')
    snapshot=load_json(tmp_path/'snapshots'/status['snapshot_id']/'manifest.json')
    generated_rows=sum(int(read_raw(path)['row_repeats'].sum()) for path in (tmp_path/'selfplay').rglob('*.npz'))
    assert snapshot['rows']==generated_rows and generated_rows>c['replay']['min_rows']
    assert state['checkpoint']['total_steps']==8


def test_sampled_replay_pipeline_and_resume(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config)
    c['replay']['keep_target_rows']=12;c['shuffle']['waves']=3
    state=run_training(tmp_path,c,BINARY,max_iteration=2)
    for iteration in (1,2):
        status=load_json(tmp_path/'.internal/iterations'/f'{iteration:06d}'/'status.json')
        snapshot=load_json(tmp_path/'snapshots'/status['snapshot_id']/'manifest.json')
        assert snapshot['keep_prob']<1 and 0<snapshot['rows']<snapshot['window_rows']
    assert state['checkpoint']['total_steps']==8
    assert run_training(tmp_path,c,BINARY,resume=True,max_iteration=2)==state


def test_native_parity_all_sizes_rules_and_match(pipeline,gpu_config):
    root,state=pipeline
    checkpoint=load_checkpoint(root,state["checkpoint"],gpu_config)
    model=make_network(gpu_config).cuda().eval();model.load_state_dict(inference_weights(checkpoint))
    canvas=gpu_config["network"]["canvas"]
    for size in (5,6):
        for rule in ("freestyle","standard","renju"):
            command=[str(BINARY),"infer","--config",str(root/"config/effective.cfg"),"--model",str(root/state["model"]["path"]),
                     "--model-id",state["model"]["id"],"--device","cuda:0","--size",str(size),"--rule",rule,
                     "--moves",f"0,{canvas},1"]
            native=json.loads(subprocess.run(command,text=True,capture_output=True,check=True).stdout)
            with torch.inference_mode():
                p,v=model(*(torch.from_numpy(x).unsqueeze(0).cuda() for x in example_inputs(canvas,size,rule,(0,canvas,1))))
            np.testing.assert_allclose(native["raw_logits"],p[0].cpu(),rtol=2e-4,atol=2e-5)
            np.testing.assert_allclose(native["raw_wdl"],v[0].float().softmax(0).cpu(),rtol=2e-4,atol=2e-5)
    model_b=root/load_json(root/".internal/iterations"/"000001"/"status.json")["model"]["path"]
    _,result=evaluate(load_evaluation_config(ROOT/"configs/smoke_test",match=True),BINARY,root,model_b=model_b,size=5,rule="renju",games=4)
    assert result["result"]["complete"] and len(result["result"]["games"])==4
    assert {g["winner"] for g in result["result"]["games"]}<={-1,0,1}
    assert all(set(g["root_visits"])=={100} for g in result["result"]["games"])
    _,result=evaluate(load_evaluation_config(ROOT/"configs/smoke_test"),BINARY,root,size=5,rule="freestyle",moves="0,5,1,6,2,7,3,8,4")
    assert result["result"]["terminal"] and result["result"]["value"]==-1


def test_root_d4_probabilities_and_policy_temperatures(pipeline,gpu_config):
    from etazero.symmetry import apply_symmetry
    root,state=pipeline
    checkpoint=load_checkpoint(root,state['checkpoint'],gpu_config)
    model=make_network(gpu_config).cuda().eval();model.load_state_dict(inference_weights(checkpoint))
    canvas=gpu_config['network']['canvas']
    obs,globals=example_inputs(canvas,5,'renju',(0,canvas))
    tensor=torch.from_numpy(obs).unsqueeze(0).cuda()
    global_tensor=torch.from_numpy(globals).unsqueeze(0).cuda()
    inverse=(0,3,2,1,4,5,6,7)
    legal=torch.from_numpy(((obs[0]>0)&(obs[1]==0)&(obs[2]==0)).flatten()).cuda()
    policies,values=[],[]
    with torch.inference_mode():
        for symmetry in range(8):
            logits,value=model(apply_symmetry(tensor,symmetry).contiguous(),global_tensor)
            restored=apply_symmetry(logits.reshape(1,canvas,canvas),inverse[symmetry]).flatten(1)
            policies.append(torch.softmax((restored/1.7).masked_fill(~legal,-torch.inf),dim=1))
            values.append(value.float().softmax(1))
    expected_policy=torch.stack(policies).mean(0)[0].cpu().numpy()
    expected_wdl=torch.stack(values).mean(0)[0].cpu().numpy()
    config=load_evaluation_config(ROOT/'configs/smoke_test')
    config['evaluation'].update(root_num_symmetries_to_sample=8,nn_policy_temperature=1.7,
                               root_policy_temperature_early=2,root_policy_temperature=1.5,
                               temperature_halflife=7,temperature_early=0.8,temperature=0.1)
    _,payload=evaluate(config,BINARY,root,size=5,rule='renju',moves='0,5')
    result=payload['result'];crop=[y*canvas+x for y in range(5) for x in range(5)]
    np.testing.assert_allclose(result['network_policy'],expected_policy[crop],rtol=3e-4,atol=2e-5)
    np.testing.assert_allclose(result['network_wdl'],expected_wdl,rtol=3e-4,atol=2e-5)
    root_temperature=1.5+0.5*0.5**(2/7*19/5)
    expected_search=expected_policy**(1/root_temperature);expected_search/=expected_search.sum()
    np.testing.assert_allclose(result['search_policy'],expected_search[crop],rtol=3e-4,atol=2e-5)
    assert result['root_visits']==100 and result['simulations']==99
    assert sum(result['visits'])==99 and result['nn_requests']>=9


def test_learner_resume_matches_uninterrupted_updates(tmp_path,pipeline,gpu_config):
    root,_=pipeline
    plan=load_json(root/".internal/iterations"/"000001"/"plan.json")
    status=load_json(root/".internal/iterations"/"000001"/"status.json")
    plan["snapshot_id"]=status["snapshot_id"]
    base=plan["input_checkpoint"]
    dirs=[tmp_path/"full",tmp_path/"resumed"]
    for dest in dirs:
        shutil.copytree(root/"checkpoints",dest/"checkpoints")
        shutil.copytree(root/"snapshots"/plan["snapshot_id"],dest/"snapshots"/plan["snapshot_id"])
    logs=[]
    def log(event,**fields):
        if event=="update":
            logs.append(fields)
    deterministic=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        full,done=train_iteration(dirs[0],gpu_config,plan,base,log)
        assert done
        logs.clear()
        partial,done=train_iteration(dirs[1],gpu_config,plan,base,log,lambda:len(logs)>=1)
        assert not done and partial['step']==1
        midway=load_checkpoint(dirs[1],partial,gpu_config)
        assert midway['optimization']['lookahead_counter']==1
        resumed,done=train_iteration(dirs[1],gpu_config,plan,base,log)
        assert done and resumed["total_steps"]==4
        a=load_checkpoint(dirs[0],full,gpu_config);b=load_checkpoint(dirs[1],resumed,gpu_config)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key],b["model"][key],rtol=0,atol=0)
        for key in a["optimizer"]["state"]:
            for field in a['optimizer']['state'][key]:
                torch.testing.assert_close(a["optimizer"]["state"][key][field],
                                           b["optimizer"]["state"][key][field],rtol=0,atol=0)
        assert a["reader"]==b["reader"]
        for key in a['optimization']['swa']:
            torch.testing.assert_close(a['optimization']['swa'][key],b['optimization']['swa'][key],rtol=0,atol=0)
        for key in a['optimization']['slow']:
            torch.testing.assert_close(a['optimization']['slow'][key],b['optimization']['slow'][key],rtol=0,atol=0)
        assert a['optimization']['norms']==b['optimization']['norms']
        torch.testing.assert_close(a["rng"]["torch"],b["rng"]["torch"],rtol=0,atol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize('kind', ['sgd', 'adamw'])
def test_optimizer_checkpoint_d4_and_swa_export(tmp_path,pipeline,gpu_config,kind):
    root,_=pipeline
    c=copy.deepcopy(gpu_config);c['optimizer']['kind']=kind
    plan=load_json(root/'.internal/iterations'/'000001'/'plan.json')
    plan['snapshot_id']=load_json(root/'.internal/iterations'/'000001'/'status.json')['snapshot_id']
    plan['train_steps']=5
    destinations=[tmp_path/'full',tmp_path/'resumed']
    for destination in destinations:
        shutil.copytree(root/'snapshots'/plan['snapshot_id'],destination/'snapshots'/plan['snapshot_id'])
    bases=[initialize(destination,c,root/plan['input_checkpoint']['path']) for destination in destinations]
    updates=[]
    def log(event,**fields):
        if event=='update':updates.append(fields)
    full,done=train_iteration(destinations[0],c,plan,bases[0],log);assert done
    uninterrupted=[u['symmetry'] for u in updates]
    updates.clear()
    partial,done=train_iteration(destinations[1],c,plan,bases[1],log,lambda:len(updates)>=1)
    assert not done
    resumed,done=train_iteration(destinations[1],c,plan,bases[1],log);assert done
    assert uninterrupted==[u['symmetry'] for u in updates]
    a=load_checkpoint(destinations[0],full,c);b=load_checkpoint(destinations[1],resumed,c)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    for key in a['optimization']['swa']:
        torch.testing.assert_close(a['optimization']['swa'][key],b['optimization']['swa'][key],rtol=0,atol=0)
    for key in a['optimizer']['state']:
        for field in a['optimizer']['state'][key]:
            torch.testing.assert_close(a['optimizer']['state'][key][field],b['optimizer']['state'][key][field],rtol=0,atol=0)
    assert a['optimization']['lookahead_counter']==0 and a['optimization']['swa']['n_averaged']==2
    assert a['optimization']['swa_samples']==c['training']['batch_size']
    for key in a['optimization']['slow']:
        torch.testing.assert_close(a['model'][key],a['optimization']['slow'][key],rtol=0,atol=0)
    assert any(not torch.equal(a['model'][k],v) for k,v in inference_weights(a).items())
    (destinations[0]/'config').mkdir()
    write_native(c,destinations[0]/'config/effective.cfg')
    info=export_model(destinations[0],c,full,BINARY)
    assert info['weights']=='swa' and info['swa_samples']==2
    assert info['verification']['native'] and info['verification']['python_scripted']


@pytest.mark.parametrize('amp',['float16','bfloat16'])
@pytest.mark.parametrize('kind',['sgd','adamw'])
def test_real_amp_updates(tmp_path,pipeline,gpu_config,amp,kind,monkeypatch):
    root,_=pipeline;c=copy.deepcopy(gpu_config);c["training"]["amp"]=amp
    c['optimizer']['kind']=kind
    plan=load_json(root/".internal/iterations"/"000001"/"plan.json");plan["train_steps"]=2
    plan["snapshot_id"]=load_json(root/".internal/iterations"/"000001"/"status.json")["snapshot_id"]
    shutil.copytree(root/"snapshots"/plan["snapshot_id"],tmp_path/"snapshots"/plan["snapshot_id"])
    base=initialize(tmp_path,c)
    if amp=="float16":
        saved=load_checkpoint(tmp_path,base,c)
        model=make_network(c).cuda();model.load_state_dict(saved["model"])
        optimizer=optimizer_for(model,c);optimizer.load_state_dict(saved["optimizer"])
        scaler=torch.amp.GradScaler("cuda",init_scale=2**32)
        restore_rng(saved["rng"])
        optimization=Optimization(model,c,optimizer,saved['optimization'])
        base=commit_checkpoint(tmp_path,c,model,optimizer,scaler,0,0,0,None,base,[],optimization)
        del optimization,model,optimizer,scaler,saved
    forwards=[];events=[]
    original=make_network
    def instrumented(config):
        model=original(config)
        model.register_forward_hook(lambda *a:forwards.append(1))
        return model
    monkeypatch.setattr("etazero.training.make_network",instrumented)
    result,done=train_iteration(tmp_path,c,plan,base,lambda event,**fields:events.append((event,fields)))
    assert done and result["total_steps"]==2
    assert len(forwards)==2
    if amp=="float16":
        assert any(event=="amp_overflow" for event,_ in events)
    checkpoint=load_checkpoint(tmp_path,result,c)
    assert all(torch.isfinite(v).all() for v in checkpoint["model"].values())


@pytest.mark.parametrize("phase",["shuffle","export","after_publish"])
def test_stage_failure_recovery(tmp_path,gpu_config,monkeypatch,phase):
    import etazero.runtime as runtime
    c=copy.deepcopy(gpu_config);c["training"]["replay_ratio"]=1000
    with monkeypatch.context() as patch:
        if phase=="shuffle":
            patch.setattr(runtime,"build_snapshot",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("injected stage failure")))
        elif phase=="export":
            original=runtime.export_model
            def failed_export(root,config,checkpoint,binary):
                if checkpoint["iteration"]:
                    raise RuntimeError("injected stage failure")
                return original(root,config,checkpoint,binary)
            patch.setattr(runtime,"export_model",failed_export)
        else:
            original=runtime.Controller.publish
            def failed_commit(self,model,elapsed):
                original(self,model,elapsed)
                if model["checkpoint"]["iteration"]:
                    raise RuntimeError("injected stage failure")
            patch.setattr(runtime.Controller,"publish",failed_commit)
        with pytest.raises(RuntimeError,match="injected stage failure"):
            run_training(tmp_path,c,BINARY,max_iteration=2)
    shards={p:sha256(p) for p in (tmp_path/"selfplay").rglob("*.npz")}
    plan_hash=sha256(tmp_path/".internal/iterations"/"000001"/"plan.json")
    checkpoint=None
    progress=tmp_path/".internal/iterations"/"000001"/"learner.json"
    if progress.exists():
        checkpoint=load_json(progress)["checkpoint"]
    state=run_training(tmp_path,c,BINARY,resume=True,max_iteration=2)
    assert state["checkpoint"]["total_steps"]==8 and state["target_rows"]==pytest.approx(state["replay_origin_rows"]+0.032)
    assert sha256(tmp_path/".internal/iterations"/"000001"/"plan.json")==plan_hash
    assert_preserved(tmp_path,shards)
    assert list((tmp_path/"selfplay"/"iteration_000002").rglob("*.npz"))
    if checkpoint:
        current=load_json(tmp_path/".internal/iterations"/"000001"/"status.json")["checkpoint"]
        assert (current==checkpoint)==(phase=="after_publish")


def test_forced_process_kill_during_training(tmp_path,gpu_config):
    configs=tmp_path/"configs";shutil.copytree(ROOT/"configs",configs)
    (configs/"smoke_test"/"train.cfg.local").write_text("[training]\ntrain_steps=100\ncheckpoint_every=2\nreplay_ratio=8\n")
    c=load_config(configs/"smoke_test");root=tmp_path/"run"
    env=os.environ.copy();env["PYTHONPATH"]=str(ROOT/"python")
    command=[sys.executable,"-m","etazero","run","--config-dir",str(configs/"smoke_test"),
             "--run-dir",str(root),"--iterations","1"]
    with (tmp_path/"stdout").open("w") as out,(tmp_path/"stderr").open("w") as err:
        process=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
        progress=root/".internal/iterations"/"000001"/"learner.json";deadline=time.monotonic()+60
        try:
            while time.monotonic()<deadline:
                if progress.exists() and load_json(progress)["checkpoint"]["step"]>=2:
                    process.kill();break
                if process.poll() is not None:
                    pytest.fail("Controller exited before the forced-kill point: "+(tmp_path/"stderr").read_text())
                time.sleep(0.005)
            else:
                pytest.fail("Timed out waiting for a committed training checkpoint")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    persisted=load_json(progress)["checkpoint"]
    assert 2<=persisted["step"]<100
    plan_hash=sha256(root/".internal/iterations"/"000001"/"plan.json")
    raw={p:sha256(p) for p in (root/"selfplay").rglob("*.npz")}
    state=run_training(root,c,BINARY,resume=True,max_iteration=1)
    assert state["checkpoint"]["step"]==100 and state["checkpoint"]["total_samples"]==800
    assert state["target_rows"]==state["replay_origin_rows"] and sha256(root/".internal/iterations"/"000001"/"plan.json")==plan_hash
    assert_preserved(root,raw)
    assert not (root/persisted["path"]).exists()


def test_multiple_selfplay_workers(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config);c["devices"]["selfplay"]="cuda:0,cuda:0"
    c["training"]["replay_ratio"]=1000
    root=tmp_path/"first";repeat=tmp_path/"repeat"
    run_training(root,c,BINARY,max_iteration=1)
    run_training(repeat,c,BINARY,max_iteration=1)
    def seeds(path):
        events=[json.loads(line) for line in (path/"logs/events.jsonl").read_text().splitlines()]
        # Compare the same bootstrap attempt. Stochastic row counts may change later backfill attempts.
        starts=[e for e in events if e['event']=='worker_start' and e['iteration']==0]
        first_attempt=min(e['attempt'] for e in starts)
        return {e['worker']['id']:e['worker']['seed'] for e in starts if e['attempt']==first_attempt}
    assert seeds(root)==seeds(repeat) and len(set(seeds(root).values()))==2
    sources=[];games=set()
    for path in (root/"selfplay").rglob("*.npz"):
        a=read_raw(path);m=json.loads(a["metadata"].tobytes());sources.append(m)
        for game_id in a["game_ids"]:
            key=(m["attempt_id"],m["worker_id"],int(game_id))
            assert key not in games;games.add(key)
    assert {m["worker_id"] for m in sources}=={0,1}
    assert all(m["source_id"] for m in sources)


def test_native_selfplay_signal_flushes_complete_games(tmp_path,pipeline,gpu_config):
    root,state=pipeline;output=tmp_path/"selfplay"
    command=[str(BINARY),"selfplay","--config",str(root/"config/effective.cfg"),"--model",str(root/state["model"]["path"]),
             "--model-id",state["model"]["id"],"--device","cuda:0","--games","100",
             "--output",str(output),"--run-id","signal","--attempt-id","a","--config-id","test",
             "--source-id","test","--iteration","1","--worker","0","--seed","1"]
    with (tmp_path/"stdout").open("w") as out,(tmp_path/"stderr").open("w") as err:
        process=subprocess.Popen(command,stdout=out,stderr=err)
        deadline=time.monotonic()+30
        try:
            while time.monotonic()<deadline:
                if list(output.glob("*.npz")):
                    process.send_signal(signal.SIGINT);break
                assert process.poll() is None,(tmp_path/"stderr").read_text()
                time.sleep(0.005)
            else:
                pytest.fail("No complete shard before the signal deadline")
            assert process.wait(timeout=10)==2
        finally:
            if process.poll() is None:
                process.kill();process.wait()
    games=0
    for path in output.glob("*.npz"):
        games+=len(read_raw(path)["game_ids"])
    assert 0<games<100


def test_multiple_servers_fp16_packed_waves_and_prefetch(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config)
    c["selfplay"].update(server_threads=2,inference_precision="float16",game_threads=4,search_threads=3,cache_entries=64)
    c["shuffle"].update(waves=3,temp_dir=str(tmp_path/"scratch"))
    root=tmp_path/"run"
    state=run_training(root,c,BINARY,max_iteration=2)
    assert state["checkpoint"]["total_steps"]==8
    events=[json.loads(line) for line in (root/"logs/events.jsonl").read_text().splitlines()]
    services=[e for e in events if e["event"]=="inference"]
    assert services and all(len(e["rows_by_server"])==2 for e in services)
    assert any(min(e["rows_by_server"])>0 for e in services if e["iteration"]>=2)
    assert all(e["requests"]+e["cache_hits"]==e["submitted"] for e in services)
    evaluation_config=load_evaluation_config(ROOT/'configs/smoke_test',environ={
        'EVAL_SERVER_THREADS':'2','EVAL_INFERENCE_PRECISION':'float16','EVAL_CACHE_ENTRIES':'64'})
    _,evaluation=evaluate(evaluation_config,BINARY,root,size=5,rule="freestyle",moves="0,5,1")
    # evaluate() requests the raw root NN output, then searches that same root.
    # This guarantees a real cache reuse without assuming selfplay hit frequency.
    assert evaluation["result"]["cache_hits"]>=1
    for path in (root/"selfplay").rglob("*.npz"):
        assert read_raw(path)["observations"].shape[1:]==(5,5) # ceil(6*6/8)
    assert not list((tmp_path/"scratch").iterdir())
    scripted=torch.jit.load(str(root/state["model"]["path"])).cuda().eval()
    assert state["model"]["verification"]["inference_precision"]=="float16"
    previous_autocast=torch._C._jit_set_autocast_mode(False)
    try:
        for size in (5,6):
            for rule in ("freestyle","standard","renju"):
                moves=(0,6,1)
                command=[str(BINARY),"infer","--config",str(root/"config/effective.cfg"),"--model",str(root/state["model"]["path"]),
                         "--model-id",state["model"]["id"],"--device","cuda:0","--size",str(size),"--rule",rule,"--moves","0,6,1"]
                native=json.loads(subprocess.run(command,text=True,capture_output=True,check=True).stdout)
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                    p,v=scripted(*(torch.from_numpy(x).unsqueeze(0).cuda() for x in example_inputs(6,size,rule,moves)))
                assert p.dtype==v.dtype==torch.float16
                np.testing.assert_allclose(native["raw_logits"],p[0].cpu(),rtol=3e-3,atol=3e-3)
                np.testing.assert_allclose(native["raw_wdl"],v[0].float().softmax(0).cpu(),rtol=3e-3,atol=3e-3)
    finally:
        torch._C._jit_set_autocast_mode(previous_autocast)


@pytest.mark.parametrize('amp',['off','float16','bfloat16'])
def test_compiled_training_resume(tmp_path,pipeline,gpu_config,amp):
    root,_=pipeline;c=copy.deepcopy(gpu_config);c['training'].update(compile=True,amp=amp)
    plan=load_json(root/'.internal/iterations'/'000001'/'plan.json')
    plan['snapshot_id']=load_json(root/'.internal/iterations'/'000001'/'status.json')['snapshot_id']
    destinations=[tmp_path/'full',tmp_path/'resumed']
    bases=[]
    for destination in destinations:
        shutil.copytree(root/'snapshots'/plan['snapshot_id'],destination/'snapshots'/plan['snapshot_id'])
        bases.append(initialize(destination,c,root/plan['input_checkpoint']['path']))
    logs=[]
    def log(event,**fields):
        if event=='update':logs.append(fields)
    full,done=train_iteration(destinations[0],c,plan,bases[0],log);assert done
    logs.clear()
    partial,done=train_iteration(destinations[1],c,plan,bases[1],log,lambda:len(logs)>=1)
    assert not done and partial['step']==1
    resumed,done=train_iteration(destinations[1],c,plan,bases[1],log);assert done
    a=load_checkpoint(destinations[0],full,c);b=load_checkpoint(destinations[1],resumed,c)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert a['reader']==b['reader'] and a['scaler']==b['scaler']


@pytest.mark.parametrize('amp',['off','float16'])
def test_inference_normalization_preserves_convolution_precision(gpu_config,amp):
    c=copy.deepcopy(gpu_config);c['network'].update(canvas=15,channels=64,blocks=4,value_hidden=64)
    with torch.random.fork_rng():
        torch.manual_seed(53)
        model=make_network(c).cuda().eval()
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,MaskedBatchNorm):
                    layer.running_mean.uniform_(-0.5,0.5);layer.running_var.uniform_(0.1,2)
                    layer.weight.uniform_(0.5,1.5);layer.bias.uniform_(-0.2,0.2)
        inference=torch.jit.script(inference_network(model))
        probes=[example_inputs(15,size,rule,(0,15,1)) for size in (9,15) for rule in ('freestyle','standard','renju')]
        obs=tuple(torch.from_numpy(np.stack(x)).cuda() for x in zip(*probes))
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=True),torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=amp=='float16'):
            for inputs in (obs,tuple(x[:1] for x in obs)):
                expected=model(*inputs)
                # Include the optimized graph selected after JIT profiling.
                for _ in range(6):
                    for reference,actual in zip(expected,inference(*inputs)):
                        rtol,atol=(3e-3,3e-3) if amp=='float16' else (0,0)
                        torch.testing.assert_close(actual,reference,rtol=rtol,atol=atol)


@pytest.mark.parametrize('precision',['float32','float16'])
def test_export_tf32_with_nontrivial_normalization(tmp_path,gpu_config,precision):
    c=copy.deepcopy(gpu_config)
    c['network'].update(canvas=15,channels=64,blocks=4,value_hidden=64)
    c['environment'].update(sizes='15',size_weights='1')
    c['selfplay']['inference_precision']=precision
    (tmp_path/'models').mkdir();(tmp_path/'checkpoints').mkdir()
    write_native(c,tmp_path/'config/effective.cfg')
    with torch.random.fork_rng(),torch.backends.cudnn.flags(enabled=True,allow_tf32=True):
        torch.manual_seed(53)
        model=make_network(c).cuda().eval()
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,MaskedBatchNorm):
                    layer.running_mean.uniform_(-2,2);layer.running_var.uniform_(0.03,2)
                    layer.weight.uniform_(0.5,1.5);layer.bias.uniform_(-0.2,0.2)
        optimizer=optimizer_for(model,c)
        checkpoint=commit_checkpoint(tmp_path,c,model,optimizer,
                                     torch.amp.GradScaler('cuda',enabled=False),0,0,0,None,None,[],
                                     Optimization(model,c,optimizer))
        info=export_model(tmp_path,c,checkpoint,BINARY)
    assert info['verification']['python_scripted'] and info['verification']['native']
    assert info['verification']['normalization']=='precomputed_inv_std'
    assert info['verification']['inference_precision']==precision
    assert sha256(tmp_path/checkpoint['path'])==checkpoint['sha256']


def test_persistent_native_protocol_and_signal(tmp_path,pipeline,gpu_config):
    from etazero.native import NativeWorker
    root,state=pipeline;model=state['model'];config=root/'config/effective.cfg'
    command=[str(BINARY),'worker','--config',str(config),'--device','cuda:0',
             '--run-id','protocol','--config-id','test','--source-id','test','--worker','0']
    service=NativeWorker(command,tmp_path/'stderr')
    def wait_result():
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            result=service.take()
            if result is not None:return result
            time.sleep(0.005)
        pytest.fail('Native worker response timeout')
    try:
        output=tmp_path/'带空格 data\nfirst'
        service.submit([str(root/model['path']),model['id'],1,str(output),'first',1,5,'network'],tmp_path/'first.jsonl')
        assert wait_result()['code']==0
        assert sum(len(read_raw(p)['game_ids']) for p in output.glob('*.npz'))==1
        service.release()
        output=tmp_path/'second'
        service.submit([str(root/model['path']),model['id'],100,str(output),'second',2,6,'network'],tmp_path/'second.jsonl')
        deadline=time.monotonic()+30
        while not list(output.glob('*.npz')):
            assert time.monotonic()<deadline and service.process.poll() is None
            time.sleep(0.005)
        service.process.send_signal(signal.SIGINT)
        assert wait_result()['code']==2
        assert list(output.glob('*.npz'))
        for path in output.glob('*.npz'):read_raw(path)
    finally:
        service.close()
    assert service.process.returncode==2
    idle=NativeWorker(command,tmp_path/'idle.stderr');idle.close()
    assert idle.process.returncode==0


def test_snapshot_retention_and_restore(tmp_path,gpu_config):
    from etazero.reader import BatchReader
    c=copy.deepcopy(gpu_config);c['shuffle']['snapshot_keep']=1;c['training']['replay_ratio']=1000
    state=run_training(tmp_path,c,BINARY,max_iteration=3)
    statuses=[load_json(tmp_path/'.internal/iterations'/f'{i:06d}'/'status.json') for i in range(1,4)]
    snapshots=[tmp_path/'snapshots'/s['snapshot_id'] for s in statuses]
    assert [p.joinpath('data').is_dir() for p in snapshots]==[False,False,True]
    raw={p:sha256(p) for p in (tmp_path/'selfplay').rglob('*.npz')}
    reader=BatchReader(snapshots[0],8,1,1)
    try:
        assert len(reader.next()['value'])==8
    finally:reader.close()
    assert raw=={p:sha256(p) for p in raw}
    assert state['checkpoint']['total_steps']==12
    resumed=run_training(tmp_path,c,BINARY,resume=True,max_iteration=4)
    assert resumed['checkpoint']['total_steps']==16


def test_bootstrap_backfill_balance_and_replay_reset(tmp_path,gpu_config,monkeypatch):
    import etazero.runtime as runtime
    from etazero.data import Catalog, metadata, training_view
    c=copy.deepcopy(gpu_config)
    c['selfplay']['bootstrap_games']=2
    c['replay']['min_rows']=150
    c['training']['replay_ratio']=8
    c['opening']['probability']=1
    root=tmp_path/'run'
    first_launch=runtime.Controller.launch
    def partial_bootstrap(self,plan,games):
        if plan['iteration']==0:
            first_launch(self,plan,1)
            raise RuntimeError('injected partial bootstrap')
        first_launch(self,plan,games)
    with monkeypatch.context() as patch:
        patch.setattr(runtime.Controller,'launch',partial_bootstrap)
        with pytest.raises(RuntimeError,match='partial bootstrap'):
            run_training(root,c,BINARY,max_iteration=3)
    bootstrap_before={p:sha256(p) for p in (root/'selfplay'/'iteration_000000').rglob('*.npz')}
    assert sum(len(read_raw(p)['game_ids']) for p in bootstrap_before)==1
    # Fail immediately after the durable reset, then verify it is never reset again.
    commit=runtime.Controller.state
    def fail_after_reset(self,state):
        commit(self,state)
        if state['iteration']==2:
            raise RuntimeError('injected after replay reset')
    with monkeypatch.context() as patch:
        patch.setattr(runtime.Controller,'state',fail_after_reset)
        with pytest.raises(RuntimeError,match='after replay reset'):
            run_training(root,c,BINARY,resume=True,max_iteration=3)
    origin=load_json(root/'.internal/state.json')['replay_origin_rows']
    assert origin>=150
    state=run_training(root,c,BINARY,resume=True,max_iteration=3)
    assert state['replay_origin_rows']==origin and state['target_rows']==origin+8
    assert state['checkpoint']['total_steps']==12
    bootstrap=[];balanced=0
    for path in (root/'selfplay').rglob('*.npz'):
        a=read_raw(path);m=metadata(a)
        if m['iteration_id']<=1:
            assert m['model_id'].startswith('random:') and a['opening_moves'].sum()==0
            if m['iteration_id']==0:bootstrap.extend(a['game_ids'])
        else:
            assert m['model_id'].startswith('iteration_')
            assert (a['opening_status']==1).all() and (a['opening_moves']>0).all()
            assert len(training_view(a)['value'])==m['rows']<m['plies']
            balanced+=len(a['game_ids'])
    assert len(bootstrap)==2 and balanced>=2
    assert_preserved(root,bootstrap_before)
    assert all(not p.exists() for p in bootstrap_before)
    catalog=Catalog(root,state['run_id'],runtime.fingerprint(c))
    try:
        assert catalog.iteration_counts(0)[1]==2
        assert catalog.iteration_counts(2)[1]>0 and catalog.iteration_counts(3)[1]>0
    finally:
        catalog.close()
    plans={i:load_json(root/'.internal/iterations'/f'{i:06d}'/'plan.json') for i in range(4)}
    assert plans[0]['train_steps']==0 and plans[1]['target_rows']==150
    assert plans[2]['target_rows']==origin+4 and plans[3]['target_rows']==origin+8
    hashes={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')}
    assert run_training(root,c,BINARY,max_iteration=3)==state
    assert hashes=={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')}


def test_autoexp_real_cuda_shared_weights_queue_resume_and_plots(tmp_path):
    from etazero.experiment import experiment_plan, run_experiment
    from etazero.plotting import run_history
    from etazero.schema import CONTRACT_ID
    umbrella = tmp_path/'umbrella'; umbrella.mkdir()
    (umbrella/'exp.cfg').write_text('[experiment]\nmax_iteration = 2\nmax_seconds = 0\narm_gpus = 0\nshared_init = true\n')
    for name in ('a', 'b'):
        arm = umbrella/name; arm.mkdir()
        (arm/'run.cfg').write_text(f'[run]\nextends = smoke_test\nrun_dir = {tmp_path/name}\n')
    plan = experiment_plan(umbrella, environ={}, work_dir=tmp_path/'scheduler')
    assert run_experiment(plan, BINARY) == 0
    records = [load_json(tmp_path/name/'.internal/run.json') for name in ('a', 'b')]
    assert records[0]['id'] != records[1]['id']
    assert records[0]['weights_initialization'] == records[1]['weights_initialization']
    checkpoints = []
    hashes = {}
    for name in ('a', 'b'):
        root = tmp_path/name
        state = load_json(root/'.internal/state.json')
        assert state['iteration'] == 3 and state['checkpoint']['total_steps'] == 8
        initial = next(p for p in (root/'checkpoints').glob('iteration_000000*.pt'))
        checkpoints.append(torch.load(initial, map_location='cpu', weights_only=False))
        assert (root/'training.png').stat().st_size > 10000
        assert (root/'logs/performance.png').stat().st_size > 10000
        history = run_history(root)
        assert [r['iteration'] for r in history] == [0, 1, 2]
        assert history[0]['steps'] == 0 and history[2]['steps'] == 4
        assert history[2]['avg_game_length'] >= history[2]['avg_rows_per_game']
        hashes.update({p: sha256(p) for p in (root/'selfplay').rglob('*.npz')})
    assert all(torch.equal(value, checkpoints[1]['model'][key]) for key, value in checkpoints[0]['model'].items())
    assert all(c['contract'] == CONTRACT_ID and c['total_steps'] == 0 for c in checkpoints)
    assert run_experiment(plan, BINARY) == 0
    assert all(sha256(path) == digest for path, digest in hashes.items())
    # Increase only the cumulative invocation budget, keeping effective arm config unchanged.
    plan['settings']['max_iteration'] = 3
    assert run_experiment(plan, BINARY) == 0
    for name in ('a', 'b'):
        state = load_json(tmp_path/name/'.internal/state.json')
        assert state['iteration'] == 4 and state['checkpoint']['total_steps'] == 12
    assert all(sha256(path) == digest for path, digest in hashes.items())


def test_autoexp_signal_and_real_cuda_learner_restore(tmp_path):
    umbrella = tmp_path/'umbrella'; arm = umbrella/'a'; arm.mkdir(parents=True)
    root = tmp_path/'run'; work = tmp_path/'scheduler'
    (umbrella/'exp.cfg').write_text('[experiment]\nmax_iteration = 1\nmax_seconds = 0\narm_gpus = 0\nshared_init = false\n')
    (arm/'run.cfg').write_text(f'[run]\nextends = smoke_test\nrun_dir = {root}\n')
    (arm/'train.cfg').write_text('[training]\ntrain_steps = 200\ncheckpoint_every = 20\n')
    command = [sys.executable, '-m', 'etazero.experiment', '--config-dir', str(umbrella),
               '--work-dir', str(work), '--binary', str(BINARY)]
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'python')}
    log = tmp_path/'runner.log'
    with log.open('w') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+90
        try:
            progress = root/'.internal/iterations/000001/learner.json'
            while time.monotonic() < deadline:
                if progress.exists():
                    process.send_signal(signal.SIGTERM); break
                if process.poll() is not None:
                    pytest.fail('Scheduler exited before learner checkpoint: '+log.read_text())
                time.sleep(.01)
            else:
                pytest.fail('Timed out waiting for learner checkpoint')
            assert process.wait(timeout=30) == 130, log.read_text()
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=30)
    old = load_json(progress)['checkpoint']
    old_hash = sha256(root/old['path'])
    assert old['total_steps'] >= 20
    assert load_json(work/'.internal/status.json')['arms']['a']['status'] == 'interrupted'
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    state = load_json(root/'.internal/state.json')
    assert state['iteration'] == 2 and state['checkpoint']['total_steps'] == 200
    assert not (root/old['path']).exists()
    assert_preserved(root,{root/old['path']:old_hash})
    from etazero.plotting import run_history
    assert run_history(root)[-1]['steps'] == 200
    assert (root/'training.png').is_file()


def test_autoexp_cumulative_time_budget_and_iteration_extension(tmp_path):
    from etazero.experiment import experiment_plan, run_experiment
    umbrella = tmp_path/'umbrella'; arm = umbrella/'a'; arm.mkdir(parents=True)
    root = tmp_path/'run'
    (umbrella/'exp.cfg').write_text('[experiment]\nmax_iteration = 0\nmax_seconds = 1\narm_gpus = 0\nshared_init = false\n')
    (arm/'run.cfg').write_text(f'[run]\nextends = smoke_test\nrun_dir = {root}\n')
    plan = experiment_plan(umbrella, environ={}, work_dir=tmp_path/'scheduler')
    assert run_experiment(plan, BINARY) == 0
    events = [json.loads(line) for line in (root/'logs/events.jsonl').read_text().splitlines()]
    assert load_json(root/'.internal/state.json')['elapsed_seconds'] >= 1
    info = load_json(root/'.internal/run.json')
    checksum = sha256(root/'.internal/state.json')
    assert run_experiment(plan, BINARY) == 0
    assert sha256(root/'.internal/state.json') == checksum
    plan['settings'].update(max_seconds=0, max_iteration=1)
    assert run_experiment(plan, BINARY) == 0
    state = load_json(root/'.internal/state.json')
    assert state['iteration'] == 2 and state['checkpoint']['total_steps'] == 4
    assert load_json(root/'.internal/run.json') == info


def test_eval_100_visits_and_arena_resume_elo(tmp_path,pipeline,gpu_config,monkeypatch):
    from etazero.arena import single_match, PairStore
    from etazero.elo import write_ratings
    root,state=pipeline
    ec=load_evaluation_config(ROOT/'configs/smoke_test')
    _,result=evaluate(ec,BINARY,root,size=5,rule='renju',moves='0,6')
    assert result['result']['root_visits']==100
    assert result['result']['simulations']==99 and sum(result['result']['visits'])==99
    assert 0<=result['result']['action']<25 and len(result['result']['policy'])==25
    mc=load_evaluation_config(ROOT/'configs/smoke_test',True)
    a=root/state['model']['path']
    b=root/load_json(root/'logs/iterations/000001.json')['model']['path']
    output=tmp_path/'match'
    original=PairStore.accept
    interrupted=False
    def stop_after_game(self,event):
        nonlocal interrupted
        original(self,event)
        if event['type']=='game' and not interrupted:
            interrupted=True
            raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr(PairStore,'accept',stop_after_game)
        with pytest.raises(KeyboardInterrupt):single_match(mc,BINARY,a,b,output)
    before={p:sha256(p) for p in (output/'pairs').rglob('*.json')}
    assert before
    _,games=single_match(mc,BINARY,a,b,output)
    assert len(games)==4 and all(set(g['root_visits'])=={100} for g in games)
    assert all(sha256(p)==digest for p,digest in before.items())
    preserved={p:sha256(p) for p in (output/'pairs').rglob('*.json')}
    _,extended=single_match(mc,BINARY,a,b,output,games=8)
    assert len(extended)==8 and all(sha256(p)==v for p,v in preserved.items())
    ratings=write_ratings(output,10)
    assert ratings['games']==8
    for ext in ('json','csv','png','svg'):assert (output/f'elo.{ext}').stat().st_size>100
    # CLI discovery uses committed history and preserves result identity on restart.
    out=tmp_path/'elo_history'
    command=['bash',str(ROOT/'scripts/run.sh'),'arena','--data',str(root),'--output',str(out),
             '--config-dir',str(ROOT/'configs/smoke_test'),'--games','4','--bootstrap-samples','10']
    first=subprocess.run(command,env={**os.environ,'PYTHONPATH':str(ROOT/'python')},text=True,capture_output=True,timeout=120)
    assert first.returncode==0,first.stdout+first.stderr
    hashes={p:sha256(p) for p in (out/'pairs').rglob('*.json')}
    second=subprocess.run(command,text=True,capture_output=True,timeout=120)
    assert second.returncode==0,second.stdout+second.stderr
    assert all(sha256(p)==digest for p,digest in hashes.items())


def test_mixed_rule_openings_and_training_feature_dropout(tmp_path,pipeline,gpu_config):
    from etazero.data import training_view
    from etazero.schema import unpack_observations
    root,state=pipeline
    c=copy.deepcopy(gpu_config)
    c['selfplay'].update(game_threads=1,search_threads=1,server_threads=1)
    c['opening']['probability']=1
    datasets=[]
    for dropout in (0.,.5,1.):
        c['selfplay']['forbidden_feature_dropout_prob']=dropout
        cfg=tmp_path/f'{dropout}.cfg';write_native(c,cfg)
        output=tmp_path/f'rows_{dropout}'
        command=[str(BINARY),'selfplay','--config',str(cfg),'--model',str(root/state['model']['path']),
                 '--model-id',state['model']['id'],'--device','cuda:0','--games','18','--output',str(output),
                 '--run-id','test','--attempt-id','features','--config-id','test','--source-id','test',
                 '--iteration','2','--worker','0','--seed','123']
        subprocess.run(command,text=True,capture_output=True,check=True,timeout=180)
        records=[read_raw(p) for p in sorted(output.glob('*.npz'))]
        assert set(np.concatenate([a['rules'] for a in records]))=={0,1,2}
        assert set(np.concatenate([a['sizes'] for a in records]))=={5,6}
        flags=[]
        for a in records:
            assert (a['opening_status']==1).all() and (a['opening_moves']>0).all()
            view=training_view(a)
            flags.extend(view['globals'][view['globals'][:,1]==1,3])
            if dropout==1:
                obs=unpack_observations(view['obs'],6)
                assert not obs[:,3:5].any()
        assert flags
        if dropout==.5: assert set(flags)=={0,1}
        else: assert set(flags)=={1-dropout}
        datasets.append(records)
    # With one search/game thread and identical model/seed, writing augmented rows
    # must not alter the openings, search targets, sampled sizes/rules or results.
    for other in datasets[1:]:
        assert len(other)==len(datasets[0])
        for a,b in zip(datasets[0],other):
            for key in a.keys()-{'observations','globals','metadata'}:
                np.testing.assert_array_equal(a[key],b[key])


@pytest.mark.parametrize('lookback',[1,3])
def test_reduce_visits_pcr_native_training_and_resume(tmp_path,gpu_config,lookback):
    c=copy.deepcopy(gpu_config)
    c['opening'].update(probability=0,policy_init=False,policy_after=False,policy_on_failure=False)
    c['selfplay'].update(game_threads=1,search_threads=1,policy_surprise_data_weight=0,value_surprise_data_weight=0)
    c['search'].update(simulations=40,cheap_search_probability=0.5,cheap_search_visits=4,
                       reduce_visits=True,reduce_visits_threshold=0,reduce_visits_threshold_lookback=lookback,
                       reduced_visits_min=2,reduced_visits_weight=0.1)
    root=tmp_path/'run'
    state=run_training(root,c,BINARY,max_iteration=2)
    assert state['checkpoint']['total_steps']==8 and state['checkpoint']['total_samples']==64
    files={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')}
    assert run_training(root,c,BINARY,resume=True,max_iteration=2)==state
    state=run_training(root,c,BINARY,resume=True,max_iteration=3)
    assert state['checkpoint']['total_steps']==12 and state['checkpoint']['total_samples']==96
    assert all(sha256(p)==digest for p,digest in files.items())
    cheap_count=reduced_count=reduced_after_cheap=0
    for path in sorted((root/'selfplay').rglob('*.npz')):
        a=read_raw(path)
        for game,(lo,hi) in enumerate(zip(a['game_offsets'][:-1],a['game_offsets'][1:])):
            history=[];previous_cheap=False
            ol=a['observation_offsets'][game]
            assert a['opening_moves'][game]==0
            for index in range(lo,hi):
                cheap=bool(a['cheap_search'][index])
                if cheap:
                    cheap_count+=1
                    assert a['target_weights'][index]==0
                    assert a['row_repeats'][index]==0
                    assert 0 <= a['simulations'][index] <= 3
                    assert a['visits'][index].sum()+1 >= 4
                else:
                    # Independently reconstruct certainty from stored completed WDL,
                    # rather than importing or translating the C++ limits helper.
                    certain=0
                    if len(history)>=lookback:
                        window=np.asarray(history[-lookback:])
                        if (window>0).all() or (window<0).all():
                            certain=float(np.min(np.abs(window)))
                    reduction=certain**2
                    expected_weight=1-0.9*reduction
                    unrounded_visits=41-39*reduction
                    # NPZ WDL is float32; allow only its rounding uncertainty at
                    # half-integer boundaries, not an extra search simulation.
                    root_visits=int(a['visits'][index].sum())+1
                    assert abs(root_visits-unrounded_visits)<=0.5002
                    assert a['simulations'][index]==root_visits-1
                    assert a['target_weights'][index]==pytest.approx(expected_weight,abs=2e-6)
                    if root_visits<41:
                        reduced_count+=1
                        reduced_after_cheap+=int(previous_cheap)
                player=int(a['players'][ol+index-lo])
                wdl=a['search_wdl'][index].astype(np.float64)
                history.append(player*(wdl[0]-wdl[2]))
                previous_cheap=cheap
    assert cheap_count>0 and reduced_count>0
    if lookback==1:
        assert reduced_after_cheap>0
