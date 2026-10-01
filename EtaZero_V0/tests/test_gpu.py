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
from etazero.config import ROOT, load_config
from etazero.data import read_raw
from etazero.evaluation import evaluate
from etazero.export import example_observation
from etazero.network import make_network
from etazero.runtime import run_training
from etazero.storage import load_json, sha256
from etazero.training import initialize, load_checkpoint, train_cycle, optimizer_for, commit_checkpoint, restore_rng

pytestmark=pytest.mark.skipif(os.environ.get("ETAZERO_GPU_TESTS")!="1",reason="Enable ETAZERO_GPU_TESTS=1 with host CUDA access")
BINARY=ROOT/"build"/"etazero"


@pytest.fixture(scope="module")
def gpu_config():
    assert torch.cuda.is_available(),"GPU tests require actual host CUDA access"
    torch.set_num_threads(1)
    return load_config(ROOT/"configs"/"minimal_test")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory,gpu_config):
    root=tmp_path_factory.mktemp("pipeline")
    state=run_training(root,gpu_config,BINARY,max_cycles=2)
    return root,state


def test_gpu_pipeline_and_idempotent_completed_resume(pipeline,gpu_config):
    root,state=pipeline
    assert state["checkpoint"]["total_steps"]==8
    assert state["checkpoint"]["total_samples"]==64
    shards={p:sha256(p) for p in (root/"data").rglob("*.npz")}
    assert shards
    for path in shards:
        read_raw(path)
    events=[json.loads(line) for line in (root/"events.jsonl").read_text().splitlines()]
    assert max(e["max_batch"] for e in events if e["event"]=="inference")>1
    before=len(list((root/"checkpoints").glob("*.pt")))
    resumed=run_training(root,gpu_config,BINARY,resume=True,max_cycles=2)
    assert state==resumed and len(list((root/"checkpoints").glob("*.pt")))==before
    assert shards=={p:sha256(p) for p in shards}


def test_native_parity_all_sizes_rules_and_match(pipeline,gpu_config):
    root,state=pipeline
    checkpoint=load_checkpoint(root,state["checkpoint"],gpu_config)
    model=make_network(gpu_config).cuda().eval();model.load_state_dict(checkpoint["model"])
    canvas=gpu_config["network"]["canvas"]
    for size in (5,6):
        for rule in ("freestyle","standard","renju"):
            command=[str(BINARY),"infer","--config",str(root/"effective.cfg"),"--model",str(root/state["model"]["path"]),
                     "--model-id",state["model"]["id"],"--device","cuda:0","--size",str(size),"--rule",rule,
                     "--moves",f"0,{canvas},1"]
            native=json.loads(subprocess.run(command,text=True,capture_output=True,check=True).stdout)
            with torch.inference_mode():
                p,v=model(torch.from_numpy(example_observation(canvas,size,rule,(0,canvas,1))).unsqueeze(0).cuda())
            np.testing.assert_allclose(native["raw_logits"],p[0].cpu(),rtol=2e-4,atol=2e-5)
            np.testing.assert_allclose(native["raw_value"],v[0].cpu(),rtol=2e-4,atol=2e-5)
    model_b=root/load_json(root/"cycles"/"000001"/"plan.json")["input_model"]["path"]
    _,result=evaluate(gpu_config,BINARY,root,model_b=model_b,size=5,rule="renju",games=4)
    assert result["result"]["complete"] and len(result["result"]["outcomes_a"])==4
    assert set(result["result"]["outcomes_a"])<={-1,0,1}
    _,result=evaluate(gpu_config,BINARY,root,size=5,rule="freestyle",moves="0,6,1,7,2,8,3,9,4")
    assert result["result"]["terminal"] and result["result"]["value"]==-1


def test_learner_resume_matches_uninterrupted_updates(tmp_path,pipeline,gpu_config):
    root,_=pipeline
    plan=load_json(root/"cycles"/"000001"/"plan.json")
    status=load_json(root/"cycles"/"000001"/"status.json")
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
        full,done=train_cycle(dirs[0],gpu_config,plan,base,log)
        assert done
        logs.clear()
        partial,done=train_cycle(dirs[1],gpu_config,plan,base,log,lambda:len(logs)>=2)
        assert not done and partial["step"]==2
        resumed,done=train_cycle(dirs[1],gpu_config,plan,base,log)
        assert done and resumed["total_steps"]==4
        a=load_checkpoint(dirs[0],full,gpu_config);b=load_checkpoint(dirs[1],resumed,gpu_config)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key],b["model"][key],rtol=0,atol=0)
        for key in a["optimizer"]["state"]:
            torch.testing.assert_close(a["optimizer"]["state"][key]["momentum_buffer"],
                                       b["optimizer"]["state"][key]["momentum_buffer"],rtol=0,atol=0)
        assert a["reader"]==b["reader"]
        torch.testing.assert_close(a["rng"]["torch"],b["rng"]["torch"],rtol=0,atol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("amp",["float16","bfloat16"])
def test_real_amp_updates(tmp_path,pipeline,gpu_config,amp,monkeypatch):
    root,_=pipeline;c=copy.deepcopy(gpu_config);c["training"]["amp"]=amp
    plan=load_json(root/"cycles"/"000001"/"plan.json");plan["train_steps"]=2
    plan["snapshot_id"]=load_json(root/"cycles"/"000001"/"status.json")["snapshot_id"]
    shutil.copytree(root/"snapshots"/plan["snapshot_id"],tmp_path/"snapshots"/plan["snapshot_id"])
    base=initialize(tmp_path,c)
    if amp=="float16":
        saved=load_checkpoint(tmp_path,base,c)
        model=make_network(c).cuda();model.load_state_dict(saved["model"])
        optimizer=optimizer_for(model,c);optimizer.load_state_dict(saved["optimizer"])
        scaler=torch.amp.GradScaler("cuda",init_scale=2**32)
        restore_rng(saved["rng"])
        base=commit_checkpoint(tmp_path,c,model,optimizer,scaler,0,0,0,None,base,[])
        del model,optimizer,scaler,saved
    forwards=[];events=[]
    original=make_network
    def instrumented(config):
        model=original(config)
        model.register_forward_hook(lambda *a:forwards.append(1))
        return model
    monkeypatch.setattr("etazero.training.make_network",instrumented)
    result,done=train_cycle(tmp_path,c,plan,base,lambda event,**fields:events.append((event,fields)))
    assert done and result["total_steps"]==2
    assert len(forwards)==2
    if amp=="float16":
        assert any(event=="amp_overflow" for event,_ in events)
    checkpoint=load_checkpoint(tmp_path,result,c)
    assert all(torch.isfinite(v).all() for v in checkpoint["model"].values())


@pytest.mark.parametrize("phase",["shuffle","export","after_publish"])
def test_stage_failure_recovery(tmp_path,gpu_config,monkeypatch,phase):
    import etazero.runtime as runtime
    c=copy.deepcopy(gpu_config);c["replay"]["replay_ratio"]=1000
    with monkeypatch.context() as patch:
        if phase=="shuffle":
            patch.setattr(runtime,"build_snapshot",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("injected stage failure")))
        elif phase=="export":
            original=runtime.export_model
            def failed_export(root,config,checkpoint,binary):
                if checkpoint["cycle"]:
                    raise RuntimeError("injected stage failure")
                return original(root,config,checkpoint,binary)
            patch.setattr(runtime,"export_model",failed_export)
        else:
            original=runtime.Controller.publish
            def failed_commit(self,model):
                original(self,model)
                if model["checkpoint"]["cycle"]:
                    raise RuntimeError("injected stage failure")
            patch.setattr(runtime.Controller,"publish",failed_commit)
        with pytest.raises(RuntimeError,match="injected stage failure"):
            run_training(tmp_path,c,BINARY,max_cycles=2)
    shards={p:sha256(p) for p in (tmp_path/"data").rglob("*.npz")}
    plan_hash=sha256(tmp_path/"cycles"/"000001"/"plan.json")
    checkpoint=None
    progress=tmp_path/"cycles"/"000001"/"learner.json"
    if progress.exists():
        checkpoint=load_json(progress)["checkpoint"]
    state=run_training(tmp_path,c,BINARY,resume=True,max_cycles=2)
    assert state["checkpoint"]["total_steps"]==8 and state["target_rows"]==pytest.approx(0.064)
    assert sha256(tmp_path/"cycles"/"000001"/"plan.json")==plan_hash
    assert shards=={p:sha256(p) for p in (tmp_path/"data").rglob("*.npz")} # cycle 2 has zero sample deficit
    if checkpoint:
        assert load_json(tmp_path/"cycles"/"000001"/"status.json")["checkpoint"]==checkpoint


def test_forced_process_kill_during_training(tmp_path,gpu_config):
    configs=tmp_path/"configs";shutil.copytree(ROOT/"configs",configs)
    (configs/"minimal_test"/"train.cfg.local").write_text("[training]\ntrain_steps=100\ncheckpoint_every=2\n[replay]\nreplay_ratio=8\n")
    c=load_config(configs/"minimal_test");root=tmp_path/"run"
    env=os.environ.copy();env["PYTHONPATH"]=str(ROOT/"python")
    command=[sys.executable,"-m","etazero","run","--config-dir",str(configs/"minimal_test"),
             "--run-dir",str(root),"--cycles","1"]
    with (tmp_path/"stdout").open("w") as out,(tmp_path/"stderr").open("w") as err:
        process=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
        progress=root/"cycles"/"000001"/"learner.json";deadline=time.monotonic()+60
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
    plan_hash=sha256(root/"cycles"/"000001"/"plan.json")
    raw={p:sha256(p) for p in (root/"data").rglob("*.npz")}
    state=run_training(root,c,BINARY,resume=True,max_cycles=1)
    assert state["checkpoint"]["step"]==100 and state["checkpoint"]["total_samples"]==800
    assert state["target_rows"]==100 and sha256(root/"cycles"/"000001"/"plan.json")==plan_hash
    assert raw=={p:sha256(p) for p in (root/"data").rglob("*.npz")}


def test_multiple_selfplay_workers(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config);c["devices"]["selfplay"]="cuda:0,cuda:0"
    c["replay"]["replay_ratio"]=1000
    root=tmp_path/"first";repeat=tmp_path/"repeat"
    run_training(root,c,BINARY,max_cycles=1)
    run_training(repeat,c,BINARY,max_cycles=1)
    def seeds(path):
        events=[json.loads(line) for line in (path/"events.jsonl").read_text().splitlines()]
        return {e["worker"]["id"]:e["worker"]["seed"] for e in events if e["event"]=="worker_start"}
    assert seeds(root)==seeds(repeat) and len(set(seeds(root).values()))==2
    sources=[];games=set()
    for path in (root/"data").rglob("*.npz"):
        a=read_raw(path);m=json.loads(a["metadata"].tobytes());sources.append(m)
        for game_id in a["game_ids"]:
            key=(m["attempt_id"],m["worker_id"],int(game_id))
            assert key not in games;games.add(key)
    assert {m["worker_id"] for m in sources}=={0,1}
    assert all(m["source_id"] for m in sources)


def test_native_selfplay_signal_flushes_complete_games(tmp_path,pipeline,gpu_config):
    root,state=pipeline;output=tmp_path/"data"
    command=[str(BINARY),"selfplay","--config",str(root/"effective.cfg"),"--model",str(root/state["model"]["path"]),
             "--model-id",state["model"]["id"],"--device","cuda:0","--games","100",
             "--output",str(output),"--run-id","signal","--attempt-id","a","--config-id","test",
             "--source-id","test","--cycle","1","--worker","0","--seed","1"]
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
