from collections import Counter
import copy
import json
import math
from pathlib import Path
import shutil
import numpy as np
import pytest
import torch
from etazero.config import ROOT, load_config, validate
from etazero.data import read_raw, training_view, validate_raw, Catalog
from etazero.export import example_observation
from etazero.network import MaskedBatchNorm, make_network, losses
from etazero.reader import BatchReader
from etazero.schema import CONTRACT_ID, RAW_DTYPES
from etazero.shuffle import build_snapshot, desired_window
from etazero.storage import save_npz, sha256


@pytest.fixture
def config():
    return load_config(ROOT/"configs"/"minimal_test")


def winning_record(rule=0):
    canvas,size=6,5
    actions=[0,6,1,7,2,8,3,9,4]
    t=len(actions)
    a={k:np.zeros(0,dtype=dtype) for k,dtype in RAW_DTYPES.items()}
    a["observations"]=np.stack([example_observation(canvas,size,("freestyle","standard","renju")[rule],actions[:i])
                                 for i in range(t+1)]).astype(np.uint8)
    a["players"]=np.array([1,-1]*5,np.int8)
    a["actions"]=np.array(actions,np.int32)
    a["visits"]=np.zeros((t,canvas*canvas),np.int64);a["visits"][np.arange(t),actions]=1
    a["policies"]=a["visits"].astype(np.float32)
    a["simulations"]=np.ones(t,np.int32);a["temperatures"]=np.ones(t,np.float32)
    a["rewards"]=np.zeros(t,np.float32);a["rewards"][-1]=1
    a["game_offsets"]=np.array([0,t],np.int64);a["observation_offsets"]=np.array([0,t+1],np.int64)
    for key,value in (("game_ids",0),("seeds",7),("sizes",size),("rules",rule),("winners",1),("reasons",1)):
        a[key]=np.array([value],dtype=RAW_DTYPES[key])
    m={"contract":CONTRACT_ID,"canvas":canvas,"rows":t,"games":1,"run_id":"test","attempt_id":str(rule),
       "worker_id":0,"cycle_id":1,"model_id":"model","config_id":"config","shard_id":str(rule),"created_ns":rule}
    a["metadata"]=np.frombuffer(json.dumps(m).encode(),np.uint8)
    return a,m


def test_configuration_inheritance_and_fail_fast(tmp_path,config):
    assert config["network"]["canvas"]==6 and config["optimizer"]["momentum"]==0.9
    shutil.copytree(ROOT/"configs",tmp_path/"configs")
    current=tmp_path/"configs"/"minimal_test"
    (current/"train.cfg.local").write_text("[training]\ntrain_steps=7\n")
    assert load_config(current)["training"]["train_steps"]==7
    (current/"net.cfg.local").write_text("[training]\ntrain_steps=7\n")
    with pytest.raises(ValueError,match="ownership"):
        load_config(current)
    (current/"net.cfg.local").unlink()
    (tmp_path/"configs"/"baseline"/"run.cfg").write_text("[run]\nextends=minimal_test\n")
    with pytest.raises(ValueError,match="cycle"):
        load_config(current)


@pytest.mark.parametrize("algorithm,root,nonroot,message",[
    ("alphazero","puct","gumbel","not an allowed"),
    ("muzero","puct","puct","not implemented"),
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
    torch.testing.assert_close(result[0,0,0],torch.tensor([-1.,1.]),rtol=1e-5,atol=1e-5)
    assert result[0,0,1].eq(0).all()
    torch.testing.assert_close(bn.running_mean,torch.tensor([0.2]))
    obs=torch.from_numpy(example_observation(6,5,"standard",(0,))).unsqueeze(0)
    policy=torch.zeros(1,36);policy[0,1]=1
    model=make_network(config)
    loss,pl,vl,reg=losses(torch.zeros(1,36),torch.zeros(1),obs,policy,torch.ones(1),model,0)
    assert float(pl)==pytest.approx(math.log(25),abs=1e-6) # includes the occupied point, excludes padding
    assert float(vl.detach())==1 and float(reg.detach())==0
    assert float(loss.detach())==pytest.approx(math.log(25)+1,abs=1e-6)


def test_full_trajectory_targets_and_corruption(tmp_path):
    a,_=winning_record()
    path=tmp_path/"game.npz";save_npz(path,a)
    loaded=read_raw(path);view=training_view(loaded)
    np.testing.assert_array_equal(view["value"],[1,-1,1,-1,1,-1,1,-1,1])
    assert len(loaded["observations"])==len(view["obs"])+1
    bad=copy.deepcopy(a);bad["policies"][0]=0
    with pytest.raises(ValueError,match="completed visits"):
        validate_raw(bad)
    bad=copy.deepcopy(a);bad["observations"][1,0,0,0]=1
    with pytest.raises(ValueError,match="transition mismatch"):
        validate_raw(bad)


def test_shuffle_conservation_and_reader_restoration(tmp_path,config):
    config["replay"].update(window_mode="fixed",window_rows=100,shuffle_bucket_rows=2,training_shard_rows=2)
    entries=[];expected=Counter()
    for rule in range(3):
        a,m=winning_record(rule);path=tmp_path/"data"/f"{rule}.npz";save_npz(path,a)
        entries.append({"path":str(path.relative_to(tmp_path)),"sha256":sha256(path),"metadata":m})
        view=training_view(a)
        expected.update((obs.tobytes(),p.tobytes(),v.tobytes()) for obs,p,v in zip(view["obs"],view["policy"],view["value"]))
    identity=build_snapshot(tmp_path,1,entries,config)
    snapshot=tmp_path/"snapshots"/identity
    manifest=json.loads((snapshot/"manifest.json").read_text());actual=Counter()
    for item in manifest["files"]:
        with np.load(snapshot/"data"/item["path"]) as a:
            actual.update((obs.tobytes(),p.tobytes(),v.tobytes()) for obs,p,v in zip(a["obs"],a["policy"],a["value"]))
    assert expected==actual and manifest["rows"]==27
    first=BatchReader(snapshot,17,2,1);first.next();state=first.state()
    second=BatchReader(snapshot,17,2,999,state)
    try:
        for _ in range(5):
            a,b=first.next(),second.next()
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])
            assert len(a["value"])==17
    finally:
        first.close();second.close()


def test_katago_window_hand_values(config):
    replay=config["replay"]
    replay.update(window_min_rows=100,taper_scale=100,taper_exponent=0.5,expand_per_row=1,window_max_rows=1000)
    assert desired_window(100,replay)==100
    assert desired_window(400,replay)==300 # sqrt(400)-sqrt(100), divided by 0.5/sqrt(100), plus 100
    assert desired_window(100000,replay)==1000


def test_unique_catalog_is_idempotent(tmp_path):
    a,_=winning_record();save_npz(tmp_path/"data"/"a.npz",a)
    catalog=Catalog(tmp_path,"test","config")
    try:
        catalog.scan({1:"model"});catalog.scan({1:"model"})
        assert catalog.counts(10)==(9,1,9.)
        save_npz(tmp_path/"data"/"duplicate.npz",a)
        with pytest.raises(Exception):
            catalog.scan({1:"model"})
        assert catalog.counts(10)==(9,1,9.)
    finally:
        catalog.close()
