"""Named network exports expose the same public diagnostic API as search."""
import json
import copy
import os
import subprocess

import numpy as np
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import ROOT, load_config, write_native
from etazero.engine_config import load_engine_config
from etazero.models.export import example_inputs
from etazero.models.network import make_network, inference_network


@pytest.mark.parametrize('architecture,channels,blocks', [
    ('b1c6nbt-fson-mish',6,1), ('b5c192h3nbttfrs-fson-silu-rsnh',192,5)])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0',marks=pytest.mark.skipif(
    os.environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA required'))])
def test_named_raw_analysis_and_genmove(tmp_path,architecture,channels,blocks,device):
    torch.set_num_threads(1)
    c=load_config(CONFIGS/'smoke_test')
    c['network'].update(canvas=6,architecture=architecture,channels=channels,blocks=blocks)
    model=make_network(c).eval().to(device)
    if device=='cpu':model.network.use_flex_attention=False
    exported=torch.jit.script(inference_network(model))
    path=tmp_path/'model.pt';exported.save(str(path))
    # CPU load of a CUDA-traced graph also exercises tensor-derived devices in
    # the diagnostic method, independently of the native FP16 execution below.
    restored=torch.jit.load(str(path),map_location='cpu')
    obs,glob=example_inputs(6,5,'freestyle',(0,))
    inputs=torch.from_numpy(obs)[None],torch.from_numpy(glob)[None]
    with torch.inference_mode():
        policy,value,_,_=model.forward_all(*(t.to(device) for t in inputs))
        actual=restored.raw(*inputs)
        cpu_policy,cpu_value,_,_=copy.deepcopy(model).cpu().forward_all(*inputs)
    for a,b in zip(actual,(cpu_policy,cpu_value)):
        torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
    config=load_engine_config(CONFIGS/'baseline',environ={},overrides={'analysis':dict(
        device=device,board_size=5,rule='freestyle',visits=16,search_threads=1,
        nn_randomize=False,root_num_symmetries_to_sample=1,cache_entries=0,
        inference_precision='float16' if device.startswith('cuda') else 'float32')})
    cfg=tmp_path/'engine.cfg';write_native({**config,'network':{'canvas':6}},cfg)
    jobs=[dict(id='new',command='new',size=5,rule='freestyle'),dict(id='play',command='play',action=0),
          dict(id='raw',command='raw_nn'),dict(id='analysis',command='analyze',visits=16),
          dict(id='move',command='genmove',visits=16),dict(id='state',command='state')]
    command=[str(ROOT/'build/etazero'),'analysis','--stream','true','--protocol','json',
             '--config',str(cfg),'--model',str(path),'--model-id','named','--device',device]
    run=subprocess.run(command,input=''.join(json.dumps(j)+'\n' for j in jobs),text=True,capture_output=True,timeout=90)
    assert run.returncode==0,run.stderr
    replies={r['id']:r for r in map(json.loads,run.stdout.splitlines()) if r.get('final')}
    assert all(replies[j['id']]['ok'] for j in jobs),replies
    for head,index in zip(replies['raw']['network_planes']['heads'],(0,1,4,5)):
        expected=policy[0,index].reshape(6,6)[:5,:5].flatten().float().softmax(0).cpu().numpy()
        np.testing.assert_allclose(head['probabilities'],expected,atol=2e-4,rtol=2e-3)
    assert replies['analysis']['state']['moves']==[0]
    assert replies['move']['played'] and len(replies['state']['state']['moves'])==2
    gtp=subprocess.run([str(ROOT/'build/etazero'),'gtp',*command[6:]],
                       input='1 boardsize 5\n2 play b A1\n3 kata-raw-nn 0\n4 genmove w\n5 quit\n',
                       text=True,capture_output=True,timeout=90)
    assert gtp.returncode==0 and '?3' not in gtp.stdout and '?4' not in gtp.stdout,gtp.stdout+gtp.stderr
    assert '=3' in gtp.stdout and '=4' in gtp.stdout
