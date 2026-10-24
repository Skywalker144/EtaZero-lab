"""Actual backend rule masks, including latent descendants and metadata policy."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from etazero.models.katago_network import KataGoNet
from etazero.models.muzero.network import NetworkConfig, MuZeroNet
from etazero.models.network import inference_network
from etazero.models.cuda_export import export_source

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
@pytest.mark.parametrize('backend',['torch','cuda'])
@pytest.mark.parametrize('precision',['float32','float16'])
def test_rules_mask_before_softmax_and_survive_latent(tmp_path,algorithm,backend,precision):
    gpu=os.environ.get('ETAZERO_GPU_TESTS')=='1'
    if not gpu and (backend=='cuda' or precision=='float16'):pytest.skip('Host CUDA acceptance')
    if algorithm=='alphazero':
        model=KataGoNet('b2c16-fson-mish',6,game='go').eval();head=model.network.value_head
    else:
        model=MuZeroNet(NetworkConfig(6,16,24,1,24,1,24,0,architecture='nbt',game='go')).eval();head=model.prediction.value_head
    with torch.no_grad():
        head.linear_valuehead.weight.zero_()
        # Source order W/L/NR/D; exercise NR dominance without losing other logits.
        head.linear_valuehead.bias.copy_(torch.tensor([0.,-1.,1000.,-2.]))
    if backend=='torch':
        path=tmp_path/'model.pt';torch.jit.script(inference_network(model)).save(str(path))
    else:
        path=tmp_path/'model.bin'
        (tmp_path/'manifest.json').write_text(json.dumps(dict(algorithm=algorithm,canvas=6,game='go')))
        if algorithm=='alphazero':export_source(model.network,path,'rules')
        else:
            export_source(model.representation.body,tmp_path/'representation.bin','representation')
            export_source(model.dynamics.body,tmp_path/'dynamics.bin','dynamics')
            export_source(model.prediction.export_model(),path,'prediction')
    for rule in ('tromp-taylor','japanese'):
        result=subprocess.run([str(ROOT/'build/inference_rules_probe'),str(path),algorithm,rule,'cuda:0' if gpu else 'cpu',precision],capture_output=True,text=True,check=True)
        rows=[json.loads(line) for line in result.stdout.splitlines()]
        assert len(rows)==(1 if algorithm=='alphazero' else 4)
        expected=np.array([1.,np.exp(-2),np.exp(-1),0.]);expected/=expected.sum()
        if rule=='japanese':expected=np.array([0.,0.,0.,1.])
        for row in rows:np.testing.assert_allclose(row['wdl'],expected,rtol=1e-6,atol=1e-7)

@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA acceptance')
def test_metadata_optimism_is_disabled_in_actual_cuda_backend(tmp_path):
    torch.manual_seed(85)
    model=KataGoNet('b2c16-fson-mish-meta',6,game='go').eval();path=tmp_path/'human.bin'
    export_source(model.network,path,'metadata')
    result=subprocess.run([str(ROOT/'build/inference_rules_probe'),str(path),'alphazero','japanese','cuda:0','float16','rank_1d_9d'],capture_output=True,text=True,check=True)
    row=json.loads(result.stdout)
    assert not row['optimism_allowed'] and row['same_policy']
