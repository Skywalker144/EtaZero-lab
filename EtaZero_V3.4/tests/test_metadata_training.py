"""Real metadata survives collection, shuffle, training, export and resume."""
import json
import os
import numpy as np
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import ROOT, load_config
from etazero.run.runtime import run_training
from etazero.train.learner import load_checkpoint, train_iteration
from etazero.data.source_reader import SourceBatchReader
from etazero.storage import load_json

CUDA=pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA acceptance')

@pytest.mark.parametrize('device,compiled', [('cpu',False),pytest.param('cuda:0',False,marks=CUDA),pytest.param('cuda:0',True,marks=CUDA)])
def test_metadata_selfplay_training_export_and_resume(tmp_path,device,compiled):
    torch.set_num_threads(1)
    c=load_config(CONFIGS/'smoke_test')
    c['network']['architecture']='b2c16-fson-mish-meta'
    c['devices'].update(train=device,selfplay=device)
    c['training'].update(compile=compiled,no_repeat_files=True,skip_validation=True,amp='float16' if device.startswith('cuda') else 'off')
    if device=='cpu':c['inference']['inference_precision']='float32'
    first=run_training(tmp_path,c,ROOT/'build/etazero',max_iteration=1)
    state=run_training(tmp_path,c,ROOT/'build/etazero',max_iteration=2)
    assert state['checkpoint']['total_steps']>first['checkpoint']['total_steps']>0
    checkpoint=load_checkpoint(tmp_path,state['checkpoint'],c)
    assert checkpoint['reader']['no_repeat_files']
    original=load_checkpoint(tmp_path,first['checkpoint'],c)
    meta_keys=[k for k in checkpoint['model'] if 'metadata_encoder' in k and k.endswith('weight')]
    assert meta_keys and any(not torch.equal(checkpoint['model'][k],original['model'][k]) for k in meta_keys)
    for path in (tmp_path/'selfplay').rglob('*.npz'):
        with np.load(path) as raw:
            meta=raw['metadataInputNC']
            assert meta.shape==(len(raw['globalTargetsNC']),192) and np.isfinite(meta).all()
            assert not meta[:,:2].any() # Both players are AI, not human profiles.
            assert np.all(meta[:,4:6]==1) # Unknown ranks.
            assert meta.any()
    manifest=load_json(tmp_path/'models'/state['model']['id']/'manifest.json')
    assert manifest['metadata_profile']=='selfplay'
    if device.startswith('cuda'):
        import subprocess
        directory=tmp_path/'models'/state['model']['id']
        results=[]
        for name in ('model.bin','reference.pt'):
            result=subprocess.run([str(ROOT/'build/etazero'),'infer','--model',str(directory/name),
                '--model-id','metadata','--config',str(tmp_path/'config/effective.cfg'),
                '--device',device,'--size','5','--rule','freestyle','--moves','0,6'],
                text=True,capture_output=True,timeout=90)
            assert result.returncode==0,result.stderr
            results.append(json.loads(result.stdout))
        np.testing.assert_allclose(results[0]['raw_wdl'],results[1]['raw_wdl'],atol=3e-3,rtol=3e-3)
    assert run_training(tmp_path,c,ROOT/'build/etazero',max_iteration=2)==state
    # Consume the same immutable snapshot to exhaustion, then resume on it:
    # no file may be recycled and no extra optimizer step may be charged.
    status=load_json(tmp_path/'.internal/iterations/000002/status.json')
    plan=dict(iteration=3,snapshot_id=status['snapshot_id'],train_steps=100,batch_size=c['training']['batch_size'])
    end,done=train_iteration(tmp_path,c,plan,state['checkpoint'],lambda *a,**k:None,lambda:False)
    assert done and end['step']<100
    exhausted=load_checkpoint(tmp_path,end,c)
    reader=SourceBatchReader(tmp_path/'snapshots'/status['snapshot_id'],8,1,19,4,1,exhausted['reader'],no_repeat_files=True)
    try:
        with pytest.raises(StopIteration):reader.next()
    finally:reader.close()
    plan['iteration']=4
    again,done=train_iteration(tmp_path,c,plan,end,lambda *a,**k:None,lambda:False)
    assert done and again['step']==0 and again['total_steps']==end['total_steps']
