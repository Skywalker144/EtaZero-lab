"""Real GPU checks for weight/buffer replacement and CPU-only runtime retention."""
import json
import os
import subprocess
import numpy as np
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import ROOT, load_config
from etazero.network import make_network, inference_network, MaskedBatchNorm
from etazero.runtime import run_training

pytestmark=pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA required')


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
@pytest.mark.parametrize('precision',['float32','float16'])
def test_reloaded_weights_buffers_executable_and_gpu_release(tmp_path,algorithm,precision):
    torch.set_num_threads(1)
    config=load_config(CONFIGS/'smoke_test')
    config['agent']['algorithm']=algorithm
    if algorithm=='muzero':
        config['muzero']=dict(latent_channels=16,dynamics_channels=24,dynamics_blocks=1,
                              prediction_channels=24,prediction_blocks=1)
        config['unroll']=dict(steps=3,hidden_gradient_scale=.5)
    models=[];paths=[]
    for index in range(3):
        torch.manual_seed(71+index)
        model=make_network(config).cuda().eval()
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,MaskedBatchNorm):
                    layer.running_mean.uniform_(-.3,.3);layer.running_std.uniform_(.5,1.5)
                    layer.weight.uniform_(-.2,.2);layer.bias.uniform_(-.1,.1)
        # The tensor layout alone cannot detect a changed executable.
        if index==2:
            head=model.value_head if algorithm=='alphazero' else model.prediction.value_head
            head.act=torch.nn.ReLU()
        scripted=torch.jit.script(inference_network(model));models.append(scripted)
        path=tmp_path/f'{index}.pt';scripted.save(str(path));paths.append(path)
    order=(0,1,2,1)
    result=subprocess.run([str(ROOT/'build/model_reload_probe'),algorithm,'cuda:0',precision,
                           *[str(paths[i]) for i in order]],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    rounds=json.loads(result.stdout)
    assert [r['reused'] for r in rounds]==[False,True,False,False]
    assert all(r['allocated_after_offload']==0 for r in rounds)
    spatial=torch.zeros(5,5,6,6,device='cuda');spatial[:,0]=1
    for i in range(5):spatial[i,1,0,i]=1;spatial[i,2,1,i]=1
    globals=torch.zeros(5,6,device='cuda')
    def flatten(output):
        if algorithm=='alphazero':logits,wdl,optimistic,error=output;latent=None
        else:latent,logits,wdl,optimistic,error=output
        values=torch.cat((logits.float(),wdl.float().softmax(1),optimistic.float(),error.float()[:,None]),1)
        if latent is not None:values=torch.cat((values,latent.float().flatten(1)),1)
        return values.cpu().numpy().ravel()
    with torch.inference_mode(),torch.autocast('cuda',enabled=precision=='float16',dtype=torch.float16):
        for index,record in zip(order,rounds):
            model=models[index]
            if algorithm=='alphazero':outputs=[model(spatial,globals),model(spatial[:1],globals[:1])]
            else:
                roots=model.initial(spatial,globals)
                outputs=[roots,model.recurrent(roots[0],torch.arange(1,6,device='cuda')),
                         model.initial(spatial[:1],globals[:1])]
            expected=np.concatenate([flatten(output) for output in outputs])
            tolerance=3e-3 if precision=='float16' else 3e-5
            np.testing.assert_allclose(record['values'],expected,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_training_rounds_reuse_only_after_model_publication(tmp_path,algorithm):
    config=load_config(CONFIGS/'smoke_test')
    config['agent']['algorithm']=algorithm
    config['training']['compile']=False
    config['inference']['server_threads']=2
    if algorithm=='muzero':
        config['muzero']=dict(latent_channels=16,dynamics_channels=24,dynamics_blocks=1,
                              prediction_channels=24,prediction_blocks=1)
        config['unroll']=dict(steps=3,hidden_gradient_scale=.5)
        config['search']['reuse_tree']=False;config['graph_search']['use_graph_search']=False
        config['symmetry']['root_num_symmetries_to_sample']=1
    state=run_training(tmp_path,config,ROOT/'build/etazero',max_iteration=3)
    events=[json.loads(s) for s in (tmp_path/'logs/events.jsonl').read_text().splitlines()]
    setup=[e for e in events if e['event']=='inference_setup']
    assert [(e['iteration'],e['reused_runtime']) for e in setup]==[(2,False),(3,True)]
    starts={e['iteration']:e for e in events if e['event']=='worker_start'}
    assert starts[2]['pid']==starts[3]['pid']
    assert starts[2]['command'][starts[2]['command'].index('--model-id')+1]!=starts[3]['command'][starts[3]['command'].index('--model-id')+1]
    assert state['checkpoint']['total_steps']==3*config['training']['train_steps']
    assert run_training(tmp_path,config,ROOT/'build/etazero',max_iteration=3)==state
