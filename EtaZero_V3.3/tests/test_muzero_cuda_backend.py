"""Native GPU MuZero agrees with source-component PyTorch through recurrent steps."""
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from etazero.games import SPECS
from etazero.models.muzero.network import NetworkConfig, MuZeroNet
from etazero.models.cuda_export import export_source

@pytest.fixture(autouse=True)
def full_precision_reference():
    conv,matmul=torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    yield
    torch.backends.cudnn.allow_tf32=conv;torch.backends.cuda.matmul.allow_tf32=matmul

pytestmark=pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA acceptance')


@pytest.mark.parametrize('game',['gomoku','hex','go','xiangqi'])
@pytest.mark.parametrize('architecture',['nbt','resnet'])
@pytest.mark.parametrize('precision',['float32','float16'])
@pytest.mark.parametrize('batch',[1,17])
def test_cuda_latents_initial_recurrent_and_parent_reuse(tmp_path,game,architecture,precision,batch):
    torch.manual_seed(27)
    canvas=10 if game=='xiangqi' else 6
    c=NetworkConfig(canvas,16,24,1,24,1,24,0,architecture=architecture,game=game)
    m=MuZeroNet(c).eval()
    export_source(m.representation.body,tmp_path/'representation.bin','representation')
    export_source(m.dynamics.body,tmp_path/'dynamics.bin','dynamics')
    export_source(m.prediction.export_model(),tmp_path/'model.bin','prediction')
    binary=Path(__file__).resolve().parents[1]/'build/muzero_cuda_probe'
    result=subprocess.run([str(binary),str(tmp_path/'model.bin'),game,str(canvas),str(batch),precision,str(tmp_path/'inputs')],capture_output=True,text=True,env={**os.environ,"NVIDIA_TF32_OVERRIDE":"0"})
    assert result.returncode==0,result.stderr
    actual=np.loadtxt(result.stdout.splitlines(),ndmin=2).reshape(5,batch,-1)
    spec=SPECS[game];area=canvas*canvas;features=spec.planes*area
    inputs=np.fromfile(tmp_path/'inputs',np.float32).reshape(batch,-1)
    obs=torch.from_numpy(inputs[:,:features].reshape(batch,spec.planes,canvas,canvas)).cuda()
    g=torch.from_numpy(inputs[:,features:]).cuda();m=m.cuda()
    with torch.no_grad(),torch.autocast('cuda',enabled=precision=='float16'):
        h=m.representation(obs,g);root=h
        for step in range(5):
            if step:
                action=area if game=='go' and step==3 else 0 if step==4 else step-1
                h=m.dynamics(root if step==4 else h,torch.full((batch,),action,device='cuda',dtype=torch.long))
            p,v,_,e=m.prediction(h)
            expected=torch.cat((h.flatten(1),p[:,0].float(),v.float().softmax(1),p[:,5].float(),.5*torch.nn.functional.softplus(e.float()[:,None]*.5)),1).cpu().numpy()
            valid=np.ones_like(expected,dtype=bool)
            offset=17*area
            if game!='xiangqi':
                mask=obs[:,0].flatten(1).cpu().numpy().astype(bool)
                if game=='go':mask=np.c_[mask,np.ones(batch,bool)]
                valid[:,offset:offset+spec.actions(canvas)]=mask
                valid[:,offset+spec.actions(canvas)+4:offset+2*spec.actions(canvas)+4]=mask
            tol=5e-3 if precision=='float16' else 1e-4
            np.testing.assert_allclose(actual[step][valid],expected[valid],rtol=tol,atol=tol,err_msg=f'step {step}')
    # The final expansion reuses the original parent, exactly matching step 1.
    np.testing.assert_array_equal(actual[4],actual[1])
