"""Independent PyTorch/source-kernel parity for exact dynamic GPU batches."""
from pathlib import Path
import os
import subprocess
import numpy as np
import pytest
import torch
from etazero.models.katago_network import KataGoNet
from etazero.models.cuda_export import export_source
from etazero.games import SPECS

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA acceptance')


@pytest.fixture(autouse=True)
def reference_precision():
    # Training enables autotuning globally. FP32 oracle comparisons require
    # full-precision cuDNN/GEMM independently of earlier training cases.
    precision=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        with torch.backends.cudnn.flags(benchmark=False,allow_tf32=False):
            yield
    finally:torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize('game', ['gomoku','hex','go','xiangqi'])
@pytest.mark.parametrize('precision', ['float32','float16'])
@pytest.mark.parametrize('batch', [1,17,33])
def test_source_cuda_outputs(tmp_path, game, precision, batch):
    torch.manual_seed(17)
    canvas = 10 if game == 'xiangqi' else 6
    spec = SPECS[game]
    model = KataGoNet('b2c16-fson-mish',canvas,game=game).eval()
    path = tmp_path/'model.bin'
    export_source(model.network, path, 'parity')
    binary = Path(__file__).resolve().parents[1]/'build/cuda_inference_probe'
    result = subprocess.run([str(binary),str(path),game,str(canvas),str(batch),precision,str(tmp_path/'inputs')],
                            check=True,capture_output=True,text=True,
                            env={**os.environ, **({'NVIDIA_TF32_OVERRIDE':'0'} if precision=='float32' else {})})
    actual = np.loadtxt(result.stdout.splitlines(),ndmin=2)
    inputs = np.fromfile(tmp_path/'inputs',np.float32).reshape(batch,-1)
    obs = torch.from_numpy(inputs[:,:spec.planes*canvas*canvas].reshape(batch,spec.planes,canvas,canvas)).cuda()
    globals = torch.from_numpy(inputs[:,spec.planes*canvas*canvas:]).cuda()
    model = model.cuda()
    with torch.no_grad(), torch.autocast('cuda',enabled=precision=='float16'):
        policy, value, _, error, _, variance = model.training_outputs(obs,globals)[0]
    expected = torch.cat((policy[:,0].float(),value.float().softmax(1),policy[:,5].float(),
                          .5*torch.nn.functional.softplus(error.float()[:,None]*.5),40*torch.nn.functional.softplus(variance.float()[:,None])),1).cpu().numpy()
    valid = np.ones_like(expected,dtype=bool)
    if game != 'xiangqi':
        mask=obs[:,0].flatten(1).cpu().numpy().astype(bool)
        if game == 'go':mask=np.c_[mask,np.ones(batch,bool)]
        valid[:,:spec.actions(canvas)]=mask
        valid[:,spec.actions(canvas)+4:2*spec.actions(canvas)+4]=mask
    # Source cuDNN may use TF32 even in its FP32 mode; disable it in the
    # subprocess for this numerical oracle, without changing production flags.
    tolerance = 2e-3 if precision == 'float16' else 1e-4
    np.testing.assert_allclose(actual[valid],expected[valid],atol=tolerance,rtol=tolerance)


@pytest.mark.parametrize('channels',[3,4])
def test_real_human_profile_and_source_three_outcome_model(tmp_path,channels):
    torch.manual_seed(85)
    model=KataGoNet('b2c16-fson-mish-meta',6,game='go').eval()
    if channels==3:
        model.network.value_head.linear_valuehead=torch.nn.Linear(model.network.value_head.linear_valuehead.in_features,3)
    path=tmp_path/'human.bin'
    export_source(model.network,path,'human')
    binary=Path(__file__).resolve().parents[1]/'build/cuda_inference_probe'
    result=subprocess.run([str(binary),str(path),'go','6','2','float16',str(tmp_path/'inputs'),'rank_1d_9d'],check=True,capture_output=True,text=True)
    inputs=np.fromfile(tmp_path/'inputs',np.float32).reshape(2,-1)
    meta=inputs[:,-192:]
    assert meta.any() and not np.array_equal(meta[0],meta[1])
    # Source encodes the relative player and opponent rank in swapped blocks.
    np.testing.assert_array_equal(meta[0,6:40],meta[1,40:74])
    np.testing.assert_array_equal(meta[0,40:74],meta[1,6:40])
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        outputs=model.cuda().source_outputs(torch.from_numpy(inputs[:,:22*36].reshape(2,22,6,6)).cuda(),
            torch.from_numpy(inputs[:,22*36:22*36+19]).cuda(),torch.from_numpy(meta).cuda())[0]
    expected=outputs[1].float().softmax(1).cpu().numpy()
    expected=np.stack((expected[:,0],expected[:,3] if channels==4 else np.zeros(2),expected[:,1],expected[:,2]),1)
    actual=np.loadtxt(result.stdout.splitlines(),ndmin=2)
    np.testing.assert_allclose(actual[:,37:41],expected,atol=2e-3,rtol=2e-3)


@pytest.mark.parametrize('architecture',['b5c192nbt-fson-mish','b5c192h3nbttfrs','b7c96h6kv3qk32v16tflrs-fson-silu'])
def test_production_source_presets(tmp_path,architecture):
    torch.manual_seed(61)
    model=KataGoNet(architecture,15).eval()
    path=tmp_path/'model.bin';export_source(model.network,path,'source_preset')
    binary=Path(__file__).resolve().parents[1]/'build/cuda_inference_probe'
    result=subprocess.run([str(binary),str(path),'gomoku','15','3','float16',str(tmp_path/'inputs')],check=True,capture_output=True,text=True)
    inputs=np.fromfile(tmp_path/'inputs',np.float32).reshape(3,-1)
    obs=torch.from_numpy(inputs[:,:5*225].reshape(3,5,15,15)).cuda();g=torch.from_numpy(inputs[:,5*225:]).cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):p,v,_,e,_,variance=model.cuda().training_outputs(obs,g)[0]
    actual=np.loadtxt(result.stdout.splitlines(),ndmin=2)
    expected=torch.cat((p[:,0].float(),v.float().softmax(1),p[:,5].float(),.5*torch.nn.functional.softplus(e.float()[:,None]*.5),40*torch.nn.functional.softplus(variance.float()[:,None])),1).cpu().numpy()
    mask=obs[:,0].flatten(1).cpu().numpy().astype(bool)
    valid=np.c_[mask,np.ones((3,4),bool),mask,np.ones((3,2),bool)]
    np.testing.assert_allclose(actual[valid],expected[valid],atol=3e-3,rtol=3e-3)


def test_concurrent_public_analysis_profiles_are_isolated(tmp_path):
    import json
    from test_multigame_diagnostics import cpu_model
    from etazero.config import write_native
    from test_multigame import BINARY
    _,_,config,profile=cpu_model(tmp_path,'go','alphazero',5,6)
    torch.manual_seed(185)
    main=KataGoNet('b2c16-fson-mish',6,game='go').eval()
    human=KataGoNet('b2c16-fson-mish-meta',6,game='go').eval()
    export_source(main.network,tmp_path/'main.bin','main')
    export_source(human.network,tmp_path/'human.bin','human')
    profile['analysis'].update(device='cuda:0',human_model=str(tmp_path/'human.bin'),human_sl_profile='',
        human_sl_root_weightless=.5,num_analysis_threads=2,search_threads=1,
        min_playouts_per_thread=0,max_playouts=100000,max_time=30,root_symmetry_pruning=False,wide_root_noise=0,nn_randomize=False)
    text=config.read_text();native=tmp_path/'engine.cfg';write_native(profile,native)
    config.write_text(text[:text.index('[analysis]')]+native.read_text())
    base=dict(boardXSize=3,boardYSize=5,rules='chinese',moves=[],maxVisits=12,includePolicy=True)
    # Repeated identical requests must retain their metadata across concurrent jobs.
    requests=[dict(base,id=str(i),overrideSettings={'humanSLProfile':name}) for i,name in enumerate(['rank_1d','rank_20k','rank_1d','rank_20k'])]
    result=subprocess.run([str(BINARY),'analysis','--stream','true','--protocol','katago','--config',str(config),
        '--model',str(tmp_path/'main.bin'),'--model-id','profiles','--device','cuda:0'],input=''.join(json.dumps(x)+'\n' for x in requests),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    reports=sorted([json.loads(x) for x in result.stdout.splitlines()],key=lambda x:x['id'])
    assert len(reports)==4 and all('error' not in x for x in reports),reports
    assert all(x['rootInfo']['rawVarTimeLeft']>0 for x in reports)
    assert all('humanPolicy' in x for x in reports)
    np.testing.assert_allclose(reports[0]['humanPolicy'],reports[2]['humanPolicy'],atol=1e-6)
    np.testing.assert_allclose(reports[1]['humanPolicy'],reports[3]['humanPolicy'],atol=1e-6)
    assert not np.allclose(reports[0]['humanPolicy'],reports[1]['humanPolicy'],atol=1e-6)
    # Main network policy is unchanged by metadata overrides on the secondary model.
    np.testing.assert_allclose(reports[0]['policy'],reports[2]['policy'],atol=1e-6)
    np.testing.assert_allclose(reports[1]['policy'],reports[3]['policy'],atol=1e-6)
    # Weightless human exploration consumes playout budget without adding root visits.
    assert all(0<x['rootInfo']['visits']<=12 for x in reports)
    assert any(x['rootInfo']['visits']<12 for x in reports)


@pytest.mark.parametrize('compressed',[False,True])
@pytest.mark.parametrize('metadata',[False,True])
def test_standalone_source_model_as_primary(tmp_path,compressed,metadata):
    import gzip
    import json
    from test_multigame_diagnostics import cpu_model
    from test_multigame import BINARY
    from etazero.config import write_native
    from etazero.evaluation.analysis import model_info
    from etazero.models.katago.model_pytorch import Model
    from etazero.models.katago_configs import resolve
    _,_,config,profile=cpu_model(tmp_path,'go','alphazero',5,6)
    source=Model(resolve('b2c16-fson-mish'+('-meta' if metadata else '')),6).eval()
    source.initialize()
    directory=tmp_path/'standalone';directory.mkdir()
    path=directory/'source.bin';export_source(source,path,'source')
    if compressed:
        payload=path.read_bytes();path=path.with_suffix('.bin.gz');path.write_bytes(gzip.compress(payload))
    _,info=model_info(path,6)
    assert (info['game'],info['algorithm'],info['canvas'])==('go','alphazero',6)
    profile['analysis'].update(device='cuda:0',human_model='',human_sl_profile='rank_1d' if metadata else '',
        num_analysis_threads=2,search_threads=1,nn_randomize=False,nn_symmetry=0)
    text=config.read_text();native=tmp_path/'engine.cfg';write_native(profile,native)
    config.write_text(text[:text.index('[analysis]')]+native.read_text())
    base=dict(boardXSize=3,boardYSize=5,rules='chinese',komi=.5,moves=[],maxVisits=12,includePolicy=True)
    requests=[dict(base,id=str(i),**({'overrideSettings':{'humanSLProfile':name}} if metadata else {}))
              for i,name in enumerate(['rank_1d','rank_20k','rank_1d','rank_20k'])]
    result=subprocess.run([str(BINARY),'analysis','--stream','true','--protocol','katago','--config',str(config),
        '--model',str(path),'--model-id','standalone','--device','cuda:0'],input=''.join(json.dumps(x)+'\n' for x in requests),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    reports=sorted(map(json.loads,result.stdout.splitlines()),key=lambda x:x['id'])
    assert len(reports)==4 and all('error' not in x and x.get('moveInfos') for x in reports),reports
    np.testing.assert_allclose(reports[0]['policy'],reports[2]['policy'],atol=1e-6)
    np.testing.assert_allclose(reports[1]['policy'],reports[3]['policy'],atol=1e-6)
    if metadata:assert not np.allclose(reports[0]['policy'],reports[1]['policy'],atol=1e-6)


@pytest.mark.parametrize('backend',['cuda','torch'])
def test_model_postprocess_multipliers_reach_native_outputs(tmp_path,backend):
    import json
    import math
    from etazero.models.katago_network import inference_network
    model=KataGoNet('b2c16-fson-mish',6).eval()
    # Constant raw auxiliary outputs make the expected source formulas independent
    # of network implementation, board inputs, batching and FP32 kernel choices.
    with torch.no_grad():
        head=model.network.value_head
        head.linear_miscvaluehead.weight[3].zero_();head.linear_miscvaluehead.bias[3].zero_()
        head.linear_moremiscvaluehead.weight[0].zero_();head.linear_moremiscvaluehead.bias[0].zero_()
    model.network.shortterm_value_error_multiplier=1.0
    model.network.variance_time_multiplier=80.0
    root=Path(__file__).resolve().parents[1]
    if backend=='cuda':
        path=tmp_path/'model.bin';export_source(model.network,path,'postprocess')
        result=subprocess.run([str(root/'build/cuda_inference_probe'),str(path),'gomoku','6','3','float32'],capture_output=True,text=True,check=True)
        actual=np.loadtxt(result.stdout.splitlines(),ndmin=2)[:,-2:]
    else:
        path=tmp_path/'model.pt';torch.jit.script(inference_network(model)).save(str(path))
        result=subprocess.run([str(root/'build/inference_probe'),str(path),'cuda:0','float32'],capture_output=True,text=True,check=True)
        actual=np.concatenate([np.asarray(batch)[:,-2:] for batch in json.loads(result.stdout)])
    np.testing.assert_allclose(actual,np.broadcast_to([math.log(2),80*math.log(2)],actual.shape),rtol=1e-6,atol=1e-6)
