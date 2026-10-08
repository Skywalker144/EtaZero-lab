"""Hex configuration, physical/model coordinates and real CUDA training acceptance."""
import copy
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import ROOT, load_config, validate, validate_hex_opening
from etazero.engine_config import load_engine_config
from etazero.schema import GLOBALS, HEX_GLOBAL, HEX_WHITE_GLOBAL
from etazero.symmetry import augment_batch
from etazero.runtime import run_training


def hex_config(algorithm='alphazero'):
    c=load_config(CONFIGS/'smoke_test')
    c['environment'].update(sizes='5,6',size_weights='1,1',rules='hex',rule_weights='1',forbidden_feature_dropout_prob=0)
    c['symmetry'].update(root_num_symmetries_to_sample=2,nn_randomize=True,nn_symmetry=0)
    c['hex_opening'].update(probability=1,make_fair_probability=1)
    c['policy_init'].update(policy_init=False)
    c['selfplay']['bootstrap_games']=8
    c['training'].update(train_steps=2,batch_size=4,replay_ratio=.5,compile=False,amp='float16',checkpoint_every=1,skip_validation=True)
    c['replay'].update(min_rows=8,keep_target_rows=256)
    c['inference'].update(cache_entries=64,inference_precision='float16',max_batch=8)
    c['hint_positions']['hint_positions_prob']=0
    c['game_forks'].update(early_fork_game_prob=0,fork_game_prob=0)
    if algorithm=='muzero':
        c['agent']['algorithm']='muzero'
        c['muzero']=dict(latent_channels=16,dynamics_channels=24,dynamics_blocks=1,prediction_channels=24,prediction_blocks=1)
        c['unroll']=dict(steps=2,hidden_gradient_scale=.5)
        c['symmetry']['root_num_symmetries_to_sample']=1
        c['search']['reuse_tree']=False
        c['graph_search']['use_graph_search']=False
    validate(c)
    return c


def test_hex_configuration_and_invalid_symmetries():
    for algorithm in ('alphazero','muzero'):
        c=hex_config(algorithm)
        for key,value in (('nn_symmetry',2),('root_num_symmetries_to_sample',3)):
            broken=copy.deepcopy(c);broken['symmetry'][key]=value
            with pytest.raises(ValueError):validate(broken)
    for key,value in (('min_accept_rate',0),('probability',1.1),('balance_exponent',float('nan'))):
        broken=copy.deepcopy(c['hex_opening']);broken[key]=value
        with pytest.raises(ValueError):validate_hex_opening(broken)
    broken=copy.deepcopy(c['hex_opening']);broken['make_fair_probability']=.98
    with pytest.raises(ValueError):validate_hex_opening(broken,match=True)


@pytest.mark.parametrize('symmetry',range(8))
def test_mixed_game_coordinates_and_fixed_muzero_root(symmetry):
    # Same physical action; row0 Gomoku, row1 Black Hex, row2 White Hex.
    n=5;actions=torch.tensor([[1,7],[1,7],[1,7]])
    obs=torch.zeros(3,5,n,n);obs[:,0]=1;obs[:,1,0,1]=1
    globals=torch.zeros(3,len(GLOBALS));globals[1:,HEX_GLOBAL]=1;globals[2,HEX_WHITE_GLOBAL]=1
    targets=torch.nn.functional.one_hot(actions,n*n).float()
    batch=dict(obs=obs,globals=globals,actions=actions,policy=targets,opponent_policy=targets,q_values=targets,q_visits=targets)
    result=augment_batch(batch,symmetry)
    assert torch.equal(result['actions'],result['policy'].argmax(-1))
    for key in ('opponent_policy','q_values','q_visits'):assert torch.equal(result[key],result['policy'])
    # Hand-computed Hex mappings, with one fixed orientation across the unroll.
    for row in (1,2):
        expected=[]
        for a in actions[row].tolist():
            y,x=divmod(a,n)
            if symmetry%2:x,y=n-1-x,n-1-y
            if row==2:x,y=y,x
            expected.append(y*n+x)
        assert result['actions'][row].tolist()==expected
        assert result['obs'][row,1].flatten().argmax()==expected[0]
    assert torch.equal(result['globals'],globals)


def test_match_hex_parameters_are_independent(tmp_path):
    p=tmp_path/'match.cfg'
    p.write_text(f'@include "{CONFIGS / "baseline/match.cfg"}"\n[match]\nrule=hex\n[hex_opening]\nbalance_exponent=9\nmin_accept_rate=.02\n')
    result=load_engine_config(p,True,environ={'MATCH_HEX_OPENING_BALANCE_EXPONENT':'11'})
    assert result['hex_opening']['balance_exponent']==11
    assert result['hex_opening']['min_accept_rate']==.02
    assert load_config(CONFIGS/'baseline')['hex_opening']['balance_exponent']==6
    p.write_text(p.read_text()+'make_fair_probability=.98\n')
    with pytest.raises(ValueError,match='both balanced opening'):load_engine_config(p,True,environ={})


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS')!='1',reason='Requires host CUDA')
@pytest.mark.parametrize('algorithm',('alphazero','muzero'))
def test_hex_cuda_training_resume_analysis_match_and_web(tmp_path,algorithm):
    from etazero.runtime import run_training
    from etazero.data import read_raw,metadata
    from etazero.storage import load_json,sha256
    from etazero.analysis import analyze
    from etazero.arena import single_match
    from etazero.export import example_inputs
    from web.engine import Engine
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    c=hex_config(algorithm);root=tmp_path/'run';binary=ROOT/'build/etazero'
    state=run_training(root,c,binary,max_iteration=2)
    checkpoint=state['checkpoint'];assert checkpoint['total_steps']>0
    shards={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')};assert shards
    accepted=0
    for p in shards:
        raw=read_raw(p)
        assert (raw['rules']==3).all() and (raw['reasons']==3).all() and (raw['winners']!=0).all()
        assert (raw['globals'][:,HEX_GLOBAL]==1).all()
        assert np.array_equal(raw['globals'][:,HEX_WHITE_GLOBAL],raw['players']==-1)
        assert not raw['observations'][:,3:5].any()
        if 'random' not in p.parts:
            accepted+=int((raw['balanced_moves']==1).sum())
        assert ((raw['balanced_moves']==0)|(raw['balanced_moves']==1)).all()
    assert accepted>0
    state=run_training(root,c,binary,max_iteration=3)
    assert state['checkpoint']['total_steps']>checkpoint['total_steps']
    assert all(sha256(p)==digest for p,digest in shards.items())
    model=root/load_json(root/'models/current.json')['model']['path']
    engine=load_engine_config(CONFIGS/('muzero_minimal_test' if algorithm=='muzero' else 'smoke_test'))
    engine['analysis'].update(rule='hex',board_size=5,visits=9,inference_precision='float16')
    _, payload=analyze(engine,binary,root,model=model,size=5,rule='hex',moves='1')
    result=payload['result']
    # Independently compare physical White logits to a transposed scripted input/output.
    obs,g=example_inputs(6,5,'hex',(1,));script=torch.jit.load(str(model),map_location='cuda').eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        o=torch.from_numpy(obs).unsqueeze(0).to('cuda').transpose(2,3).contiguous()
        gl=torch.from_numpy(g).unsqueeze(0).to('cuda')
        output=script.initial(o,gl) if algorithm=='muzero' else script(o,gl)
        logits=output[1] if algorithm=='muzero' else output[0]
    expected=logits[0].reshape(6,6).transpose(0,1)[:5,:5].float().cpu().flatten().numpy()
    np.testing.assert_allclose(np.array(result['raw_logits']),expected,rtol=3e-3,atol=3e-3)
    match=load_engine_config(CONFIGS/('muzero_minimal_test' if algorithm=='muzero' else 'smoke_test'),True)
    match['match'].update(rule='hex',board_size=5,visits=9,games=4,inference_precision='float16')
    single_match(match,binary,model,model,tmp_path/'match')
    openings=list((tmp_path/'match/pairs').rglob('openings/*.json'));assert len(openings)==2
    for path in openings:
        data=json.loads(path.read_text());assert len(data['moves'])==1 and data['balanced_moves']==1
        assert set(data['balance_evaluators'])=={1}
    engine.update(opening=match['opening'],hex_opening=match['hex_opening'])
    with Engine(binary,model,engine) as session:
        empty=session.command('new 5 hex')['state'];assert empty['turn']==0
        opening=session.command('new 5 hex balanced 31 10000')['state'];assert opening['turn']==1 and opening['player']==-1
        reply=session.command('analyze 9');assert reply['analysis']['network_planes']['heads']
        assert session.command('genmove 9')['state']['turn']==2
