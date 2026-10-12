"""Gumbel targets reuse the existing AZ/MZ data and learner layouts."""
import copy
import json
import os
import subprocess
import numpy as np
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import ROOT, GUMBEL_DEFAULTS, load_config, validate, write_native
from etazero.data import read_raw, metadata, training_view
from etazero.engine_config import load_engine_config, validate_engine_config
from etazero.runtime import run_training
from etazero.training import load_checkpoint
from etazero.storage import sha256


def small_config(algorithm, nonroot):
    if algorithm == 'muzero':
        from test_muzero_pipeline import small_config as muzero_config
        c=muzero_config()
        c['network']['architecture']='resnet'
        c['muzero_training']=dict(auxiliary_losses=True,katago_optimizer=False,learning_rate=.001,weight_decay=.0003)
        c['optimizer']['kind']='adamw'
    else:
        c=load_config(CONFIGS/'smoke_test')
    c['agent'].update(root_search_algo='gumbel',nonroot_search_algo=nonroot)
    c['gumbel']={**GUMBEL_DEFAULTS,'max_num_considered_actions':4}
    c['search'].update(reuse_tree=False,full_search_visits=12,cheap_search_visits=3,cheap_search_probs=.5)
    c['graph_search']['use_graph_search']=False
    c['symmetry']['root_num_symmetries_to_sample']=1
    c['parallelism'].update(game_threads=2,search_threads=2)
    c['inference'].update(max_batch=8,server_threads=1,cache_entries=0)
    c['opening']['probability']=0;c['hex_opening']['probability']=0
    c['policy_init'].update(policy_init=False,policy_after=False,policy_on_failure=False)
    c['side_positions']['side_position_prob']=.4
    c['training'].update(skip_validation=True,compile=False)
    c['replay']['keep_target_rows']='all'
    validate(c)
    return c


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
@pytest.mark.parametrize('nonroot',['puct','gumbel'])
def test_native_targets_side_reanalysis_and_unroll(tmp_path,algorithm,nonroot):
    c=small_config(algorithm,nonroot)
    c['gumbel']['action_selection']='visit'
    c['environment'].update(sizes='5',size_weights='1',rules='hex',rule_weights='1')
    c['reanalysis']=dict(use_reanalyze=True,reanalyze_prop=1,reanalyze_policy_surprise_weight=1,
                       reanalyze_value_surprise_weight=0,reanalyze_surprise_exponent=1,reanalyze_use_outcome_targets=False)
    validate(c); path=tmp_path/'effective.cfg';write_native(c,path)
    subprocess.run([str(ROOT/'build/etazero'),'selfplay','--config',str(path),'--evaluator','random',
                    '--model','random','--model-id','random','--device','cpu','--games','4',
                    '--output',str(tmp_path/'raw'),'--run-id','test','--attempt-id','test','--config-id','test',
                    '--source-id','test','--iteration','1','--worker','0','--seed','734'],
                   check=True,capture_output=True,text=True,timeout=60)
    unvisited=side=reanalyzed=0
    for path in (tmp_path/'raw').glob('*.npz'):
        a=read_raw(path); m=metadata(a); v=training_view(a)
        assert m['root_search_algo']=='gumbel' and m['nonroot_search_algo']==nonroot
        assert np.isfinite(v['policy']).all()
        unvisited+=((a['policies']>0)&(a['visits']==0)).sum()
        side+=len(a['side_game_indices']);reanalyzed+=a['reanalyzed'].sum()
        # A positive completed policy must still have zero mass outside actual legal moves.
        if algorithm=='muzero':
            assert v['policy'].shape[1]==c['unroll']['steps']+1
            assert (a['trajectory_policy'][a['train_mask'].astype(bool)].sum(1)>0).all()
    assert unvisited>0 and side>0 and reanalyzed>0


def test_illegal_dense_target_is_rejected():
    from test_python import winning_record
    a,m=winning_record(0);m.update(root_search_algo='gumbel',nonroot_search_algo='puct')
    a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    a['policies'][0,5]=1 # padded point, despite allowing unvisited legal actions
    from etazero.data import validate_raw
    with pytest.raises(ValueError,match='occupied/padded'):
        validate_raw(a)


def test_gumbel_config_and_engine_boundaries():
    c=small_config('alphazero','puct')
    for group,key in [('search','reuse_tree'),('graph_search','use_graph_search')]:
        invalid=copy.deepcopy(c);invalid[group][key]=True
        with pytest.raises(ValueError,match='Gumbel requires'):validate(invalid)
    for key,value in [('max_num_considered_actions',0),('c_scale',0),('noise_scale',-1),('action_selection','unknown')]:
        invalid=copy.deepcopy(c);invalid['gumbel'][key]=value
        with pytest.raises(ValueError,match='Gumbel'):validate(invalid)
    e=load_engine_config(CONFIGS/'baseline')
    e['analysis'].update(root_search_algo='gumbel',nonroot_search_algo='gumbel',reuse_tree=False,use_graph_search=False)
    validate_engine_config(e)
    e['analysis']['root_search_algo']='puct'
    with pytest.raises(ValueError,match='not an allowed'):validate_engine_config(e)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA acceptance')
@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
@pytest.mark.parametrize('nonroot',['puct','gumbel'])
def test_cuda_pipeline_resume_analysis_and_match(tmp_path,algorithm,nonroot):
    torch.set_num_threads(1);torch._dynamo.reset()
    c=small_config(algorithm,nonroot)
    rule='hex' if nonroot=='gumbel' else 'renju'
    c['environment'].update(sizes='5',size_weights='1',rules=rule,rule_weights='1')
    c['training']['compile']=True
    c['gumbel']['action_selection']='gumbel'
    validate(c);root=tmp_path/'run'
    state=run_training(root,c,ROOT/'build/etazero',max_iteration=2)
    saved=load_checkpoint(root,state['checkpoint'],c)
    assert saved['optimizer_steps']>0
    files={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')}
    assert files
    for path in files:
        a=read_raw(path);assert metadata(a)['root_search_algo']=='gumbel'
    assert run_training(root,c,ROOT/'build/etazero',resume=True,max_iteration=2)==state
    assert all(sha256(p)==digest for p,digest in files.items())
    if nonroot=='gumbel':
        from test_gpu import assert_compiled_partial_resume
        assert_compiled_partial_resume(tmp_path/'partial_resume',root,c)
    from etazero.analysis import analyze
    evaluation=load_engine_config(CONFIGS/'muzero' if algorithm=='muzero' else CONFIGS/'baseline')
    evaluation['analysis'].update(root_search_algo='gumbel',nonroot_search_algo=nonroot,reuse_tree=False,
                                  use_graph_search=False,root_num_symmetries_to_sample=1,
                                  visits=12,board_size=5,rule=rule,search_threads=2,max_batch=8,cache_entries=0)
    _, result=analyze(evaluation,ROOT/'build/etazero',root,moves='0,6')
    assert result['result']['root_visits']==12 and result['result']['action'] not in (0,6)
    if nonroot=='gumbel':
        from etazero.arena import single_match
        match=load_engine_config(CONFIGS/'muzero' if algorithm=='muzero' else CONFIGS/'baseline',match=True)
        match['match'].update(root_search_algo='gumbel',nonroot_search_algo=nonroot,reuse_tree=False,
                              use_graph_search=False,root_num_symmetries_to_sample=1,visits=7,
                              board_size=5,rule=rule,game_threads=2,search_threads=2,max_batch=8,cache_entries=0)
        model=root/state['model']['path']
        _, games=single_match(match,ROOT/'build/etazero',model,model,tmp_path/'match',games=4)
        assert len(games)==4
