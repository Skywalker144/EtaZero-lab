from etazero.schema import GLOBALS
"""Independent finite-trajectory, gradient, validity and storage contract checks."""
from config_samples import CONFIGS
import copy
from pathlib import Path
import json
import math
import numpy as np
import pytest
import torch
from etazero.data import trajectory_td_targets, training_view, validate_raw
from etazero.network import auxiliary_losses, error_variance, make_network, inference_network, TrainingForward
from etazero.schema import RAW_DTYPES, pack_observations
from test_python import winning_record, config


def test_td_finite_horizon_fixed_perspective_and_actual_area():
    values = np.array([[.6,.1,.3],[.2,.3,.5],[.1,.4,.5]],np.float32)
    players = np.array([1,-1,1])
    terminal = np.array([0.,1.,0.])
    for area in (25,121,225):
        td = trajectory_td_targets(values,players,terminal,area)
        # Direct finite sum independent of the production reverse recurrence.
        fixed = [values[0],values[1][::-1],values[2]]
        for t in range(3):
            for k,c in enumerate((.176,.056,.016)):
                a=1/(1+c*area)
                expected=(1-a)**(3-t)*terminal
                for j in range(t,3):
                    expected=expected+a*(1-a)**(j-t)*fixed[j]
                if players[t]==-1: expected=expected[::-1]
                np.testing.assert_allclose(td[t,k],expected,rtol=1e-6,atol=1e-7)
    assert not np.allclose(trajectory_td_targets(values,players,terminal,25),
                           trajectory_td_targets(values,players,terminal,225))


def test_error_source_surrogate_gradient_floor():
    raw=torch.tensor([-100.,0.,4.],requires_grad=True)
    variance=error_variance(raw)
    np.testing.assert_allclose(variance.detach(),[.25*math.log1p(math.exp(x/2))**2 for x in (-100,0,4)],rtol=1e-6,atol=1e-8)
    variance.sum().backward()
    np.testing.assert_allclose(raw.grad,[.25*(.05+.95/(1+math.exp(-x))) for x in (-100,0,4)],rtol=1e-6)


@pytest.mark.parametrize('disable',[False,True])
def test_auxiliary_probabilities_gradients_and_gates(disable):
    logits=torch.zeros(2,6,4,requires_grad=True)
    obs=torch.zeros(2,5,2,2);obs[:,0].flatten(1)[:,:3]=1
    policy=torch.tensor([[0.,10.,0.,0.],[0.,0.,10.,0.]])
    target=torch.tensor([[0.,1.,0.],[.5,0.,.5]])
    td_target=target[:,None].expand(-1,3,-1)
    td=torch.zeros(2,3,3,requires_grad=True)
    raw=torch.zeros(2,requires_grad=True)
    gate=torch.tensor([1.,0.])
    result=auxiliary_losses(logits,td,raw,obs,policy,target,td_target,gate,disable)
    entropy=np.array([0.,math.log(2)])
    assert [float(x.detach()) for x in result[:3]]==pytest.approx([.72*(math.log(3)-entropy.mean())]*3)
    long_weight=.5 if disable else .25/2
    short_weight=.5 if disable else 1/(1+math.exp(4.5))/2
    assert float(result[3].detach())==pytest.approx(.1*long_weight*math.log(3))
    assert float(result[4].detach())==pytest.approx(.2*short_weight*math.log(3))
    e=.25*math.log(2)**2
    assert float(result[5].detach())==pytest.approx((e-1e-8)**2/2,rel=2e-6)
    # Error target detaches TD prediction, optimistic weights detach both predictions.
    result[5].backward(retain_graph=True)
    assert td.grad is None and raw.grad[0]!=0 and raw.grad[1]==0
    raw.grad=None
    (result[3]+result[4]).backward(retain_graph=True)
    assert raw.grad is None and td.grad is None
    assert not logits.grad[:,:,3].any()
    if not disable: assert not logits.grad[1,4:6].any()
    sum(result[:3]).backward()
    expected=.72*(np.full((2,3,3),1/3)-td_target.numpy())/2
    np.testing.assert_allclose(td.grad,expected,rtol=1e-6,atol=1e-8)


def test_side_has_own_search_value_and_no_complete_game_targets():
    a,m=winning_record(2)
    full=training_view(a)
    a['side_observations']=a['observations'][2:3].copy()
    a['side_globals']=a['globals'][2:3].copy()
    a['side_players']=a['players'][2:3].copy()
    a['side_game_indices']=np.array([0],np.int32)
    a['side_policies']=a['policies'][2:3].copy()
    a['side_visits']=a['visits'][2:3].copy()
    a['side_q_visits']=a['q_visits'][2:3].copy()
    a['side_q_values']=np.repeat(a['q_values'][2:3],2,axis=0)
    a['side_wdl']=np.array([[.2,.3,.5]],np.float32)
    a['side_row_repeats']=np.array([2],np.int32)
    a['side_target_weights']=np.array([2.],np.float32)
    a['side_forbidden_input']=np.array([1,0],np.uint8)
    m['rows']+=2;a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    validate_raw(a)
    view=training_view(a)
    for key in full: np.testing.assert_array_equal(view[key][:-2],full[key])
    np.testing.assert_array_equal(view['value'][-2:],np.repeat(a['side_wdl'],2,axis=0))
    np.testing.assert_array_equal(view['td_value'][-2:],np.tile(a['side_wdl'][:,None],(2,3,1)))
    assert not view['full_game_weight'][-2:].any() and not view['opponent_policy_weight'][-2:].any()
    assert view['globals'][-2:,3].tolist()==[1,0]
    assert (view['opponent_policy'][-2:]==1).all()
    # Corrupting the parent outcome never supplies a label to side positions.
    a['winners'][0]=-1
    np.testing.assert_array_equal(training_view(a)['value'][-2:],view['value'][-2:])


def test_all_heads_and_inference_contract():
    from etazero.config import ROOT,load_config
    c=load_config(CONFIGS / 'smoke_test')
    model=make_network(c)
    obs=torch.zeros(2,5,6,6);obs[:,0,:5,:5]=1
    globals=torch.zeros(2,len(GLOBALS))
    policy,value,td,raw=model.forward_all(obs,globals)
    assert policy.shape==(2,6,36) and value.shape==(2,3) and td.shape==(2,3,3) and raw.shape==(2,)
    target=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    p=torch.zeros(2,36);p[:,0]=10
    out=TrainingForward(model,8,False)(obs,globals,p,p,torch.ones(2),target,target[:,None].expand(-1,3,-1),torch.ones(2))
    assert len(out)==12 and torch.isfinite(torch.stack(out)).all()
    out[0].backward()
    assert model.value_head.out.weight.grad[3:].abs().sum()>0
    assert model.policy_head.out.weight.grad[4:].abs().sum()>0
    model.eval();scripted=torch.jit.script(inference_network(model))
    with torch.no_grad():
        p,v,t,e=model.forward_all(obs,globals)
        exported=scripted(obs,globals)
    expected=(p[:,0],v,p[:,5],.5*torch.nn.functional.softplus(e/2))
    for a,b in zip(exported,expected): torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)


def test_native_side_writer_catalog_shuffle_reader_roundtrip(tmp_path):
    import subprocess
    from etazero.config import ROOT,load_config
    from etazero.data import read_raw,Catalog
    from etazero.reader import BatchReader
    from etazero.shuffle import build_snapshot
    from etazero.storage import save_json,sha256
    binary=ROOT/'build/record_contract_test'
    if not binary.exists(): pytest.skip('Build C++ record_contract_test first')
    directory=tmp_path/'selfplay/worker'
    subprocess.run([str(binary),str(directory)],check=True,capture_output=True,text=True)
    raw=read_raw(next(directory.glob('*.npz')));view=training_view(raw)
    assert len(view['value'])==11 and raw['policies'].dtype==np.int16
    assert not view['full_game_weight'][-2:].any()
    catalog=Catalog(tmp_path,'test','config')
    try:
        catalog.scan({1:'model'})
        assert catalog.counts()==(11,1)
        assert catalog.previous_rows_per_game(2)==11.
        c=load_config(CONFIGS / 'smoke_test')
        c['replay'].update(min_rows=11,keep_target_rows='all')
        c['shuffle'].update(bucket_rows=11,training_shard_rows=11)
        info=build_snapshot(tmp_path,1,catalog.entries(),c)
    finally: catalog.close()
    reader=BatchReader(tmp_path/'snapshots'/info,11,1,1)
    try: batch=reader.next()
    finally: reader.close()
    assert sorted(batch['full_game_weight'].tolist())==[0.,0.]+[1.]*9
    side=batch['full_game_weight']==0
    np.testing.assert_allclose(batch['value'][side],[[.2,.3,.5]]*2)
    np.testing.assert_allclose(batch['td_value'][side],[[[.2,.3,.5]]*3]*2)


def test_native_empty_side_arrays_keep_valid_zip_crc(tmp_path):
    import subprocess
    from etazero.config import ROOT
    from etazero.data import read_raw
    binary=ROOT/'build/record_contract_test'
    if not binary.exists(): pytest.skip('Build C++ record_contract_test first')
    directory=tmp_path/'worker'
    subprocess.run([str(binary),str(directory),'no-side'],check=True,capture_output=True,text=True)
    raw=read_raw(next(directory.glob('*.npz')))
    assert len(raw['side_game_indices'])==0 and len(training_view(raw)['value'])==9


@pytest.mark.parametrize('symmetry',range(8))
def test_auxiliary_scalars_and_policy_quantization_survive_d4(symmetry):
    from etazero.schema import unpack_observations
    from etazero.symmetry import augment_batch,apply_symmetry
    raw,_=winning_record()
    view=training_view(raw)
    view['obs']=unpack_observations(view['obs'],6)
    batch={k:torch.from_numpy(v).float() for k,v in view.items()}
    transformed=augment_batch(batch,symmetry)
    for key in ('value','td_value','full_game_weight','opponent_policy_weight','globals'):
        torch.testing.assert_close(transformed[key],batch[key],rtol=0,atol=0)
    for key in ('policy','opponent_policy'):
        torch.testing.assert_close(transformed[key],apply_symmetry(batch[key].reshape(-1,6,6),symmetry).flatten(1),rtol=0,atol=0)
        torch.testing.assert_close(transformed[key].sum(1),batch[key].sum(1),rtol=0,atol=0)


@pytest.mark.parametrize('use_outcome',[False,True])
@pytest.mark.parametrize('compact',[False,True])
def test_pda_globals_and_reanalysis_outcome_gate(use_outcome,compact):
    from test_python import compact_search
    a,m=winning_record(2)
    players=a['players'].astype(np.float32)
    a['globals'][:,4]=1;a['globals'][:,5]=1.5*players
    if compact:
        # An omitted earlier position and repeated reanalysis rows distinguish
        # compact search indices from full trajectory and output-row indices.
        a['row_repeats'][1]=0;a['row_repeats'][2]=3
        a['target_weights']=a['row_repeats'].astype(np.float32)
        compact_search(a)
        m['rows']=10;a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    original=training_view(a)
    a['cheap_search'][2]=1;a['reanalyzed'][2]=1
    a['reanalysis_used_outcome'][2]=use_outcome
    a['reanalysis_original_visits'][2]=4
    a['reanalysis_policy_surprise'][2]=2;a['reanalysis_value_surprise'][2]=.3
    sample=1 if compact else 2
    if not use_outcome:
        a['opponent_policy_weights'][sample]=0;a['opponent_policies'][sample]=1
    validate_raw(a,m)
    view=training_view(a)
    turns=[0,2,2,2,3,4,5,6,7,8] if compact else list(range(9))
    expected_gate=[1,int(use_outcome),int(use_outcome),int(use_outcome),1,1,1,1,1,1] if compact else [1,1,int(use_outcome),1,1,1,1,1,1]
    np.testing.assert_array_equal(view['full_game_weight'],expected_gate)
    np.testing.assert_array_equal(view['globals'][:,4],1)
    np.testing.assert_array_equal(view['globals'][:,5],a['globals'][turns,5])
    assert view['opponent_policy_weight'][2]==int(use_outcome)
    np.testing.assert_array_equal(view['value'][2],[1,0,0])
    # The auxiliary outcome gate leaves main WDL, TD and Q supervision intact.
    for key in ('value','td_value','policy','q_values','q_visits'):
        np.testing.assert_array_equal(view[key],original[key])
    assert not np.array_equal(view['td_value'][2],np.repeat(a['search_wdl'][2,None],3,axis=0))
    broken=copy.deepcopy(a);broken['globals'][3,5]*=-1
    with pytest.raises(ValueError,match='PDA conditioning'):validate_raw(broken,m)
    broken=copy.deepcopy(a);broken['globals'][:,4]=0
    with pytest.raises(ValueError,match='PDA conditioning'):validate_raw(broken,m)


@pytest.mark.parametrize('use_outcome',[False,True])
@pytest.mark.parametrize('disable',[False,True])
def test_reanalysis_outcome_gate_reaches_loss_gradients(use_outcome,disable):
    from etazero.network import losses
    from etazero.schema import unpack_observations
    a,m=winning_record(2)
    a['cheap_search'][2]=a['reanalyzed'][2]=1
    a['reanalysis_used_outcome'][2]=use_outcome
    a['reanalysis_original_visits'][2]=4
    if not use_outcome:
        a['opponent_policy_weights'][2]=0;a['opponent_policies'][2]=1
    validate_raw(a,m)
    view=training_view(a)
    obs=torch.from_numpy(unpack_observations(view['obs'],6)).float()
    batch={key:torch.from_numpy(value) for key,value in view.items() if key!='obs'}
    policy=torch.zeros(9,7,36,requires_grad=True)
    value=torch.zeros(9,3,requires_grad=True)
    td=torch.zeros(9,3,3,requires_grad=True)
    error=torch.zeros(9,requires_grad=True)
    result=losses(policy,value,td,error,obs,batch['policy'],batch['opponent_policy'],
                  batch['opponent_policy_weight'],batch['value'],batch['td_value'],
                  batch['full_game_weight'],8,disable,batch['q_values'],batch['q_visits'])
    result[0].backward()
    # Check the row's actual parameter gradients, rather than just its flag.
    assert value.grad[2].abs().sum()>0 and td.grad[2].abs().sum()>0
    assert policy.grad[2,0].abs().sum()>0 and policy.grad[2,6].abs().sum()>0
    if use_outcome:
        assert error.grad[2]!=0
    else:
        assert error.grad[2]==0 and not policy.grad[2,1].any()
    if use_outcome or disable:
        assert (policy.grad[2,4:6].abs().sum(1)>0).all()
    else:
        assert not policy.grad[2,4:6].any()
    # Neighboring ordinary rows remain supervised in the same mixed batch.
    assert error.grad[0]!=0 and (policy.grad[0,4:6].abs().sum(1)>0).all()


def test_sampling_configuration_source_defaults_and_reanalysis_required_parameters(tmp_path,config):
    from etazero.config import load_config,ROOT,validate
    assert load_config(CONFIGS / 'baseline')['pda']==dict(normal_asymmetric_playout_prob=.01,max_asymmetric_ratio=8)
    assert config['pda']['normal_asymmetric_playout_prob']==0
    # Project sampling rates are intentional experiment settings, independent
    # of the source algorithm's defaults.
    assert config['side_positions']['side_position_prob']==.04
    assert config['reanalysis']=={'use_reanalyze':False}
    current=tmp_path/'cfg';current.mkdir();(current/'run.cfg').write_text(f'[run]\nextends=smoke_test\nrun_dir={tmp_path/"run"}\n')
    (current/'selfplay.cfg').write_text('[reanalysis]\nuse_reanalyze=true\n')
    with pytest.raises(ValueError,match='Missing required key: reanalysis.reanalyze_prop'):load_config(current)
    (current/'selfplay.cfg').write_text('[reanalysis]\nuse_reanalyze=true\nreanalyze_prop=0.7\nreanalyze_policy_surprise_weight=1\nreanalyze_value_surprise_weight=2\nreanalyze_surprise_exponent=0.5\nreanalyze_use_outcome_targets=false\n')
    c=load_config(current);assert c['reanalysis']['reanalyze_prop']==.7
    for section,key,value in [('pda','normal_asymmetric_playout_prob',1.01),('pda','max_asymmetric_ratio',.9),('pda','max_asymmetric_ratio',101),('side_positions','side_position_prob',1.01),('reanalysis','reanalyze_surprise_exponent',11)]:
        broken=copy.deepcopy(c);broken[section][key]=value
        with pytest.raises(ValueError):validate(broken)


def test_hint_and_fork_raw_prefix_contract():
    a,m=winning_record()
    # Use the two opening moves of the independent completed trajectory as an external hint prefix.
    a['initial_position_moves'][0]=a['opening_moves'][0]=2
    a['initial_position_kind'][0]=1;a['hint_actions'][0]=2
    a['train_mask'][:2]=a['row_repeats'][:2]=0;a['target_weights'][:2]=0
    a['simulations'][:2]=0;a['temperatures'][:2]=0
    from test_python import compact_search
    compact_search(a);m['rows']=7
    a['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
    validate_raw(a)
    assert len(training_view(a)['value'])==7
    for field,value in [('initial_position_moves',1),('policy_moves',1),('hint_actions',0),('hint_actions',5)]:
        broken=copy.deepcopy(a);broken[field][0]=value
        with pytest.raises(ValueError):validate_raw(broken)
    fork=copy.deepcopy(a);fork['initial_position_kind'][0]=4;fork['hint_actions'][0]=-1;validate_raw(fork)
    fork['globals'][:,4]=1;fork['globals'][:,5]=.5*fork['players']
    with pytest.raises(ValueError,match='fork positions'):validate_raw(fork)


def test_hint_game_fork_defaults_bounds_and_input_identity(tmp_path):
    from etazero.config import load_config,validate,fingerprint
    baseline=load_config(CONFIGS / 'baseline')
    assert baseline['hint_positions']['hint_positions_prob']==0
    assert baseline['game_forks']==dict(early_fork_game_prob=.04,fork_game_prob=.01,
        early_fork_game_expected_move_prop=.025,fork_game_min_choices=3,early_fork_game_max_choices=12,fork_game_max_choices=36)
    for field,value in [('early_fork_game_prob',1.1),('fork_game_min_choices',0),('early_fork_game_max_choices',2),('fork_game_max_choices',101),('early_fork_game_expected_move_prop',1.1)]:
        bad=copy.deepcopy(baseline);bad['game_forks'][field]=value
        with pytest.raises(ValueError):validate(bad)
    bad=copy.deepcopy(baseline);bad['hint_positions']['hint_positions_prob']=1
    with pytest.raises(ValueError,match='requires positions_file'):validate(bad)
    configs=tmp_path/'configs';configs.mkdir();profile=configs/'hints';profile.mkdir()
    (profile/'run.cfg').write_text(f'[run]\nextends=smoke_test\nrun_dir={tmp_path/"run"}\n')
    (profile/'selfplay.cfg').write_text('[hint_positions]\nhint_positions_prob=1\npositions_file=positions.txt\n')
    (profile/'positions.txt').write_text('5 renju 1 12 2 0 5\n')
    one=load_config(profile);assert one['hint_positions']['positions_file']==str(profile/'positions.txt')
    (profile/'positions.txt').write_text('5 renju 1 13 2 0 5\n')
    two=load_config(profile);assert fingerprint(one)!=fingerprint(two)
