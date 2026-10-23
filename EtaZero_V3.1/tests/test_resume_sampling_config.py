"""Optimizer resume identity and Python/native sampling validation agreement."""
import copy
import subprocess

import pytest
from config_samples import CONFIGS
from etazero.config import ROOT, load_config, resume_config, validate, write_native
from etazero.data import read_raw
from etazero.training import initialize, load_checkpoint


@pytest.mark.parametrize('lookahead_print', [False, True])
def test_checkpoint_resume_rejects_changed_norm_sampling(tmp_path, lookahead_print):
    c=load_config(CONFIGS/'smoke_test')
    c['devices'].update(train='cpu',selfplay='cpu')
    c['optimizer'].update(norm_only_at_print=False,lookahead_alpha=.5,
                          lookahead_print=lookahead_print)
    checkpoint=initialize(tmp_path,c)
    saved=load_checkpoint(tmp_path,checkpoint,c)
    assert saved['resume_config']['optimizer']['lookahead_print']==lookahead_print
    changed=copy.deepcopy(c)
    changed['optimizer']['lookahead_print']=not lookahead_print
    assert resume_config(changed)!=resume_config(c)
    with pytest.raises(ValueError,match='Checkpoint identity/configuration mismatch'):
        load_checkpoint(tmp_path,checkpoint,changed)
    # Pure logging cadence remains adjustable on resume.
    changed=copy.deepcopy(c);changed['training']['log_every']+=1
    assert load_checkpoint(tmp_path,checkpoint,changed)['id']==saved['id']


def sampling_config(enabled):
    c=load_config(CONFIGS/'smoke_test')
    c['opening'].update(probability=float(enabled),avg_dist_factor=.8,balance_exponent=1,
                        rejection_probability=0,rejection_probability_fallback=0,max_tries=3)
    c['policy_init'].update(policy_init=enabled,policy_after=enabled,policy_on_failure=enabled,
                            policy_init_mean=0,policy_temperature=1)
    c['game_forks'].update(early_fork_game_prob=.5 if enabled else 0,fork_game_prob=.5 if enabled else 0,
                          early_fork_game_expected_move_prop=.1,fork_game_min_choices=1,
                          early_fork_game_max_choices=2,fork_game_max_choices=3)
    return c


def native_selfplay(tmp_path,c):
    write_native(c,tmp_path/'effective.cfg')
    return subprocess.run([str(ROOT/'build/etazero'),'selfplay','--config',str(tmp_path/'effective.cfg'),
                           '--evaluator','random','--model-id','random:config','--device','cpu',
                           '--games','1','--output',str(tmp_path/'raw'),'--run-id','config-test',
                           '--attempt-id','config-test','--config-id','config-test','--source-id','config-test',
                           '--iteration','0','--worker','0','--seed','1'],
                          capture_output=True,text=True,timeout=30)


@pytest.mark.parametrize('field', ['cheap','reduced','playouts','full'])
def test_pda_budget_bounds_fail_before_any_game(tmp_path,field):
    c=load_config(CONFIGS/'smoke_test')
    c['pda'].update(normal_asymmetric_playout_prob=.000001,max_asymmetric_ratio=8)
    c['search'].update(full_search_visits=100,cheap_search_visits=30,cheap_search_probs=.5,max_playouts=2**31-1)
    c['reduce_visits'].update(reduce_visits=True,reduced_visits_min=30)
    if field=='cheap':c['search']['cheap_search_visits']=20
    if field=='reduced':c['reduce_visits']['reduced_visits_min']=20
    if field=='playouts':c['search'].update(cheap_search_probs=0,max_playouts=20);c['reduce_visits']['reduce_visits']=False
    if field=='full':c['search'].update(full_search_visits=20,cheap_search_probs=0);c['reduce_visits']['reduce_visits']=False
    with pytest.raises(ValueError, match='PDA'):validate(c)
    result=native_selfplay(tmp_path,c)
    assert result.returncode!=0 and 'PDA' in result.stderr,result.stderr
    assert not list((tmp_path/'raw').rglob('*.npz'))


@pytest.mark.parametrize('ratio,probability,minimum', [(8,1,21),(1,1,2),(100,0,2)])
def test_pda_budget_rounding_and_disabled_boundaries(tmp_path,ratio,probability,minimum):
    c=sampling_config(False)
    c['pda'].update(normal_asymmetric_playout_prob=probability,max_asymmetric_ratio=ratio)
    c['environment'].update(sizes='5',size_weights='1',rules='freestyle',rule_weights='1')
    c['parallelism'].update(game_threads=1,search_threads=1)
    c['search'].update(full_search_visits=minimum,cheap_search_visits=minimum)
    c['reduce_visits']['reduced_visits_min']=minimum
    validate(c)
    result=native_selfplay(tmp_path,c)
    assert result.returncode==0,result.stderr


def all_cheap_pda_config():
    c=sampling_config(False)
    c['environment'].update(sizes='5',size_weights='1',rules='freestyle',rule_weights='1')
    c['parallelism'].update(game_threads=1,search_threads=1)
    c['side_positions']['side_position_prob']=0
    c['pda'].update(normal_asymmetric_playout_prob=1,max_asymmetric_ratio=8)
    c['search'].update(full_search_visits=100,cheap_search_visits=30,
                       cheap_search_probs=1,cheap_search_target_weight=.25)
    c['reduce_visits'].update(reduce_visits=True,reduced_visits_min=2)
    c['reanalysis'].update(use_reanalyze=False,reanalyze_prop=0,
                           reanalyze_policy_surprise_weight=1,reanalyze_value_surprise_weight=0,
                           reanalyze_surprise_exponent=1,reanalyze_use_outcome_targets=False)
    return c


@pytest.mark.parametrize('full,playouts,reanalysis', [
    (100,2**31-1,False), (2**31-1,2**31-1,False),
    (100,2**31-2,False), (100,2**31-1,True)])
def test_pda_ignores_unreachable_full_and_reduced_budgets(tmp_path,full,playouts,reanalysis):
    c=all_cheap_pda_config()
    c['search'].update(full_search_visits=full,max_playouts=playouts)
    c['reanalysis']['use_reanalyze']=reanalysis  # Enabled with zero proportion still cannot force full.
    validate(c)
    result=native_selfplay(tmp_path,c)
    assert result.returncode==0,result.stderr
    shards=list((tmp_path/'raw').glob('*.npz'))
    assert shards
    for path in shards:
        raw=read_raw(path)
        assert raw['cheap_search'].all() and not raw['reanalyzed'].any()


@pytest.mark.parametrize('branch', ['ordinary','hint','reanalysis'])
@pytest.mark.parametrize('budget', ['reduced'])
def test_pda_checks_full_budgets_when_the_branch_is_reachable(tmp_path,branch,budget):
    c=all_cheap_pda_config()
    if budget=='full':
        # Keep the hint's fourfold cap representable before PDA scaling.
        c['search']['full_search_visits']=400_000_000 if branch=='hint' else 2**31-1
        c['reduce_visits']['reduced_visits_min']=30
    if branch=='ordinary':c['search']['cheap_search_probs']=.999999
    if branch=='hint':
        c['hint_positions'].update(hint_positions_prob=.000001,positions_file='unused-before-budget-validation')
    if branch=='reanalysis':c['reanalysis'].update(use_reanalyze=True,reanalyze_prop=.000001)
    with pytest.raises(ValueError,match='PDA'):validate(c)
    result=native_selfplay(tmp_path,c)
    assert result.returncode!=0 and 'PDA' in result.stderr,result.stderr
    assert not list((tmp_path/'raw').rglob('*.npz'))


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('section,key,value',[
    ('opening','probability',-.1),('opening','probability',1.1),
    ('opening','rejection_probability',-.1),('opening','rejection_probability',1.1),
    ('opening','rejection_probability_fallback',-.1),('opening','rejection_probability_fallback',1.1),
    ('opening','avg_dist_factor',-1),('opening','avg_dist_factor',101),
    ('opening','balance_exponent',-1),('opening','balance_exponent',101),
    ('opening','max_tries',0),('opening','max_tries',1001),
    ('policy_init','policy_init_mean',-1),('policy_init','policy_init_mean',101),
    ('policy_init','policy_temperature',0),('policy_init','policy_temperature',5.1),
    ('game_forks','early_fork_game_prob',-.1),('game_forks','fork_game_prob',1.1),
    ('game_forks','early_fork_game_expected_move_prop',-.1),('game_forks','early_fork_game_expected_move_prop',1.1),
    ('game_forks','fork_game_min_choices',0),('game_forks','fork_game_min_choices',3),
    ('game_forks','early_fork_game_max_choices',101),('game_forks','fork_game_max_choices',101),
    *[(section,key,float('nan')) for section,key in (
        ('opening','probability'),('opening','rejection_probability'),('opening','rejection_probability_fallback'),
        ('opening','avg_dist_factor'),('opening','balance_exponent'),
        ('policy_init','policy_init_mean'),('policy_init','policy_temperature'),
        ('game_forks','early_fork_game_expected_move_prop'))],
])
def test_python_and_native_reject_invalid_sampling_fields(tmp_path,enabled,section,key,value):
    c=sampling_config(enabled);c[section][key]=value
    with pytest.raises(ValueError,match=r'(?i)opening|policy[ _]init|fork'):
        validate(c)
    result=native_selfplay(tmp_path,c)
    assert result.returncode!=0
    assert 'Invalid opening' in result.stderr or 'Invalid early/game fork' in result.stderr,result.stderr


def test_disabled_sampling_accepts_valid_boundaries_and_generates_games(tmp_path):
    c=sampling_config(False)
    c['opening'].update(avg_dist_factor=100,balance_exponent=100,max_tries=1000,
                        rejection_probability=1,rejection_probability_fallback=1)
    c['policy_init'].update(policy_init_mean=100,policy_temperature=5)
    c['game_forks'].update(early_fork_game_expected_move_prop=1,fork_game_min_choices=100,
                          early_fork_game_max_choices=100,fork_game_max_choices=100)
    c['environment'].update(sizes='5',size_weights='1',rules='freestyle',rule_weights='1')
    c['search']['cheap_search_probs']=0
    c['parallelism'].update(game_threads=1,search_threads=1)
    validate(c)
    result=native_selfplay(tmp_path,c)
    assert result.returncode==0,result.stderr
    shards=list((tmp_path/'raw').glob('*.npz'))
    assert shards
    for path in shards:
        raw=read_raw(path)
        assert not raw['balanced_moves'].any() and not raw['policy_moves'].any()
        assert not raw['initial_position_kind'].any()


@pytest.mark.parametrize('hint', [False,True])
def test_pda_accepts_source_int64_caps_and_saturating_hint(hint):
    c=all_cheap_pda_config()
    c['search'].update(full_search_visits=2**50,max_playouts=2**50,cheap_search_probs=.5)
    c['reduce_visits']['reduced_visits_min']=30
    if hint:c['hint_positions'].update(hint_positions_prob=.000001,positions_file='unused')
    validate(c)
    c['search']['full_search_visits']=2**50+1
    with pytest.raises(ValueError):validate(c)
