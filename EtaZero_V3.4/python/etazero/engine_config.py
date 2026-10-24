"""Shared engine settings and independent analysis/match profiles.

Profiles explicitly include their engine and parent profiles. The selected mode
is flattened once for native consumers and provenance; no training files are read.
"""
import math
import os
from pathlib import Path
from etazero.config import PATTERN_DEFAULTS, resolve_pattern_corpora, HUMAN_SL_DEFAULTS, TIME_CONTROL_DEFAULTS, ROOT, boolean, FIELDS, SEARCH_PARAMETERS, validate_search_parameters, validate_hex_opening, validate_game_symmetry
from etazero.schema import RULES
from etazero.config_files import read_config_file
from etazero.config import GUMBEL_FIELDS, GUMBEL_DEFAULTS, validate_gumbel, validate_search_combination

ENGINE_FIELDS = dict(go_rules=str, komi=float, playout_doubling_advantage=float,playout_doubling_advantage_player=str,seed=int, board_size=int, rule=str, device=str, cpu_threads=int,
              visits=int, search_threads=int, c_puct=float, virtual_loss=float, reuse_tree=boolean,
              temperature=float, temperature_early=float, max_batch=int, server_threads=int, cache_entries=int,
              inference_precision=str, batch_wait_us=int, muzero_queue_capacity=int,nn_mutex_pool_power_of_two=int,
              use_fpu=boolean, fpu_reduction_max=float, root_fpu_reduction_max=float,
              fpu_parent_weight_by_visited_policy_pow=float, use_lcb=boolean,
              lcb_stdevs=float, min_visit_prop_for_lcb=float, policy_target_pruning=boolean, **SEARCH_PARAMETERS)
ENGINE_FIELDS.update(report_analysis_winrates_as=str,analysis_pv_len=int,always_compute_pass_alive_under_suicide_rules=str,exclude_territory_adjacent_to_atari=str,analysis_wide_root_noise=float,analysis_ignore_pre_root_history=boolean,root_noise_enabled=boolean,noise_fraction=float,dirichlet_total_concentration=float,shaped_dirichlet_noise=boolean,root_desired_per_child_visits_coeff=float,board_width=int,board_height=int,allow_resignation=boolean,resign_threshold=float,resign_consec_turns=int,delay_move_scale=float,delay_move_max=float,num_analysis_threads=int,pondering_enabled=boolean,root_search_algo=str, nonroot_search_algo=str,
                     **{'gumbel_'+key: value for key, value in GUMBEL_FIELDS.items()})


RESIGNATION_FIELDS = dict(allow_resignation=boolean,resign_threshold=float,resign_consec_turns=int)

# Setup::loadParams (analysis/match) at the pinned source. Execution fields with
# no source analogue are local defaults; explicit experiment profiles override all.
ENGINE_DEFAULTS = dict(report_analysis_winrates_as='SIDETOMOVE',always_compute_pass_alive_under_suicide_rules='auto',exclude_territory_adjacent_to_atari='auto',analysis_wide_root_noise=.04,analysis_ignore_pre_root_history=False,seed=0,board_size=19,rule='go',device='cuda:0',cpu_threads=1,
    use_non_buggy_lcb=True,root_noise_enabled=False,noise_fraction=.25,dirichlet_total_concentration=10.83,shaped_dirichlet_noise=True,root_desired_per_child_visits_coeff=0,board_width=0,board_height=0,delay_move_scale=0,delay_move_max=1000000,
    visits=2**50,c_puct=1,virtual_loss=3,reuse_tree=True,temperature=.1,temperature_early=.5,
    max_batch=16,server_threads=1,inference_precision='auto',batch_wait_us=0,
    muzero_queue_capacity=1024,use_fpu=True,fpu_reduction_max=.2,root_fpu_reduction_max=.1,
    fpu_parent_weight_by_visited_policy=True,fpu_parent_weight_by_visited_policy_pow=2,
    fpu_parent_weight=0,fpu_loss_prop=0,root_fpu_loss_prop=0,use_lcb=True,lcb_stdevs=5,
    min_visit_prop_for_lcb=.15,policy_target_pruning=True,c_puct_log=.45,c_puct_base=500,
    c_puct_stdev_prior=.4,c_puct_stdev_prior_weight=2,c_puct_stdev_scale=.85,
    value_weight_exponent=.25,chosen_move_subtract=0,chosen_move_prune=1,
    root_num_symmetries_to_sample=1,nn_randomize=True,nn_symmetry=0,nn_policy_temperature=1,
    root_policy_temperature=1,root_policy_temperature_early=1,temperature_halflife=19,
    temperature_only_below_prob=1,use_graph_search=True,graph_search_catch_up_leak_prob=0,
    use_uncertainty=True,uncertainty_coeff=.25,uncertainty_exponent=1,uncertainty_max_weight=8,
    policy_optimism=1,root_policy_optimism=.2,use_noise_pruning=True,noise_prune_utility_scale=.15,
    noise_pruning_cap=1e50,subtree_value_bias_factor=0,subtree_value_bias_free_prop=.8,
    subtree_value_bias_weight_exponent=.5,subtree_value_bias_table_num_shards=65536,
    max_playouts=2**50,max_time=1e20,playout_doubling_advantage=0,playout_doubling_advantage_player='black')

def load_engine_config(source, match=False, environ=None, *, overrides=None, gtp=False):
    result = _load_profile(source, match, environ, overrides=overrides, gtp=gtp)
    group=result['match' if match else 'analysis']
    base=Path(source);base=base if base.is_dir() else base.parent
    group['pattern_corpora']=resolve_pattern_corpora(group['pattern_corpora'],base)
    if group['auto_avoid_repeat_dir']:
        group['auto_avoid_repeat_dir']=str((base/group['auto_avoid_repeat_dir']).resolve())
    if group['human_model']:
        path=Path(group['human_model']);base=Path(source);base=base if base.is_dir() else base.parent
        group['human_model']=str((base/path).resolve() if not path.is_absolute() else path)
    if not 1<=group['num_analysis_threads']<=16384:raise ValueError('numAnalysisThreads must be in [1,16384]')
    validate_engine_config(result, match)
    return result


def load_match_opening_config(source, environ=None):
    """Read match's opening alone, without imposing match search or game budgets."""
    profiles = _load_profile(source, True, environ, groups=('opening',))
    validate_match_opening(profiles['opening'])
    return profiles['opening']


def load_match_hex_opening_config(source, environ=None):
    opening = _load_profile(source, True, environ, groups=('hex_opening',))['hex_opening']
    validate_hex_opening(opening,match=True)
    return opening


def _load_profile(source, match, environ, *, groups=None, overrides=None, gtp=False):
    source = Path(source)
    if not source.is_absolute() and not source.exists():
        source = ROOT/source
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f'Engine configuration source does not exist: {source}')
    section = 'match' if match else 'analysis'
    path = source/(section+'.cfg') if source.is_dir() else source
    fields = {section: dict(ENGINE_FIELDS, **({'games': int, 'game_threads': int} if match else {}))}
    if match:
        fields['opening'] = {**FIELDS['opening'][1], **FIELDS['policy_init'][1]}
        fields['hex_opening'] = FIELDS['hex_opening'][1]
    overrides = overrides or {}
    for group, keys in overrides.items():
        if group not in fields or set(keys) - set(fields[group]):
            raise ValueError(f'Unknown engine override: {group}')
    values, common = {}, {}

    def read(path, *, local=False):
        # Local engine fields override the selected mode in file/include order.
        entries = read_config_file(path, {'engine': ENGINE_FIELDS, **fields},
                                   section_aliases={'engine': section} if local else None)
        for (group, key), value in entries.items():
            if group == 'engine':
                common[key] = value
            elif group in fields:
                values[group, key] = value
    if not path.is_file():
        raise FileNotFoundError(f'Engine profile does not exist: {path}')
    read(path)
    for local in (path.parent/'engine.cfg.local', Path(str(path)+'.local')):
        if local.is_file():
            read(local, local=True)
    env = os.environ if environ is None else environ
    prefix = 'MATCH_' if match else 'ANALYSIS_'
    result = {}
    for group, keys in fields.items():
        if groups is not None and group not in groups:
            continue
        result[group] = {}
        for key, convert in keys.items():
            override = prefix+((group.upper()+'_') if group in ('opening','hex_opening') else '')+key.upper()
            raw = values.get((group, key), common.get(key) if group == section else None)
            if group == section and key in ENGINE_FIELDS:
                raw = env.get('ENGINE_'+key.upper(), raw)
            raw = env.get(override, raw)
            raw = overrides.get(group, {}).get(key, raw)
            if raw is None and group == "opening":
                defaults = {"policy_init": "false", "policy_temperature": "1", "policy_after": "true", "policy_on_failure": "true"}
                if key == "policy_init_mean" and not result[group]["policy_init"]:
                    raw = "0"
                else:
                    raw = defaults.get(key)
            if raw is None and group == section:
                if key in ENGINE_DEFAULTS: raw = str(ENGINE_DEFAULTS[key]).lower()
                elif key == 'analysis_pv_len': raw = '13' if gtp else '15'
                elif key == 'num_analysis_threads': raw = '1'
                elif key == 'allow_resignation': raw = 'false'
                elif key == 'resign_threshold': raw = '-0.95'
                elif key == 'resign_consec_turns': raw = '6'
                elif key in PATTERN_DEFAULTS: raw=str(PATTERN_DEFAULTS[key])
                elif key in HUMAN_SL_DEFAULTS: raw = str(HUMAN_SL_DEFAULTS[key]).lower() if isinstance(HUMAN_SL_DEFAULTS[key],bool) else str(HUMAN_SL_DEFAULTS[key])
                elif key in TIME_CONTROL_DEFAULTS: raw = str(TIME_CONTROL_DEFAULTS[key])
                elif key == 'graph_search_rep_bound': raw = '11'
                elif key == 'node_table_shards_power_of_two': raw = '16'
                elif key == 'min_playouts_per_thread': raw = '8' if group=='analysis' else '0'
                elif key == 'pondering_enabled': raw = 'false'
                elif key == 'root_symmetry_pruning': raw = 'true' if group=='analysis' else 'false'
                elif key == 'use_eval_cache': raw = 'false'
                elif key == 'eval_cache_min_visits': raw = '100'
                elif key == 'wide_root_noise': raw = '.04' if group=='analysis' and not gtp else '0'
                elif key in ('futile_visits_threshold','visit_cap_contempt','visit_cap_contempt_player'): raw = '0'
                elif key == 'cache_entries': raw = str(1 << (21 if match else 20 if gtp else 23))
                elif key == 'nn_mutex_pool_power_of_two': raw = '16' if gtp else '17'
                elif key in ('ignore_pre_root_history','ignore_all_history'): raw = 'false'
                elif key == 'go_rules': raw = 'chinese'
                elif key == 'komi': raw = '7.5'
                elif key in ('root_search_algo', 'nonroot_search_algo'):
                    raw = 'puct'
                elif key.startswith('gumbel_'):
                    value = GUMBEL_DEFAULTS[key.removeprefix('gumbel_')]
                    raw = str(0 if key == 'gumbel_noise_scale' else value).lower()
            if raw is None:
                raise ValueError(f'Missing engine field: {group}.{key}')
            try:
                if key=='visit_cap_contempt_player':raw={'b':'1','black':'1','w':'-1','white':'-1','unspecified':'0'}.get(str(raw).lower(),raw)
                result[group][key] = convert(str(raw))
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid {group}.{key}: {raw}') from error
        if group==section:
            if gtp and not any((group,key) in values or key in common or prefix+key.upper() in env or 'ENGINE_'+key.upper() in env or key in overrides.get(group,{}) for key in ('visits','max_playouts','max_time')):
                # Source GTP supplies a short clock only when no search limit was specified.
                result[group]['gtp_default_time_control']=True
            for key in ('playout_doubling_advantage_player','visit_cap_contempt_player'):
                value=result[group][key]
                if key=='playout_doubling_advantage_player':result[group][key]={'b':'black','w':'white'}.get(str(value).lower(),str(value).lower())
            # Source defaults inherit explicitly supplied parent parameters.
            for child,parent in [('root_policy_temperature_early','root_policy_temperature'),('root_fpu_loss_prop','fpu_loss_prop')]:
                if ((group,child) not in values and child not in common and prefix+child.upper() not in env
                        and 'ENGINE_'+child.upper() not in env and child not in overrides.get(group,{})):
                    result[group][child]=result[group][parent]
            if ((group,'root_policy_optimism') not in values and 'root_policy_optimism' not in common
                    and prefix+'ROOT_POLICY_OPTIMISM' not in env and 'ENGINE_ROOT_POLICY_OPTIMISM' not in env
                    and 'root_policy_optimism' not in overrides.get(group,{})):
                result[group]['root_policy_optimism']=min(result[group]['policy_optimism'],.2)
            if result[group]['auto_avoid_repeat_dir']:
                for suffix in ('utility','lambda','min_turn_number','max_turn_number','max_poses'):
                    key='auto_avoid_repeat_'+suffix
                    if ((group,key) not in values and key not in common and prefix+key.upper() not in env and 'ENGINE_'+key.upper() not in env and key not in overrides.get(group,{})):result[group].pop(key,None)
    return result


def validate_engine_config(config, match=False):
    c = config['match' if match else 'analysis']
    validate_search_combination(c.get('root_search_algo', 'puct'), c.get('nonroot_search_algo', 'puct'))
    if c.get('root_search_algo') == 'gumbel':
        validate_gumbel({key: c.get('gumbel_'+key, 0 if key=='noise_scale' else value)
                         for key, value in GUMBEL_DEFAULTS.items()})
        if c['use_graph_search'] or c['reuse_tree']:
            raise ValueError('Gumbel requires use_graph_search=false and reuse_tree=false')
    for group, values in config.items():
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or (value < 0 and key not in ('komi','nn_mutex_pool_power_of_two','visit_cap_contempt_player','resign_threshold','avoid_repeated_pattern_utility','auto_avoid_repeat_utility')):
                    raise ValueError(f'Invalid {group}.{key}')
    if not 0 <= c['seed'] < 2**64 or not (2 if c['rule']=='go' else 5) <= c['board_size'] <= 25:
        raise ValueError('Invalid engine seed or board size')
    if any(c[k]!=0 and not 2<=c[k]<=25 for k in ('board_width','board_height')) or (c['rule']!='go' and (c['board_width'] or c['board_height'])):raise ValueError('Rectangular dimensions require Go sizes in [2,25]')
    if c['rule'] not in RULES:
        raise ValueError('Invalid engine rule')
    from etazero.games import spec_for_rule
    spec = spec_for_rule(c['rule'])
    if not spec.subtree_value_bias and c['subtree_value_bias_factor'] != 0:
        raise ValueError(f'{spec.name} does not support subtree value bias; requires subtree_value_bias_factor=0')
    if c['rule']=='go':
        from etazero.games import validate_go_komi
        validate_go_komi(c['komi'])
    if not 1 <= c['visits'] <= 2**50 or any(c[k] < 1 for k in ('cpu_threads','search_threads','max_batch','server_threads','muzero_queue_capacity')) or c['c_puct'] < 0:
        raise ValueError('Engine requires visits >= 1 and positive execution/search sizes')
    if c['lcb_stdevs'] <= 0 or c['fpu_parent_weight_by_visited_policy_pow'] < 0 or c['min_visit_prop_for_lcb'] > 1:
        raise ValueError('Invalid engine FPU/LCB settings')
    if not 0<=c['playout_doubling_advantage']<=math.log2(100) or c['playout_doubling_advantage_player'] not in ('black','white'):
        raise ValueError('Invalid engine PDA condition')
    if c['report_analysis_winrates_as'].upper() not in ('BLACK','WHITE','SIDETOMOVE'):raise ValueError('Invalid reportAnalysisWinratesAs')
    if not 1<=c['analysis_pv_len']<=1000 or not 0<=c['analysis_wide_root_noise']<=5:raise ValueError('Invalid analysis report settings')
    for key in ('always_compute_pass_alive_under_suicide_rules','exclude_territory_adjacent_to_atari'):
        if c[key].lower() not in ('true','false','auto'):raise ValueError('Invalid Go history mode')
    validate_search_parameters(c)
    if not (0 if c['cache_entries'] else -1)<=c['nn_mutex_pool_power_of_two']<=24:
        raise ValueError('Invalid NN cache mutex pool power')
    validate_game_symmetry(c,spec)
    if c['rule']=='xiangqi' and c['board_size']!=10: raise ValueError('Xiangqi requires board_size=10')
    if c['device'] != 'cpu' and not (c['device'].startswith('cuda:') and c['device'][5:].isdigit()):
        raise ValueError('Engine device must be cpu or cuda:<index>')
    if c['inference_precision'] not in ('auto', 'float32', 'float16') or (c['device'] == 'cpu' and c['inference_precision'] == 'float16'):
        raise ValueError('Invalid engine inference precision/device')
    if match:
        if not -1 <= c['resign_threshold'] <= 0 or not 1 <= c['resign_consec_turns'] <= 100:
            raise ValueError('Invalid match resignation settings')
        if c['games'] < 1 or c['game_threads'] < 1:
            raise ValueError('Match games and game threads must be positive')
        if c['rule']=='hex': validate_hex_opening(config['hex_opening'],match=True)
        elif c['rule'] not in ('go','xiangqi'): validate_match_opening(config['opening'])


def validate_match_opening(o):
    for key, value in o.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or (value < 0 and key != 'komi'):
                raise ValueError(f'Invalid opening.{key}')
    if o['probability'] != 1 or not 0 <= o['rejection_probability'] <= 1 or not 0 <= o['rejection_probability_fallback'] < 1:
        raise ValueError('Matches require successful balanced openings and fallback rejection below one')
    if not 1 <= o['max_tries'] <= 1000 or any(o[k] > 100 for k in ('avg_dist_factor','balance_exponent','policy_init_mean')) or not .1 <= o['policy_temperature'] <= 5:
        raise ValueError('Invalid match opening configuration')
