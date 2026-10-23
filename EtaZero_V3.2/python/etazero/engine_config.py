"""Shared engine settings and independent analysis/match profiles.

Profiles explicitly include their engine and parent profiles. The selected mode
is flattened once for native consumers and provenance; no training files are read.
"""
import math
import os
from pathlib import Path
from .config import PATTERN_DEFAULTS, resolve_pattern_corpora, HUMAN_SL_DEFAULTS, TIME_CONTROL_DEFAULTS, ROOT, boolean, FIELDS, SEARCH_PARAMETERS, validate_search_parameters, validate_hex_opening, validate_game_symmetry
from .schema import RULES
from .config_files import read_config_file
from .config import GUMBEL_FIELDS, GUMBEL_DEFAULTS, validate_gumbel, validate_search_combination

ENGINE_FIELDS = dict(go_rules=str, komi=float, playout_doubling_advantage=float,playout_doubling_advantage_player=str,seed=int, board_size=int, rule=str, device=str, cpu_threads=int,
              visits=int, search_threads=int, c_puct=float, virtual_loss=float, reuse_tree=boolean,
              temperature=float, temperature_early=float, max_batch=int, server_threads=int, cache_entries=int,
              inference_precision=str, batch_wait_us=int, muzero_queue_capacity=int,nn_mutex_pool_power_of_two=int,
              use_fpu=boolean, fpu_reduction_max=float, root_fpu_reduction_max=float,
              fpu_parent_weight_by_visited_policy_pow=float, use_lcb=boolean,
              lcb_stdevs=float, min_visit_prop_for_lcb=float, policy_target_pruning=boolean, **SEARCH_PARAMETERS)
ENGINE_FIELDS.update(pondering_enabled=boolean,root_search_algo=str, nonroot_search_algo=str,
                     **{'gumbel_'+key: value for key, value in GUMBEL_FIELDS.items()})


def load_engine_config(source, match=False, environ=None, *, overrides=None):
    result = _load_profile(source, match, environ, overrides=overrides)
    group=result['match' if match else 'analysis']
    base=Path(source);base=base if base.is_dir() else base.parent
    group['pattern_corpora']=resolve_pattern_corpora(group['pattern_corpora'],base)
    if group['human_model']:
        path=Path(group['human_model']);base=Path(source);base=base if base.is_dir() else base.parent
        group['human_model']=str((base/path).resolve() if not path.is_absolute() else path)
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


def _load_profile(source, match, environ, *, groups=None, overrides=None):
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
                if key in PATTERN_DEFAULTS: raw=str(PATTERN_DEFAULTS[key])
                elif key in HUMAN_SL_DEFAULTS: raw = str(HUMAN_SL_DEFAULTS[key]).lower() if isinstance(HUMAN_SL_DEFAULTS[key],bool) else str(HUMAN_SL_DEFAULTS[key])
                elif key in TIME_CONTROL_DEFAULTS: raw = str(TIME_CONTROL_DEFAULTS[key])
                elif key == 'graph_search_rep_bound': raw = '11'
                elif key == 'node_table_shards_power_of_two': raw = '16'
                elif key == 'min_playouts_per_thread': raw = '8' if group=='analysis' else '0'
                elif key == 'pondering_enabled': raw = 'false'
                elif key == 'root_symmetry_pruning': raw = 'true' if group=='analysis' else 'false'
                elif key == 'use_eval_cache': raw = 'false'
                elif key == 'eval_cache_min_visits': raw = '100'
                elif key == 'wide_root_noise': raw = '.04' if group=='analysis' else '0'
                elif key in ('futile_visits_threshold','visit_cap_contempt','visit_cap_contempt_player'): raw = '0'
                elif key == 'nn_mutex_pool_power_of_two': raw = '17'
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
                result[group][key] = convert(str(raw))
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid {group}.{key}: {raw}') from error
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
                if not math.isfinite(value) or (value < 0 and key not in ('komi','nn_mutex_pool_power_of_two','visit_cap_contempt_player')):
                    raise ValueError(f'Invalid {group}.{key}')
    if not 0 <= c['seed'] < 2**64 or not 5 <= c['board_size'] <= 25:
        raise ValueError('Invalid engine seed or board size')
    if c['rule'] not in RULES:
        raise ValueError('Invalid engine rule')
    from .games import spec_for_rule
    spec = spec_for_rule(c['rule'])
    if not spec.subtree_value_bias and c['subtree_value_bias_factor'] != 0:
        raise ValueError(f'{spec.name} does not support subtree value bias; requires subtree_value_bias_factor=0')
    if c['rule']=='go':
        from .games import validate_go_komi
        validate_go_komi(c['komi'])
    if not 1 <= c['visits'] <= 2**50 or any(c[k] < 1 for k in ('cpu_threads','search_threads','max_batch','server_threads','muzero_queue_capacity')) or c['c_puct'] < 0:
        raise ValueError('Engine requires visits >= 1 and positive execution/search sizes')
    if c['lcb_stdevs'] <= 0 or c['fpu_parent_weight_by_visited_policy_pow'] < 0 or c['min_visit_prop_for_lcb'] > 1:
        raise ValueError('Invalid engine FPU/LCB settings')
    if not 0<=c['playout_doubling_advantage']<=math.log2(100) or c['playout_doubling_advantage_player'] not in ('black','white'):
        raise ValueError('Invalid engine PDA condition')
    validate_search_parameters(c)
    if not -1<=c['nn_mutex_pool_power_of_two']<=24:
        raise ValueError('Invalid NN cache mutex pool power')
    validate_game_symmetry(c,spec)
    if c['rule']=='xiangqi' and c['board_size']!=10: raise ValueError('Xiangqi requires board_size=10')
    if c['device'] != 'cpu' and not (c['device'].startswith('cuda:') and c['device'][5:].isdigit()):
        raise ValueError('Engine device must be cpu or cuda:<index>')
    if c['inference_precision'] not in ('auto', 'float32', 'float16') or (c['device'] == 'cpu' and c['inference_precision'] == 'float16'):
        raise ValueError('Invalid engine inference precision/device')
    if match:
        if c['games'] < 4 or c['games'] % 4 or c['game_threads'] < 1:
            raise ValueError('Match games must be a positive multiple of four')
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
