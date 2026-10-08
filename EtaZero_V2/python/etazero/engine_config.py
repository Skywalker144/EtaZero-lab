"""Shared engine settings and independent analysis/match profiles.

Profiles explicitly include their engine and parent profiles. The selected mode
is flattened once for native consumers and provenance; no training files are read.
"""
import configparser
import math
import os
import shlex
from pathlib import Path
from .config import ROOT, boolean, FIELDS, SEARCH_PARAMETERS, validate_search_parameters, validate_hex_opening, validate_hex_symmetry
from .schema import RULES

ENGINE_FIELDS = dict(playout_doubling_advantage=float,playout_doubling_advantage_player=str,seed=int, board_size=int, rule=str, device=str, cpu_threads=int,
              visits=int, search_threads=int, c_puct=float, virtual_loss=float, reuse_tree=boolean,
              temperature=float, temperature_early=float, max_batch=int, server_threads=int, cache_entries=int,
              inference_precision=str, batch_wait_us=int, queue_capacity=int,
              use_fpu=boolean, fpu_reduction_max=float, root_fpu_reduction_max=float,
              fpu_parent_weight_by_visited_policy_pow=float, use_lcb=boolean,
              lcb_stdevs=float, min_visit_prop_for_lcb=float, policy_target_pruning=boolean, **SEARCH_PARAMETERS)


def load_engine_config(source, match=False, environ=None, *, umbrella=False):
    result = _load_profile(source, match, environ, umbrella=umbrella)
    validate_engine_config(result, match)
    return result


def load_match_opening_config(source, environ=None, *, umbrella=False):
    """Read match's opening alone, without imposing match search or game budgets."""
    profiles = _load_profile(source, True, environ, umbrella=umbrella, groups=('opening',))
    validate_match_opening(profiles['opening'])
    return profiles['opening']


def load_match_hex_opening_config(source, environ=None, *, umbrella=False):
    opening = _load_profile(source, True, environ, umbrella=umbrella, groups=('hex_opening',))['hex_opening']
    validate_hex_opening(opening,match=True)
    return opening


def _load_profile(source, match, environ, *, umbrella=False, groups=None):
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
    values, common = {}, {}

    def read(path, ancestors=(), *, local=False):
        path = path.resolve()
        if path in ancestors:
            raise ValueError('Engine configuration include cycle: ' + ' -> '.join(map(str, (*ancestors, path))))
        lines, includes = [], []
        for line in path.read_text().splitlines():
            if line.lstrip().startswith('@include'):
                if any(v.lstrip().startswith('[') for v in lines):
                    raise ValueError(f'@include must precede sections: {path}')
                tokens = shlex.split(line, comments=True)
                if len(tokens) != 2 or tokens[0] != '@include':
                    raise ValueError(f'Invalid engine include: {path}: {line}')
                includes.append(path.parent/tokens[1])
            else:
                lines.append(line)
        for included in includes:
            read(included, (*ancestors, path), local=local)
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string('\n'.join(lines))
        if parser.defaults():
            raise ValueError(f'DEFAULT fields are not supported: {path}')
        for group in parser.sections():
            keys = ENGINE_FIELDS if group == 'engine' else fields.get(group)
            if keys is None:
                raise ValueError(f'Unknown engine section: {path}: {group}')
            for key, value in parser.items(group):
                if key not in keys:
                    raise ValueError(f'Unknown engine key: {path}: {group}.{key}')
                if group == 'engine' and not local:
                    common[key] = value
                else:
                    values[section if group == 'engine' else group, key] = value
    if umbrella:
        # An experiment may omit a match profile and use the default protocol.
        read(ROOT/'configs/baseline'/f'{section}.cfg')
    if path.is_file():
        read(path)
    elif not umbrella:
        raise FileNotFoundError(f'Engine profile does not exist: {path}')
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
            if raw is None and group == "opening":
                defaults = {"policy_init": "false", "policy_temperature": "1", "policy_after": "true", "policy_on_failure": "true"}
                if key == "policy_init_mean" and not result[group]["policy_init"]:
                    raw = "0"
                else:
                    raw = defaults.get(key)
            if raw is None:
                raise ValueError(f'Missing engine field: {group}.{key}')
            try:
                result[group][key] = convert(raw)
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid {group}.{key}: {raw}') from error
    return result


def validate_engine_config(config, match=False):
    c = config['match' if match else 'analysis']
    for group, values in config.items():
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid {group}.{key}')
    if not 0 <= c['seed'] < 2**64 or not 5 <= c['board_size'] <= 25:
        raise ValueError('Invalid engine seed or board size')
    if c['rule'] not in RULES:
        raise ValueError('Invalid engine rule')
    if not 1 <= c['visits'] <= 2**31-1 or any(c[k] < 1 for k in ('cpu_threads','search_threads','max_batch','server_threads','queue_capacity')) or c['c_puct'] <= 0:
        raise ValueError('Engine requires visits >= 1 and positive execution/search sizes')
    if c['lcb_stdevs'] <= 0 or c['fpu_parent_weight_by_visited_policy_pow'] < 0 or c['min_visit_prop_for_lcb'] > 1:
        raise ValueError('Invalid engine FPU/LCB settings')
    if not 0<=c['playout_doubling_advantage']<=math.log2(100) or c['playout_doubling_advantage_player'] not in ('black','white'):
        raise ValueError('Invalid engine PDA condition')
    validate_search_parameters(c)
    if c['rule']=='hex': validate_hex_symmetry(c)
    if c['device'] != 'cpu' and not (c['device'].startswith('cuda:') and c['device'][5:].isdigit()):
        raise ValueError('Engine device must be cpu or cuda:<index>')
    if c['inference_precision'] not in ('auto', 'float32', 'float16') or (c['device'] == 'cpu' and c['inference_precision'] == 'float16'):
        raise ValueError('Invalid engine inference precision/device')
    if match:
        if c['games'] < 4 or c['games'] % 4 or c['game_threads'] < 1:
            raise ValueError('Match games must be a positive multiple of four')
        if c['rule']=='hex': validate_hex_opening(config['hex_opening'],match=True)
        else: validate_match_opening(config['opening'])


def validate_match_opening(o):
    for key, value in o.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid opening.{key}')
    if o['probability'] != 1 or not 0 <= o['rejection_probability'] <= 1 or not 0 <= o['rejection_probability_fallback'] < 1:
        raise ValueError('Matches require successful balanced openings and fallback rejection below one')
    if not 1 <= o['max_tries'] <= 1000 or any(o[k] > 100 for k in ('avg_dist_factor','balance_exponent','policy_init_mean')) or not .1 <= o['policy_temperature'] <= 5:
        raise ValueError('Invalid match opening configuration')
