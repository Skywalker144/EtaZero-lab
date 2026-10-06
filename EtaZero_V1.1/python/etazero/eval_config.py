"""Independent evaluation profiles; training configuration is never read or fingerprinted."""
import configparser
import math
import os
from pathlib import Path
from .config import ROOT, boolean, FIELDS, SEARCH_PARAMETERS, validate_search_parameters

SEARCH = dict(playout_doubling_advantage=float,playout_doubling_advantage_player=str,seed=int, board_size=int, rule=str, device=str, cpu_threads=int,
              visits=int, search_threads=int, c_puct=float, virtual_loss=float, reuse_tree=boolean,
              temperature=float, temperature_early=float, max_batch=int, server_threads=int, cache_entries=int,
              inference_precision=str, batch_wait_us=int, queue_capacity=int,
              use_fpu=boolean, fpu_reduction_max=float, root_fpu_reduction_max=float,
              fpu_parent_weight_by_visited_policy_pow=float, use_lcb=boolean,
              lcb_stdevs=float, min_visit_prop_for_lcb=float, policy_target_pruning=boolean, **SEARCH_PARAMETERS)


def load_evaluation_config(directory, match=False, environ=None, *, umbrella=False):
    directory = Path(directory)
    if not directory.is_absolute() and not directory.exists():
        directory = ROOT/directory
    directory = directory.resolve()
    name = 'match' if match else 'eval'
    section = 'match' if match else 'evaluation'
    fields = {section: dict(SEARCH, **({'games': int, 'game_threads': int} if match else {}))}
    if match:
        fields['opening'] = {**FIELDS['opening'][1], **FIELDS['policy_init'][1]}
    values = {}

    def read(current, local=False):
        path = current/(name+'.cfg'+('.local' if local else ''))
        if not path.exists():
            return
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string(path.read_text())
        if parser.defaults():
            raise ValueError(f'DEFAULT fields are not supported: {path}')
        for group in parser.sections():
            if group not in fields:
                raise ValueError(f'Unknown evaluation section: {path}: {group}')
            for key, value in parser.items(group):
                if key not in fields[group]:
                    raise ValueError(f'Unknown evaluation key: {path}: {group}.{key}')
                values[group, key] = value

    def inherit(current, ancestors):
        if current in ancestors:
            raise ValueError('Configuration inheritance cycle')
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string((current/'run.cfg').read_text())
        parent = parser.get('run', 'extends', fallback=None)
        if parent:
            if Path(parent).name != parent or parent in ('.', '..'):
                raise ValueError('extends must name a directory under configs/')
            target = directory.parent/parent
            if not (target/'run.cfg').is_file():
                target = ROOT/'configs'/parent
            inherit(target, ancestors+[current])
        read(current)
    if umbrella:
        # An experiment umbrella has no run.cfg. Its one shared profile overlays
        # baseline directly, independently of all arm training configurations.
        read(ROOT/'configs/baseline')
        read(directory)
    else:
        inherit(directory, [])
    read(directory, True)
    env = os.environ if environ is None else environ
    prefix = 'MATCH_' if match else 'EVAL_'
    result = {}
    for group, keys in fields.items():
        result[group] = {}
        for key, convert in keys.items():
            override = prefix+('OPENING_' if group == 'opening' else '')+key.upper()
            raw = env.get(override, values.get((group, key)))
            if raw is None and group == "opening":
                defaults = {"policy_init": "false", "policy_temperature": "1", "policy_after": "true", "policy_on_failure": "true"}
                if key == "policy_init_mean" and not result[group]["policy_init"]:
                    raw = "0"
                else:
                    raw = defaults.get(key)
            if raw is None:
                raise ValueError(f'Missing evaluation field: {group}.{key}')
            result[group][key] = convert(raw)
    validate_evaluation(result, match)
    return result


def validate_evaluation(config, match=False):
    c = config['match' if match else 'evaluation']
    for group, values in config.items():
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid {group}.{key}')
    if not 0 <= c['seed'] < 2**64 or not 5 <= c['board_size'] <= 25:
        raise ValueError('Invalid evaluation seed or board size')
    if c['rule'] not in ('freestyle', 'standard', 'renju'):
        raise ValueError('Invalid evaluation rule')
    if not 1 <= c['visits'] <= 2**31-1 or any(c[k] < 1 for k in ('cpu_threads','search_threads','max_batch','server_threads','queue_capacity')) or c['c_puct'] <= 0:
        raise ValueError('Evaluation requires visits >= 1 and positive execution/search sizes')
    if c['lcb_stdevs'] <= 0 or c['fpu_parent_weight_by_visited_policy_pow'] < 0 or c['min_visit_prop_for_lcb'] > 1:
        raise ValueError('Invalid evaluation FPU/LCB settings')
    if not 0<=c['playout_doubling_advantage']<=math.log2(100) or c['playout_doubling_advantage_player'] not in ('black','white'):
        raise ValueError('Invalid evaluation PDA condition')
    validate_search_parameters(c)
    if c['device'] != 'cpu' and not (c['device'].startswith('cuda:') and c['device'][5:].isdigit()):
        raise ValueError('Evaluation device must be cpu or cuda:<index>')
    if c['inference_precision'] not in ('auto', 'float32', 'float16') or (c['device'] == 'cpu' and c['inference_precision'] == 'float16'):
        raise ValueError('Invalid evaluation inference precision/device')
    if match:
        if c['games'] < 4 or c['games'] % 4 or c['game_threads'] < 1:
            raise ValueError('Match games must be a positive multiple of four')
        o = config['opening']
        if o['probability'] != 1 or not 0 <= o['rejection_probability'] <= 1 or not 0 <= o['rejection_probability_fallback'] < 1:
            raise ValueError('Matches require successful balanced openings and fallback rejection below one')
        if not 1 <= o['max_tries'] <= 1000 or any(o[k] > 100 for k in ('avg_dist_factor','balance_exponent','policy_init_mean')) or not .1 <= o['policy_temperature'] <= 5:
            raise ValueError('Invalid match opening configuration')
