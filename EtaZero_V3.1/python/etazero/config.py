"""Strict, directory-based INI configuration. Native consumers receive resolved values."""
import configparser
from .config_files import read_config_file
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise ValueError("Boolean values must be true or false")
    return value.lower() == "true"


def row_target(value):
    return "all" if value.lower() == "all" else int(value)


GUMBEL_FIELDS = dict(max_num_considered_actions=int, c_visit=float, c_scale=float,
                     noise_scale=float, rescale_q_values=boolean, action_selection=str)
GUMBEL_DEFAULTS = dict(max_num_considered_actions=16, c_visit=50., c_scale=.1,
                       noise_scale=1., rescale_q_values=True, action_selection='gumbel')


def validate_gumbel(c):
    if (type(c['max_num_considered_actions']) is not int or not 1 <= c['max_num_considered_actions'] <= 2086
            or not math.isfinite(c['c_visit']) or c['c_visit'] < 0
            or not math.isfinite(c['c_scale']) or c['c_scale'] <= 0
            or not math.isfinite(c['noise_scale']) or c['noise_scale'] < 0
            or c['action_selection'] not in ('gumbel', 'visit')):
        raise ValueError('Invalid Gumbel candidate count, scaling or action selection')


def validate_search_combination(root, nonroot):
    if root not in ('puct', 'gumbel') or nonroot not in ('puct', 'gumbel'):
        raise ValueError('Invalid algorithm/search enum')
    if root == 'puct' and nonroot != 'puct':
        raise ValueError('PUCT root + Gumbel nonroot is not an allowed project combination')


# Each section belongs to exactly one file; baseline supplies every required key.
# run_dir is optional and belongs only to the selected configuration directory.
SEARCH_PARAMETERS = dict(use_uncertainty=boolean, uncertainty_coeff=float, uncertainty_exponent=float, uncertainty_max_weight=float, policy_optimism=float, root_policy_optimism=float, use_noise_pruning=boolean, noise_prune_utility_scale=float, noise_pruning_cap=float, use_graph_search=boolean, graph_search_catch_up_leak_prob=float, graph_search_rep_bound=int, max_playouts=int, max_time=float, nn_randomize=boolean, nn_symmetry=int,
                         subtree_value_bias_factor=float, subtree_value_bias_free_prop=float,
                         subtree_value_bias_weight_exponent=float, subtree_value_bias_table_num_shards=int,
                         fpu_parent_weight_by_visited_policy=boolean, fpu_parent_weight=float, value_weight_exponent=float, chosen_move_subtract=float, chosen_move_prune=float,
                         fpu_loss_prop=float, root_fpu_loss_prop=float, c_puct_log=float, c_puct_base=float,
                         c_puct_stdev_prior=float, c_puct_stdev_prior_weight=float, c_puct_stdev_scale=float,
                         root_num_symmetries_to_sample=int, nn_policy_temperature=float,
                         root_policy_temperature_early=float, root_policy_temperature=float,
                         temperature_halflife=float, temperature_only_below_prob=float)
SEARCH_PARAMETERS.update(root_symmetry_pruning=boolean)
SEARCH_PARAMETERS.update(use_eval_cache=boolean,eval_cache_min_visits=int)
SEARCH_PARAMETERS.update(wide_root_noise=float,futile_visits_threshold=float,visit_cap_contempt=int,visit_cap_contempt_player=int)
SEARCH_PARAMETERS.update(node_table_shards_power_of_two=int,min_playouts_per_thread=float,ignore_pre_root_history=boolean,ignore_all_history=boolean)


PATTERN_DEFAULTS={'avoid_repeated_pattern_utility':0.0,'pattern_corpora':'[]'}
PATTERN_FIELDS={'avoid_repeated_pattern_utility':float,'pattern_corpora':str}
SEARCH_PARAMETERS.update(PATTERN_FIELDS)

def resolve_pattern_corpora(value,base):
    import json
    sources=json.loads(value)
    if not isinstance(sources,list): raise ValueError('pattern_corpora must be a JSON array')
    for source in sources:
        if not isinstance(source,dict) or set(source)-{'paths','penalty','decay','min_turn','max_files','players'}: raise ValueError('Invalid pattern corpus fields')
        if not isinstance(source.get('paths'),list) or not source['paths']: raise ValueError('Pattern corpus requires paths')
        for key,low,high in [('penalty',0,3),('decay',0,1),('min_turn',0,2**50),('max_files',0,2**50)]:
            if key in source and not low<=source[key]<=high: raise ValueError(f'Invalid pattern corpus {key}')
        source['paths']=[str((Path(base)/p).resolve()) for p in source['paths']]
    return json.dumps(sources,separators=(',',':'))

HUMAN_SL_DEFAULTS = {'human_model': '', 'human_sl_cpuct_exploration': 1.0, 'human_sl_cpuct_permanent': 0.0, 'human_sl_root_weightless': 0.0, 'human_sl_root_weightful': 0.0, 'human_sl_player_weightless': 0.0, 'human_sl_player_weightful': 0.0, 'human_sl_opponent_weightless': 0.0, 'human_sl_opponent_weightful': 0.0, 'human_sl_chosen_move_proportion': 0.0, 'human_sl_chosen_move_pikl_lambda': 1000000000.0, 'human_sl_chosen_move_ignore_pass': False}
HUMAN_SL_FIELDS = {'human_model':str, 'human_sl_cpuct_exploration':float, 'human_sl_cpuct_permanent':float, 'human_sl_root_weightless':float, 'human_sl_root_weightful':float, 'human_sl_player_weightless':float, 'human_sl_player_weightful':float, 'human_sl_opponent_weightless':float, 'human_sl_opponent_weightful':float, 'human_sl_chosen_move_proportion':float, 'human_sl_chosen_move_pikl_lambda':float, 'human_sl_chosen_move_ignore_pass':boolean}
SEARCH_PARAMETERS.update(HUMAN_SL_FIELDS)

TIME_CONTROL_DEFAULTS = {'max_visits_pondering': 1125899906842624, 'max_playouts_pondering': 1125899906842624, 'max_time_pondering': 1e+20, 'lag_buffer': 0.0, 'tree_reuse_carry_over_time_factor': 0.0, 'overallocate_time_factor': 1.0, 'midgame_time_factor': 1.0, 'midgame_turn_peak_time': 130.0, 'endgame_turn_time_decay': 100.0, 'obvious_moves_time_factor': 1.0, 'obvious_moves_policy_entropy_tolerance': 0.3, 'obvious_moves_policy_surprise_tolerance': 0.15}
TIME_CONTROL_FIELDS = {'max_visits_pondering':int, 'max_playouts_pondering':int, 'max_time_pondering':float, 'lag_buffer':float, 'tree_reuse_carry_over_time_factor':float, 'overallocate_time_factor':float, 'midgame_time_factor':float, 'midgame_turn_peak_time':float, 'endgame_turn_time_decay':float, 'obvious_moves_time_factor':float, 'obvious_moves_policy_entropy_tolerance':float, 'obvious_moves_policy_surprise_tolerance':float}
SEARCH_PARAMETERS.update(TIME_CONTROL_FIELDS)

def validate_search_parameters(c):
    if not 0<=c.get("avoid_repeated_pattern_utility",0)<=3: raise ValueError("Invalid repeated-pattern utility")
    for key,value in HUMAN_SL_DEFAULTS.items():
        if isinstance(value,(str,bool)):continue
        maximum=1e9 if key.endswith('pikl_lambda') else 1000 if 'cpuct' in key else 1
        if not 0<=c.get(key,value)<=maximum:raise ValueError(f'Invalid {key}')
    if not c.get('human_model','') and any(c.get(k,0)>0 for k in HUMAN_SL_DEFAULTS if k.endswith(('weightless','weightful','proportion'))):
        raise ValueError('Secondary policy mixing or exploration requires human_model')
    if not 0<=c.get('wide_root_noise',0)<=5 or not 0<=c.get('futile_visits_threshold',0)<=1:
        raise ValueError('Invalid root exploration noise or futile visits threshold')
    if not 1<=c.get('eval_cache_min_visits',100)<=2**50: raise ValueError('Invalid eval cache visit threshold')
    cap=c.get('visit_cap_contempt',0)
    if cap==1 or not 0<=cap<=2**50 or c.get('visit_cap_contempt_player',0) not in (-1,0,1):
        raise ValueError('Visit cap must be 0 or [2,2^50], player must be -1,0,1')
    if not 8<=c.get('node_table_shards_power_of_two',16)<=24 or not 0<=c.get('min_playouts_per_thread',0)<=1e20:
        raise ValueError('Invalid search shard power or minimum playouts per thread')
    for key in ('max_visits_pondering','max_playouts_pondering'):
        if not 1<=c.get(key,2**50)<=2**50: raise ValueError(f'Invalid {key}')
    time_bounds=dict(max_time_pondering=(0,1e20),lag_buffer=(0,3600),tree_reuse_carry_over_time_factor=(0,1),
        overallocate_time_factor=(.01,100),midgame_time_factor=(.01,100),midgame_turn_peak_time=(0,1000),
        endgame_turn_time_decay=(0,1000),obvious_moves_time_factor=(.01,1),obvious_moves_policy_entropy_tolerance=(.001,2),
        obvious_moves_policy_surprise_tolerance=(.001,2))
    for key,(low,high) in time_bounds.items():
        if not low<=c.get(key,TIME_CONTROL_DEFAULTS[key])<=high: raise ValueError(f'Invalid {key}')
    bounds={'uncertainty_coeff' :(.0001,1),'uncertainty_exponent':(0,2),'uncertainty_max_weight':(1,100),
            'policy_optimism':(0,1),'root_policy_optimism':(0,1),'noise_prune_utility_scale':(.001,10),'noise_pruning_cap':(0,1e50),
            'subtree_value_bias_factor':(0,1),'subtree_value_bias_free_prop':(0,1),'subtree_value_bias_weight_exponent':(0,1)}
    for key,(low,high) in bounds.items():
        if key.startswith('uncertainty_') and not c['use_uncertainty']: continue
        if key.startswith('noise_') and not c['use_noise_pruning']: continue
        if not math.isfinite(c[key]) or not low<=c[key]<=high:
            raise ValueError(f'Invalid search correction: {key}')
    if type(c['subtree_value_bias_table_num_shards']) is not int or not 1<=c['subtree_value_bias_table_num_shards']<=2**31-1:
        raise ValueError('subtree_value_bias_table_num_shards must be a positive int32')
    if type(c['graph_search_rep_bound']) is not int or not 3 <= c['graph_search_rep_bound'] <= 50:
        raise ValueError('graph_search_rep_bound must be an integer in [3,50]')
    if not 0 <= c['nn_symmetry'] < 8 or not 0 <= c['max_playouts'] <= 2**50:
        raise ValueError('Invalid NN symmetry or playout cap')
    if not 1 <= c['root_num_symmetries_to_sample'] <= 8:
        raise ValueError('Root symmetry count must be in [1,8]')
    if any(c[k] > 1 for k in ('graph_search_catch_up_leak_prob','fpu_loss_prop','root_fpu_loss_prop','c_puct_stdev_scale','temperature_only_below_prob','fpu_parent_weight')):
        raise ValueError('Search proportions must be <= 1')
    if any(c[k] <= 0 for k in ('c_puct_base','c_puct_stdev_prior','nn_policy_temperature',
                              'root_policy_temperature','root_policy_temperature_early','temperature_halflife')):
        raise ValueError('Policy temperatures, halflife and PUCT scales must be positive')
    if any(not .01<=c[key]<=100 for key in ('root_policy_temperature','root_policy_temperature_early')):
        raise ValueError('Root policy temperatures must be in [0.01,100]')

OPTION_DEFAULTS = {
    'patterns':PATTERN_DEFAULTS,
    'human_sl': HUMAN_SL_DEFAULTS,
    'time_control': TIME_CONTROL_DEFAULTS,
    'losses': dict(value_scale=0.6, td_long_scale=0.6, td_mid_scale=0.6, td_short_scale=0.6,
                   variance_time_scale=1.0, main_head_scale=-1.0, intermediate_head_scale=-1.0),
    'optimizer': dict(muon_adam_lr_factor=1.0, wd_floor_frac=0.0, ns_steps=5,
                      use_polar_express=False, aurora_pp_iterations=2, aurora_pp_beta=0.5),
    'lr_schedule': dict(mode='constant', auto2_factor=1.0, points='', cycle_low=1.0,
                        cycle_high=1.0, cycle_up_rounds=0, cycle_down_rounds=0, cycle_rounds=1),
    'attention_penalty': dict(cap=-1.0, coefficient=1e-3, batch_fraction=1.0),
    'data_weight': dict(scale_data_weight=1.0),
    'start_positions': dict(probability=0.0, files='', excludes='', load_probability=1.0, turn_weight_lambda=0.0,
                            policy_init=False, policy_area_proportion=0.0, policy_gamma_shape=1.0, policy_temperature=1.0),
    'graph_search': dict(node_table_shards_power_of_two=16,use_eval_cache=False,eval_cache_min_visits=100),
    'parallelism': dict(min_playouts_per_thread=0.0),
    'inference': dict(nn_mutex_pool_power_of_two=16),
    'search_extras': dict(root_symmetry_pruning=False,wide_root_noise=0.0,futile_visits_threshold=0.0,visit_cap_contempt=0,visit_cap_contempt_player=0),
    'search_history': dict(ignore_pre_root_history=False,ignore_all_history=False),
}


FIELDS = {
    'patterns':('selfplay',PATTERN_FIELDS),
    'human_sl': ('selfplay',HUMAN_SL_FIELDS),
    'time_control': ('selfplay',TIME_CONTROL_FIELDS),
    "run": ("run", {"run_dir": str, "seed": int, "max_iteration": int,
                    "max_seconds": float, "cpu_threads": int}),
    "agent": ("run", {"algorithm": str, "root_search_algo": str, "nonroot_search_algo": str}),
    "gumbel": ("selfplay", GUMBEL_FIELDS),
    "devices": ("run", {"train": str, "selfplay": str}),
    "go": ("env", {"rules": str, "komi": float}),
    "environment": ("env", {"sizes": str, "size_weights": str, "rules": str, "rule_weights": str,
                              "forbidden_feature_dropout_prob": float}),
    "network": ("net", {"architecture": str, "canvas": int, "channels": int, "blocks": int, "predict_q_values":boolean}),
    "muzero": ("net", {"latent_channels": int, "dynamics_channels": int, "dynamics_blocks": int,
                         "prediction_channels": int, "prediction_blocks": int}),
    "unroll": ("train", {"steps": int, "hidden_gradient_scale": float}),
    "muzero_training": ("train", {"auxiliary_losses": boolean, "katago_optimizer": boolean,
                                  "learning_rate": float, "weight_decay": float}),
    "plain_optimizer": ("train", {"learning_rate": float, "weight_decay": float}),
    "brenorm": ("train", {"avg_momentum": float, "target_rmax": float,
                           "target_dmax": float, "adjustment_scale": float}),
    'losses': ('train', {key: float for key in OPTION_DEFAULTS['losses']}),
    'lr_schedule': ('train', {key: type(value) for key, value in OPTION_DEFAULTS['lr_schedule'].items()}),
    'attention_penalty': ('train', {key: float for key in OPTION_DEFAULTS['attention_penalty']}),
    "training": ("train", {"train_steps": int, "batch_size": int, "prefetch_depth": int,
                           'cuda_prefetch':boolean,'compile':boolean,'sub_epochs':int,'skip_validation':boolean,'randomize_validation_files':boolean,'max_validation_samples':int,
                           "log_every": int, "checkpoint_every": int, "checkpoint_keep": int, "amp": str, "gradient_clip": float,
                           "replay_ratio": float, "d4_augmentation": boolean, "soft_policy_weight_scale": float, "disable_optimistic_policy": boolean}),
    "optimizer": ("train", {"kind": str, "lr_scale": float, "lr_warmup": boolean,
                             **{key: boolean if type(value) is bool else type(value) for key, value in OPTION_DEFAULTS['optimizer'].items()},
                             "head_lr_factor": float, "noreg_lr_factor": float,
                             "input_wd_factor": float, "normal_wd_factor": float, "normal_attn_wd_factor": float,
                             "norm_interval": int, "norm_only_at_print": boolean, "lookahead_print": boolean,
                             "lookahead_k": int, "lookahead_alpha": float,
                             "swa_period_samples": int, "swa_scale": float}),
    "replay": ("train", {"min_rows": int, "taper_exponent": float,
                         "expand_per_row": float, "keep_target_rows": row_target, "taper_scale":float, "add_to_data_rows":float, "max_rows":row_target}),
    "shuffle": ("train", {"workers": int, "group_rows": int, "bucket_rows": int,
                          "training_shard_rows": int, "waves": int,
                          "temp_dir": str, "snapshot_keep": int, "compress_temp": boolean}),
    'search': ("selfplay", {'full_search_visits': int, 'cheap_search_visits': int,
                            'cheap_search_probs': float, 'cheap_search_target_weight': float,
                            'reuse_tree': boolean, 'clear_before_search': boolean, 'max_playouts': int, 'max_time': float}),
    'graph_search': ("selfplay", {'use_graph_search': boolean, 'graph_search_catch_up_leak_prob': float, 'graph_search_rep_bound': int,'node_table_shards_power_of_two':int,'use_eval_cache':boolean,'eval_cache_min_visits':int}),
    'uncertainty': ("selfplay", dict(use_uncertainty=boolean,uncertainty_coeff=float,uncertainty_exponent=float,uncertainty_max_weight=float)),
    'optimistic_policy': ("selfplay", dict(policy_optimism=float,root_policy_optimism=float)),
    'noise_pruning': ("selfplay", dict(use_noise_pruning=boolean,noise_prune_utility_scale=float,noise_pruning_cap=float)),
    'subtree_value_bias': ("selfplay", {key: SEARCH_PARAMETERS[key] for key in (
        'subtree_value_bias_factor','subtree_value_bias_free_prop','subtree_value_bias_weight_exponent','subtree_value_bias_table_num_shards')}),
    'reanalysis': ("selfplay", dict(use_reanalyze=boolean,reanalyze_prop=float,reanalyze_policy_surprise_weight=float,reanalyze_value_surprise_weight=float,reanalyze_surprise_exponent=float,reanalyze_use_outcome_targets=boolean)),
    'hint_positions': ("selfplay", dict(hint_positions_prob=float,positions_file=str)),
    'data_weight': ('selfplay', dict(scale_data_weight=float)),
    'search_extras': ('selfplay',dict(root_symmetry_pruning=boolean,wide_root_noise=float,futile_visits_threshold=float,visit_cap_contempt=int,visit_cap_contempt_player=int)),
    'search_history': ('selfplay', dict(ignore_pre_root_history=boolean,ignore_all_history=boolean)),
    'start_positions': ('selfplay', {key: boolean if type(value) is bool else type(value) for key,value in OPTION_DEFAULTS['start_positions'].items()}),
    'game_forks': ("selfplay", dict(early_fork_game_prob=float,fork_game_prob=float,early_fork_game_expected_move_prop=float,
                                  fork_game_min_choices=int,early_fork_game_max_choices=int,fork_game_max_choices=int)),
    'side_positions': ("selfplay", dict(side_position_prob=float)),
    'pda': ("selfplay", dict(normal_asymmetric_playout_prob=float,max_asymmetric_ratio=float)),
    'selfplay': ("selfplay", {'bootstrap_games': int}),
    'reduce_visits': ("selfplay", {'reduce_visits': boolean, 'reduce_visits_threshold': float,
                                   'reduce_visits_threshold_lookback': int, 'reduced_visits_min': int,
                                   'reduced_visits_weight': float}),
    'puct': ("selfplay", {'c_puct': float, 'c_puct_log': float, 'c_puct_base': float,
                          'c_puct_stdev_prior': float, 'c_puct_stdev_prior_weight': float,
                          'c_puct_stdev_scale': float, 'virtual_loss': float}),
    'fpu': ("selfplay", {'use_fpu': boolean, 'fpu_reduction_max': float, 'root_fpu_reduction_max': float,
                         'fpu_parent_weight_by_visited_policy_pow': float, 'fpu_loss_prop': float,
                         'root_fpu_loss_prop': float, 'fpu_parent_weight_by_visited_policy': boolean, 'fpu_parent_weight': float}),
    'value_weighting': ("selfplay", {'value_weight_exponent': float}),
    'forced_playouts': ("selfplay", {'root_desired_per_child_visits_coeff': float}),
    'policy_target': ("selfplay", {'policy_target_pruning': boolean, 'chosen_move_subtract': float,
                                   'chosen_move_prune': float}),
    'lcb': ("selfplay", {'use_lcb': boolean, 'lcb_stdevs': float, 'min_visit_prop_for_lcb': float}),
    'symmetry': ("selfplay", {'root_num_symmetries_to_sample': int, 'nn_randomize': boolean, 'nn_symmetry': int}),
    'temperature': ("selfplay", {'nn_policy_temperature': float, 'root_policy_temperature_early': float,
                                 'root_policy_temperature': float, 'temperature': float,
                                 'final_temperature': float, 'temperature_halflife': float,
                                 'temperature_only_below_prob': float}),
    'dirichlet_noise': ("selfplay", {'dirichlet_total_concentration': float,
                                     'shaped_dirichlet_noise': boolean, 'noise_fraction': float}),
    'opening': ("selfplay", {'probability': float, 'avg_dist_factor': float, 'balance_exponent': float,
                             'rejection_probability': float, 'rejection_probability_fallback': float,
                             'max_tries': int}),
    'hex_opening': ("selfplay", {'probability': float, 'make_fair_probability': float,
                                 'balance_exponent': float, 'min_accept_rate': float}),
    'policy_init': ("selfplay", {'policy_init': boolean, 'policy_after': boolean,
                                 'policy_on_failure': boolean, 'policy_init_mean': float,
                                 'policy_temperature': float}),
    'surprise_weighting': ("selfplay", {'policy_surprise_data_weight': float,
                                        'value_surprise_data_weight': float, 'use_search_value_surprise': boolean}),
    'parallelism': ("selfplay", {'game_threads': int, 'search_threads': int,'min_playouts_per_thread':float}),
    'inference': ("selfplay", {'server_threads': int, 'max_batch': int, 'inference_precision': str,
                               'nn_mutex_pool_power_of_two':int,
                               'cache_entries': int, 'batch_wait_us': int, 'muzero_queue_capacity': int}),
    'writer': ("selfplay", {'writer_queue': int, 'shard_rows': int, 'first_file_min_random_proportion': float}),
}
FILES = tuple(dict.fromkeys(v[0] for v in FIELDS.values()))


def network_widths(channels, architecture='nbt'):
    """Widths of the selected source preset; NBT retains its width scaling."""
    if architecture == 'plain':
        if channels != 128:
            raise ValueError('plain requires b10c128-fson-mish: channels=128, blocks=10')
        return dict(mid=128, gpool=32, policy=32, value=32, value_hidden=80)
    if architecture == 'transformer':
        if channels != 192:
            raise ValueError('transformer requires b5c192h3nbttfrs: channels=192, blocks=5')
        return dict(mid=96, gpool=32, policy=32, value=32, value_hidden=64)
    if architecture != 'nbt':
        raise ValueError(f'Unsupported network architecture: {architecture}')
    if channels < 24 or channels % 2:
        raise ValueError("NBT network.channels must be even and >= 24")
    return {"mid": channels // 2, "gpool": 8 * ((channels + 47) // 48),
            "policy": 8 * ((channels + 47) // 48), "value": 8 * ((channels + 47) // 48),
            "value_hidden": 8 * ((5 * channels + 95) // 96)}


def _read(directory, local=False):
    values = {}
    for name in FILES:
        path = directory / (name + ".cfg" + (".local" if local else ""))
        if path.exists():
            fields = {section: keys for section, (owner, keys) in FIELDS.items() if owner == name}
            values.update(read_config_file(path, fields, local_only=(("run", "run_dir"),)))
    return values


def load_config(directory, run_dir=None):
    directory = Path(directory).resolve()
    if not (directory / "run.cfg").is_file():
        raise ValueError(f"Missing configuration directory/run.cfg: {directory}")
    values = _read(directory)
    # Only the selected directory supplies machine overrides and output paths.
    values.update(_read(directory, local=True))
    config = {}
    for section, (_, fields) in FIELDS.items():
        if section == 'go' and not any(s == section for s, _ in values):
            continue
        if section == 'gumbel' and values.get(('agent', 'root_search_algo')) != 'gumbel':
            continue
        if section == 'muzero_training' and not any(s == section for s, _ in values):
            continue
        if section == 'plain_optimizer' and not any(s == section for s, _ in values):
            continue
        if section == 'brenorm' and not any(s == section for s, _ in values):
            continue
        if section in ('muzero', 'unroll', 'muzero_training') and values.get(('agent', 'algorithm')) != 'muzero':
            if any(s == section for s, _ in values):
                raise ValueError(f'{section} configuration requires agent.algorithm = muzero')
            continue
        config[section] = {}
        for key, convert in fields.items():
            if section == 'network' and key in ('channels', 'blocks') and (section, key) not in values:
                from .katago_configs import is_preset, resolve
                architecture = values.get(('network', 'architecture'), '')
                if values.get(('agent', 'algorithm')) == 'alphazero' and is_preset(architecture):
                    preset = resolve(architecture)
                    config[section][key] = (preset['trunk_num_channels'] if key == 'channels'
                                            else len(preset['block_kind']))
                    continue
            if section == "inference" and key == "muzero_queue_capacity" and values.get(("agent", "algorithm")) != "muzero":
                continue
            if section=="reanalysis" and key!="use_reanalyze" and (section,key) not in values and not config[section]["use_reanalyze"]:
                continue
            if (section, key) not in values:
                if key in OPTION_DEFAULTS.get(section, {}):
                    config[section][key] = OPTION_DEFAULTS[section][key]
                    continue
                if section == 'gumbel':
                    config[section][key] = GUMBEL_DEFAULTS[key]
                    continue
                defaults = {("graph_search", "graph_search_rep_bound"): "11", ("run", "run_dir"): "", ("training", "disable_optimistic_policy"): "false", ("policy_init", "policy_init_mean"): "12", ("policy_init", "policy_temperature"): "1",
                            ("policy_init", "policy_after"): "true", ("policy_init", "policy_on_failure"): "true"}
                if (section,key) not in defaults:
                    raise ValueError(f"Missing required key: {section}.{key}")
                values[section,key] = defaults[section,key]
            try:
                config[section][key] = convert(values[(section, key)])
            except ValueError as error:
                raise ValueError(f"Invalid {section}.{key}: {error}") from error
    hints=config['hint_positions']
    for key in ('files', 'excludes'):
        config['start_positions'][key] = ','.join(str((directory/Path(path)).resolve()) for path in csv(config['start_positions'][key]))
    if hints['positions_file']:
        path=Path(hints['positions_file']);path=path if path.is_absolute() else directory/path
        hints['positions_file']=str(path.resolve())
        if hints['hint_positions_prob'] > 0 and not path.is_file():raise ValueError('Missing hint positions file')
    validate(config)
    if run_dir is not None:
        if not str(run_dir).strip():
            raise ValueError("--run-dir must not be empty")
        destination = Path(run_dir)
    elif config['run']['run_dir']:
        destination = ROOT/config['run']['run_dir']
    else:
        try:
            relative = directory.relative_to((ROOT/'configs').resolve())
        except ValueError as error:
            raise ValueError(f"Configuration outside {ROOT/'configs'} requires an explicit run_dir or --run-dir: {directory}") from error
        destination = ROOT/'data'/relative
    config['patterns']['pattern_corpora']=resolve_pattern_corpora(config['patterns']['pattern_corpora'],directory)
    if config['agent']['algorithm']=='muzero' and (config['patterns']['avoid_repeated_pattern_utility']!=0 or config['patterns']['pattern_corpora']!='[]'):
        raise ValueError('Pattern bonuses require a real-board AlphaZero search')
    if config['human_sl']['human_model']:
        path=Path(config['human_sl']['human_model']);config['human_sl']['human_model']=str((directory/path).resolve() if not path.is_absolute() else path)
    config['run']['run_dir'] = str(destination.resolve())
    return config


def csv(value, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def validate_hex_opening(o, *, match=False):
    if any(not math.isfinite(v) for v in o.values()) or any(not 0<=o[k]<=1 for k in ('probability','make_fair_probability')) or not 0<=o['balance_exponent']<=100 or not 0<o['min_accept_rate']<=1:
        raise ValueError('Invalid Hex opening gates, exponent or minimum acceptance rate')
    if match and (o['probability']!=1 or o['make_fair_probability']!=1):
        raise ValueError('Hex matches require both balanced opening probabilities to be one')


def validate_game_symmetry(c, spec):
    if not 1<=c['root_num_symmetries_to_sample']<=spec.symmetries or not 0<=c['nn_symmetry']<spec.symmetries:
        raise ValueError(f'{spec.name} supports {spec.symmetries} symmetries; nn_symmetry must be in [0,{spec.symmetries-1}]')


# Lower bounds belong to the module that consumes each field.
NUMERIC_MINIMUMS = {
    'run': (('cpu_threads',), ('seed', 'max_iteration', 'max_seconds')),
    'environment': ((), ('forbidden_feature_dropout_prob',)),
    'network': (('canvas', 'channels'), ('blocks',)),
    'muzero': (('latent_channels', 'dynamics_channels', 'prediction_channels'), ('dynamics_blocks', 'prediction_blocks')),
    'unroll': (('steps',), ('hidden_gradient_scale',)),
    'muzero_training': (('learning_rate',), ('weight_decay',)),
    'plain_optimizer': (('learning_rate',), ('weight_decay',)),
    'training': (('train_steps', 'batch_size', 'sub_epochs', 'log_every', 'checkpoint_keep', 'replay_ratio'), ('checkpoint_every', 'prefetch_depth', 'max_validation_samples', 'gradient_clip', 'soft_policy_weight_scale')),
    'optimizer': (('lr_scale', 'head_lr_factor', 'noreg_lr_factor', 'norm_interval', 'lookahead_k', 'lookahead_alpha', 'swa_scale'), ('input_wd_factor', 'normal_wd_factor', 'normal_attn_wd_factor', 'swa_period_samples')),
    'replay': (('min_rows', 'taper_exponent', 'keep_target_rows', 'max_rows'), ('expand_per_row', 'taper_scale')),
    'shuffle': (('workers', 'group_rows', 'bucket_rows', 'training_shard_rows', 'waves', 'snapshot_keep'), ()),
    'search': (('full_search_visits', 'cheap_search_visits'), ('cheap_search_probs', 'cheap_search_target_weight', 'max_playouts', 'max_time')),
    'graph_search': ((), ('graph_search_catch_up_leak_prob',)),
    'uncertainty': (('uncertainty_coeff', 'uncertainty_max_weight'), ('uncertainty_exponent',)),
    'optimistic_policy': ((), ('policy_optimism', 'root_policy_optimism')),
    'noise_pruning': (('noise_prune_utility_scale',), ('noise_pruning_cap',)),
    'subtree_value_bias': (('subtree_value_bias_table_num_shards',), ('subtree_value_bias_factor','subtree_value_bias_free_prop','subtree_value_bias_weight_exponent')),
    'reanalysis': ((), ('reanalyze_prop', 'reanalyze_policy_surprise_weight', 'reanalyze_value_surprise_weight', 'reanalyze_surprise_exponent')),
    'hint_positions': ((), ('hint_positions_prob',)),
    'game_forks': (('fork_game_min_choices', 'early_fork_game_max_choices', 'fork_game_max_choices'), ('early_fork_game_prob', 'fork_game_prob', 'early_fork_game_expected_move_prop')),
    'side_positions': ((), ('side_position_prob',)),
    'pda': (('max_asymmetric_ratio',), ('normal_asymmetric_playout_prob',)),
    'selfplay': (('bootstrap_games',), ()),
    'reduce_visits': (('reduced_visits_min',), ('reduce_visits_threshold_lookback','reduce_visits_threshold', 'reduced_visits_weight')),
    'puct': (('c_puct_base', 'c_puct_stdev_prior'), ('c_puct','c_puct_log', 'c_puct_stdev_prior_weight', 'c_puct_stdev_scale', 'virtual_loss')),
    'fpu': ((), ('fpu_reduction_max', 'root_fpu_reduction_max', 'fpu_parent_weight_by_visited_policy_pow', 'fpu_loss_prop', 'root_fpu_loss_prop', 'fpu_parent_weight')),
    'value_weighting': ((), ('value_weight_exponent',)),
    'forced_playouts': ((), ('root_desired_per_child_visits_coeff',)),
    'policy_target': ((), ('chosen_move_subtract', 'chosen_move_prune')),
    'lcb': (('lcb_stdevs',), ('min_visit_prop_for_lcb',)),
    'symmetry': (('root_num_symmetries_to_sample',), ('nn_symmetry',)),
    'temperature': (('nn_policy_temperature', 'root_policy_temperature_early', 'root_policy_temperature', 'temperature_halflife'), ('temperature', 'final_temperature', 'temperature_only_below_prob')),
    'dirichlet_noise': (('dirichlet_total_concentration',), ('noise_fraction',)),
    'opening': (('max_tries',), ('probability', 'avg_dist_factor', 'balance_exponent', 'rejection_probability', 'rejection_probability_fallback')),
    'hex_opening': (('min_accept_rate',), ('probability', 'make_fair_probability', 'balance_exponent')),
    'policy_init': (('policy_temperature',), ('policy_init_mean',)),
    'surprise_weighting': ((), ('policy_surprise_data_weight', 'value_surprise_data_weight')),
    'parallelism': (('game_threads', 'search_threads'), ()),
    'inference': (('server_threads', 'max_batch', 'muzero_queue_capacity'), ('cache_entries', 'batch_wait_us')),
    'writer': (('writer_queue', 'shard_rows'), ('first_file_min_random_proportion',)),
}

def validate(c):
    from .schema import RULES
    a = c["agent"]
    if not 0 <= c["run"]["seed"] < 2**64:
        raise ValueError("Run seed must fit uint64")
    if a["algorithm"] not in ("alphazero", "muzero"):
        raise ValueError("Invalid algorithm/search enum")
    if 'plain_optimizer' in c and a['algorithm'] != 'alphazero':
        raise ValueError('plain_optimizer requires AlphaZero; MuZero uses muzero_training')
    validate_search_combination(a['root_search_algo'], a['nonroot_search_algo'])
    if a['root_search_algo'] == 'gumbel':
        validate_gumbel(c.setdefault('gumbel', GUMBEL_DEFAULTS.copy()))
        if c['graph_search']['use_graph_search'] or c['search']['reuse_tree']:
            raise ValueError('Gumbel requires use_graph_search=false and reuse_tree=false')
    if a['algorithm'] == 'muzero':
        if c['graph_search']['use_eval_cache']: raise ValueError('MuZero has no real-board graph evaluation cache')
        if c['patterns']['avoid_repeated_pattern_utility'] or c['patterns']['pattern_corpora']!='[]': raise ValueError('Pattern bonuses require real-board AlphaZero search')
        if c['subtree_value_bias']['subtree_value_bias_factor'] != 0:
            raise ValueError('MuZero latent search requires subtree_value_bias_factor=0')
        from .muzero.network import network_config
        network_config(c)
        if c['unroll']['steps'] < 1 or not 0 <= c['unroll']['hidden_gradient_scale'] <= 1:
            raise ValueError('MuZero unroll.steps must be positive, hidden_gradient_scale in [0,1]')
        if not c.get('muzero_training', {}).get('auxiliary_losses', True) and c['network']['predict_q_values']:
            raise ValueError('MuZero without auxiliary losses requires predict_q_values=false')
        if c['graph_search']['use_graph_search'] or c['search']['reuse_tree'] or c['symmetry']['root_num_symmetries_to_sample'] != 1:
            raise ValueError('MuZero requires use_graph_search=false, reuse_tree=false and root_num_symmetries_to_sample=1')
    inactive = set()
    for section, flag in (('uncertainty','use_uncertainty'), ('noise_pruning','use_noise_pruning'),
                          ('fpu','use_fpu'), ('reanalysis','use_reanalyze'), ('reduce_visits','reduce_visits')):
        if not c[section][flag]:
            inactive.add(section)
    # Opening/policy-init/fork objects are constructed even when disabled;
    # retain their native numeric constraints at configuration load time.
    for section, fields in c.items():
        for key, value in fields.items():
            if isinstance(value, (int,float)) and not isinstance(value,bool) and not math.isfinite(value):
                raise ValueError(f'{section}.{key} must be finite')
        if section in inactive:
            continue
        positive, nonnegative = NUMERIC_MINIMUMS.get(section, ((),()))
        for keys, allow_zero in ((positive,False),(nonnegative,True)):
            for key in keys:
                if section == "inference" and key == "muzero_queue_capacity" and a["algorithm"] != "muzero":
                    continue
                if section == 'search' and key in ('cheap_search_visits','cheap_search_target_weight') and c['search']['cheap_search_probs'] == 0:
                    continue
                value = fields[key]
                if value == 'all': continue
                if value < 0 or (not allow_zero and value == 0):
                    raise ValueError(f'{section}.{key} must be {"nonnegative" if allow_zero else "positive"}')
    env, net = c["environment"], c["network"]
    # MuZero ResNet keeps the NBT head width rules; the AZ presets are unchanged.
    from .katago_configs import is_preset, resolve, validate_dimensions
    named = a['algorithm'] == 'alphazero' and is_preset(net['architecture'])
    if named:
        preset = resolve(net['architecture'])
        validate_dimensions(preset)
        for key, actual in (('channels', preset['trunk_num_channels']), ('blocks', len(preset['block_kind']))):
            if net[key] != actual:
                raise ValueError(f"{net['architecture']} fixes network.{key}={actual}, got {net[key]}")
        if net['predict_q_values'] and preset['version'] < 17:
            raise ValueError('predict_q_values requires a KataGo v17 preset')
    else:
        network_widths(net["channels"], 'nbt' if a['algorithm'] == 'muzero' else net['architecture'])
    if 'brenorm' in c:
        if not named or preset['norm_kind'] not in ('brenorm', 'fixbrenorm'):
            raise ValueError('brenorm configuration requires a named KataGo batch-renorm preset')
        brn = c['brenorm']
        if not 0 < brn['avg_momentum'] <= 1 or brn['target_rmax'] < 1 or brn['target_dmax'] < 0 or brn['adjustment_scale'] <= 0:
            raise ValueError('Invalid batch-renorm schedule')
    if a['algorithm']=='alphazero' and not named and net['predict_q_values'] and net['architecture']!='transformer':
        raise ValueError('predict_q_values requires the v17 Transformer preset')
    if net['architecture'] == 'plain' and net['blocks'] != 10:
        raise ValueError('plain requires b10c128-fson-mish: channels=128, blocks=10')
    if net['architecture'] == 'transformer' and net['blocks'] != 5:
        raise ValueError('transformer requires b5c192h3nbttfrs: channels=192, blocks=5')
    sizes = csv(env["sizes"], int)
    rules = csv(env["rules"])
    if not sizes or len(set(sizes)) != len(sizes) or any(x < 5 or x > net["canvas"] for x in sizes) or not 5 <= net["canvas"] <= 25:
        raise ValueError("Sizes must be unique, in [5, canvas], with canvas <= 25")
    if not rules or len(set(rules)) != len(rules) or any(x not in RULES for x in rules):
        raise ValueError("Invalid rule distribution")
    from .games import spec_for_config, spec_for_rule
    spec = spec_for_config(c)
    if not spec.subtree_value_bias and c['subtree_value_bias']['subtree_value_bias_factor'] != 0:
        raise ValueError(f'{spec.name} does not support subtree value bias; requires subtree_value_bias_factor=0')
    if 'xiangqi' in rules:
        if sizes != [10] or net['canvas'] != 10:
            raise ValueError('Xiangqi requires sizes=10 and canvas=10 (9 files, 10 ranks)')
    if 'go' in c:
        from .games import validate_go_komi
        validate_go_komi(c['go']['komi'])
    for label, entries in (("size", sizes), ("rule", rules)):
        weights = csv(env[label + "_weights"], float)
        if len(weights) != len(entries) or any(not math.isfinite(x) or x < 0 for x in weights) or not any(x > 0 for x in weights):
            raise ValueError(f"Invalid {label} weights")
    validate_hex_opening(c['hex_opening'])
    for rule, weight in zip(rules, csv(env['rule_weights'],float)):
        if weight>0:validate_game_symmetry(c['symmetry'],spec_for_rule(rule))
    if c['dirichlet_noise']['noise_fraction'] > 1 or c['environment']['forbidden_feature_dropout_prob'] > 1:
        raise ValueError("Noise fraction and feature dropout must be <= 1")
    if c['writer']['first_file_min_random_proportion'] > 1:
        raise ValueError('writer.first_file_min_random_proportion must be in [0,1]')
    search = c['search']
    validate_search_parameters({key: value for section in ('search', 'graph_search', 'uncertainty', 'optimistic_policy', 'noise_pruning', 'subtree_value_bias', 'fpu', 'puct', 'symmetry', 'temperature','parallelism','search_extras','search_history','time_control','human_sl','patterns')
                                for key, value in c[section].items()})
    if (not 1 <= search['full_search_visits'] <= 2**50 or search['cheap_search_probs'] > 1
            or (search['cheap_search_probs'] > 0 and not 1 <= search['cheap_search_visits'] <= min(10000000,search['full_search_visits']))
            or (search['cheap_search_probs'] > 0 and search['cheap_search_target_weight'] > 1) or c['lcb']['min_visit_prop_for_lcb'] > 1):
        raise ValueError('Invalid cheap search cap/probability/weight or LCB visit proportion')
    if (search['cheap_search_probs'] == 1 and search['cheap_search_target_weight'] == 0 and
            not (c['reanalysis']['use_reanalyze'] and c['reanalysis']['reanalyze_prop']>0)):
        raise ValueError('All-cheap zero-weight searches produce no training rows')
    if c['reduce_visits']['reduce_visits'] and (c['reduce_visits']['reduce_visits_threshold'] > 0.999999 or
            not 0 <= c['reduce_visits']['reduce_visits_threshold_lookback'] <= 1000 or
            not 1 <= c['reduce_visits']['reduced_visits_min'] <= min(10000000,search['full_search_visits']) or
            c['reduce_visits']['reduced_visits_weight'] > 1):
        raise ValueError('Invalid Reduce Visits threshold/lookback/minimum/weight')
    forks=c['game_forks'];hints=c['hint_positions']
    if any(forks[k]>.5 for k in ('early_fork_game_prob','fork_game_prob')) or not 0<=hints['hint_positions_prob']<=1:
        raise ValueError('Invalid hint/game fork probability')
    if hints['hint_positions_prob']>0 and not hints['positions_file']:
        raise ValueError('Enabled hint sampling requires positions_file')
    if not 1<=forks['fork_game_min_choices']<=min(forks['early_fork_game_max_choices'],forks['fork_game_max_choices']) or max(forks['early_fork_game_max_choices'],forks['fork_game_max_choices'])>100 or forks['early_fork_game_expected_move_prop']>1:
        raise ValueError('Invalid game fork candidate range')
    ra=c['reanalysis']
    if ra['use_reanalyze']:
        for key,limit in (('reanalyze_prop',1),('reanalyze_policy_surprise_weight',100),('reanalyze_value_surprise_weight',100),('reanalyze_surprise_exponent',10)):
            if not 0<=ra[key]<=limit:raise ValueError(f'Invalid reanalysis field: {key}')
    if not 0<=c['side_positions']['side_position_prob']<=1:
        raise ValueError('Invalid side position probability')
    if not 0<=c['pda']['normal_asymmetric_playout_prob']<=1 or not 1<=c['pda']['max_asymmetric_ratio']<=100:
        raise ValueError('Invalid PDA probability/maximum ratio')
    if c['pda']['normal_asymmetric_playout_prob'] > 0 and c['pda']['max_asymmetric_ratio'] > 1:
        # Bound every attainable PCR / Reduce Visits budget before sampling a
        # game. Positive std::round in the native path is floor(x + 0.5).
        # KataGo makes PCR and Reduce Visits exclusive. Hints lower PCR's
        # probability for six turns; reanalysis can explicitly force full.
        full_possible = (search['cheap_search_probs'] < 1 or hints['hint_positions_prob'] > 0 or
                         (ra['use_reanalyze'] and ra['reanalyze_prop'] > 0))
        budgets = []
        if full_possible:
            budgets.append(search['full_search_visits'])
            if c['reduce_visits']['reduce_visits']:
                budgets.append(c['reduce_visits']['reduced_visits_min'])
            budgets.append(search['max_playouts'])
        if search['cheap_search_probs'] > 0:
            budgets.append(search['cheap_search_visits'])
        if hints['hint_positions_prob'] > 0:
            budgets.append(min(2**50,4 * search['full_search_visits']))
            budgets.append(min(2**50,4 * search['max_playouts']))
        ratio = c['pda']['max_asymmetric_ratio']
        if (math.floor(min(budgets) * 2 / (1 + ratio) + .5) < 5 or
                math.floor(max(budgets) * 2 * ratio / (1 + ratio) + .5) > 2**62):
            raise ValueError('PDA-scaled full/cheap/reduced/hint/playout budgets must stay in [5, 2^62]')
    if c['surprise_weighting']['policy_surprise_data_weight'] + c['surprise_weighting']['value_surprise_data_weight'] > 1:
        raise ValueError('Surprise weight fractions must sum to <= 1')
    if c["training"]["amp"] not in ("off", "float16", "bfloat16"):
        raise ValueError("Invalid optimizer or AMP mode")
    o = c['optimizer']
    start=c['start_positions']
    if (not 0<=start['probability']<=1 or not 0<=start['load_probability']<=1 or not -10<=start['turn_weight_lambda']<=10 or
            not 0<=start['policy_area_proportion']<=1 or not .5<=start['policy_gamma_shape']<=10 or not .1<=start['policy_temperature']<=5):
        raise ValueError('Invalid external starting-position configuration')
    if start['probability']>0 and not start['files']:
        raise ValueError('External starting positions require files')
    if not .01<=c['data_weight']['scale_data_weight']<=10:
        raise ValueError('Invalid data weight scale')
    for key, value in c.get('losses', {}).items():
        if not math.isfinite(value) or value < (-1 if key.endswith('head_scale') else 0):
            raise ValueError(f'Invalid losses.{key}')
    if o['kind'] not in ('sgd', 'adamw', 'muon', 'normuon', 'aurora'):
        raise ValueError('Optimizer kind must be sgd, adamw, muon, normuon or aurora')
    if (o['muon_adam_lr_factor'] < 0 or o['wd_floor_frac'] < 0 or o['ns_steps'] < 1 or
            o['aurora_pp_iterations'] < 1 or not 0 <= o['aurora_pp_beta'] <= 1):
        raise ValueError('Invalid Muon optimizer options')
    if o['wd_floor_frac'] > 0 and o['kind'] not in ('muon','normuon','aurora'):
        raise ValueError('Weight-decay floors require a Muon-family optimizer')
    from .lr_schedule import validate as validate_lr_schedule
    validate_lr_schedule(c['lr_schedule'])
    if c['lr_schedule']['mode']!='constant' and ('plain_optimizer' in c or not c.get('muzero_training',{}).get('katago_optimizer',True)):
        raise ValueError('Fixed plain optimizer does not use the KataGo learning-rate schedule')
    if c['lr_schedule']['mode'] != 'constant' and o['lr_scale'] != 1.0:
        raise ValueError('Scheduled LR requires optimizer.lr_scale=1')
    penalty = c['attention_penalty']
    if (not all(math.isfinite(x) for x in penalty.values()) or penalty['cap'] < -1 or
            penalty['coefficient'] < 0 or not 0 < penalty['batch_fraction'] <= 1):
        raise ValueError('Invalid attention penalty')
    if not 0 < o['lookahead_alpha'] <= 1 or o['swa_scale'] < 1:
        raise ValueError('lookahead_alpha must be in (0,1] and swa_scale must be >= 1')
    if c['inference']['inference_precision'] not in ("float32", "float16"):
        raise ValueError("Invalid inference precision")
    if not -1<=c['inference']['nn_mutex_pool_power_of_two']<=24:
        raise ValueError('Invalid NN cache mutex pool power')
    if c['inference']['inference_precision'] == "float16" and "cpu" in csv(c["devices"]["selfplay"]):
        raise ValueError("FP16 inference requires CUDA selfplay devices")
    opening, policy_init = c["opening"], c["policy_init"]
    if any(opening[k] > 1 for k in ("probability", "rejection_probability", "rejection_probability_fallback")):
        raise ValueError("Opening probabilities must be in [0,1]")
    if opening["max_tries"] > 1000 or any(opening[k] > 100 for k in ("avg_dist_factor", "balance_exponent")):
        raise ValueError("Invalid opening configuration")
    if policy_init['policy_init_mean'] > 100 or not .1 <= policy_init['policy_temperature'] <= 5:
        raise ValueError('Invalid policy init configuration')
    r, s = c["replay"], c["shuffle"]
    if not 0 < r["taper_exponent"]:
        raise ValueError("Invalid replay window exponent")
    if r["keep_target_rows"] != "all" and not isinstance(r["keep_target_rows"],int):
        raise ValueError("keep_target_rows must be a positive integer or all")
    if r['max_rows']!='all' and (not isinstance(r['max_rows'],int) or r['max_rows']<r['min_rows']):
        raise ValueError('replay.max_rows must be all or >= min_rows')
    scale=r['taper_scale'] or r['min_rows']
    if scale+r['add_to_data_rows']<=0:
        raise ValueError('replay taper_scale + add_to_data_rows must be positive')
    if s["bucket_rows"] % s["training_shard_rows"] != 0:
        raise ValueError("shuffle.bucket_rows must be a multiple of training_shard_rows")
    devices = csv(c["devices"]["selfplay"]) + [c["devices"]["train"]]
    if not devices or any(x != "cpu" and not (x.startswith("cuda:") and x[5:].isdigit()) for x in devices):
        raise ValueError("Devices must be explicit cpu or cuda:<index>")
    if not csv(c["devices"]["selfplay"]):
        raise ValueError("At least one selfplay device is required")


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def native_text(config):
    import io
    parser = configparser.ConfigParser()
    for section, fields in config.items():
        parser[section] = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in fields.items()}
    out=io.StringIO();parser.write(out)
    return out.getvalue()


def write_native(config, path):
    from .storage import atomic_write
    atomic_write(path,lambda p:p.write_text(native_text(config)))


def resume_config(config):
    """State/layout requirements, independent of paths and execution resources."""
    sections = ('agent', 'network', 'muzero', 'unroll', 'muzero_training', 'plain_optimizer', 'brenorm', 'losses', 'lr_schedule', 'attention_penalty', 'optimizer', 'training')
    result = {name: dict(config[name]) for name in sections if name in config}
    for key in ('log_every','checkpoint_every','checkpoint_keep','prefetch_depth','cuda_prefetch','compile',
                'max_validation_samples','randomize_validation_files'):
        result['training'].pop(key, None)
    # lookahead_print selects norm samples and therefore adaptive weight decay.
    # It is part of optimizer identity even though its name suggests logging.
    result['seed'] = config['run']['seed']
    return result
