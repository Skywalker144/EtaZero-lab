"""Strict, directory-based INI configuration. Native consumers receive resolved values."""
import configparser
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


# Each section belongs to exactly one file; baseline supplies every required key.
# run_dir is optional and belongs only to the selected configuration directory.
SEARCH_PARAMETERS = dict(use_uncertainty=boolean, uncertainty_coeff=float, uncertainty_exponent=float, uncertainty_max_weight=float, policy_optimism=float, root_policy_optimism=float, use_noise_pruning=boolean, noise_prune_utility_scale=float, noise_pruning_cap=float, use_graph_search=boolean, graph_search_catch_up_leak_prob=float, max_playouts=int, max_time=float, nn_randomize=boolean, nn_symmetry=int,
                         fpu_parent_weight_by_visited_policy=boolean, fpu_parent_weight=float, value_weight_exponent=float, chosen_move_subtract=float, chosen_move_prune=float,
                         fpu_loss_prop=float, root_fpu_loss_prop=float, c_puct_log=float, c_puct_base=float,
                         c_puct_stdev_prior=float, c_puct_stdev_prior_weight=float, c_puct_stdev_scale=float,
                         root_num_symmetries_to_sample=int, nn_policy_temperature=float,
                         root_policy_temperature_early=float, root_policy_temperature=float,
                         temperature_halflife=float, temperature_only_below_prob=float)


def validate_search_parameters(c):
    bounds={'uncertainty_coeff':(.0001,1),'uncertainty_exponent':(0,2),'uncertainty_max_weight':(1,100),
            'policy_optimism':(0,1),'root_policy_optimism':(0,1),'noise_prune_utility_scale':(.001,10),'noise_pruning_cap':(0,1e50)}
    for key,(low,high) in bounds.items():
        if not math.isfinite(c[key]) or not low<=c[key]<=high:
            raise ValueError(f'Invalid search correction: {key}')
    if not 0 <= c['nn_symmetry'] < 8 or not 0 <= c['max_playouts'] <= 2**31-1:
        raise ValueError('Invalid NN symmetry or playout cap')
    if not 1 <= c['root_num_symmetries_to_sample'] <= 8:
        raise ValueError('Root symmetry count must be in [1,8]')
    if any(c[k] > 1 for k in ('graph_search_catch_up_leak_prob','fpu_loss_prop','root_fpu_loss_prop','c_puct_stdev_scale','temperature_only_below_prob','fpu_parent_weight')):
        raise ValueError('Search proportions must be <= 1')
    if any(c[k] <= 0 for k in ('c_puct_base','c_puct_stdev_prior','nn_policy_temperature',
                              'root_policy_temperature','root_policy_temperature_early','temperature_halflife')):
        raise ValueError('Policy temperatures, halflife and PUCT scales must be positive')

FIELDS = {
    "run": ("run", {"run_dir": str, "seed": int, "max_iteration": int,
                    "max_seconds": float, "cpu_threads": int}),
    "agent": ("run", {"algorithm": str, "root_search_algo": str, "nonroot_search_algo": str}),
    "devices": ("run", {"train": str, "selfplay": str}),
    "environment": ("env", {"sizes": str, "size_weights": str, "rules": str, "rule_weights": str,
                              "forbidden_feature_dropout_prob": float}),
    "network": ("net", {"architecture": str, "canvas": int, "channels": int, "blocks": int, "predict_q_values":boolean}),
    "muzero": ("net", {"latent_channels": int, "dynamics_channels": int, "dynamics_blocks": int,
                         "prediction_channels": int, "prediction_blocks": int}),
    "unroll": ("train", {"steps": int, "hidden_gradient_scale": float}),
    "muzero_training": ("train", {"auxiliary_losses": boolean, "katago_optimizer": boolean,
                                  "learning_rate": float, "weight_decay": float}),
    "training": ("train", {"train_steps": int, "batch_size": int, "prefetch_depth": int,
                           'cuda_prefetch':boolean,'compile':boolean,'sub_epochs':int,'no_repeat_files':boolean,'skip_validation':boolean,'randomize_validation_files':boolean,'max_validation_samples':int,
                           "checkpoint_every": int, "checkpoint_keep": int, "amp": str, "gradient_clip": float,
                           "replay_ratio": float, "d4_augmentation": boolean, "soft_policy_weight_scale": float, "disable_optimistic_policy": boolean}),
    "optimizer": ("train", {"kind": str, "lr_scale": float, "lr_warmup": boolean,
                             "head_lr_factor": float, "noreg_lr_factor": float,
                             "input_wd_factor": float, "normal_wd_factor": float, "normal_attn_wd_factor": float,
                             "norm_interval": int, "norm_only_at_print": boolean, "lookahead_print": boolean,
                             "lookahead_k": int, "lookahead_alpha": float,
                             "swa_period_samples": int, "swa_scale": float}),
    "replay": ("train", {"min_rows": int, "taper_exponent": float,
                         "expand_per_row": float, "keep_target_rows": row_target, "taper_scale":float, "add_to_data_rows":float, "max_rows":row_target}),
    "shuffle": ("train", {"workers": int, "group_rows": int, "bucket_rows": int,
                          "training_shard_rows": int, "waves": int, "memory_mb": int,
                          "temp_dir": str, "snapshot_keep": int, "compress_temp": boolean}),
    'search': ("selfplay", {'full_search_visits': int, 'cheap_search_visits': int,
                            'cheap_search_probs': float, 'cheap_search_target_weight': float,
                            'reuse_tree': boolean, 'clear_before_search': boolean, 'max_playouts': int, 'max_time': float}),
    'graph_search': ("selfplay", {'use_graph_search': boolean, 'graph_search_catch_up_leak_prob': float}),
    'uncertainty': ("selfplay", dict(use_uncertainty=boolean,uncertainty_coeff=float,uncertainty_exponent=float,uncertainty_max_weight=float)),
    'optimistic_policy': ("selfplay", dict(policy_optimism=float,root_policy_optimism=float)),
    'noise_pruning': ("selfplay", dict(use_noise_pruning=boolean,noise_prune_utility_scale=float,noise_pruning_cap=float)),
    'reanalysis': ("selfplay", dict(use_reanalyze=boolean,reanalyze_prop=float,reanalyze_policy_surprise_weight=float,reanalyze_value_surprise_weight=float,reanalyze_surprise_exponent=float,reanalyze_use_outcome_targets=boolean)),
    'hint_positions': ("selfplay", dict(hint_positions_prob=float,positions_file=str)),
    'game_forks': ("selfplay", dict(early_fork_game_prob=float,fork_game_prob=float,early_fork_game_expected_move_prop=float,
                                  fork_game_min_choices=int,early_fork_game_max_choices=int,fork_game_max_choices=int)),
    'side_positions': ("selfplay", dict(side_position_prob=float)),
    'pda': ("selfplay", dict(normal_asymmetric_playout_prob=float,max_asymmetric_ratio=float)),
    'selfplay': ("selfplay", {'bootstrap_games': int, 'recent_games': int}),
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
    'policy_init': ("selfplay", {'policy_init': boolean, 'policy_after': boolean,
                                 'policy_on_failure': boolean, 'policy_init_mean': float,
                                 'policy_temperature': float}),
    'surprise_weighting': ("selfplay", {'policy_surprise_data_weight': float,
                                        'value_surprise_data_weight': float, 'use_search_value_surprise': boolean}),
    'parallelism': ("selfplay", {'game_threads': int, 'search_threads': int}),
    'inference': ("selfplay", {'server_threads': int, 'max_batch': int, 'inference_precision': str,
                               'cache_entries': int, 'batch_wait_us': int, 'queue_capacity': int}),
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
        if not path.exists():
            continue
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string(path.read_text())
        if parser.defaults():
            raise ValueError(f"DEFAULT fields are not supported: {path}")
        for section in parser.sections():
            if section not in FIELDS or FIELDS[section][0] != name:
                raise ValueError(f"Wrong section/file ownership: {path}: [{section}]")
            for key, value in parser.items(section):
                if section == "run" and key == "extends" and not local:
                    values[(section, key)] = value
                    continue
                if key not in FIELDS[section][1]:
                    raise ValueError(f"Unknown key: {path}: {section}.{key}")
                values[(section, key)] = value
    return values


def load_config(directory, run_dir=None):
    directory = Path(directory).resolve()
    configs_root = directory.parent

    def inherit(current, ancestors):
        if current in ancestors:
            raise ValueError(f"Configuration inheritance cycle: {current}")
        if not (current / "run.cfg").exists():
            raise ValueError(f"Missing configuration directory/run.cfg: {current}")
        values = _read(current)
        parent = values.pop(("run", "extends"), None)
        result = {}
        if parent:
            if Path(parent).name != parent or parent in (".", ".."):
                raise ValueError("extends must name a directory under configs/")
            parent_dir = configs_root / parent
            if not (parent_dir / "run.cfg").is_file():
                parent_dir = ROOT / "configs" / parent
            result.update(inherit(parent_dir, ancestors + [current]))
            # Output locations belong to the selected config, not its parent.
            result.pop(("run", "run_dir"), None)
        result.update(values)
        # Machine overrides are deliberately applied only at the selected directory.
        return result

    values = inherit(directory, [])
    values.update(_read(directory, local=True))
    config = {}
    for section, (_, fields) in FIELDS.items():
        if section == 'muzero_training' and not any(s == section for s, _ in values):
            continue
        if section in ('muzero', 'unroll', 'muzero_training') and values.get(('agent', 'algorithm')) != 'muzero':
            if any(s == section for s, _ in values):
                raise ValueError(f'{section} configuration requires agent.algorithm = muzero')
            continue
        config[section] = {}
        for key, convert in fields.items():
            if section=="reanalysis" and key!="use_reanalyze" and (section,key) not in values and not config[section]["use_reanalyze"]:
                continue
            if (section, key) not in values:
                defaults = {("run", "run_dir"): "", ("training", "disable_optimistic_policy"): "false", ("policy_init", "policy_init_mean"): "12", ("policy_init", "policy_temperature"): "1",
                            ("policy_init", "policy_after"): "true", ("policy_init", "policy_on_failure"): "true"}
                if (section,key) not in defaults:
                    raise ValueError(f"Missing required key: {section}.{key}")
                values[section,key] = defaults[section,key]
            try:
                config[section][key] = convert(values[(section, key)])
            except ValueError as error:
                raise ValueError(f"Invalid {section}.{key}: {error}") from error
    hints=config['hint_positions']
    if hints['positions_file']:
        path=Path(hints['positions_file']);path=path if path.is_absolute() else directory/path
        hints['positions_file']=str(path.resolve())
        if not path.is_file():raise ValueError('Missing hint positions file')
        hints['positions_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
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
    config['run']['run_dir'] = str(destination.resolve())
    return config


def csv(value, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def validate(c):
    from .schema import RULES
    a = c["agent"]
    if not 0 <= c["run"]["seed"] < 2**64:
        raise ValueError("Run seed must fit uint64")
    if a["algorithm"] not in ("alphazero", "muzero") or any(
            a[k] not in ("puct", "gumbel") for k in ("root_search_algo", "nonroot_search_algo")):
        raise ValueError("Invalid algorithm/search enum")
    if a["root_search_algo"] == "puct" and a["nonroot_search_algo"] == "gumbel":
        raise ValueError("PUCT root + Gumbel nonroot is not an allowed project combination")
    if any(a[k] != 'puct' for k in ('root_search_algo', 'nonroot_search_algo')):
        raise ValueError("Selected algorithm/search combination is not implemented in this version")
    if a['algorithm'] == 'muzero':
        from .muzero.network import network_config
        network_config(c)
        if not 1 <= c['unroll']['steps'] <= 32 or not 0 <= c['unroll']['hidden_gradient_scale'] <= 1:
            raise ValueError('MuZero unroll.steps must be in [1,32], hidden_gradient_scale in [0,1]')
        if c['network']['architecture'] != 'nbt':
            raise ValueError('MuZero requires the NBT architecture')
        if not c.get('muzero_training', {}).get('auxiliary_losses', True) and c['network']['predict_q_values']:
            raise ValueError('MuZero without auxiliary losses requires predict_q_values=false')
        if c['graph_search']['use_graph_search'] or c['search']['reuse_tree'] or c['symmetry']['root_num_symmetries_to_sample'] != 1:
            raise ValueError('MuZero requires use_graph_search=false, reuse_tree=false and root_num_symmetries_to_sample=1')
    for section, fields in c.items():
        for key, value in fields.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or (value < 0 and (section,key)!=("replay","add_to_data_rows")):
                    raise ValueError(f"{section}.{key} must be nonnegative and finite")
                zero_allowed = {"dynamics_blocks", "prediction_blocks", "hidden_gradient_scale", "first_file_min_random_proportion","hint_positions_prob","early_fork_game_prob","fork_game_prob","early_fork_game_expected_move_prop","reanalyze_prop","reanalyze_policy_surprise_weight","reanalyze_value_surprise_weight","reanalyze_surprise_exponent","side_position_prob","normal_asymmetric_playout_prob","uncertainty_exponent", "policy_optimism", "root_policy_optimism", "noise_pruning_cap", "graph_search_catch_up_leak_prob", "max_playouts", "max_time", "nn_symmetry", "fpu_parent_weight", "fpu_parent_weight_by_visited_policy_pow", "seed", "max_iteration", "max_seconds", "blocks", "batch_wait_us",
                                "weight_decay", "virtual_loss", "noise_fraction", "temperature", "temperature_early",
                                "final_temperature", "gradient_clip", "soft_policy_weight_scale", "swa_period_samples", "input_wd_factor", "normal_wd_factor", "normal_attn_wd_factor", "forbidden_feature_dropout_prob", "expand_per_row", "cache_entries",
                                "policy_surprise_data_weight", "value_surprise_data_weight", "cheap_search_probs",
                                "cheap_search_target_weight", "fpu_reduction_max", "root_fpu_reduction_max",
                                "reduce_visits_threshold", "reduced_visits_weight",
                                "min_visit_prop_for_lcb", "root_desired_per_child_visits_coeff", "value_weight_exponent",
                                "chosen_move_subtract", "chosen_move_prune", "fpu_loss_prop", "root_fpu_loss_prop", "c_puct_log",
                                "c_puct_stdev_prior_weight", "c_puct_stdev_scale", "temperature_only_below_prob",
                                "max_validation_samples","taper_scale","add_to_data_rows","probability", "avg_dist_factor", "balance_exponent", "rejection_probability",
                                "rejection_probability_fallback", "policy_init_mean"}
                if key not in zero_allowed and value == 0:
                    raise ValueError(f"{section}.{key} must be positive")
    if c['training']['sub_epochs']>c['training']['train_steps']:
        raise ValueError('training.sub_epochs cannot exceed the fixed round batch budget')
    env, net = c["environment"], c["network"]
    network_widths(net["channels"], net['architecture'])
    if a['algorithm']=='alphazero' and net['predict_q_values'] and net['architecture']!='transformer':
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
    for label, entries in (("size", sizes), ("rule", rules)):
        weights = csv(env[label + "_weights"], float)
        if len(weights) != len(entries) or any(not math.isfinite(x) or x < 0 for x in weights) or not any(x > 0 for x in weights):
            raise ValueError(f"Invalid {label} weights")
    if c['dirichlet_noise']['noise_fraction'] > 1 or c['environment']['forbidden_feature_dropout_prob'] > 1:
        raise ValueError("Noise fraction and feature dropout must be <= 1")
    if c['writer']['first_file_min_random_proportion'] > 1:
        raise ValueError('writer.first_file_min_random_proportion must be in [0,1]')
    search = c['search']
    validate_search_parameters({key: value for section in ('search', 'graph_search', 'uncertainty', 'optimistic_policy', 'noise_pruning', 'fpu', 'puct', 'symmetry', 'temperature')
                                for key, value in c[section].items()})
    if (not 2 <= search['full_search_visits'] <= 2**31 - 1 or search['cheap_search_probs'] > 1
            or not 2 <= search['cheap_search_visits'] <= search['full_search_visits']
            or search['cheap_search_target_weight'] > 1 or c['lcb']['min_visit_prop_for_lcb'] > 1):
        raise ValueError('Invalid cheap search cap/probability/weight or LCB visit proportion')
    if (search['cheap_search_probs'] == 1 and search['cheap_search_target_weight'] == 0 and
            not (c['reanalysis']['use_reanalyze'] and c['reanalysis']['reanalyze_prop']>0)):
        raise ValueError('All-cheap zero-weight searches produce no training rows')
    if (c['reduce_visits']['reduce_visits_threshold'] > 0.999999 or
            not 1 <= c['reduce_visits']['reduce_visits_threshold_lookback'] <= 1000 or
            not 2 <= c['reduce_visits']['reduced_visits_min'] <= search['full_search_visits'] or
            c['reduce_visits']['reduced_visits_weight'] > 1):
        raise ValueError('Invalid Reduce Visits threshold/lookback/minimum/weight')
    forks=c['game_forks'];hints=c['hint_positions']
    if any(forks[k]>1 for k in ('early_fork_game_prob','fork_game_prob')) or not 0<=hints['hint_positions_prob']<=1:
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
    if c['surprise_weighting']['policy_surprise_data_weight'] + c['surprise_weighting']['value_surprise_data_weight'] > 1:
        raise ValueError('Surprise weight fractions must sum to <= 1')
    if c["training"]["amp"] not in ("off", "float16", "bfloat16"):
        raise ValueError("Invalid optimizer or AMP mode")
    o = c['optimizer']
    if o['kind'] not in ('sgd', 'adamw'):
        raise ValueError('Optimizer kind must be sgd or adamw')
    if not 0 < o['lookahead_alpha'] <= 1 or o['swa_scale'] < 1:
        raise ValueError('lookahead_alpha must be in (0,1] and swa_scale must be >= 1')
    if c['inference']['inference_precision'] not in ("float32", "float16"):
        raise ValueError("Invalid inference precision")
    if c['inference']['inference_precision'] == "float16" and "cpu" in csv(c["devices"]["selfplay"]):
        raise ValueError("FP16 inference requires CUDA selfplay devices")
    opening, policy_init = c["opening"], c["policy_init"]
    if any(opening[k] > 1 for k in ("probability", "rejection_probability", "rejection_probability_fallback")):
        raise ValueError("Opening probabilities must be in [0,1]")
    if (opening["max_tries"] > 1000 or any(opening[k] > 100 for k in
            ("avg_dist_factor", "balance_exponent")) or policy_init["policy_init_mean"] > 100 or
            not 0.1 <= policy_init["policy_temperature"] <= 5):
        raise ValueError("Invalid opening configuration")
    r, s = c["replay"], c["shuffle"]
    if not 0 < r["taper_exponent"] <= 1:
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
