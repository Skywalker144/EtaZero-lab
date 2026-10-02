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
SEARCH_PARAMETERS = dict(value_weight_exponent=float, chosen_move_subtract=float, chosen_move_prune=float,
                         fpu_loss_prop=float, root_fpu_loss_prop=float, c_puct_log=float, c_puct_base=float,
                         c_puct_stdev_prior=float, c_puct_stdev_prior_weight=float, c_puct_stdev_scale=float,
                         root_num_symmetries_to_sample=int, nn_policy_temperature=float,
                         root_policy_temperature_early=float, root_policy_temperature=float,
                         temperature_halflife=float, temperature_only_below_prob=float)


def validate_search_parameters(c):
    if not 1 <= c['root_num_symmetries_to_sample'] <= 8:
        raise ValueError('Root symmetry count must be in [1,8]')
    if any(c[k] > 1 for k in ('fpu_loss_prop','root_fpu_loss_prop','c_puct_stdev_scale','temperature_only_below_prob')):
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
    "network": ("net", {"canvas": int, "channels": int, "blocks": int}),
    "training": ("train", {"train_steps": int, "batch_size": int, "prefetch_depth": int,
                           'cuda_prefetch':boolean,'compile':boolean,
                           "checkpoint_every": int, "checkpoint_keep": int, "amp": str, "gradient_clip": float,
                           "replay_ratio": float, "d4_augmentation": boolean, "soft_policy_weight_scale": float}),
    "optimizer": ("train", {"kind": str, "lr_scale": float, "lr_warmup": boolean,
                             "head_lr_factor": float, "noreg_lr_factor": float,
                             "input_wd_factor": float, "normal_wd_factor": float,
                             "norm_interval": int, "lookahead_k": int, "lookahead_alpha": float,
                             "swa_period_samples": int, "swa_scale": float}),
    "replay": ("train", {"min_rows": int, "taper_exponent": float,
                         "expand_per_row": float, "keep_target_rows": row_target}),
    "shuffle": ("train", {"workers": int, "group_rows": int, "bucket_rows": int,
                          "training_shard_rows": int, "waves": int, "memory_mb": int,
                          "temp_dir": str, "snapshot_keep": int, "compress_temp": boolean}),
    'search': ("selfplay", {'full_search_visits': int, 'cheap_search_visits': int,
                            'cheap_search_probs': float, 'cheap_search_target_weight': float,
                            'reuse_tree': boolean, 'clear_before_search': boolean}),
    'selfplay': ("selfplay", {'bootstrap_games': int, 'recent_games': int}),
    'reduce_visits': ("selfplay", {'reduce_visits': boolean, 'reduce_visits_threshold': float,
                                   'reduce_visits_threshold_lookback': int, 'reduced_visits_min': int,
                                   'reduced_visits_weight': float}),
    'puct': ("selfplay", {'c_puct': float, 'c_puct_log': float, 'c_puct_base': float,
                          'c_puct_stdev_prior': float, 'c_puct_stdev_prior_weight': float,
                          'c_puct_stdev_scale': float, 'virtual_loss': float}),
    'fpu': ("selfplay", {'use_fpu': boolean, 'fpu_reduction_max': float, 'root_fpu_reduction_max': float,
                         'fpu_parent_weight_by_visited_policy_pow': float, 'fpu_loss_prop': float,
                         'root_fpu_loss_prop': float}),
    'value_weighting': ("selfplay", {'value_weight_exponent': float}),
    'forced_playouts': ("selfplay", {'root_desired_per_child_visits_coeff': float}),
    'policy_target': ("selfplay", {'policy_target_pruning': boolean, 'chosen_move_subtract': float,
                                   'chosen_move_prune': float}),
    'lcb': ("selfplay", {'use_lcb': boolean, 'lcb_stdevs': float, 'min_visit_prop_for_lcb': float}),
    'symmetry': ("selfplay", {'root_num_symmetries_to_sample': int}),
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
                                        'value_surprise_data_weight': float}),
    'parallelism': ("selfplay", {'game_threads': int, 'search_threads': int}),
    'inference': ("selfplay", {'server_threads': int, 'max_batch': int, 'inference_precision': str,
                               'cache_entries': int, 'batch_wait_us': int, 'queue_capacity': int}),
    'writer': ("selfplay", {'writer_queue': int, 'shard_rows': int, 'flush_seconds': float}),
}
FILES = tuple(dict.fromkeys(v[0] for v in FIELDS.values()))


def network_widths(channels):
    """Scale b5c192nbt widths, rounding heads up to multiples of eight."""
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


def load_config(directory):
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
        result.update(values)
        # Machine overrides are deliberately applied only at the selected directory.
        return result

    values = inherit(directory, [])
    values.update(_read(directory, local=True))
    config = {}
    for section, (_, fields) in FIELDS.items():
        config[section] = {}
        for key, convert in fields.items():
            if (section, key) not in values:
                raise ValueError(f"Missing required key: {section}.{key}")
            try:
                config[section][key] = convert(values[(section, key)])
            except ValueError as error:
                raise ValueError(f"Invalid {section}.{key}: {error}") from error
    validate(config)
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
    if tuple(a[k] for k in ("algorithm", "root_search_algo", "nonroot_search_algo")) != ("alphazero", "puct", "puct"):
        raise ValueError("Selected algorithm/search combination is not implemented in V0")
    for section, fields in c.items():
        for key, value in fields.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{section}.{key} must be nonnegative and finite")
                zero_allowed = {"seed", "max_iteration", "max_seconds", "blocks", "batch_wait_us",
                                "virtual_loss", "noise_fraction", "temperature", "temperature_early",
                                "final_temperature", "gradient_clip", "soft_policy_weight_scale", "swa_period_samples", "input_wd_factor", "normal_wd_factor", "forbidden_feature_dropout_prob", "expand_per_row", "cache_entries",
                                "policy_surprise_data_weight", "value_surprise_data_weight", "cheap_search_probs",
                                "cheap_search_target_weight", "fpu_reduction_max", "root_fpu_reduction_max",
                                "reduce_visits_threshold", "reduced_visits_weight",
                                "min_visit_prop_for_lcb", "root_desired_per_child_visits_coeff", "value_weight_exponent",
                                "chosen_move_subtract", "chosen_move_prune", "fpu_loss_prop", "root_fpu_loss_prop", "c_puct_log",
                                "c_puct_stdev_prior_weight", "c_puct_stdev_scale", "temperature_only_below_prob",
                                "probability", "avg_dist_factor", "balance_exponent", "rejection_probability",
                                "rejection_probability_fallback", "policy_init_mean"}
                if key not in zero_allowed and value == 0:
                    raise ValueError(f"{section}.{key} must be positive")
    env, net = c["environment"], c["network"]
    network_widths(net["channels"])
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
    search = c['search']
    validate_search_parameters({key: value for section in ('fpu', 'puct', 'symmetry', 'temperature')
                                for key, value in c[section].items()})
    if (not 2 <= search['full_search_visits'] <= 2**31 - 1 or search['cheap_search_probs'] > 1
            or not 2 <= search['cheap_search_visits'] <= search['full_search_visits']
            or search['cheap_search_target_weight'] > 1 or c['lcb']['min_visit_prop_for_lcb'] > 1):
        raise ValueError('Invalid cheap search cap/probability/weight or LCB visit proportion')
    if search['cheap_search_probs'] == 1 and search['cheap_search_target_weight'] == 0:
        raise ValueError('All-cheap zero-weight searches produce no training rows')
    if (c['reduce_visits']['reduce_visits_threshold'] > 0.999999 or
            not 1 <= c['reduce_visits']['reduce_visits_threshold_lookback'] <= 1000 or
            not 2 <= c['reduce_visits']['reduced_visits_min'] <= search['full_search_visits'] or
            c['reduce_visits']['reduced_visits_weight'] > 1):
        raise ValueError('Invalid Reduce Visits threshold/lookback/minimum/weight')
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
    if c['writer']['shard_rows'] + net["canvas"] ** 2 > s["group_rows"]:
        raise ValueError("shuffle.group_rows must hold one raw shard including its final whole game")
    if s["training_shard_rows"] > s["bucket_rows"]:
        raise ValueError("Training shard rows must be <= shuffle bucket rows")
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
