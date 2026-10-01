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


# Each section belongs to exactly one file; baseline supplies every required key.
FIELDS = {
    "run": ("run", {"run_dir": str, "seed": int, "max_cycles": int,
                    "max_seconds": float, "cpu_threads": int}),
    "agent": ("run", {"algorithm": str, "root_search_algo": str, "nonroot_search_algo": str}),
    "devices": ("run", {"train": str, "selfplay": str}),
    "environment": ("env", {"sizes": str, "size_weights": str, "rules": str, "rule_weights": str}),
    "network": ("net", {"canvas": int, "channels": int, "blocks": int, "value_hidden": int}),
    "selfplay": ("selfplay", {"game_threads": int, "search_threads": int, "max_batch": int,
                              "batch_wait_us": int, "queue_capacity": int, "writer_queue": int,
                              "shard_rows": int, "flush_seconds": float, "probe_games": int,
                              "recent_games": int}),
    "search": ("selfplay", {"simulations": int, "c_puct": float, "virtual_loss": float,
                            "reuse_tree": boolean}),
    "exploration": ("selfplay", {"dirichlet_alpha": float, "noise_fraction": float,
                                 "temperature": float, "temperature_moves": int,
                                 "final_temperature": float}),
    "training": ("train", {"train_steps": int, "batch_size": int, "prefetch_depth": int,
                           "checkpoint_every": int, "amp": str, "gradient_clip": float}),
    "optimizer": ("train", {"type": str, "learning_rate": float, "momentum": float}),
    "loss": ("train", {"l2": float}),
    "replay": ("train", {"replay_ratio": float, "min_rows": int, "window_mode": str,
                         "window_rows": int, "window_min_rows": int, "window_max_rows": int,
                         "taper_exponent": float, "expand_per_row": float, "taper_scale": float,
                         "shuffle_workers": int, "shuffle_group_rows": int,
                         "shuffle_bucket_rows": int, "training_shard_rows": int}),
    "evaluation": ("eval", {"simulations": int, "search_threads": int, "max_batch": int}),
    "match": ("match", {"games": int, "game_threads": int, "simulations": int,
                        "search_threads": int, "max_batch": int, "temperature": float,
                        "temperature_moves": int}),
}
FILES = tuple(dict.fromkeys(v[0] for v in FIELDS.values()))


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
            result.update(inherit(configs_root / parent, ancestors + [current]))
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
                zero_allowed = {"seed", "max_cycles", "max_seconds", "blocks", "batch_wait_us",
                                "virtual_loss", "noise_fraction", "temperature", "temperature_moves",
                                "final_temperature", "gradient_clip", "momentum", "l2", "expand_per_row"}
                if key not in zero_allowed and value == 0:
                    raise ValueError(f"{section}.{key} must be positive")
    env, net = c["environment"], c["network"]
    sizes = csv(env["sizes"], int)
    rules = csv(env["rules"])
    if not sizes or len(set(sizes)) != len(sizes) or any(x < 5 or x > net["canvas"] for x in sizes) or not 5 <= net["canvas"] <= 25:
        raise ValueError("Sizes must be unique, in [5, canvas], with canvas <= 25")
    if not rules or len(set(rules)) != len(rules) or any(x not in RULES for x in rules):
        raise ValueError("Invalid rule distribution")
    for label, entries in (("size", sizes), ("rule", rules)):
        weights = csv(env[label + "_weights"], float)
        if len(weights) != len(entries) or any(not math.isfinite(x) or x <= 0 for x in weights):
            raise ValueError(f"Invalid {label} weights")
    if c["exploration"]["noise_fraction"] > 1 or c["optimizer"]["momentum"] >= 1:
        raise ValueError("Noise fraction must be <= 1 and momentum < 1")
    if c["optimizer"]["type"] not in ("sgd", "adam") or c["training"]["amp"] not in ("off", "float16", "bfloat16"):
        raise ValueError("Invalid optimizer or AMP mode")
    r = c["replay"]
    if r["window_mode"] not in ("fixed", "katago") or not 0 < r["taper_exponent"] <= 1:
        raise ValueError("Invalid replay window mode/exponent")
    if r["window_max_rows"] < r["window_min_rows"] or r["taper_scale"] < r["window_min_rows"]:
        raise ValueError("Window maximum/scale must be >= window minimum")
    if c["selfplay"]["shard_rows"] + net["canvas"] ** 2 > r["shuffle_group_rows"]:
        raise ValueError("shuffle_group_rows must hold one raw shard including its final whole game")
    if r["training_shard_rows"] > r["shuffle_bucket_rows"]:
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
