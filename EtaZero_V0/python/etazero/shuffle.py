"""KataGo-style power-law windows and two-phase, disk-backed uniform shuffling.

The window function is adapted from KataGo shuffle.py (MIT; see THIRD_PARTY.md).
Random bucket assignment, then permutation within each bucket, follows its shardify/merge design.
"""
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import uuid
import numpy as np
from .data import read_raw, training_view
from .storage import save_json, save_npz, sha256, sync_directory


def desired_window(rows, replay):
    if replay["window_mode"] == "fixed":
        return replay["window_rows"]
    minimum, scale, exponent = replay["window_min_rows"], replay["taper_scale"], replay["taper_exponent"]
    x = rows - minimum + scale
    unscaled = x**exponent - scale**exponent
    scaled = unscaled / (exponent * scale**(exponent-1))
    desired = int(scaled * replay["expand_per_row"] + minimum)
    return min(max(desired, minimum), replay["window_max_rows"])


def _concat(views):
    return {key: np.concatenate([a[key] for a in views]) for key in views[0]}


def _load_view(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def _scatter(task):
    index, sources, root, buckets, seed = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    views = []
    for source in sources:
        path, expected_hash = source
        if sha256(path) != expected_hash:
            raise ValueError(f"Raw source changed after snapshot selection: {path}")
        views.append(training_view(read_raw(path)))
    arrays = _concat(views)
    assignments = rng.integers(buckets, size=len(arrays["value"]))
    counts = []
    for bucket in range(buckets):
        select = assignments == bucket
        n = int(select.sum())
        if n:
            path = Path(root) / f"bucket_{bucket}" / f"group_{index}.npz"
            save_npz(path, {k: v[select] for k, v in arrays.items()})
            counts.append((bucket, str(path), n))
    return counts


def _merge_bucket(task):
    paths, total, scratch, output, limit, shard_rows, seed, label = task
    rng = np.random.default_rng(seed)
    if total > limit:
        # Random buckets can exceed their expected size. Re-scatter large buckets
        # instead of loading them all or silently exceeding the configured bound.
        subcount = max(2, math.ceil(total/limit)*2)
        children = [[] for _ in range(subcount)]
        counts = [0]*subcount
        for i, path in enumerate(paths):
            arrays = _load_view(path)
            for start in range(0, len(arrays["value"]), limit):
                part = {k: a[start:start+limit] for k, a in arrays.items()}
                assigned = rng.integers(subcount, size=len(part["value"]))
                for b in range(subcount):
                    take = assigned == b
                    n = int(take.sum())
                    if n:
                        dest = Path(scratch) / f"{label}_{i}_{start}_{b}.npz"
                        save_npz(dest, {k: a[take] for k, a in part.items()})
                        children[b].append(str(dest)); counts[b] += n
        result = []
        for b, child in enumerate(children):
            if child:
                result.extend(_merge_bucket((child, counts[b], scratch, output, limit, shard_rows,
                                              int(rng.integers(2**63)), f"{label}_{b}")))
        for child in children:
            for path in child:
                Path(path).unlink()
        return result
    if not total:
        return []
    arrays = _concat([_load_view(p) for p in paths])
    if len(arrays["value"]) != total:
        raise ValueError("Intermediate shuffle count mismatch")
    permutation = rng.permutation(total)
    result = []
    for start in range(0, total, shard_rows):
        selection = permutation[start:start+shard_rows]
        path = Path(output) / f"train_{label}_{start}.npz"
        save_npz(path, {k: a[selection] for k, a in arrays.items()})
        result.append({"path": path.name, "rows": len(selection), "sha256": sha256(path)})
    return result


def build_snapshot(run_dir, cycle, entries, config):
    root = Path(run_dir)
    replay = config["replay"]
    desired = desired_window(sum(e["metadata"]["rows"] for e in entries), replay)
    chosen, rows = [], 0
    # As in KataGo, retain complete recent shards until the requested window is covered.
    for entry in reversed(entries):
        chosen.append(entry); rows += entry["metadata"]["rows"]
        if rows >= desired:
            break
    if not rows:
        raise ValueError("Cannot shuffle an empty window")
    identity = f"cycle_{cycle:06d}_{uuid.uuid4().hex}"
    snapshots = root / "snapshots"; snapshots.mkdir(exist_ok=True)
    stage = snapshots / (".tmp_" + identity)
    stage.mkdir(); scratch = stage / "scratch"; scratch.mkdir()
    output = stage / "data"; output.mkdir()
    groups, group, count = [], [], 0
    for entry in chosen:
        n = entry["metadata"]["rows"]
        if n > replay["shuffle_group_rows"]:
            raise ValueError("Raw shard exceeds shuffle_group_rows")
        if group and count+n > replay["shuffle_group_rows"]:
            groups.append(group); group, count = [], 0
        group.append((str(root/entry["path"]), entry["sha256"])); count += n
    if group:
        groups.append(group)
    buckets = max(1, math.ceil(rows/replay["shuffle_bucket_rows"]))
    seed = config["run"]["seed"] + cycle
    try:
        with ProcessPoolExecutor(max_workers=replay["shuffle_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            scatter_tasks = [(i,g,str(scratch),buckets,seed) for i,g in enumerate(groups)]
            assignments = list(pool.map(_scatter, scatter_tasks))
            files, counts = [[] for _ in range(buckets)], [0]*buckets
            for group_output in assignments:
                for bucket, path, n in group_output:
                    files[bucket].append(path); counts[bucket] += n
            merge_tasks = [(files[b], counts[b], str(scratch), str(output), replay["shuffle_bucket_rows"],
                            replay["training_shard_rows"], seed+b+1, str(b)) for b in range(buckets)]
            outputs = [e for group_output in pool.map(_merge_bucket, merge_tasks) for e in group_output]
        outputs.sort(key=lambda x: x["path"])
        if sum(e["rows"] for e in outputs) != rows:
            raise ValueError("Shuffle did not conserve rows")
        shutil.rmtree(scratch)
        manifest = {"id": identity, "cycle": cycle, "seed": seed, "rows": rows, "desired_rows": desired,
                    "sources": chosen, "files": outputs}
        save_json(stage/"manifest.json", manifest, immutable=True)
        sync_directory(stage); os.rename(stage, snapshots/identity); sync_directory(snapshots)
        return identity
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
