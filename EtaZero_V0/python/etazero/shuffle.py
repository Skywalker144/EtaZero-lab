"""Compact, bounded KataGo-style shardify/merge and per-row random waves.

The window formula and wave organization are adapted from KataGo (MIT).
Raw complete games remain authoritative; temporaries can always be rebuilt.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import uuid
import fcntl
import numpy as np
from .data import read_raw, training_view
from .schema import CONTRACT_ID
from .storage import atomic_write, load_json, save_json, save_npz, sha256, sync_directory


def desired_window(rows, replay):
    minimum, exponent = replay["min_rows"], replay["taper_exponent"]
    scaled = (rows**exponent - minimum**exponent) / (exponent * minimum**(exponent-1))
    return max(int(scaled * replay["expand_per_row"] + minimum), minimum)


def sampling_probability(rows, replay):
    target = replay['keep_target_rows']
    return 1.0 if target == 'all' else min(1.0, target/rows)


def _concat(views):
    return {key: np.concatenate([a[key] for a in views]) for key in views[0]}


def _load_view(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def _temporary(path, arrays):
    # Scratch is private to an unpublished attempt: no compression, hashing or fsync.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(file, **arrays)


def partition_rows(rows, buckets, rng):
    """IID uniform labels represented by counts and a uniform row permutation.

    Conditional on multinomial counts, every partition is equally likely. This
    is KataGo's contiguous-slice organization, avoiding O(rows*buckets) scans.
    """
    counts = np.bincount(rng.integers(buckets, size=rows), minlength=buckets)
    return rng.permutation(rows), np.concatenate(([0], np.cumsum(counts)))


def _source_view(path, expected_hash, cache):
    path, cache = Path(path), Path(cache)
    view_path, certificate = cache/(expected_hash+'.npz'), cache/(expected_hash+'.json')
    stat = path.stat()
    stamp = [stat.st_size, stat.st_mtime_ns]
    if certificate.exists() and view_path.exists():
        info = load_json(certificate)
        if info['contract'] != CONTRACT_ID or info['source_sha256'] != expected_hash:
            raise ValueError(f'Invalid training-view certificate: {certificate}')
        if info['source_stat'] != stamp:
            if sha256(path) != expected_hash:
                raise ValueError(f'Raw source changed after snapshot selection: {path}')
            info['source_stat'] = stamp
            save_json(certificate, info)
        if sha256(view_path) != info['sha256']:
            raise ValueError(f'Training-view checksum mismatch: {view_path}')
        return _load_view(view_path)
    if sha256(path) != expected_hash:
        raise ValueError(f'Raw source changed after snapshot selection: {path}')
    arrays = training_view(read_raw(path))
    # A derived, authenticated cache. Full trajectories remain the source of truth.
    atomic_write(view_path, lambda p: _temporary(p, arrays))
    save_json(certificate, {'contract':CONTRACT_ID, 'source_sha256':expected_hash,
                           'source_stat':stamp, 'sha256':sha256(view_path)})
    return arrays


def _scatter(task):
    index, sources, raw, root, buckets, seed, cache, keep_prob = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    views = []
    for path, expected_hash in sources:
        if raw:
            view = _source_view(path, expected_hash, cache)
        else:
            view = _load_view(path)
        views.append(view)
    arrays = _concat(views)
    del views, view
    if keep_prob < 1:
        count = int(round(len(arrays['value'])*keep_prob))
        selection = rng.permutation(len(arrays['value']))[:count]
        arrays = {k: v[selection] for k,v in arrays.items()}
    order, offsets = partition_rows(len(arrays['value']), buckets, rng)
    counts = []
    for bucket in range(buckets):
        select = order[offsets[bucket]:offsets[bucket+1]]
        if len(select):
            path = Path(root) / f"bucket_{bucket}" / f"group_{index}.npz"
            _temporary(path, {k: v[select] for k, v in arrays.items()})
            counts.append((bucket, str(path), len(select)))
    return counts


def _merge_bucket(task):
    paths, total, scratch, output, limit, shard_rows, seed, label = task
    rng = np.random.default_rng(seed)
    if total > limit:
        # Unusually large random buckets are re-scattered, with no row loss.
        subcount = max(2, math.ceil(total/limit)*2)
        children, counts = [[] for _ in range(subcount)], [0]*subcount
        for i, path in enumerate(paths):
            arrays = _load_view(path)
            for start in range(0, len(arrays["value"]), limit):
                part = {k: a[start:start+limit] for k, a in arrays.items()}
                order, offsets = partition_rows(len(part['value']), subcount, rng)
                for b in range(subcount):
                    take = order[offsets[b]:offsets[b+1]]
                    if len(take):
                        dest = Path(scratch) / f"{label}_{i}_{start}_{b}.npz"
                        _temporary(dest, {k: a[take] for k, a in part.items()})
                        children[b].append(str(dest)); counts[b] += len(take)
            del arrays, part
            Path(path).unlink()
        result = []
        for b, child in enumerate(children):
            if child:
                result.extend(_merge_bucket((child, counts[b], scratch, output, limit, shard_rows,
                                              int(rng.integers(2**63)), f"{label}_{b}")))
        return result
    if not total:
        return []
    # Copy each input directly into one fixed-size destination, keeping only one
    # decompressed input live instead of all fragments plus concatenated copies.
    arrays, offset = None, 0
    for path in paths:
        part = _load_view(path)
        if arrays is None:
            arrays = {k: np.empty((total, *v.shape[1:]), dtype=v.dtype) for k, v in part.items()}
        n = len(part["value"])
        if offset+n > total or set(part) != set(arrays):
            raise ValueError("Intermediate shuffle count/fields mismatch")
        for k, a in arrays.items():
            if part[k].dtype != a.dtype or part[k].shape[1:] != a.shape[1:]:
                raise ValueError("Intermediate shuffle layout mismatch")
            a[offset:offset+n] = part[k]
        offset += n
        del part
        Path(path).unlink()
    if offset != total:
        raise ValueError("Intermediate shuffle count mismatch")
    permutation = rng.permutation(total)
    result = []
    for start in range(0, total, shard_rows):
        selection = permutation[start:start+shard_rows]
        path = Path(output) / f"train_{label}_{start}.npz"
        save_npz(path, {k: a[selection] for k, a in arrays.items()})
        result.append({"path": path.name, "rows": len(selection), "sha256": sha256(path)})
    return result


def resource_plan(rows, max_raw_rows, groups, config):
    """Conservative array estimates, excluding interpreter/allocator and OS overhead."""
    s, canvas = config["shuffle"], config["network"]["canvas"]
    train_bytes = 4*4 + 5*((canvas*canvas+7)//8) + 4*canvas*canvas + 12
    # Raw includes int64 visits and at most twice as many observations as moves.
    raw_bytes = 12*canvas*canvas + 10*((canvas*canvas+7)//8) + 160
    budget = s["memory_mb"]*1024**2
    per_worker = budget//s["workers"]
    group_rows = min(s["group_rows"], per_worker//(3*raw_bytes))
    bucket_rows = min(s["bucket_rows"], per_worker//(4*train_bytes+16))
    if group_rows < max_raw_rows or bucket_rows < s["training_shard_rows"]:
        raise ValueError("shuffle.memory_mb is too small for one raw shard or training shard per worker")
    keep_prob = sampling_probability(rows, config['replay'])
    # Each source group independently rounds its requested sample count.
    output_rows = min(rows, math.ceil(rows*keep_prob+groups/2))
    waves = s["waves"]
    wave_rows = math.ceil(output_rows/waves)
    buckets = max(1, math.ceil(wave_rows/bucket_rows))
    # Wave draws and recursive overflow make these planning estimates, not hard
    # file/disk limits. The disk estimate allows all rows to land in one wave.
    temp_bytes = output_rows*train_bytes*(3 if waves>1 else 2)
    peak_files = groups*waves + math.ceil(wave_rows/group_rows)*buckets if waves>1 else groups*buckets
    return {"training_bytes_per_row": train_bytes, "raw_bytes_per_row_estimate": raw_bytes,
            "output_rows_estimate":output_rows,"keep_prob":keep_prob,
            "training_view_cache_bytes_estimate":rows*train_bytes*2,
            "group_rows": group_rows, "bucket_rows": bucket_rows, "waves": waves,
            "buckets_per_wave": buckets, "array_memory_budget_bytes": budget,
            "scatter_array_bytes_estimate": group_rows*3*raw_bytes*s["workers"],
            "merge_array_bytes_estimate": bucket_rows*(4*train_bytes+16)*s["workers"],
            "temporary_bytes_estimate": temp_bytes, "peak_temporary_files_estimate": peak_files}


def _groups(items, limit):
    groups, group, count = [], [], 0
    for path, checksum, n in items:
        if n > limit:
            raise ValueError("Raw/intermediate shard exceeds effective shuffle group limit")
        if group and count+n > limit:
            groups.append(group); group, count = [], 0
        group.append((path, checksum)); count += n
    if group:
        groups.append(group)
    return groups


def _source_groups(root, chosen, limit, seed):
    items = [(str(root/e['path']),e['sha256'],max(e['metadata']['plies'], e['metadata']['rows'])) for e in chosen]
    rng = np.random.default_rng(np.random.SeedSequence([seed,2**32-1]))
    return _groups([items[i] for i in rng.permutation(len(items))],limit)


def _two_phase(pool, groups, raw, rows, scratch, output, plan, shard_rows, seed, label, cache, keep_prob=1.0):
    buckets = max(1, math.ceil(rows/plan["bucket_rows"]))
    assignments = pool.map(_scatter, [(i,g,raw,str(scratch),buckets,seed,str(cache),keep_prob) for i,g in enumerate(groups)])
    files, counts = [[] for _ in range(buckets)], [0]*buckets
    for group_output in assignments:
        for bucket, path, n in group_output:
            files[bucket].append(path); counts[bucket] += n
    tasks = [(files[b],counts[b],str(scratch),str(output),plan["bucket_rows"],shard_rows,
              seed+b+1,f"{label}_{b}") for b in range(buckets)]
    return [e for group in pool.map(_merge_bucket,tasks) for e in group]


def _write_data(root, stage, scratch, chosen, config, plan, seed, pool=None):
    s = config['shuffle']
    rows = sum(e['metadata']['rows'] for e in chosen)
    items = [(str(root/e['path']),e['sha256'],e['metadata']['rows']) for e in chosen]
    groups = _source_groups(root,chosen,plan['group_rows'],seed)
    sizes = {path:n for path,_,n in items}
    keep_prob = sampling_probability(rows, config['replay'])
    kept_rows = sum(int(round(sum(sizes[path] for path,_ in group)*keep_prob)) for group in groups)
    if not kept_rows:
        raise ValueError('keep_target_rows produced an empty sample after per-group rounding')
    output = stage/'data';output.mkdir()
    cache = root/'.internal/training_views'
    owner = nullcontext(pool) if pool is not None else ProcessPoolExecutor(
        max_workers=s['workers'],mp_context=multiprocessing.get_context('spawn'))
    with owner as workers:
        if plan['waves'] == 1:
            outputs = _two_phase(workers,groups,True,kept_rows,scratch,output,plan,s['training_shard_rows'],
                                 seed,'0',cache,keep_prob)
        else:
            waves = [[] for _ in range(plan['waves'])]
            scatter = workers.map(_scatter,[(i,g,True,str(scratch/'waves'),plan['waves'],seed,str(cache),keep_prob)
                                             for i,g in enumerate(groups)])
            for assignments in scatter:
                for wave,path,n in assignments:
                    waves[wave].append((path,None,n))
            outputs = []
            for wave,parts in enumerate(waves):
                if not parts:
                    continue
                wave_scratch=scratch/f'merge_{wave}';wave_scratch.mkdir()
                wave_groups=_groups(parts,plan['group_rows'])
                outputs.extend(_two_phase(workers,wave_groups,False,sum(p[2] for p in parts),wave_scratch,
                                          output,plan,s['training_shard_rows'],seed+wave+1,str(wave),cache))
                shutil.rmtree(wave_scratch)
                shutil.rmtree(scratch/'waves'/f'bucket_{wave}')
    outputs.sort(key=lambda x:x['path'])
    if sum(e['rows'] for e in outputs) != kept_rows:
        raise ValueError('Shuffle did not conserve sampled rows')
    return outputs


def _check_space(root, temp_parent, output_parent, chosen, plan):
    rows=sum(e['metadata']['rows'] for e in chosen)
    cache=root/'.internal/training_views'
    cold_rows=sum(e['metadata']['rows'] for e in chosen
                  if not (cache/(e['sha256']+'.npz')).exists() or not (cache/(e['sha256']+'.json')).exists())
    requirements={}
    for path,size in ((temp_parent,plan['temporary_bytes_estimate']),
                      (output_parent,plan['output_rows_estimate']*plan['training_bytes_per_row']*2),
                      (root,cold_rows*plan['training_bytes_per_row']*2)):
        device=path.stat().st_dev
        if device not in requirements:
            requirements[device]=[path,0]
        requirements[device][1]+=size
    for path,required in requirements.values():
        if shutil.disk_usage(path).free<required:
            raise OSError(f'Insufficient shuffle disk space at {path}: estimated {required} bytes including cold view cache')


def build_snapshot(run_dir, iteration, entries, config, pool=None, total_rows=None):
    root, r, s = Path(run_dir), config["replay"], config['shuffle']
    total = sum(e['metadata']['rows'] for e in entries) if total_rows is None else total_rows
    if total < r['min_rows']:
        raise ValueError('Cannot shuffle before replay.min_rows is reached')
    desired = desired_window(total, r)
    chosen, rows = [], 0
    for entry in reversed(entries):
        chosen.append(entry); rows += entry["metadata"]["rows"]
        if rows >= desired:
            break
    if not rows:
        raise ValueError("Cannot shuffle an empty window")
    max_raw = max(max(e["metadata"]["plies"],e["metadata"]["rows"]) for e in chosen)
    seed = config["run"]["seed"]+iteration
    rough_groups = math.ceil(rows/s["group_rows"])
    plan = resource_plan(rows,max_raw,rough_groups,config)
    groups = _source_groups(root,chosen,plan["group_rows"],seed)
    plan = resource_plan(rows,max_raw,len(groups),config)
    identity = f"iteration_{iteration:06d}_{uuid.uuid4().hex}"
    snapshots = root/"snapshots"; snapshots.mkdir(exist_ok=True)
    stage = snapshots/(".tmp_"+identity)
    temp_parent = Path(s["temp_dir"]).expanduser().resolve() if s["temp_dir"] else snapshots
    temp_parent.mkdir(parents=True,exist_ok=True)
    scratch = temp_parent/(".shuffle_"+identity)
    _check_space(root,temp_parent,snapshots,chosen,plan)
    try:
        stage.mkdir();scratch.mkdir()
        outputs = _write_data(root,stage,scratch,chosen,config,plan,seed,pool)
        manifest = {"id":identity,"contract":CONTRACT_ID,"canvas":config["network"]["canvas"],
                    "iteration":iteration,"seed":seed,"rows":sum(e['rows'] for e in outputs),
                    "window_rows":rows,"desired_rows":desired,"keep_prob":plan['keep_prob'],"resource_plan":plan,
                    'sources':chosen,'files':outputs,
                    'recipe':{'replay':r,'shuffle':s,'network':{'canvas':config['network']['canvas']}}}
        save_json(stage/"manifest.json",manifest,immutable=True)
        sync_directory(stage);os.rename(stage,snapshots/identity);sync_directory(snapshots)
        return identity
    finally:
        # Only this uniquely named unpublished workspace is eligible for cleanup.
        if scratch.exists():
            shutil.rmtree(scratch)
        if stage.exists():
            shutil.rmtree(stage)


def restore_snapshot(snapshot):
    """Rebuild an evicted derived payload, requiring the original file hashes."""
    snapshot = Path(snapshot)
    with (snapshot/'.restore.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (snapshot/'data').is_dir():
            return
        manifest = load_json(snapshot/'manifest.json')
        if manifest['contract']!=CONTRACT_ID or manifest['id']!=snapshot.name:
            raise ValueError(f'Invalid snapshot identity: {snapshot}')
        root = snapshot.parent.parent
        stage = snapshot/('.restore_'+uuid.uuid4().hex)
        temp_dir=manifest['recipe']['shuffle']['temp_dir']
        temp_parent=Path(temp_dir).expanduser().resolve() if temp_dir else snapshot.parent
        temp_parent.mkdir(parents=True,exist_ok=True)
        scratch=temp_parent/('.shuffle_restore_'+uuid.uuid4().hex)
        _check_space(root,temp_parent,snapshot,manifest['sources'],manifest['resource_plan'])
        # The lock proves no cooperating restorer/evictor owns this scratch.
        for garbage in list(snapshot.glob('.evicted_*'))+list(snapshot.glob('.restore_*')):
            if garbage.is_dir():
                shutil.rmtree(garbage)
        try:
            stage.mkdir();scratch.mkdir()
            outputs = _write_data(root,stage,scratch,manifest['sources'],manifest['recipe'],
                                  manifest['resource_plan'],manifest['seed'])
            if outputs != manifest['files']:
                raise ValueError(f'Rebuilt snapshot differs from original manifest: {snapshot}')
            sync_directory(stage);os.rename(stage/'data',snapshot/'data');sync_directory(snapshot)
        finally:
            if scratch.exists():
                shutil.rmtree(scratch)
            if stage.exists():
                shutil.rmtree(stage)


def prune_derived(run_dir, snapshots, keep_views):
    """Evict reproducible arrays only; preserve manifests, raw games and checkpoints."""
    root = Path(run_dir)
    for identity in snapshots:
        path = root/'snapshots'/identity
        with (path/'.restore.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for garbage in path.glob('.evicted_*'):
                if garbage.is_dir():
                    shutil.rmtree(garbage)
            if (path/'data').is_dir():
                # Atomic removal from the readable namespace; a crash can only
                # leave disposable scratch, never a partially readable payload.
                garbage = path/('.evicted_'+uuid.uuid4().hex)
                os.rename(path/'data',garbage);sync_directory(path)
                shutil.rmtree(garbage)
    cache = root/'.internal/training_views'
    for path in cache.glob('*'):
        if path.suffix in ('.npz','.json') and path.stem not in keep_views:
            path.unlink()
