"""Compact, bounded KataGo-style shardify/merge and per-row random waves.

The window formula and wave organization are adapted from KataGo (MIT).
Raw complete games remain authoritative; temporaries can always be rebuilt.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import math
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import uuid
import fcntl
import numpy as np
from .data import read_raw, training_view
from .schema import CONTRACT_ID
from .storage import atomic_write, load_json, save_json, save_npz, sha256, sync_directory, write_npz


def desired_window(rows, replay):
    minimum, exponent = replay['min_rows'], replay['taper_exponent']
    scale=replay['taper_scale'] or minimum
    x=rows-minimum+scale+replay['add_to_data_rows']
    if x<0:raise ValueError('Replay power-law base is negative')
    scaled=(x**exponent-scale**exponent)/(exponent*scale**(exponent-1))
    desired=max(int(scaled*replay['expand_per_row']+minimum),minimum)
    return desired if replay['max_rows']=='all' else min(desired,replay['max_rows'])


def replay_counts(entries, minimum):
    random_rows=postrandom=0
    for entry in entries:
        rows=entry['metadata']['rows']
        if entry['metadata']['model_id'].startswith('random:'):random_rows+=rows
        else:postrandom+=rows
    return {'raw_rows':random_rows+postrandom,'random_rows':random_rows,
            'postrandom_rows':postrandom,'usable_rows':min(random_rows,minimum)+postrandom}


def window_sources(entries, replay):
    counts=replay_counts(entries,replay['min_rows'])
    if counts['raw_rows']<replay['min_rows']:raise ValueError('Cannot shuffle before replay.min_rows is reached')
    desired=desired_window(counts['usable_rows'],replay)
    # Catalogue mtimes come from stat, never iteration or writer wall-clock metadata.
    ordered=sorted(entries,key=lambda e:e['mtime_ns'])
    chosen=[];rows=0
    for entry in reversed(ordered):
        if entry['metadata']['rows']<=0:continue
        chosen.append(entry);rows+=entry['metadata']['rows']
        if rows>=desired:break
    if not rows:raise ValueError('Cannot shuffle an empty window')
    return chosen,dict(counts,desired_rows=desired,window_rows=rows,
                       range=[counts['raw_rows']+int(replay['add_to_data_rows'])-rows,
                              counts['raw_rows']+int(replay['add_to_data_rows'])])


def validation_file(path):
    return int(hashlib.md5(Path(path).name.encode()).hexdigest()[:13],16)/2**52>=.99


def sampling_probability(rows, replay):
    target = replay['keep_target_rows']
    return 1.0 if target == 'all' else min(1.0, target/rows)


def _concat(views):
    return {key: np.concatenate([a[key] for a in views]) for key in views[0]}


def _load_view(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def _temporary(path, arrays, compressed=True):
    # Private scratch uses fast compression and skips hashing/fsync.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_npz(path, arrays, compressed=compressed)


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
    index, sources, raw, root, buckets, seed, cache, keep_prob, compressed = task
    rng = np.random.default_rng(np.random.SeedSequence([*seed,index]))
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
        path = Path(root) / f"bucket_{bucket}" / f"group_{index}.npz"
        _temporary(path,{k:v[select] for k,v in arrays.items()},compressed)
        counts.append((bucket,str(path),len(select)))
    return counts


def _merge_bucket(task):
    paths, total, scratch, output, limit, shard_rows, seed, label, compressed = task
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    if total > limit:
        raise ValueError('Actual random bucket exceeds shuffle array memory budget; increase memory_mb or reduce bucket_rows')
    if not paths:return []
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
    # Source plans a fixed files-per-bucket count before random assignment.
    # Equal slices may be larger/smaller than the nominal file target.
    file_count = shard_rows
    for index in range(file_count):
        start, stop = index*total//file_count, (index+1)*total//file_count
        selection = permutation[start:stop]
        path = Path(output) / f"train_{label}_{index}.npz"
        save_npz(path, {k: a[selection] for k, a in arrays.items()})
        result.append({"path": path.name, "rows": len(selection), "sha256": sha256(path)})
    return result


def resource_plan(rows, max_raw_rows, groups, config, keep_prob=None):
    """Conservative array estimates, excluding interpreter/allocator and OS overhead."""
    s, canvas = config["shuffle"], config["network"]["canvas"]
    # Six globals, four spatial float targets (two policies + W-L Q/visits),
    # opponent/full-game gates, three TD WDLs and main WDL.
    train_bytes = 6*4 + 5*((canvas*canvas+7)//8) + 16*canvas*canvas + 2*4 + 9*4 + 12
    # Raw includes int64 visits and at most twice as many observations as moves.
    raw_bytes = 20*canvas*canvas + 10*((canvas*canvas+7)//8) + 164
    if config.get('agent', {}).get('algorithm') == 'muzero':
        steps = config['unroll']['steps']
        train_bytes += steps * (16*canvas*canvas + 56) + 8*steps + 9*(steps+1)
        raw_bytes += 6*canvas*canvas
    budget = s["memory_mb"]*1024**2
    per_worker = budget//s["workers"]
    group_rows=min(s['group_rows'],max(1,per_worker//(3*train_bytes)-max_raw_rows))
    nominal=s['training_shard_rows']
    bucket_rows=min(s['bucket_rows'],(per_worker//(4*train_bytes+16)//nominal)*nominal)
    merge_limit=per_worker//(4*train_bytes+16)
    if 3*max_raw_rows*raw_bytes>per_worker or bucket_rows<nominal:
        raise ValueError('shuffle.memory_mb is too small for one raw shard or training shard per worker')
    if keep_prob is None:keep_prob = sampling_probability(rows, config['replay'])
    # Each source group independently rounds its requested sample count.
    output_rows = min(rows, math.ceil(rows*keep_prob+groups/2))
    waves = s["waves"]
    wave_rows = math.ceil(output_rows/waves)
    buckets = max(1, int(round(rows*keep_prob/waves/bucket_rows)))
    # Random bucket/wave draws make these planning estimates, not hard
    # file/disk limits. The disk estimate allows all rows to land in one wave.
    temp_bytes = output_rows*train_bytes*(3 if waves>1 else 2)
    peak_files = groups*waves + math.ceil(wave_rows/group_rows)*buckets if waves>1 else groups*buckets
    return {"training_bytes_per_row": train_bytes, "raw_bytes_per_row_estimate": raw_bytes,
            "output_rows_estimate":output_rows,"keep_prob":keep_prob,
            "training_view_cache_bytes_estimate":rows*train_bytes*2,
            "group_rows": group_rows, "bucket_rows": bucket_rows, "waves": waves,
            "compress_temp": s['compress_temp'],
            "buckets_per_wave": buckets,"files_per_bucket":bucket_rows//nominal,"merge_limit_rows":merge_limit, "array_memory_budget_bytes": budget,
            "scatter_array_bytes_estimate": group_rows*3*raw_bytes*s["workers"],
            "merge_array_bytes_estimate": bucket_rows*(4*train_bytes+16)*s["workers"],
            "temporary_bytes_estimate": temp_bytes, "peak_temporary_files_estimate": peak_files}


def _groups(items, limit):
    groups, group, count = [], [], 0
    for path, checksum, n in items:
        if n<=0:continue
        group.append((path,checksum));count+=n
        if count>=limit:
            groups.append(group);group=[];count=0
    if group:groups.append(group)
    return groups


def _source_groups(root, chosen, limit, seed, partition=0):
    items = [(str(root/e['path']),e['sha256'],e['metadata']['rows']) for e in chosen]
    rng = np.random.default_rng(np.random.SeedSequence([seed,2**32-1,partition]))
    return _groups([items[i] for i in rng.permutation(len(items))],limit)


def _two_phase(pool, groups, raw, rows, scratch, output, plan, shard_rows, seed, label, cache, keep_prob=1.0, partition=0):
    if not groups:return []
    buckets = plan['buckets_per_wave']
    namespace=(seed,1,partition) if raw else (seed,2,partition,int(label))
    assignments = pool.map(_scatter, [(i,g,raw,str(scratch),buckets,namespace,str(cache),keep_prob,plan['compress_temp']) for i,g in enumerate(groups)])
    files, counts = [[] for _ in range(buckets)], [0]*buckets
    for group_output in assignments:
        for bucket, path, n in group_output:
            files[bucket].append(path); counts[bucket] += n
    tasks = [(files[b],counts[b],str(scratch),str(output),plan['merge_limit_rows'],plan['files_per_bucket'],
              (seed,3,partition,int(label),b),f"{label}_{b}",plan['compress_temp']) for b in range(buckets)]
    return [e for group in pool.map(_merge_bucket,tasks) for e in group]


def _write_data(root, stage, scratch, chosen, config, plan, seed, pool=None, output_name="data", partition=0):
    if not chosen:return []
    s = config['shuffle']
    rows = sum(e['metadata']['rows'] for e in chosen)
    items = [(str(root/e['path']),e['sha256'],e['metadata']['rows']) for e in chosen]
    groups = _source_groups(root,chosen,plan['group_rows'],seed,partition)
    sizes = {path:n for path,_,n in items}
    keep_prob = plan['keep_prob']
    kept_rows = sum(int(round(sum(sizes[path] for path,_ in group)*keep_prob)) for group in groups)
    if not kept_rows:return []
    output = stage/output_name;output.mkdir(parents=True,exist_ok=True)
    cache = root/'.internal/training_views'
    owner = nullcontext(pool) if pool is not None else ProcessPoolExecutor(
        max_workers=s['workers'],mp_context=multiprocessing.get_context('spawn'))
    with owner as workers:
        if plan['waves'] == 1:
            outputs = _two_phase(workers,groups,True,kept_rows,scratch,output,plan,s['training_shard_rows'],
                                 seed,'0',cache,keep_prob,partition)
        else:
            waves = [[] for _ in range(plan['waves'])]
            scatter = workers.map(_scatter,[(i,g,True,str(scratch/'waves'),plan['waves'],(seed,1,partition),str(cache),keep_prob,plan['compress_temp'])
                                             for i,g in enumerate(groups)])
            for assignments in scatter:
                for wave,path,n in assignments:
                    waves[wave].append((path,None,n))
            outputs = []
            for wave,parts in enumerate(waves):
                if not sum(p[2] for p in parts):
                    shutil.rmtree(scratch/'waves'/f'bucket_{wave}')
                    continue
                wave_scratch=scratch/f'merge_{wave}';wave_scratch.mkdir()
                wave_groups=_groups(parts,plan['group_rows'])
                outputs.extend(_two_phase(workers,wave_groups,False,sum(p[2] for p in parts),wave_scratch,
                                          output,plan,s['training_shard_rows'],seed,str(wave),cache,partition=partition))
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


def build_snapshot(run_dir, iteration, entries, config, pool=None):
    root, r, s = Path(run_dir), config["replay"], config['shuffle']
    # All entries are required to compute capped random/usable totals. Quota
    # totals are separate from the usable replay rows.
    entries=[{**e,'mtime_ns':(root/e['path']).stat().st_mtime_ns} for e in entries]
    chosen,selection=window_sources(entries,r)
    rows=selection['window_rows'];desired=selection['desired_rows']
    seed = config["run"]["seed"]+iteration
    keep_prob=sampling_probability(rows,r) # Before MD5 filtering, like the source.
    skip=config['training']['skip_validation']
    train_sources=[e for e in chosen if skip or not validation_file(e['path'])]
    val_sources=[] if skip else [e for e in chosen if validation_file(e['path'])]
    def partition_plan(sources,partition):
        subset_rows=sum(e['metadata']['rows'] for e in sources)
        max_raw=max((max(e['metadata']['plies'],e['metadata']['rows']) for e in sources),default=0)
        rough=resource_plan(subset_rows,max_raw,math.ceil(subset_rows/s['group_rows']),config,keep_prob)
        groups=_source_groups(root,sources,rough['group_rows'],seed,partition)
        return resource_plan(subset_rows,max_raw,len(groups),config,keep_prob)
    plan=partition_plan(train_sources,0);val_plan=partition_plan(val_sources,1)
    identity = f"iteration_{iteration:06d}_{uuid.uuid4().hex}"
    snapshots = root/"snapshots"; snapshots.mkdir(exist_ok=True)
    stage = snapshots/(".tmp_"+identity)
    temp_parent = Path(s["temp_dir"]).expanduser().resolve() if s["temp_dir"] else snapshots
    temp_parent.mkdir(parents=True,exist_ok=True)
    scratch = temp_parent/(".shuffle_"+identity)
    space_plan=dict(plan,temporary_bytes_estimate=plan['temporary_bytes_estimate']+val_plan['temporary_bytes_estimate'],
                    output_rows_estimate=plan['output_rows_estimate']+val_plan['output_rows_estimate'])
    _check_space(root,temp_parent,snapshots,chosen,space_plan)
    try:
        stage.mkdir();scratch.mkdir()
        outputs = _write_data(root,stage,scratch/'train',train_sources,config,plan,seed,pool)
        if not outputs or not sum(e['rows'] for e in outputs):
            raise ValueError('keep_target_rows produced an empty sample after per-group rounding or MD5 holdout')
        validation_outputs=_write_data(root,stage,scratch/'val',val_sources,config,val_plan,seed,pool,'data/validation',partition=1)
        for files,partition in ((outputs,plan),(validation_outputs,val_plan)):
            usable=sum(e['rows']//config['training']['batch_size']*config['training']['batch_size'] for e in files)
            partition.update(usable_rows_per_pass=usable,complete_batches_per_pass=usable//config['training']['batch_size'],
                             discarded_tail_rows_per_pass=sum(e['rows'] for e in files)-usable)
        manifest = {"id":identity,"contract":CONTRACT_ID,"canvas":config["network"]["canvas"],
                    "iteration":iteration,"seed":seed,"rows":sum(e['rows'] for e in outputs),
                    "window_rows":rows,"desired_rows":desired,"keep_prob":plan['keep_prob'],"resource_plan":plan,
                    'sources':train_sources,'files':outputs,'selection':selection,
                    'validation_sources':val_sources,'validation_files':validation_outputs,
                    'validation_rows':sum(e['rows'] for e in validation_outputs),'validation_resource_plan':val_plan,
                    'skip_validation':skip,
                    'recipe':{'replay':r,'shuffle':s,'network':{'canvas':config['network']['canvas']}}}
        if config['agent']['algorithm'] == 'muzero':
            from .muzero.data import replay_weight_mean
            if any(e['metadata'].get('unroll_steps') != config['unroll']['steps'] or
                   e['metadata'].get('algorithm') != 'muzero' for e in chosen):
                raise ValueError('MuZero replay algorithm/unroll configuration mismatch')
            manifest.update(algorithm='muzero', unroll_steps=config['unroll']['steps'],
                            unroll_weight_mean=replay_weight_mean(root, train_sources))
            manifest['recipe'].update(agent=config['agent'], unroll=config['unroll'])
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
        space_plan=dict(manifest['resource_plan'])
        for key in ('temporary_bytes_estimate','output_rows_estimate'):
            space_plan[key]+=manifest['validation_resource_plan'][key]
        _check_space(root,temp_parent,snapshot,manifest['sources']+manifest['validation_sources'],space_plan)
        # The lock proves no cooperating restorer/evictor owns this scratch.
        for garbage in list(snapshot.glob('.evicted_*'))+list(snapshot.glob('.restore_*')):
            if garbage.is_dir():
                shutil.rmtree(garbage)
        try:
            stage.mkdir();scratch.mkdir()
            outputs = _write_data(root,stage,scratch/'train',manifest['sources'],manifest['recipe'],
                                  manifest['resource_plan'],manifest['seed'])
            validation_outputs=_write_data(root,stage,scratch/'val',manifest['validation_sources'],manifest['recipe'],
                                           manifest['validation_resource_plan'],manifest['seed'],output_name='data/validation',partition=1)
            if outputs != manifest['files'] or validation_outputs!=manifest['validation_files']:
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
