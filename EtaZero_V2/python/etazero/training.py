"""Fixed-size learner updates, immutable checkpoints and exact data/RNG restoration."""
import contextlib
import gc
import math
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from .config import resume_config
from .network import make_network, TrainingForward
from .optimization import optimization_for, optimizer_for
from .reader import BatchReader, BatchPrefetcher, CudaBatchPrefetcher
from .schema import CONTRACT_ID
from .storage import atomic_write, load_json, save_json, sync_directory
from .symmetry import augment_batch


def model_identity(config):
    fields = {'network_config': config['network']}
    if config['agent']['algorithm'] == 'muzero':
        fields.update(algorithm='muzero', muzero_config=config['muzero'])
    return fields


def check_model_identity(saved, config):
    expected = model_identity(config)
    if (saved.get('algorithm', 'alphazero') != config['agent']['algorithm'] or
            any(saved.get(key) != value for key, value in expected.items())):
        raise ValueError('Model algorithm/network configuration mismatch')


def training_forward(model, config):
    if config['agent']['algorithm'] == 'muzero':
        from .muzero.training import TrainingForward as MuZeroForward
        return MuZeroForward(model, config)
    return TrainingForward(model, config['training']['soft_policy_weight_scale'], config['training']['disable_optimistic_policy'])


def forward_batch(forward, tensors):
    keys = ('obs', 'globals', 'policy', 'opponent_policy', 'opponent_policy_weight',
            'value', 'td_value', 'full_game_weight', 'q_values', 'q_visits')
    if 'actions' in tensors:
        keys += ('actions', 'step_weights', 'sequence_mask')
        return forward(*(tensors.get(key) for key in keys))
    return forward(*(tensors.get(key) for key in keys)), None


def augment_training_batch(tensors, symmetry):
    if 'actions' in tensors:
        from .muzero.training import augment_batch as muzero_augment
        return muzero_augment(tensors, symmetry)
    return augment_batch(tensors, symmetry)


def device_check(device):
    d = torch.device(device)
    if d.type == "cuda":
        if not torch.cuda.is_available() or d.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
        torch.empty(1, device=d)
    return d


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device topology differs from the checkpoint")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def load_checkpoint(run_dir, reference, config, device="cpu"):
    path = Path(run_dir)/reference["path"]
    value = torch.load(path, map_location=device, weights_only=False)
    if (value["contract"] != CONTRACT_ID or value["resume_config"] != resume_config(config) or
            value["id"] != reference["id"] or value["network_config"] != config["network"]):
        raise ValueError("Checkpoint identity/configuration mismatch")
    check_model_identity(value, config)
    return value


def commit_checkpoint(run_dir, config, model, optimizer, scaler, iteration, step, total_steps,
                      reader_state, optimization):
    identity = f"iteration_{iteration:06d}_step_{step:08d}_{uuid.uuid4().hex}"
    path = Path("checkpoints")/(identity+".pt")
    value = {"id": identity, "contract": CONTRACT_ID, "resume_config": resume_config(config),
             **model_identity(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "optimization": optimization.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "reader": reader_state,
             "iteration": iteration, "step": step, "total_steps": total_steps,
             "total_samples": optimization.consumed_samples, "optimizer_steps": optimization.optimizer_steps,
             "source_id":load_json(Path(run_dir)/".internal/session.json")["source_id"] if (Path(run_dir)/".internal/session.json").exists() else None}
    atomic_write(Path(run_dir)/path, lambda p: torch.save(value,p), immutable=True)
    reference = {"id": identity, "path": str(path), "iteration": iteration, "step": step, "total_steps": total_steps, "total_samples": value["total_samples"], "optimizer_steps": value["optimizer_steps"]}
    index_path = Path(run_dir)/'.internal/checkpoints.json'
    index = load_json(index_path) if index_path.exists() else []
    save_json(index_path, index + [reference])
    return reference


def prune_checkpoints(run_dir, keep):
    """Prune registered completed-round payloads using the bounded live index."""
    if keep < 1:
        raise ValueError("checkpoint_keep must be positive")
    root = Path(run_dir)
    state = load_json(root/'.internal/state.json')
    if state['checkpoint'] is None:
        return []
    index_path = root/'.internal/checkpoints.json'
    index = load_json(index_path)
    if state['checkpoint'] not in index:
        raise ValueError('Committed checkpoint is absent from the live index')
    latest = {}
    for reference in index:
        relative = Path(reference['path'])
        if (relative.parent != Path('checkpoints') or relative.suffix != '.pt'
                or relative.stem != reference['id']):
            raise ValueError(f'Invalid checkpoint path: {relative}')
        if reference['iteration'] < state['iteration']:
            latest[reference['iteration']] = reference
    # The authority is the final checkpoint selected by state, even if a
    # previous process saved another payload before dying at a pointer write.
    latest[state['checkpoint']['iteration']] = state['checkpoint']
    retained = {r['id'] for _,r in sorted(latest.items(),reverse=True)[:keep]}
    retained.update(r['id'] for r in index if r['iteration'] >= state['iteration'])
    for reference in index:
        if reference['id'] in retained and not (root/reference['path']).is_file():
            raise FileNotFoundError(f"Retained checkpoint is missing: {reference['path']}")
    removed = []
    for reference in index:
        if reference['id'] not in retained:
            path = root/reference['path']
            if path.exists():
                path.unlink(); removed.append(reference['path'])
    if removed:
        sync_directory(root/'checkpoints')
    save_json(index_path, [r for r in index if r['id'] in retained])
    return removed


def initialize(run_dir, config, weights=None):
    seed = config["run"]["seed"]
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    device = device_check(config["devices"]["train"])
    model = make_network(config).to(device)
    if weights:
        source = torch.load(weights, map_location=device, weights_only=False)
        if source["contract"] != CONTRACT_ID or source["network_config"] != config["network"]:
            raise ValueError("Imported weights do not match the configured model contract")
        check_model_identity(source, config)
        model.load_state_dict(source["model"], strict=True)
    optimizer = optimizer_for(model,config)
    optimization = optimization_for(model, config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] == "float16")
    reference = commit_checkpoint(run_dir,config,model,optimizer,scaler,0,0,0,None,optimization)
    del optimization,model,optimizer,scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reference


LOSS_NAMES=('loss','policy_loss','opponent_policy_loss','soft_policy_loss',
            'soft_opponent_policy_loss','value_loss','td_value_long_loss','td_value_mid_loss',
            'td_value_short_loss','long_optimistic_policy_loss','short_optimistic_policy_loss','shortterm_value_error_loss')


def validate_epoch(snapshot,model,forward,config,iteration,device,log):
    """Raw model, per-file full batches, source random D4 and post-batch cap."""
    if config['training']['skip_validation']:return
    manifest=load_json(Path(snapshot)/'manifest.json')
    batch_size=config['training']['batch_size']
    usable=sum(f['rows']//batch_size*batch_size for f in manifest['validation_files'])
    if not usable:
        log('validation',iteration=iteration,samples=0,batches=0,reason='no_complete_validation_batch')
        return
    reader=BatchReader(snapshot,batch_size,config['training']['prefetch_depth'],
                       config['run']['seed']+iteration,no_repeat_files=True,split='validation',
                       shuffle_files=config['training']['randomize_validation_files'])
    # Independent stream keeps validation from changing the learner's next RNG
    # state. The source promises uniform D4, not a shared seeded draw sequence.
    symmetries=random.Random((config['run']['seed']+iteration)^0x56414c)
    amp=config['training']['amp'];maximum=config['training']['max_validation_samples']
    names=LOSS_NAMES+(('q_winloss_loss',) if model.predict_q_values else ())
    sums=[0.0]*len(names);samples=batches=0;draws=[0]*8;step_sums=None
    was_training=model.training
    try:
        model.eval()
        with torch.no_grad():
            while True:
                try:batch=reader.next()
                except StopIteration:break
                tensors={k:torch.from_numpy(a).to(device=device,dtype=torch.int64 if k=='actions' else torch.float32)
                         for k,a in batch.items()}
                symmetry=symmetries.randrange(8);draws[symmetry]+=1
                tensors=augment_training_batch(tensors,symmetry) # Always on for validation.
                context=torch.autocast('cuda',dtype=torch.float16 if amp=='float16' else torch.bfloat16) if amp!='off' else contextlib.nullcontext()
                with context:
                    components,step_losses=forward_batch(forward,tensors)
                values=torch.stack(components).tolist()
                if not all(math.isfinite(v) for v in values):raise FloatingPointError('Nonfinite validation loss')
                sums=[a+v*batch_size for a,v in zip(sums,values)]
                if step_losses is not None:
                    measured=step_losses.tolist()
                    if not all(math.isfinite(v) for v in measured):raise FloatingPointError('Nonfinite validation step loss')
                    if step_sums is None:step_sums=[0.0]*len(measured)
                    step_sums=[a+v*batch_size for a,v in zip(step_sums,measured)]
                samples+=batch_size;batches+=1
                if maximum and samples>maximum:break
        log('validation',iteration=iteration,samples=samples,batches=batches,model='raw',
            symmetry_counts=draws,**dict(zip(names,(v/samples for v in sums))),
            **({'step_losses':[v/samples for v in step_sums]} if step_sums is not None else {}))
    finally:
        model.train(was_training);reader.close()


def subepoch_ends(steps,segments):
    """Even, nonempty integer segments within the actual consumed-batch budget."""
    if not 1<=segments<=steps:raise ValueError('Subepochs require at least one consumed batch each')
    return [(i+1)*steps//segments for i in range(segments)]


def train_iteration(run_dir, config, plan, base, log, stopping=lambda:False):
    root, iteration = Path(run_dir), plan["iteration"]
    progress_path = root/".internal/iterations"/f"{iteration:06d}"/"learner.json"
    progress = load_json(progress_path) if progress_path.exists() else None
    if progress and progress['snapshot_id'] != plan['snapshot_id']:
        raise ValueError('Learner checkpoint snapshot differs from the iteration plan')
    reference = progress["checkpoint"] if progress else base
    device = device_check(config["devices"]["train"])
    checkpoint = load_checkpoint(root,reference,config,device)
    if progress and (checkpoint["iteration"] != iteration or checkpoint["step"] > plan["train_steps"]):
        raise ValueError("Learner checkpoint belongs to a different iteration or exceeds its budget")
    model = make_network(config).to(device); model.load_state_dict(checkpoint["model"],strict=True); model.train()
    forward=training_forward(model,config)
    if config['training']['compile']:
        from torch._inductor import config as compiler_config
        compiler_config.compile_threads=config['run']['cpu_threads']
        torch._dynamo.config.fail_on_recompile_limit_hit=True
        # Learner batches have a fixed canvas and batch size. Keep separate
        # graphs for different network/precision runs instead of auto-generalizing.
        # The Transformer uses the source's NCHW/SDPA path. In torch 2.12,
        # automatic channels-last rewriting corrupts its BF16 attention
        # backward (finite gradients around 1e36, versus ~2.6 in eager).
        # Preserve that storage layout while retaining full-graph compilation.
        options={'layout_optimization':False} if model.architecture=='transformer' else None
        muzero = config['agent']['algorithm'] == 'muzero'
        forward=torch.compile(forward,fullgraph=not muzero,dynamic=muzero,options=options)
    optimizer = optimizer_for(model,config); optimizer.load_state_dict(checkpoint["optimizer"])
    optimization = optimization_for(model, config, optimizer, checkpoint['optimization'])
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] == "float16")
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng(checkpoint["rng"])
    reader = BatchReader(root/"snapshots"/plan["snapshot_id"],plan["batch_size"],config["training"]["prefetch_depth"],
                         config["run"]["seed"]+iteration, checkpoint["reader"] if progress else None,no_repeat_files=True)
    step, total = (checkpoint["step"] if progress else 0), checkpoint["total_steps"]
    ends=subepoch_ends(plan['train_steps'],min(config['training']['sub_epochs'],plan['train_steps']))
    if not progress:
        optimization.begin_round()
    if optimization.consumed_samples != total * plan["batch_size"] or optimization.round_batches != step:
        raise ValueError("Learner consumption counters differ from optimization state")
    expected_segment=sum(step>end for end in ends[:-1])
    segment_start=0 if expected_segment==0 else ends[expected_segment-1]
    if optimization.subepoch!=expected_segment or optimization.subepoch_batches!=step-segment_start:
        raise ValueError('Learner segment counters differ from the consumed batch cursor')
    amp = config["training"]["amp"]
    if device.type != "cuda" and amp != "off":
        raise ValueError("AMP training requires a CUDA device")
    remaining=sum(reader.files[i]['rows']//plan['batch_size'] for i in reader.order[reader.index:])-reader.offset//plan['batch_size']
    if remaining<plan['train_steps']-step:
        reader.close()
        raise ValueError('Single-pass snapshot has insufficient complete file batches for the actual round budget')
    unsaved = False
    consumed_state = reader.state()
    prepared=BatchPrefetcher(reader,config['training']['prefetch_depth'])
    prefetch = None
    parameters=list(model.parameters())
    module_parameters={name:list(getattr(model,name).parameters())
                       for name in ('representation','dynamics','prediction')} if config['agent']['algorithm']=='muzero' else {}
    try:
        if device.type=='cuda' and config['training']['cuda_prefetch']:
            prefetch=CudaBatchPrefetcher(prepared,device)
        while step < plan["train_steps"] and not stopping():
            if step==ends[optimization.subepoch]:
                optimization.begin_subepoch()
            if optimization.subepoch_batches==0:
                log('subepoch_start',iteration=iteration,subepoch=optimization.subepoch,
                    consumed_step=step,end_step=ends[optimization.subepoch],lookahead_counter=optimization.counter)
            if prefetch:
                tensors,consumed_state = prefetch.next(prefetch_next=step+1<plan["train_steps"])
            else:
                batch,consumed_state = prepared.next();tensors = {}
                for key,array in batch.items():
                    tensor = torch.from_numpy(array).to(dtype=torch.int64 if key=='actions' else torch.float32)
                    if device.type == "cuda":
                        tensor = tensor.pin_memory()
                    tensors[key] = tensor.to(device, non_blocking=device.type == "cuda")
            symmetry = int(torch.randint(8, ()).item()) if config['training']['d4_augmentation'] else 0
            # White Hex canonicalization also applies when augmentation is off.
            tensors = augment_training_batch(tensors, symmetry)
            optimization.before_step()
            learning_rates = {g["group_name"]: g["lr"] for g in optimizer.param_groups}
            weight_decays = {g["group_name"]: g["weight_decay"] for g in optimizer.param_groups}
            context = torch.autocast("cuda",dtype=torch.float16 if amp=="float16" else torch.bfloat16) if amp!="off" else contextlib.nullcontext()
            with context:
                components,step_losses=forward_batch(forward,tensors)
            loss,pl,opl,spl,sopl,vl=components[:6]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss * optimization.backward_scale).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None])
            module_norms={}
            detailed = step == 0 or (step+1) % config['optimizer']['norm_interval'] == 0 or step+1 == plan['train_steps']
            for name,params in module_parameters.items() if detailed else ():
                gradients=[p.grad for p in params if p.grad is not None]
                module_norms[name]=(torch.nn.utils.get_total_norm(gradients) if gradients else grad_norm.new_zeros(())) / optimization.backward_scale
            # Losses, norms and unroll diagnostics share one device-to-host
            # transfer. Reuse these values for both checks and journal metrics.
            diagnostics_tensors = [] if step_losses is None else list(step_losses.unbind())
            # Reuse GradScaler's unscale check rather than synchronizing twice
            # through get_scale(). These accessors are verified by CUDA tests.
            amp_status = ([scaler._get_scale_async(),
                           sum(scaler._found_inf_per_device(optimizer).values())]
                          if scaler.is_enabled() else [])
            values = torch.stack([x.detach() for x in components] +
                                 [grad_norm / optimization.backward_scale] +
                                 list(module_norms.values()) + diagnostics_tensors + amp_status).tolist()
            if not all(math.isfinite(v) for v in values[:len(components)]):
                raise FloatingPointError(f"Nonfinite loss at iteration {iteration}, step {step+1}")
            finite_norm = math.isfinite(values[len(components)])
            scale_before, found_inf = values[-2:] if amp_status else (1., 0.)
            if scale_before <= 0 or not math.isfinite(scale_before):
                raise FloatingPointError("Invalid AMP loss scale")
            skipped = found_inf != 0
            if finite_norm:
                torch.nn.utils.clip_grads_with_norm_(parameters, optimization.gradient_cap(config['training']['gradient_clip']), grad_norm)
            elif not scaler.is_enabled():
                raise FloatingPointError(f"Nonfinite gradients at iteration {iteration}, step {step+1}")
            elif not skipped:
                raise FloatingPointError("Nonfinite gradient norm despite finite unscaled gradients")
            scaler.step(optimizer)
            scaler.update()
            if skipped:
                log("amp_overflow", iteration=iteration, step=step+1,
                    scale_before=scale_before, scale_after=scaler.get_scale(), skipped=True)
            optimization.after_step(successful=not skipped)
            step += 1; total += 1
            if step == plan['train_steps']:
                optimization.finish_round()
            unsaved = True
            diagnostics={}
            if step_losses is not None:
                start=len(components)+1
                diagnostics={'step_losses':values[start+len(module_norms):start+len(module_norms)+len(diagnostics_tensors)]}
                if module_norms:
                    diagnostics['grad_norms']=dict(zip(module_norms,values[start:start+len(module_norms)]))
            loss_names=LOSS_NAMES
            if model.predict_q_values:loss_names+=('q_winloss_loss',)
            log('update',iteration=iteration,step=step,total_steps=total,
                optimizer_steps=optimization.optimizer_steps,total_samples=optimization.consumed_samples,
                subepoch=optimization.subepoch,subepoch_batches=optimization.subepoch_batches,
                **dict(zip((*loss_names,'grad_norm'),values)),amp_skipped=skipped,
                **diagnostics,
                symmetry=symmetry, learning_rates=learning_rates,weight_decays=weight_decays,
                lookahead_counter=optimization.counter,swa_samples=optimization.swa_count)
            del loss,pl,opl,spl,sopl,vl,components,tensors,step_losses
            if step==plan['train_steps']:
                validate_epoch(reader.snapshot,model,forward,config,iteration,device,log)
            if step % config["training"]["checkpoint_every"] == 0 or step == plan["train_steps"] or stopping():
                reference = commit_checkpoint(root,config,model,optimizer,scaler,iteration,step,total,consumed_state,optimization)
                log("checkpoint",iteration=iteration,checkpoint=reference)
                if hasattr(log,'flush'):
                    log.flush()
                save_json(progress_path,{"checkpoint":reference,"snapshot_id":plan["snapshot_id"]})
                unsaved = False
        # Commit consumed batches, including scaler skips, at a normal stop.
        if unsaved:
            reference = commit_checkpoint(root,config,model,optimizer,scaler,iteration,step,total,consumed_state,optimization)
            log("checkpoint",iteration=iteration,checkpoint=reference)
            if hasattr(log,'flush'):
                log.flush()
            save_json(progress_path,{"checkpoint":reference,"snapshot_id":plan["snapshot_id"]})
        return reference, step == plan["train_steps"]
    finally:
        try:
            if prefetch:
                prefetch.close()
        finally:
            prepared.close();reader.close()
        del optimization,forward,model,optimizer,scaler,checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
