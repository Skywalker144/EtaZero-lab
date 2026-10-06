"""Fixed-size learner updates, immutable checkpoints and exact data/RNG restoration."""
import contextlib
import gc
import math
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from .config import fingerprint
from .network import make_network, TrainingForward
from .optimization import optimization_for, optimizer_for
from .reader import BatchReader, BatchPrefetcher, CudaBatchPrefetcher
from .schema import CONTRACT_ID
from .storage import atomic_write, load_json, save_json, sha256, sync_directory
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
        return forward(*(tensors[key] for key in keys))
    return forward(*(tensors[key] for key in keys)), None


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
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    value = torch.load(path, map_location=device, weights_only=False)
    if (value["contract"] != CONTRACT_ID or value["config_id"] != fingerprint(config) or
            value["id"] != reference["id"] or value["network_config"] != config["network"]):
        raise ValueError("Checkpoint identity/configuration mismatch")
    check_model_identity(value, config)
    return value


def commit_checkpoint(run_dir, config, model, optimizer, scaler, iteration, step, total_steps,
                      reader_state, parent, updates, optimization):
    identity = f"iteration_{iteration:06d}_step_{step:08d}_{uuid.uuid4().hex}"
    path = Path("checkpoints")/(identity+".pt")
    value = {"id": identity, "contract": CONTRACT_ID, "config_id": fingerprint(config),
             **model_identity(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "optimization": optimization.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "reader": reader_state,
             "iteration": iteration, "step": step, "total_steps": total_steps,
             "total_samples": optimization.consumed_samples, "optimizer_steps": optimization.optimizer_steps, "parent": parent,
             "committed_updates": updates,
             "source_id":load_json(Path(run_dir)/".internal/session.json")["source_id"] if (Path(run_dir)/".internal/session.json").exists() else None}
    atomic_write(Path(run_dir)/path, lambda p: torch.save(value,p), immutable=True)
    reference = {"id": identity, "path": str(path), "sha256": sha256(Path(run_dir)/path),
                 "iteration": iteration, "step": step, "total_steps": total_steps, "total_samples": value["total_samples"], "optimizer_steps": value["optimizer_steps"]}
    save_json((Path(run_dir)/path).with_suffix(".json"), {**reference,"parent":parent,"committed_updates":updates}, immutable=True)
    return reference


def prune_checkpoints(run_dir, keep):
    """Prune committed payloads only; JSON lineage and inference models stay intact."""
    if keep < 1:
        raise ValueError("checkpoint_keep must be positive")
    root = Path(run_dir)
    # Read the durable authority rather than accepting a candidate round state.
    state = load_json(root/".internal/state.json")
    if state["iteration"] == 0 or state["checkpoint"] is None:
        return []
    reference = state["checkpoint"]
    seen, rounds, retained, payloads = set(), set(), set(), []
    previous_iteration = state["iteration"]
    while reference:
        relative = Path(reference["path"])
        iteration = reference["iteration"]
        if relative.parent != Path("checkpoints") or relative.suffix != ".pt":
            raise ValueError(f"Invalid checkpoint path: {relative}")
        if reference["id"] in seen:
            raise ValueError("Checkpoint lineage contains a cycle")
        if not 0 <= iteration <= previous_iteration or iteration >= state["iteration"]:
            raise ValueError("Checkpoint lineage contains an uncommitted or unordered round")
        seen.add(reference["id"])
        previous_iteration = iteration
        path = root/relative
        sidecar = load_json(path.with_suffix(".json"))
        if any(sidecar.get(key) != value for key, value in reference.items()):
            raise ValueError(f"Checkpoint metadata differs from its reference: {relative}")
        # The first checkpoint encountered for each round is its final one.
        if iteration not in rounds:
            rounds.add(iteration)
            if len(rounds) <= keep:
                if not path.is_file():
                    raise FileNotFoundError(f"Retained checkpoint is missing: {path}")
                retained.add(path)
        payloads.append(path)
        reference = sidecar["parent"]
    # Validate the whole chain before deleting; interrupted deletion is retryable.
    removed = []
    for path in payloads:
        if path not in retained and path.exists():
            path.unlink()
            removed.append(str(path.relative_to(root)))
    if removed:
        sync_directory(root/"checkpoints")
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
    reference = commit_checkpoint(run_dir,config,model,optimizer,scaler,0,0,0,None,None,[],optimization)
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
                tensors={k:torch.from_numpy(a).float().to(device) for k,a in batch.items()}
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
    """Even, nonempty integer segments within the fixed consumed-batch budget."""
    if not 1<=segments<=steps:raise ValueError('Subepochs require at least one consumed batch each')
    return [(i+1)*steps//segments for i in range(segments)]


def train_iteration(run_dir, config, plan, base, log, stopping=lambda:False):
    root, iteration = Path(run_dir), plan["iteration"]
    progress_path = root/".internal/iterations"/f"{iteration:06d}"/"learner.json"
    progress = load_json(progress_path) if progress_path.exists() else None
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
                         config["run"]["seed"]+iteration, checkpoint["reader"] if progress else None,no_repeat_files=config["training"]["no_repeat_files"])
    step, total = (checkpoint["step"] if progress else 0), checkpoint["total_steps"]
    ends=subepoch_ends(plan['train_steps'],config['training']['sub_epochs'])
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
    if config['training']['no_repeat_files']:
        remaining=sum(reader.files[i]['rows']//plan['batch_size'] for i in reader.order[reader.index:])-reader.offset//plan['batch_size']
        if remaining<plan['train_steps']-step:
            reader.close()
            raise ValueError('No-repeat snapshot has insufficient complete file batches for the fixed round budget')
    updates = []
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
                    tensor = torch.from_numpy(array).float()
                    if device.type == "cuda":
                        tensor = tensor.pin_memory()
                    tensors[key] = tensor.to(device, non_blocking=device.type == "cuda")
            symmetry = int(torch.randint(8, ()).item()) if config['training']['d4_augmentation'] else 0
            if config['training']['d4_augmentation']:
                tensors = augment_training_batch(tensors, symmetry)
            optimization.before_step()
            learning_rates = {g["group_name"]: g["lr"] for g in optimizer.param_groups}
            weight_decays = {g["group_name"]: g["weight_decay"] for g in optimizer.param_groups}
            context = torch.autocast("cuda",dtype=torch.float16 if amp=="float16" else torch.bfloat16) if amp!="off" else contextlib.nullcontext()
            with context:
                components,step_losses=forward_batch(forward,tensors)
            loss,pl,opl,spl,sopl,vl=components[:6]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at iteration {iteration}, step {step+1}")
            optimizer.zero_grad(set_to_none=True)
            scale_before = scaler.get_scale()
            if scale_before <= 0 or not math.isfinite(scale_before):
                raise FloatingPointError("Invalid AMP loss scale")
            scaler.scale(loss * optimization.backward_scale).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None])
            module_norms={}
            for name,params in module_parameters.items():
                gradients=[p.grad for p in params if p.grad is not None]
                module_norms[name]=(torch.nn.utils.get_total_norm(gradients) if gradients else grad_norm.new_zeros(())) / optimization.backward_scale
            finite_norm = bool(torch.isfinite(grad_norm))
            if finite_norm:
                torch.nn.utils.clip_grads_with_norm_(parameters, optimization.gradient_cap(config['training']['gradient_clip']), grad_norm)
            elif not scaler.is_enabled():
                raise FloatingPointError(f"Nonfinite gradients at iteration {iteration}, step {step+1}")
            elif all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None):
                raise FloatingPointError("Nonfinite gradient norm despite finite unscaled gradients")
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
            if not finite_norm and not skipped:
                raise FloatingPointError("Nonfinite gradients without an AMP skipped update")
            if skipped:
                log("amp_overflow", iteration=iteration, step=step+1,
                    scale_before=scale_before, scale_after=scaler.get_scale(), skipped=True)
            optimization.after_step(successful=not skipped)
            step += 1; total += 1
            if step == plan['train_steps']:
                optimization.finish_round()
            update = uuid.uuid4().hex; updates.append(update)
            values=torch.stack([x.detach() for x in components]+[grad_norm / optimization.backward_scale]+list(module_norms.values())).tolist()
            diagnostics={}
            if step_losses is not None:
                diagnostics={'step_losses':step_losses.tolist(),
                             'grad_norms':dict(zip(module_norms,values[len(components)+1:]))}
            loss_names=LOSS_NAMES
            if model.predict_q_values:loss_names+=('q_winloss_loss',)
            log('update',iteration=iteration,step=step,total_steps=total,update_id=update,
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
                reference = commit_checkpoint(root,config,model,optimizer,scaler,iteration,step,total,consumed_state,reference,updates,optimization)
                log("checkpoint",iteration=iteration,checkpoint=reference,committed_updates=updates)
                if hasattr(log,'flush'):
                    log.flush()
                save_json(progress_path,{"checkpoint":reference,"snapshot_id":plan["snapshot_id"]})
                updates = []
        # Commit consumed batches, including scaler skips, at a normal stop.
        if updates:
            reference = commit_checkpoint(root,config,model,optimizer,scaler,iteration,step,total,consumed_state,reference,updates,optimization)
            log("checkpoint",iteration=iteration,checkpoint=reference,committed_updates=updates)
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
