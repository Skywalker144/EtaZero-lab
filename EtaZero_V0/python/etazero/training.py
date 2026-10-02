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
from .optimization import Optimization, optimizer_for
from .reader import BatchReader, BatchPrefetcher, CudaBatchPrefetcher
from .schema import CONTRACT_ID
from .storage import atomic_write, load_json, save_json, sha256, sync_directory
from .symmetry import augment_batch

MAX_AMP_RETRIES = 32


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
    return value


def commit_checkpoint(run_dir, config, model, optimizer, scaler, iteration, step, total_steps,
                      reader_state, parent, updates, optimization):
    identity = f"iteration_{iteration:06d}_step_{step:08d}_{uuid.uuid4().hex}"
    path = Path("checkpoints")/(identity+".pt")
    value = {"id": identity, "contract": CONTRACT_ID, "config_id": fingerprint(config),
             "network_config": config["network"], "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "optimization": optimization.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "reader": reader_state,
             "iteration": iteration, "step": step, "total_steps": total_steps,
             "total_samples": total_steps*config["training"]["batch_size"], "parent": parent,
             "committed_updates": updates,
             "source_id":load_json(Path(run_dir)/".internal/session.json")["source_id"] if (Path(run_dir)/".internal/session.json").exists() else None}
    atomic_write(Path(run_dir)/path, lambda p: torch.save(value,p), immutable=True)
    reference = {"id": identity, "path": str(path), "sha256": sha256(Path(run_dir)/path),
                 "iteration": iteration, "step": step, "total_steps": total_steps, "total_samples": value["total_samples"]}
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
        model.load_state_dict(source["model"], strict=True)
    optimizer = optimizer_for(model,config)
    optimization = Optimization(model, config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] == "float16")
    reference = commit_checkpoint(run_dir,config,model,optimizer,scaler,0,0,0,None,None,[],optimization)
    del optimization,model,optimizer,scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reference


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
    forward=TrainingForward(model,config['training']['soft_policy_weight_scale'])
    if config['training']['compile']:
        from torch._inductor import config as compiler_config
        compiler_config.compile_threads=config['run']['cpu_threads']
        torch._dynamo.config.fail_on_recompile_limit_hit=True
        # Learner batches have a fixed canvas and batch size. Keep separate
        # graphs for different network/precision runs instead of auto-generalizing.
        forward=torch.compile(forward,fullgraph=True,dynamic=False)
    optimizer = optimizer_for(model,config); optimizer.load_state_dict(checkpoint["optimizer"])
    optimization = Optimization(model, config, optimizer, checkpoint['optimization'])
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] == "float16")
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng(checkpoint["rng"])
    reader = BatchReader(root/"snapshots"/plan["snapshot_id"],plan["batch_size"],config["training"]["prefetch_depth"],
                         config["run"]["seed"]+iteration, checkpoint["reader"] if progress else None)
    step, total = (checkpoint["step"] if progress else 0), checkpoint["total_steps"]
    amp = config["training"]["amp"]
    if device.type != "cuda" and amp != "off":
        raise ValueError("AMP training requires a CUDA device")
    updates = []
    consumed_state = reader.state()
    prepared=BatchPrefetcher(reader,config['training']['prefetch_depth'])
    prefetch = None
    parameters=list(model.parameters())
    compiler_settings=contextlib.ExitStack()
    try:
        if config['training']['compile'] and scaler.is_enabled():
            from torch._functorch import config as autograd_config
            # FP16 retries reuse one graph. AOT buffer donation can consume its
            # saved intermediates, so disable it for this learner invocation.
            compiler_settings.enter_context(autograd_config.patch(donated_buffer=False))
        if device.type=='cuda' and config['training']['cuda_prefetch']:
            prefetch=CudaBatchPrefetcher(prepared,device)
        while step < plan["train_steps"] and not stopping():
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
                tensors = augment_batch(tensors, symmetry)
            optimization.configure(total)
            context = torch.autocast("cuda",dtype=torch.float16 if amp=="float16" else torch.bfloat16) if amp!="off" else contextlib.nullcontext()
            with context:
                loss,pl,opl,spl,sopl,vl=forward(tensors['obs'],tensors['globals'],tensors['policy'],
                                             tensors['opponent_policy'],tensors['opponent_policy_weight'],tensors['value'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at iteration {iteration}, step {step+1}")
            for retry in range(MAX_AMP_RETRIES+1):
                optimizer.zero_grad(set_to_none=True)
                scale_before=scaler.get_scale()
                if scale_before<=0 or not math.isfinite(scale_before):
                    raise FloatingPointError("Invalid AMP loss scale")
                # Retain the same forward graph for FP16 overflow retries. BatchNorm
                # statistics, random draws and the data cursor advance only once.
                scaler.scale(loss * plan['batch_size']).backward(retain_graph=scaler.is_enabled())
                scaler.unscale_(optimizer)
                grad_norm=torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None])
                if torch.isfinite(grad_norm):
                    cap = optimization.gradient_cap(config['training']['gradient_clip'])
                    torch.nn.utils.clip_grads_with_norm_(parameters,cap,grad_norm)
                    scaler.step(optimizer);scaler.update()
                    optimization.after_step()
                    break
                if not scaler.is_enabled():
                    raise FloatingPointError(f"Nonfinite gradients at iteration {iteration}, step {step+1}")
                # GradScaler skips this optimizer step and backs off its scale.
                # The discarded attempt is explicitly logged and is not an update.
                scaler.step(optimizer);scaler.update()
                scale_after=scaler.get_scale()
                if scale_after>=scale_before:
                    raise FloatingPointError("Nonfinite gradient norm without a recoverable AMP overflow")
                log("amp_overflow",iteration=iteration,step=step+1,retry=retry+1,
                    scale_before=scale_before,scale_after=scale_after)
                if retry==MAX_AMP_RETRIES:
                    raise FloatingPointError(f"AMP overflow persisted after {MAX_AMP_RETRIES} retries")
            step += 1; total += 1
            if step == plan['train_steps']:
                optimization.finish_round()
            update = uuid.uuid4().hex; updates.append(update)
            values=torch.stack([loss.detach(),pl.detach(),opl.detach(),spl.detach(),sopl.detach(),
                                vl.detach(),grad_norm / plan['batch_size']]).tolist()
            log('update',iteration=iteration,step=step,total_steps=total,update_id=update,
                **dict(zip(('loss','policy_loss','opponent_policy_loss','soft_policy_loss',
                            'soft_opponent_policy_loss','value_loss','grad_norm'),values)),amp_retries=retry,
                symmetry=symmetry, learning_rates={g['group_name']:g['lr'] for g in optimizer.param_groups},
                weight_decays={g['group_name']:g['weight_decay'] for g in optimizer.param_groups},
                lookahead_counter=optimization.counter,swa_samples=int(optimization.swa.n_averaged.item()))
            del loss,pl,opl,spl,sopl,vl,tensors
            if step % config["training"]["checkpoint_every"] == 0 or step == plan["train_steps"] or stopping():
                reference = commit_checkpoint(root,config,model,optimizer,scaler,iteration,step,total,consumed_state,reference,updates,optimization)
                log("checkpoint",iteration=iteration,checkpoint=reference,committed_updates=updates)
                if hasattr(log,'flush'):
                    log.flush()
                save_json(progress_path,{"checkpoint":reference,"snapshot_id":plan["snapshot_id"]})
                updates = []
        # A stop arriving between updates still commits all successfully completed updates.
        if updates:
            reference = commit_checkpoint(root,config,model,optimizer,scaler,iteration,step,total,consumed_state,reference,updates,optimization)
            log("checkpoint",iteration=iteration,checkpoint=reference,committed_updates=updates)
            if hasattr(log,'flush'):
                log.flush()
            save_json(progress_path,{"checkpoint":reference,"snapshot_id":plan["snapshot_id"]})
        return reference, step == plan["train_steps"]
    finally:
        compiler_settings.close()
        try:
            if prefetch:
                prefetch.close()
        finally:
            prepared.close();reader.close()
        del optimization,forward,model,optimizer,scaler,checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
