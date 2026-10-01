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
from .network import make_network, losses
from .reader import BatchReader
from .schema import CONTRACT_ID
from .storage import atomic_write, load_json, save_json, sha256

MAX_AMP_RETRIES = 32


def device_check(device):
    d = torch.device(device)
    if d.type == "cuda":
        if not torch.cuda.is_available() or d.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
        torch.empty(1, device=d)
    return d


def optimizer_for(model, config):
    o = config["optimizer"]
    if o["type"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=o["learning_rate"], momentum=o["momentum"], weight_decay=0)
    return torch.optim.Adam(model.parameters(), lr=o["learning_rate"], weight_decay=0)


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


def commit_checkpoint(run_dir, config, model, optimizer, scaler, cycle, step, total_steps,
                      reader_state, parent, updates):
    identity = f"cycle_{cycle:06d}_step_{step:08d}_{uuid.uuid4().hex}"
    path = Path("checkpoints")/(identity+".pt")
    value = {"id": identity, "contract": CONTRACT_ID, "config_id": fingerprint(config),
             "network_config": config["network"], "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": None, "scaler": scaler.state_dict(), "rng": rng_state(), "reader": reader_state,
             "cycle": cycle, "step": step, "total_steps": total_steps,
             "total_samples": total_steps*config["training"]["batch_size"], "parent": parent,
             "committed_updates": updates,
             "source_id":load_json(Path(run_dir)/"session.json")["source_id"] if (Path(run_dir)/"session.json").exists() else None}
    atomic_write(Path(run_dir)/path, lambda p: torch.save(value,p), immutable=True)
    reference = {"id": identity, "path": str(path), "sha256": sha256(Path(run_dir)/path),
                 "cycle": cycle, "step": step, "total_steps": total_steps, "total_samples": value["total_samples"]}
    save_json((Path(run_dir)/path).with_suffix(".json"), {**reference,"parent":parent,"committed_updates":updates}, immutable=True)
    return reference


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
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] == "float16")
    reference = commit_checkpoint(run_dir,config,model,optimizer,scaler,0,0,0,None,None,[])
    del model,optimizer,scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reference


def train_cycle(run_dir, config, plan, base, log, stopping=lambda:False):
    root, cycle = Path(run_dir), plan["cycle"]
    progress_path = root/"cycles"/f"{cycle:06d}"/"learner.json"
    progress = load_json(progress_path) if progress_path.exists() else None
    reference = progress["checkpoint"] if progress else base
    device = device_check(config["devices"]["train"])
    checkpoint = load_checkpoint(root,reference,config,device)
    if progress and (checkpoint["cycle"] != cycle or checkpoint["step"] > plan["train_steps"]):
        raise ValueError("Learner checkpoint belongs to a different cycle or exceeds its budget")
    model = make_network(config).to(device); model.load_state_dict(checkpoint["model"],strict=True); model.train()
    optimizer = optimizer_for(model,config); optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] == "float16")
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng(checkpoint["rng"])
    reader = BatchReader(root/"snapshots"/plan["snapshot_id"],plan["batch_size"],config["training"]["prefetch_depth"],
                         config["run"]["seed"]+cycle, checkpoint["reader"] if progress else None)
    step, total = (checkpoint["step"] if progress else 0), checkpoint["total_steps"]
    amp = config["training"]["amp"]
    if device.type != "cuda" and amp != "off":
        raise ValueError("AMP training requires a CUDA device")
    updates = []
    try:
        while step < plan["train_steps"] and not stopping():
            batch = reader.next()
            tensors = {}
            for key,array in batch.items():
                tensor = torch.from_numpy(array).float()
                if device.type == "cuda":
                    tensor = tensor.pin_memory()
                tensors[key] = tensor.to(device, non_blocking=device.type == "cuda")
            context = torch.autocast("cuda",dtype=torch.float16 if amp=="float16" else torch.bfloat16) if amp!="off" else contextlib.nullcontext()
            with context:
                policy,value = model(tensors["obs"])
                loss,pl,vl,reg = losses(policy,value,tensors["obs"],tensors["policy"],tensors["value"],model,config["loss"]["l2"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at cycle {cycle}, step {step+1}")
            parameters=list(model.parameters())
            for retry in range(MAX_AMP_RETRIES+1):
                optimizer.zero_grad(set_to_none=True)
                scale_before=scaler.get_scale()
                if scale_before<=0 or not math.isfinite(scale_before):
                    raise FloatingPointError("Invalid AMP loss scale")
                # Retain the same forward graph for FP16 overflow retries. BatchNorm
                # statistics, random draws and the data cursor advance only once.
                scaler.scale(loss).backward(retain_graph=scaler.is_enabled())
                scaler.unscale_(optimizer)
                grad_norm=torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None])
                if torch.isfinite(grad_norm):
                    clip=config["training"]["gradient_clip"]
                    if clip:
                        torch.nn.utils.clip_grads_with_norm_(parameters,clip,grad_norm)
                    scaler.step(optimizer);scaler.update()
                    break
                if not scaler.is_enabled():
                    raise FloatingPointError(f"Nonfinite gradients at cycle {cycle}, step {step+1}")
                # GradScaler skips this optimizer step and backs off its scale.
                # The discarded attempt is explicitly logged and is not an update.
                scaler.step(optimizer);scaler.update()
                scale_after=scaler.get_scale()
                if scale_after>=scale_before:
                    raise FloatingPointError("Nonfinite gradient norm without a recoverable AMP overflow")
                log("amp_overflow",cycle=cycle,step=step+1,retry=retry+1,
                    scale_before=scale_before,scale_after=scale_after)
                if retry==MAX_AMP_RETRIES:
                    raise FloatingPointError(f"AMP overflow persisted after {MAX_AMP_RETRIES} retries")
            step += 1; total += 1
            update = uuid.uuid4().hex; updates.append(update)
            log("update",cycle=cycle,step=step,total_steps=total,update_id=update,loss=float(loss.detach()),
                policy_loss=float(pl.detach()),value_loss=float(vl.detach()),l2=float(reg.detach()),
                grad_norm=float(grad_norm),amp_retries=retry)
            del loss,pl,vl,reg,policy,value,tensors,batch
            if step % config["training"]["checkpoint_every"] == 0 or step == plan["train_steps"] or stopping():
                reference = commit_checkpoint(root,config,model,optimizer,scaler,cycle,step,total,reader.state(),reference,updates)
                save_json(progress_path,{"checkpoint":reference,"snapshot_id":plan["snapshot_id"]})
                log("checkpoint",cycle=cycle,checkpoint=reference,committed_updates=updates)
                updates = []
        # A stop arriving between updates still commits all successfully completed updates.
        if updates:
            reference = commit_checkpoint(root,config,model,optimizer,scaler,cycle,step,total,reader.state(),reference,updates)
            save_json(progress_path,{"checkpoint":reference,"snapshot_id":plan["snapshot_id"]})
            log("checkpoint",cycle=cycle,checkpoint=reference,committed_updates=updates)
        return reference, step == plan["train_steps"]
    finally:
        reader.close(); del model,optimizer,scaler,checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
