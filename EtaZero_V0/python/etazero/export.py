"""Export and verify an immutable inference model before updating any publication pointer."""
import json
import os
from pathlib import Path
import subprocess
import uuid
import numpy as np
import torch
from .network import make_network
from .schema import CONTRACT_ID
from .storage import load_json, save_json, sha256, sync_directory
from .training import load_checkpoint, device_check
from .config import csv


def example_observation(canvas, size, rule, moves=()):
    board = np.zeros((canvas,canvas),np.int8)
    for i,a in enumerate(moves):
        board[a//canvas,a%canvas] = 1 if i%2==0 else -1
    player = 1 if len(moves)%2==0 else -1
    obs = np.zeros((6,canvas,canvas),np.float32)
    obs[0] = board==player; obs[1] = board==-player
    obs[2,:size,:size] = player==1; obs[3,:size,:size] = 1
    obs[4,:size,:size] = rule=="standard"; obs[5,:size,:size] = rule=="renju"
    return obs


def export_model(run_dir, config, checkpoint, binary):
    root = Path(run_dir)
    destination = root/"models"/checkpoint["id"]
    if destination.exists():
        info = load_json(destination/"manifest.json")
        if info["checkpoint"] != checkpoint or sha256(destination/"model.pt") != info["sha256"]:
            raise ValueError("Existing exported model does not match its source checkpoint")
        return info
    device = device_check(config["devices"]["train"])
    saved = load_checkpoint(root,checkpoint,config)
    with torch.random.fork_rng(devices=[]):
        model = make_network(config); model.load_state_dict(saved["model"],strict=True);model.eval()
        scripted = torch.jit.script(model)
    parent = destination.parent; parent.mkdir(exist_ok=True)
    stage = parent/(".tmp_"+uuid.uuid4().hex);stage.mkdir()
    scripted.save(str(stage/"model.pt"))
    with (stage/"model.pt").open("rb") as file:
        os.fsync(file.fileno())
    # Mixed rules and sizes exercise the exported model before it is published.
    sizes, rules = csv(config["environment"]["sizes"],int), csv(config["environment"]["rules"])
    canvas = config["network"]["canvas"]
    inputs = np.stack([example_observation(canvas,s,r,(0,canvas)) for s in sizes for r in rules])
    model.to(device);scripted.to(device)
    with torch.inference_mode():
        eager = model(torch.from_numpy(inputs).to(device))
        jit = scripted(torch.from_numpy(inputs).to(device))
        for x,y in zip(eager,jit):
            torch.testing.assert_close(x,y,rtol=2e-4,atol=2e-5)
            if not torch.isfinite(y).all():
                raise ValueError("Nonfinite exported model output")
    command = [str(binary),"infer","--config",str(root/"effective.cfg"),"--model",str(stage/"model.pt"),
               "--model-id",checkpoint["id"],"--device",config["devices"]["train"],"--size",str(sizes[0]),
               "--rule",rules[0],"--moves",f"0,{canvas}"]
    output = subprocess.run(command,text=True,capture_output=True,check=True)
    native = json.loads(output.stdout)
    np.testing.assert_allclose(native["raw_logits"],eager[0][0].cpu().numpy(),rtol=2e-4,atol=2e-5)
    np.testing.assert_allclose(native["raw_value"],eager[1][0].cpu().numpy(),rtol=2e-4,atol=2e-5)
    info = {"id":checkpoint["id"],"checkpoint":checkpoint,"contract":CONTRACT_ID,"canvas":canvas,
            "path":str(Path("models")/checkpoint["id"]/"model.pt"),"sha256":sha256(stage/"model.pt"),
            "verification":{"python_scripted":True,"native":True,"rtol":2e-4,"atol":2e-5}}
    save_json(stage/"manifest.json",info,immutable=True)
    sync_directory(stage);os.rename(stage,destination);sync_directory(parent)
    return info
