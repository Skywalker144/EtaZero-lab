"""Export and verify an immutable inference model before updating any publication pointer."""
import json
import os
from pathlib import Path
import subprocess
import uuid
import numpy as np
import torch
from .network import make_network, inference_network
from .optimization import inference_weights
from .schema import CONTRACT_ID
from .storage import load_json, save_json, sha256, sync_directory
from .training import load_checkpoint, device_check
from .config import csv


def example_inputs(canvas, size, rule, moves=()):
    """Sparse export probes only; arbitrary Renju positions must use native Game."""
    if len(moves) > 3:
        raise ValueError("Export probes require at most three stones (no possible forbidden points)")
    board = np.zeros((canvas,canvas),np.int8)
    for i,a in enumerate(moves):
        board[a//canvas,a%canvas] = 1 if i%2==0 else -1
    player = 1 if len(moves)%2==0 else -1
    obs = np.zeros((5,canvas,canvas),np.float32)
    obs[0,:size,:size] = 1
    obs[1] = board==player; obs[2] = board==-player
    globals = np.array([rule=="standard", rule=="renju", -player if rule=="renju" else 0,
                        rule=="renju"],np.float32)
    return obs, globals


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
        model = make_network(config); model.load_state_dict(inference_weights(saved),strict=True);model.eval();model.to(device)
        scripted = torch.jit.script(inference_network(model))
    parent = destination.parent; parent.mkdir(exist_ok=True)
    stage = parent/(".tmp_"+uuid.uuid4().hex);stage.mkdir()
    scripted.save(str(stage/"model.pt"))
    with (stage/"model.pt").open("rb") as file:
        os.fsync(file.fileno())
    # Mixed rules and sizes exercise the exported model before it is published.
    sizes, rules = csv(config["environment"]["sizes"],int), csv(config["environment"]["rules"])
    canvas = config["network"]["canvas"]
    probes = [example_inputs(canvas,s,r,(0,canvas)) for s in sizes for r in rules]
    tensor_inputs = tuple(torch.from_numpy(np.stack(x)).to(device) for x in zip(*probes))
    with torch.inference_mode():
        eager = model(*tensor_inputs)
        # Validate both initial execution and the graph optimized after profiling.
        for _ in range(3):
            jit = scripted(*tensor_inputs)
            for x,y in zip(eager,jit):
                torch.testing.assert_close(x,y,rtol=2e-4,atol=2e-5)
                if not torch.isfinite(y).all():
                    raise ValueError("Nonfinite exported model output")
    command = [str(binary),"infer","--config",str(root/"config/effective.cfg"),"--model",str(stage/"model.pt"),
               "--model-id",checkpoint["id"],"--device",config["devices"]["train"],"--size",str(sizes[0]),
               "--rule",rules[0],"--moves",f"0,{canvas}"]
    output = subprocess.run(command,text=True,capture_output=True,check=True)
    native = json.loads(output.stdout)
    precision=config["selfplay"]["inference_precision"]
    rtol,atol=(3e-3,3e-3) if precision=="float16" else (2e-4,2e-5)
    # Native infer evaluates one position. TF32 convolution kernels can differ
    # with batch size, so use the same shape for its eager reference.
    with torch.inference_mode():
        native_reference=model(*(x[:1] for x in tensor_inputs))
    np.testing.assert_allclose(native["raw_logits"],native_reference[0][0].cpu().numpy(),rtol=rtol,atol=atol)
    wdl=torch.softmax(native_reference[1][0].float(),dim=0).cpu().numpy()
    np.testing.assert_allclose(native["raw_wdl"],wdl,rtol=rtol,atol=atol)
    np.testing.assert_allclose(native["raw_value"],wdl[0]-wdl[2],rtol=rtol,atol=atol)
    averaged = int(saved['optimization']['swa']['n_averaged'].item())
    info = {"id":checkpoint["id"],"checkpoint":checkpoint,"contract":CONTRACT_ID,"canvas":canvas,
            'weights': 'swa' if averaged else 'model', 'swa_samples': averaged,
            "path":str(Path("models")/checkpoint["id"]/"model.pt"),"sha256":sha256(stage/"model.pt"),
            "verification":{"python_scripted":True,"native":True,"rtol":rtol,"atol":atol,
                            'inference_precision':precision,'backend':'libtorch','normalization':'precomputed_inv_std'}}
    save_json(stage/"manifest.json",info,immutable=True)
    sync_directory(stage);os.rename(stage,destination);sync_directory(parent)
    return info
