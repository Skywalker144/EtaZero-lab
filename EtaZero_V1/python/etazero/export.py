"""Export an immutable inference model before updating any publication pointer."""
import os
from pathlib import Path
import uuid
import numpy as np
import torch
from .network import make_network, inference_network
from .optimization import inference_weights
from .schema import CONTRACT_ID
from .storage import load_json, save_json, sha256, sync_directory
from .training import load_checkpoint, device_check


def example_inputs(canvas, size, rule, moves=()):
    """Sparse inputs for inference tests; arbitrary Renju positions must use native Game."""
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
                        rule=="renju",0,0],np.float32)
    return obs, globals


def verify_export(run_dir, info, canvas):
    """Verify the immutable artifact before restoring or publishing its pointer."""
    relative = Path('models')/info['id']/'model.pt'
    if (Path(info['path']) != relative or info['contract'] != CONTRACT_ID or
            info['canvas'] != canvas or info['checkpoint']['id'] != info['id'] or
            info['weights'] not in ('model','swa')):
        raise ValueError('Exported model identity/contract mismatch')
    path = Path(run_dir)/relative
    if load_json(path.parent/'manifest.json') != info or sha256(path) != info['sha256']:
        raise ValueError('Exported model manifest/checksum mismatch')
    return info


def export_model(run_dir, config, checkpoint):
    root = Path(run_dir)
    destination = root/"models"/checkpoint["id"]
    if destination.exists():
        info = load_json(destination/"manifest.json")
        if info["checkpoint"] != checkpoint:
            raise ValueError("Existing exported model does not match its source checkpoint")
        return verify_export(root,info,config['network']['canvas'])
    device = device_check(config["devices"]["train"])
    saved = load_checkpoint(root,checkpoint,config)
    with torch.random.fork_rng(devices=[]):
        model = make_network(config); model.load_state_dict(inference_weights(saved),strict=True);model.eval();model.to(device)
        inference = inference_network(model)
        scripted = torch.jit.script(inference)
    parent = destination.parent; parent.mkdir(exist_ok=True)
    stage = parent/(".tmp_"+uuid.uuid4().hex);stage.mkdir()
    scripted.save(str(stage/"model.pt"))
    with (stage/"model.pt").open("rb") as file:
        os.fsync(file.fileno())
    canvas = config["network"]["canvas"]
    swa = saved['optimization']['swa']
    averaged = 0 if swa is None else int(swa['n_averaged'].item())
    info = {"id":checkpoint["id"],"checkpoint":checkpoint,"contract":CONTRACT_ID,"canvas":canvas,
            'weights': 'swa' if averaged else 'model', 'swa_samples': averaged,
            "path":str(Path("models")/checkpoint["id"]/"model.pt"),"sha256":sha256(stage/"model.pt"),
            'inference_precision':config['inference']['inference_precision'],'backend':'libtorch',
            'normalization':'masked_fixup_bias' if model.norm_kind=='fixup' else 'precomputed_inv_std'}
    if config['agent']['algorithm'] == 'muzero':
        info['algorithm'] = 'muzero'
        info['muzero_config'] = config['muzero']
    save_json(stage/"manifest.json",info,immutable=True)
    sync_directory(stage);os.rename(stage,destination);sync_directory(parent)
    return info
