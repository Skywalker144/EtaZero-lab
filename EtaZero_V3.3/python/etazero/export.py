"""Export an immutable inference model before updating any publication pointer."""
import os
import copy
from pathlib import Path
import uuid
import numpy as np
import torch
from .network import make_network, inference_network
from .games import layout_metadata
from .optimization import inference_weights
from .storage import load_json, save_json, sync_directory
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
                        rule=="renju",0,0,rule=="hex",rule=="hex" and player==-1],np.float32)
    return obs, globals


def verify_export(run_dir, info, canvas):
    """Verify the immutable artifact before restoring or publishing its pointer."""
    filename = 'model.bin' if info['backend'] == 'cuda' else 'model.pt'
    relative = Path('models')/info['id']/filename
    if (Path(info['path']) != relative or info['canvas'] != canvas or info['checkpoint']['id'] != info['id'] or
            info['weights'] not in ('model','swa')):
        raise ValueError('Exported model identity/canvas mismatch')
    path = Path(run_dir)/relative
    if load_json(path.parent/'manifest.json') != info or path.stat().st_size != info['bytes']:
        raise ValueError('Exported model manifest/size mismatch')
    expected = {'model.bin', 'reference.pt'} if info['backend'] == 'cuda' else {'model.pt'}
    if info['backend'] == 'cuda' and info['algorithm'] == 'muzero':
        expected.update(('representation.bin', 'dynamics.bin'))
    if set(info['files']) != expected:
        raise ValueError('Exported model package components mismatch')
    for name, size in info['files'].items():
        if (path.parent/name).stat().st_size != size:
            raise ValueError(f'Exported model component size mismatch: {name}')
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
    # An explicitly configured CPU run uses the reference implementation.
    native_cuda = config['devices']['selfplay'].startswith('cuda')
    filename = 'model.bin' if native_cuda else 'model.pt'
    scripted.save(str(stage/('reference.pt' if native_cuda else filename)))
    if native_cuda:
        from .cuda_export import export_source
        if config['agent']['algorithm']=='muzero':
            export_source(copy.deepcopy(model.representation.body).cpu().eval(),stage/'representation.bin',checkpoint['id']+'-representation')
            export_source(copy.deepcopy(model.dynamics.body).cpu().eval(),stage/'dynamics.bin',checkpoint['id']+'-dynamics')
            export_source(model.prediction.export_model().cpu().eval(),stage/filename,checkpoint['id']+'-prediction')
        else:
            export_source(copy.deepcopy(model.network).cpu().eval(), stage/filename, checkpoint['id'])
    files = {p.name: p.stat().st_size for p in stage.iterdir()}
    for name in files:
        with (stage/name).open('rb') as file:
            os.fsync(file.fileno())
    canvas = config["network"]["canvas"]
    swa = saved['optimization']['swa']
    averaged = 0 if swa is None else int(swa['n_averaged'].item())
    info = {**layout_metadata(config), "id":checkpoint["id"],"checkpoint":checkpoint,"canvas":canvas,
            'weights': 'swa' if averaged else 'model', 'swa_samples': averaged,
            "path":str(Path("models")/checkpoint["id"]/filename),
            "bytes":(stage/filename).stat().st_size, "files":files,
            'inference_precision':config['inference']['inference_precision'],'backend':'cuda' if native_cuda else 'libtorch','algorithm':config['agent']['algorithm'],
            'normalization': {'fixup': 'masked_fixup_bias', 'fixscaleonenorm': 'precomputed_inv_std',
                              'masked_layernorm': 'masked_layernorm', 'fixscale': 'masked_fixscale',
                              'bnorm': 'masked_batchnorm', 'brenorm': 'masked_batchrenorm',
                              'fixbrenorm': 'masked_fixbrenorm'}[model.norm_kind]}
    if hasattr(model, 'source_config'):
        info['katago_architecture'] = model.architecture
        info['katago_config'] = model.source_config
        info['normalization'] = 'source_' + model.norm_kind
    if config['agent']['algorithm'] == 'muzero':
        info['algorithm'] = 'muzero'
        info['muzero_config'] = config['muzero']
    save_json(stage/"manifest.json",info,immutable=True)
    sync_directory(stage);os.rename(stage,destination);sync_directory(parent)
    return info
