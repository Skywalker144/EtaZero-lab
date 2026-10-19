"""Physical-coordinate records -> model coordinates and game-valid augmentation."""
from functools import lru_cache
import torch
from .schema import HEX_GLOBAL, HEX_WHITE_GLOBAL


def apply_symmetry(tensor, symmetry):
    if tensor.shape[-1] != tensor.shape[-2] or symmetry not in range(8):
        raise ValueError('D4 requires a square tensor and a symmetry in [0,7]')
    if symmetry == 0:
        return tensor
    if symmetry == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symmetry == 2:
        return tensor.flip(-1).flip(-2)
    if symmetry == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symmetry == 4:
        return tensor.transpose(-2, -1)
    if symmetry == 5:
        return tensor.flip(-1)
    if symmetry == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    return tensor.flip(-2)


@lru_cache(maxsize=32)
def coordinate_maps(canvas, device):
    # Build both directions once on CPU. Spatial targets gather output->input;
    # physical actions use input->output. No per-batch board-sized sort.
    grid = torch.arange(canvas*canvas).reshape(canvas, canvas)
    spatial = torch.stack([apply_symmetry(grid, s).flatten() for s in range(8)])
    return spatial.to(device), spatial.argsort(1).to(device)


def augment_batch(batch, symmetry):
    if symmetry not in range(8):
        raise ValueError('D4 requires a symmetry in [0,7]')
    canvas = batch['obs'].shape[-1]
    globals = batch['globals']
    # Hex's 180-degree rotation followed by White transpose is D4 6;
    # the other Hex cases are D4 0, 2, 4. Select one map per row and reuse
    # it for every target instead of materializing both games' transforms.
    choices = torch.where(globals[:, HEX_GLOBAL] != 0,
                          2*(symmetry%2) + 4*(globals[:, HEX_WHITE_GLOBAL] != 0).long(), symmetry)
    spatial, actions = coordinate_maps(canvas, batch['obs'].device)
    mapping = spatial[choices]
    def transform(value):
        shape = value.shape
        flat = value.reshape(len(value), -1, canvas*canvas)
        return flat.gather(2, mapping[:, None].expand(-1, flat.shape[1], -1)).reshape(shape)
    result={**batch,'obs':transform(batch['obs'])}
    for key in ('policy','opponent_policy','q_values','q_visits'):
        if key in batch:
            result[key]=transform(batch[key])
    if 'actions' in batch:
        result['actions']=actions[choices].gather(1,batch['actions'].long())
    return result
