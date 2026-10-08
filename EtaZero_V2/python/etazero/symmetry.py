"""Physical-coordinate records -> model coordinates and game-valid augmentation."""
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


def transform_rows(tensor, globals, symmetry):
    """Hex uses identity/180 and White transpose; Gomoku keeps uniform D4.

    A MuZero sequence uses its root's globals for every target and action.
    """
    shape=(len(tensor),)+(1,)*(tensor.ndim-1)
    hex=(globals[:,HEX_GLOBAL]!=0).reshape(shape)
    white=(globals[:,HEX_WHITE_GLOBAL]!=0).reshape(shape)
    rotated=apply_symmetry(tensor,2*(symmetry%2))
    canonical=torch.where(white,rotated.transpose(-2,-1),rotated)
    return torch.where(hex,canonical,apply_symmetry(tensor,symmetry))


def augment_batch(batch, symmetry):
    canvas = batch['obs'].shape[-1]
    result={**batch,'obs':transform_rows(batch['obs'],batch['globals'],symmetry).contiguous()}
    for key in ('policy','opponent_policy','q_values','q_visits'):
        if key in batch:
            value=batch[key]
            result[key]=transform_rows(value.reshape(*value.shape[:-1],canvas,canvas),batch['globals'],symmetry).reshape_as(value).contiguous()
    if 'actions' in batch:
        grid=torch.arange(canvas*canvas,device=batch['actions'].device).reshape(1,canvas,canvas).expand(len(batch['actions']),-1,-1)
        inverse=transform_rows(grid,batch['globals'],symmetry).flatten(1).argsort(1)
        result['actions']=inverse.gather(1,batch['actions'].long())
    return result
