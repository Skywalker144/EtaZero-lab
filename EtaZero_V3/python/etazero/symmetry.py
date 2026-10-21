"""Physical-coordinate records -> model coordinates and game-valid augmentation."""
from functools import lru_cache
import torch
from .schema import HEX_GLOBAL, HEX_WHITE_GLOBAL
from .games import XIANGQI_MOVES, XIANGQI_INDEX, SPECS


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


@lru_cache(maxsize=16)
def xiangqi_maps(canvas, device):
    spatial, actions = [], []
    for choice in range(4):
        def xy(x,y):
            if choice & 2: x,y = 8-x,9-y
            if choice & 1: x = 8-x
            return x,y
        forward = list(range(canvas*canvas))
        for y in range(10):
            for x in range(9):
                u,v = xy(x,y); forward[y*canvas+x] = v*canvas+u
        spatial.append(torch.tensor(forward).argsort())
        actions.append(torch.tensor([XIANGQI_INDEX[(*xy(x,y),*xy(u,v))] for x,y,u,v in XIANGQI_MOVES]))
    return torch.stack(spatial).to(device), torch.stack(actions).to(device)


def augment_batch(batch, symmetry, spec=SPECS['gomoku']):
    if symmetry not in range(8):
        raise ValueError('D4 requires a square tensor and a symmetry in [0,7]')
    obs, globals = batch['obs'], batch['globals']
    canvas = obs.shape[-1]
    if spec.symmetry == 'xiangqi':
        choices = (symmetry%2) + 2*(globals[:,0] == 0).long()
        spatial, forward = xiangqi_maps(canvas, obs.device)
        policy = forward.argsort(1)
    else:
        choices = (torch.where(globals[:, HEX_GLOBAL] != 0,
                   2*(symmetry%2) + 4*(globals[:, HEX_WHITE_GLOBAL] != 0).long(), symmetry)
                   if spec.name in ('gomoku', 'hex') else torch.full((len(obs),),symmetry,device=obs.device,dtype=torch.long))
        spatial, forward = coordinate_maps(canvas, obs.device)
        policy = spatial
        if spec.action_layout == 'points_pass':
            extra = torch.full((8,1),canvas*canvas,device=obs.device,dtype=torch.long)
            policy, forward = torch.cat((policy,extra),1), torch.cat((forward,extra),1)
    def transform(value, mapping):
        shape = value.shape
        flat = value.reshape(len(value), -1, mapping.shape[1])
        return flat.gather(2, mapping[:, None].expand(-1, flat.shape[1], -1)).reshape(shape)
    spatial_mapping = spatial[choices]
    policy_mapping = spatial_mapping if spec.action_layout == 'points' else policy[choices]
    result={**batch,'obs':transform(obs,spatial_mapping)}
    for key in ('policy','opponent_policy','q_values','q_visits'):
        if key in batch:
            result[key]=transform(batch[key],policy_mapping)
    if 'actions' in batch:
        result['actions']=forward[choices].gather(1,batch['actions'].long())
    return result
