"""Training D4 transforms, numbered as in SkyZero V8.1's KataGo reader."""


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


def augment_batch(batch, symmetry):
    canvas = batch['obs'].shape[-1]
    result={**batch,'obs':apply_symmetry(batch['obs'],symmetry).contiguous()}
    for key in ('policy','opponent_policy','q_values','q_visits'):
        if key in batch:
            result[key]=apply_symmetry(batch[key].reshape(-1,canvas,canvas),symmetry).reshape(-1,canvas*canvas).contiguous()
    return result
