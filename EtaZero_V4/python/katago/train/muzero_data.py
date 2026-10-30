"""MuZero continuation targets attached to source-sampled training roots.

Raw shards carry complete games even when their sampled roots span files. The
shuffle window determines a single mean unrounded main-position weight, counting
each game once. Only recurrent real positions receive that normalization; root
multiplicity already implements the source sampler.
"""
import math
import numpy as np
from ..game import xiangqi as xq

TARGETS = ('policyTargetsNCMove', 'globalTargetsNC', 'valueTargetsNCHW', 'qValueTargetsNCMove')
RAW_KEYS = {
    'muZeroRootNC', 'muZeroGameOffsets', 'muZeroGameHash', 'muZeroGameFinished',
    'muZeroActions', 'muZeroWeights', *('muZero' + key[0].upper() + key[1:] for key in TARGETS),
}
INACTIVE, REAL, TERMINAL, TRUNCATED = range(4)


def trajectory_key(key):
    return 'muZero' + key[0].upper() + key[1:]


def view_keys(include_qvalues=True):
    targets = TARGETS if include_qvalues else TARGETS[:-1]
    return ('muZeroActions', 'muZeroStepKind', 'muZeroWeights', *(trajectory_key(k) for k in targets))


def weight_statistics(data):
    """Per-game sums/counts, excluding the final observed state and side roots."""
    offsets, hashes, weights = (data[key] for key in
                               ('muZeroGameOffsets', 'muZeroGameHash', 'muZeroWeights'))
    if (offsets.ndim != 1 or offsets.dtype != np.int64 or hashes.dtype != np.uint64
            or hashes.shape != (len(offsets)-1, 2) or weights.dtype != np.float32
            or weights.ndim != 1 or len(offsets) < 2 or offsets[0] != 0
            or offsets[-1] != len(weights) or np.any(np.diff(offsets) < 1)):
        raise ValueError('Invalid MuZero game boundaries or weight layout')
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('MuZero sampling weights must be finite and nonnegative')
    return [(tuple(map(int, h)), float(weights[lo:hi-1].sum(dtype=np.float64)), int(hi-lo-1))
            for h, lo, hi in zip(hashes, offsets[:-1], offsets[1:])]


def replay_weight_mean(paths):
    """Call on the source shuffle's selected file window, before root subsampling."""
    games = {}
    for path in paths:
        with np.load(path) as data:
            for identity, total, count in weight_statistics(data):
                if identity in games and games[identity] != (total, count):
                    raise ValueError(f'Conflicting MuZero trajectory weights for game {identity}')
                games[identity] = (total, count)
    total = math.fsum(value[0] for value in games.values())
    count = sum(value[1] for value in games.values())
    if count == 0 or total <= 0:
        raise ValueError('MuZero replay window has no positive main-position weight')
    return total / count


def expand_training_rows(data, unroll_steps):
    """Resolve raw references into fixed-K views without sampling or augmentation.

    The ordinary root arrays are unchanged. Added target arrays contain K future
    steps; step kind/weight include the root. Side roots have no recurrent steps.
    A truncated game includes its final observed bootstrap, then stops. Only an
    actual terminal game is extended with absorbing targets and random actions.
    """
    if type(unroll_steps) is not int or unroll_steps < 1:
        raise ValueError('MuZero unroll_steps must be a positive integer')
    # Materialize once: NpzFile decompresses again on every __getitem__.
    data = dict(data)
    if not RAW_KEYS.issubset(data):
        raise ValueError('MuZero raw shard is missing complete trajectories')
    weight_statistics(data)
    roots = data['muZeroRootNC']
    offsets = data['muZeroGameOffsets']
    finished = data['muZeroGameFinished']
    actions = data['muZeroActions']
    n = len(data['globalInputNC'])
    m = int(offsets[-1])
    if (roots.dtype != np.int64 or roots.shape != (n, 2)
            or finished.dtype != np.uint8 or finished.shape != (len(offsets)-1,)
            or np.any(finished > 1) or actions.dtype != np.int32 or actions.shape != (m,)):
        raise ValueError('Invalid MuZero root, endpoint, or action layout')
    if np.any(roots[:, 0] < 0) or np.any(roots[:, 0] >= len(finished)):
        raise ValueError('MuZero root references an absent game')
    lengths = np.diff(offsets) - 1
    if np.any(roots[:, 1] < -1) or np.any(roots[:, 1] >= lengths[roots[:, 0]]):
        raise ValueError('MuZero root references an absent main position')
    for key in TARGETS:
        target = data[trajectory_key(key)]
        root = data[key]
        if target.shape != (m, *root.shape[1:]) or target.dtype != root.dtype:
            raise ValueError(f'MuZero trajectory layout disagrees with roots: {key}')
    area = int(np.prod(data['valueTargetsNCHW'].shape[-2:]))
    policy_size = data['policyTargetsNCMove'].shape[-1]
    xiangqi = policy_size == len(xq.MOVES)
    if not xiangqi and policy_size not in (area, area+1):
        raise ValueError('Unsupported MuZero action layout')
    if xiangqi and (data['binaryInputNCHWPacked'].shape[1] != xq.SPATIAL_FEATURES or
                    data['valueTargetsNCHW'].shape[-2] < xq.HEIGHT or
                    data['valueTargetsNCHW'].shape[-1] < xq.WIDTH):
        raise ValueError('Xiangqi MuZero requires its observation layout and physical board')
    rotate_actions = np.asarray(xq.action_permutation(black=True)) if xiangqi else None
    endpoints = offsets[1:]-1
    if np.any(actions[endpoints] != -1):
        raise ValueError('MuZero final observed states must have no real action')
    real = np.ones(m, dtype=bool)
    real[endpoints] = False
    if np.any(actions[real] < 0) or np.any(actions[real] >= policy_size):
        raise ValueError('MuZero real action outside the model canvas')

    out = {key: value for key, value in data.items() if key not in RAW_KEYS}
    k = unroll_steps
    out['muZeroActions'] = np.full((n, k), -1, np.int32)
    out['muZeroStepKind'] = np.zeros((n, k+1), np.uint8)
    out['muZeroStepKind'][:, 0] = REAL
    out['muZeroWeights'] = np.zeros((n, k+1), np.float32)
    out['muZeroWeights'][:, 0] = 1
    for key in TARGETS:
        out[trajectory_key(key)] = np.zeros((n, k, *data[key].shape[1:]), data[key].dtype)
    # Even masked target rows have finite normalized policy distributions.
    out['muZeroPolicyTargetsNCMove'].fill(1)
    if xiangqi:
        static = np.ones((n, policy_size), np.uint8)
    else:
        static = np.unpackbits(data['binaryInputNCHWPacked'][:, 0], axis=-1)[:, :area]
        if policy_size > area:
            static = np.concatenate((static, np.ones((n, 1), np.uint8)), axis=1)
    for row, (game, start) in enumerate(roots):
        if start < 0:  # Side positions retain the source root targets only.
            continue
        lo, hi = offsets[game:game+2]
        end = hi-lo-1
        for step in range(k+1):
            turn = start + step
            if turn > end and not finished[game]:
                break
            index = lo + min(turn, end)
            if step < k and turn < end:
                out['muZeroActions'][row, step] = rotate_actions[actions[index]] if xiangqi and step % 2 else actions[index]
                if not static[row, actions[index]]:
                    raise ValueError('MuZero real action outside the root board mask')
            if not step:
                continue  # Root quantization/history masks stay exactly as written.
            kind = REAL if turn < end else TERMINAL if finished[game] else TRUNCATED
            out['muZeroStepKind'][row, step] = kind
            out['muZeroWeights'][row, step] = data['muZeroWeights'][index]
            for key in TARGETS:
                out[trajectory_key(key)][row, step-1] = data[trajectory_key(key)][index]
            # Native Xiangqi rows use their own player coordinates. Latent
            # targets retain root coordinates, including terminal board geometry
            # reused beyond the last observed turn. Values still use each ply's
            # player perspective and are not rotated or negated here.
            if xiangqi and (min(turn, end)-start) % 2:
                for key in ('muZeroPolicyTargetsNCMove', 'muZeroQValueTargetsNCMove'):
                    row_target = out[key][row, step-1]
                    row_target[:] = row_target[:, rotate_actions]
                future = out['muZeroValueTargetsNCHW'][row, step-1, :, :xq.HEIGHT, :xq.WIDTH]
                future[:] = future[:, ::-1, ::-1]
            target_global = out['muZeroGlobalTargetsNC'][row, step-1]
            if kind == TERMINAL:
                # Uniform static-domain policies and zero searched Q are the
                # MuZero absorbing targets, not invented real searches.
                out['muZeroPolicyTargetsNCMove'][row, step-1] = static[row]
                target_global[[26, 28]] = 1
                target_global[51] += turn-end
                if (turn-end) % 2:
                    for win, loss in ((0,1), (4,5), (8,9), (12,13), (16,17)):
                        target_global[win], target_global[loss] = target_global[loss], target_global[win]
                    target_global[[47, 57, 61]] *= -1
                    out['muZeroValueTargetsNCHW'][row, step-1] *= -1
    return out


def prepare_batch(data, mean_weight, rng, model_config, device, *, randomize_symmetries=True, history_matrices=None):
    """Prepare one shuffled view batch for unroll_loss, using the caller's RNG.

    Absorbing actions are redrawn on each consumption from the static root
    domain, including Go pass. One source symmetry is shared by the root,
    actions and every future target. Future observations are never materialized.
    """
    # Shuffle workers need only NumPy; do not import the training runtime there.
    import torch
    from . import modelconfigs
    from .data_processing_pytorch import (
        apply_history_matrices, build_history_matrices, apply_symmetry, apply_symmetry_policy, apply_xiangqi_history,
    )

    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError('MuZero replay mean weight must be positive and finite')
    kinds = data['muZeroStepKind']
    actions = data['muZeroActions'].astype(np.int64, copy=True)
    n, k = actions.shape
    if kinds.shape != (n,k+1) or np.any(kinds[:, 0] != REAL) or np.any(kinds > TRUNCATED):
        raise ValueError('Invalid MuZero step kinds')
    active = kinds != INACTIVE
    if np.any(active[:, 1:] & ~active[:, :-1]):
        raise ValueError('MuZero inactive sequence cannot resume')
    weights = data['muZeroWeights'][:, 1:].copy()
    weights = np.where(kinds[:, 1:] == REAL, weights / mean_weight, weights)
    spatial_shape = data['valueTargetsNCHW'].shape[-2:]
    h, w = spatial_shape
    if h != w:
        raise ValueError('Source training augmentation requires a square canvas')
    if 'binaryInputNCHW' in data:
        spatial = data['binaryInputNCHW']
    else:
        spatial = np.unpackbits(data['binaryInputNCHWPacked'], axis=-1)[..., :h*w]
        spatial = spatial.reshape(n, -1, h, w).astype(np.float32)
    has_pass = modelconfigs.has_pass(model_config)
    xiangqi = model_config['version'] == 19
    for row in range(n):
        domain = np.arange(len(xq.MOVES)) if xiangqi else np.flatnonzero(spatial[row, 0].ravel())
        if has_pass:
            domain = np.r_[domain, h*w]
        if len(domain) == 0:
            raise ValueError('MuZero root board mask is empty')
        absorbing = np.flatnonzero((kinds[row, :-1] == TERMINAL) & active[row, 1:])
        actions[row, absorbing] = rng.choice(domain, len(absorbing))
        # Inactive transitions are removed before dynamics; give them an inert
        # in-domain value so they never carry the on-disk sentinel to a kernel.
        actions[row, ~active[row, 1:]] = domain[0]
        if not np.all(np.isin(actions[row], domain)):
            raise ValueError('MuZero action is outside the static root domain')

    def tensor(array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)

    root = {'binaryInputNCHW': tensor(spatial), 'globalInputNC': tensor(data['globalInputNC'])}
    if 'metadataInputNC' in data:
        root['metadataInputNC'] = tensor(data['metadataInputNC'])
    target_keys = TARGETS if 'qValueTargetsNCMove' in data else TARGETS[:-1]
    predicts_q = model_config['version'] == 16 or (
        model_config['version'] >= 17 and bool(model_config.get('predict_q_values')))
    if predicts_q and 'qValueTargetsNCMove' not in data:
        raise ValueError('MuZero Q-predicting model requires Q targets')
    targets = [{key: tensor(data[key].astype(np.float32, copy=False)) for key in target_keys}]
    for step in range(k):
        values = {key: data[trajectory_key(key)][:, step].astype(np.float32, copy=True) for key in target_keys}
        values['globalTargetsNC'][:, 25] *= weights[:, step]
        targets.append({key: tensor(value) for key, value in values.items()})
    placement = model_config['version'] == 18
    if xiangqi:
        root['binaryInputNCHW'], root['globalInputNC'] = apply_xiangqi_history(root['binaryInputNCHW'], root['globalInputNC'])
    elif not placement:
        base, builder = (build_history_matrices(model_config, device)
                         if history_matrices is None else history_matrices)
        root['binaryInputNCHW'], root['globalInputNC'] = apply_history_matrices(
            model_config, root['binaryInputNCHW'], root['globalInputNC'], targets[0]['globalTargetsNC'], base, builder)
    if randomize_symmetries:
        sym = int(rng.integers(0, 8))
        hex_rows = data['globalInputNC'][:, 6].astype(bool) if placement else np.zeros(n, bool)
        mirror_actions = np.asarray(xq.action_permutation(mirror=True)) if xiangqi and sym % 2 else None
        mirror_tensor = tensor(mirror_actions) if mirror_actions is not None else None

        def transform(value, policy=False):
            if xiangqi:
                if mirror_tensor is None:
                    return value
                if policy:
                    return value.index_select(-1, mirror_tensor)
                mirrored = value.clone()
                mirrored[..., :xq.HEIGHT, :xq.WIDTH] = value[..., :xq.HEIGHT, :xq.WIDTH].flip(-1)
                return mirrored
            fn = (lambda x, s: apply_symmetry_policy(x, s, w, has_pass)) if policy else apply_symmetry
            transformed = fn(value, sym)
            if placement:
                selector = tensor(hex_rows).reshape((-1,) + (1,)*(value.ndim-1))
                transformed = torch.where(selector, fn(value, 2*(sym % 2)), transformed)
            return transformed.contiguous()

        root['binaryInputNCHW'] = transform(root['binaryInputNCHW'])
        for target in targets:
            for key in ('policyTargetsNCMove', 'qValueTargetsNCMove'):
                if key in target:
                    target[key] = transform(target[key], policy=True)
            target['valueTargetsNCHW'] = transform(target['valueTargetsNCHW'])
        # Invert the transformed index grid: map original actions to new cells.
        if xiangqi:
            if mirror_actions is not None:
                actions = mirror_actions[actions]
        else:
            labels = torch.arange(h*w).reshape(1, 1, h, w)
            for chosen, rows in ((sym, ~hex_rows), (2*(sym % 2), hex_rows)):
                mapping = apply_symmetry(labels, chosen).flatten().numpy().argsort()
                if has_pass:
                    mapping = np.r_[mapping, h*w]
                actions[rows] = mapping[actions[rows]]
    root['binaryInputNCHW'] = root['binaryInputNCHW'].contiguous()
    # CPU activity lets unroll remove inactive rows without a device sync.
    return root, targets, tensor(actions), torch.from_numpy(active)


def read_npz_muzero_data(npz_files, batch_size, pos_len, device, model_config, *,
                         unroll_steps, mean_weight, rng, include_meta=False,
                         randomize_symmetries=True, prefetch_depth=1):
    """Single-learner file reader with source prefetch and whole-batch consumption.

    Input is a source-shuffled snapshot. Its metadata supplies unroll_steps and
    mean_weight; rng belongs to the learner and is consumed only on this thread,
    never by file prefetch workers. Each yielded tuple is unroll_loss's root,
    K+1 target dictionaries, actions and CPU activity mask.
    """
    import logging
    from . import modelconfigs
    from .data_processing_pytorch import prefetch_training_files, build_history_matrices

    if batch_size < 1 or unroll_steps < 1:
        raise ValueError('MuZero batch size and unroll length must be positive')
    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError('MuZero snapshot requires a positive replay mean weight')
    include_qvalues = model_config['version'] == 16 or (
        model_config['version'] >= 17 and bool(model_config.get('predict_q_values')))
    keys = ['binaryInputNCHWPacked', 'globalInputNC', *TARGETS[:-1]]
    if include_qvalues:
        keys.append('qValueTargetsNCMove')
    if include_meta:
        keys.append('metadataInputNC')
    keys += view_keys(include_qvalues)
    spatial_channels = modelconfigs.get_num_bin_input_features(model_config)
    global_channels = modelconfigs.get_num_global_input_features(model_config)
    action_size = modelconfigs.get_policy_size(model_config, pos_len)
    history = None if model_config['version'] in (18, 19) else build_history_matrices(model_config, device)

    def load_file(path):
        with np.load(path) as npz:
            if 'muZeroRootNC' in npz or 'muZeroStepKind' not in npz:
                raise ValueError(f'{path}: MuZero reader requires a shuffled unroll view')
            n = len(npz['globalInputNC'])
            used = n // batch_size * batch_size
            data = {key: npz[key][:used] for key in keys}
        shapes = {
            'binaryInputNCHWPacked': (used, spatial_channels, (pos_len**2+7)//8),
            'globalInputNC': (used, global_channels),
            'policyTargetsNCMove': (used, 2, action_size),
            'globalTargetsNC': (used, 80),
            'valueTargetsNCHW': (used, 2, pos_len, pos_len),
            'muZeroActions': (used, unroll_steps),
            'muZeroStepKind': (used, unroll_steps+1),
            'muZeroWeights': (used, unroll_steps+1),
        }
        if include_qvalues:
            shapes['qValueTargetsNCMove'] = (used, 2, action_size)
        for key in TARGETS:
            if key in shapes:
                shapes[trajectory_key(key)] = (used, unroll_steps, *shapes[key][1:])
        for key, shape in shapes.items():
            if data[key].shape != shape:
                raise ValueError(f'{path}: invalid MuZero layout for {key}, expected {shape}')
        packed = data.pop('binaryInputNCHWPacked')
        spatial = np.unpackbits(packed, axis=-1)[..., :pos_len**2]
        data['binaryInputNCHW'] = spatial.reshape(used, spatial_channels, pos_len, pos_len).astype(np.float32)
        return path, data

    for path, data in prefetch_training_files(npz_files, load_file, prefetch_depth):
        steps = len(data['globalInputNC']) // batch_size
        logging.info(f'Beginning {path} with {steps} usable MuZero batches')
        for index in range(steps):
            batch = {key: value[index*batch_size:(index+1)*batch_size] for key, value in data.items()}
            yield prepare_batch(batch, mean_weight, rng, model_config, device,
                                randomize_symmetries=randomize_symmetries, history_matrices=history)
