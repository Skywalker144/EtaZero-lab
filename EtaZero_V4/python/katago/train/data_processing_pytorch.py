import logging
import os

import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional

from ..train import modelconfigs
from ..game import xiangqi as xq

# Needs to be kept in sync with GLOBAL_TARGET_NUM_CHANNELS in trainingwrite.cpp C++ code among other places.
# Data format version 2 files (recorded in channel 63 of each row) had only 64 channels; they are zero-padded
# up to this width when loading, which correctly encodes "not reanalyzed" for the version 3 channels.
GLOBAL_TARGETS_NC_CHANNELS = 80

def pad_global_targets_nc(globalTargetsNC: np.ndarray) -> np.ndarray:
    """Zero-pad older-format globalTargetsNC rows up to the current channel count."""
    num_channels = globalTargetsNC.shape[1]
    if num_channels == GLOBAL_TARGETS_NC_CHANNELS:
        return globalTargetsNC
    assert num_channels < GLOBAL_TARGETS_NC_CHANNELS, f"globalTargetsNC has {num_channels} channels, more than the expected {GLOBAL_TARGETS_NC_CHANNELS}"
    padded = np.zeros((globalTargetsNC.shape[0], GLOBAL_TARGETS_NC_CHANNELS), dtype=globalTargetsNC.dtype)
    padded[:, :num_channels] = globalTargetsNC
    return padded

def prefetch_training_files(npz_files, load_npz_file, prefetch_depth=1):
    """Source ordered file prefetch, shared by ordinary and MuZero readers.

    Keep the current file and up to prefetch_depth files ahead. Completion order
    never changes consumption order; close joins loaders on early generator exit.
    """
    if not npz_files:
        return
    prefetch_depth = max(1, prefetch_depth)
    with ThreadPoolExecutor(max_workers=prefetch_depth) as executor:
        pending = deque()
        next_index = 0
        while next_index < len(npz_files) and len(pending) <= prefetch_depth:
            pending.append(executor.submit(load_npz_file, npz_files[next_index]))
            next_index += 1
        while pending:
            loaded = pending.popleft().result()
            if next_index < len(npz_files):
                logging.info(f"Preloading {npz_files[next_index]} while processing this file")
                pending.append(executor.submit(load_npz_file, npz_files[next_index]))
                next_index += 1
            yield loaded


def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: bool,
    include_meta: bool,
    model_config: modelconfigs.ModelConfig,
    prefetch_depth: int = 1,
    rng=None,
    skip_batches: int = 0,
):
    rand = np.random.default_rng(seed=list(os.urandom(12))) if rng is None else rng
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    placement = model_config["version"] == 18
    xiangqi = model_config["version"] == 19
    if xiangqi and (include_meta or pos_len < xq.HEIGHT):
        raise ValueError('Xiangqi data requires a canvas containing 9 x 10 and no Go metadata')
    h_base, h_builder = (None, None) if placement or xiangqi else build_history_matrices(model_config, device)
    mirror_actions = torch.tensor(xq.action_permutation(mirror=True), device=device) if xiangqi and randomize_symmetries else None

    # Version 16 always predicts q values; version 17+ does so only when configured.
    include_qvalues = model_config["version"] == 16 or (
        model_config["version"] >= 17 and bool(model_config.get("predict_q_values"))
    )

    def load_npz_file(npz_file):
        # Select only THIS rank's rows up front, while the arrays are still in
        # their compact on-disk dtypes (packed bits / int8 / int16), so the
        # expensive unpackbits + float32 expansion runs on 1/world_size of the
        # data rather than the whole shard in every rank.
        with np.load(npz_file) as npz:
            if 'muZeroRootNC' in npz or 'muZeroStepKind' in npz:
                raise ValueError(f"{npz_file}: MuZero data requires the trajectory reader")
            num_samples = npz["globalInputNC"].shape[0]
            num_whole_steps = num_samples // (batch_size * world_size)
            used = num_whole_steps * world_size * batch_size

            def select_rank_rows(arr):
                # Keep only the rows this rank will consume:
                # reshape the used prefix to (steps, world_size, batch, ...) and
                # take this rank's slice. For world_size>1 the trailing reshape
                # forces a compact 1/world_size-size copy and lets the full
                # decompressed array be freed immediately.
                # Drop any trailing suffix that doesn't match the overall world batch size.
                arr = arr[:used]
                rest = arr.shape[1:]
                arr = arr.reshape(num_whole_steps, world_size, batch_size, *rest)
                arr = arr[:, rank]
                return arr.reshape(num_whole_steps * batch_size, *rest)

            binaryInputNCHWPacked = select_rank_rows(npz["binaryInputNCHWPacked"])
            globalInputNC = select_rank_rows(npz["globalInputNC"])
            policyTargetsNCMove = select_rank_rows(npz["policyTargetsNCMove"]).astype(np.float32)
            globalTargetsNC = pad_global_targets_nc(select_rank_rows(npz["globalTargetsNC"]))
            if "scoreDistrN" in npz:
                raise ValueError(f"{npz_file}: score supervision is unsupported")
            if policyTargetsNCMove.shape[2] != modelconfigs.get_policy_size(model_config, pos_len):
                raise ValueError(f"{npz_file}: policy action layout does not match the model")
            valueTargetsNCHW = select_rank_rows(npz["valueTargetsNCHW"]).astype(np.float32)
            if valueTargetsNCHW.shape[1] != 2:
                raise ValueError(f"{npz_file}: expected two future-position channels; ownership/seki/scoring targets are unsupported")
            if include_meta:
                metadataInputNC = select_rank_rows(npz["metadataInputNC"]).astype(np.float32)
            else:
                metadataInputNC = None
            if include_qvalues:
                qValueTargetsNCMove = select_rank_rows(npz["qValueTargetsNCMove"]).astype(np.float32)
                if qValueTargetsNCMove.shape[2] != modelconfigs.get_policy_size(model_config, pos_len):
                    raise ValueError(f"{npz_file}: Q action layout does not match the model")
                if qValueTargetsNCMove.shape[1] != 2:
                    raise ValueError(f"{npz_file}: expected Q winloss and visits channels; Q-score is unsupported")
            else:
                qValueTargetsNCMove = None
        del npz

        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
        )).astype(np.float32)

        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        return (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove)

    for loaded in prefetch_training_files(npz_files, load_npz_file, prefetch_depth):
        (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove) = loaded
        num_whole_steps = binaryInputNCHW.shape[0] // batch_size
        logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")

        for n in range(num_whole_steps):
            if skip_batches:
                skip_batches -= 1
                continue
            start = n * batch_size
            end = start + batch_size

            batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
            batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
            batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
            batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
            batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)
            if include_meta:
                batch_metadataInputNC = torch.from_numpy(metadataInputNC[start:end]).to(device)
            if include_qvalues:
                batch_qValueTargetsNCMove = torch.from_numpy(qValueTargetsNCMove[start:end]).to(device)

            if xiangqi:
                batch_binaryInputNCHW, batch_globalInputNC = apply_xiangqi_history(batch_binaryInputNCHW, batch_globalInputNC)
            elif not placement:
                (batch_binaryInputNCHW, batch_globalInputNC) = apply_history_matrices(
                    model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder
                )

            if randomize_symmetries:
                symm = int(rand.integers(0,8))
                # Hex permits identity and 180 degrees, without exchanging colors.
                # The same random draw stays uniform on both the 8- and 2-element sets.
                hex_rows = batch_globalInputNC[:, 6].bool() if placement else None
                def transform(tensor, policy=False):
                    if xiangqi:
                        if symm % 2 == 0:
                            return tensor
                        if policy:
                            return tensor.index_select(-1, mirror_actions)
                        mirrored = tensor.clone()
                        mirrored[..., :xq.HEIGHT, :xq.WIDTH] = tensor[..., :xq.HEIGHT, :xq.WIDTH].flip(-1)
                        return mirrored
                    if policy:
                        fn = lambda t, sym: apply_symmetry_policy(t, sym, pos_len, has_pass=not placement)
                    else:
                        fn = apply_symmetry
                    transformed = fn(tensor, symm)
                    if placement:
                        hex_transformed = fn(tensor, 2 * (symm % 2))
                        selector = hex_rows.reshape((-1,) + (1,) * (tensor.ndim-1))
                        transformed = torch.where(selector, hex_transformed, transformed)
                    return transformed
                batch_binaryInputNCHW = transform(batch_binaryInputNCHW)
                batch_policyTargetsNCMove = transform(batch_policyTargetsNCMove, policy=True)
                batch_valueTargetsNCHW = transform(batch_valueTargetsNCHW)
                if include_qvalues:
                    batch_qValueTargetsNCMove = transform(batch_qValueTargetsNCMove, policy=True)

            batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
            batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
            batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()
            if include_qvalues:
                batch_qValueTargetsNCMove = batch_qValueTargetsNCMove.contiguous()

            batch = dict(
                binaryInputNCHW = batch_binaryInputNCHW,
                globalInputNC = batch_globalInputNC,
                policyTargetsNCMove = batch_policyTargetsNCMove,
                globalTargetsNC = batch_globalTargetsNC,
                valueTargetsNCHW = batch_valueTargetsNCHW,
            )
            if include_meta:
                batch["metadataInputNC"] = batch_metadataInputNC
            if include_qvalues:
                batch["qValueTargetsNCMove"] = batch_qValueTargetsNCMove

            yield batch


def apply_xiangqi_history(spatial, global_input):
    # Same source geometric history dropout, applied to seven complete past
    # piece frames. Rule counters remain visible when feature history is hidden.
    keep = torch.cumsum(torch.rand_like(global_input[:, :xq.HISTORY_FRAMES-1]) >= 0.98, dim=1) == 0
    keep = torch.nn.functional.pad(keep, (1, 0), value=True).to(spatial.dtype)
    frames = spatial[:, 1:].reshape(spatial.shape[0], xq.HISTORY_FRAMES, 14, *spatial.shape[-2:])
    frames = frames * keep[:, :, None, None, None]
    spatial = torch.cat((spatial[:, :1], frames.flatten(1, 2)), dim=1)
    global_input = global_input.clone()
    global_input[:, 5] = torch.minimum(global_input[:, 5], keep.sum(dim=1) / xq.HISTORY_FRAMES)
    return spatial, global_input


def apply_symmetry_policy(tensor, symm, pos_len, has_pass=True):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    if not has_pass:
        return apply_symmetry(tensor.view(batch_size, channels, pos_len, pos_len), symm).reshape(batch_size, channels, pos_len*pos_len)
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)


def build_history_matrices(model_config: modelconfigs.ModelConfig, device):
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    assert num_bin_features == 22, "Currently this code is hardcoded for this many features"

    h_base = torch.diag(
        torch.tensor(
            [
                1.0,  # 0
                1.0,  # 1
                1.0,  # 2
                1.0,  # 3
                1.0,  # 4
                1.0,  # 5
                1.0,  # 6
                1.0,  # 7
                1.0,  # 8
                0.0,  # 9   Location of move 1 turn ago
                0.0,  # 10  Location of move 2 turns ago
                0.0,  # 11  Location of move 3 turns ago
                0.0,  # 12  Location of move 4 turns ago
                0.0,  # 13  Location of move 5 turns ago
                1.0,  # 14  Ladder-threatened stone
                0.0,  # 15  Ladder-threatened stone, 1 turn ago
                0.0,  # 16  Ladder-threatened stone, 2 turns ago
                1.0,  # 17
                1.0,  # 18
                1.0,  # 19
                1.0,  # 20
                1.0,  # 21
            ],
            device=device,
            requires_grad=False,
        )
    )
    # Because we have ladder features that express past states rather than past diffs,
    # the most natural encoding when we have no history is that they were always the
    # same, rather than that they were all zero. So rather than zeroing them we have no
    # history, we add entries in the matrix to copy them over.
    # By default, without history, the ladder features 15 and 16 just copy over from 14.
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    h0 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prev move, we enable feature 9 and 15
    h0[9, 9] = 1.0  # Enable 9 -> 9
    h0[14, 15] = -1.0  # Stop copying 14 -> 15
    h0[14, 16] = -1.0  # Stop copying 14 -> 16
    h0[15, 15] = 1.0  # Enable 15 -> 15
    h0[15, 16] = 1.0  # Start copying 15 -> 16

    h1 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prevprev move, we enable feature 10 and 16
    h1[10, 10] = 1.0  # Enable 10 -> 10
    h1[15, 16] = -1.0  # Stop copying 15 -> 16
    h1[16, 16] = 1.0  # Enable 16 -> 16

    h2 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h2[11, 11] = 1.0

    h3 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h3[12, 12] = 1.0

    h4 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h4[13, 13] = 1.0

    # (1, n_bin, n_bin)
    h_base = h_base.reshape((1, num_bin_features, num_bin_features))
    # (5, n_bin, n_bin)
    h_builder = torch.stack((h0, h1, h2, h3, h4), dim=0)

    return (h_base, h_builder)


def apply_history_matrices(model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder):
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    # include_history = batch_globalTargetsNC[:,36:41]
    should_stop_history = torch.rand_like(batch_globalTargetsNC[:,36:41]) >= 0.98
    include_history = (torch.cumsum(should_stop_history,axis=1,dtype=torch.float32) <= 0.1).to(torch.float32)

    # include_history: (N, 5)
    # bi * ijk -> bjk, (N, 5) * (5, n_bin, n_bin) -> (N, n_bin, n_bin)
    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)


    # batch_binaryInputNCHW: (N, n_bin_in, 19, 19)
    # h_matrix: (N, n_bin_in, n_bin_out)
    # Result: (N, n_bin_out, 19, 19)
    batch_binaryInputNCHW = torch.einsum("bijk,bil->bljk", batch_binaryInputNCHW, h_matrix)

    # First 5 global input features exactly correspond to include_history, pointwise multiply to
    # enable/disable them
    batch_globalInputNC = batch_globalInputNC * torch.nn.functional.pad(
        include_history, ((0, num_global_features - include_history.shape[1])), value=1.0
    )
    return batch_binaryInputNCHW, batch_globalInputNC
