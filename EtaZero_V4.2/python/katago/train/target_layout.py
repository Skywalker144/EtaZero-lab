import numpy as np

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
