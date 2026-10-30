"""Single-device RNG and AMP state at the source learner's checkpoint boundary."""
import random
import numpy as np
import torch


def capture_runtime(train_rng, val_rng, scaler, device):
    numpy_state = np.random.get_state()
    return dict(train_rng=train_rng.bit_generator.state, val_rng=val_rng.bit_generator.state,
                python_rng=random.getstate(),
                numpy_rng=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                scaler=scaler.state_dict() if scaler is not None else None)


def restore_runtime(state, train_rng, val_rng, scaler, device):
    train_rng.bit_generator.state = state['train_rng']
    val_rng.bit_generator.state = state['val_rng']
    random.setstate(state['python_rng'])
    numpy_state = state['numpy_rng']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch_rng'].cpu())
    if device.type == 'cuda' and state['cuda_rng'] is not None:
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
    if scaler is not None and state['scaler'] is not None:
        scaler.load_state_dict(state['scaler'])


