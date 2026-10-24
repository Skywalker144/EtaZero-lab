"""Terminal reason codes shared with native rules, records and evaluation.

Winner is absolute (+1 first player, -1 second player). The shared categorical
head is [win, draw, loss, no_result]; utility is P(win)-P(loss). No-result is a
rules verdict and receives value supervision, but supplies no arena score.
"""
from enum import IntEnum


class Reason(IntEnum):
    DRAW = 0
    FIVE = 1
    FORBIDDEN = 2
    CONNECTION = 3
    SCORE = 4
    MATE = 5
    REPETITION = 6
    MOVE_LIMIT = 7
    NO_RESULT = 8
    INSUFFICIENT_MATERIAL = 9
    RESIGNATION = 10


REASONS = tuple(reason.name.lower() for reason in Reason)
OUTCOME_CHANNELS = 4
REVERSE_OUTCOME = (2, 1, 0, 3)


def has_result(reason):
    return reason != Reason.NO_RESULT


def ordinary_wdl(value, dtype=None):
    """One-hot W/D/L/NR for a known ordinary result in the requested perspective."""
    import numpy as np
    value = np.asarray(value)
    return np.stack((value == 1, value == 0, value == -1, np.zeros_like(value, dtype=bool)), axis=-1).astype(dtype or np.float32)


def terminal_wdl(winner, reason, player=1, dtype=None):
    import numpy as np
    wdl = ordinary_wdl(np.asarray(winner) * player, dtype)
    ordinary = has_result(reason)
    if not (ordinary if np.isscalar(ordinary) else ordinary.all()):
        wdl = np.where(np.asarray(ordinary)[..., None], wdl, np.array([0, 0, 0, 1], dtype=wdl.dtype))
    return wdl


def generate_cpp():
    return 'enum class Reason { ' + ', '.join(f'{r.name}={r.value}' for r in Reason) + ' };\n'
