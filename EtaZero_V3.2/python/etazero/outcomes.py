"""Terminal reason codes shared with native rules, records and evaluation.

Winner is absolute (+1 first player, -1 second player). No-result has neutral
search WDL but supplies no outcome-supervised training rows or arena score.
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


REASONS = tuple(reason.name.lower() for reason in Reason)


def has_result(reason):
    return reason != Reason.NO_RESULT


def ordinary_wdl(value, dtype=None):
    """One-hot W/D/L for a known ordinary result in the requested perspective."""
    import numpy as np
    value = np.asarray(value)
    return np.stack((value == 1, value == 0, value == -1), axis=-1).astype(dtype or np.float32)


def terminal_wdl(winner, reason, player=1, dtype=None):
    import numpy as np
    wdl = ordinary_wdl(np.asarray(winner) * player, dtype)
    ordinary = has_result(reason)
    if not (ordinary if np.isscalar(ordinary) else ordinary.all()):
        wdl = np.where(np.asarray(ordinary)[..., None], wdl, np.array([.5, 0, .5], dtype=wdl.dtype))
    return wdl


def generate_cpp():
    return 'enum class Reason { ' + ', '.join(f'{r.name}={r.value}' for r in Reason) + ' };\n'
