"""Observed controller wall time and process-local compiler receipts."""
import json
import math
import time
import warnings


class RoundClock:
    def __init__(self, progress, now=None):
        self.progress = progress
        self.now = time.monotonic if now is None else now
        self.last = self.now()

    def settle(self):
        current = self.now()
        delta = current - self.last
        if not math.isfinite(delta) or delta < 0:
            raise RuntimeError('Controller monotonic clock is invalid')
        self.progress['wall_seconds'] += delta
        self.last = current


def compiler_seconds(receipt, stage_wall):
    try:
        record = json.loads(receipt.read_text())
        value = record['compile_seconds']
        if (record['timing_valid'] and type(value) in (int, float) and math.isfinite(value)
                and 0 <= value <= record['wall_seconds'] <= stage_wall):
            return value
        raise ValueError(f'invalid compiler interval: {record}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.warn(f'Learner compiler timing unavailable ({receipt}): {error}; '
                      'using observed wall time for budgets, excluding this run from equal-time comparisons', RuntimeWarning)
        return None


def committed_timing(previous, progress):
    wall = previous['wall_seconds'] + progress['wall_seconds']
    compiled = (None if previous['compile_seconds'] is None or progress['compile_seconds'] is None else
                previous['compile_seconds'] + progress['compile_seconds'])
    valid = previous['timing_valid'] and progress['timing_valid'] and compiled is not None
    if valid and not 0 <= compiled <= wall:
        raise RuntimeError('Committed compiler time exceeds observed round wall time')
    return dict(wall_seconds=wall, compile_seconds=compiled, timing_valid=valid,
                elapsed_seconds=wall - compiled if valid else wall,
                timing_basis='committed_wall_excluding_compile' if valid else 'observed_wall_incomplete')
