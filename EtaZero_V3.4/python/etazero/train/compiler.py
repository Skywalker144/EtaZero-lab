"""Persistent compiler caches and wall-time accounting for synchronous training."""
import math
import os
from pathlib import Path
import time
import warnings


def configure_cache():
    # Run before importing torch, including in fresh autoexp/spawn children.
    # Keep explicit overrides, and let the compiler create directories on demand.
    cache = Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'etazero/compile-v1'
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(cache/'inductor'))
    os.environ.setdefault('TRITON_CACHE_DIR', str(cache/'triton'))


def compilation_seconds():
    from torch._dynamo.utils import calculate_time_spent
    # PyTorch's non-overlapping outer forward and lazy-backward compile spans.
    # Summing backend/Triton timings would count nested/parallel work twice.
    return calculate_time_spent()['total_wall_time']


def compilation_enabled(config):
    # Muon compiles its batched orthogonalization even with an eager network.
    return config['training']['compile'] or (
        config['optimizer']['kind'] == 'muon' and os.environ.get('KATAGO_MUON_BATCHED_NS', '1') == '1')


class WorkTimer:
    """Measure a phase/round, excluding compiler work without skipping updates.

    Counters are process cumulative; snapshot them for every attempt, including
    resumes. This includes cache loading, later recompiles and validation graphs.
    The controller runs one learner at a time and never resets Dynamo mid-round.
    """
    def __init__(self, compiled):
        self.compiled = compiled
        self.compile_start = self._compile_seconds() if compiled else 0.0
        self.start = time.monotonic()

    @staticmethod
    def _compile_seconds():
        try:
            return compilation_seconds()
        except Exception as error:
            warnings.warn(f'Compiler timing unavailable: {error}', RuntimeWarning)
            return float('nan')

    def finish(self):
        wall = time.monotonic()-self.start
        compile_time = (self._compile_seconds()-self.compile_start) if self.compiled else 0.0
        if not all(math.isfinite(v) for v in (wall, compile_time)) or not 0 <= compile_time <= wall:
            warnings.warn(f'Invalid compiler timing: wall={wall}, compile={compile_time}; '
                          'using wall time for budgets, excluding this run from equal-time comparisons', RuntimeWarning)
            return dict(seconds=wall, wall_seconds=wall, compile_seconds=0., timing_valid=False)
        return dict(seconds=wall-compile_time, wall_seconds=wall, compile_seconds=compile_time, timing_valid=True)


def add_timing(previous, measured):
    return {**{key: previous[key]+measured[key] for key in ('seconds', 'wall_seconds', 'compile_seconds')},
            'timing_valid': previous.get('timing_valid', True) and measured['timing_valid']}
