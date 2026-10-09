"""Persistent compiler caches and wall-time accounting for synchronous training."""
import math
import os
from pathlib import Path
import time


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


class WorkTimer:
    """Measure a phase/round, excluding compiler work without skipping updates.

    Counters are process cumulative; snapshot them for every attempt, including
    resumes. This includes cache loading, later recompiles and validation graphs.
    The controller runs one learner at a time and never resets Dynamo mid-round.
    """
    def __init__(self, compiled):
        self.compiled = compiled
        self.compile_start = compilation_seconds() if compiled else 0.0
        self.start = time.monotonic()

    def finish(self):
        wall = time.monotonic()-self.start
        compile_time = (compilation_seconds()-self.compile_start) if self.compiled else 0.0
        if not all(math.isfinite(v) for v in (wall, compile_time)) or not 0 <= compile_time <= wall:
            raise RuntimeError(f'Invalid compiler timing: wall={wall}, compile={compile_time}')
        return dict(seconds=wall-compile_time, wall_seconds=wall, compile_seconds=compile_time)
