"""Execute the unchanged source learner and record its own compiler wall spans.

Single device only, as required by the outer controller. No warmup forwards,
extra optimizer steps, RNG resets, or device synchronizations are introduced.
"""
import math
from pathlib import Path
import runpy
import sys
import time
import warnings

from .run import save_json
from katago.train.stopping import defer_signals


def compilation_seconds():
    from torch._dynamo.utils import calculate_time_spent
    # These are non-overlapping outer forward and lazy-backward compilation
    # intervals. Summing backend/Triton phases would double-count nested work.
    return calculate_time_spent()['total_wall_time']


def main():
    receipt = Path(sys.argv[1])
    sys.argv = sys.argv[2:]
    started = time.monotonic()
    try:
        initial = compilation_seconds()
    except Exception as error:
        warnings.warn(f'Compiler timing unavailable: {error}', RuntimeWarning)
        initial = None
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        try:
            compiled = None if initial is None else compilation_seconds() - initial
            wall = time.monotonic() - started
            valid = compiled is not None and math.isfinite(compiled) and 0 <= compiled <= wall
            import torch
            save_json(receipt, dict(wall_seconds=wall, compile_seconds=compiled if valid else None,
                                   timing_valid=valid, torch_version=torch.__version__))
        except Exception as error:
            # A receipt is timing evidence, not the learner's training commit.
            # The controller marks missing timing invalid without undoing work.
            warnings.warn(f'Could not record learner compiler timing: {error}', RuntimeWarning)


if __name__ == '__main__':
    with defer_signals():
        main()
