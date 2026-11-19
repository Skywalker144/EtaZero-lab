"""Generate current-format numerical test data with the native random evaluator.

This is a test fixture, not a training run or a strength measurement. The native
selfplay sampler chooses its seeds; regeneration need not produce identical rows.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        models = temp / 'models'
        models.mkdir()
        cfg = temp / 'selfplay.cfg'
        cfg.write_text(
            f'@include {os.path.relpath(root / "configs/checks/source_go.cfg", temp)}\n'
            'dataBoardLen = 19\nbSizes = 19\nmaxMovesPerGame = 64\nmaxVisits = 4\n'
            'maxRowsPerTrainFile = 4096\nnumGameThreads = 2\n'
        )
        subprocess.run([
            str(root / 'build/katago'), 'selfplay', '-config', str(cfg),
            '-models-dir', str(models), '-output-dir', str(temp / 'selfplay'), '-max-games-total', '32',
        ], check=True)
        arrays = {}
        for file in sorted((temp / 'selfplay/random/tdata').glob('*.npz')):
            with np.load(file) as data:
                for key in data.files:
                    arrays.setdefault(key, []).append(data[key])
        data = {key: np.concatenate(parts)[:1024] for key, parts in arrays.items()}
        if (data['globalTargetsNC'].shape != (1024, 80)
                or data['valueTargetsNCHW'].shape != (1024, 2, 19, 19)
                or data['qValueTargetsNCMove'].shape != (1024, 2, 362)
                or 'scoreDistrN' in data):
            raise RuntimeError('Native writer did not generate 1024 rows with the current target layout')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **data)


if __name__ == '__main__':
    main()
