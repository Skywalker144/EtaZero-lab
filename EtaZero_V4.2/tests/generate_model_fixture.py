"""Regenerate the small four-result v17 fixture used by the native model parser tests."""
import gzip
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from katago.train import modelconfigs
from katago.train.model_pytorch import Model


def main():
    root = Path(__file__).resolve().parents[1]
    torch.manual_seed(4)
    config = dict(modelconfigs.config_of_name['b2c16-fson-mish'], version=17, predict_q_values=False)
    model = Model(config, pos_len=7)
    model.initialize()
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        checkpoint = temp / 'model.ckpt'
        torch.save({'config': config, 'model': model.state_dict()}, checkpoint)
        subprocess.run([sys.executable, str(root / 'python/export_model_pytorch.py'), '-checkpoint', str(checkpoint),
                        '-export-dir', str(temp), '-model-name', 'four_results', '-filename-prefix', 'model'], check=True)
        (root / 'cpp/tests/models/four-results.bin.gz').write_bytes(gzip.compress((temp / 'model.bin').read_bytes(), mtime=0))


if __name__ == '__main__':
    main()
