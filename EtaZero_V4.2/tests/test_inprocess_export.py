"""Source CLI and controller exports agree, without leaking process state."""
import logging
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.optim.swa_utils import AveragedModel

from etazero import ROOT
from etazero.run import execute
from katago.train.model_pytorch import Model
from katago.train.muzero_learner import checkpoint_config
from test_muzero_model import component_config, make_model, limited_cpu_threads


def make_checkpoint(path, muzero):
    if muzero:
        model = make_model('renju')
    else:
        model = Model(component_config('renju'), pos_len=5)
        model.initialize()
    swa = AveragedModel(model)
    with torch.no_grad():
        next(swa.parameters()).add_(0.1)
    swa.n_averaged.fill_(1)
    torch.save(dict(config=checkpoint_config(model), model=model.state_dict(),
                    swa_model=swa.state_dict(), train_state={'global_step_samples': 32}), path)


def command(checkpoint, output, muzero, swa):
    return [sys.executable, ROOT / 'python' / ('export_muzero.py' if muzero else 'export_model_pytorch.py'),
            '-checkpoint', checkpoint, '-export-dir', output, '-model-name', 'export-test',
            *(['-pos-len', '5'] if muzero else ['-filename-prefix', 'model']),
            *(['-use-swa'] if swa else [])]


@pytest.mark.parametrize('muzero', [False, True])
@pytest.mark.parametrize('swa', [False, True])
def test_export_matches_cli_and_restores_caller(tmp_path, muzero, swa):
    checkpoint = tmp_path / 'model.ckpt'
    make_checkpoint(checkpoint, muzero)
    checkpoint_bytes = checkpoint.read_bytes()
    reference = tmp_path / 'cli'
    result = subprocess.run(list(map(str, command(checkpoint, reference, muzero, swa))),
                            capture_output=True, text=True,
                            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    payloads = {p.name: p.read_bytes() for p in reference.iterdir() if p.suffix in {'.bin', '.json'}}
    cpu_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    print_options = np.get_printoptions()
    handlers, level, argv = logging.root.handlers[:], logging.root.level, sys.argv
    stdout, stderr = sys.stdout, sys.stderr
    cuda_initialized = torch.cuda.is_initialized()
    for i in range(2):
        output = tmp_path / f'inprocess{i}'
        execute(command(checkpoint, output, muzero, swa), tmp_path / f'export{i}.log')
        assert {p.name: p.read_bytes() for p in output.iterdir() if p.suffix in {'.bin', '.json'}} == payloads
        assert logging.root.handlers == handlers and logging.root.level == level
        assert sys.argv is argv and sys.stdout is stdout and sys.stderr is stderr
        assert np.get_printoptions() == print_options
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        after_numpy = np.random.get_state()
        np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
        assert after_numpy[0] == numpy_rng[0] and after_numpy[2:] == numpy_rng[2:]
        assert random.getstate() == python_rng
        assert torch.cuda.is_initialized() == cuda_initialized
    assert checkpoint.read_bytes() == checkpoint_bytes


def test_export_failure_logs_traceback_and_restores_caller(tmp_path, monkeypatch):
    import export_model_pytorch
    checkpoint = tmp_path / 'model.ckpt'
    make_checkpoint(checkpoint, False)
    cpu_rng = torch.get_rng_state()
    handlers, level, argv = logging.root.handlers[:], logging.root.level, sys.argv

    def fail(*args, **kwargs):
        raise ValueError('injected after model initialization')

    monkeypatch.setattr(export_model_pytorch, 'prepare_model_for_export', fail)
    log = tmp_path / 'failed.log'
    with pytest.raises(RuntimeError, match='Source export failed'):
        execute(command(checkpoint, tmp_path / 'partial', False, False), log)
    assert 'Traceback' in log.read_text() and 'injected after model initialization' in log.read_text()
    assert logging.root.handlers == handlers and logging.root.level == level and sys.argv is argv
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert not (tmp_path / 'partial/model.bin').exists()
