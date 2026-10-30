"""Shared initial weights made by the same source learner used for training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from . import ROOT
from .run import learner_command, save_json


def initialization_key(config):
    # Search, optimizer, unroll length and budgets do not change network weights.
    # Keep the game/canvas explicit even where architectures happen to coincide.
    network = config.muzero_network if config.is_muzero else config.model_kind
    identity = dict(game=config.game, canvas=config.canvas, muzero=config.is_muzero, network=network)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def prepare_shared_initializations(work, arms, *, lock_fd=None):
    """Return name -> immutable checkpoint for new arms with matching networks.

    Each arm is a dict with name, run_dir and a validated RunConfig. Existing
    runs own their initialization and do not require the shared origin again.
    Failed temporary directories retain native logs for diagnosis.
    """
    work = Path(work).resolve()
    result = {}
    for arm in arms:
        if (Path(arm['run_dir']) / 'state.json').is_file():
            continue
        config = arm['config'].validate()
        key = initialization_key(config)
        destination = work / 'initializations' / key
        if not destination.is_dir():
            temporary = destination.with_name(key + '.tmp.' + uuid.uuid4().hex)
            temporary.mkdir(parents=True)
            save_json(temporary / 'config.json', config.serializable())
            if config.is_muzero:
                save_json(temporary / 'muzero_network.json', config.muzero_network)
            # Initializing on CPU consumes no arm's GPU/RNG stream. No forward,
            # warmup, file consumption or optimizer update occurs in the learner.
            command = [*learner_command(temporary, config), '-initialize-only', '-no-compile']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            with (temporary / 'initialize.log').open('w') as log:
                completed = subprocess.run(list(map(str, command)), cwd=ROOT / 'python', env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=() if lock_fd is None else (lock_fd,))
            if completed.returncode:
                raise RuntimeError(f'Shared initialization failed; see {temporary / "initialize.log"}')
            if not (temporary / 'train/checkpoint.ckpt').is_file():
                raise RuntimeError(f'Source learner did not save shared initialization: {temporary}')
            temporary.rename(destination)
        result[arm['name']] = destination / 'train/checkpoint.ckpt'
    return result
