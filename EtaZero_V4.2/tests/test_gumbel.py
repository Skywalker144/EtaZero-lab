"""Gumbel root through the native KataGo search, writer, and outer learner."""
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from etazero.config import RunConfig
from etazero.run import run
from test_placement_selfplay import selfplay_config


def gumbel_config(path, game, threads=1, interior="puct", budget=4):
    selfplay_config(path, game, 100)
    with path.open('a') as out:
        out.write(f'rootSearchAlgorithm = gumbel\nnonrootSearchAlgorithm = {interior}\n'
                  f'numSearchThreads = {threads}\nuseGraphSearch = false\nrootSymmetryPruning = false\n'
                  f'maxVisits = {budget}\ngumbelMaxConsideredActions = 3\nrootNoiseEnabled = true\n')
    return path


@pytest.mark.parametrize('game', ['go', 'renju', 'hex'])
@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('interior', ['puct', 'gumbel'])
def test_native_gumbel_dense_targets_and_budget(tmp_path, game, threads, interior):
    budget = 12
    cfg = gumbel_config(tmp_path / 'selfplay.cfg', game, threads, interior, budget)
    result = subprocess.run([str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
                             '-models-dir', str(tmp_path / 'models'), '-output-dir', str(tmp_path / 'data'),
                             '-max-games-total', '4'], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    shards = list((tmp_path / 'data/random/tdata').glob('*.npz'))
    assert shards
    dense_unvisited = 0
    for path in shards:
        with np.load(path) as shard:
            policy = shard['policyTargetsNCMove'][:, 0]
            visits = shard['qValueTargetsNCMove'][:, 1]
            targets = shard['globalTargetsNC']
            assert (policy.max(1) == 30000).all()
            assert (targets[:, 60] == budget).all()
            assert (visits.sum(1) == budget - 1).all()  # Initial root evaluation consumes one visit.
            dense_unvisited += np.count_nonzero((visits == 0) & (policy > 0))
            assert np.isfinite(targets).all()
    assert dense_unvisited > 0


@pytest.mark.parametrize('root,interior,message', [
    ('puct', 'gumbel', 'requires rootSearchAlgorithm=gumbel'),
    ('gumbel', 'unknown', 'Unknown nonrootSearchAlgorithm'),
])
def test_invalid_search_combinations_are_rejected(tmp_path, root, interior, message):
    cfg = gumbel_config(tmp_path / 'selfplay.cfg', 'hex', interior=interior)
    cfg.write_text(cfg.read_text().replace('rootSearchAlgorithm = gumbel\n',
                                          f'rootSearchAlgorithm = {root}\n', 1))
    result = subprocess.run([str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
                             '-models-dir', str(tmp_path / 'models'), '-output-dir', str(tmp_path / 'data'),
                             '-max-games-total', '1'], text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert message in result.stdout + result.stderr


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires CUDA build and GPU')
def test_gumbel_two_round_cuda_training(tmp_path):
    budget = 12
    cfg = gumbel_config(tmp_path / 'selfplay.cfg', 'renju', 3, 'gumbel', budget)
    config = RunConfig(selfplay_config=str(cfg), model_kind='b2c16-fson-mish', game='renju',
                       algorithm='gumbel_alphazero', canvas=5, batch_size=8, bootstrap_games=4,
                       train_steps=2, shuffle_rows_per_file=8, shuffle_keep_rows=1024,
                       shuffle_args=('-min-rows','1'),
                       train_args=('-no-compile','-use-fp16','-no-longterm-checkpoints')).validate()
    directory = tmp_path / 'run'
    state = run(directory, config, 2)
    assert state['completed_rounds'] == 2 and state['total_samples'] >= 32
    report = json.loads((directory / 'rounds/000002/result.json').read_text())
    assert report['input_model'].startswith('round-000001-s')
    checkpoint = torch.load(directory / 'train/checkpoint.ckpt', map_location='cpu', weights_only=False)
    assert checkpoint['config']['version'] == 18
    paths = list((directory / 'selfplay/000002' / report['input_model'] / 'tdata').glob('*.npz'))
    assert paths
    for path in paths:
        with np.load(path) as shard:
            assert (shard['policyTargetsNCMove'][:, 0].max(1) == 30000).all()
            assert (shard['qValueTargetsNCMove'][:, 1].sum(1) == budget - 1).all()
