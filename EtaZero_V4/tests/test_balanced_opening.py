"""Native opening prefixes through real CUDA selfplay, match and learner."""
from dataclasses import asdict
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from etazero import ROOT
from etazero.evaluation import run_match
from katago.train.load_model import load_checkpoint
from test_match import export_models, match_config
from test_muzero_data import shuffle_source
from test_muzero_learner import run_cli
from test_muzero_model import make_model

gpu = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA')


def configure(path, game, **settings):
    path.write_text(f'@include {os.path.relpath(ROOT / "configs/checks/source_go.cfg", path.parent)}\n'
                    f'game = {game}\nkomiMean = 0\nbSizes = 5\ndataBoardLen = 5\nmaxMovesPerGame = 4\n')
    values = dict(balanceOpeningProb=1, maxVisits=8, numSearchThreads=2,
                  useGraphSearch='false', cudaUseFP16='true',
                  nnRandSeed='balanced-opening', initGamesWithPolicy='false', **settings)
    if game == 'hex':
        values['makeOpeningFairRate'] = 1
    with path.open('a') as stream:
        stream.write(''.join(f'{key} = {value}\n' for key, value in values.items()))
    return path


def selfplay(tmp_path, cfg, models, games=4):
    result = subprocess.run([str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
        '-models-dir', str(models), '-output-dir', str(tmp_path / 'data'), '-max-games-total', str(games)],
        capture_output=True, text=True, timeout=120)
    (tmp_path / 'native.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return list((tmp_path / 'data').glob('*/tdata/*.npz'))


@pytest.mark.parametrize('game', ['renju', 'hex'])
def test_random_cold_start_skips_model_balancing(tmp_path, game):
    cfg = configure(tmp_path / 'selfplay.cfg', game)
    files = selfplay(tmp_path, cfg, tmp_path / 'models')
    assert files
    for file in files:
        with np.load(file) as data:
            assert (data['globalTargetsNC'][:, 53] == 0).all()
    assert 'Balanced opening:' not in (tmp_path / 'native.log').read_text()


@gpu
@pytest.mark.parametrize('game', ['renju', 'hex'])
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_balanced_prefix_through_native_selfplay_and_training(tmp_path, game, muzero, gumbel):
    ordinary, bundle = export_models(tmp_path, game)
    models = tmp_path / 'models'; models.mkdir()
    if muzero:
        shutil.copytree(bundle, models / 'published')
    else:
        (models / 'published').mkdir()
        shutil.copyfile(ordinary, models / 'published/model.bin')
    cfg = configure(tmp_path / 'selfplay.cfg', game, useMuZero=str(muzero).lower(),
                    rootSearchAlgorithm='gumbel' if gumbel else 'puct', gumbelMaxConsideredActions=4,
                    subtreeValueBiasFactor=0, avoidRepeatedPatternUtility=0)
    files = selfplay(tmp_path, cfg, models)
    assert files and 'Balanced opening:' in (tmp_path / 'native.log').read_text()
    for file in files:
        with np.load(file) as data:
            target = data['globalTargetsNC']
            assert (target[:, 53] >= 1).all()
            assert (target[:, 51] >= target[:, 53]).all()
            if game == 'hex':
                assert (target[:, 53] == 1).all()
            if muzero:
                trajectory = data['muZeroGlobalTargetsNC']
                for start in data['muZeroGameOffsets'][:-1]:
                    assert trajectory[start, 51] == trajectory[start, 53] >= 1
                for row, (identity, turn) in enumerate(data['muZeroRootNC']):
                    assert turn >= 0
                    assert target[row, 51] - target[row, 53] == turn
    # Ordinary and MuZero use the unchanged source shuffle and formal learner.
    shuffled = shuffle_source(files[0].parent, tmp_path, steps=3 if muzero else 0, rows_per_file=128, files_per_bucket=1)
    data_dir = tmp_path / 'learner-data'; data_dir.mkdir()
    shutil.move(shuffled, data_dir / 'train')
    shutil.move(shuffled.with_suffix('.json'), data_dir / 'train.json')
    (data_dir / 'val').mkdir()
    options = ['-model-kind', 'b2c16-fson-mish', '-input-features', 'placement']
    if muzero:
        network = tmp_path / 'network.json'
        network.write_text(json.dumps(asdict(make_model(game).network_config)))
        options = ['-muzero-config', network]
    command = [ROOT / 'python/train.py', '-traindir', tmp_path / 'train', '-datadir', data_dir,
               *options, '-pos-len', '5', '-batch-size', '4', '-samples-per-epoch', '8',
               '-max-epochs-this-instance', '1', '-no-compile', '-use-adamw', '-use-fp16',
               '-lookahead-k', '2', '-no-export', '-no-longterm-checkpoints', '-quit-if-no-data']
    run_cli(command, tmp_path / 'learner.log', dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    checkpoint = load_checkpoint(tmp_path / 'train/checkpoint.ckpt')
    assert checkpoint['train_state']['global_step_samples'] > 0
    assert checkpoint['learner_runtime']['scaler']


@gpu
@pytest.mark.parametrize('game', ['renju', 'hex'])
def test_mixed_match_uses_native_opening_and_source_elo(tmp_path, game):
    ordinary, bundle = export_models(tmp_path, game)
    cfg = match_config(tmp_path / 'match.cfg', [(ordinary, False), (bundle, True)], game, games=4)
    with cfg.open('a') as stream:
        stream.write('balanceOpeningProb = 1\n')
        if game == 'hex': stream.write('makeOpeningFairRate = 1\n')
    result = run_match(tmp_path / 'match', cfg)
    assert result['games'] == 4
    logs = '\n'.join(p.read_text() for p in (tmp_path / 'match').rglob('*.log'))
    assert logs.count('Balanced opening:') >= 4
