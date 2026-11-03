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
from export_muzero import export_components
from katago.train.load_model import load_checkpoint
from test_match import export_models, match_config
from test_muzero_data import shuffle_source
from test_muzero_learner import run_cli
from test_muzero_model import make_model

gpu = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA')


def configure(path, game, **settings):
    path.write_text(f'@include {os.path.relpath(ROOT / "configs/checks/source_go.cfg", path.parent)}\n'
                    f'game = {game}\nkomiMean = 0\nmaxMovesPerGame = 4\n')
    values = dict(bSizes=5, dataBoardLen=5, balanceOpeningProb=1, maxVisits=8, numSearchThreads=2,
                  useGraphSearch='false', cudaUseFP16='true',
                  nnRandSeed='balanced-opening', initGamesWithPolicy='false')
    values.update(settings)
    if game == 'hex':
        values['makeOpeningFairRate'] = 1
        values.setdefault('specialOpeningProb', 0)
        values.setdefault('completelyRandomOpeningProb', 0)
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
def test_random_cold_start_uses_katagomo_balancing(tmp_path, game):
    cfg = configure(tmp_path / 'selfplay.cfg', game)
    files = selfplay(tmp_path, cfg, tmp_path / 'models')
    assert files
    for file in files:
        with np.load(file) as data:
            prefix = data['globalTargetsNC'][:, 53]
            assert (prefix >= 1).all()
    assert 'Balanced opening:' in (tmp_path / 'native.log').read_text()


@pytest.mark.parametrize('game', ['freestyle', 'standard', 'renju'])
def test_gomoku_policy_uses_move_mean_instead_of_go_area_settings(tmp_path, game):
    cfg = configure(tmp_path / 'selfplay.cfg', game, balanceOpeningProb=0,
                    initGamesWithPolicy='true', policyInitAvgMoveNum=0,
                    policyInitAreaProp=1, policyInitGammaShape=10)
    files = selfplay(tmp_path, cfg, tmp_path / 'models')
    assert files
    for file in files:
        with np.load(file) as data:
            assert (data['globalTargetsNC'][:, 53] == 0).all()


@gpu
@pytest.mark.parametrize('game,opening', [
    ('renju', 'balanced'), ('hex', 'balanced'), ('hex', 'policy'),
    ('hex', 'special'), ('hex', 'random-fill'),
])
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_balanced_prefix_through_native_selfplay_and_training(tmp_path, game, opening, muzero, gumbel):
    # The 19x19 setup path includes the value-filtered infinite template.
    size = 19 if opening == 'special' else 5
    ordinary, bundle = export_models(tmp_path, game)
    if muzero and size != 5:
        bundle = tmp_path / 'muzero-large'
        export_components(make_model(game, pos_len=size), bundle, 'muzero-large')
    models = tmp_path / 'models'; models.mkdir()
    if muzero:
        shutil.copytree(bundle, models / 'published')
    else:
        (models / 'published').mkdir()
        shutil.copyfile(ordinary, models / 'published/model.bin')
    opening_settings = {}
    setup = opening in ('special', 'random-fill')
    if setup:
        opening_settings.update(balanceOpeningProb=0,
            specialOpeningProb=int(opening == 'special'),
            completelyRandomOpeningProb=int(opening == 'random-fill'),
            completelyRandomOpeningFillRateAvg=0.08)
    cfg = configure(tmp_path / 'selfplay.cfg', game, useMuZero=str(muzero).lower(),
                    bSizes=size, dataBoardLen=size,
                    rootSearchAlgorithm='gumbel' if gumbel else 'puct', gumbelMaxConsideredActions=4,
                    subtreeValueBiasFactor=0, avoidRepeatedPatternUtility=0,
                    initGamesWithPolicy=str(game != 'hex' or opening == 'policy').lower(),
                    policyInitAvgMoveNum=6, **opening_settings)
    files = selfplay(tmp_path, cfg, models, games=8)
    expected_log = f'Hex opening: {opening}' if setup else 'Balanced opening:'
    assert files and expected_log in (tmp_path / 'native.log').read_text()
    if setup:
        sgfs = '\n'.join(p.read_text() for p in (tmp_path / 'data').rglob('*.sgfs'))
        assert 'AB[' in sgfs or 'AW[' in sgfs
    prefixes = []
    for file in files:
        with np.load(file) as data:
            target = data['globalTargetsNC']
            prefixes.extend(target[:, 53])
            assert (target[:, 53] >= (0 if setup else 1)).all()
            assert (target[:, 51] >= target[:, 53]).all()
            if setup:
                assert (target[:, 53] == 0).all()
            elif game == 'hex' and opening == 'balanced':
                assert (target[:, 53] == 1).all()
            if muzero:
                trajectory = data['muZeroGlobalTargetsNC']
                for start in data['muZeroGameOffsets'][:-1]:
                    assert trajectory[start, 51] == trajectory[start, 53] >= (0 if setup else 1)
                for row, (identity, turn) in enumerate(data['muZeroRootNC']):
                    assert turn >= 0
                    assert target[row, 51] - target[row, 53] == turn
    if opening == 'policy':
        assert max(prefixes) > 1
    # Ordinary and MuZero use the unchanged source shuffle and formal learner.
    shuffled = shuffle_source(files[0].parent, tmp_path, steps=3 if muzero else 0, rows_per_file=128, files_per_bucket=1)
    data_dir = tmp_path / 'learner-data'; data_dir.mkdir()
    shutil.move(shuffled, data_dir / 'train')
    shutil.move(shuffled.with_suffix('.json'), data_dir / 'train.json')
    (data_dir / 'val').mkdir()
    options = ['-model-kind', 'b2c16-fson-mish', '-input-features', 'placement']
    if muzero:
        network = tmp_path / 'network.json'
        network.write_text(json.dumps(asdict(make_model(game, pos_len=size).network_config)))
        options = ['-muzero-config', network]
    command = [ROOT / 'python/train.py', '-traindir', tmp_path / 'train', '-datadir', data_dir,
               *options, '-pos-len', str(size), '-batch-size', '4', '-samples-per-epoch', '8',
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
    result = run_match(tmp_path / 'match', cfg)
    assert result['games'] == 4
    logs = '\n'.join(p.read_text() for p in (tmp_path / 'match').rglob('*.log'))
    assert logs.count('Balanced opening:') >= 4
