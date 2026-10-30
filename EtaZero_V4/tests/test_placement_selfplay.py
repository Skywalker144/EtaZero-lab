"""Real native placement games and the source learner/controller boundary."""
import json
import os
import re
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from etazero.config import RunConfig
from etazero.run import run
from katago.train.metrics_pytorch import Metrics
from katago.train.load_model import load_model_state_dict
from test_placement_network import placement_model


def selfplay_config(path, game, max_moves):
    path.write_text(f'@include {os.path.relpath(ROOT / "configs/checks/source_go.cfg", path.parent)}\n'
                    f'game = {game}\nkomiMean = 0\nbSizes = 5\ndataBoardLen = 5\nmaxMovesPerGame = {max_moves}\n'
                    'subtreeValueBiasFactor = 0.4\navoidRepeatedPatternUtility = 0.03\n')
    return path


@pytest.mark.parametrize('game', ['freestyle', 'standard', 'renju', 'hex'])
@pytest.mark.parametrize('max_moves', [5, 100])
def test_native_selfplay_terminals_and_truncation(tmp_path, game, max_moves):
    cfg = selfplay_config(tmp_path / 'selfplay.cfg', game, max_moves)
    result = subprocess.run([str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
                             '-models-dir', str(tmp_path / 'models'), '-output-dir', str(tmp_path / 'data'),
                             '-max-games-total', '4'], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    files = list((tmp_path / 'data/random/tdata').glob('*.npz'))
    assert files
    arrays = {}
    for file in files:
        with np.load(file) as shard:
            for key in shard.files:
                arrays.setdefault(key, []).append(shard[key])
    arrays = {key: np.concatenate(values) for key, values in arrays.items()}
    target = arrays['globalTargetsNC']
    assert target.shape[1] == 80
    assert arrays['binaryInputNCHWPacked'].shape[1:] == (5, 4)
    assert arrays['globalInputNC'].shape[1] == 8
    assert arrays['policyTargetsNCMove'].shape[1:] == (2, 25)
    assert arrays['qValueTargetsNCMove'].shape[1:] == (2, 25)
    assert (target[:, [2, 6, 10, 14, 18]] == 0).all()
    for cols in ([0, 70, 1], [4, 71, 5], [8, 72, 9], [12, 73, 13]):
        np.testing.assert_allclose(target[:, cols].sum(1), 1, atol=2e-6)
    if max_moves == 5:
        assert (target[:, 52] == 1).all() and (target[:, 62] == 0).all()
        assert (target[:, 35] == 1).all() and (target[:, 27] == 0).all()
        assert (target[:, 24] == 0).all()  # Keep bootstrapped TD supervision.
        assert (target[:, 75:77] == 1).all()
        assert np.any((target[:, 0] > 0) & (target[:, 0] < 1))
    else:
        assert (target[:, 52] == 0).all() and (target[:, 62] == 1).all()
        assert (target[:, 35] == 0).all() and (target[:, 27] == 1).all()
        assert (target[:, 75:77] == 0).all()
        assert np.isin(target[:, [0, 70, 1]], [0, 1]).all()
    records = ''.join(p.read_text() for p in (tmp_path / 'data/random/sgfs').glob('*.sgfs'))
    assert f'GM[{11 if game == "hex" else 4}]' in records
    assert f'RU[{game}]' in records
    results = re.findall(r'RE\[([^]]+)\]', records)
    assert not results if max_moves == 5 else results and set(results) <= {'B+', 'W+', '0'}


def test_future_horizons_mask_only_unobserved_targets():
    model = placement_model()
    metrics = Metrics(1, model)
    pred = torch.zeros(3, 2, 7, 7, requires_grad=True)
    targets = torch.ones_like(pred)
    horizon = torch.tensor([[0., 0.], [1., 0.], [1., 1.]])
    loss = metrics.loss_futurepos_samplewise(pred, targets, torch.ones(3), torch.ones(3, 7, 7),
                                            torch.full((3,), 49.), torch.ones(3), horizon)
    torch.testing.assert_close(loss, torch.tensor([0., 1.75, 2.1875]))
    loss.sum().backward()
    assert torch.count_nonzero(pred.grad[0]) == 0
    assert torch.count_nonzero(pred.grad[1, 1]) == 0
    assert (pred.grad[1, 0] != 0).all() and (pred.grad[2] != 0).all()


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA build and GPU')
@pytest.mark.parametrize('game', ['renju', 'hex'])
def test_placement_model_consumed_in_second_round(tmp_path, game):
    cfg = selfplay_config(tmp_path / 'selfplay.cfg', game, 100)
    config = RunConfig(selfplay_config=str(cfg), model_kind='b2c16-fson-mish', game=game,
                       canvas=5, batch_size=8, games_per_round=4, samples_per_epoch=16,
                       shuffle_rows_per_file=128, shuffle_keep_rows=1024, shuffle_args=('-min-rows','1'),
                       train_args=(('-no-compile',) if game == 'renju' else ('-use-fp16',)) +
                                  ('-lookahead-k','6','-no-longterm-checkpoints')).validate()
    directory = tmp_path / 'run'
    first = run(directory, config, 1)
    second = run(directory, config, 2)
    assert second['completed_rounds'] == 2 and second['total_samples'] > first['total_samples'] > 0
    checkpoint = torch.load(directory / 'train/checkpoint.ckpt', map_location='cpu', weights_only=False)
    assert checkpoint['config']['version'] == 18
    weights = load_model_state_dict(checkpoint)
    assert weights['conv_spatial.weight'].shape[1] == 5
    assert not any('linear_pass' in name for name in weights)
    report = json.loads((directory / 'rounds/000002/result.json').read_text())
    assert report['input_model'] == 'round-000001'
    assert list((directory / 'selfplay/000002/round-000001/tdata').glob('*.npz'))
    assert run(directory, config, 2) == second


@pytest.mark.parametrize('game', ['renju', 'hex'])
def test_source_policy_init_side_and_reanalysis_on_placement(tmp_path, game):
    cfg = selfplay_config(tmp_path / 'selfplay.cfg', game, 100)
    with cfg.open('a') as stream:
        stream.write('numGameThreads = 1\nmaxVisits = 24\n'
                     'initGamesWithPolicy = true\npolicyInitAreaProp = 0.15\n'
                     'sidePositionProb = 1\ncheapSearchProb = 1\ncheapSearchVisits = 8\n'
                     'cheapSearchTargetWeight = 1\nuseReanalyze = true\nreanalyzeProp = 1\n'
                     'reanalyzePolicySurpriseWeight = 1\nreanalyzeValueSurpriseWeight = 1\n'
                     'reanalyzeSurpriseExponent = 1\nreanalyzeUseOutcomeTargets = false\n'
                     'earlyForkGameProb = 0.5\nforkGameProb = 0.5\n'
                     'normalAsymmetricPlayoutProb = 1\nmaxAsymmetricRatio = 2\n')
    result = subprocess.run([str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
                             '-models-dir', str(tmp_path / 'models'), '-output-dir', str(tmp_path / 'data'),
                             '-max-games-total', '8'], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    targets = []
    for path in (tmp_path / 'data/random/tdata').glob('*.npz'):
        with np.load(path) as shard:
            targets.append(shard['globalTargetsNC'])
            assert np.isfinite(shard['globalInputNC']).all()
    target = np.concatenate(targets)
    assert np.isfinite(target).all()
    assert np.any(target[:, 64] == 1)  # Actual post-game reanalysis.
    reanalyzed = target[target[:, 64] == 1]
    assert (reanalyzed[:, [27, 28, 33]] == 0).all()  # Source outcome-target switch.
    assert (reanalyzed[:, 35] == 0).all()  # Main result remains supervised.
    assert np.any(target[:, 62] == 0)  # Off-trajectory side positions were written too.
    assert (target[:, [2, 6, 10, 14, 18]] == 0).all()


def imported_start_config(tmp_path, game, kind):
    cfg = selfplay_config(tmp_path / 'selfplay.cfg', game, 100)
    source = tmp_path / 'positions'
    source.mkdir()
    if kind == 'sgf':
        (source / 'position.sgf').write_text(
            f'(;GM[{11 if game == "hex" else 4}]RU[{game}]SZ[5]'
            'AB[cc]AW[bc][cb][dc][cd]PL[W];W[aa];B[ee])')
        options = (f'startPosesFromSgfDir = {source}\nstartPosesProb = 1\n'
                   'startPosesLoadProb = 1\nstartPosesTurnWeightLambda = 0\n')
    else:
        sample = dict(game=game, xSize=5, ySize=5, board='...../..O../.OXO./..O../...../',
                      nextPla='W', moveLocs=['A5', 'E1'], movePlas=['W', 'B'],
                      initialTurnNumber=5, hintLoc='A1', weight=1, trainingWeight=0.375)
        (source / 'position.hintposes.txt').write_text(json.dumps(sample) + '\n')
        options = f'hintPosesDir = {source}\nhintPosesProb = 1\n'
    with cfg.open('a') as stream:
        stream.write(options)
    return cfg


@pytest.mark.parametrize('game', ['freestyle', 'standard', 'renju', 'hex'])
@pytest.mark.parametrize('kind', ['sgf', 'hint'])
def test_placement_imported_start_selfplay(tmp_path, game, kind):
    cfg = imported_start_config(tmp_path, game, kind)
    result = subprocess.run([str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
                             '-models-dir', str(tmp_path / 'models'), '-output-dir', str(tmp_path / 'data'),
                             '-max-games-total', '4'], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = []
    boards = []
    for file in (tmp_path / 'data/random/tdata').glob('*.npz'):
        with np.load(file) as shard:
            rows.append(shard['globalTargetsNC'])
            boards.append(np.unpackbits(shard['binaryInputNCHWPacked'], axis=-1)[..., :25])
    target = np.concatenate(rows)
    spatial = np.concatenate(boards)
    assert np.isin(target[:, 55], [4] if kind == 'sgf' else [5, 6]).all()
    assert (target[:, 56] == 5).all()
    assert ((spatial[:, 1] + spatial[:, 2]).sum(1) >= 5).all()
    assert ((spatial[:, 1, 12] + spatial[:, 2, 12]) == 1).all()  # Enclosed stone never captured.
    assert (target[:, [2, 6, 10, 14, 18]] == 0).all()
    assert (target[:, 52] == 0).all()
    if kind == 'hint':
        assert (target[target[:, 55] == 5, 53] == 2).all()
        assert (target[target[:, 55] == 6, 53] == 3).all()
        assert (target[:, 25] == 0.375).all()
        # trainingWeight is applied to row weight, not to the categorical result.
        assert np.isin(target[:, [0, 70, 1]], [0, 1]).all()
    records = ''.join(p.read_text() for p in (tmp_path / 'data/random/sgfs').glob('*.sgfs'))
    assert f'RU[{game}]' in records and 'KM[' not in records and 'HA[' not in records
    if game in {'renju', 'hex'}:
        assert 'AB[cc]' in records and 'PL[W]' in records


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA build and GPU')
@pytest.mark.parametrize('game,kind', [('renju', 'sgf'), ('hex', 'hint')])
def test_imported_start_training_and_next_model(tmp_path, game, kind):
    cfg = imported_start_config(tmp_path, game, kind)
    config = RunConfig(selfplay_config=str(cfg), model_kind='b2c16-fson-mish', game=game,
                       canvas=5, batch_size=8, games_per_round=4, samples_per_epoch=16,
                       shuffle_rows_per_file=128, shuffle_keep_rows=1024, shuffle_args=('-min-rows','1'),
                       train_args=('-no-compile','-use-fp16','-no-longterm-checkpoints')).validate()
    directory = tmp_path / 'run'
    state = run(directory, config, 2)
    assert state['completed_rounds'] == 2 and state['total_samples'] >= 32
    report = json.loads((directory / 'rounds/000002/result.json').read_text())
    assert report['input_model'] == 'round-000001'
    files = list((directory / 'selfplay/000002/round-000001/tdata').glob('*.npz'))
    assert files
    for file in files:
        with np.load(file) as shard:
            assert np.isin(shard['globalTargetsNC'][:, 55], [4] if kind == 'sgf' else [5, 6]).all()
            assert np.isfinite(shard['globalTargetsNC']).all()
            assert (shard['globalTargetsNC'][:, 56] == 5).all()
    checkpoint = torch.load(directory / 'train/checkpoint.ckpt', map_location='cpu', weights_only=False)
    assert checkpoint['config']['version'] == 18
    assert_terminal_sgf_gtp(tmp_path, directory, game)


def assert_terminal_sgf_gtp(tmp_path, directory, game):
    terminal = tmp_path / 'terminal.sgf'
    stones = '[aa][ab][ac][ad][ae]' if game == 'hex' else '[aa][ba][ca][da][ea]'
    terminal.write_text(f'(;RU[{game}]SZ[5]AB{stones}PL[W])')
    gtp_config = tmp_path / 'gtp.cfg'
    gtp_config.write_text(f'rules = {game}\nnumSearchThreads = 1\nmaxVisits = 4\n'
                          'cudaUseFP16 = true\nlogSearchInfo = false\nlogToStderr = false\n'
                          'logAllGTPCommunication = false\nponderingEnabled = false\n')
    result = subprocess.run([str(ROOT / 'build/katago'), 'gtp', '-config', str(gtp_config),
                             '-model', str(directory / 'models/round-000002/model.bin')],
                            input=f'10 loadsgf {terminal}\n11 final_score\n12 kata-raw-nn 0\nquit\n',
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '=10' in result.stdout and '=11 B+' in result.stdout, result.stdout
    assert '?12 No policy is defined at a terminal placement position' in result.stdout


@pytest.mark.parametrize('game', ['freestyle', 'standard', 'renju', 'hex'])
def test_source_sgf_sampler_and_position_viewer(tmp_path, game):
    imported_start_config(tmp_path, game, 'sgf')
    output = tmp_path / 'sampled'
    result = subprocess.run([str(ROOT / 'build/katago'), 'samplesgfs', '-sgfdir', str(tmp_path / 'positions'),
                             '-outdir', str(output), '-sample-prob', '1', '-turn-weight-lambda', '0',
                             '-max-komi', '0', '-flip-if-pass', '-training-weight', '0.375', '-for-testing'],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    files = list(output.glob('*.startposes.txt'))
    assert files, result.stdout + result.stderr
    samples = [json.loads(line) for path in files for line in path.read_text().splitlines()]
    assert len(samples) == 3
    for sample in samples:
        assert sample['game'] == game and sample['trainingWeight'] == 0.375
        assert sample['initialTurnNumber'] == 5
        assert sample['board'].count('X') + sample['board'].count('O') == 5
        if game in {'renju', 'hex'}:
            assert sample['nextPla'] == 'W'
    for path in files:
        viewed = subprocess.run([str(ROOT / 'build/katago'), 'viewstartposes',
                                 '-start-poses-file', str(path), '-check-legality'],
                                text=True, capture_output=True, timeout=30)
        assert viewed.returncode == 0, viewed.stdout + viewed.stderr
        assert 'Illegal' not in viewed.stdout and 'Mismatching' not in viewed.stdout
