"""Public selfplay and the outer controller consume the native joint-action model."""
from dataclasses import asdict, replace
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from etazero.config import RunConfig, load_config
from etazero.run import learner_command, run
from katago.train.load_model import load_checkpoint
from test_config_profiles import write_profile, shell


def selfplay_config(path, canvas=10, **options):
    # Test-owned settings for the currently connected Xiangqi mechanisms.
    values = dict(game='xiangqi', bSizesXY='9-10',
                  bSizeRelProbs=1, komiMean=0, dataBoardLen=canvas, maxMovesPerGame=12,
                  subtreeValueBiasFactor=0, avoidRepeatedPatternUtility=0,
                  numGameThreads=2, numSearchThreads=2, maxVisits=16,
                  rootNumSymmetriesToSample=1, rootSymmetryPruning='false',
                  useGraphSearch='true', rootSearchAlgorithm='puct', nonrootSearchAlgorithm='puct',
                  cudaUseFP16='true', cudaUseNHWC='true', logGamesEvery=1,
                  logSearchInfo='false', logMoves='false', logToStdout='true',
                  maxDataQueueSize=16, maxRowsPerTrainFile=128, firstFileRandMinProp=1,
                  switchNetsMidGame='false', initGamesWithPolicy='false', sidePositionProb=0,
                  earlyForkGameProb=0, earlyForkGameExpectedMoveProp=.025,
                  forkGameProb=0, forkGameMinChoices=3, earlyForkGameMaxChoices=12, forkGameMaxChoices=36,
                  cheapSearchProb=0, cheapSearchVisits=4, cheapSearchTargetWeight=1,
                  reduceVisits='false', reduceVisitsThreshold=.9, reduceVisitsThresholdLookback=3,
                  reducedVisitsMin=4, reducedVisitsWeight=.1,
                  policySurpriseDataWeight=.5, valueSurpriseDataWeight=.1,
                  handicapAsymmetricPlayoutProb=0, normalAsymmetricPlayoutProb=0, maxAsymmetricRatio=2,
                  nnMaxBatchSize=4, nnCacheSizePowerOfTwo=10, nnMutexPoolSizePowerOfTwo=8)
    values.update(options)
    path.write_text(''.join(f'{key} = {value}\n' for key, value in values.items()))
    return path


def native_selfplay(tmp_path, cfg):
    return subprocess.run([str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
        '-models-dir', str(tmp_path / 'models'), '-output-dir', str(tmp_path / 'output'),
        '-max-games-total', '4', '-result-file', str(tmp_path / 'result.json')],
        capture_output=True, text=True, timeout=60)


def verify_native_output(output, canvas):
    files = list(output.glob('*/tdata/*.npz'))
    assert files
    for file in files:
        with np.load(file) as shard:
            assert shard['binaryInputNCHWPacked'].shape[1:] == (113, (canvas * canvas + 7) // 8)
            assert shard['globalInputNC'].shape[1] == 8
            for name in ('policyTargetsNCMove', 'qValueTargetsNCMove'):
                assert shard[name].shape[1:] == (2, 2086)
            targets = shard['globalTargetsNC']
            assert np.isfinite(targets).all()
            assert not np.any(targets[:, [2, 6, 10, 14, 18]])
            for columns in ([0, 70, 1], [4, 71, 5], [8, 72, 9], [12, 73, 13]):
                np.testing.assert_allclose(targets[:, columns].sum(1), 1, atol=2e-6)
    sgfs = ''.join(file.read_text() for file in output.glob('*/sgfs/*.sgfs'))
    assert 'GM[7]' in sgfs and 'SZ[9:10]' in sgfs and 'XQFEN[' in sgfs
    assert 'KM[' not in sgfs and 'HA[' not in sgfs


@pytest.mark.parametrize('canvas', [10, 12])
@pytest.mark.parametrize('gumbel,nonroot', [(False, False), (True, False), (True, True)])
def test_native_cold_start_writes_joint_data_and_full_sgf(tmp_path, canvas, gumbel, nonroot):
    cfg = selfplay_config(tmp_path / 'selfplay.cfg', canvas, rootSearchAlgorithm='gumbel' if gumbel else 'puct',
                          nonrootSearchAlgorithm='gumbel' if nonroot else 'puct', useGraphSearch=str(not gumbel).lower(),
                          subtreeValueBiasFactor=.4, avoidRepeatedPatternUtility=.03)
    result = native_selfplay(tmp_path, cfg)
    assert result.returncode == 0, result.stdout + result.stderr
    verify_native_output(tmp_path / 'output', canvas)
    assert json.loads((tmp_path / 'result.json').read_text()) == {'games_requested': 4, 'completed': True}


@pytest.mark.parametrize('options,error', [
    ({'inputsVersion': 8}, 'inputsVersion does not match'),
    ({'dataBoardLen': 9}, 'dataBoardLen >= 10'),
])
def test_native_layout_fails_before_model_polling(tmp_path, options, error):
    result = native_selfplay(tmp_path, selfplay_config(tmp_path / 'selfplay.cfg', **options))
    assert result.returncode != 0 and error in result.stdout + result.stderr
    assert not list(tmp_path.rglob('*.npz'))


def test_native_rejects_a_model_from_another_game_family(tmp_path):
    from export_model_pytorch import write_model_binary
    from test_placement_network import placement_model
    model_dir = tmp_path / 'models' / 'placement'
    model_dir.mkdir(parents=True)
    with (model_dir / 'model.bin').open('wb') as out:
        write_model_binary(placement_model().eval(), out, 'placement')
    result = native_selfplay(tmp_path, selfplay_config(tmp_path / 'selfplay.cfg'))
    assert result.returncode != 0 and 'other model families cannot play Xiangqi' in result.stdout + result.stderr
    assert not list(tmp_path.rglob('*.npz'))


@pytest.mark.parametrize('kind', ['sgf', 'hint'])
def test_native_imported_full_state_reaches_real_terminal(tmp_path, kind):
    # Every legal next move reaches the rule's 120-ply no-progress draw.
    fen = '3aka3/9/9/9/4P4/9/9/9/9/R3K4 w - - 119 60'
    positions = tmp_path / 'positions'
    positions.mkdir()
    if kind == 'sgf':
        (positions / 'start.sgf').write_text(f'(;GM[7]SZ[9:10]XQFEN[{fen}]XQTURN[73])')
        options = dict(startPosesFromSgfDir=positions, startPosesProb=1,
                       startPosesLoadProb=1, startPosesTurnWeightLambda=0)
    else:
        sample = dict(game='xiangqi', xSize=9, ySize=10, xiangqiFen=fen, xiangqiHistory=[],
                      nextPla='B', moveLocs=[], movePlas=[], initialTurnNumber=73,
                      hintLoc='null', weight=1, trainingWeight=.375)
        (positions / 'start.hintposes.txt').write_text(json.dumps(sample) + '\n')
        options = dict(hintPosesDir=positions, hintPosesProb=1)
    cfg = selfplay_config(tmp_path / 'selfplay.cfg', **options)
    result = native_selfplay(tmp_path, cfg)
    assert result.returncode == 0, result.stdout + result.stderr
    verify_native_output(tmp_path / 'output', 10)
    for file in (tmp_path / 'output').glob('*/tdata/*.npz'):
        with np.load(file) as shard:
            target = shard['globalTargetsNC']
            np.testing.assert_array_equal(target[:, [0, 1, 2]], 0)
            np.testing.assert_array_equal(target[:, 70], 1)
            assert np.all(target[:, 35] == 0) and np.all(target[:, 27] == 1)
            assert np.all(target[:, 56] == 73)
            if kind == 'hint':
                assert np.all(target[:, 25] == .375)
    records = ''.join(file.read_text() for file in (tmp_path / 'output').glob('*/sgfs/*.sgfs'))
    assert records.count('RE[0]') == 4 and 'XQTURN[73]' in records


def controller_config(tmp_path, **changes):
    cfg = selfplay_config(tmp_path / 'selfplay.cfg')
    return replace(RunConfig(selfplay_config=str(cfg), game='xiangqi', model_kind='b2c16-fson-mish',
        canvas=10, batch_size=4, bootstrap_games=4, train_steps=2,
        shuffle_rows_per_file=8, shuffle_keep_rows=1024, min_rows=1,
        train_args=('-no-compile', '-use-fp16', '-lookahead-k', '2', '-no-longterm-checkpoints'),
        auto_plot=False), **changes)


def test_outer_layout_and_model_family_boundaries(tmp_path):
    config = controller_config(tmp_path).validate()
    argv = learner_command(tmp_path, config)
    assert argv[argv.index('-input-features') + 1] == 'xiangqi'
    for changes, message in [
        ({'canvas': 9}, 'canvas >= 10'),
        ({'algorithm': 'muzero'}, 'MuZero uses muzero_network'),
        ({'algorithm': 'gumbel_muzero', 'model_kind': None}, 'MuZero requires muzero_network'),
    ]:
        with pytest.raises(ValueError, match=message):
            replace(config, **changes).validate()


@pytest.mark.parametrize('algorithm', ['alphazero', 'muzero', 'gumbel_muzero'])
def test_xiangqi_profile_and_script_entrypoints(tmp_path, algorithm):
    from test_muzero_model import make_model
    config = controller_config(tmp_path, algorithm=algorithm)
    if config.is_muzero:
        config = replace(config, model_kind=None, muzero_network=asdict(make_model('xiangqi').network_config),
                         muzero_unroll_steps=3)
    config.validate()
    profile = tmp_path / 'profile'
    write_profile(profile, config)
    assert load_config(profile).serializable() == config.serializable()
    result = shell('run.sh', '--config-dir', profile, '--run-dir', tmp_path / 'run', '--dry-run')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'xiangqi' in result.stdout and not (tmp_path / 'run').exists()
    (tmp_path / 'exp.cfg').write_text('[experiment]\niterations = 2\nshared_init = true\n[arms]\nxiangqi = profile\n')
    result = shell('autoexp.sh', '--config-dir', tmp_path, '--work-dir', tmp_path / 'experiment', '--dry-run')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'xiangqi' in result.stdout and not (tmp_path / 'experiment').exists()


@pytest.mark.parametrize('muzero', [False, True])
def test_shared_initialization_uses_native_xiangqi_model(tmp_path, muzero):
    from etazero.initialization import prepare_shared_initializations
    from test_muzero_model import make_model
    options = dict(algorithm='muzero', model_kind=None,
                   muzero_network=asdict(make_model('xiangqi').network_config), muzero_unroll_steps=3) if muzero else {}
    config = controller_config(tmp_path, **options).validate()
    algorithms = ('muzero', 'gumbel_muzero') if muzero else ('alphazero', 'gumbel_alphazero')
    arms = [dict(name=algorithm, run_dir=tmp_path / algorithm, config=replace(config, algorithm=algorithm))
            for algorithm in algorithms]
    weights = prepare_shared_initializations(tmp_path / 'experiment', arms)
    assert weights[algorithms[0]] == weights[algorithms[1]]
    checkpoint = load_checkpoint(weights[algorithms[0]])
    if muzero:
        assert checkpoint['config'] == {'algorithm': 'muzero', 'network': config.muzero_network}
        for component in ('representation.', 'dynamics.', 'prediction.'):
            assert any(key.startswith(component) for key in checkpoint['model'])
    else:
        assert checkpoint['config']['version'] == 19
    assert checkpoint['train_state']['global_step_samples'] == 0
    assert not checkpoint['optimizer']['state']


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and FP16 learner')
@pytest.mark.parametrize('algorithm,weights,canvas', [('alphazero', 'ordinary', 10), ('gumbel_alphazero', 'swa', 12)])
def test_xiangqi_second_round_consumes_published_model(tmp_path, algorithm, weights, canvas):
    check_xiangqi_two_rounds(tmp_path, algorithm, weights, canvas)


def check_xiangqi_two_rounds(tmp_path, algorithm, weights, canvas, *, binary=None, backend_config=None):
    config = controller_config(tmp_path, algorithm=algorithm, publish_weights=weights, canvas=canvas)
    selfplay_config(tmp_path / 'selfplay.cfg', canvas, useGraphSearch=str(not config.is_gumbel).lower(),
                    subtreeValueBiasFactor=.4, avoidRepeatedPatternUtility=.03, **(backend_config or {}))
    config.validate()
    directory = tmp_path / 'run'
    first = run(directory, config, 1, binary=binary)
    checkpoint = load_checkpoint(directory / 'train/checkpoint.ckpt')
    first_weights = {key: value.clone() for key, value in checkpoint['model'].items()}
    second = run(directory, config, 2, binary=binary)
    assert second['completed_rounds'] == 2 and second['total_samples'] > first['total_samples'] > 0
    checkpoint = load_checkpoint(directory / 'train/checkpoint.ckpt')
    assert checkpoint['config']['version'] == 19
    assert checkpoint['optimizer']['state'] and checkpoint['learner_runtime']['scaler']
    assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
    assert any(key.endswith('.weight') and not torch.equal(value, first_weights[key])
               for key, value in checkpoint['model'].items())
    report = json.loads((directory / 'rounds/000002/result.json').read_text())
    assert report['input_model'].startswith('round-000001-s') and report['published_weights'] == weights
    assert report['samples_before'] == first['total_samples']
    output = directory / 'selfplay/000002'
    assert (output / report['input_model']).is_dir() and not (output / 'random').exists()
    verify_native_output(output, canvas)
    modified = (directory / 'train/checkpoint.ckpt').stat().st_mtime_ns
    assert run(directory, config, 2, binary=binary) == second
    assert (directory / 'train/checkpoint.ckpt').stat().st_mtime_ns == modified
