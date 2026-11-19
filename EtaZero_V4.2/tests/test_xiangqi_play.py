"""Real source GameRunner samples, including forks and reanalysis, feed the learner."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from katago.train.load_model import load_checkpoint
from test_xiangqi_data import feature_reference, load, oriented_action, shuffle
from test_xiangqi_network import model


def run_play(path, network='-', fp16=False, **overrides):
    path.mkdir()
    config = dict(game='xiangqi', bSizesXY='9-10', bSizeRelProbs='1', dataBoardLen='10',
                  logSearchInfo='true', logMoves='true', maxMovesPerGame='12', testFP16=str(fp16).lower())
    config.update(overrides)
    cfg = path / 'play.cfg'
    cfg.write_text(''.join(f'{key} = {value}\n' for key, value in config.items()))
    output = path / 'output'
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-xiangqi-play', str(network), str(cfg), str(output)],
                            capture_output=True, text=True, timeout=180)
    (path / 'native.log').write_text(result.stdout + result.stderr)
    return result, output


def verify_samples(output):
    report = json.loads((output / 'report.json').read_text())
    assert {row['mode'] for row in report} == {0, 1, 2}
    assert sum(row['tree_side'] for row in report) > 0
    assert sum(row['opening'] for row in report if row['game'] == 0) > 0
    assert all(row['used_fork'] == (row['game'] != 0) for row in report)
    for game in report:
        assert game['roots'] == game['reanalyzed'] > 0
        assert game['truncated'] == (game['game'] != 2)
        records = {}
        for side, positions in ((False, game['positions']), (True, game['side'])):
            for position in positions:
                spatial, globals_ = feature_reference(position, 10, side=True)
                policy = np.zeros(2086, dtype=np.int16)
                for action, value in position['policy']:
                    policy[oriented_action(action, position['player'])] = value
                assert policy.sum() > 0
                records[spatial.astype(np.uint8).tobytes(), policy.tobytes(), side] = (globals_, position['entropy'])
        rows = 0
        for file in (output / f"{game['mode']}-{game['game']}").glob('*.npz'):
            data = load(file)
            assert data['policyTargetsNCMove'].shape[1:] == (2, 2086)
            assert data['qValueTargetsNCMove'].shape[1:] == (2, 2086)
            binary = np.unpackbits(data['binaryInputNCHWPacked'], axis=2)[..., :100].reshape(-1, 113, 10, 10)
            targets = data['globalTargetsNC']
            assert np.isfinite(targets).all()
            assert not np.any(targets[:, [2, 6, 10, 14, 18]])
            assert np.all(targets[:, 25] > 0)
            for i, target in enumerate(targets):
                side = not bool(target[64])  # Every played turn was reanalyzed in this fixture.
                globals_, entropy = records[binary[i].tobytes(), data['policyTargetsNCMove'][i, 0].tobytes(), side]
                np.testing.assert_allclose(data['globalInputNC'][i], globals_, atol=1e-7, rtol=0)
                np.testing.assert_allclose(target[59], entropy, atol=5e-7, rtol=0)
                if game['truncated']:
                    assert target[35] == 1 and target[27] == 0 and target[62] == 0
                elif not side:
                    assert target[35] == 0 and target[62] == 1
                    np.testing.assert_array_equal(target[[0, 70, 1, 2]], [0, 1, 0, 0])
                if side or not game['outcome_targets']:
                    assert target[28] == 0 and target[33] == 0
            rows += len(targets)
        assert rows == game['rows'] > 0
    log = (output / 'play.log').read_text()
    assert 'Xiangqi FEN:' in log and 'Reanalyzed' in log


def test_source_game_runner_random_evaluator(tmp_path):
    result, output = run_play(tmp_path / 'play')
    assert result.returncode == 0, result.stdout + result.stderr
    verify_samples(output)


@pytest.mark.parametrize('overrides, error', [
    ({'bSizesXY': '10-10'}, 'requires bSizesXY = 9-10'),
    ({'handicapProb': '1'}, 'Go handicap initialization cannot be used'),
])
def test_game_initialization_boundaries(tmp_path, overrides, error):
    result, _ = run_play(tmp_path / 'invalid', **overrides)
    assert result.returncode != 0 and error in result.stdout + result.stderr


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and FP16 learner')
def test_source_cuda_selfplay_shuffle_learner_and_export(tmp_path):
    net = model(with_q=True).eval()
    network = tmp_path / 'model.bin'
    with network.open('wb') as out:
        write_model_binary(net, out, 'xiangqi-play')
    result, output = run_play(tmp_path / 'play', network, fp16=True)
    assert result.returncode == 0, result.stdout + result.stderr
    verify_samples(output)
    shuffled = shuffle(output, tmp_path / 'shuffled')
    directory = tmp_path / 'data'; directory.mkdir()
    for split in ('train', 'val'):
        shutil.copytree(shuffled, directory / split)
        shutil.copyfile(shuffled.with_suffix('.json'), directory / (split+'.json'))
    train = tmp_path / 'train'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(ROOT / 'python/train.py'), '-traindir', str(train), '-datadir', str(directory),
               '-pos-len', '10', '-batch-size', '4', '-samples-per-epoch', '8', '-max-epochs-this-instance', '1',
               '-lookahead-k', '2', '-use-fp16', '-no-compile', '-no-export', '-no-longterm-checkpoints',
               '-quit-if-no-data', '-model-kind', 'b2c16-fson-mish', '-input-features', 'xiangqi']
    log = tmp_path / 'learner.log'
    with log.open('w') as out:
        result = subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, log.read_text()[-10000:]
    checkpoint = load_checkpoint(train / 'checkpoint.ckpt')
    assert checkpoint['config']['version'] == 19
    assert checkpoint['optimizer']['state'] and checkpoint['learner_runtime']['scaler']
    assert checkpoint['train_state']['global_step_samples'] > 0
    assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
    export = tmp_path / 'export'
    result = subprocess.run([sys.executable, str(ROOT / 'python/export_model_pytorch.py'), '-checkpoint',
        str(train / 'checkpoint.ckpt'), '-export-dir', str(export), '-model-name', 'xiangqi-trained',
        '-filename-prefix', 'model'], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    request = tmp_path / 'search.json'
    request.write_text(json.dumps(dict(evaluate=True, search=True, side=10, fp16=True,
                                      inputs_nhwc=False, gumbel=True, threads=4)))
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-xiangqi-inference',
        str(export / 'model.bin'), str(request), str(tmp_path / 'search-result.json')],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
