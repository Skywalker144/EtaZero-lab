"""Native Xiangqi rows through source shuffle, augmentation and learner."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from etazero import ROOT
from katago.game import xiangqi as xq
from katago.train import data_processing_pytorch as reader
from katago.train.load_model import load_checkpoint
from test_xiangqi_network import model, loss
from test_muzero_model import limited_cpu_threads


@pytest.fixture(scope='module')
def native_data(tmp_path_factory):
    path = tmp_path_factory.mktemp('xiangqi-data')
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-xiangqi-data', str(path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return path


def load(path):
    with np.load(path) as archive:
        return dict(archive)


def feature_reference(position, size, side=False):
    mine = 1 if position['player'] == 1 else -1
    planes = np.zeros((113, size, size), dtype=np.float32)
    planes[0, :10, :9] = 1
    for frame, pieces in enumerate(position['frames']):
        for index, piece in enumerate(pieces):
            if not piece:
                continue
            y, x = divmod(index, 9)
            if mine == -1:
                x, y = 8-x, 9-y
            channel = 1 + frame*14 + (0 if np.sign(piece) == mine else 7) + abs(piece)-1
            planes[channel, y, x] = 1
    global_input = [mine == 1, position['rule60']/120, (position['repetitions']-1)/2,
                    not side, 0 if side else mine*.25, len(position['frames'])/8,
                    position['check'], position['ply']/200]
    return planes, np.asarray(global_input, dtype=np.float32)


def oriented_action(action, player):
    x, y, u, v = xq.MOVES[action]
    return xq.INDEX[8-x, 9-y, 8-u, 9-v] if player == 2 else action


@pytest.mark.parametrize('canvas', [10, 12])
@pytest.mark.parametrize('ending', ['terminal', 'truncated'])
def test_native_features_joint_targets_and_source_weights(native_data, canvas, ending):
    directory = native_data / f'{canvas}-{ending}'
    expected = json.loads((directory / 'expected.json').read_text())
    data = load(next(directory.glob('*.npz')))
    n = len(expected['positions'])-1
    assert len(data['globalInputNC']) == expected['rows'] == n+1
    assert data['policyTargetsNCMove'].shape == data['qValueTargetsNCMove'].shape == (n+1, 2, 2086)
    assert data['policyTargetsNCMove'].dtype == data['qValueTargetsNCMove'].dtype == np.int16
    assert data['valueTargetsNCHW'].dtype == np.int8
    binary = np.unpackbits(data['binaryInputNCHWPacked'], axis=2)[..., :canvas*canvas].reshape(-1, 113, canvas, canvas)
    records = data['globalTargetsNC']
    np.testing.assert_array_equal(records[:, 25], .75)
    np.testing.assert_array_equal(records[:, [2, 6, 10, 14, 18]], 0)
    assert Counter(records[:-1, 51]) == Counter([1, 1, *range(3, n+1)])
    assert records[-1, 60] == 10000
    white = np.array([[.2+.005*t, .1, .7-.005*t, 0] for t in range(n)])
    endpoint = [.38, .1, .52, 0] if ending == 'truncated' else [1, 0, 0, 0]
    white = np.vstack((white, endpoint))
    for row, target in enumerate(records):
        side = target[60] == 10000
        turn = int(target[51])-1
        position = expected['side'] if side else expected['positions'][turn]
        spatial, global_input = feature_reference(position, canvas, side)
        np.testing.assert_array_equal(binary[row], spatial)
        np.testing.assert_allclose(data['globalInputNC'][row], global_input, atol=1e-7, rtol=0)
        player = position['player']
        disabled = side or turn == 2
        for channel in range(2):
            source = None if channel and (disabled or turn+1 == n) else position if channel == 0 else expected['positions'][turn+1]
            policy = np.ones(2086, dtype=np.int16) if source is None else np.zeros(2086, dtype=np.int16)
            if source is not None:
                for action, value in source['policy']:
                    policy[oriented_action(action, player)] = value
            np.testing.assert_array_equal(data['policyTargetsNCMove'][row, channel], policy)
            assert target[26 if channel == 0 else 28] == (source is not None)
        q = np.zeros((2, 2086), dtype=np.int16)
        for action, value, visits in position['q']:
            q[:, oriented_action(action, player)] = [value * (32000 if player == 2 else -32000), visits]
        np.testing.assert_array_equal(data['qValueTargetsNCMove'][row], q)
        assert target[35] == (ending == 'truncated')
        assert target[27] == (not disabled and ending == 'terminal')
        assert target[33] == (not disabled)
        future = np.zeros((2, canvas, canvas), dtype=np.int8)
        for channel, horizon in enumerate((8, 32)):
            assert target[75+channel] == (not disabled and ending == 'truncated' and turn+horizon > n)
            if not disabled:
                pieces = np.asarray(expected['positions'][min(turn+horizon, n)]['frames'][0]).reshape(10, 9)
                occupancy = np.sign(pieces)
                if player == 2:
                    occupancy = -occupancy[::-1, ::-1]
                future[channel, :10, :9] = occupancy
        np.testing.assert_array_equal(data['valueTargetsNCHW'][row], future)
        for factor, columns in ((0, [0, 70, 1, 2]), (.176, [4, 71, 5, 6]),
                                (.056, [8, 72, 9, 10]), (.016, [12, 73, 13, 14])):
            if side:
                value = white[0]
            elif factor == 0:
                value = white[-1]
            else:
                alpha = 1/(1+90*factor)
                value = (white[turn:n] * (alpha*(1-alpha)**np.arange(n-turn))[:, None]).sum(axis=0)
                value += (1-alpha)**(n-turn)*white[-1]
            if player == 1:
                value = value[[2, 1, 0, 3]]
            np.testing.assert_allclose(target[columns], value, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize('symmetry', range(8))
def test_reader_mirrors_every_target_on_physical_board(native_data, monkeypatch, symmetry):
    path = next((native_data / '12-truncated').glob('*.npz'))
    data = load(path)
    monkeypatch.setattr(torch, 'rand_like', lambda tensor: torch.zeros_like(tensor))
    class FixedRng:
        def integers(self, low, high):
            assert (low, high) == (0, 8)
            return symmetry
    batches = list(reader.read_npz_training_data([path], batch_size=2, world_size=1, rank=0,
        pos_len=12, device='cpu', randomize_symmetries=True, include_meta=False,
        model_config=model(with_q=True, pos_len=12).config, rng=FixedRng()))
    count = len(batches)*2
    transformed = {key: torch.cat([b[key] for b in batches]).numpy() for key in batches[0]}
    binary = np.unpackbits(data.pop('binaryInputNCHWPacked'), axis=2)[..., :144].reshape(-1, 113, 12, 12)
    data['binaryInputNCHW'] = binary
    for key, value in data.items():
        expected = value[:count].copy()
        if symmetry % 2:
            if key in ('binaryInputNCHW', 'valueTargetsNCHW'):
                expected[..., :10, :9] = value[:count, ..., :10, :9][..., ::-1]
            elif key in ('policyTargetsNCMove', 'qValueTargetsNCMove'):
                for action, (x, y, u, v) in enumerate(xq.MOVES):
                    expected[..., xq.INDEX[8-x, y, 8-u, v]] = value[:count, ..., action]
        np.testing.assert_array_equal(transformed[key], expected, err_msg=key)


def test_history_dropout_keeps_current_frame_and_rule_counters(monkeypatch):
    spatial = torch.ones(3, 113, 12, 12)
    global_input = torch.tensor([[1, .5, 1, 0, 0, 1, 1, .4], [0, .8, .5, 0, 0, .5, 0, .8], [1, .3, 0, 0, 0, 1, 0, 1]])
    draws = torch.tensor([[.979, .98, 0, 0, 0, 0, 0], [.1, .1, .1, .1, .1, .99, 0], [0, 0, 0, 0, 0, 0, 0]])
    monkeypatch.setattr(torch, 'rand_like', lambda tensor: draws)
    features, globals_ = reader.apply_xiangqi_history(spatial, global_input)
    for row, frames in enumerate((2, 6, 8)):
        assert torch.all(features[row, :1+frames*14] == 1)
        assert torch.all(features[row, 1+frames*14:] == 0)
    torch.testing.assert_close(globals_[:, 5], torch.tensor([.25, .5, 1.]))
    torch.testing.assert_close(globals_[:, [0, 1, 2, 3, 4, 6, 7]], global_input[:, [0, 1, 2, 3, 4, 6, 7]])
    assert torch.all(spatial == 1) and global_input[0, 5] == 1


def shuffle(source, output):
    command = [sys.executable, str(ROOT / 'python/shuffle.py'), str(source), '-out-dir', str(output),
               '-out-tmp-dir', str(output.with_name(output.name+'-scratch')), '-min-rows', '1',
               '-keep-target-rows', 'all', '-num-processes', '2', '-approx-rows-per-out-file', '1000']
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def row_hashes(data):
    return Counter(hashlib.sha256(b''.join(data[k][i].tobytes() for k in sorted(data))).hexdigest()
                   for i in range(len(data['globalInputNC'])))


def test_source_shuffle_and_joint_gradient(native_data, tmp_path):
    source = native_data / '10-truncated'
    output = shuffle(source, tmp_path / 'shuffled')
    expected = row_hashes(load(next(source.glob('*.npz'))))
    actual = Counter()
    for path in output.glob('*.npz'):
        actual.update(row_hashes(load(path)))
    assert actual == expected
    net = model(with_q=True).train()
    batch = next(reader.read_npz_training_data(sorted(output.glob('*.npz')), batch_size=4, world_size=1, rank=0,
        pos_len=10, device='cpu', randomize_symmetries=True, include_meta=False, model_config=net.config))
    raw = net(batch['binaryInputNCHW'], batch['globalInputNC'])
    total = loss(net, net.postprocess_output(raw), batch)
    assert total.isfinite()
    gradient = torch.autograd.grad(total, raw[0][1], retain_graph=True)[0]
    assert torch.count_nonzero(gradient) == 0  # Truncated data never invents a final result.
    total.backward()
    assert net.policy_head.conv2p.weight.grad.isfinite().all()
    assert torch.count_nonzero(net.policy_head.conv2p.weight.grad) > 0


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA learner and export')
def test_source_fp16_learner_resume_and_export(native_data, tmp_path):
    output = shuffle(native_data / '10-truncated', tmp_path / 'shuffled')
    directory = tmp_path / 'data'; directory.mkdir()
    for split in ('train', 'val'):
        shutil.copytree(output, directory / split)
        shutil.copyfile(output.with_suffix('.json'), directory / (split+'.json'))
    rows = sum(len(load(p)['globalInputNC'])//4*4 for p in output.glob('*.npz'))
    train = tmp_path / 'train'
    command = [sys.executable, str(ROOT / 'python/train.py'), '-traindir', str(train), '-datadir', str(directory),
               '-pos-len', '10', '-batch-size', '4', '-samples-per-epoch', '8', '-max-epochs-this-instance', '1',
               '-lookahead-k', '2', '-swa-period-samples', '8', '-use-fp16', '-no-compile', '-no-export',
               '-no-longterm-checkpoints', '-quit-if-no-data']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    checkpoints = []
    for index, options in enumerate((['-initialize-only', '-model-kind', 'b2c16-fson-mish', '-input-features', 'xiangqi'], [], [])):
        log = tmp_path / f'learner-{index}.log'
        with log.open('w') as out:
            result = subprocess.run(command+options, env=env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, log.read_text()[-10000:]
        checkpoints.append(load_checkpoint(train / 'checkpoint.ckpt'))
    initial, first, second = checkpoints
    assert [c['train_state']['global_step_samples'] for c in checkpoints] == [0, rows, 2*rows]
    assert all(c['config']['version'] == 19 for c in checkpoints)
    assert first['optimizer']['state'] and second['learner_runtime']['scaler']
    assert second['learner_runtime']['cuda_rng'] is not None
    assert not torch.equal(first['learner_runtime']['cuda_rng'], second['learner_runtime']['cuda_rng'])
    assert first['learner_runtime']['train_rng'] != second['learner_runtime']['train_rng']
    assert first['swa_model']['n_averaged'] == rows//8
    for old, new in ((initial, first), (first, second)):
        assert not torch.equal(old['model']['policy_head.conv2p.weight'], new['model']['policy_head.conv2p.weight'])
    saved = (train / 'checkpoint.ckpt').read_bytes()
    mismatch = subprocess.run(command+['-input-features', 'go'], env=env, cwd=ROOT,
                              capture_output=True, text=True, timeout=60)
    assert mismatch.returncode != 0 and 'do not match the requested game family' in mismatch.stdout+mismatch.stderr
    assert (train / 'checkpoint.ckpt').read_bytes() == saved
    for use_swa in (False, True):
        export = tmp_path / ('swa' if use_swa else 'ordinary')
        result = subprocess.run([sys.executable, str(ROOT / 'python/export_model_pytorch.py'), '-checkpoint',
            str(train / 'checkpoint.ckpt'), '-export-dir', str(export), '-model-name', 'xiangqi-data',
            '-filename-prefix', 'model', *(['-use-swa'] if use_swa else [])], env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        request = tmp_path / f'{use_swa}-request.json'
        request.write_text(json.dumps(dict(evaluate=True, search=True, side=10, fp16=True,
                                          inputs_nhwc=False, gumbel=True, threads=4)))
        result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-xiangqi-inference',
            str(export / 'model.bin'), str(request), str(tmp_path / f'{use_swa}-response.json')],
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
