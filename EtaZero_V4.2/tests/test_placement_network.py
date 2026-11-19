"""Placement features, action layout, valid augmentation and native export parity."""
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

from etazero import ROOT
from katago.train import data_processing_pytorch as data
from katago.train import modelconfigs
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Model


def placement_model(with_q=False, intermediate=False):
    torch.manual_seed(91)
    config = dict(modelconfigs.config_of_name['b2c16-fson-mish'], version=18,
                  predict_q_values=with_q, has_intermediate_head=intermediate, intermediate_head_blocks=1)
    model = Model(config, 7)
    model.initialize()
    return model.eval()


def inputs(game, white):
    binary = torch.zeros(1, 5, 7, 7)
    binary[:, 0] = 1
    binary[0, 2 if white else 1, 4, 1] = 1  # Black B3; GTP counts rows from the bottom.
    binary[0, 1 if white else 2, 1, 4] = 1  # White E6.
    global_input = torch.tensor([[game == 'standard', game == 'renju',
                                  (1 if white else -1) if game == 'renju' else 0,
                                  game == 'renju', 0, 0, game == 'hex', game == 'hex' and white]], dtype=torch.float32)
    return binary, global_input


@pytest.mark.parametrize('with_q', [False, True])
@pytest.mark.parametrize('intermediate', [False, True])
def test_placement_actions_and_masked_nr_gradient(with_q, intermediate):
    model = placement_model(with_q, intermediate)
    binary, global_input = inputs('renju', False)
    binary = binary.expand(3, -1, -1, -1).contiguous()
    global_input = global_input.expand(3, -1).contiguous()
    assert not any('pass' in name for name in model.state_dict())
    raw = model(binary, global_input)
    outputs = model.postprocess_output(raw)
    targets = torch.zeros(3, 80)
    for columns in ([0, 70, 1], [4, 71, 5], [8, 72, 9], [12, 73, 13]):
        targets[torch.arange(3), columns] = 1
    targets[:, 25:29] = 1
    batch = dict(binaryInputNCHW=binary, globalInputNC=global_input, globalTargetsNC=targets,
                 policyTargetsNCMove=torch.ones(3, 2, 49), valueTargetsNCHW=torch.zeros(3, 2, 7, 7))
    if with_q:
        batch['qValueTargetsNCMove'] = torch.zeros(3, 2, 49)
        batch['qValueTargetsNCMove'][:, 1] = 1
    losses = Metrics(1, model).metrics_dict_batchwise(
        model, outputs, extra_outputs=None, batch=batch, is_training=True,
        soft_policy_weight_scale=1., disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
        value_loss_scale=1., td_value_loss_scales=[1., 1., 1.], variance_time_loss_scale=.5,
        main_loss_scale=1., intermediate_loss_scale=.5 if intermediate else None, include_model_norms=False)
    assert torch.isfinite(losses['loss_sum'])
    for index, output in enumerate(outputs):
        assert output[0].shape == (3, 7 if with_q else 6, 49)
        assert torch.count_nonzero(output[1].softmax(-1)[:, 3]) == 0
        assert torch.count_nonzero(output[2].softmax(-1)[:, :, 3]) == 0
        value_gradient = torch.autograd.grad(losses['loss_sum'], raw[index][1], retain_graph=True)[0]
        assert torch.count_nonzero(value_gradient[:, 3]) == 0
        assert torch.count_nonzero(value_gradient[:, :3]) > 0
    losses['loss_sum'].backward()
    assert model.conv_spatial.weight.grad.isfinite().all()


@pytest.mark.parametrize('key,value', [('metadata_encoder', {}),
    ('always_compute_pass_alive_under_suicide_rules', True), ('exclude_territory_adjacent_to_atari', True)])
def test_go_feature_options_rejected(key, value):
    with pytest.raises(ValueError, match='Go metadata or Go feature options'):
        Model(dict(modelconfigs.config_of_name['b2c16-fson-mish'], version=18, **{key: value}), 7)


@pytest.mark.parametrize('symmetry', range(8))
def test_reader_transforms_mixed_games_consistently(tmp_path, monkeypatch, symmetry):
    # Force each augmentation, and keep one asymmetric nonzero point in every target.
    class FixedRng:
        def integers(self, low, high):
            return symmetry
    monkeypatch.setattr(data.np.random, 'default_rng', lambda **kwargs: FixedRng())
    binary = np.zeros((2, 5, 7, 7), dtype=np.uint8)
    binary[:, 0] = 1
    binary[:, 1, 1, 2] = 1
    global_input = np.zeros((2, 8), dtype=np.float32)
    global_input[1, 6] = 1  # First row: freestyle; second row: Hex.
    policy = np.zeros((2, 2, 49), dtype=np.int16)
    policy[:, :, 9] = 11
    q = np.zeros_like(policy)
    q[:, :, 9] = 17
    future = np.zeros((2, 2, 7, 7), dtype=np.int8)
    future[:, :, 1, 2] = 1
    path = tmp_path / 'mixed.npz'
    np.savez(path, binaryInputNCHWPacked=np.packbits(binary.reshape(2, 5, 49), axis=2),
             globalInputNC=global_input, policyTargetsNCMove=policy, qValueTargetsNCMove=q,
             globalTargetsNC=np.zeros((2, 80), dtype=np.float32), valueTargetsNCHW=future)
    batch, = data.read_npz_training_data([path], batch_size=2, world_size=1, rank=0, pos_len=7,
        device='cpu', randomize_symmetries=True, include_meta=False,
        model_config=placement_model(with_q=True).config)
    # Independent NumPy rotation/transpose oracle for the source's augmentation convention.
    marker = binary[0, 1]
    expected_five = np.rot90(marker, symmetry % 4)
    if symmetry >= 4:
        expected_five = expected_five.T
    expected_hex = marker if symmetry % 2 == 0 else np.rot90(marker, 2)
    for row, expected in enumerate((expected_five, expected_hex)):
        np.testing.assert_array_equal(batch['binaryInputNCHW'][row, 1], expected)
        for channel in range(2):
            np.testing.assert_array_equal(batch['policyTargetsNCMove'][row, channel].reshape(7, 7), expected * 11)
            np.testing.assert_array_equal(batch['qValueTargetsNCMove'][row, channel].reshape(7, 7), expected * 17)
            np.testing.assert_array_equal(batch['valueTargetsNCHW'][row, channel], expected)
    torch.testing.assert_close(batch['globalInputNC'], torch.from_numpy(global_input))


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA build and GPU')
@pytest.mark.parametrize('fp16,with_q', [(False, False), (True, True)])
def test_placement_export_matches_native(tmp_path, fp16, with_q):
    check_placement_export(tmp_path, ROOT / 'build/katago', fp16, with_q)


def check_placement_export(tmp_path, binary, fp16, with_q, *, backend_config=None):
    model = placement_model(with_q)
    checkpoint = tmp_path / 'checkpoint.ckpt'
    torch.save(dict(config=model.config, model=model.state_dict()), checkpoint)
    subprocess.run([sys.executable, str(ROOT / 'python/export_model_pytorch.py'), '-checkpoint', str(checkpoint),
                    '-export-dir', str(tmp_path), '-model-name', 'placement', '-filename-prefix', 'model'],
                   capture_output=True, text=True, check=True)
    config = tmp_path / 'gtp.cfg'
    config.write_text(f'rules = freestyle\nnumSearchThreads = 1\nmaxVisits = 4\nrootNumSymmetriesToSample = 8\ncudaUseFP16 = {str(fp16).lower()}\n'
                      'logSearchInfo = false\nlogToStderr = false\nlogAllGTPCommunication = false\nponderingEnabled = false\n')
    if backend_config:
        with config.open('a') as stream:
            stream.write(''.join(f'{key} = {value}\n' for key, value in backend_config.items()))
    commands = 'boardsize 7\nplay b B3\nplay w E6\n'
    for game in ['freestyle', 'standard', 'renju', 'hex']:
        commands += f'kata-set-rules {game}\nkata-raw-nn b 0 0\nkata-raw-nn w 0 0\n'
    commands += '90 kata-raw-nn 1\n91 kata-raw-nn all\n92 genmove b\nquit\n'
    result = subprocess.run([str(binary), 'gtp', '-config', str(config),
                             '-model', str(tmp_path / 'model.bin')], input=commands, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(re.findall(r'^\?', result.stdout, re.M)) == 1, result.stdout
    assert '?90 Requested symmetry is invalid for the game' in result.stdout
    assert re.findall(r'symmetry (\d+)', result.stdout) == ['0'] * 8 + ['0', '3']
    assert re.search(r'=92 (?!pass)[A-Z][0-9]+', result.stdout), result.stdout
    matches = re.findall(r'whiteWin ([\d.]+)\ndraw ([\d.]+)\nwhiteLoss ([\d.]+)\nnoResult ([\d.]+)', result.stdout)
    policies = re.findall(r'\npolicy\n(.*?)\npolicyPass\s+NAN', result.stdout, re.S)
    assert len(policies) == 10, result.stdout
    for i, (game, white) in enumerate((g, w) for g in ['freestyle', 'standard', 'renju', 'hex'] for w in [False, True]):
        binary, global_input = inputs(game, white)
        with torch.no_grad():
            outputs = model.postprocess_output(model(binary, global_input))[0]
        expected_value = outputs[1][0].softmax(-1).numpy()
        if not white:
            expected_value = expected_value[[2, 1, 0, 3]]
        np.testing.assert_allclose(np.array(matches[i], dtype=float), expected_value, atol=5e-4 if fp16 else 2e-5, rtol=0)
        logits = outputs[0][0, 0].detach().clone()
        logits[[29, 11]] = -torch.inf
        actual_policy = np.array(policies[i].split(), dtype=float)
        legal = np.isfinite(actual_policy)
        assert not legal[29] and not legal[11] and legal.sum() == 47
        np.testing.assert_allclose(actual_policy[legal], logits.softmax(-1).numpy()[legal],
                                   atol=5e-5 if fp16 else 3e-6, rtol=0)
