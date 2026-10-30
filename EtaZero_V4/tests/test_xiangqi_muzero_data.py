"""Native Xiangqi trajectories in the fixed root frame used by latent search."""
import numpy as np
import pytest
import torch

from katago.game import xiangqi as xq
from katago.train import checkpoint_runtime
from katago.train.muzero_data import REAL, TERMINAL, TRUNCATED, TARGETS, trajectory_key, expand_training_rows, prepare_batch, read_npz_muzero_data
from test_muzero_data import raw_data, files, load, combined_view, FixedRng
from test_muzero_model import make_model, limited_cpu_threads


def rotate_actions():
    # Independent physical 180-degree mapping; padding is not part of the board.
    return np.array([xq.INDEX[8-x, 9-y, 8-u, 9-v] for x, y, u, v in xq.MOVES])


def spatial_transform(value, *, rotate=False, mirror=False):
    expected = value.copy()
    board = value[..., :10, :9]
    if rotate:
        board = np.rot90(board, 2, axes=(-2, -1))
    if mirror:
        board = np.flip(board, axis=-1)
    expected[..., :10, :9] = board
    return expected


@pytest.mark.parametrize('case', [7, 8, 9])
def test_joint_trajectory_root_coordinates_endpoints_and_sides(raw_data, case):
    rotated = rotate_actions()
    for path in files(raw_data, case):
        raw = load(path)
        view = expand_training_rows(raw, 12)
        planned = (['f9e9', 'f8e8', 'e9f9', 'e8f8', 'f9e9', 'f8e8', 'e9f9'] if case == 9 else
                   ['b9c7', 'c2b0', 'c7b9', 'b0c2', 'b9c7', 'c2b0', 'c7b9'])
        absolute_actions = np.array([xq.INDEX[ord(m[0])-97, int(m[1]), ord(m[2])-97, int(m[3])] for m in planned])
        for lo, hi in zip(raw['muZeroGameOffsets'][:-1], raw['muZeroGameOffsets'][1:]):
            turns = hi-lo-1
            expected = absolute_actions[:turns].copy()
            expected[::2] = rotated[expected[::2]]  # First row is P_WHITE (black pieces).
            np.testing.assert_array_equal(raw['muZeroActions'][lo:hi-1], expected)
            np.testing.assert_array_equal(raw['muZeroPolicyTargetsNCMove'][lo:hi-1, 0].argmax(-1), expected)
        for key in ('binaryInputNCHWPacked', 'globalInputNC', *TARGETS):
            np.testing.assert_array_equal(view[key], raw[key])
        for row, (game, start) in enumerate(raw['muZeroRootNC']):
            kinds = view['muZeroStepKind'][row]
            if start == -1:
                np.testing.assert_array_equal(kinds, [REAL]+[0]*12)
                continue
            lo, hi = raw['muZeroGameOffsets'][game:game+2]
            end = hi-lo-1
            # Root action coordinates depend only on its player, for all future
            # real moves, including the rounded-away turn with weight 0.125.
            expected_actions = absolute_actions[start:end]
            if start % 2 == 0:
                expected_actions = rotated[expected_actions]
            np.testing.assert_array_equal(view['muZeroActions'][row, :end-start], expected_actions)
            for step in range(1, 13):
                turn = start+step
                if turn > end and case == 8:
                    assert kinds[step] == 0
                    continue
                index = lo+min(turn, end)
                expected_future = raw['muZeroValueTargetsNCHW'][index]
                if (min(turn, end)-start) % 2:
                    expected_future = spatial_transform(expected_future, rotate=True)
                if turn > end and (turn-end) % 2:
                    expected_future = -expected_future
                np.testing.assert_array_equal(view['muZeroValueTargetsNCHW'][row, step-1], expected_future)
                for key in ('policyTargetsNCMove', 'qValueTargetsNCMove'):
                    actual = view[trajectory_key(key)][row, step-1]
                    expected = raw[trajectory_key(key)][index]
                    if (min(turn, end)-start) % 2:
                        expected = expected[:, rotated]
                    if turn >= end and case != 8 and key == 'policyTargetsNCMove':
                        expected = np.ones_like(expected)
                    np.testing.assert_array_equal(actual, expected)
                value = view['muZeroGlobalTargetsNC'][row, step-1]
                if turn < end:
                    assert kinds[step] == REAL
                    np.testing.assert_array_equal(value, raw['muZeroGlobalTargetsNC'][index])
                elif case == 8:
                    assert kinds[step] == TRUNCATED
                    assert value[[24, 26, 27, 28, 33]].sum() == 0 and value[35] == 1
                else:
                    assert kinds[step] == TERMINAL
                    expected_wdl = [0, 1, 0, 0] if case == 7 else ([0, 0, 1, 0] if turn % 2 else [1, 0, 0, 0])
                    np.testing.assert_array_equal(value[[0, 70, 1, 2]], expected_wdl)
                    assert value[26] == value[28] == 1 and value[22] == 0


@pytest.mark.parametrize('symmetry', range(8))
def test_fixed_root_mirror_all_joint_actions_and_history(raw_data, symmetry, monkeypatch):
    view = combined_view(raw_data, 9, 12)
    # Keep all available frames for this geometry check. Actual history RNG and
    # restore semantics are covered below without replacing the dropout.
    monkeypatch.setattr(torch, 'rand_like', lambda value: torch.zeros_like(value))
    rng = FixedRng(symmetry)
    root, targets, actions, active = prepare_batch(view, 2., rng, make_model('xiangqi').prediction.config, 'cpu')
    original = np.unpackbits(view['binaryInputNCHWPacked'], axis=-1)[..., :144].reshape(-1, 113, 12, 12)
    mirror = bool(symmetry % 2)
    mapping = np.array([xq.INDEX[8-x, y, 8-u, v] for x, y, u, v in xq.MOVES])
    np.testing.assert_array_equal(root['binaryInputNCHW'], spatial_transform(original, mirror=mirror))
    np.testing.assert_array_equal(root['globalInputNC'], view['globalInputNC'])
    for domain in rng.domains:
        np.testing.assert_array_equal(domain, np.arange(2086))
    expected_actions = view['muZeroActions'].astype(np.int64).copy()
    for row, kinds in enumerate(view['muZeroStepKind']):
        absorbing = np.flatnonzero((kinds[:-1] == TERMINAL) & (kinds[1:] != 0))
        expected_actions[row, absorbing] = np.arange(len(absorbing))
        expected_actions[row, kinds[1:] == 0] = 0
    np.testing.assert_array_equal(actions, mapping[expected_actions] if mirror else expected_actions)
    np.testing.assert_array_equal(active, view['muZeroStepKind'] != 0)
    for step, target in enumerate(targets):
        for key in ('policyTargetsNCMove', 'qValueTargetsNCMove', 'valueTargetsNCHW'):
            expected = view[key] if step == 0 else view[trajectory_key(key)][:, step-1]
            if key == 'valueTargetsNCHW':
                expected = spatial_transform(expected, mirror=mirror)
            elif mirror:
                expected = expected[..., mapping]
            np.testing.assert_array_equal(target[key], expected)
        weight = 1 if step == 0 else view['muZeroWeights'][:, step] / np.where(view['muZeroStepKind'][:, step] == REAL, 2., 1.)
        np.testing.assert_allclose(target['globalTargetsNC'][:, 25], .75*weight)


def test_reader_restores_root_history_absorbing_actions_and_mirror_rng(raw_data, tmp_path):
    view = combined_view(raw_data, 9, 12)
    path = tmp_path / 'view.npz'
    np.savez_compressed(path, **view)
    config = make_model('xiangqi').prediction.config
    rng, validation = np.random.default_rng(45), np.random.default_rng(46)
    state = checkpoint_runtime.capture_runtime(rng, validation, None, torch.device('cpu'))
    def read():
        return list(read_npz_muzero_data([path], 2, 12, 'cpu', config, unroll_steps=12, mean_weight=2., rng=rng))
    expected = read()
    advanced = rng.bit_generator.state
    checkpoint_runtime.restore_runtime(state, rng, validation, None, torch.device('cpu'))
    actual = read()
    assert len(expected) == len(actual) > 0 and advanced == rng.bit_generator.state
    for a, b in zip(actual, expected):
        for left, right in zip([a[0], *a[1]], [b[0], *b[1]]):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        torch.testing.assert_close(a[3], b[3], rtol=0, atol=0)


def test_joint_action_and_observation_boundaries(raw_data):
    raw = load(files(raw_data, 9)[0])
    actions = raw['muZeroActions'].copy()
    actions[0] = 2086
    with pytest.raises(ValueError, match='real action outside'):
        expand_training_rows(dict(raw, muZeroActions=actions), 3)
    with pytest.raises(ValueError, match='observation layout'):
        expand_training_rows(dict(raw, binaryInputNCHWPacked=raw['binaryInputNCHWPacked'][:, :-1]), 3)
    view = expand_training_rows(raw, 3)
    view['muZeroActions'][0, 0] = 2086
    with pytest.raises(ValueError, match='static root domain'):
        prepare_batch(view, 1., np.random.default_rng(3), make_model('xiangqi').prediction.config, 'cpu')
