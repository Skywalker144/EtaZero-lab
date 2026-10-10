"""Probability roundoff must not turn TD target entropy into NaN on resume."""
import warnings
import numpy as np
import pytest
import torch
from etazero.reader import BatchReader, check_wdl_targets
from etazero.storage import sha256
from test_replay import id_snapshot


@pytest.mark.parametrize('shape', [(4, 3), (4, 3, 3), (4, 6, 3, 3)])
def test_roundoff_correction_preserves_valid_targets_and_finite_entropy(shape):
    targets = np.broadcast_to(np.array([.2, .3, .5], np.float32), shape).copy()
    targets.reshape(-1, 3)[0] = [-5e-11, 1., 7e-9]
    original = targets.copy()
    with pytest.warns(RuntimeWarning, match='corrected WDL roundoff.*td_value'):
        check_wdl_targets({'td_value': targets}, 'sample.npz')
    np.testing.assert_array_equal(targets.reshape(-1, 3)[1:], original.reshape(-1, 3)[1:])
    assert targets.reshape(-1, 3)[0, 0] == 0
    assert (targets >= 0).all() and (targets <= 1).all()
    np.testing.assert_allclose(targets.sum(-1), 1, atol=1e-7)
    p = torch.from_numpy(targets)
    assert torch.isfinite(-(p * (p+1e-30).log()).sum(-1)).all()
    with warnings.catch_warnings(record=True) as seen:
        check_wdl_targets({'td_value': targets}, 'sample.npz')
    assert not seen


@pytest.mark.parametrize('target', [[-.01, .51, .5], [1.01, 0, 0], [0, 0, 0],
                                   [-5e-7, 0, 0], [.2, .3, .4], [np.nan, 0, 1], [np.inf, 0, 1]])
def test_invalid_probabilities_are_rejected_with_source_and_index(target):
    arrays = {'value': np.array([[0, 0, 1], target], np.float32)}
    with pytest.raises(ValueError, match=r'sample.npz: invalid value WDL at index \(1,\)'):
        check_wdl_targets(arrays, 'sample.npz')


@pytest.mark.parametrize('name', ['value', 'td_value'])
@pytest.mark.parametrize('active', [False, True])
def test_near_zero_mass_is_rejected_before_clipping_even_with_sequence_mask(name, active):
    target = np.array([[[-5e-7, 0, 0]]], np.float32)
    if name == 'td_value':
        target = np.repeat(target[:, :, None], 3, axis=2)
    original = target.copy()
    arrays = {name: target, 'sequence_mask': np.array([[active]], np.float32)}
    with np.errstate(divide='raise', invalid='raise'):
        with pytest.raises(ValueError, match=f'invalid {name} WDL'):
            check_wdl_targets(arrays, 'near-zero.npz')
    np.testing.assert_array_equal(target, original)


def test_muzero_absent_continuations_keep_zero_targets():
    value = np.array([[[.2, .3, .5], [0, 0, 0]]], np.float32)
    arrays = {'value': value.copy(), 'td_value': np.repeat(value[:, :, None], 3, axis=2),
              'sequence_mask': np.array([[1, 0]], np.float32)}
    check_wdl_targets(arrays, 'sequence.npz')
    np.testing.assert_array_equal(arrays['value'], value)
    assert not arrays['td_value'][0, 1].any()
    arrays['sequence_mask'][0, 1] = 1
    with pytest.raises(ValueError, match='invalid value WDL'):
        check_wdl_targets(arrays, 'sequence.npz')


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_reader_corrects_roundoff_after_resume_without_rewriting_snapshot(tmp_path, split):
    snapshot = id_snapshot(tmp_path, [8], validation=True)
    directory = snapshot/'data'
    if split == 'validation':
        directory /= 'validation'
    path = directory/'000.npz'
    with np.load(path) as file:
        arrays = {key: file[key] for key in file.files}
    arrays['value'][5] = [-5e-11, 1., 7e-9]
    arrays['td_value'][5] = arrays['value'][5]
    np.savez(path, **arrays)
    digest = sha256(path)
    reader = BatchReader(snapshot, 4, 1, 17, no_repeat_files=True, split=split)
    try:
        with pytest.warns(RuntimeWarning, match='corrected WDL roundoff'):
            reader.next()
        cursor = reader.state()
    finally:
        reader.close()
    resumed = BatchReader(snapshot, 4, 1, 99, cursor, no_repeat_files=True, split=split)
    try:
        with pytest.warns(RuntimeWarning, match='corrected WDL roundoff'):
            batch = resumed.next()
        assert batch['globals'][:, 5].tolist() == [4, 5, 6, 7]
        assert batch['value'][1, 0] == 0 and (batch['td_value'] >= 0).all()
        assert resumed.state()['offset'] == 8
    finally:
        resumed.close()
    assert sha256(path) == digest
    with np.load(path) as file:
        assert file['value'][5, 0] < 0
