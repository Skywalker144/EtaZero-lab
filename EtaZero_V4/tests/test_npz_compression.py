"""Fast compression preserves NPY payloads, including optional targets and slices."""
import zipfile

import numpy as np
import pytest

from shuffle import save_output_npz


@pytest.mark.parametrize('optional', [False, True])
def test_shuffled_arrays_are_lossless_and_readable(optional, tmp_path):
    # Non-contiguous views exercise the streaming NPY writer as well as exact
    # dtypes, negative zero, nonfinite floats and a selected row interval.
    values = np.array([0., -0., np.nan, np.inf, -np.inf, 1.2345], dtype=np.float32)
    arrays = [np.arange(48, dtype=np.uint8).reshape(6, 8)[:, ::2],
              values[:, None], np.arange(36, dtype=np.int16).reshape(6, 2, 3),
              np.tile(values[:, None], (1, 80)), np.arange(24, dtype=np.float32).reshape(6, 2, 2),
              np.ones((6, 3), dtype=np.float32) if optional else None,
              np.zeros((6, 2, 3), dtype=np.int16) if optional else None]
    names = ['binaryInputNCHWPacked', 'globalInputNC', 'policyTargetsNCMove', 'globalTargetsNC',
             'valueTargetsNCHW', 'metadataInputNC', 'qValueTargetsNCMove']
    output, reference = tmp_path / 'fast.npz', tmp_path / 'reference.npz'
    save_output_npz(output, arrays, optional, optional, 1, 5)
    expected = {name: value[1:5] for name, value in zip(names, arrays) if value is not None}
    np.savez_compressed(reference, **expected)
    with zipfile.ZipFile(output) as actual, zipfile.ZipFile(reference) as baseline:
        assert actual.namelist() == baseline.namelist()
        for name in baseline.namelist():
            assert actual.read(name) == baseline.read(name)
    with np.load(output) as actual:
        for name, value in expected.items():
            assert actual[name].dtype == value.dtype
            np.testing.assert_array_equal(actual[name], value)
