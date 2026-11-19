"""Compare ordered rows and RNG state against the source permutation algorithm."""
import json

import numpy as np
import pytest

import shuffle


@pytest.mark.parametrize('optional', [False, True])
@pytest.mark.parametrize('keep_prob', [0.0, 0.4, 1.0])
def test_sharding_and_merging_preserve_source_permutations(tmp_path, monkeypatch, optional, keep_prob):
    entropy = b'\x01' * 4
    monkeypatch.setattr(shuffle.os, 'urandom', lambda n: entropy[:n])
    seeds = [int.from_bytes(entropy, 'little')]
    keys = ['binaryInputNCHWPacked', 'globalInputNC', 'policyTargetsNCMove',
            'globalTargetsNC', 'valueTargetsNCHW']
    shapes = [(2, 1), (3,), (2, 5), (80,), (2, 2, 2)]
    dtypes = [np.uint8, np.float32, np.int16, np.float32, np.int8]
    if optional:
        keys += ['metadataInputNC', 'qValueTargetsNCMove']
        shapes += [(4,), (2, 5)]
        dtypes += [np.float32, np.int16]
    raw = []
    paths = []
    for group, rows in enumerate([7, 6, 2]):
        values = {k: (np.arange(rows * np.prod(s)).reshape(rows, *s) + group * 10).astype(d)
                  for k, s, d in zip(keys, shapes, dtypes)}
        raw.append(values)
        path = tmp_path / f'raw{group}.npz'
        np.savez_compressed(path, **values)
        paths.append(str(path))
    buckets = [tmp_path / f'bucket{i}' for i in range(7)]
    for path in buckets:
        path.mkdir()
    expected_buckets = [[] for _ in buckets]
    for group, source_indices in enumerate(([0, 1], [2])):
        combined = {k: np.concatenate([raw[i][k] for i in source_indices]) for k in keys}
        count = len(combined[keys[0]])
        kept = min(count, int(round(count * keep_prob)))
        rng = np.random.RandomState(seeds * 4)
        order = rng.permutation(count)[:kept]
        counts = np.bincount(rng.randint(len(buckets), size=kept), minlength=len(buckets))
        shuffled = {k: a[order] for k, a in combined.items()}
        shuffle.shardify(group, [paths[i] for i in source_indices], len(buckets),
                         list(map(str, buckets)), keep_prob, optional, optional)
        np.testing.assert_array_equal(np.random.randint(1000, size=8), rng.randint(1000, size=8))
        offset = 0
        for bucket, rows in enumerate(counts):
            expected = {k: a[offset:offset + rows] for k, a in shuffled.items()}
            expected_buckets[bucket].append(expected)
            with np.load(buckets[bucket] / f'{group}.npz') as actual:
                assert set(actual.files) == set(keys)
                for key in keys:
                    assert actual[key].tobytes() == expected[key].tobytes()
            offset += rows
    for bucket, parts in zip(buckets, expected_buckets):
        combined = {k: np.concatenate([part[k] for part in parts]) for k in keys}
        count = len(combined[keys[0]])
        rng = np.random.RandomState(seeds * 5)
        order = rng.permutation(count)
        expected = {k: a[order] for k, a in combined.items()}
        outputs = [tmp_path / f'{bucket.name}-out{i}.npz' for i in range(2)]
        shuffle.merge_bucket(list(map(str, outputs)), 2, str(bucket), optional, optional)
        np.testing.assert_array_equal(np.random.randint(1000, size=8), rng.randint(1000, size=8))
        for i, path in enumerate(outputs):
            start, stop = i * count // 2, (i + 1) * count // 2
            with np.load(path) as actual:
                for key in keys:
                    assert actual[key].tobytes() == expected[key][start:stop].tobytes()
            assert json.loads(path.with_suffix('.json').read_text())['num_rows'] == stop - start
