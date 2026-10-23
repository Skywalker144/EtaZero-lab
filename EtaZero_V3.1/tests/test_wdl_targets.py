"""Producer targets stay valid without reader-wide scans or historical repair."""
import numpy as np
import pytest
import torch
from etazero.data import trajectory_td_targets
from etazero.reader import BatchReader
from etazero.storage import sha256
from test_replay import id_snapshot


@pytest.mark.parametrize('area', [25, 121, 225, 361])
def test_td_convex_recurrence_preserves_boundary_probabilities(area):
    search = np.tile(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1],
                               [0, 1-1e-7, 1e-7]], np.float32), (40, 1))
    players = np.resize([1, -1], len(search))
    for terminal in np.eye(3):
        td = trajectory_td_targets(search, players, terminal, area)
        assert np.isfinite(td).all() and (td >= 0).all() and (td <= 1).all()
        np.testing.assert_allclose(td.sum(-1), 1, atol=1e-7)
        p = torch.from_numpy(td)
        assert torch.isfinite(-(p*(p+1e-30).log()).sum(-1)).all()


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_reader_resume_preserves_targets_without_scanning_or_repair(tmp_path, split):
    snapshot = id_snapshot(tmp_path, [8], validation=True)
    directory = snapshot/'data'
    if split == 'validation':
        directory /= 'validation'
    path = directory/'000.npz'
    with np.load(path) as file:
        arrays = {key: file[key] for key in file.files}
    # Deliberately invalid: the reader now only owns layout/cursor validation.
    # The learner's every-batch nonfinite-loss check handles this on consumption.
    arrays['td_value'][5] = [-5e-11, 1., 7e-9]
    np.savez(path, **arrays)
    digest = sha256(path)
    reader = BatchReader(snapshot, 4, 1, 17, split=split)
    try:
        reader.next()
        cursor = reader.state()
    finally:
        reader.close()
    resumed = BatchReader(snapshot, 4, 1, 99, cursor, split=split)
    try:
        batch = resumed.next()
        assert batch['globals'][:, 5].tolist() == [4, 5, 6, 7]
        np.testing.assert_array_equal(batch['td_value'], arrays['td_value'][4:8])
        assert resumed.state()['offset'] == 8
    finally:
        resumed.close()
    assert sha256(path) == digest


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'field'])
def test_reader_still_rejects_invalid_layout(tmp_path, bad):
    snapshot = id_snapshot(tmp_path, [8])
    path = snapshot/'data/000.npz'
    with np.load(path) as file:
        arrays = {k: file[k] for k in file.files}
    if bad == 'dtype':
        arrays['value'] = arrays['value'].astype(np.float64)
    elif bad == 'shape':
        arrays['value'] = arrays['value'][:, :2]
    else:
        del arrays['value']
    np.savez(path, **arrays)
    reader = BatchReader(snapshot, 4, 1, 17)
    try:
        with pytest.raises(ValueError, match='Invalid training'):
            reader.next()
    finally:
        reader.close()
