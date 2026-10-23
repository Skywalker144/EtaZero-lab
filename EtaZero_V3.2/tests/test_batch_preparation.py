"""Coordinate, random-stream and asynchronous upload invariants."""
from concurrent.futures import ThreadPoolExecutor
import copy
import os
import random
import threading

import numpy as np
import pytest
import torch

from config_samples import CONFIGS
from etazero.config import load_config
from etazero.muzero.data import prepare_batch
from etazero.network import make_network
from etazero.optimization import optimizer_for, optimization_for
from etazero.reader import CudaBatchPrefetcher
from etazero.schema import GLOBALS, HEX_GLOBAL, HEX_WHITE_GLOBAL
from etazero.symmetry import augment_batch


CUDA = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA acceptance')


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=CUDA)])
@pytest.mark.parametrize('symmetry', range(8))
def test_cached_coordinates_match_physical_positions(device, symmetry):
    n = 5
    globals = torch.zeros(3, len(GLOBALS), device=device)
    globals[1:, HEX_GLOBAL] = 1; globals[2, HEX_WHITE_GLOBAL] = 1
    grid = torch.arange(n*n, device=device).reshape(1, 1, n, n)
    obs = grid.expand(3, 5, -1, -1).float()
    target = grid.flatten(2).expand(3, 4, -1).float()
    actions = torch.tensor([[0, 1, 7, 24]] * 3, device=device)
    batch = dict(obs=obs, globals=globals, actions=actions, policy=target,
                 opponent_policy=target, q_values=target, q_visits=target)
    actual = augment_batch(batch, symmetry)
    for row in range(3):
        expected = torch.empty(n*n, device=device)
        forward = []
        for a in range(n*n):
            y, x = divmod(a, n)
            if row == 0:
                yy, xx = [(y,x),(n-1-x,y),(n-1-y,n-1-x),(x,n-1-y),
                          (x,y),(y,n-1-x),(n-1-x,n-1-y),(n-1-y,x)][symmetry]
            else:
                yy, xx = (n-1-y,n-1-x) if symmetry%2 else (y,x)
                if row == 2: yy, xx = xx, yy
            b = yy*n+xx
            expected[b] = a; forward.append(b)
        torch.testing.assert_close(actual['obs'][row], expected.reshape(1,n,n).expand(5,-1,-1), rtol=0, atol=0)
        for key in ('policy','opponent_policy','q_values','q_visits'):
            torch.testing.assert_close(actual[key][row], expected.expand(4,-1), rtol=0, atol=0)
        assert actual['actions'][row].tolist() == [forward[a] for a in [0,1,7,24]]
    assert actual['globals'] is globals
    assert all(actual[k].is_contiguous() for k in ('obs','policy','actions'))


@pytest.mark.parametrize('absorbing_mode', ['none', 'mixed', 'all'])
def test_muzero_preparation_preserves_draws_weights_and_side_order(absorbing_mode):
    obs = np.zeros((6,5,5,5), np.float32)
    obs[:,0,:3,:4] = 1; obs[::2,0,4,4] = 1
    absorbing = np.zeros((6,5), np.uint8)
    if absorbing_mode == 'all': absorbing[:] = 1
    if absorbing_mode == 'mixed': absorbing[1,2:] = 1; absorbing[4,1:] = 1
    mask = np.ones((6,5), np.float32); mask[[1,3],1:] = 0
    batch = dict(obs=obs, actions=np.zeros((6,4),np.int64),
                 step_weights=np.arange(30,dtype=np.float32).reshape(6,5)/30,
                 absorbing=absorbing, sequence_mask=mask)
    original = copy.deepcopy(batch)
    expected = batch['actions'].copy(); reference_rng = random.Random(91)
    # Historical row-major draws, including the same nonrectangular masks.
    for row in range(6):
        legal = np.flatnonzero(obs[row,0].reshape(-1))
        for step in np.flatnonzero(absorbing[row,:-1]):
            expected[row,step] = legal[reference_rng.randrange(len(legal))]
    rng = random.Random(91)
    result = prepare_batch(dict(batch), .375, rng)
    np.testing.assert_array_equal(result['actions'], expected)
    assert rng.getstate() == reference_rng.getstate()
    np.testing.assert_array_equal(result['recurrent_indices'], [0,2,4,5])
    assert result['recurrent_indices'].dtype == np.int64
    np.testing.assert_array_equal(result['step_weights'][:,0], original['step_weights'][:,0])
    np.testing.assert_array_equal(result['step_weights'][:,1:],
                                  np.where(absorbing[:,1:], 1., original['step_weights'][:,1:]/.375))
    for key in batch: np.testing.assert_array_equal(batch[key], original[key])


@pytest.mark.parametrize('algorithm', ['alphazero', 'muzero'])
def test_optimizer_defers_norms_to_schedule_initialization(monkeypatch, algorithm):
    from test_muzero_pipeline import small_config
    import etazero.optimization as optimization
    c = small_config() if algorithm == 'muzero' else load_config(CONFIGS/'smoke_test')
    model = make_network(c)
    with monkeypatch.context() as patch:
        patch.setattr(optimization, 'model_norms', lambda *a: pytest.fail('Unused optimizer-init reduction'))
        optimizer = optimizer_for(model, c)
    control = optimization_for(model, c, optimizer)
    assert control.baselines and all(v > 0 for v in control.baselines.values())
    before = [(g['lr'],g['weight_decay']) for g in optimizer.param_groups]
    control.configure()
    assert [(g['lr'],g['weight_decay']) for g in optimizer.param_groups] == before


@CUDA
def test_cuda_prefetch_returns_current_while_next_cpu_batch_is_blocked():
    torch.cuda.init()
    entered, release = threading.Event(), threading.Event()
    class Reader:
        calls = 0
        def next(self):
            i = self.calls; self.calls += 1
            if i == 1:
                entered.set()
                if not release.wait(15): raise TimeoutError('Test did not release next CPU batch')
            if i == 6: raise ValueError('producer failure')
            return {'obs':np.full((8,5,6,6), i, np.float32),
                    'recurrent_indices':np.arange(i%3, dtype=np.int64)}, {'offset':i+1}
    reader = Reader(); prefetch = CudaBatchPrefetcher(reader, torch.device('cuda:0'))
    caller = ThreadPoolExecutor(max_workers=1)
    try:
        future = caller.submit(prefetch.next)
        assert entered.wait(10)
        # This must finish BEFORE the following batch becomes available.
        batches = [future.result(timeout=2)]
        release.set()
        for _ in range(5): batches.append(prefetch.next())
        with pytest.raises(ValueError, match='producer failure'): prefetch.next()
        # Keep earlier GPU tensors alive across pinned-slot reuse and changing
        # recurrent counts; neither their contents nor cursors may be overwritten.
        for i,(tensors,cursor) in enumerate(batches):
            assert cursor == {'offset':i+1}
            assert tensors['obs'].cpu().eq(i).all()
            assert tensors['recurrent_indices'].dtype == torch.int64
            assert tensors['recurrent_indices'].cpu().tolist() == list(range(i%3))
    finally:
        release.set(); caller.shutdown(wait=True); prefetch.close()


@CUDA
def test_cuda_prefetch_final_batch_does_not_request_more():
    class Reader:
        calls = 0
        def next(self):
            self.calls += 1
            if self.calls != 1: raise AssertionError('Read past final batch')
            return {'obs': np.ones((2,5,6,6), np.float32)}, {'offset':2}
    reader = Reader(); prefetch = CudaBatchPrefetcher(reader, torch.device('cuda:0'))
    try:
        tensors,cursor = prefetch.next(prefetch_next=False)
        assert tensors['obs'].cpu().eq(1).all() and cursor == {'offset':2}
    finally: prefetch.close()
    assert reader.calls == 1
