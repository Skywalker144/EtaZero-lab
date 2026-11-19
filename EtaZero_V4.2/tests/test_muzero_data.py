"""Source writer roots, complete continuations, and actual unroll updates."""
import os
import subprocess
import json
import hashlib
from collections import Counter

import numpy as np
import pytest
import torch

from etazero import ROOT
from katago.train.muzero_data import (
    TARGETS, REAL, TERMINAL, TRUNCATED, trajectory_key, weight_statistics,
    replay_weight_mean, expand_training_rows, prepare_batch,
    read_npz_muzero_data,
)
from katago.train.muzero_training import unroll_loss
from test_muzero_model import make_model, metric_options


@pytest.fixture(scope='module')
def raw_data(tmp_path_factory):
    path = tmp_path_factory.mktemp('native-muzero-data')
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-muzero-data', str(path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return path


def load(path):
    with np.load(path) as data:
        return dict(data)


def files(path, case):
    return sorted((path / str(case) / 'mu').glob('*.npz'))


def combined_view(path, case, steps):
    views = [expand_training_rows(load(p), steps) for p in files(path, case)]
    return {key: np.concatenate([view[key] for view in views]) for key in views[0]}


@pytest.mark.parametrize('case', range(10))
def test_native_sampler_unchanged_and_complete_trajectory(raw_data, case):
    mu_files = files(raw_data, case)
    assert {p.name for p in mu_files} == {p.name for p in (raw_data / str(case) / 'az').glob('*.npz')}
    seen = {}
    all_root_turns = []
    for path in mu_files:
        data = load(path)
        ordinary = load(path.parent.parent / 'az' / path.name)
        # Same names and exact contents prove the full pass draws no source RNG,
        # nor changes first-file random size, duplicate roots, or side ordering.
        for key in ordinary:
            np.testing.assert_array_equal(data[key], ordinary[key], err_msg=key)
        assert 1 <= len(ordinary['globalInputNC']) <= 4
        offsets = data['muZeroGameOffsets']
        for game, identity in enumerate(data['muZeroGameHash']):
            lo, hi = offsets[game:game+2]
            identity = tuple(identity)
            target_arrays = [data[key][lo:hi] for key in
                             ('muZeroActions', 'muZeroWeights', *(trajectory_key(k) for k in TARGETS))]
            if identity in seen:
                for actual, expected in zip(target_arrays, seen[identity]):
                    np.testing.assert_array_equal(actual, expected)
            seen[identity] = target_arrays
            assert data['muZeroActions'][hi-1] == -1
            assert data['muZeroWeights'][lo+1] == .125
            # Reanalyzed, rounded-away step retains its full current policy/TD/Q.
            assert data['muZeroGlobalTargetsNC'][lo+1, 64] == 1
            assert data['muZeroGlobalTargetsNC'][lo+1, 26] == 1
            assert data['muZeroGlobalTargetsNC'][lo+1, 28] == 0
            assert data['muZeroGlobalTargetsNC'][lo+1, 33] == 0
            assert data['muZeroQValueTargetsNCMove'][lo+1, 1].sum() == 24
        for row, (game, turn) in enumerate(data['muZeroRootNC']):
            all_root_turns.append(int(turn))
            if turn < 0:
                continue
            index = offsets[game] + turn
            for key in TARGETS:
                expected = data[trajectory_key(key)][index]
                actual = data[key][row]
                if key == 'globalTargetsNC':
                    # History masks are independent per source-sampled root.
                    channels = np.r_[0:36, 41:80]
                    np.testing.assert_array_equal(actual[channels], expected[channels])
                elif key == 'qValueTargetsNCMove':
                    assert np.max(np.abs(actual.astype(int)-expected.astype(int))) <= 1
                else:
                    np.testing.assert_array_equal(actual, expected)
    assert len(seen) == 2
    assert 1 not in all_root_turns and -1 in all_root_turns
    assert all_root_turns.count(0) >= 4
    stats = {identity: (total, count) for path in mu_files for identity, total, count in weight_statistics(load(path))}
    expected_mean = sum(v[0] for v in stats.values()) / sum(v[1] for v in stats.values())
    assert replay_weight_mean(mu_files) == expected_mean
    assert replay_weight_mean(mu_files * 2) == expected_mean


@pytest.mark.parametrize('case', range(10))
def test_complete_td_matches_geometric_reference(raw_data, case):
    data = load(files(raw_data, case)[0])
    lo, hi = data['muZeroGameOffsets'][:2]
    values = data['muZeroGlobalTargetsNC'][lo:hi]
    t = len(values)-1
    # Native fixture stores white-perspective search probabilities; starts White.
    white = np.array([[.1+.02*i, .2, .7-.02*i, 0.] for i in range(t)])
    last = values[-1, [0,70,1,2]].astype(float)
    if t % 2:
        last = last[[2,1,0,3]]
    white = np.concatenate((white, last[None]))
    for turn in range(t):
        for factor, columns in ((.176, [4,71,5,6]), (.056, [8,72,9,10]), (.016, [12,73,13,14])):
            alpha = 1/(1+(90 if case >= 7 else 9)*factor)
            powers = np.arange(t-turn)
            expected = (white[turn:t] * (alpha*(1-alpha)**powers)[:, None]).sum(axis=0)
            expected += (1-alpha)**(t-turn)*white[-1]
            if turn % 2:
                expected = expected[[2,1,0,3]]
            np.testing.assert_allclose(values[turn, columns], expected, atol=1e-7, rtol=1e-6)
        wl = white[:, 0] - white[:, 2]
        variance = sum((i-turn)*(wl[i]-wl[i-1])**2 for i in range(turn+1,t+1))
        assert values[turn, 22] == pytest.approx(variance, rel=2e-7, abs=3e-7)


@pytest.mark.parametrize('case', range(7))
def test_unroll_continuity_endpoint_and_side_semantics(raw_data, case):
    for path in files(raw_data, case):
        data = load(path)
        view = expand_training_rows(data, 12)
        for key in ('binaryInputNCHWPacked', 'globalInputNC', *TARGETS):
            np.testing.assert_array_equal(view[key], data[key])
        for row, (game, start) in enumerate(data['muZeroRootNC']):
            kind = view['muZeroStepKind'][row]
            if start < 0:
                np.testing.assert_array_equal(kind, [REAL]+[0]*12)
                assert not view['muZeroWeights'][row, 1:].any()
                continue
            lo, hi = data['muZeroGameOffsets'][game:game+2]
            end = hi-lo-1
            for step in range(1,13):
                turn = start + step
                if turn < end:
                    assert kind[step] == REAL
                    for key in TARGETS:
                        np.testing.assert_array_equal(view[trajectory_key(key)][row, step-1],
                                                      data[trajectory_key(key)][lo+turn])
                    assert view['muZeroWeights'][row, step] == data['muZeroWeights'][lo+turn]
                elif data['muZeroGameFinished'][game]:
                    assert kind[step] == TERMINAL
                    target = view['muZeroGlobalTargetsNC'][row, step-1]
                    expected = data['muZeroGlobalTargetsNC'][hi-1, [0,70,1,2]]
                    if (turn-end) % 2:
                        expected = expected[[2,1,0,3]]
                    for columns in ([0,70,1,2], [4,71,5,6], [8,72,9,10], [12,73,13,14]):
                        np.testing.assert_array_equal(target[columns], expected)
                    assert target[26] == target[28] == 1
                    assert target[22] == 0
                    assert view['muZeroWeights'][row, step] == 1
                    assert not view['muZeroQValueTargetsNCMove'][row, step-1].any()
                elif turn == end:
                    assert kind[step] == TRUNCATED
                    target = view['muZeroGlobalTargetsNC'][row, step-1]
                    assert target[35] == 1 and target[24] == 0  # TD bootstrap only.
                    assert target[[26,27,28,33]].sum() == 0
                else:
                    assert kind[step] == 0


class FixedRng:
    def __init__(self, symmetry):
        self.symmetry = symmetry
        self.domains = []

    def integers(self, low, high):
        assert (low, high) == (0, 8)
        return self.symmetry

    def choice(self, domain, size):
        self.domains.append(domain.copy())
        return np.resize(domain, size)


def numpy_symmetry(x, sym):
    # Independent spatial reference in NumPy, using the documented source IDs.
    return np.rot90(x, sym, axes=(-2,-1)) if sym < 4 else np.rot90(x.swapaxes(-2,-1), 4-sym, axes=(-2,-1))


@pytest.mark.parametrize('case,game', [(0,'go'), (3,'renju'), (5,'hex')])
@pytest.mark.parametrize('symmetry', range(8))
def test_batch_weights_fixed_orientation_and_absorbing_domain(raw_data, case, game, symmetry):
    view = combined_view(raw_data, case, 10)
    model = make_model(game)
    rng = FixedRng(symmetry)
    root, targets, actions, active = prepare_batch(view, 2., rng, model.prediction.config, 'cpu')
    original = np.unpackbits(view['binaryInputNCHWPacked'], axis=-1)[..., :25].reshape(-1, view['binaryInputNCHWPacked'].shape[1], 5, 5)
    chosen = 2*(symmetry % 2) if game == 'hex' else symmetry
    np.testing.assert_array_equal(root['binaryInputNCHW'][:, 0], numpy_symmetry(original[:, 0], chosen))
    for domain in rng.domains:
        expected = [0,1,2,5,6,7,10,11,12] + ([25] if game == 'go' else [])
        np.testing.assert_array_equal(domain, expected)
    grid = np.arange(25).reshape(5,5)
    reverse = numpy_symmetry(grid, chosen).ravel()
    for row in range(len(actions)):
        for step in range(actions.shape[1]):
            if view['muZeroStepKind'][row, step] == REAL and active[row, step+1]:
                actual = int(actions[row, step])
                assert (25 if actual == 25 else reverse[actual]) == view['muZeroActions'][row, step]
    np.testing.assert_array_equal(active, view['muZeroStepKind'] != 0)
    for step, target in enumerate(targets):
        raw = view if step == 0 else {key: view[trajectory_key(key)][:, step-1] for key in TARGETS}
        policy = raw['policyTargetsNCMove']
        expected = numpy_symmetry(policy[..., :25].reshape(-1, 2, 5, 5), chosen).reshape(-1,2,25)
        if game == 'go':
            expected = np.concatenate((expected, policy[..., -1:]), axis=-1)
        np.testing.assert_array_equal(target['policyTargetsNCMove'], expected)
        np.testing.assert_array_equal(target['valueTargetsNCHW'], numpy_symmetry(raw['valueTargetsNCHW'], chosen))
        weights = np.ones(len(actions)) if step == 0 else view['muZeroWeights'][:, step] / np.where(view['muZeroStepKind'][:, step] == REAL, 2., 1.)
        np.testing.assert_allclose(target['globalTargetsNC'][:, 25], .75 * weights)


def test_replay_rng_is_owned_by_caller_and_restorable(raw_data):
    view = combined_view(raw_data, 3, 10)
    config = make_model().prediction.config
    first = np.random.default_rng(808)
    saved = first.bit_generator.state
    a = prepare_batch(view, 1., first, config, 'cpu')
    restored = np.random.default_rng()
    restored.bit_generator.state = saved
    b = prepare_batch(view, 1., restored, config, 'cpu')
    for left, right in zip(a[0].values(), b[0].values()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
    assert first.bit_generator.state == restored.bit_generator.state


@pytest.mark.parametrize('case,game', [(0, 'go'), (3, 'renju'), (7, 'xiangqi')])
def test_ordinary_reader_augmentation_rng_is_restorable(raw_data, case, game):
    from katago.train import checkpoint_runtime
    from katago.train.data_processing_pytorch import read_npz_training_data
    paths = sorted((raw_data / str(case) / 'az').glob('*.npz'))
    config = make_model(game).prediction.config
    canvas = load(paths[0])['valueTargetsNCHW'].shape[-1]
    rng = np.random.default_rng(872)
    validation_rng = np.random.default_rng(873)
    saved = checkpoint_runtime.capture_runtime(rng, validation_rng, None, torch.device('cpu'))

    def read():
        return list(read_npz_training_data(paths, 2, 1, 0, canvas, 'cpu', True, False, config, rng=rng))

    expected = read()
    advanced = rng.bit_generator.state
    assert expected and saved['train_rng'] != advanced
    # The source Go history dropout uses Torch, while symmetry uses NumPy.
    checkpoint_runtime.restore_runtime(saved, rng, validation_rng, None, torch.device('cpu'))
    actual = read()
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert left.keys() == right.keys()
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    assert rng.bit_generator.state == advanced


@pytest.mark.parametrize('case,game', [(0,'go'), (3,'renju'), (4,'renju'), (5,'hex'),
                                      (7,'xiangqi'), (8,'xiangqi'), (9,'xiangqi')])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_data_to_source_unroll_update(raw_data, tmp_path, case, game, device):
    if device == 'cuda' and os.environ.get('ETAZERO_GPU_TESTS') != '1':
        pytest.skip('requires real CUDA')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(7)
        canvas = load(files(raw_data, case)[0])['valueTargetsNCHW'].shape[-1]
        model = make_model(game, pos_len=canvas).to(device).train()
        output = shuffle_source(raw_data / str(case) / 'mu', tmp_path, steps=4, rows_per_file=1000, files_per_bucket=1)
        metadata = json.loads(output.with_suffix('.json').read_text())
        shuffled_files = sorted(output.glob('*.npz'))
        assert len(shuffled_files) == 1
        batch_size = len(load(shuffled_files[0])['globalInputNC'])
        batches = list(read_npz_muzero_data(
            shuffled_files, batch_size, canvas, device, model.prediction.config,
            unroll_steps=metadata['muzero_unroll_steps'], mean_weight=metadata['muzero_weight_mean'],
            rng=np.random.default_rng(12)))
        assert len(batches) == 1
        root, targets, actions, active = batches[0]
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        scaler = torch.amp.GradScaler(device, init_scale=1., enabled=device == 'cuda')
        before = {key: value.detach().clone() for key, value in model.named_parameters()}
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
            loss, logs = unroll_loss(model, root, targets, actions, active_steps=active, **metric_options())
        assert torch.isfinite(loss) and len(logs) >= 2
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for component in (model.representation, model.dynamics, model.prediction):
            grads = [p.grad for p in component.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum().item() for g in grads) > 0
        scaler.step(optimizer)
        scaler.update()
        for prefix in ('representation.', 'dynamics.', 'prediction.'):
            assert any(not torch.equal(p, before[name]) for name, p in model.named_parameters() if name.startswith(prefix))
    finally:
        torch.set_num_threads(previous)


def test_invalid_trajectory_boundaries_actions_and_weights(raw_data):
    original = load(files(raw_data, 3)[0])
    for field, mutate, message in (
        ('muZeroGameOffsets', lambda a: a.__setitem__(1, 0), 'boundaries'),
        ('muZeroActions', lambda a: a.__setitem__(0, 25), 'canvas'),
        ('muZeroWeights', lambda a: a.__setitem__(0, np.nan), 'finite'),
        ('muZeroRootNC', lambda a: a.__setitem__((0,0), 100), 'absent game'),
        ('muZeroRootNC', lambda a: a.__setitem__((0,1), 100), 'absent main'),
    ):
        data = {k: v.copy() for k,v in original.items()}
        mutate(data[field])
        with pytest.raises(ValueError, match=message):
            expand_training_rows(data, 3)


def row_fingerprints(data):
    # Compare whole rows, including every future tensor; catches permutations
    # that accidentally detach targets/actions from their sampled roots.
    return Counter(hashlib.sha256(b''.join(data[key][row].tobytes() for key in sorted(data))).hexdigest()
                   for row in range(len(data['globalInputNC'])))


def shuffle_source(source, tmp_path, *, steps=5, waves=1, keep='all', exclude_q=False, rows_per_file=4, files_per_bucket=2):
    output = tmp_path / 'shuffled'
    command = [str(ROOT / 'python/shuffle.py'), str(source), '-out-dir', str(output),
               '-out-tmp-dir', str(tmp_path / 'scratch'), '-min-rows', '1',
               '-keep-target-rows', keep, '-num-processes', '2', '-worker-group-size', '5',
               '-approx-rows-per-out-file', str(rows_per_file), '-approx-rows-per-bucket', str(files_per_bucket*rows_per_file),
               '-num-waves', str(waves), '-muzero-unroll-steps', str(steps)]
    if exclude_q:
        command.append('-exclude-qvalues')
    import sys
    result = subprocess.run([sys.executable, *command], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


@pytest.mark.parametrize('waves,keep,exclude_q', [(1,'all',False), (3,'all',False), (2,'10',True)])
@pytest.mark.parametrize('case', [3, 7, 9])
def test_source_shuffle_preserves_complete_unroll_rows(raw_data, tmp_path, waves, keep, exclude_q, case):
    source = raw_data / str(case) / 'mu'
    output = shuffle_source(source, tmp_path, waves=waves, keep=keep, exclude_q=exclude_q)
    expected = combined_view(raw_data, case, 5)
    if exclude_q:
        del expected['qValueTargetsNCMove']
        del expected['muZeroQValueTargetsNCMove']
    expected_rows = row_fingerprints(expected)
    actual_rows = Counter()
    for file in output.glob('*.npz'):
        actual_rows += row_fingerprints(load(file))
    if keep == 'all':
        assert actual_rows == expected_rows
    else:
        assert 0 < sum(actual_rows.values()) < sum(expected_rows.values())
        assert not (actual_rows - expected_rows)
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['muzero_unroll_steps'] == 5
    assert metadata['muzero_weight_mean'] == replay_weight_mean(files(raw_data, case))
    # Excluding Q keeps a valid consumer layout for a prediction without that head.
    if exclude_q:
        config = make_model('xiangqi' if case >= 7 else 'renju').prediction.config.copy()
        config['predict_q_values'] = False
        nonempty = next(load(p) for p in output.glob('*.npz') if len(load(p)['globalInputNC']))
        _, targets, _, _ = prepare_batch(nonempty, metadata['muzero_weight_mean'], np.random.default_rng(5), config, 'cpu')
        assert all('qValueTargetsNCMove' not in t for t in targets)


def test_ordered_prefetch_batch_tails_rng_and_algorithm_boundary(raw_data, tmp_path):
    from katago.train.data_processing_pytorch import read_npz_training_data
    view = combined_view(raw_data, 3, 5)
    config = make_model().prediction.config
    # Different lengths, including a file too short for one batch.
    paths = []
    cursor = 0
    for index, length in enumerate((5,1,7)):
        path = tmp_path / f'{index}.npz'
        np.savez_compressed(path, **{k:v[cursor:cursor+length] for k,v in view.items()})
        paths.append(path)
        cursor += length
    reference_rng = np.random.default_rng(44)
    expected = []
    for path in paths:
        data = load(path)
        for start in range(0,len(data['globalInputNC'])//2*2,2):
            expected.append(prepare_batch({k:v[start:start+2] for k,v in data.items()}, 2., reference_rng, config, 'cpu'))
    for depth in (1,3):
        rng = np.random.default_rng(44)
        actual = list(read_npz_muzero_data(paths, 2, 5, 'cpu', config, unroll_steps=5,
                                         mean_weight=2., rng=rng, prefetch_depth=depth))
        assert len(actual) == len(expected) == 5
        for a,b in zip(actual,expected):
            for x,y in zip(a[0].values(),b[0].values()):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            for step_a,step_b in zip(a[1],b[1]):
                for key in step_a:
                    torch.testing.assert_close(step_a[key],step_b[key],rtol=0,atol=0)
            torch.testing.assert_close(a[2],b[2],rtol=0,atol=0)
            torch.testing.assert_close(a[3],b[3],rtol=0,atol=0)
        assert rng.bit_generator.state == reference_rng.bit_generator.state
    with pytest.raises(ValueError, match='trajectory reader'):
        list(read_npz_training_data(paths,2,1,0,5,'cpu',False,False,config))
    with pytest.raises(ValueError, match='shuffled unroll view'):
        list(read_npz_muzero_data(files(raw_data,3),2,5,'cpu',config,unroll_steps=5,
                                 mean_weight=2.,rng=np.random.default_rng(44)))


def test_version_16_implicit_q_prediction(raw_data, tmp_path):
    view = combined_view(raw_data, 0, 3)
    config = make_model('go').prediction.config.copy()
    config['version'] = 16
    config.pop('predict_q_values')
    path = tmp_path / 'view.npz'
    np.savez_compressed(path, **view)
    batches = list(read_npz_muzero_data([path], 2, 5, 'cpu', config, unroll_steps=3,
                                       mean_weight=1., rng=np.random.default_rng(6)))
    assert batches
    assert all('qValueTargetsNCMove' in step for batch in batches for step in batch[1])
    del view['qValueTargetsNCMove']
    with pytest.raises(ValueError, match='requires Q targets'):
        prepare_batch(view,1.,np.random.default_rng(6),config,'cpu')
