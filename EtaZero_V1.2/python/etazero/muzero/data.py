"""Complete-trajectory unrolls; row multiplicity samples starts only.

Every view row is one start, with K+1 targets and K canvas actions. Future
real-row weights are normalized by the snapshot's mean main-row weight; roots
already reflect writer multiplicity. Absorbing actions are drawn by the reader's
checkpointed RNG, while absent side continuations are fully masked.
"""
import numpy as np

from ..data import metadata, alphazero_training_view, trajectory_td_targets, read_raw
from ..schema import PLANES, GLOBALS

TARGETS = ('policy', 'opponent_policy', 'opponent_policy_weight', 'value', 'td_value',
           'full_game_weight', 'q_values', 'q_visits')
EXTRA = ('actions', 'step_weights', 'sequence_mask', 'absorbing')


def validate_trajectory(a):
    m = metadata(a); t, area = m['plies'], m['canvas'] ** 2
    if type(m.get('unroll_steps')) is not int or not 1 <= m['unroll_steps'] <= 32:
        raise ValueError('Invalid MuZero unroll length')
    for name in ('trajectory_policy', 'trajectory_q_values', 'trajectory_q_visits'):
        if a[name].shape != (t, area) or a[name].dtype != np.int16:
            raise ValueError(f'Invalid MuZero {name} layout')
    train = a['train_mask'].astype(bool)
    if (a['trajectory_policy'] < 0).any() or (a['trajectory_q_visits'] < 0).any():
        raise ValueError('Negative MuZero policy/visit target')
    if not (a['trajectory_policy'][train].sum(1) > 0).all():
        raise ValueError('Missing MuZero per-step policy, including cheap rows')
    if any(a[key][~train].any() for key in ('trajectory_policy', 'trajectory_q_values', 'trajectory_q_visits')):
        raise ValueError('MuZero opening prefix has supervision')
    if not np.array_equal(a['policies'], a['trajectory_policy'][a['sample_indices']]):
        raise ValueError('MuZero root and trajectory policies disagree')
    if np.abs(a['trajectory_q_values'].astype(np.int32)).max(initial=0) > 32000:
        raise ValueError('MuZero Q target outside [-1,1]')


def training_view(a):
    root = alphazero_training_view(a)
    m = metadata(a); k = m['unroll_steps']; n = len(root['value']); area = m['canvas'] ** 2
    result = {key: root[key] for key in ('obs', 'globals')}
    for key in TARGETS:
        result[key] = np.zeros((n, k+1, *root[key].shape[1:]), dtype=np.float32)
        result[key][:, 0] = root[key]
    result['actions'] = np.zeros((n, k), np.int64)
    result['step_weights'] = np.zeros((n, k+1), np.float32)
    result['sequence_mask'] = np.zeros((n, k+1), np.float32)
    result['absorbing'] = np.zeros((n, k+1), np.uint8)
    result['step_weights'][:, 0] = result['sequence_mask'][:, 0] = 1
    # The writer stores complete games even when a shard contains only a span
    # of their repeated start rows. Construct the same ordering as AlphaZero.
    starts = []; td = {}
    for g, size in enumerate(a['sizes']):
        lo, hi = a['game_offsets'][g:g+2]; ol = a['observation_offsets'][g]
        starts.extend((g, int(t)) for t in np.repeat(np.arange(lo, hi), a['row_repeats'][lo:hi]))
        side_count = int(a['side_row_repeats'][a['side_game_indices'] == g].sum())
        starts.extend((g, -1) for _ in range(side_count))
        winner = a['winners'][g]
        terminal = np.array([winner == 1, winner == 0, winner == -1], np.float64)
        td[g] = trajectory_td_targets(a['search_wdl'][lo:hi], a['players'][ol:ol+hi-lo], terminal, int(size)**2)
    starts = starts[m['row_begin']:m['row_begin'] + n]
    if len(starts) != n:
        raise ValueError('MuZero start-row ordering mismatch')
    for row, (g, start) in enumerate(starts):
        # Normalizable masked targets also keep loss evaluation finite when a
        # side-only batch has no recurrent rows.
        mask = np.zeros(area, np.float32); size = int(a['sizes'][g]); canvas = m['canvas']
        mask.reshape(canvas, canvas)[:size, :size] = 1
        uniform = mask / mask.sum()
        result['policy'][row, 1:] = result['opponent_policy'][row, 1:] = uniform
        if start < 0:
            continue
        lo, hi = a['game_offsets'][g:g+2]; ol = a['observation_offsets'][g]
        player = int(a['players'][ol + start-lo]); winner = int(a['winners'][g])
        result['sequence_mask'][row] = 1
        for step in range(k+1):
            current = start + step
            if step < k:
                result['actions'][row, step] = a['actions'][current] if current < hi else -1
            if not step:
                continue
            outcome = winner * player * (-1 if step % 2 else 1)
            terminal = np.array([outcome == 1, outcome == 0, outcome == -1], np.float32)
            result['value'][row, step] = terminal
            if current >= hi:
                result['absorbing'][row, step] = 1
                result['step_weights'][row, step] = 1
                result['opponent_policy_weight'][row, step] = 1
                result['full_game_weight'][row, step] = 1
                result['td_value'][row, step] = terminal
                # No fabricated searched Q/visits in absorbing states.
                continue
            result['step_weights'][row, step] = a['target_weights'][current]
            result['policy'][row, step] = a['trajectory_policy'][current]
            result['q_values'][row, step] = a['trajectory_q_values'][current].astype(np.float32) / 32000
            result['q_visits'][row, step] = a['trajectory_q_visits'][current]
            gate = not a['reanalyzed'][current] or bool(a['reanalysis_used_outcome'][current])
            result['full_game_weight'][row, step] = gate
            result['td_value'][row, step] = td[g][current-lo]
            if current+1 < hi and gate:
                result['opponent_policy'][row, step] = a['trajectory_policy'][current+1]
                result['opponent_policy_weight'][row, step] = 1
    return result


def replay_weight_mean(root, sources):
    """Mean over distinct real trainable main positions in the selected window."""
    seen = set(); total = 0.; count = 0
    for entry in sources:
        a = read_raw(root / entry['path']); m = metadata(a)
        if m.get('algorithm') != 'muzero':
            raise ValueError('MuZero snapshot contains another algorithm')
        for g, game in enumerate(a['game_ids']):
            key = (m['run_id'], m['attempt_id'], m['worker_id'], int(game))
            if key in seen:
                continue
            seen.add(key); lo, hi = a['game_offsets'][g:g+2]
            valid = a['train_mask'][lo:hi].astype(bool)
            total += float(a['target_weights'][lo:hi][valid].sum(dtype=np.float64)); count += int(valid.sum())
    if count == 0 or total <= 0:
        raise ValueError('MuZero replay has no positive main-position weight')
    return total / count


def validate_view(arrays, rows, canvas, steps):
    shapes = {'obs': (rows, len(PLANES), (canvas*canvas+7)//8), 'globals': (rows, len(GLOBALS)),
              'actions': (rows, steps), 'step_weights': (rows, steps+1),
              'sequence_mask': (rows, steps+1), 'absorbing': (rows, steps+1)}
    for key in TARGETS:
        trailing = (canvas*canvas,) if key in ('policy', 'opponent_policy', 'q_values', 'q_visits') else (3,3) if key == 'td_value' else (3,) if key == 'value' else ()
        shapes[key] = (rows, steps+1, *trailing)
    if set(arrays) != set(shapes):
        raise ValueError('MuZero training-view fields mismatch')
    for key, shape in shapes.items():
        dtype = np.uint8 if key in ('obs', 'absorbing') else np.int64 if key == 'actions' else np.float32
        if arrays[key].shape != shape or arrays[key].dtype != dtype or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid MuZero training {key}')


def prepare_batch(batch, mean_weight, rng):
    batch['step_weights'] = batch['step_weights'].copy()
    batch['step_weights'][:, 1:] = np.where(batch['absorbing'][:, 1:], 1., batch['step_weights'][:, 1:] / mean_weight)
    batch['actions'] = batch['actions'].copy()
    for row in range(len(batch['actions'])):
        actions = np.flatnonzero(batch['obs'][row, 0].reshape(-1))
        for step in np.flatnonzero(batch['absorbing'][row, :-1]):
            batch['actions'][row, step] = actions[rng.randrange(len(actions))]
    del batch['absorbing']
    return batch
