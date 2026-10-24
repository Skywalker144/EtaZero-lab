"""Complete-trajectory unrolls; row multiplicity samples starts only.

Every view row is one start, with K+1 targets and K canvas actions. Future
real-row weights are normalized by the snapshot's mean main-row weight; roots
already reflect writer multiplicity. Absorbing actions are drawn by the reader's
checkpointed RNG, while absent side continuations are fully masked.
"""
from etazero.outcomes import terminal_wdl, REVERSE_OUTCOME
import math
import numpy as np

from etazero.data.targets import metadata, alphazero_training_view, full_trajectory_td, full_trajectory_auxiliary, TRAIN_TARGETS, TRAIN_POLICY_TARGETS
from etazero.schema import PLANES, GLOBALS, RULES
from etazero.games import spec_for_metadata, static_action_mask, SPECS

TARGETS = TRAIN_TARGETS
EXTRA = ('actions', 'step_weights', 'sequence_mask', 'absorbing')


def validate_trajectory(a):
    m = metadata(a); t, area = m['plies'], spec_for_metadata(m).actions(m['canvas'])
    if type(m.get('unroll_steps')) is not int or not m['unroll_steps'] >= 1:
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


def training_view(a, targets=TARGETS):
    td = full_trajectory_td(a) if 'td_value' in targets else None
    auxiliary = full_trajectory_auxiliary(a) if 'future_positions' in targets else None
    root = alphazero_training_view(a, targets, td=td, auxiliary=auxiliary)
    m = metadata(a); k = m['unroll_steps']; n = len(root['value']); area = spec_for_metadata(m).actions(m['canvas'])
    result = {key: root[key] for key in ('obs', 'globals')}
    for key in targets:
        result[key] = np.zeros((n, k+1, *root[key].shape[1:]),
                               dtype=np.int16 if key in TRAIN_POLICY_TARGETS else np.float32)
        result[key][:, 0] = root[key]
    result['actions'] = np.zeros((n, k), np.int64)
    result['step_weights'] = np.zeros((n, k+1), np.float32)
    result['sequence_mask'] = np.zeros((n, k+1), np.float32)
    result['absorbing'] = np.zeros((n, k+1), np.uint8)
    result['step_weights'][:, 0] = result['sequence_mask'][:, 0] = 1
    # The writer stores complete games even when a shard contains only a span
    # of their repeated start rows. Construct the same ordering as AlphaZero.
    starts = []
    for g in range(len(a['sizes'])):
        lo, hi = a['game_offsets'][g:g+2]
        starts.extend((g, int(t)) for t in np.repeat(np.arange(lo, hi), a['row_repeats'][lo:hi]))
        side_count = int(a['side_row_repeats'][a['side_game_indices'] == g].sum())
        starts.extend((g, -1) for _ in range(side_count))
    starts = starts[m['row_begin']:m['row_begin'] + n]
    if len(starts) != n:
        raise ValueError('MuZero start-row ordering mismatch')
    outcomes = terminal_wdl(a['winners'], a['reasons'])
    for row, (g, start) in enumerate(starts):
        if 'training_weight' in targets:
            result['training_weight'][row] = a['training_weights'][g]
        # Normalizable masked targets also keep loss evaluation finite when a
        # side-only batch has no recurrent rows.
        mask = static_action_mask(RULES[a['rules'][g]], int(a['sizes'][g]), m['canvas'])
        if RULES[a['rules'][g]]=='go':mask=np.concatenate((np.unpackbits(root['obs'][row,0],bitorder='big')[:m['canvas']**2],np.ones(1,np.uint8)))
        # Unit integer weights normalize to the same on-board uniform policy.
        result['policy'][row, 1:] = mask
        if 'opponent_policy' in targets:
            result['opponent_policy'][row, 1:] = mask
        if start < 0:
            continue
        lo, hi = a['game_offsets'][g:g+2]; ol = a['observation_offsets'][g]
        player = int(a['players'][ol + start-lo])
        result['sequence_mask'][row] = 1
        for key in ('policy_weight','value_weight','td_value_weight'):
            if key in targets:result[key][row,1:]=1
        for step in range(k+1):
            current = start + step
            if step < k:
                result['actions'][row, step] = a['actions'][current] if current < hi else -1
            if not step:
                continue
            perspective = player * (-1 if step % 2 else 1)
            terminal = outcomes[g] if perspective == 1 else outcomes[g, list(REVERSE_OUTCOME)]
            result['value'][row, step] = terminal
            if auxiliary is not None:
                result['future_positions'][row, step] = (auxiliary[0][current] if current < hi
                    else auxiliary[2][ol+hi-lo] * perspective)
                result['variance_time'][row, step] = auxiliary[1][current] if current < hi else 0
            gate = current >= hi or not a['reanalyzed'][current] or bool(a['reanalysis_used_outcome'][current])
            if 'future_position_weight' in targets:result['future_position_weight'][row,step]=gate
            if 'value_aux_weight' in targets:result['value_aux_weight'][row,step]=gate * (terminal[3] == 0)
            if current >= hi:
                result['absorbing'][row, step] = 1
                result['step_weights'][row, step] = 1
                if 'opponent_policy' in targets:
                    result['opponent_policy_weight'][row, step] = 1
                if 'td_value' in targets:
                    result['td_value'][row, step] = terminal
                # No fabricated searched Q/visits in absorbing states.
                continue
            result['step_weights'][row, step] = a['target_weights'][current]
            result['policy'][row, step] = a['trajectory_policy'][current]
            if 'q_values' in targets:
                result['q_values'][row, step] = a['trajectory_q_values'][current].astype(np.float32) / 32000
                result['q_visits'][row, step] = a['trajectory_q_visits'][current]
            if 'td_value' in targets:
                result['td_value'][row, step] = td[current]
            if 'opponent_policy' in targets and current+1 < hi and gate:
                result['opponent_policy'][row, step] = a['trajectory_policy'][current+1]
                result['opponent_policy_weight'][row, step] = 1
    return result


def replay_weight_mean(root, sources):
    """Mean over distinct real trainable main positions in the selected window."""
    seen = set(); total = 0.; count = 0
    for entry in sources:
        if entry['metadata'].get('algorithm') != 'muzero':
            raise ValueError('MuZero snapshot contains another algorithm')
        for game in entry['weight_stats']:
            if game['id'] in seen:
                continue
            seen.add(game['id'])
            total += game['sum']; count += game['count']
    if count <= 0 or not math.isfinite(total) or total <= 0:
        raise ValueError('MuZero replay has no positive main-position weight')
    return total / count


def validate_view(arrays, rows, canvas, steps, targets=TARGETS, game_metadata=None):
    spec = spec_for_metadata(game_metadata or {})
    shapes = {'obs': (rows, spec.planes, (canvas*canvas+7)//8), 'globals': (rows, spec.globals),
              'actions': (rows, steps), 'step_weights': (rows, steps+1),
              'sequence_mask': (rows, steps+1), 'absorbing': (rows, steps+1)}
    for key in targets:
        trailing = (spec.actions(canvas),) if key in ('policy', 'opponent_policy', 'q_values', 'q_visits') else (3,4) if key == 'td_value' else (4,) if key == 'value' else ()
        if key == 'future_positions': trailing = (2, canvas, canvas)
        shapes[key] = (rows, steps+1, *trailing)
    if set(arrays) != set(shapes):
        raise ValueError('MuZero training-view fields mismatch')
    for key, shape in shapes.items():
        dtype = (np.uint8 if key in ('obs', 'absorbing') else np.int64 if key == 'actions'
                 else np.int16 if key in TRAIN_POLICY_TARGETS else np.float32)
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise ValueError(f'Invalid MuZero training {key}')


def prepare_batch(batch, mean_weight, rng, spec=SPECS['gomoku']):
    batch['step_weights'] = batch['step_weights'].copy()
    batch['step_weights'][:, 1:] = np.where(batch['absorbing'][:, 1:], 1., batch['step_weights'][:, 1:] / mean_weight)
    batch['actions'] = batch['actions'].copy()
    absorbing = batch['absorbing'][:, :-1]
    for row in np.flatnonzero(absorbing.any(axis=1)):
        actions = (np.arange(spec.fixed_actions) if spec.action_layout == 'moves' else
                   np.flatnonzero(batch['obs'][row, 0].reshape(-1)))
        if spec.action_layout == 'points_pass': actions = np.r_[actions, batch['obs'].shape[-1]**2]
        for step in np.flatnonzero(absorbing[row]):
            batch['actions'][row, step] = actions[rng.randrange(len(actions))]
    del batch['absorbing']
    # The mask is known on CPU; avoid CUDA nonzero and its host synchronization
    # inside forward. Keep the original row order and exclude side rows from BN.
    batch['recurrent_indices'] = np.flatnonzero(batch.pop('sequence_mask')[:, 1]).astype(np.int64)
    return batch
