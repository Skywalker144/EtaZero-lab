"""Adaptive sampling and signed sample credit for complete-file epochs."""
import math
import random


def count_rows(directory):
    from shuffle import compute_num_rows
    total = 0
    for path in directory.glob('*.npz'):
        _, rows = compute_num_rows(path)
        if rows is None:
            raise ValueError(f'Cannot count committed replay shard: {path}')
        total += rows
    return total


def plan_sampling(config, replay, iteration):
    if iteration == 0:
        return dict(games=config.bootstrap_games, target_rows=0, rows_per_game=None)
    recent = replay['history'][-2:]
    games = sum(item['games'] for item in recent)
    average = sum(item['rows'] for item in recent) / games if games else 0
    if average <= 0:
        raise ValueError('Preceding selfplay rounds provide no valid rows/game estimate')
    target = (config.min_rows if iteration == 1 else
              replay['rows'] + config.samples_per_epoch / config.replay_ratio)
    deficit = max(0, target - replay['rows'])
    quota = math.ceil(deficit / average * (config.bootstrap_game_multiplier if iteration == 1 else 1))
    return dict(games=max(1, quota) if iteration >= 2 else quota,
                target_rows=target, rows_per_game=average)


def plan_training(config, replay, iteration, new_rows, snapshot, *, rng=None):
    # Per-file truncation matches both ordinary and expanded MuZero readers.
    from train import get_npz_num_rows
    cache = {}
    files = [(str(p.resolve()), get_npz_num_rows(str(p), cache) // config.batch_size)
             for p in sorted((snapshot / 'train').glob('*.npz'))]
    files = [(path, count) for path, count in files if count > 0]
    batches = sum(count for _, count in files)
    if batches == 0:
        raise ValueError('Shuffle snapshot has no complete training batch')
    # Bootstrap receives one baseline grant, not credit for all bootstrap rows.
    credit = (float(config.samples_per_epoch) if iteration == 1 else
              replay['credit'] + new_rows * config.replay_ratio)
    if not math.isfinite(credit):
        raise ValueError('Replay training credit is not finite')
    (rng if rng is not None else random.SystemRandom()).shuffle(files)
    target = max(0.0, credit) / config.batch_size
    selected, steps = [], 0
    for path, count in files:
        # Nearest prefix in random file order. Ties choose the shorter prefix;
        # zero files is valid, so small grants cannot force perpetual borrowing.
        if steps + count >= target:
            if abs(steps + count - target) < abs(steps - target):
                selected.append(path)
                steps += count
            break
        selected.append(path)
        steps += count
    return dict(train_steps=steps, snapshot_batches=batches, available_credit=credit,
                target_samples=max(0.0, credit), files=selected)


def commit_replay(replay, iteration, sampling, plan, consumed_samples):
    return dict(rows=replay['rows'] + sampling['rows'],
                credit=plan['available_credit'] - consumed_samples if iteration >= 1 else 0.0,
                history=[*replay['history'], sampling][-2:])
