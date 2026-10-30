"""Outer replay credits; the source learner still consumes complete epochs/files."""
import math
import logging


def count_rows(directory):
    # Use the source header reader: no decompression of trajectory/target payloads.
    from shuffle import compute_num_rows
    total = 0
    for path in directory.glob('*.npz'):
        _, rows = compute_num_rows(path)
        if rows is None:
            raise ValueError(f'Cannot count committed replay shard: {path}')
        total += rows
    return total


def plan_games(config, history):
    if config.replay_ratio is None or not history:
        return config.games_per_round
    recent = history[-2:]
    rows_per_game = sum(item['rows'] for item in recent) / sum(item['games'] for item in recent)
    if rows_per_game <= 0:
        logging.warning('Recent selfplay rounds produced no rows; recalibrating with games_per_round games')
        return config.games_per_round
    target_rows = config.samples_per_epoch * config.epochs_per_round / config.replay_ratio
    return max(1, math.ceil(target_rows / rows_per_game))


def plan_training(config, replay, new_rows):
    if config.replay_ratio is None:
        return {'epochs': config.epochs_per_round}
    if replay['origin_rows'] is None:
        # Cold-start updates establish the first trained model, without issuing
        # ordinary replay credit for the bootstrap corpus.
        return {'epochs': config.epochs_per_round, 'available_credit': 0.0}
    available = replay['credit'] + new_rows * config.replay_ratio
    if not math.isfinite(available):
        raise ValueError('Replay credit is not finite; replay_ratio exceeds the accounting range')
    epochs = min(config.epochs_per_round, max(0, math.floor(available / config.samples_per_epoch)))
    return {'epochs': epochs, 'available_credit': available}


def commit_replay(replay, plan, sampling, consumed_samples):
    rows = replay['rows'] + sampling['rows']
    bootstrap = replay['origin_rows'] is None
    return {'rows': rows, 'origin_rows': rows if bootstrap else replay['origin_rows'],
            'credit': 0.0 if bootstrap else plan['available_credit'] - consumed_samples,
            'history': [*replay['history'], sampling][-2:]}
