"""Read native log records verbatim; plotting never loads checkpoints or NPZs."""
import json
import math
import re
import warnings


OUTCOMES = {
    'black_wins': 'Final black wins: ',
    'white_wins': 'Final white wins: ',
    'draws': 'Final draws: ',
    'no_results': 'Final no results: ',
    'truncated': 'Final truncated games: ',
}


def learner_display_scales(directory):
    """Read the structured record emitted after the learner resolves head weights."""
    settings = []
    for path in sorted(directory.glob('train*.log')):
        try:
            with path.open() as stream:
                for line in stream:
                    if line.startswith('learner_display '):
                        record = json.loads(line.removeprefix('learner_display '))
                        if record['batch_size'] <= 0 or not all(math.isfinite(x) for x in
                                (*record['loss'].values(), *record['heads'].values())):
                            raise ValueError('invalid learner display settings')
                        settings.append(record)
                        break
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            warnings.warn(f'Cannot read learner display scales at {path}: {error}', RuntimeWarning)
            return {}
    if not settings or any(item != settings[0] for item in settings[1:]):
        warnings.warn(f'Missing or conflicting learner display scales at {directory}; leaving overview gaps', RuntimeWarning)
        return {}
    return settings[0]


def learner_log(path):
    """Keep the source's logged normalization, cadence, precision and ordering."""
    from katago.train.metric_history import recovery_ranges, abandoned
    try:
        lines = path.read_bytes().splitlines(keepends=True)
        ranges = recovery_ranges(path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.warn(f'Native metric log unavailable: {path}: {error}', RuntimeWarning)
        return []
    records = []
    offset = 0
    cursor = 'nsamp_train' if path.name == 'metrics_val.json' else 'nsamp'
    for number, line in enumerate(lines, 1):
        position, offset = offset, offset + len(line)
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('expected a metrics object')
            if abandoned(position, record, ranges, cursor):
                continue
            for key, value in record.items():
                if type(value) in (int, float) and not math.isfinite(value):
                    warnings.warn(f'Nonfinite native metric {key} in {path}:{number}; leaving a gap', RuntimeWarning)
                    record[key] = None
            records.append(record)
        except (ValueError, TypeError) as error:
            warnings.warn(f'Unreadable native metric record {path}:{number}: {error}', RuntimeWarning)
    return records


def selfplay_metrics(output, attempts, *, muzero=False):
    fields = {
        'games': 'Final games finished: ',
        'moves_played': 'Final moves played: ',
        'rows': 'Final data rows: ',
        'nn_rows': 'Final NN rows: ',
        'nn_batches': 'Final NN batches: ',
        'cache_hits': 'Final NN cache hits: ',
        'seconds': 'Total selfplay runtime (seconds): ',
    }
    all_fields = {**fields, **OUTCOMES}
    records = {key: [] for key in all_fields}
    # Read the native logger, independent of logToStdout. Failed attempts must
    # not silently turn a partial subset into whole-round measurements.
    for path in sorted(output.glob('log*.log')):
        for line in path.read_text().splitlines():
            for key, marker in all_fields.items():
                if marker in line:
                    value = (float if key == 'seconds' else int)(line.split(marker, 1)[1])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f'Invalid native selfplay {key}')
                    records[key].append(value)
    totals = {key: sum(records[key]) for key in fields if attempts > 0 and len(records[key]) == attempts}
    if not totals:
        raise ValueError(f'Selfplay native logs do not cover all {attempts} drained attempts')
    missing = fields.keys() - totals.keys()
    if missing:
        warnings.warn(f'Incomplete native selfplay logs for {attempts} attempts; leaving gaps: {sorted(missing)}', RuntimeWarning)
    # Old logs have no outcomes. Partial attempts must not turn their observed
    # subset into a whole-round distribution, even if one category is complete.
    if any(records[key] for key in OUTCOMES):
        if 'games' in totals and all(len(records[key]) == attempts for key in OUTCOMES):
            if any(sum(records[key][i] for key in OUTCOMES) != games
                   for i, games in enumerate(records['games'])):
                warnings.warn('Native selfplay outcomes do not sum to completed games; leaving a gap', RuntimeWarning)
            else:
                totals.update({key: sum(records[key]) for key in OUTCOMES})
        else:
            warnings.warn('Incomplete native selfplay outcomes; leaving a gap', RuntimeWarning)
    # The source random evaluator does not increment NN rows/batches. Its cache
    # hits must not be presented as measured neural network inference.
    if totals.get('nn_rows', 0) <= 0 or totals.get('nn_batches', 0) <= 0:
        for key in ('nn_rows', 'nn_batches', 'cache_hits'):
            totals.pop(key, None)
    elif not muzero and 'cache_hits' in totals:
        totals['cache_hit_rate'] = totals['cache_hits'] / (totals['cache_hits'] + totals['nn_rows'])
    # MuZero logs total initial + recurrent rows, not initial rows separately.
    # No position-cache denominator or full game lengths can be
    # recovered from these logs; do not fill them from SGFs or training shards.
    return totals


def gatekeeper_metrics(directory):
    pattern = r'Candidate (?:won|lost) match, score ([\d.]+) to ([\d.]+) in (\d+) games,'
    results = []
    for path in sorted((directory / 'matches').glob('*/log*.log')):
        for line in path.read_text().splitlines():
            match = re.search(pattern, line)
            if match:
                results.append(dict(candidate_points=float(match[1]), baseline_points=float(match[2]), games=int(match[3])))
    if len(results) != 1:
        warnings.warn(f'Expected one final gatekeeper match in native logs at {directory}; leaving a gap', RuntimeWarning)
        return {}
    return results[0]
