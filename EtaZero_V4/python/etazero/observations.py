"""Read native log records verbatim; plotting never loads checkpoints or NPZs."""
import json
import math
import re
import warnings


def learner_log(path):
    """Keep the source's logged normalization, cadence, precision and ordering."""
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        warnings.warn(f'Native metric log unavailable: {path}: {error}', RuntimeWarning)
        return []
    records = []
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('expected a metrics object')
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
    records = {key: [] for key in fields}
    # Read the native logger, independent of logToStdout. Failed attempts must
    # not silently turn a partial subset into whole-round measurements.
    for path in sorted(output.glob('log*.log')):
        for line in path.read_text().splitlines():
            for key, marker in fields.items():
                if marker in line:
                    value = (float if key == 'seconds' else int)(line.split(marker, 1)[1])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f'Invalid native selfplay {key}')
                    records[key].append(value)
    totals = {key: sum(values) for key, values in records.items() if attempts > 0 and len(values) == attempts}
    if not totals:
        raise ValueError(f'Selfplay native logs do not cover all {attempts} drained attempts')
    missing = fields.keys() - totals.keys()
    if missing:
        warnings.warn(f'Incomplete native selfplay logs for {attempts} attempts; leaving gaps: {sorted(missing)}', RuntimeWarning)
    # The source random evaluator does not increment NN rows/batches. Its cache
    # hits must not be presented as measured neural network inference.
    if totals.get('nn_rows', 0) <= 0 or totals.get('nn_batches', 0) <= 0:
        for key in ('nn_rows', 'nn_batches', 'cache_hits'):
            totals.pop(key, None)
    elif not muzero and 'cache_hits' in totals:
        totals['cache_hit_rate'] = totals['cache_hits'] / (totals['cache_hits'] + totals['nn_rows'])
    # MuZero logs total initial + recurrent rows, not initial rows separately.
    # No position-cache denominator, result counts or full game lengths can be
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
