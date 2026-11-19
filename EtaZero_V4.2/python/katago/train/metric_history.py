"""Checkpoint boundaries for filtering abandoned observations without rewriting logs."""
import json
from pathlib import Path
import warnings


def checkpoint_offsets(*streams):
    offsets = {}
    for stream in streams:
        try:
            stream.flush()
            offsets[Path(stream.name).name] = stream.tell()
        except (OSError, ValueError) as error:
            warnings.warn(f'Cannot checkpoint metric log position: {error}', RuntimeWarning)
    return offsets


def record_resume(directory, offsets, samples, *streams):
    """Record only abandoned suffixes, before any new observation is appended."""
    ranges = {}
    for stream in streams:
        name = Path(stream.name).name
        try:
            end = stream.tell()
            start = offsets.get(name) if offsets is not None else None
            if start is not None and start > end:
                warnings.warn(f'Metric log {name} is shorter than its checkpoint; history is incomplete', RuntimeWarning)
                start = None
            if end and start != end:
                ranges[name] = dict(start=start, end=end)
        except (OSError, ValueError) as error:
            warnings.warn(f'Cannot read metric log position: {error}', RuntimeWarning)
    if ranges:
        try:
            with (Path(directory) / 'metrics_recovery.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(samples=samples, logs=ranges)) + '\n')
        except OSError as error:
            warnings.warn(f'Cannot record abandoned metric log ranges: {error}', RuntimeWarning)


def recovery_ranges(path):
    history = path.parent / 'metrics_recovery.jsonl'
    if not history.exists():
        return []
    ranges = []
    for line in history.read_text().splitlines():
        record = json.loads(line)
        if path.name in record['logs']:
            interval = record['logs'][path.name]
            start, end = interval['start'], interval['end']
            if type(end) is not int or end < 0 or (start is not None and (type(start) is not int or not 0 <= start <= end)):
                raise ValueError('Invalid metric recovery log interval')
            ranges.append((start, end, record['samples']))
    return ranges


def abandoned(offset, record, ranges, cursor):
    for start, end, samples in ranges:
        if offset < end:
            if start is not None and offset >= start:
                return True
            # Existing checkpoints without byte positions can still identify
            # updates beyond their saved sample count. Validation at that count
            # may precede its checkpoint, so leave it unclaimed on this fallback.
            if start is None and (record.get(cursor, -1) > samples or
                                  cursor == 'nsamp_train' and record.get(cursor) == samples):
                return True
    return False
