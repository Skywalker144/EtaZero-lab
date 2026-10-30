"""Decode the native raw-NN text response for display, without recomputing heads."""
import math


def parse_raw_nn(text, state):
    """Read one symmetry, retaining native rounding and masked policy entries."""
    def value(token, *, policy=False):
        if policy and token == 'NAN':
            return None
        result = float(token)
        if not math.isfinite(result):
            raise ValueError('Nonfinite native raw-NN value')
        return result

    header, policy = text.split('\npolicy\n')
    lines = header.splitlines()
    label, symmetry = lines.pop(0).split()
    if label != 'symmetry':
        raise ValueError('Missing native raw-NN symmetry')
    values = {}
    for line in lines:
        key, token = line.split()
        values[key] = value(token)
    if not {'whiteWin', 'draw', 'whiteLoss', 'noResult'} <= values.keys():
        raise ValueError('Missing native raw-NN result values')

    rows = policy.splitlines()
    entries = []
    if state['game'] == 'xiangqi':
        for row in rows:
            move, token = row.split()
            entries.append(dict(move=move, probability=value(token, policy=True)))
        if len(entries) != 2086:
            raise ValueError('Incomplete native Xiangqi raw-NN policy')
    else:
        nx, ny = state['boardXSize'], state['boardYSize']
        if len(rows) != ny + 1:
            raise ValueError('Native raw-NN policy dimensions differ from the position')
        for y, row in enumerate(rows[:-1]):
            tokens = row.split()
            if len(tokens) != nx:
                raise ValueError('Native raw-NN policy row width differs from the position')
            for x, token in enumerate(tokens):
                columns = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'
                column = (columns[x // 25 - 1] if x >= 25 else '') + columns[x % 25]
                entries.append(dict(move=f'{column}{ny-y}',
                                    probability=value(token, policy=True)))
        label, token = rows[-1].split()
        if label != 'policyPass':
            raise ValueError('Missing native raw-NN pass entry')
        entries.append(dict(move='pass', probability=value(token, policy=True)))
    return dict(symmetry=int(symmetry), values=values, policy=entries)
