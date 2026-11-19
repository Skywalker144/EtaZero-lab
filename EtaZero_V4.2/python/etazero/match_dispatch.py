"""Order native game IDs by model residency; no pairing or inference logic."""
from collections import defaultdict

REQUEST_PREFIX = 'ETAZERO_MATCH_DISPATCH '


def plan_windows(games, capacity):
    """games contains (saved ID, black service, white service) from native match."""
    pending = defaultdict(list)
    for index, first, second in games:
        pending[tuple(sorted((first, second)))].append(index)
    if capacity is None:
        return [sorted(index for ids in pending.values() for index in ids)] if pending else []
    if type(capacity) is not int or capacity < 2:
        raise ValueError('max_loaded_models must be an integer of at least 2')
    windows, resident = [], set()
    while pending:
        models, chosen = set(), []
        while True:
            candidates = [pair for pair in pending if len(models | set(pair)) <= capacity]
            if not candidates:
                break
            # Shared work first, then prior residents and the largest group.
            # The earliest saved ID breaks ties without altering source pairing.
            pair = max(candidates, key=lambda pair: (
                len(models & set(pair)), len(resident & set(pair)),
                len(pending[pair]), -pending[pair][0]))
            models.update(pair)
            chosen.extend(pending.pop(pair))
        windows.append(sorted(chosen))
        resident = models
    return windows
