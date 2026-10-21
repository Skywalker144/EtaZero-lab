"""Reuse raw paired matches for immutable model paths and match conditions."""
from .arena import PairStore, identity
from .storage import load_json, save_json


def pair_seed(manifest, pair):
    return (manifest['config']['match']['seed'] + int(pair['id'][:16], 16)) % (1 << 64)


def pair_identity(manifest, pair):
    players = {p['id']: p for p in manifest['players']}
    config = {group: dict(values) for group, values in manifest['config'].items()}
    config['match'].pop('games', None)
    # Published model paths are immutable. No weight-file hashing is needed.
    return {'models': [players[p]['model'] for p in pair['players']],
            'config': config, 'binary_sha256': manifest['binary_sha256'],
            'execution': manifest.get('execution', {}), 'seed': pair_seed(manifest, pair)}


def merge_pair(source, source_total, destination, destination_total, seed):
    """Keep the first recorded trial for each opening; never count reruns twice."""
    src, dst = PairStore(source, source_total), PairStore(destination, destination_total)
    source_records, destination_records = src.read(), dst.read()
    src.tasks(seed, source_records)  # Check seeds without rereading records.
    dst.tasks(seed, destination_records)
    openings, games = source_records
    destination_openings, destination_games = destination_records
    provenance_path = destination/'provenance.json'
    provenance = load_json(provenance_path) if provenance_path.exists() else {}
    source_provenance = (load_json(source/'provenance.json')
                         if (source/'provenance.json').exists() else {})
    source_path = str(source.resolve())
    changed = False
    for opening in openings.values():
        if opening['id'] >= destination_total // 2:
            continue
        # Independent historical reruns can have different stochastic outcomes.
        # Keep one trial per opening, with both colors coming from that trial.
        origin = provenance.get(str(opening['id']))
        source_origin = source_provenance.get(str(opening['id']), source_path)
        if opening['id'] in destination_openings and (destination_openings[opening['id']] != opening or
                               (origin is not None and origin != source_origin)):
            continue
        if opening['id'] not in destination_openings:
            dst.accept(opening)
            changed = True
        if origin is None:
            provenance[str(opening['id'])] = source_origin
            changed = True
        for index in (2*opening['id'], 2*opening['id']+1):
            if index not in games:
                continue
            game = games[index]
            if index in destination_games:
                # Resuming our own snapshot must agree; other historical trials
                # remain intact in their original sessions and are not merged.
                if destination_games[index] != game:
                    raise ValueError(f'Conflicting cached trial: {destination}/games/{index:08d}.json')
            else:
                dst.accept(game)
                changed = True
    if changed:
        save_json(provenance_path, provenance)


def cache_capacity(directory, minimum):
    ids = [int(p.stem) for p in (directory/'games').glob('*.json')]
    openings = [int(p.stem)*2+1 for p in (directory/'openings').glob('*.json')]
    maximum = max(ids + openings, default=-1) + 1
    return max(minimum, ((maximum+3)//4)*4)


def prepare_pairs(base, output, manifest):
    """Import matching historical trials, then hydrate this immutable schedule."""
    targets = {identity(pair_identity(manifest, pair)): pair for pair in manifest['pairs']}
    pool = base/'pair_cache'
    def complete(key):
        return all((pool/key/'games'/f'{i:08d}.json').is_file()
                   for i in range(manifest['games_per_pair']))
    for key, pair in targets.items():
        directory = pool/key
        payload = pair_identity(manifest, pair)
        metadata = directory/'identity.json'
        if metadata.exists():
            if load_json(metadata) != payload:
                raise ValueError(f'Cached match identity changed: {directory}')
        else:
            save_json(metadata, payload, immutable=True)
        # Preserve any progress in this session before considering older trials.
        source = output/'pairs'/pair['id']
        if source.exists():
            merge_pair(source, manifest['games_per_pair'], directory,
                       cache_capacity(directory, manifest['games_per_pair']), pair_seed(manifest, pair))
    # Complete cached pairs need no historical import. Stable ordering still
    # selects the first recorded trial when filling a missing pair or opening.
    needed = {key for key in targets if not complete(key)}
    for session in sorted(base.iterdir()) if needed else ():
        if not needed:
            break
        if session.is_symlink() or session == output or not (session/'manifest.json').is_file():
            continue
        old = load_json(session/'manifest.json')
        # These trials used content-derived inference seeds, not match.seed.
        if any('sha256' in player for player in old['players']):
            continue
        for pair in old['pairs']:
            key = identity(pair_identity(old, pair))
            if key not in needed:
                continue
            directory = pool/key
            merge_pair(session/'pairs'/pair['id'], old['games_per_pair'], directory,
                       cache_capacity(directory, old['games_per_pair']), pair_seed(old, pair))
            if complete(key):
                needed.remove(key)
    for key, pair in targets.items():
        directory = pool/key
        merge_pair(directory, cache_capacity(directory, manifest['games_per_pair']),
                   output/'pairs'/pair['id'], manifest['games_per_pair'], pair_seed(manifest, pair))
    return sum(len(PairStore(output/'pairs'/p['id'], manifest['games_per_pair']).games())
               for p in manifest['pairs'])


def retain_pairs(base, output, manifest):
    """Publish completed games even when the native schedule was interrupted."""
    for pair in manifest['pairs']:
        directory = base/'pair_cache'/identity(pair_identity(manifest, pair))
        merge_pair(output/'pairs'/pair['id'], manifest['games_per_pair'], directory,
                   cache_capacity(directory, manifest['games_per_pair']), pair_seed(manifest, pair))
