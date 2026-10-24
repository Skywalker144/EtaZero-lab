"""KataGo candidate/base decision with resumable independently initialized games."""
from dataclasses import asdict
from pathlib import Path
from threading import Event, Thread

from etazero.evaluation.arena import PairStore, Player, execute_pair, identity, native_config, save_manifest
from etazero.config import native_text
from etazero.engine_config import load_engine_config
from etazero.storage import load_json, save_json


def decision_from_points(points, played, total, required):
    """Source gatekeeper.cpp: clinch early; ties at the threshold accept."""
    remaining = total-played
    if remaining > 0:
        if points >= total*required:
            return True
        if points+remaining+1e-10 < total*required:
            return False
        return None
    return not (points+1e-10 < required*played)


def consider_candidate(root, config, binary, base, candidate, should_stop=lambda: False):
    root = Path(root)
    settings = config['gatekeeper']
    directory = root/'gatekeeper'/candidate['id']
    result_path = directory/'decision.json'
    identities = dict(candidate=candidate['id'], base=base['id'] if base else None)
    if result_path.exists():
        result = load_json(result_path)
        if any(result[k] != v for k, v in identities.items()):
            raise ValueError('Gatekeeper candidate/base identity changed')
        return result
    def finish(accepted, reason, played=0, points=0.0):
        result = dict(**identities, accepted=accepted, reason=reason, played=played,
                      candidate_points=points, settings=settings)
        save_json(result_path, result, immutable=True)
        return result
    if not settings['enabled']:
        return finish(True, 'gatekeeper_disabled')
    if base is None:
        return finish(True, 'first_network')
    if candidate['checkpoint']['iteration'] <= base['checkpoint']['iteration']:
        return finish(False, 'older_than_accepted_network')
    total, required = settings['games'], settings['required_candidate_win_prop']
    if total == 0:
        return finish(True, 'zero_games')
    profile = load_engine_config(settings['match_config'], match=True,
                                 overrides={'match': {'games': total}})
    players = tuple(Player(name, name, info['checkpoint']['iteration'], 0,
                           str(root/info['path'])) for name, info in (('base', base), ('candidate', candidate)))
    pair = dict(id=identity(('base', 'candidate')), players=['base', 'candidate'])
    manifest = dict(players=[asdict(p) for p in players], config=profile, games_per_pair=total,
                    anchor='base', pairs=[pair], gatekeeper=settings)
    native = native_config(profile, players)
    save_manifest(directory/'manifest.json', manifest)
    path = directory/'resolved.cfg'; path.write_text(native_text(native))
    store = PairStore(directory/'pairs'/pair['id'], total)
    records = store.games()
    def candidate_points(row):
        return (1-row['winner'])/2 if row['black_a'] else (1+row['winner'])/2
    points, played = sum(map(candidate_points, records)), len(records)
    decision = decision_from_points(points, played, total, required)
    if decision is None:
        cancel, done = Event(), Event()
        def watch():
            while not done.wait(0.1):
                if should_stop():
                    cancel.set(); return
        watcher = Thread(target=watch); watcher.start()
        def on_game(row):
            nonlocal points, played, decision
            points += candidate_points(row); played += 1
            decision = decision_from_points(points, played, total, required)
            return decision is not None
        try:
            seed = (profile['match']['seed']+int(pair['id'][:16],16)) % (1<<64)
            execute_pair(binary, path, players, store, seed, cancel=cancel, on_game=on_game)
        finally:
            done.set(); watcher.join()
    if decision is None:
        raise InterruptedError('Gatekeeper interrupted; completed games retained')
    return finish(decision, 'match', played, points)
