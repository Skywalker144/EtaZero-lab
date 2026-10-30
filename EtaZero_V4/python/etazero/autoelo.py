"""Incremental historical Elo from immutable native pair-game prefixes."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from . import ROOT
from .arena import make_plan, match_overlay, plot_timeline
from .evaluation import estimate, load_games, run_match_batch, validate_fit_options, validate_model_window
from .run import save_json, snapshot_source


SELECTION = {'stride', 'sample_seconds', 'neighbors', 'cross_seconds', 'final_cross',
             'games_per_pair', 'anchor', 'include_first'}
FIT = {'prior_games', 'first_player_advantage'}
EXECUTION = {'max_loaded_models'}


def validate_options(options):
    validate_model_window(options.get('max_loaded_models'))
    for key in ('stride', 'neighbors', 'games_per_pair'):
        if key in options and (type(options[key]) is not int or options[key] < 1):
            raise ValueError(f'{key} must be a positive integer')
    for key in ('sample_seconds', 'cross_seconds'):
        if key in options and (type(options[key]) not in (int, float) or not math.isfinite(options[key]) or options[key] <= 0):
            raise ValueError(f'{key} must be finite and positive')
    for key in ('include_first', 'final_cross'):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if options.get('anchor') is not None and (not isinstance(options['anchor'], str) or not options['anchor']):
        raise ValueError('anchor must be a nonempty model ID')
    validate_fit_options(options.get('prior_games', 2.), options.get('first_player_advantage', False))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolved_pair(config, overlay, binary):
    """Use the source parser, including includes/quoting; never parse cfg in Python."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg') as stream:
        stream.write(overlay)
        stream.flush()
        command = [str(binary), 'match', '-config', str(config), '-config',
                   os.path.relpath(stream.name, config.parent), '-print-config']
        result = subprocess.run(command, cwd=config.parent, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f'Native match config failed to parse: {result.stderr or result.stdout}')
    return dict(working_directory=str(config.parent), **json.loads(result.stdout))


def autoelo_plan(base, data, config, *, binary=None, arms=None, **selection):
    validate_options(selection)
    base, config = Path(base).resolve(), Path(config).resolve()
    binary = Path(binary or ROOT / 'build/katago').resolve()
    if not config.is_file() or not binary.is_file():
        raise ValueError('Autoelo requires a source match config and native binary')
    anchor_path = base / 'anchor.json'
    pinned = json.loads(anchor_path.read_text()) if anchor_path.exists() else None
    if pinned:
        if selection.get('anchor') not in (None, pinned['id']):
            raise ValueError('Autoelo anchor is pinned; use a separate output root to change it')
        selection['anchor'] = pinned['id']
    plan = make_plan(data, arms=arms, **selection)
    indexed = {p['id']: p for p in plan['players']}
    anchor = indexed[plan['anchor']]
    record = dict(id=anchor['id'], model=anchor['model'])
    if pinned and pinned != record:
        raise ValueError('Pinned anchor model changed; use a separate output root')
    pairs = []
    for pair in plan['pairs']:
        pair_plan = dict(players=[indexed[name] for name in pair], pairs=[pair],
                         games_per_pair=plan['games_per_pair'])
        resolved = resolved_pair(config, match_overlay(pair_plan), binary)
        # Counts only select a prefix. All other actual native settings, model
        # paths and the resource cwd distinguish independently reusable games.
        key = identity({k: v for k, v in resolved.items() if k != 'resolved_config'})
        pairs.append(dict(players=pair, cache_key=key, conditions=resolved['extension_config']))
    plan.update(pairs=pairs, config=str(config), anchor_record=record)
    return plan


def collect_games(session, plan):
    players = plan['players']
    indices = {p['id']: i for i, p in enumerate(players)}
    games = []
    for pair in plan['pairs']:
        native, records = load_games(session / 'pairs' / pair['cache_key'], plan['games_per_pair'])
        if native.get('extension_config') != pair['conditions']:
            raise ValueError('Native pair conditions differ from the saved autoelo plan')
        if {bot['name'] for bot in native['bots']} != set(pair['players']):
            raise ValueError('Native pair participants differ from the saved autoelo plan')
        mapping = {i: indices[bot['name']] for i, bot in enumerate(native['bots'])}
        if any(bot['model'] != players[mapping[i]]['model'] or bot['muzero'] != players[mapping[i]]['muzero']
               for i, bot in enumerate(native['bots'])):
            raise ValueError('Native pair model differs from the saved autoelo plan')
        games.extend(dict(record, black=mapping[record['black']], white=mapping[record['white']]) for record in records)
    # Only the in-memory indices are remapped; native records remain unchanged.
    return dict(bots=[dict(name=p['id'], model=p['model']) for p in players]), games


def fit_autoelo(session, *, prior_games=2., first_player_advantage=False):
    session = Path(session).resolve()
    plan = json.loads((session / 'plan.json').read_text())
    native, games = collect_games(session, plan)
    result = estimate(native, games, prior_games=prior_games, first_player_advantage=first_player_advantage,
                      anchor=plan['anchor'])
    history = {p['id']: p for p in plan['players']}
    result['players'] = [dict(p, **{k: history[p['name']][k] for k in ('arm', 'iteration', 'seconds')})
                         for p in result['players']]
    save_json(session / 'elo.json', result)
    plot_timeline(plan, result, session)
    return result


def archive_prefix(source, destination, count):
    # Records are already atomically published by the native writer. Hardlinks
    # preserve their bytes and let old sessions survive later cache extensions.
    load_games(source, count)
    destination.mkdir(parents=True, exist_ok=True)
    plan = destination / 'plan.json'
    if not plan.exists():
        temporary = plan.with_suffix('.tmp')
        shutil.copyfile(source / 'plan.json', temporary)
        os.replace(temporary, plan)
    for index in range(count):
        target = destination / f'game-{index}.json'
        if not target.exists():
            os.link(source / target.name, target)


def run_autoelo(output, config=None, *, data=None, binary=None, arms=None, fit_only=False, dry_run=False,
                prior_games=2., first_player_advantage=False, inherited_fds=(), env=None, max_loaded_models=None, **selection):
    validate_fit_options(prior_games, first_player_advantage)
    validate_model_window(max_loaded_models)
    base = Path(output).resolve()
    if fit_only:
        if config is not None or data is not None or arms is not None or selection or dry_run or max_loaded_models is not None:
            raise ValueError('Autoelo fit-only uses a saved session without new inputs or selection')
        with (base / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fit_autoelo(base, prior_games=prior_games, first_player_advantage=first_player_advantage)
    if data is None or config is None:
        raise ValueError('Autoelo requires data and an independent match config')
    capacity = 2 if max_loaded_models is None else max_loaded_models
    if dry_run:
        return dict(autoelo_plan(base, data, config, binary=binary, arms=arms, **selection),
                    execution=dict(max_loaded_models=capacity))
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another autoelo controller or native match is using this output root') from error
        plan = autoelo_plan(base, data, config, binary=binary, arms=arms, **selection)
        session = base / 'sessions' / identity(plan)
        session.mkdir(parents=True, exist_ok=True)
        with (session / '.lock').open('a') as session_lock:
            fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not (base / 'anchor.json').exists():
                save_json(base / 'anchor.json', plan['anchor_record'])
            if not (session / 'plan.json').exists():
                snapshot_source(session / 'source.tar.gz.tmp')
                os.replace(session / 'source.tar.gz.tmp', session / 'source.tar.gz')
                save_json(session / 'plan.json', plan)
            save_json(session / 'status.json', dict(status='running'))
            try:
                indexed = {p['id']: p for p in plan['players']}
                tasks = []
                for pair in plan['pairs']:
                    cache = base / 'pair_cache' / pair['cache_key']
                    cache.mkdir(parents=True, exist_ok=True)
                    pair_plan = dict(players=[indexed[name] for name in pair['players']], pairs=[pair['players']],
                                     games_per_pair=plan['games_per_pair'])
                    extra = cache / 'participants.cfg'
                    # Budget may grow or shrink; source validates every other
                    # resolved setting and preserves all already published games.
                    temporary = extra.with_suffix('.tmp')
                    temporary.write_text(match_overlay(pair_plan))
                    os.replace(temporary, extra)
                    tasks.append((cache / 'match', extra))
                run_match_batch(tasks, plan['config'], session, binary=binary,
                                inherited_fds=(lock.fileno(), session_lock.fileno(), *inherited_fds), env=env,
                                source_snapshot=session / 'source.tar.gz', max_loaded_models=capacity)
                for pair, (match, _) in zip(plan['pairs'], tasks):
                    archive_prefix(match / 'results', session / 'pairs' / pair['cache_key'], plan['games_per_pair'])
                result = fit_autoelo(session, prior_games=prior_games, first_player_advantage=first_player_advantage)
                save_json(session / 'status.json', dict(status='complete', games=result['games']))
                temporary = base / ('.latest.' + uuid.uuid4().hex)
                try:
                    temporary.symlink_to(session.relative_to(base), target_is_directory=True)
                    os.replace(temporary, base / 'latest')
                finally:
                    temporary.unlink(missing_ok=True)
                print(f'Autoelo: {session / "elo.png"}', flush=True)
                return result
            except BaseException as error:
                save_json(session / 'status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                                        error=str(error)))
                raise
