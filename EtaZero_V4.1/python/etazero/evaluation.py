"""Source match orchestration and Elo fitting from committed, color-aware games."""
from contextlib import contextmanager, ExitStack
from collections import Counter
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import uuid
import warnings

from . import ROOT
from .match_cache import MatchCache, STATE_PREFIX, READY_MARKER, read_game
from .match_dispatch import REQUEST_PREFIX, plan_windows
from .run import save_json, snapshot_source


def load_games(directory, limit=None):
    directory = Path(directory)
    plan = json.loads((directory / 'plan.json').read_text())
    games = []
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(plan['matchups'])):
        raise ValueError('Requested game prefix exceeds the saved schedule')
    for index, pair in enumerate(plan['matchups'][:limit]):
        path = directory / f'game-{index}.json'
        if not path.is_file():
            raise ValueError(f'Match is incomplete: missing game {index}; resume match before fitting')
        games.append(read_game(path, index, pair))
    if not games:
        raise ValueError('Cannot fit an empty match')
    return plan, games


def summarize_endings(reasons):
    total = sum(reasons.values())
    truncated = reasons['move_limit']
    # Old archived records have no ending metadata. Do not report missing
    # observations as zero truncations, or include unplayed games in the rate.
    return dict(end_reasons=dict(reasons), games_truncated=truncated,
                truncation_fraction=truncated / total if total and not reasons['unknown'] else None)


def summarize_match(directory):
    """Rebuild the observational summary from native records, including a partial match.

    The caller holds the output lock and has waited for the native writer to
    finish. Summaries are never used to decide which games need to be resumed.
    """
    directory = Path(directory)
    plan = json.loads((directory / 'plan.json').read_text())
    counts = Counter()
    reasons = Counter()
    for index, pair in enumerate(plan['matchups']):
        path = directory / f'game-{index}.json'
        if path.is_file():
            game = read_game(path, index, pair)
            reasons[game.get('end_reason', 'unknown')] += 1
            counts[game['black'], game['white'], game['outcome']] += 1
    pairs = sorted({(black, white) for black, white, _ in counts})
    rows = [dict(black=black, white=white, black_wins=counts[black, white, 'B'],
                 white_wins=counts[black, white, 'W'], draws=counts[black, white, 'D'],
                 no_results=counts[black, white, 'NR']) for black, white in pairs]
    total, completed = len(plan['matchups']), sum(counts.values())
    result = dict(completed=completed == total, games_requested=total, games_completed=completed,
                  results=rows, **summarize_endings(reasons))
    save_json(directory / 'summary.json', result)
    return result


def observe_match(directory):
    # The game records are authoritative; a failure to rebuild a display must
    # not discard them or prevent an otherwise complete Elo fit.
    try:
        summarize_match(directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.warn(f'Could not rebuild match summary in {directory}: {error}', RuntimeWarning)


def validate_fit_options(prior_games, first_player_advantage):
    if type(prior_games) not in (int, float) or not math.isfinite(prior_games) or prior_games < 0:
        raise ValueError('prior_games must be finite and nonnegative')
    if type(first_player_advantage) is not bool:
        raise ValueError('first_player_advantage must be boolean')


def validate_model_window(max_loaded_models):
    if max_loaded_models is not None and (type(max_loaded_models) is not int or max_loaded_models < 2):
        raise ValueError('max_loaded_models must be an integer of at least 2')


def estimate(plan, games, *, prior_games=2., first_player_advantage=False, anchor=None):
    from katago.utils import elo
    validate_fit_options(prior_games, first_player_advantage)
    participating = sorted({game[color] for game in games for color in ('black', 'white')})
    names = {index: plan['bots'][index]['name'] for index in participating}
    if len(set(names.values())) != len(names) or elo.P1_ADVANTAGE_NAME in names.values():
        raise ValueError('Participating bot names must be unique and cannot be P1Advantage')
    if len(names) < 2:
        raise ValueError('Elo needs at least two participating bots')
    anchor = names[participating[0]] if anchor is None else anchor
    if anchor not in names.values():
        raise ValueError(f'Anchor is not a participating bot: {anchor}')
    edges = {index: set() for index in participating}
    for game in games:
        edges[game['black']].add(game['white'])
        edges[game['white']].add(game['black'])
    visited, pending = set(), [participating[0]]
    while pending:
        current = pending.pop()
        if current not in visited:
            visited.add(current)
            pending.extend(edges[current] - visited)
    if len(visited) != len(names):
        raise ValueError('Elo comparison graph is disconnected; priors cannot replace actual cross-play')

    summary = elo.GameResultSummary(prior_games, first_player_advantage)
    outcomes = dict.fromkeys(('B', 'W', 'D', 'NR'), 0)
    for game in games:
        outcome = game['outcome']
        outcomes[outcome] += 1
        summary.add_game_record(elo.GameRecord(names[game['black']], names[game['white']],
            win=int(outcome == 'B'), loss=int(outcome == 'W'), draw=int(outcome in ('D', 'NR'))))
    info = summary.get_elos()
    players = []
    for index, name in names.items():
        value, stderr = info.get_elo_difference(name, anchor), info.get_approx_elo_difference_stderr(name, anchor)
        if not math.isfinite(value) or not math.isfinite(stderr):
            raise RuntimeError('Source Elo fit did not produce finite estimates and covariance')
        players.append({'bot_index': index, 'name': name, 'model': plan['bots'][index]['model'],
                        'elo': value, 'stderr': stderr})
    los = {}
    for name in names.values():
        los[name] = {}
        for other in names.values():
            value = float(info.get_approx_likelihood_of_superiority(name, other))
            if not math.isfinite(value):
                warnings.warn(f'Source LOS is undefined for {name} versus {other}', RuntimeWarning)
            los[name][other] = value if math.isfinite(value) else None
    result = {'anchor': anchor, 'games': len(games), 'outcomes': outcomes, 'prior_games': prior_games,
              **summarize_endings(Counter(game.get('end_reason', 'unknown') for game in games)),
              'estimate_first_player_advantage': first_player_advantage, 'players': players, 'los': los,
              'elo_covariance': {p: {q: info.elo_covariance[p, q] for q in info.players} for p in info.players}}
    if first_player_advantage:
        result['first_player_advantage'] = {'elo': info.get_elo(elo.P1_ADVANTAGE_NAME),
                                          'stderr': info.get_approx_elo_stderr(elo.P1_ADVANTAGE_NAME)}
    return result


def plot_ratings(result, output):
    import matplotlib as mpl
    from .plotting import THEME, BLUE, GREY, _figure
    players = sorted(result['players'], key=lambda player: -player['elo'])
    with mpl.rc_context(THEME):
        figure, (axis,) = _figure(['Relative Elo'], [('Bot', 'Elo')],
            'EtaZero V4 · source Elo fit', columns=1, figsize=(max(8, len(players) * .6), 6))
        positions = list(range(len(players)))
        axis.errorbar(positions, [p['elo'] for p in players], yerr=[p['stderr'] for p in players],
                      color=BLUE, fmt='o', capsize=4, linestyle='none')
        axis.set_xticks(positions, [p['name'] for p in players], rotation=30, ha='right')
        axis.axhline(0, linewidth=.7, color=GREY)
        figure.supxlabel(f'{result["games"]} games · anchor: {result["anchor"]} · error bars: approximately one standard error\n'
                            'KataGo likelihood, priors and covariance; D and NR each count half a point.', fontsize=10, color=GREY)
        try:
            for extension in ('png', 'svg'):
                destination = output / f'elo.{extension}'
                temporary = destination.with_name(destination.name + '.tmp.' + uuid.uuid4().hex)
                try:
                    figure.savefig(temporary, format=extension, dpi=160, bbox_inches='tight')
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            figure.clear()


def fit_match(directory, output, **options):
    plan, games = load_games(directory)
    result = estimate(plan, games, **options)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'elo.json', result)
    destination = output / 'elo.csv'
    temporary = destination.with_name(destination.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('bot_index', 'name', 'model', 'elo', 'stderr'))
            writer.writeheader()
            writer.writerows(result['players'])
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    plot_ratings(result, output)
    return result


def match_inputs(config, binary, extra_config=None):
    if config is None:
        raise ValueError('match requires a source match config, or --fit-only')
    config = Path(config).resolve()
    binary = Path(binary or ROOT / 'build/katago').resolve()
    if not config.is_file() or not binary.is_file():
        raise ValueError('Match config and native binary must exist')
    if extra_config is not None:
        extra_config = Path(extra_config).resolve()
        if not extra_config.is_file():
            raise ValueError('Additional match config must exist')
    return config, binary, extra_config


@contextmanager
def match_output(output, config=None, extra_config=None, extend_pair=False, source_snapshot=None):
    """Hold the same per-match lock for single and shared-residency invocations."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another evaluation controller is using this output directory') from error
        if config is not None:
            identity = {'config': str(config), 'working_directory': str(config.parent)}
            if extend_pair:
                # Native resolved config and cwd own the cache identity. Changing
                # only the profile filename does not change match semantics.
                identity = {'working_directory': str(config.parent), 'extend_pair': True}
            if extra_config is not None:
                identity['extra_config'] = str(extra_config)
            manifest = output / 'invocation.json'
            if manifest.exists():
                if json.loads(manifest.read_text()) != identity:
                    raise ValueError('Match config location differs; use a new output directory')
            else:
                if any(path.name != '.lock' for path in output.iterdir()):
                    raise ValueError('A new evaluation requires an empty output directory')
                if source_snapshot is None:
                    snapshot_source(output / 'source.tar.gz')
                save_json(output / 'source.json', {
                    'repository_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                    'reference': json.loads((ROOT / 'reference_sources.json').read_text()),
                    'snapshot': str(source_snapshot) if source_snapshot is not None else 'source.tar.gz'})
                save_json(manifest, identity)
            (output / 'invocations').mkdir(exist_ok=True)
        yield lock.fileno()


def execute_match(command, cwd, log_path, inherited_fds, env, *, max_loaded_models=None):
    print(f'Running source match; log: {log_path}', flush=True)
    external = any(flag in command for flag in ('-results-dir', '-pair-jobs'))
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE if external else log,
                                   stdin=subprocess.PIPE if external else None, stderr=subprocess.STDOUT,
                                   pass_fds=inherited_fds, env=env, text=True)
        try:
            if external:
                preparing, caches = True, {}
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.rstrip('\n') == READY_MARKER:
                        preparing = False
                        process.stdin.close()
                    if not preparing:
                        continue
                    response = None
                    if line.startswith(STATE_PREFIX):
                        request = json.loads(line[len(STATE_PREFIX):])
                        directory = request['directory']
                        if request['action'] == 'load':
                            caches[directory] = MatchCache(request)
                            response = caches[directory].state()
                        elif request['action'] == 'schedule':
                            response = caches[directory].schedule(request['proposal'])
                        else:
                            raise ValueError(f'Unknown native match state request: {request["action"]}')
                    elif line.startswith(REQUEST_PREFIX):
                        request = json.loads(line[len(REQUEST_PREFIX):])
                        response = dict(windows=plan_windows(request['games'], max_loaded_models))
                        dispatch_dir = log_path.parent / 'dispatch'
                        dispatch_dir.mkdir(exist_ok=True)
                        save_json(dispatch_dir / f'{log_path.stem}.json',
                                  dict(max_loaded_models=max_loaded_models, **request, **response))
                    if response is not None:
                        process.stdin.write(json.dumps(response) + '\n')
                        process.stdin.flush()
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            if external:
                process.stdin.close()
                process.stdout.close()
    if returncode:
        raise RuntimeError(f'Native match failed; completed games retained. See {log_path}')


def run_match(output, config=None, *, binary=None, fit_only=False, prior_games=2., first_player_advantage=False,
              anchor=None, extra_config=None, inherited_fds=(), extend_pair=False, fit=True, game_limit=None,
              env=None, source_snapshot=None, max_loaded_models=None):
    validate_fit_options(prior_games, first_player_advantage)
    validate_model_window(max_loaded_models)
    if fit_only and max_loaded_models is not None:
        raise ValueError('fit-only does not load models or take a residency window')
    output = Path(output).resolve()
    if not fit_only:
        config, binary, extra_config = match_inputs(config, binary, extra_config)
    elif config is not None:
        raise ValueError('--fit-only uses saved match records and does not take a new config')
    if fit_only and not (output / 'results/plan.json').is_file():
        raise ValueError('No saved match schedule to fit')
    with match_output(output, config, extra_config, extend_pair, source_snapshot) as fd:
        if not fit_only:
            logs = output / 'invocations'
            attempt = uuid.uuid4().hex
            command = [str(binary), 'match', '-config', str(config), '-results-dir', str(output / 'results'),
                       '-sgf-output-dir', str(output / 'sgfs'), '-log-file', str(logs / f'{attempt}.native.log')]
            if extend_pair:
                command.append('-extend-pair')
            if max_loaded_models is not None:
                command.append('-external-dispatch')
            if extra_config is not None:
                # The source treats later config files as includes and requires
                # relative paths. Keep the original profile's resource cwd.
                command += ['-config', os.path.relpath(extra_config, config.parent)]
            save_json(logs / f'{attempt}.json', {'command': command, 'working_directory': str(config.parent),
                'max_loaded_models': max_loaded_models,
                'source_snapshot': str(source_snapshot) if source_snapshot is not None else None})
            execute_match(command, config.parent, logs / f'{attempt}.log', (fd, *inherited_fds), env,
                          max_loaded_models=max_loaded_models)
            observe_match(output / 'results')
        if not fit:
            return load_games(output / 'results', game_limit)
        return fit_match(output / 'results', output, prior_games=prior_games,
                         first_player_advantage=first_player_advantage, anchor=anchor)


def run_match_batch(tasks, config, session, *, binary=None, inherited_fds=(), env=None,
                    source_snapshot=None, max_loaded_models=2):
    """Dispatch autoelo pair caches in one native match; keep each cache's inputs."""
    validate_model_window(max_loaded_models)
    config, binary, _ = match_inputs(config, binary)
    tasks = [(Path(output).resolve(), match_inputs(config, binary, extra)[2]) for output, extra in tasks]
    if not tasks:
        return
    with ExitStack() as stack:
        fds = tuple(stack.enter_context(match_output(output, config, extra, True, source_snapshot))
                    for output, extra in tasks)
        logs = Path(session).resolve() / 'invocations'
        logs.mkdir(parents=True, exist_ok=True)
        attempt = uuid.uuid4().hex
        jobs = logs / f'{attempt}.jobs.json'
        save_json(jobs, [dict(extra_config=os.path.relpath(extra, config.parent),
                             results_dir=str(output / 'results'), sgf_output_dir=str(output / 'sgfs'))
                         for output, extra in tasks])
        native_log = logs / f'{attempt}.native.log'
        log = logs / f'{attempt}.log'
        command = [str(binary), 'match', '-config', str(config), '-pair-jobs', str(jobs), '-log-file', str(native_log),
                   '-external-dispatch']
        invocation = dict(command=command, working_directory=str(config.parent), log=str(log), native_log=str(native_log),
                          max_loaded_models=max_loaded_models,
                          source_snapshot=str(source_snapshot) if source_snapshot is not None else None)
        save_json(logs / f'{attempt}.json', invocation)
        for index, (output, _) in enumerate(tasks):
            save_json(output / 'invocations' / f'{attempt}.json', dict(invocation, pair_job=index))
        execute_match(command, config.parent, log, (*fds, *inherited_fds), env, max_loaded_models=max_loaded_models)
        for output, _ in tasks:
            observe_match(output / 'results')
