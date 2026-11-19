"""Outer cache decisions preserve native color rounds and immutable originals."""
import json
import os

import pytest

from etazero import match_cache
from etazero.evaluation import run_match
from etazero.match_cache import MatchCache
from test_match import match_config
from test_match_windows import check_games, window_config


def request(path, count, *, per_pair=0, extension=''):
    path.mkdir(exist_ok=True)
    (path / '.native.lock').touch()
    return dict(directory=str(path), resolved_config=f'games={count}', extension_config=extension,
                working_directory=str(path.parent), num_games=count, games_per_pair=per_pair)


def proposal(pairs):
    return dict(bots=[dict(name=f'bot{i}', model=f'/model{i}') for i in range(3)],
                game='hex', matchups=pairs)


def write_game(path, index, pair, outcome='D'):
    (path / f'game-{index}.json').write_text(json.dumps(
        dict(game_index=index, black=pair[0], white=pair[1], outcome=outcome)))


def test_pair_quotas_keep_first_native_occurrences_and_resume_noncontiguous_games(tmp_path):
    directory = tmp_path / 'results'
    inputs = request(directory, 3, per_pair=1)
    cache = MatchCache(inputs)
    assert cache.state() == dict(plan=None, completed=[], generate=True, proposal_games=None)
    source = proposal([[1, 2], [1, 0], [0, 2], [2, 0], [2, 1], [0, 1]])
    state = cache.schedule(source)
    assert state['plan']['matchups'] == [[1, 2], [1, 0], [0, 2]]
    assert source['matchups'] == [[1, 2], [1, 0], [0, 2], [2, 0], [2, 1], [0, 1]]
    write_game(directory, 0, [1, 2])
    write_game(directory, 2, [0, 2], 'W')
    originals = {p: p.read_bytes() for p in directory.glob('*.json')}
    recovered = MatchCache(inputs).state()
    assert recovered == dict(plan=state['plan'], completed=[0, 2], generate=False, proposal_games=0)
    assert all(p.read_bytes() == content for p, content in originals.items())
    with pytest.raises(ValueError, match='configuration differs'):
        MatchCache(dict(inputs, resolved_config='changed search'))
    with pytest.raises(ValueError, match='working directory changed'):
        MatchCache(dict(inputs, working_directory='/different'))
    write_game(directory, 2, [2, 0], 'W')
    with pytest.raises(ValueError, match='committed schedule'):
        MatchCache(inputs)


def test_extending_an_odd_prefix_adds_reverse_color_before_native_rounds(tmp_path):
    directory = tmp_path / 'results'
    inputs = request(directory, 3, per_pair=3, extension='same conditions')
    first = MatchCache(inputs).schedule(proposal([[0, 1], [1, 0], [1, 0], [0, 1]]))['plan']
    for i, pair in enumerate(first['matchups']):
        write_game(directory, i, pair)
    originals = {p: p.read_bytes() for p in directory.glob('game-*.json')}
    archive = tmp_path / 'archived-plan.json'
    os.link(directory / 'plan.json', archive)
    archived_bytes = archive.read_bytes()
    cache = MatchCache(request(directory, 6, per_pair=6, extension='same conditions'))
    assert cache.state()['proposal_games'] == 2
    result = cache.schedule(proposal([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1]]))
    assert result['plan']['matchups'] == [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]
    assert result['completed'] == [0, 1, 2] and not result['generate']
    assert json.loads((directory / 'plans/3.json').read_text()) == first
    assert archive.read_bytes() == archived_bytes
    assert all(p.read_bytes() == content for p, content in originals.items())
    smaller = MatchCache(request(directory, 2, per_pair=2, extension='same conditions')).state()
    assert smaller == result  # The requested prefix does not shrink saved evidence.
    with pytest.raises(ValueError, match='configuration differs'):
        MatchCache(request(directory, 8, per_pair=8, extension='different conditions'))


def test_schedule_publication_failure_does_not_start_games_and_can_resume(tmp_path, monkeypatch):
    config = window_config(tmp_path / 'match.cfg')
    root = tmp_path / 'evaluation'
    snapshot = tmp_path / 'source.tar.gz'; snapshot.touch()

    def fail_write(*args):
        raise OSError('schedule publication failed')

    with monkeypatch.context() as patch:
        patch.setattr(match_cache, 'save_json', fail_write)
        with pytest.raises(OSError, match='schedule publication failed'):
            run_match(root, config, max_loaded_models=2, source_snapshot=snapshot, fit=False)
    assert not (root / 'results/plan.json').exists()
    assert not list((root / 'results').glob('game-*.json'))
    run_match(root, config, max_loaded_models=2, source_snapshot=snapshot, fit=False)
    check_games(root / 'results', 3)


def test_odd_pair_extension_needing_only_reverse_color_draws_no_new_pairings(tmp_path):
    config = match_config(tmp_path / 'match.cfg', [('/dev/null', False)] * 2, games=3)
    with config.open('a') as out:
        out.write('numGamesPerPair = 3\ndebugSkipNeuralNet = true\n')
    root = tmp_path / 'evaluation'
    snapshot = tmp_path / 'source.tar.gz'; snapshot.touch()
    plan, _ = run_match(root, config, extend_pair=True, source_snapshot=snapshot, fit=False)
    original = {p: p.read_bytes() for p in (root / 'results').glob('game-*.json')}
    logs = set((root / 'invocations').glob('*.log'))
    config.write_text(config.read_text().replace('numGamesTotal = 3\n', 'numGamesTotal = 4\n')
                      .replace('numGamesPerPair = 3\n', 'numGamesPerPair = 4\n'))
    extended, games = run_match(root, config, extend_pair=True, source_snapshot=snapshot, fit=False)
    assert len(games) == 4 and extended['matchups'] == plan['matchups'] + [list(reversed(plan['matchups'][-1]))]
    assert all(p.read_bytes() == content for p, content in original.items())
    log = next(p for p in set((root / 'invocations').glob('*.log')) - logs if not p.name.endswith('.native.log'))
    requests = [json.loads(line[len(match_cache.STATE_PREFIX):]) for line in log.read_text().splitlines()
                if line.startswith(match_cache.STATE_PREFIX)]
    proposals = [r['proposal']['matchups'] for r in requests if r['action'] == 'schedule']
    assert proposals == [[]]
