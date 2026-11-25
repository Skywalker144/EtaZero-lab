"""Native prefix reuse, source config identity, and independent evaluation recovery."""
from collections import Counter
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path

import pytest

from etazero import autoelo
from etazero.arena import match_overlay
from etazero.evaluation import load_games
from etazero.experiment import experiment_plan, run_experiment
from etazero.run import run, save_json
from test_arena import history
from test_controller_recovery import recovery_config
from test_experiment import plan_for, fake_source_children
from test_gatekeeper import gate_config


def test_source_parser_includes_and_budget_independent_pair_identity(tmp_path):
    data = tmp_path / 'runs'
    history(data / 'a # x', [5, 10, 15]); history(data / 'b', [6, 11, 16])
    included = gate_config(tmp_path / 'included.cfg')
    profile = tmp_path / 'profile.cfg'
    profile.write_text('@include included.cfg\n')
    output = tmp_path / 'elo'
    first = autoelo.run_autoelo(output, profile, data=data, dry_run=True, games_per_pair=3)
    second = autoelo.run_autoelo(output, profile, data=data, dry_run=True, games_per_pair=8)
    assert not output.exists()
    assert [p['cache_key'] for p in first['pairs']] == [p['cache_key'] for p in second['pairs']]
    assert 'a # x' in first['pairs'][0]['conditions']
    included.write_text(included.read_text().replace('maxVisits = 8', 'maxVisits = 9'))
    changed = autoelo.autoelo_plan(output, data, profile, games_per_pair=3)
    assert not ({p['cache_key'] for p in first['pairs']} & {p['cache_key'] for p in changed['pairs']})
    output.mkdir()
    save_json(output / 'anchor.json', first['anchor_record'])
    with pytest.raises(ValueError, match='pinned'):
        autoelo.autoelo_plan(output, data, profile, anchor='b:00000001')
    pinned = autoelo.autoelo_plan(output, data, profile, stride=3, include_first=False)
    assert first['anchor'] in {p['id'] for p in pinned['players']}


def test_pair_local_settings_follow_the_rated_player(tmp_path):
    gate_config(tmp_path / 'common.cfg')
    config = tmp_path / 'match.cfg'
    config.write_text('@include common.cfg\nmaxVisits0 = 16\nmaxVisits1 = 256\n')
    output = tmp_path / 'elo'
    data = history(tmp_path / 'runs/a', [10, 20])
    assert len(autoelo.run_autoelo(output, config, data=data, dry_run=True)['pairs']) == 1
    history(data, [10, 20, 30])
    for dry_run in (True, False):
        with pytest.raises(ValueError, match='a:00000002.*different search settings'):
            autoelo.run_autoelo(output, config, data=data, dry_run=dry_run)
    assert not (output / 'sessions').exists()
    config.write_text('@include common.cfg\nmaxVisits0 = 08\n')
    plan = autoelo.run_autoelo(output, config, data=data, dry_run=True)
    assert len(plan['pairs']) == 2
    for pair in plan['pairs']:
        pair.pop('search_strategies')
        pair['conditions'] += 'maxVisits1 = 256\n'
    session = tmp_path / 'old-session'
    session.mkdir()
    save_json(session / 'plan.json', plan)
    config.unlink()
    with pytest.raises(ValueError, match='a:00000002.*different search settings'):
        autoelo.run_autoelo(session, fit_only=True)


def test_saved_settings_checked_even_if_player_keeps_same_slot(tmp_path):
    data = history(tmp_path / 'runs/a', [10, 20, 30])
    config = gate_config(tmp_path / 'match.cfg')
    plan = autoelo.run_autoelo(tmp_path / 'elo', config, data=data, neighbors=2, dry_run=True)
    # Simulate a profile edit between two resolutions in a fixed-slot schedule.
    plan['pairs'] = [pair for pair in plan['pairs'] if pair['players'][0] == plan['anchor']]
    assert len(plan['pairs']) == 2
    config.write_text(config.read_text().replace('maxVisits = 8', 'maxVisits = 9'))
    changed = autoelo.run_autoelo(tmp_path / 'elo', config, data=data, neighbors=2, dry_run=True)
    plan['pairs'][1]['search_strategies'] = changed['pairs'][1]['search_strategies']
    session = tmp_path / 'saved'
    session.mkdir()
    save_json(session / 'plan.json', plan)
    with pytest.raises(ValueError, match='a:00000001.*different search settings'):
        autoelo.run_autoelo(session, fit_only=True, binary='/missing')


def test_archived_settings_preserve_literal_tabs_in_sgf_paths(tmp_path):
    data = history(tmp_path / 'runs/a', [10, 20, 30])
    config = gate_config(tmp_path / 'match.cfg')
    plan = autoelo.run_autoelo(tmp_path / 'elo', config, data=data, dry_run=True)
    # Old plans store literal decoded values, including a tab or a hash sign.
    for pair, directory in zip(plan['pairs'], ['a\tb # patterns', 'atb # patterns'], strict=True):
        pair.pop('search_strategies')
        pair['conditions'] += f'avoidSgfPatternUtility = -1\navoidSgfPatternDirs = {directory}\n'
    session = tmp_path / 'old'
    session.mkdir()
    save_json(session / 'plan.json', plan)
    config.unlink()
    with pytest.raises(ValueError, match='a:00000002.*different search settings'):
        autoelo.run_autoelo(session, fit_only=True)


@pytest.mark.parametrize(('overrides', 'conflict'), [
    ('useFP16-0 = true\nuseFP16-1 = false\n', True),
    ('avoidSgfPatternUtility0 = -1\navoidSgfPatternDirs0 = "missing # patterns"\n', True),
    ('maxBoardSizeForNNBuffer0 = 4\n', True),
    ('maxBoardSizeForNNBuffer0 = 3\nnnRandSeed0 = a\nnnRandSeed1 = b\n'
     'gpuToUseModel0 = 0\ngpuToUseModel1 = 1\n', False),
])
def test_autoelo_resolves_settings_without_loading_model_or_sgf(tmp_path, overrides, conflict):
    data = history(tmp_path / 'runs/a', [10, 20, 30])
    gate_config(tmp_path / 'base.cfg')
    config = tmp_path / 'match.cfg'
    config.write_text('@include base.cfg\n' + overrides)
    if conflict:
        with pytest.raises(ValueError, match='a:00000002.*different search settings'):
            autoelo.run_autoelo(tmp_path / 'elo', config, data=data, dry_run=True)
    else:
        plan = autoelo.run_autoelo(tmp_path / 'elo', config, data=data, dry_run=True)
        assert len(plan['pairs']) == 2


@pytest.mark.parametrize('option', [dict(games_per_pair=0), dict(neighbors=True), dict(sample_seconds=float('nan')),
    dict(final_cross='true'), dict(prior_games=-1), dict(first_player_advantage='false')])
def test_experiment_rejects_bad_evaluation_before_training(tmp_path, option):
    plan_for(tmp_path)
    gate_config(tmp_path / 'match.cfg')
    path = tmp_path / 'exp.json'
    save_json(path, {**json.loads(path.read_text()), 'evaluation': dict(config='match.cfg', **option)})
    with pytest.raises(ValueError):
        experiment_plan(path, tmp_path / 'work')
    assert not (tmp_path / 'work').exists()


def test_failed_post_training_evaluation_retries_without_retraining(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    fake_source_children(tmp_path, monkeypatch)
    plan['evaluation'] = dict(config='/independent/profile.cfg')
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError('injected evaluation failure')
        return {'games': 2}
    monkeypatch.setattr(autoelo, 'run_autoelo', evaluate)
    with pytest.raises(RuntimeError, match='injected'):
        run_experiment(plan, '/bin/true')
    work = Path(plan['work_dir'])
    assert json.loads((work / 'elo_status.json').read_text())['training'] == 'complete'
    before = {p: p.read_bytes() for p in (work / 'runs').glob('*/observed.json')}
    assert run_experiment(plan, '/bin/true') == 0
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert json.loads((work / 'elo_status.json').read_text())['status'] == 'complete'
    assert calls[-1]['arms'] == [(a['name'], a['run_dir']) for a in plan['arms']]
    assert calls[-1]['inherited_fds']


def check_trained_autoelo(tmp_path, arms):
    tmp_path.mkdir(parents=True, exist_ok=True)
    profile = gate_config(tmp_path / 'match.cfg', moves=100)
    output = tmp_path / 'elo'
    args = dict(data=tmp_path, arms=arms, final_cross=True)
    first = autoelo.run_autoelo(output, profile, games_per_pair=3, **args)
    session = (output / 'latest').resolve()
    plan = json.loads((session / 'plan.json').read_text())
    assert first['games'] == 3 * len(plan['pairs'])
    saved = {p: p.read_bytes() for p in (output / 'pair_cache').glob('*/match/results/game-*.json')}
    old_snapshot = {p: p.read_bytes() for p in (session / 'pairs').glob('*/*.json')}
    # Native cached completion must work with no visible GPU.
    cached = autoelo.run_autoelo(output, profile, games_per_pair=3, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), **args)
    assert cached == first
    extended = autoelo.run_autoelo(output, profile, games_per_pair=6, **args)
    assert extended['games'] == first['games'] * 2
    assert all(p.read_bytes() == raw for p, raw in saved.items())
    assert all(p.read_bytes() == raw for p, raw in old_snapshot.items())
    for pair in plan['pairs']:
        native, games = load_games(output / 'pair_cache' / pair['cache_key'] / 'match/results')
        assert len(games) == 6
        assert Counter(g['black'] for g in games) == {0: 3, 1: 3}
        assert native['matchups'][3] == list(reversed(native['matchups'][2]))
        assert all('RE[' in g['sgf'] for g in games)
    fewer = autoelo.run_autoelo(output, profile, games_per_pair=2, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), **args)
    assert fewer['games'] == 2 * len(plan['pairs'])
    assert all(p.read_bytes() == raw for p, raw in saved.items())
    # Same semantic config at another filename in the same resource cwd reuses games.
    renamed = tmp_path / 'renamed.cfg'
    renamed.write_bytes(profile.read_bytes())
    assert autoelo.run_autoelo(output, renamed, games_per_pair=3, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), **args) == first
    assert len(list((output / 'pair_cache').iterdir())) == len(plan['pairs'])
    # A changed search profile creates independent trials, rather than mixing conditions.
    profile.write_text(profile.read_text().replace('maxVisits = 8', 'maxVisits = 9'))
    changed = autoelo.run_autoelo(output, profile, games_per_pair=1, **args)
    assert changed['games'] == len(plan['pairs'])
    assert len(list((output / 'pair_cache').iterdir())) == 2 * len(plan['pairs'])
    profile.unlink(); renamed.unlink()
    assert autoelo.run_autoelo(session, fit_only=True, binary='/missing') == first
    assert (session / 'elo.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (session / 'elo.svg').read_text()
    assert all(p.read_bytes() == raw for p, raw in old_snapshot.items())
    return first


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires actual training and native CUDA match')
def test_real_autoelo_from_trained_histories(tmp_path):
    arms = []
    for name, algorithm in [('a', 'alphazero'), ('b', 'gumbel_muzero')]:
        config_dir = tmp_path / ('config-' + name); config_dir.mkdir()
        config = replace(recovery_config(config_dir, algorithm), auto_plot=False)
        destination = tmp_path / 'training' / name
        run(destination, config, 2)
        arms.append((name, destination))
    check_trained_autoelo(tmp_path / 'evaluation', arms)
