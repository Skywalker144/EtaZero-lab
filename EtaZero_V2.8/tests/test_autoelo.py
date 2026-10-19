from config_samples import CONFIGS, derive_config
from pathlib import Path
import sys
import io
import threading
import time

import pytest

from etazero.arena import Player, PairStore, build_schedule, discover_players, execute_pair, execute_arena, identity, run_arena
from etazero.autoelo import autoelo_plan, load_elo_config, load_elo_profile, publish_latest
from etazero.autoelo import run_autoelo
from etazero.elo_cache import pair_identity, pair_seed, prepare_pairs, retain_pairs
from etazero.config import ROOT, load_config
from etazero.engine_config import load_engine_config
from etazero.experiment import experiment_plan, run_experiment, Scheduler
from etazero.schema import CONTRACT_ID
from etazero.storage import load_json, save_json


def match_profile(directory, contents=''):
    (directory/'match.cfg').write_text(f'@include "{CONFIGS / "baseline/match.cfg"}"\n'+contents)


def history(root, times, algorithm=None):
    save_json(root/'logs/iterations/000000.json', {'iteration': 0, 'elapsed_seconds': 0, 'model': None})
    for i, seconds in enumerate(times, 1):
        model = root/'models'/str(i)/'model.pt'
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(f'{root.name}:{i}'.encode())
        info = {'id': str(i), 'path': str(model.relative_to(root)),
                'canvas': 15, 'contract': CONTRACT_ID, 'bytes': model.stat().st_size}
        if algorithm:
            info['algorithm'] = algorithm
        save_json(model.parent/'manifest.json', info)
        save_json(root/'logs/iterations'/f'{i:06d}.json',
                  {'iteration': i, 'elapsed_seconds': seconds, 'model': info})
    save_json(root/'.internal/state.json', {'iteration': len(times)+1, 'elapsed_seconds': times[-1]})


def test_explicit_arm_names_required_models_and_missing_history(tmp_path):
    history(tmp_path/'a', [1, 20, 100])
    history(tmp_path/'b', [1, 100])
    players = discover_players(tmp_path, stride=4, include_first=False,
                               arms=[('custom-name', tmp_path/'a'), ('b', tmp_path/'b')],
                               required_ids=('custom-name:00000002',))
    assert [(p.arm, p.seconds) for p in players] == [('custom-name', 20), ('custom-name', 100), ('b', 100)]
    with pytest.raises(ValueError, match='no committed state'):
        discover_players(tmp_path, arms=[('missing', tmp_path/'missing')])


def test_fixed_iteration_sampling_and_bridges_survive_extension_and_new_arm(tmp_path):
    history(tmp_path/'a', list(range(1, 28)))
    history(tmp_path/'b', list(range(1, 28)))
    players = discover_players(tmp_path, stride=8, include_first=False)
    assert [p.iteration for p in players if p.arm == 'a'] == [8, 16, 24, 27]
    edges = build_schedule(players, 2, cross_seconds=10, stride=8, final_cross=True)
    assert ('a:00000008', 'b:00000008') in edges
    # The 20-second target ties 16 and 24; choose the earlier checkpoint.
    assert ('a:00000016', 'b:00000016') in edges
    history(tmp_path/'a', list(range(1, 44)))
    history(tmp_path/'b', list(range(1, 44)))
    history(tmp_path/'c', [1, 2])
    players = discover_players(tmp_path, stride=8, include_first=False)
    assert [p.iteration for p in players if p.arm == 'a'] == [8, 16, 24, 32, 40, 43]
    extended = build_schedule(players, 2, cross_seconds=10, stride=8, final_cross=True)
    stable = [edge for edge in edges if not any(p.endswith('00000027') for p in edge)]
    assert set(stable) <= set(extended)
    assert ('a:00000043', 'c:00000002') in extended


def test_nearest_time_sampling_ties_deduplication_and_final(tmp_path):
    history(tmp_path/'a', [27*60, 32*60, 58*60, 63*60])
    players = discover_players(tmp_path, sample_seconds=1800, include_first=False)
    assert [(p.iteration, p.seconds) for p in players] == [(2, 32*60), (3, 58*60)]
    history(tmp_path/'a', [28, 32, 70])
    players = discover_players(tmp_path, sample_seconds=30, include_first=False)
    assert [p.iteration for p in players] == [1, 3]  # 30 ties; 60 selects final.
    history(tmp_path/'a', [1, 100, 101])
    players = discover_players(tmp_path, sample_seconds=20, include_first=False)
    assert [p.iteration for p in players] == [1, 2]  # Five targets, two fixed models; close tail omitted.
    history(tmp_path/'a', [1, 2])
    players = discover_players(tmp_path, sample_seconds=20, include_first=False,
                               required_ids=('a:00000001',))
    assert [p.iteration for p in players] == [1, 2]  # Short history plus pinned anchor.


@pytest.mark.parametrize('last, expected', [(34, [1, 2, 3]), (35, [1, 2, 3]),
                                          (36, [1, 2, 3, 4])])
def test_time_sampling_adds_tail_only_beyond_half_interval(tmp_path, last, expected):
    history(tmp_path/'a', [10, 20, 30, last])
    players = discover_players(tmp_path, sample_seconds=10, include_first=False)
    assert [p.iteration for p in players] == expected


def test_omitted_tail_preserves_covered_grid_and_selected_final_cross(tmp_path):
    history(tmp_path/'a', [10, 20, 29, 32])
    history(tmp_path/'b', [10, 20, 30, 32])
    players = discover_players(tmp_path, sample_seconds=10, include_first=False)
    assert [p.iteration for p in players if p.arm == 'a'] == [1, 2, 3]
    edges = build_schedule(players, 1, cross_seconds=30, sample_seconds=10)
    # Both histories covered 30 seconds, even though a's nearest model is at 29.
    assert ('a:00000003', 'b:00000003') in edges
    with_final = build_schedule(players, 1, cross_seconds=100, sample_seconds=10, final_cross=True)
    assert ('a:00000003', 'b:00000003') in with_final
    assert all(not p.endswith('00000004') for edge in with_final for p in edge)
    # An explicitly pinned omitted tail stays in the graph, without displacing
    # the closer fixed model from the 30-second bridge.
    pinned = discover_players(tmp_path, sample_seconds=10, include_first=False,
                              required_ids=('a:00000004',))
    assert ('a:00000003', 'b:00000003') in build_schedule(
        pinned, 1, cross_seconds=30, sample_seconds=10)


def test_time_samples_and_bridges_survive_extension_and_new_arm(tmp_path):
    for arm in ('a', 'b'):
        history(tmp_path/arm, [5, 9, 12, 19, 22, 29, 32, 35])
    players = discover_players(tmp_path, sample_seconds=10, include_first=False)
    assert [p.iteration for p in players if p.arm == 'a'] == [2, 4, 6, 8]
    edges = build_schedule(players, 2, cross_seconds=20, sample_seconds=10, final_cross=True)
    # Model at 19 seconds covers the 20-second grid; do not defer this bridge.
    assert ('a:00000004', 'b:00000004') in edges
    assert ('a:00000008', 'b:00000008') in edges
    for arm in ('a', 'b'):
        history(tmp_path/arm, [5, 9, 12, 19, 22, 29, 32, 35, 38, 42, 47])
    history(tmp_path/'c', [1, 2])
    players = discover_players(tmp_path, sample_seconds=10, include_first=False)
    assert [p.iteration for p in players if p.arm == 'a'] == [2, 4, 6, 9, 11]
    extended = build_schedule(players, 2, cross_seconds=20, sample_seconds=10, final_cross=True)
    stable = [edge for edge in edges if not any(p.endswith('00000008') for p in edge)]
    assert set(stable) <= set(extended)
    assert ('a:00000011', 'c:00000002') in extended
    assert ('a:00000009', 'b:00000009') in extended


def complete_fake_pair(manifest, pair, directory, limit=None):
    store = PairStore(directory, limit or manifest['games_per_pair'])
    for task in store.tasks(pair_seed(manifest, pair)):
        if task['moves'] is None:
            store.accept(dict(type='opening', id=task['id'], generator=task['generator'],
                              seed=task['seed'], moves=[12], value=0))
        for color in (0, 1):
            if task['mask'] & (1 << color):
                store.accept(dict(type='game', id=task['id']*2+color,
                                  opening_id=task['id'], black_a=color == 0,
                                  winner=0, moves=[12, 13], seconds=1))


def test_incremental_autoelo_reuses_after_new_arm_more_games_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-not-in-resolved-environment')
    config = tmp_path/'config'; config.mkdir()
    (config/'elo.cfg').write_text('[elo]\nsample_seconds=8\ncross_seconds=1000\ngames_per_pair=4\nbootstrap_samples=2\n')
    match_profile(config, '[match]\ndevice=cpu\n')
    data = tmp_path/'data'
    for arm in ('b', 'c'):
        history(data/arm, list(range(1, 17)))
    monkeypatch.setattr('etazero.autoelo.verify_build', lambda _: 'binary')
    calls = []
    def native(output, manifest, binary, samples, workers, env):
        assert 'CUDA_VISIBLE_DEVICES' not in env
        for pair in manifest['pairs']:
            directory = output/'pairs'/pair['id']
            calls.extend((pair['id'], task['id']) for task in
                         PairStore(directory, manifest['games_per_pair']).tasks(pair_seed(manifest, pair)))
            complete_fake_pair(manifest, pair, directory)
        from etazero.elo import write_ratings
        return write_ratings(output, samples)
    monkeypatch.setattr('etazero.autoelo.run_arena', native)
    # The test native still publishes the same snapshot contract as run_arena.
    def publish_and_run(output, manifest, *args):
        save_json(output/'manifest.json', manifest)
        return native(output, manifest, *args)
    monkeypatch.setattr('etazero.autoelo.run_arena', publish_and_run)
    def run(**overrides):
        plan = autoelo_plan(config, tmp_path/'binary', data=data, environ={}, overrides=overrides)
        calls.clear()
        run_autoelo(plan, tmp_path/'binary')
        perf = load_json(next((Path(plan['output'])/'invocations').glob('*.json')))
        return plan, list(calls), perf
    first, tasks, perf = run()
    assert len(tasks) == 6 and perf['new_games'] == 12
    # A new alphabetically earlier arm must not replace the persisted anchor.
    history(data/'a', list(range(1, 17)))
    second, tasks, perf = run()
    assert second['manifest']['anchor'] == first['manifest']['anchor']
    assert len(tasks) == 6 and perf['reused_games'] == 12 and perf['new_games'] == 12
    third, tasks, perf = run(games_per_pair='8')
    assert len(tasks) == 12 and perf['reused_games'] == 24 and perf['new_games'] == 24
    _, tasks, _ = run(games_per_pair='8')
    assert tasks == []
    # Smaller schedules read only the requested prefix of the larger cache.
    plan = autoelo_plan(config, tmp_path/'binary', data=data, environ={}, overrides={'games_per_pair':'4'})
    assert prepare_pairs(Path(plan['base']), Path(plan['output']), plan['manifest']) == 24
    for arm in ('a', 'b', 'c'):
        history(data/arm, list(range(1, 25)))
    extended, tasks, perf = run()
    assert extended['manifest']['anchor'] == first['manifest']['anchor']
    assert perf['reused_games'] == 12 and perf['new_games'] == 36
    assert len(tasks) == 18
    off_grid = autoelo_plan(config, tmp_path/'binary', data=data, environ={}, overrides={'sample_seconds':'10'})
    assert first['manifest']['anchor'] in {p['id'] for p in off_grid['manifest']['players']}


def test_cache_identity_invalidates_match_changes_but_not_fit_or_budget(tmp_path):
    manifest = {'players': [{'id':'a','model':'/a/model.pt'}, {'id':'b','model':'/b/model.pt'}],
                'config': {'match': {'seed':1, 'visits':100, 'games':4}},
                'binary_sha256':'bin', 'execution': {'max_loaded_models':4},
                'source_sha256': {'engine_config.py':'compiler', 'elo.py':'fit'}}
    pair = {'id':identity(('a','b')), 'players':['a','b']}
    import copy
    original = pair_identity(manifest, pair)
    other = copy.deepcopy(manifest)
    other.update(anchor='b', sampling={'sample_seconds':32}, games_per_pair=40)
    other['config']['match']['games'] = 40
    other['source_sha256'].update({'elo.py': 'new-fit', 'engine_config.py': 'new-comment'})
    other['players'][0].update(seconds=1000, sample_horizon=2000)
    assert pair_identity(other, pair) == original
    for group, key, value in [('config', 'visits', 200), ('config', 'seed', 17), ('execution', 'max_loaded_models', 8)]:
        changed = copy.deepcopy(manifest)
        target = changed[group]['match'] if group == 'config' else changed[group]
        target[key] = value
        assert pair_identity(changed, pair) != original
    changed = copy.deepcopy(manifest); changed['binary_sha256'] = 'new-bin'
    assert pair_identity(changed, pair) != original
    changed = copy.deepcopy(manifest); changed['players'][0]['model'] = '/new/model.pt'
    assert pair_identity(changed, pair) != original


def test_cache_keeps_partial_opening_and_does_not_mix_independent_trials(tmp_path):
    manifest = {'players':[{'id':'a','model':'/a/model.pt'}, {'id':'b','model':'/b/model.pt'}],
                'config':{'match':{'seed':1}}, 'binary_sha256':'bin', 'games_per_pair':4}
    pair = {'id':identity(('a','b')), 'players':['a','b']}
    manifest['pairs'] = [pair]
    old = tmp_path/'old'; directory = old/'pairs'/pair['id']
    save_json(old/'manifest.json', manifest)
    store = PairStore(directory, 4); task = store.tasks(pair_seed(manifest, pair))[0]
    store.accept(dict(type='opening', id=0, generator=0, seed=task['seed'], moves=[12], value=0))
    store.accept(dict(type='game', id=0, opening_id=0, black_a=True, winner=0, moves=[12,13], seconds=1))
    output = tmp_path/'new'
    assert prepare_pairs(tmp_path, output, manifest) == 1
    pending = PairStore(output/'pairs'/pair['id'],4).tasks(pair_seed(manifest,pair))
    assert pending[0]['mask'] == 2 and pending[0]['moves'] == [12]
    complete_fake_pair(manifest, pair, output/'pairs'/pair['id'])
    retain_pairs(tmp_path, output, manifest)
    # Completing the other side propagates the original trial lineage.
    assert prepare_pairs(tmp_path, tmp_path/'third', manifest) == 4
    conflicting = tmp_path/'zz-rerun'
    save_json(conflicting/'manifest.json', manifest)
    complete_fake_pair(manifest, pair, conflicting/'pairs'/pair['id'])
    p = conflicting/'pairs'/pair['id']/'games/00000000.json'
    game = load_json(p); game['winner'] = 1; save_json(p,game)
    assert prepare_pairs(tmp_path, tmp_path/'fourth', manifest) == 4
    assert load_json(tmp_path/'fourth'/'pairs'/pair['id']/'games/00000000.json')['winner'] == 0


def test_unchanged_cache_merge_reads_each_record_once_and_writes_nothing(tmp_path, monkeypatch):
    from collections import Counter
    from etazero.elo_cache import merge_pair
    manifest = {'config': {'match': {'seed': 1}}, 'games_per_pair': 4}
    pair = {'id': identity(('a', 'b'))}
    source, destination = tmp_path/'source', tmp_path/'destination'
    complete_fake_pair(manifest, pair, source)
    seed = pair_seed(manifest, pair)
    merge_pair(source, 4, destination, 4, seed)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*.json')}
    reads = Counter()
    original = Path.read_text
    def read(path, *args, **kwargs):
        reads[path] += 1
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_text', read)
        merge_pair(source, 4, destination, 4, seed)
    assert len(reads) == 13 and set(reads.values()) == {1}
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*.json')}


def test_complete_cache_does_not_rescan_historical_sessions(tmp_path, monkeypatch):
    manifest = {'players': [{'id':'a','model':'/a'}, {'id':'b','model':'/b'}],
                'config': {'match': {'seed': 1}}, 'binary_sha256': 'bin', 'games_per_pair': 4,
                'pairs': [{'id':identity(('a','b')), 'players':['a','b']}]}
    old = tmp_path/'old'
    save_json(old/'manifest.json', manifest)
    pair = manifest['pairs'][0]
    complete_fake_pair(manifest, pair, old/'pairs'/pair['id'])
    assert prepare_pairs(tmp_path, tmp_path/'first', manifest) == 4
    original = Path.iterdir
    def iterdir(path):
        assert path != tmp_path, 'Complete cache must not scan session history'
        return original(path)
    monkeypatch.setattr(Path, 'iterdir', iterdir)
    assert prepare_pairs(tmp_path, tmp_path/'second', manifest) == 4


def test_source_changes_do_not_change_session_or_match_identity(tmp_path, monkeypatch):
    match_profile(tmp_path, '[match]\ndevice=cpu\n')
    data = tmp_path/'data'
    for arm in ('a', 'b'):
        history(data/arm, [10, 20])
    monkeypatch.setattr('etazero.autoelo.verify_build', lambda _: 'binary')
    monkeypatch.setattr('etazero.autoelo.sha256', lambda _: 'first-source')
    first = autoelo_plan(tmp_path, tmp_path/'binary', data=data, environ={})
    monkeypatch.setattr('etazero.autoelo.sha256', lambda _: 'changed-comment-or-fit')
    second = autoelo_plan(tmp_path, tmp_path/'binary', data=data, environ={})
    assert first['source_sha256'] != second['source_sha256']
    assert first['output'] == second['output'] and first['manifest'] == second['manifest']


def test_cache_excludes_trials_with_content_derived_inference_seeds(tmp_path):
    import copy
    manifest = {'players': [{'id': 'a', 'model': '/a/model.pt'}, {'id': 'b', 'model': '/b/model.pt'}],
                'config': {'match': {'seed': 1}}, 'binary_sha256': 'bin', 'games_per_pair': 4,
                'pairs': [{'id': identity(('a', 'b')), 'players': ['a', 'b']}]}
    old = copy.deepcopy(manifest)
    for player in old['players']:
        player['sha256'] = 'old-content-hash'
    previous = tmp_path/'old'
    save_json(previous/'manifest.json', old)
    complete_fake_pair(old, old['pairs'][0], previous/'pairs'/old['pairs'][0]['id'])
    assert prepare_pairs(tmp_path, tmp_path/'new', manifest) == 0
    assert len(PairStore(previous/'pairs'/old['pairs'][0]['id'], 4).games()) == 4


def test_arena_deduplicates_paths_and_uses_config_seed_independent_of_schedule(tmp_path, monkeypatch):
    from dataclasses import asdict
    import shlex
    players = [Player('a', 'a', 1, 1, str(tmp_path/'a.pt')),
               Player('b', 'b', 1, 1, str(tmp_path/'b.pt')),
               Player('alias', 'alias', 1, 1, str(tmp_path/'a.pt'))]
    manifest = {'players': [asdict(p) for p in players], 'games_per_pair': 4,
                'config': {'match': {'seed': 37, 'device': 'cpu', 'board_size': 5, 'rule': 'freestyle'}},
                'pairs': [{'id': identity(pair), 'players': list(pair)}
                          for pair in [('a', 'b'), ('alias', 'b')]]}
    monkeypatch.setattr('etazero.arena.model_info', lambda p: (Path(p), {'id': 'same-export-label'}))
    for order in (0, 1):
        manifest['pairs'].reverse()
        output = tmp_path/str(order)
        output.mkdir()
        def complete(*args):
            for pair in manifest['pairs']:
                complete_fake_pair(manifest, pair, output/'pairs'/pair['id'])
        monkeypatch.setattr('etazero.arena._run_native', complete)
        execute_arena(Path('/unused'), tmp_path/'cfg', output, manifest, 2, threading.Event())
        models = [shlex.split(row) for row in (output/'models.txt').read_text().splitlines()]
        assert len(models) == 2
        assert {path: int(seed) for _, path, seed in models} == {p.model: 37 for p in players}
        tasks = [row.split() for row in (output/'tasks.txt').read_text().splitlines()]
        assert len({tuple(row[1:3]) for row in tasks}) == 1


def test_autoelo_failure_retains_cache_and_publishes_only_after_recovery(tmp_path, monkeypatch):
    config = tmp_path/'config'; config.mkdir()
    (config/'elo.cfg').write_text('[elo]\nsample_seconds=8\ngames_per_pair=4\nbootstrap_samples=2\n')
    match_profile(config, '[match]\ndevice=cpu\n')
    data = tmp_path/'data'
    for arm in ('a','b'):
        history(data/arm, list(range(1,9)))
    monkeypatch.setattr('etazero.autoelo.verify_build', lambda _: 'binary')
    plan = autoelo_plan(config, tmp_path/'binary', data=data, environ={})
    base, output = Path(plan['base']), Path(plan['output'])
    previous = base/'previous'; previous.mkdir(parents=True)
    publish_latest(base, previous)
    def fail(output, manifest, *_):
        save_json(output/'manifest.json', manifest)
        pair = manifest['pairs'][0]
        store = PairStore(output/'pairs'/pair['id'],4)
        task = store.tasks(pair_seed(manifest,pair))[0]
        store.accept(dict(type='opening',id=0,generator=0,seed=task['seed'],moves=[12],value=0))
        store.accept(dict(type='game',id=0,opening_id=0,black_a=True,winner=0,moves=[12,13],seconds=1))
        raise RuntimeError('interrupted native')
    monkeypatch.setattr('etazero.autoelo.run_arena', fail)
    with pytest.raises(RuntimeError,match='interrupted native'):
        run_autoelo(plan,tmp_path/'binary')
    assert load_json(output/'status.json')['status'] == 'failed'
    assert (base/'latest').resolve() == previous
    assert len(list((base/'pair_cache').glob('*/games/*.json'))) == 1
    def recover(output, manifest, binary, samples, *_):
        for pair in manifest['pairs']:
            complete_fake_pair(manifest,pair,output/'pairs'/pair['id'])
        from etazero.elo import write_ratings
        return write_ratings(output,samples)
    monkeypatch.setattr('etazero.autoelo.run_arena',recover)
    run_autoelo(plan,tmp_path/'binary')
    assert (base/'latest').resolve() == output
    performance = load_json(Path(load_json(output/'status.json')['performance']))
    assert performance['reused_games'] == 1 and performance['new_games'] == 3


def test_shared_profiles_are_independent_and_strict(tmp_path):
    (tmp_path/'elo.cfg').write_text('[elo]\nsample_seconds=4\ngame_threads=7\nmax_loaded_models=3\n')
    (tmp_path/'elo.cfg.local').write_text('[elo]\nsample_seconds=5\n')
    assert load_elo_config(tmp_path, environ={})['sample_seconds'] == 5
    assert load_elo_config(tmp_path, environ={'ELO_SAMPLE_SECONDS': '6'})['sample_seconds'] == 6
    assert load_elo_config(tmp_path, environ={'ELO_SAMPLE_SECONDS': '6'}, overrides={'sample_seconds': '7'})['sample_seconds'] == 7
    assert load_elo_config(tmp_path, environ={'ELO_MODEL_LOAD_WORKERS': '2'})['model_load_workers'] == 2
    for overrides in [{'games_per_pair': '6'}, {'sample_seconds': '0'}, {'game_threads': '0'},
                      {'model_load_workers': '0'},
                      {'max_loaded_models': '1'}, {'pair_workers': '1'},
                      {'sample_seconds': 'nan'}, {'sample_seconds': 'inf'}, {'sample_seconds': '-1'},
                      {'cross_seconds': 'nan'}, {'cross_seconds': '0'}]:
        with pytest.raises(ValueError):
            load_elo_config(tmp_path, environ={}, overrides=overrides)
    training = load_config(CONFIGS / 'muzero')
    match_profile(tmp_path, '[match]\ngame_threads=20\nreuse_tree=false\nuse_graph_search=false\n')
    match = load_engine_config(tmp_path, match=True, environ={})
    assert match['match']['game_threads'] == 20 and not match['match']['reuse_tree']
    assert load_config(CONFIGS / 'muzero') == training
    (tmp_path/'match.cfg').write_text('[match]\nunknown=1\n')
    with pytest.raises(ValueError, match='Unknown key'):
        load_engine_config(tmp_path, match=True, environ={})


def test_optional_elo_overrides_and_explicit_shared_match(tmp_path):
    baseline = load_elo_config(tmp_path, environ={})
    with pytest.raises(FileNotFoundError, match='match.cfg'):
        load_elo_profile(tmp_path, environ={})
    # Neither the scheduler nor an arm's profile supplies the shared protocol.
    (tmp_path/'exp.cfg').write_text('Ignored by independent Elo.\n')
    arm = tmp_path/'arm'; arm.mkdir()
    match_profile(arm, '[match]\nvisits=999\n')
    match_profile(tmp_path, '[match]\nvisits=17\n')
    (tmp_path/'engine.cfg.local').write_text('[engine]\nvisits=19\n')
    (tmp_path/'match.cfg.local').write_text('[match]\nvisits=23\n')
    profile = load_elo_profile(tmp_path, environ={'MATCH_VISITS': '29', 'MATCH_GAMES': 'invalid'})
    assert profile['settings'] == baseline
    assert profile['config']['match']['visits'] == 29
    assert profile['config']['match']['games'] == baseline['games_per_pair']
    assert profile['config']['match']['game_threads'] == baseline['game_threads']
    assert load_engine_config(tmp_path, True, environ={})['match']['visits'] == 23


@pytest.mark.parametrize('contents', ['[elo]\nunknown=1\n', '[match]\nvisits=17\n',
                                     '[experiment]\nautoelo=false\n'])
def test_elo_file_ownership_is_strict(tmp_path, contents):
    (tmp_path/'elo.cfg').write_text(contents)
    with pytest.raises(ValueError, match='Unknown key|Wrong section'):
        load_elo_config(tmp_path, environ={})


def test_resolved_elo_profile_survives_config_and_environment_changes(tmp_path, monkeypatch):
    config = tmp_path/'configs'; config.mkdir()
    match_profile(config, '[match]\nvisits=17\n')
    (config/'elo.cfg').write_text('[elo]\nsample_seconds=10\ngames_per_pair=12\n')
    profile = load_elo_profile(config, environ={'CUDA_VISIBLE_DEVICES': 'GPU-original'})
    (config/'match.cfg').unlink()
    (config/'elo.cfg').write_text('Changed after planning.\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-changed')
    monkeypatch.setattr('etazero.autoelo.verify_build', lambda _: 'binary')
    data = tmp_path/'data'
    history(data/'a', [10, 20])
    history(data/'b', [10, 20])
    plan = autoelo_plan(config, tmp_path/'binary', data=data, profile=profile)
    assert plan['manifest']['config']['match']['visits'] == 17
    assert plan['manifest']['games_per_pair'] == 12
    assert plan['manifest']['execution']['cuda_visible_devices'] == 'GPU-original'
    plan['manifest']['config']['match']['visits'] = 99
    assert profile['config']['match']['visits'] == 17


def test_independent_elo_discovers_config_arms_without_scheduler_settings(tmp_path, monkeypatch):
    directory = tmp_path/'configs'; directory.mkdir()
    (directory/'exp.cfg').write_text('Unused by independent evaluation.\n')
    match_profile(directory, '[match]\nboard_size=5\n')
    (directory/'elo.cfg').write_text('[elo]\nsample_seconds=10\n')
    for name in ('a', 'b'):
        derive_config(directory/name, CONFIGS/'smoke_test', f'run_dir={tmp_path/name}\n')
        history(tmp_path/name, [10, 20])
    monkeypatch.setattr('etazero.autoelo.verify_build', lambda _: 'binary')
    plan = autoelo_plan(directory, tmp_path/'binary', environ={})
    assert plan['summary']['players_per_arm'] == {'a': 2, 'b': 2}


def test_plan_is_read_only_preserves_identity_and_ignores_untrained_config_arms(tmp_path, monkeypatch):
    config = tmp_path/'config'; config.mkdir()
    (config/'elo.cfg').write_text('[elo]\nsample_seconds=10\n')
    match_profile(config, '[match]\nreuse_tree=false\nuse_graph_search=false\ncache_entries=0\n')
    (config/'untrained').mkdir()
    derive_config(config/'untrained', CONFIGS/'smoke_test', '')
    data = tmp_path/'data'
    history(data/'a', [10, 20, 30])
    history(data/'b', [12, 24, 36], 'muzero')
    binary = tmp_path/'binary'; binary.write_bytes(b'test')
    monkeypatch.setattr('etazero.autoelo.verify_build', lambda _: 'binary-checksum')
    plan = autoelo_plan(config, binary, data=data, environ={})
    overridden = autoelo_plan(config, binary, data=data, environ={}, overrides={'game_threads': '7'})
    assert overridden['manifest']['config']['match']['game_threads'] == 7
    assert overridden['output'] != plan['output']
    assert plan['summary']['players_per_arm'] == {'a': 3, 'b': 3}
    assert not (data/'elo').exists()
    assert autoelo_plan(config, binary, data=data, environ={})['output'] == plan['output']
    assert autoelo_plan(config, binary, data=data, environ={}, overrides={'games_per_pair': '40'})['output'] != plan['output']
    custom = autoelo_plan(config, binary, arms=[{'name':'alias-a','run_dir':str(data/'a')},
                                               {'name':'alias-b','run_dir':str(data/'b')}], environ={}, gpu='GPU-test')
    assert custom['summary']['players_per_arm'] == {'alias-a':3, 'alias-b':3}
    assert custom['manifest']['execution']['cuda_visible_devices'] == 'GPU-test'
    assert custom['manifest']['config']['match']['device'] == 'cuda:0'
    match_profile(config, '[match]\nreuse_tree=true\n')
    with pytest.raises(ValueError, match='MuZero matches require'):
        autoelo_plan(config, binary, data=data, environ={})


def test_shared_arena_never_fits_partial_results(tmp_path, monkeypatch):
    manifest = {'players': [], 'config': {'match': {'seed': 1}}, 'games_per_pair': 4,
                'pairs': [{'id': identity(i), 'players': []} for i in range(2)]}
    monkeypatch.setattr('etazero.arena.native_config', lambda *_: {})
    calls = []
    def execute(binary, config, output, manifest, limit, cancel, environ):
        assert limit == 3
        calls.append(cancel)
        raise RuntimeError('native failed')
    monkeypatch.setattr('etazero.arena.execute_arena', execute)
    monkeypatch.setattr('etazero.elo.write_ratings', lambda *_: calls.append(True))
    with pytest.raises(RuntimeError, match='native failed'):
        run_arena(tmp_path, manifest, Path('/unused'), 10, max_loaded_models=3)
    assert len(calls) == 1 and calls[0].is_set()


def test_shared_native_routes_pairs_deduplicates_models_and_resumes_one_side(tmp_path, monkeypatch):
    from dataclasses import asdict
    for arm in ('a', 'b', 'c'):
        history(tmp_path/arm, [1])
        # Different paths may carry the same exported checkpoint label.
    players = discover_players(tmp_path)
    manifest = {'players': [asdict(p) for p in players], 'games_per_pair': 4,
                'config': {'match': {'seed': 1, 'device': 'cpu', 'board_size': 5, 'rule': 'freestyle'}},
                'pairs': [{'id': identity(pair), 'players': pair} for pair in
                          ((players[0].id, players[1].id), (players[0].id, players[2].id))]}
    output = tmp_path/'results'; output.mkdir()
    binary = tmp_path/'child'
    binary.write_text(f'''#!{sys.executable}
import json,sys
from pathlib import Path
args=dict(zip(sys.argv[2::2],sys.argv[3::2]))
tasks=Path(args['--tasks']).read_text().splitlines()
print(json.dumps(dict(type='arena_progress',stage='loading',window=1,models=3,workers=3,loaded=0,seconds=0)),flush=True)
print(json.dumps(dict(type='arena_progress',stage='model_ready',window=1,models=3,workers=3,loaded=3,seconds=1)),flush=True)
print(json.dumps(dict(type='arena_progress',stage='playing',window=1,models=3,workers=3,loaded=3,seconds=1)),flush=True)
partial=not Path(args['--tasks']).with_suffix('.started').exists()
Path(args['--tasks']).with_suffix('.started').touch()
for line in tasks:
 p,a,b,i,seed,g,mask,n,*moves=line.split();i=int(i)
 if int(n)<0:
  print(json.dumps(dict(type='opening',pair=p,id=i,seed=int(seed),generator=int(g),moves=[12],value=0)),flush=True)
 for color in (0,1):
  if int(mask)&(1<<color):
   print(json.dumps(dict(type='game',pair=p,id=2*i+color,opening_id=i,black_a=color==0,winner=0,moves=[12,13],seconds=1)),flush=True)
   print(json.dumps(dict(type='game',pair=p,id=2*i+color,opening_id=i,black_a=color==0,winner=0,moves=[12,13],seconds=1)),flush=True)
   if partial: sys.exit(1)
print(json.dumps(dict(type='arena_stats',games=7)),flush=True)
''')
    binary.chmod(0o755)
    with pytest.raises(RuntimeError, match='Arena exited 1'):
        execute_arena(binary, tmp_path/'cfg', output, manifest, 3, threading.Event())
    assert len((output/'models.txt').read_text().splitlines()) == 3
    import shlex
    models = [shlex.split(line) for line in (output/'models.txt').read_text().splitlines()]
    assert {row[1] for row in models} == {player.model for player in players}
    assert all(row[0] == row[1] and int(row[2]) == 1 for row in models)
    first = output/'pairs'/manifest['pairs'][0]['id']/'games/00000000.json'
    original = first.read_bytes()
    from tqdm import tqdm
    bars = []
    def record_progress(*args, **kwargs):
        bar = tqdm(*args, **(kwargs | {'disable': False, 'file': io.StringIO()}))
        bars.append(bar)
        return bar
    monkeypatch.setattr('etazero.arena.tqdm', record_progress)
    execute_arena(binary, tmp_path/'cfg', output, manifest, 3, threading.Event())
    assert [shlex.split(line) for line in (output/'models.txt').read_text().splitlines()] == models
    progress = next(bar for bar in bars if bar.desc == 'Arena games')
    assert progress.initial == 1 and progress.n == 8
    assert progress.postfix == 'window=1 playing'
    fields = (output/'tasks.txt').read_text().splitlines()[0].split()
    assert fields[6:8] == ['2', '1']  # Only the missing white side, saved opening.
    assert first.read_bytes() == original
    assert sum(len(PairStore(output/'pairs'/p['id'], 4).games()) for p in manifest['pairs']) == 8
    binary.unlink()  # A fully complete schedule never starts the executable.
    execute_arena(binary, tmp_path/'cfg', output, manifest, 3, threading.Event())


def test_model_inspection_deduplicates_checks_and_rejects_changed_files(tmp_path, monkeypatch):
    from etazero.arena import inspect_models
    calls = []
    def inspect(path):
        calls.append(path)
        if path.name == 'changed':
            raise ValueError('changed model')
        return path, {'id': path.name}
    monkeypatch.setattr('etazero.arena.model_info', inspect)
    paths = [tmp_path/'a', tmp_path/'b', tmp_path/'a']
    assert list(inspect_models(paths)) == paths[:2]
    assert sorted(calls) == paths[:2]
    with pytest.raises(ValueError, match='changed model'):
        inspect_models([tmp_path/'changed'])


def test_autoelo_import_does_not_initialize_training_or_torch():
    import subprocess
    subprocess.run([sys.executable, '-c',
                    "import sys, etazero.autoelo; assert 'torch' not in sys.modules"], check=True)


def test_cancellation_stops_a_child_blocked_on_stdout_and_keeps_opening(tmp_path, monkeypatch):
    binary = tmp_path/'child'
    binary.write_text(f'#!{sys.executable}\nimport json,time\n'
                      'print(json.dumps(dict(type="opening",id=0,generator=0,seed=1,moves=[12],value=0)),flush=True)\n'
                      'time.sleep(5)\n')
    binary.chmod(0o755)
    output = tmp_path/'output'
    save_json(output/'manifest.json', {'config': {'match': {'device':'cpu','board_size':5,'rule':'renju'}}})
    store = PairStore(output/'pairs/ab', 4)
    cancel = threading.Event()
    original = store.accept
    def accept(event):
        original(event)
        cancel.set()
    store.accept = accept
    monkeypatch.setattr('etazero.arena.model_info', lambda path: (Path(path), {'id':str(path)}))
    players = tuple(Player(name, name, 1, 1, name) for name in ('a', 'b'))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='Match exited'):
        execute_pair(binary, tmp_path/'cfg', players, store, 1, cancel)
    assert time.monotonic()-started < 3
    assert (store.directory/'openings/00000000.json').exists()


def test_shared_arena_cancellation_flushes_an_opening(tmp_path, monkeypatch):
    from dataclasses import asdict
    pair_id = identity(('a', 'b'))
    players = [Player(name, name, 1, 1, name) for name in ('a', 'b')]
    manifest = dict(players=[asdict(p) for p in players], games_per_pair=4,
                    pairs=[dict(id=pair_id, players=['a', 'b'])],
                    config={'match': {'seed': 1, 'device': 'cpu', 'board_size': 5, 'rule': 'renju'}})
    binary = tmp_path/'child'
    binary.write_text(f'#!{sys.executable}\nimport json,time\n'
        f'print(json.dumps(dict(type="opening",pair="{pair_id}",id=0,generator=0,seed=1,moves=[12],value=0)),flush=True)\n'
        'time.sleep(5)\n')
    binary.chmod(0o755)
    monkeypatch.setattr('etazero.arena.model_info', lambda p: (Path(p), {'id': str(p)}))
    cancel = threading.Event()
    original = PairStore.accept
    def accept(self, event):
        original(self, event)
        cancel.set()
    monkeypatch.setattr(PairStore, 'accept', accept)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='Arena exited'):
        execute_arena(binary, tmp_path/'cfg', tmp_path, manifest, 2, cancel)
    assert time.monotonic()-started < 3
    assert (tmp_path/'pairs'/pair_id/'openings/00000000.json').exists()


def test_latest_pointer_and_native_performance_preserve_prior_results(tmp_path):
    first, second = tmp_path/'first', tmp_path/'second'
    first.mkdir(); second.mkdir()
    (first/'elo.png').write_bytes(b'first')
    publish_latest(tmp_path, first)
    publish_latest(tmp_path, second)
    assert (tmp_path/'latest').resolve() == second
    assert (first/'elo.png').read_bytes() == b'first'
    store = PairStore(tmp_path/'pair', 4)
    store.accept({'type':'match_stats','games':4,'seconds':1.5,'inference_a':{'requests':10}})
    assert len(list((store.directory/'performance').glob('*.json'))) == 1
    assert store.games() == []


@pytest.mark.parametrize('code', [0, 130])
def test_autoexp_invokes_elo_only_after_success_and_keeps_actual_arm_paths(tmp_path, monkeypatch, code):
    directory = tmp_path/'configs'; directory.mkdir()
    (directory/'exp.cfg').write_text('[experiment]\nmax_iteration=2\nmax_hours=0\narm_gpus=2\nshared_init=false\nautoelo=true\n')
    (directory/'a').mkdir()
    derive_config(directory/'a', CONFIGS/'smoke_test', f'run_dir={tmp_path/"custom-output"}\n')
    (directory/'b').mkdir()
    derive_config(directory/'b', CONFIGS/'smoke_test', f'run_dir={tmp_path/"another-output"}\n')
    match_profile(directory, '[match]\nboard_size=5\nvisits=17\n')
    plan = experiment_plan(directory, environ={}, work_dir=tmp_path/'controller')
    binary = tmp_path/'binary'; binary.write_bytes(b'binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: 'hash')
    def train(scheduler):
        (directory/'match.cfg').write_text('Changed during training.\n')
        (directory/'elo.cfg').write_text('Changed during training.\n')
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-changed')
        return code
    monkeypatch.setattr(Scheduler, 'run', train)
    calls = []
    def elo_plan(directory, binary, *, arms, profile):
        assert [arm['name'] for arm in arms] == ['a', 'b']
        assert profile == plan['elo']
        assert profile['config']['match']['visits'] == 17
        calls.append(([arm['run_dir'] for arm in arms], profile['cuda_visible_devices']))
        return {'summary': {}, 'output': str(tmp_path/'elo')}
    monkeypatch.setattr('etazero.autoelo.autoelo_plan', elo_plan)
    monkeypatch.setattr('etazero.autoelo.run_autoelo', lambda *_: None)
    assert run_experiment(plan, binary) == code
    assert calls == ([([str(tmp_path/'custom-output'), str(tmp_path/'another-output')], '2')]
                     if code == 0 else [])
    assert load_json(tmp_path/'controller/.internal/plan.json')['elo'] == plan['elo']
    if code == 0:
        assert load_json(tmp_path/'controller/.internal/elo_status.json')['status'] == 'complete'
        monkeypatch.setattr('etazero.autoelo.run_autoelo', lambda *_: (_ for _ in ()).throw(RuntimeError('elo failed')))
        with pytest.raises(RuntimeError, match='elo failed'):
            run_experiment(plan, binary)
        status = load_json(tmp_path/'controller/.internal/elo_status.json')
        assert status['training'] == 'complete' and status['status'] == 'failed'
