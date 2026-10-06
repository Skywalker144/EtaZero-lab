from config_samples import CONFIGS
from pathlib import Path
import sys
import threading
import time

import pytest

from etazero.arena import Player, PairStore, build_schedule, discover_players, execute_pair, identity, run_arena
from etazero.autoelo import autoelo_plan, load_elo_config, publish_latest
from etazero.autoelo import run_autoelo
from etazero.elo_cache import pair_identity, pair_seed, prepare_pairs, retain_pairs
from etazero.config import ROOT, load_config
from etazero.eval_config import load_evaluation_config
from etazero.experiment import experiment_plan, run_experiment, Scheduler
from etazero.schema import CONTRACT_ID
from etazero.storage import load_json, save_json, sha256


def history(root, times, algorithm=None):
    save_json(root/'logs/iterations/000000.json', {'iteration': 0, 'elapsed_seconds': 0, 'model': None})
    for i, seconds in enumerate(times, 1):
        model = root/'models'/str(i)/'model.pt'
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(f'{root.name}:{i}'.encode())
        info = {'id': str(i), 'path': str(model.relative_to(root)), 'sha256': sha256(model),
                'canvas': 15, 'contract': CONTRACT_ID}
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
    config = tmp_path/'config'; config.mkdir()
    (config/'elo.cfg').write_text('[elo]\nstride=8\ncross_seconds=1000\ngames_per_pair=4\nbootstrap_samples=2\n')
    (config/'match.cfg').write_text('[match]\ndevice=cpu\n')
    data = tmp_path/'data'
    for arm in ('b', 'c'):
        history(data/arm, list(range(1, 17)))
    monkeypatch.setattr('etazero.autoelo.verify_build', lambda _: 'binary')
    calls = []
    def native(output, manifest, binary, samples, workers, env):
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
    off_grid = autoelo_plan(config, tmp_path/'binary', data=data, environ={}, overrides={'stride':'10'})
    assert first['manifest']['anchor'] in {p['id'] for p in off_grid['manifest']['players']}


def test_cache_identity_invalidates_match_changes_but_not_fit_or_budget(tmp_path):
    manifest = {'players': [{'id':'a','sha256':'aa'}, {'id':'b','sha256':'bb'}],
                'config': {'match': {'seed':1, 'visits':100, 'games':4}},
                'binary_sha256':'bin', 'execution': {'pair_workers':1},
                'source_sha256': {'eval_config.py':'compiler', 'elo.py':'fit'}}
    pair = {'id':identity(('a','b')), 'players':['a','b']}
    import copy
    original = pair_identity(manifest, pair)
    other = copy.deepcopy(manifest)
    other.update(anchor='b', sampling={'stride':32}, games_per_pair=40)
    other['config']['match']['games'] = 40
    other['source_sha256']['elo.py'] = 'new-fit'
    other['players'][0].update(model='/moved/model.pt', seconds=1000)
    assert pair_identity(other, pair) == original
    for group, key, value in [('config', 'visits', 200), ('execution', 'pair_workers', 2),
                               ('source_sha256', 'eval_config.py', 'changed')]:
        changed = copy.deepcopy(manifest)
        target = changed[group]['match'] if group == 'config' else changed[group]
        target[key] = value
        assert pair_identity(changed, pair) != original
    changed = copy.deepcopy(manifest); changed['binary_sha256'] = 'new-bin'
    assert pair_identity(changed, pair) != original
    changed = copy.deepcopy(manifest); changed['players'][0]['sha256'] = 'new-model'
    assert pair_identity(changed, pair) != original


def test_cache_keeps_partial_opening_and_does_not_mix_independent_trials(tmp_path):
    manifest = {'players':[{'id':'a','sha256':'aa'}, {'id':'b','sha256':'bb'}],
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


def test_autoelo_failure_retains_cache_and_publishes_only_after_recovery(tmp_path, monkeypatch):
    config = tmp_path/'config'; config.mkdir()
    (config/'elo.cfg').write_text('[elo]\nstride=8\ngames_per_pair=4\nbootstrap_samples=2\n')
    (config/'match.cfg').write_text('[match]\ndevice=cpu\n')
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
    (tmp_path/'elo.cfg').write_text('[elo]\nstride=4\npair_workers=2\n')
    (tmp_path/'elo.cfg.local').write_text('[elo]\nstride=5\n')
    assert load_elo_config(tmp_path, environ={})['stride'] == 5
    assert load_elo_config(tmp_path, environ={'ELO_STRIDE': '6'})['stride'] == 6
    assert load_elo_config(tmp_path, environ={'ELO_STRIDE': '6'}, overrides={'stride': '7'})['stride'] == 7
    for overrides in [{'games_per_pair': '6'}, {'stride': '0'}, {'pair_workers': '0'},
                      {'cross_seconds': 'nan'}, {'cross_seconds': '0'}]:
        with pytest.raises(ValueError):
            load_elo_config(tmp_path, environ={}, overrides=overrides)
    training = load_config(CONFIGS / 'muzero')
    (tmp_path/'match.cfg').write_text('[match]\ngame_threads=20\nreuse_tree=false\nuse_graph_search=false\n')
    match = load_evaluation_config(tmp_path, match=True, umbrella=True, environ={})
    assert match['match']['game_threads'] == 20 and not match['match']['reuse_tree']
    assert load_config(CONFIGS / 'muzero') == training
    (tmp_path/'match.cfg').write_text('[match]\nunknown=1\n')
    with pytest.raises(ValueError, match='Unknown evaluation key'):
        load_evaluation_config(tmp_path, match=True, umbrella=True, environ={})


def test_plan_is_read_only_preserves_identity_and_ignores_untrained_config_arms(tmp_path, monkeypatch):
    config = tmp_path/'config'; config.mkdir()
    (config/'elo.cfg').write_text('[elo]\nstride=1\n')
    (config/'match.cfg').write_text('[match]\nreuse_tree=false\nuse_graph_search=false\ncache_entries=0\n')
    (config/'untrained').mkdir()
    (config/'untrained/run.cfg').write_text('[run]\nextends=baseline\n')
    data = tmp_path/'data'
    history(data/'a', [10, 20, 30])
    history(data/'b', [12, 24, 36], 'muzero')
    binary = tmp_path/'binary'; binary.write_bytes(b'test')
    monkeypatch.setattr('etazero.autoelo.verify_build', lambda _: 'binary-checksum')
    plan = autoelo_plan(config, binary, data=data, environ={})
    assert plan['summary']['players_per_arm'] == {'a': 3, 'b': 3}
    assert not (data/'elo').exists()
    assert autoelo_plan(config, binary, data=data, environ={})['output'] == plan['output']
    assert autoelo_plan(config, binary, data=data, environ={}, overrides={'games_per_pair': '40'})['output'] != plan['output']
    custom = autoelo_plan(config, binary, arms=[{'name':'alias-a','run_dir':str(data/'a')},
                                               {'name':'alias-b','run_dir':str(data/'b')}], environ={}, gpu='GPU-test')
    assert custom['summary']['players_per_arm'] == {'alias-a':3, 'alias-b':3}
    assert custom['manifest']['execution']['cuda_visible_devices'] == 'GPU-test'
    assert custom['manifest']['config']['match']['device'] == 'cuda:0'
    (config/'match.cfg').write_text('[match]\nreuse_tree=true\n')
    with pytest.raises(ValueError, match='MuZero matches require'):
        autoelo_plan(config, binary, data=data, environ={})


def test_concurrent_pairs_cancel_other_workers_and_never_fit_partial_results(tmp_path, monkeypatch):
    manifest = {'players': [], 'config': {'match': {'seed': 1}}, 'games_per_pair': 4,
                'pairs': [{'id': identity(i), 'players': []} for i in range(2)]}
    monkeypatch.setattr('etazero.arena.native_config', lambda *_: {})
    started = threading.Event()
    stopped = threading.Event()
    calls = []
    def execute(binary, config, players, store, seed, cancel, environ):
        if store.directory.name == manifest['pairs'][0]['id']:
            assert started.wait(3)
            raise RuntimeError('native failed')
        started.set()
        assert cancel.wait(3)
        stopped.set()
    monkeypatch.setattr('etazero.arena.execute_pair', execute)
    monkeypatch.setattr('etazero.elo.write_ratings', lambda *_: calls.append(True))
    with pytest.raises(RuntimeError, match='native failed'):
        run_arena(tmp_path, manifest, Path('/unused'), 10, pair_workers=2)
    assert stopped.is_set() and calls == []


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
    players = tuple(Player(name, name, 1, 1, name, '') for name in ('a', 'b'))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='Match exited'):
        execute_pair(binary, tmp_path/'cfg', players, store, 1, cancel)
    assert time.monotonic()-started < 3
    assert (store.directory/'openings/00000000.json').exists()


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
    (directory/'exp.cfg').write_text('[experiment]\nmax_iteration=2\nmax_seconds=0\narm_gpus=2\nshared_init=false\nautoelo=true\n')
    (directory/'a').mkdir()
    (directory/'a/run.cfg').write_text(f'[run]\nextends=smoke_test\nrun_dir={tmp_path/"custom-output"}\n')
    (directory/'b').mkdir()
    (directory/'b/run.cfg').write_text(f'[run]\nextends=smoke_test\nrun_dir={tmp_path/"another-output"}\n')
    plan = experiment_plan(directory, environ={}, work_dir=tmp_path/'controller')
    binary = tmp_path/'binary'; binary.write_bytes(b'binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: 'hash')
    monkeypatch.setattr(Scheduler, 'run', lambda _: code)
    calls = []
    def elo_plan(directory, binary, *, arms, gpu):
        assert [arm['name'] for arm in arms] == ['a', 'b']
        calls.append(([arm['run_dir'] for arm in arms], gpu))
        return {'summary': {}, 'output': str(tmp_path/'elo')}
    monkeypatch.setattr('etazero.autoelo.autoelo_plan', elo_plan)
    monkeypatch.setattr('etazero.autoelo.run_autoelo', lambda *_: None)
    assert run_experiment(plan, binary) == code
    assert calls == ([([str(tmp_path/'custom-output'), str(tmp_path/'another-output')], '2')]
                     if code == 0 else [])
    if code == 0:
        assert load_json(tmp_path/'controller/.internal/elo_status.json')['status'] == 'complete'
        monkeypatch.setattr('etazero.autoelo.run_autoelo', lambda *_: (_ for _ in ()).throw(RuntimeError('elo failed')))
        with pytest.raises(RuntimeError, match='elo failed'):
            run_experiment(plan, binary)
        status = load_json(tmp_path/'controller/.internal/elo_status.json')
        assert status['training'] == 'complete' and status['status'] == 'failed'
