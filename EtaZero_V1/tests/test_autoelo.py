from config_samples import CONFIGS
from pathlib import Path
import sys
import threading
import time

import pytest

from etazero.arena import Player, PairStore, build_schedule, discover_players, execute_pair, identity, run_arena
from etazero.autoelo import autoelo_plan, load_elo_config, publish_latest
from etazero.config import ROOT, load_config
from etazero.eval_config import load_evaluation_config
from etazero.experiment import experiment_plan, run_experiment, Scheduler
from etazero.schema import CONTRACT_ID
from etazero.storage import load_json, save_json, sha256


def history(root, times, algorithm=None):
    save_json(root/'logs/iterations/000000.json', {'iteration': 0, 'elapsed_seconds': 0, 'model': None})
    for i, seconds in enumerate(times, 1):
        model = root/'models'/str(i)/'model.pt'
        model.parent.mkdir(parents=True)
        model.write_bytes(f'{root.name}:{i}'.encode())
        info = {'id': str(i), 'path': str(model.relative_to(root)), 'sha256': sha256(model),
                'canvas': 15, 'contract': CONTRACT_ID}
        if algorithm:
            info['algorithm'] = algorithm
        save_json(model.parent/'manifest.json', info)
        save_json(root/'logs/iterations'/f'{i:06d}.json',
                  {'iteration': i, 'elapsed_seconds': seconds, 'model': info})
    save_json(root/'.internal/state.json', {'iteration': len(times)+1, 'elapsed_seconds': times[-1]})


def test_time_sampling_uses_actual_seconds_and_pins_last(tmp_path):
    history(tmp_path/'a', [1, 2, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    players = discover_players(tmp_path, points=10)
    assert [p.seconds for p in players] == list(range(10, 101, 10))
    # Fewer unique models are reported honestly; do not fabricate missing points.
    history(tmp_path/'b', [1, 100])
    assert len([p for p in discover_players(tmp_path, points=10) if p.arm == 'b']) == 2
    explicit = discover_players(tmp_path, points=1, arms=[('custom-name', tmp_path/'a'), ('b', tmp_path/'b')])
    assert [(p.arm, p.seconds) for p in explicit] == [('custom-name', 100), ('b', 100)]
    with pytest.raises(ValueError, match='no committed state'):
        discover_players(tmp_path, points=10, arms=[('missing', tmp_path/'missing')])


def test_two_level_neighbors_three_bridges_and_final_pairs():
    players = [Player(f'{arm}:{i}', arm, i, i*10, '', '')
               for arm in ('a', 'b', 'c') for i in range(1, 11)]
    schedule = build_schedule(players, 2, cross_time_fractions=(.3, .6, .9), final_cross=True)
    assert len(schedule) == 63
    assert ('a:1', 'a:3') in schedule and ('a:1', 'a:4') not in schedule
    for i in (3, 6, 9, 10):
        assert (f'a:{i}', f'b:{i}') in schedule
        assert (f'a:{i}', f'c:{i}') in schedule
        assert (f'b:{i}', f'c:{i}') in schedule
    assert len(build_schedule(players, 2, cross_time_fractions=(1,), final_cross=True)) == 54
    with pytest.raises(ValueError, match='connected'):
        build_schedule(players, 2, cross_time_fractions=(), final_cross=False)


def test_bridge_times_use_all_arms_common_horizon():
    players = [Player(f'{arm}:{i}', arm, i, i*scale, '', '')
               for arm, scale in [('a', 10), ('b', 10), ('c', 5)] for i in range(1, 11)]
    schedule = build_schedule(players, 1, cross_time_fractions=(.6,), final_cross=False)
    assert ('a:3', 'b:3') in schedule
    assert ('a:3', 'c:6') in schedule
    assert ('a:6', 'b:6') not in schedule


def test_shared_profiles_are_independent_and_strict(tmp_path):
    (tmp_path/'elo.cfg').write_text('[elo]\npoints=4\npair_workers=2\n')
    (tmp_path/'elo.cfg.local').write_text('[elo]\npoints=5\n')
    assert load_elo_config(tmp_path, environ={})['points'] == 5
    assert load_elo_config(tmp_path, environ={'ELO_POINTS': '6'})['points'] == 6
    assert load_elo_config(tmp_path, environ={'ELO_POINTS': '6'}, overrides={'points': '7'})['points'] == 7
    for overrides in [{'games_per_pair': '6'}, {'points': '0'}, {'pair_workers': '0'},
                      {'cross_time_fractions': 'nan'}, {'cross_time_fractions': '.3,.3'}]:
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
    (config/'elo.cfg').write_text('[elo]\npoints=3\n')
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
    plan = experiment_plan(directory, environ={}, work_dir=tmp_path/'controller')
    binary = tmp_path/'binary'; binary.write_bytes(b'binary')
    monkeypatch.setattr('etazero.runtime.verify_build', lambda _: 'hash')
    monkeypatch.setattr(Scheduler, 'run', lambda _: code)
    calls = []
    def elo_plan(directory, binary, *, arms, gpu):
        calls.append((arms[0]['run_dir'], gpu))
        return {'summary': {}, 'output': str(tmp_path/'elo')}
    monkeypatch.setattr('etazero.autoelo.autoelo_plan', elo_plan)
    monkeypatch.setattr('etazero.autoelo.run_autoelo', lambda *_: None)
    assert run_experiment(plan, binary) == code
    assert calls == ([(str(tmp_path/'custom-output'), '2')] if code == 0 else [])
    if code == 0:
        assert load_json(tmp_path/'controller/.internal/elo_status.json')['status'] == 'complete'
        monkeypatch.setattr('etazero.autoelo.run_autoelo', lambda *_: (_ for _ in ()).throw(RuntimeError('elo failed')))
        with pytest.raises(RuntimeError, match='elo failed'):
            run_experiment(plan, binary)
        status = load_json(tmp_path/'controller/.internal/elo_status.json')
        assert status['training'] == 'complete' and status['status'] == 'failed'
