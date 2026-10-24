import itertools
import json
from pathlib import Path

from etazero.arena import PairStore
from etazero.gatekeeper import decision_from_points


def test_source_clinch_never_changes_with_remaining_results():
    for total, required in itertools.product((1, 2, 5, 8), (0, .5, .55, 1)):
        for played in range(total+1):
            for half_points in range(2*played+1):
                points = half_points/2
                decision = decision_from_points(points, played, total, required)
                if decision is not None:
                    for tail in range(2*(total-played)+1):
                        assert decision == (points+tail/2+1e-10 >= required*total)
    assert decision_from_points(2,4,4,.5) is True
    assert decision_from_points(0,0,0,.5) is True


def test_independent_openings_round_colors_and_resume(tmp_path):
    store = PairStore(tmp_path, 7)
    tasks = store.tasks(81)
    assert len(tasks) == 7
    assert all({t['generator'] for t in tasks[i:i+2]} == {0,1} for i in (0,2,4))
    assert len({t['seed'] for t in tasks}) == 7
    t = tasks[0]
    opening = dict(type='opening',id=t['id'],generator=t['generator'],seed=t['seed'],moves=[],value=0)
    store.accept(opening)
    store.accept(dict(type='game',id=0,opening_id=0,black_a=t['generator']==0,winner=0,
                      moves=[0,1],seconds=.1))
    assert store.tasks(81) == tasks[1:]
    assert PairStore(tmp_path,9).tasks(81)[:6] == tasks[1:]


def test_cancel_before_match_launch_does_not_publish_a_decision(tmp_path, monkeypatch):
    import pytest
    from etazero import gatekeeper
    monkeypatch.setattr(gatekeeper, 'load_engine_config', lambda *a, **kw: {'match': {'seed': 17}})
    monkeypatch.setattr(gatekeeper, 'native_config', lambda *a: {})
    # execute_pair returns before launching native when cancellation wins the race.
    monkeypatch.setattr(gatekeeper, 'execute_pair', lambda *a, **kw: None)
    config = {'gatekeeper': dict(enabled=True, games=4, required_candidate_win_prop=.5,
                                match_config='match.cfg')}
    base = dict(id='base', path='models/base/model.bin', checkpoint={'iteration': 1})
    candidate = dict(id='candidate', path='models/candidate/model.bin', checkpoint={'iteration': 2})
    with pytest.raises(InterruptedError, match='completed games retained'):
        gatekeeper.consider_candidate(tmp_path, config, Path('unused'), base, candidate)
    assert not (tmp_path/'gatekeeper/candidate/decision.json').exists()
