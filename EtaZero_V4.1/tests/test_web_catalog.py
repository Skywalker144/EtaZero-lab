"""Workbench discovery follows committed publications and source config paths."""
import json
from pathlib import Path

import pytest

from etazero.web_catalog import discover_catalog, run_catalog


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def make_run(root, algorithm='alphazero', game='renju'):
    canvas = 10 if game == 'xiangqi' else 5
    write(root / 'config.json', dict(game=game, algorithm=algorithm, canvas=canvas, publish_weights='swa',
                                   selfplay_config=str(root / 'profile/selfplay.cfg')))
    for number in [1, 2, 3]:
        bundle = root / f'models/round-{number:06d}'
        bundle.mkdir(parents=True)
        if 'muzero' in algorithm:
            write(bundle / 'model.json', dict(game=game, pos_len=canvas))
            for name in ['representation.bin', 'dynamics.bin', 'prediction.bin']:
                (bundle / name).touch()
        else:
            (bundle / 'model.bin').touch()
        # Round 2 is rejected; it is still a published, reviewable generation.
        accepted = 'models/round-000001' if number < 3 else 'models/round-000003'
        state = dict(completed_rounds=number, accepted_model=accepted, elapsed_seconds=number * 10,
                     timing_valid=True, timing_basis='committed_wall_excluding_compile')
        write(root / f'rounds/{number:06d}/commit.json', dict(state=state, candidate_model=str(bundle.relative_to(root))))
    write(root / 'state.json', state)
    return root


@pytest.mark.parametrize('algorithm', ['alphazero', 'gumbel_alphazero', 'muzero', 'gumbel_muzero'])
@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
def test_published_generations_current_pointer_and_config(tmp_path, algorithm, game):
    root = make_run(tmp_path / 'experiment/runs/arm', algorithm, game)
    profile = root / 'profile/gtp.cfg'; profile.parent.mkdir(); profile.write_text('@include common.cfg\n')
    # A concurrent exporter may have both a complete candidate and temp files;
    # neither is visible before a root-pointer-bounded round commit.
    write(root / 'models/round-000004/model.json', {})
    write(root / 'models/round-000004.tmp/model.json', {})
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    catalog = discover_catalog(tmp_path)
    assert catalog['errors'] == [] and len(catalog['runs']) == 1
    run = catalog['runs'][0]
    assert run['config'] == str(profile) and run['completed_rounds'] == 3
    models = run['models']
    assert [m['iteration'] for m in models] == [3, 2, 1]
    assert [m['accepted'] for m in models] == [True, False, True]
    assert [m['training_seconds'] for m in models] == [30, 20, 10]
    assert all(m['weights'] == 'swa' and m['algorithm'] == algorithm and m['game'] == game for m in models)
    assert run['current_model'] == models[0]['path']
    assert all(Path(m['path']).is_dir() if 'muzero' in algorithm else Path(m['path']).is_file() for m in models)
    assert all(path.read_bytes() == data for path, data in before.items())
    explicit = tmp_path / 'manual.cfg'; explicit.write_text('numSearchThreads = 1\n')
    assert discover_catalog(tmp_path, config=explicit)['runs'][0]['config'] == str(explicit)


def test_empty_and_reused_generations_do_not_invent_metadata(tmp_path):
    root = make_run(tmp_path / 'run')
    last = root / 'rounds/000003/commit.json'
    receipt = json.loads(last.read_text())
    receipt['candidate_model'] = None
    receipt['state']['accepted_model'] = 'models/round-000001'
    write(last, receipt)
    write(root / 'state.json', receipt['state'])
    run = run_catalog(root)
    assert [m['iteration'] for m in run['models']] == [2, 1]
    assert run['current_model'] == run['models'][-1]['id'] and run['config'] is None
    empty = tmp_path / 'empty'; empty.mkdir()
    write(empty / 'config.json', json.loads((root / 'config.json').read_text()))
    write(empty / 'state.json', dict(completed_rounds=0, accepted_model=None))
    catalog = discover_catalog(tmp_path)
    assert len(catalog['runs']) == 2
    assert next(r for r in catalog['runs'] if r['id'] == str(empty))['models'] == []
    assert discover_catalog(tmp_path / 'not-yet-created') == dict(runs=[], errors=[])


def test_bootstrap_initial_model_is_visible_before_first_training(tmp_path):
    root = make_run(tmp_path / 'run')
    initial = root / 'models/initial'
    initial.mkdir()
    (initial / 'model.bin').touch()
    state = dict(completed_rounds=0, accepted_model='models/initial', elapsed_seconds=1,
                 timing_valid=True, timing_basis='committed_wall_excluding_compile')
    write(root / 'rounds/000000/commit.json', dict(state=state, candidate_model=None))
    write(root / 'state.json', state)
    result = run_catalog(root)
    assert len(result['models']) == 1 and result['models'][0]['iteration'] == 0
    assert result['models'][0]['weights'] == 'ordinary'
    assert result['current_model'] == str(initial / 'model.bin')
    from etazero.arena import discover_players
    write(root / 'state.json', json.loads((root / 'rounds/000001/commit.json').read_text())['state'])
    assert [p.iteration for p in discover_players(root)] == [0, 1]


@pytest.mark.parametrize('seconds,valid', [(float('nan'), True), (20, False)])
def test_missing_timing_is_not_replaced_by_zero(tmp_path, seconds, valid):
    root = make_run(tmp_path / 'run')
    path = root / 'rounds/000002/commit.json'
    receipt = json.loads(path.read_text())
    receipt['state'].update(elapsed_seconds=seconds, timing_valid=valid)
    write(path, receipt)
    model = run_catalog(root)['models'][1]
    assert model['training_seconds'] is None and model['timing_source'] is None


@pytest.mark.parametrize('failure', ['missing_model', 'wrong_game', 'pointer', 'missing_commit'])
def test_bad_run_is_visible_and_does_not_hide_other_runs(tmp_path, failure):
    bad = make_run(tmp_path / 'bad', 'muzero')
    good = make_run(tmp_path / 'good')
    if failure == 'missing_model':
        (bad / 'models/round-000001/dynamics.bin').unlink()
    elif failure == 'wrong_game':
        write(bad / 'models/round-000001/model.json', dict(game='hex', pos_len=5))
    elif failure == 'pointer':
        state = json.loads((bad / 'state.json').read_text())
        state['accepted_model'] = 'models/round-000002'
        write(bad / 'state.json', state)
    else:
        (bad / 'rounds/000002/commit.json').unlink()
    catalog = discover_catalog(tmp_path)
    assert [run['id'] for run in catalog['runs']] == [str(good)]
    assert len(catalog['errors']) == 1 and catalog['errors'][0]['run'] == str(bad)
    assert catalog['errors'][0]['error']


def test_trained_board_sizes_use_saved_sampling_weights(tmp_path):
    root = make_run(tmp_path / 'run', game='hex')
    config = json.loads((root / 'config.json').read_text())
    config['native_selfplay'] = 'bSizes = 5,3,4,7\nbSizeRelProbs = 2,0,1,1\n'
    write(root / 'config.json', config)
    assert run_catalog(root)['trained_board_sizes'] == [4, 5]
    config['native_selfplay'] = 'bSizesXY = 3-3,5-4,5-5\nbSizeRelProbs = 1,1,0\n'
    write(root / 'config.json', config)
    assert run_catalog(root)['trained_board_sizes'] == [3]
    config['native_selfplay'] = 'bSizes = 3,5\nbSizeRelProbs = 1\n'
    write(root / 'config.json', config)
    assert run_catalog(root)['trained_board_sizes'] is None
