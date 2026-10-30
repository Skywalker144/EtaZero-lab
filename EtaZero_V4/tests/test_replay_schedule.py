"""Outer accounting at source epoch boundaries, with real native/learner checks."""
from dataclasses import replace
import importlib
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from etazero.config import RunConfig
from etazero import ROOT
from etazero.replay import plan_games, plan_training, commit_replay, count_rows
from etazero.run import checkpoint_progress
from test_controller_recovery import recovery_config

controller = importlib.import_module('etazero.run')


def config(**overrides):
    return replace(RunConfig(selfplay_config=str(ROOT / 'configs/checks/source_go.cfg'),
        canvas=7, batch_size=8, games_per_round=10, samples_per_epoch=64,
        epochs_per_round=2, model_kind='b2c16-fson-mish', replay_ratio=2), **overrides).validate()


@pytest.mark.parametrize('value', [0, -1, True, '2', float('nan'), float('inf')])
def test_invalid_replay_ratio_is_rejected(value):
    with pytest.raises(ValueError, match='replay_ratio'):
        config(replay_ratio=value)


def test_replay_settings_do_not_create_two_budget_owners():
    with pytest.raises(ValueError, match='requires replay_ratio'):
        config(replay_ratio=None, bootstrap_min_rows=100)
    for rows in (True, 0, 7, 8.5):
        with pytest.raises(ValueError, match='bootstrap_min_rows'):
            config(bootstrap_min_rows=rows)
    for args in [('-max-train-bucket-per-new-data', '2'), ('-initial-train-bucket-level=9',),
                 ('-stop-when-train-bucket-limited',), ('-max-train-steps-since-last-reload', '100')]:
        with pytest.raises(ValueError, match='outer controller'):
            config(train_args=args)


def test_game_planning_uses_weighted_recent_history_and_nominal_work():
    cfg = config()
    assert plan_games(cfg, []) == 10
    # Only the last two rounds: 40 rows / 10 games => 4 rows/game.
    history = [{'rows': 10000, 'games': 1}, {'rows': 10, 'games': 1}, {'rows': 30, 'games': 9}]
    assert plan_games(cfg, history) == 16  # 128 samples / ratio 2 / 4 rows per game.
    assert plan_games(config(replay_ratio=1000), history) == 1
    assert plan_games(config(replay_ratio=None), history) == 10


def test_credit_is_charged_for_actual_samples_and_debt_survives_zero_epoch_rounds():
    cfg = config()
    replay = dict(rows=100, origin_rows=100, credit=4., history=[dict(rows=100, games=10)])
    plan = plan_training(cfg, replay, new_rows=40)
    assert plan == {'epochs': 1, 'available_credit': 84.}
    # A whole source file yields 120 samples even though the nominal epoch is 64.
    first = commit_replay(replay, plan, dict(rows=40, games=4), consumed_samples=120)
    assert first['credit'] == -36 and first['rows'] == 140
    assert replay['credit'] == 4 and replay['rows'] == 100  # Commit is not an in-place grant.
    plan = plan_training(cfg, first, new_rows=30)
    assert plan == {'epochs': 0, 'available_credit': 24.}
    second = commit_replay(first, plan, dict(rows=30, games=3), consumed_samples=0)
    assert second['credit'] == 24 and second['rows'] == 170
    assert len(second['history']) == 2 and second['origin_rows'] == 100
    assert plan_training(cfg, second, new_rows=20)['epochs'] == 1
    assert plan_training(cfg, second, new_rows=2000)['epochs'] == 2


def test_bootstrap_corpus_does_not_issue_ordinary_credit():
    cfg = config()
    replay = dict(rows=0, origin_rows=None, credit=0., history=[])
    plan = plan_training(cfg, replay, 80)
    result = commit_replay(replay, plan, dict(rows=80, games=10), consumed_samples=144)
    assert plan['epochs'] == 2
    assert result['origin_rows'] == result['rows'] == 80
    assert result['credit'] == 0


@pytest.mark.parametrize('native_names', [False, True])
def test_count_rows_uses_retained_roots_not_muzero_trajectory_positions(tmp_path, native_names):
    assert count_rows(tmp_path) == 0
    for index, rows in enumerate((2, 3)):
        path = tmp_path / f'{index}.npz'
        np.savez(path, binaryInputNCHWPacked=np.zeros((rows, 5, 4), dtype=np.uint8),
                 muZeroGlobalInputNC=np.zeros((50, 8), dtype=np.float32))
        if native_names:
            with zipfile.ZipFile(path) as archive:
                entries = {name.removesuffix('.npy'): archive.read(name) for name in archive.namelist()}
            with zipfile.ZipFile(path, 'w') as archive:
                for name, content in entries.items():
                    archive.writestr(name, content)
    assert count_rows(tmp_path) == 5


def test_zero_row_games_do_not_issue_credit_and_trigger_visible_recalibration(caplog):
    cfg = config()
    replay = dict(rows=100, origin_rows=100, credit=-1., history=[dict(rows=0, games=10)])
    plan = plan_training(cfg, replay, new_rows=0)
    assert plan == {'epochs': 0, 'available_credit': -1.}
    result = commit_replay(replay, plan, dict(rows=0, games=20), consumed_samples=0)
    assert result['credit'] == -1 and result['rows'] == 100
    assert plan_games(cfg, result['history']) == cfg.games_per_round
    assert 'recalibrating' in caplog.text


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and learner')
@pytest.mark.parametrize('algorithm', ['alphazero', 'gumbel_muzero'])
def test_real_replay_rounds_bootstrap_debt_and_resume(tmp_path, monkeypatch, algorithm):
    cfg = replace(recovery_config(tmp_path, algorithm), games_per_round=1,
                  replay_ratio=2., bootstrap_min_rows=64)
    if cfg.is_muzero:
        from test_gatekeeper import gate_config
        cfg = replace(cfg, gatekeeper_config=str(gate_config(tmp_path / 'gate.cfg', gumbel=True)),
                      gatekeeper_required_win_prop=.75)
    root = tmp_path / 'run'
    calls = []
    execute = controller.execute
    interrupt = [True]
    zero_rows = [False]

    def run_stage(command, log, **kwargs):
        stage = Path(log).name.split('.')[0]
        calls.append(stage)
        if stage == 'selfplay' and zero_rows[0]:
            command = list(command)
            index = command.index('-override-config') + 1
            command[index] += (',cheapSearchProb=1,cheapSearchVisits=8,cheapSearchTargetWeight=0,'
                               'policySurpriseDataWeight=0,valueSurpriseDataWeight=0,recordTreePositions=false')
        execute(command, log, **kwargs)
        # Exercise receipt adoption in the middle of bootstrap supplementing.
        if stage == 'selfplay' and interrupt[0]:
            interrupt[0] = False
            raise RuntimeError('injected bootstrap receipt loss')

    monkeypatch.setattr(controller, 'execute', run_stage)
    with pytest.raises(RuntimeError, match='injected bootstrap'):
        controller.run(root, cfg, 1)
    saved = json.loads((root / 'state.json').read_text())
    assert saved['replay'] == dict(rows=0, origin_rows=None, credit=0., history=[])
    initial_calls = calls.count('selfplay')
    first = controller.run(root, cfg, 1)
    data = root / 'selfplay/000001' / ('initial' if cfg.is_muzero else 'random') / 'tdata'
    rows = count_rows(data)
    assert rows >= cfg.bootstrap_min_rows and first['replay']['origin_rows'] == rows
    assert first['replay']['credit'] == 0
    assert calls.count('selfplay') > initial_calls  # Calibration required more complete games.
    assert checkpoint_progress(root / 'train/checkpoint.ckpt')[1] == cfg.epochs_per_round
    assert first['replay']['history'][0]['games'] > 1
    assert all(not (root / 'rounds/000001/logs' / name).exists() for name in ('initialize.2.log', 'initial_export.2.log'))

    # Interrupt after real training commits: resume must not issue the row credit twice.
    save = controller.save_json
    fail = [True]
    def save_state(path, value):
        if (fail[0] and Path(path).name == 'state.json' and value.get('active_round') == 2
                and 'train' in value['round_progress']['done']):
            fail[0] = False
            raise RuntimeError('injected credit commit boundary')
        save(path, value)
    monkeypatch.setattr(controller, 'save_json', save_state)
    with pytest.raises(RuntimeError, match='injected credit'):
        controller.run(root, cfg, 2)
    pending = json.loads((root / 'state.json').read_text())
    assert pending['replay'] == first['replay']
    training_calls = calls.count('train')
    second = controller.run(root, cfg, 2)
    assert calls.count('train') == training_calls
    report = json.loads((root / 'rounds/000002/result.json').read_text())
    gained = report['sampling']['rows'] * cfg.replay_ratio
    spent = second['total_samples'] - first['total_samples']
    assert second['replay']['credit'] == gained - spent
    assert report['actual_replay_ratio'] == second['total_samples'] / second['replay']['rows']
    assert report['sampling']['games'] == plan_games(cfg, first['replay']['history'])
    if cfg.is_muzero:
        assert report['gatekeeper']['accepted'] is False
        assert second['accepted_model'] == first['accepted_model']

    # Use source PCR to make the next completed games retain zero roots. This
    # guarantees no new credit: ordinary surprise weighting can otherwise vary
    # the row count enough to pay off the debt, legitimately allowing training.
    zero_rows[0] = True
    checkpoint = (root / 'train/checkpoint.ckpt').read_bytes()
    third = controller.run(root, cfg, 3)
    report3 = json.loads((root / 'rounds/000003/result.json').read_text())
    assert report3['sampling']['rows'] == 0 and report3['sampling']['games'] > 0
    available = second['replay']['credit'] + report3['sampling']['rows'] * cfg.replay_ratio
    assert report3['training_epochs'] == min(cfg.epochs_per_round, max(0, math.floor(available / cfg.samples_per_epoch)))
    assert second['replay']['credit'] < 0 and report3['training_epochs'] == 0
    assert third['total_samples'] == second['total_samples']
    assert third['accepted_model'] == second['accepted_model']
    assert (root / 'train/checkpoint.ckpt').read_bytes() == checkpoint
    assert not (root / 'models/round-000003').exists()
    assert 'train' not in report3['stage_attempts'] and 'export' not in report3['stage_attempts']
    assert 'gatekeeper' not in report3['stage_attempts']
    assert report3['gatekeeper'] == {'accepted': None, 'reason': 'no_training'}
    before = len(calls)
    assert controller.run(root, cfg, 3) == third
    assert len(calls) == before
    assert third['replay']['rows'] == second['replay']['rows']
    assert third['replay']['credit'] == second['replay']['credit']
