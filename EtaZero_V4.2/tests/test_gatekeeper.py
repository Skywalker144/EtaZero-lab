"""Source gatekeeper decisions, MuZero services and outer acceptance state."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest
import torch

from etazero import ROOT
from etazero.config import load_config
from etazero.run import checkpoint_progress
from export_muzero import export_components
from test_controller_recovery import recovery_config, wait_until
from test_muzero_model import make_model, limited_cpu_threads
from test_placement_network import placement_model

controller = importlib.import_module('etazero.run')
gpu = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and learner')


def gate_config(path, game='renju', moves=30, games=6, gumbel=False):
    path.write_text(f'@include {os.path.relpath(ROOT / "cpp/configs/training/gatekeeper1.cfg", path.parent)}\n'
        f'game = {game}\nnumGamesPerGating = {games}\nmaxMovesPerGame = {moves}\n'
        'bSizes = 3\nbSizeRelProbs = 1\nallowRectangleProb = 0\nkomiMean = 0\n'
        'numGameThreads = 2\nnumSearchThreads = 1\nmaxVisits = 8\nallowResignation = false\n'
        'nnMaxBatchSize = 4\n'
        'nnCacheSizePowerOfTwo = 8\n'
        'nnMutexPoolSizePowerOfTwo = 5\n'
        'useGraphSearch = false\nsubtreeValueBiasFactor = 0\nrootSymmetryPruning = false\n'
        f'rootSearchAlgorithm = {"gumbel" if gumbel else "puct"}\ngumbelMaxConsideredActions = 4\n'
        'cudaUseFP16 = true\n')
    return path


def test_gatekeeper_config_and_relative_path_ownership(tmp_path):
    cfg = recovery_config(tmp_path, 'alphazero')
    gate = gate_config(tmp_path / 'gate.cfg')
    for value in (-.1, 1.1, float('nan'), True):
        with pytest.raises(ValueError, match='gatekeeper_required_win_prop'):
            replace(cfg, gatekeeper_config=str(gate), gatekeeper_required_win_prop=value).validate()
    with pytest.raises(ValueError, match='requires gatekeeper_config'):
        replace(cfg, gatekeeper_required_win_prop=.6).validate()
    with pytest.raises(ValueError, match='Missing gatekeeper config'):
        replace(cfg, gatekeeper_config=str(tmp_path / 'absent')).validate()
    values = cfg.serializable()
    values.update(selfplay_config='selfplay.cfg', gatekeeper_config='gate.cfg')
    path = tmp_path / 'run.json'; path.write_text(json.dumps(values))
    assert load_config(path).gatekeeper_config == str(gate)


def test_native_gatekeeper_refuses_to_overwrite_committed_receipt(tmp_path):
    config = gate_config(tmp_path / 'gate.cfg')
    receipt = tmp_path / 'result.json'
    receipt.write_text('{"accepted": true}\n')
    committed = receipt.read_bytes()
    command = [str(ROOT / 'build/katago'), 'gatekeeper', '-config', str(config),
        '-test-models-dir', str(tmp_path / 'test'), '-accepted-models-dir', str(tmp_path / 'accepted'),
        '-rejected-models-dir', str(tmp_path / 'rejected'), '-sgf-output-dir', str(tmp_path / 'matches'),
        '-result-file', str(receipt)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'will not overwrite' in result.stdout + result.stderr
    assert receipt.read_bytes() == committed
    assert not (tmp_path / 'test').exists()


def native_command(tmp_path, game, muzero, threshold, *, moves=30, games=6, gumbel=False):
    baseline = tmp_path / 'accepted/baseline'
    baseline.mkdir(parents=True)
    if muzero:
        baseline.rmdir()
        export_components(make_model(game), baseline, 'baseline')
    else:
        model = placement_model()
        checkpoint = tmp_path / 'model.ckpt'
        torch.save({'config': model.config, 'model': model.state_dict()}, checkpoint)
        result = subprocess.run([sys.executable, str(ROOT / 'python/export_model_pytorch.py'),
            '-checkpoint', str(checkpoint), '-export-dir', str(baseline), '-model-name', 'baseline',
            '-filename-prefix', 'model'], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    candidate = tmp_path / 'test/candidate'
    shutil.copytree(baseline, candidate)
    receipt = tmp_path / 'result.json'
    config = gate_config(tmp_path / 'gate.cfg', game, moves, games, gumbel)
    command = [str(ROOT / 'build/katago'), 'gatekeeper', '-config', str(config),
        '-test-models-dir', str(candidate.parent), '-accepted-models-dir', str(baseline.parent),
        '-rejected-models-dir', str(tmp_path / 'rejected'), '-sgf-output-dir', str(tmp_path / 'matches'),
        '-required-candidate-win-prop', str(threshold), '-no-autoreject-old-models',
        '-result-file', str(receipt), '-override-config', f'useMuZero={str(muzero).lower()}']
    return command, receipt


@gpu
@pytest.mark.parametrize('game,muzero,threshold,gumbel', [
    ('renju', False, .5, False), ('renju', True, .75, True),
    ('hex', True, 0., False), ('go', True, 0., False),
])
def test_native_gatekeeper_uses_real_results_and_moves_whole_models(tmp_path, game, muzero, threshold, gumbel):
    command, receipt = native_command(tmp_path, game, muzero, threshold, gumbel=gumbel)
    original = {p.name: p.read_bytes() for p in (tmp_path / 'test/candidate').iterdir()}
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(receipt.read_text())
    assert report['candidate'] == 'candidate' and report['baseline'] == 'baseline'
    assert report['planned_games'] == 6 and 1 <= report['games'] <= 6
    assert sum(report['outcomes'].values()) == report['games']
    assert report['candidate_points'] + report['baseline_points'] == report['games']
    assert report['accepted'] == (report['candidate_points'] + 1e-10 >= threshold * report['games'])
    if game == 'renju':
        assert report['outcomes'] == dict(win=0, draw=report['games'], loss=0, no_result=0)
        assert report['candidate_points'] == report['games'] * .5
        assert report['accepted'] == (threshold == .5)
    else:
        assert report['accepted']
    destination = tmp_path / ('accepted' if report['accepted'] else 'rejected') / 'candidate'
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == original
    assert not (tmp_path / 'test/candidate').exists()
    assert list((tmp_path / 'matches/candidate').glob('*.sgfs'))


@gpu
def test_native_gatekeeper_refuses_nonterminal_placement_adjudication(tmp_path):
    command, receipt = native_command(tmp_path, 'renju', False, .5, moves=1)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'move limit without a rule result' in result.stdout + result.stderr
    assert not receipt.exists()
    assert (tmp_path / 'test/candidate').is_dir()


@gpu
def test_native_gatekeeper_signal_does_not_commit_a_partial_match(tmp_path):
    command, receipt = native_command(tmp_path, 'renju', True, .5, games=1000000)
    log = tmp_path / 'gate.log'
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: 'starting game testing candidate' in log.read_text(), seconds=30)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    assert not receipt.exists() and (tmp_path / 'test/candidate').is_dir()


@gpu
@pytest.mark.parametrize('algorithm,threshold', [('alphazero', .75), ('gumbel_muzero', .5)])
def test_outer_gatekeeper_resume_and_next_round_model(tmp_path, monkeypatch, algorithm, threshold):
    profile = gate_config(tmp_path / 'gate.cfg', gumbel='gumbel' in algorithm)
    config = replace(recovery_config(tmp_path, algorithm),
                     gatekeeper_config=str(profile), gatekeeper_required_win_prop=threshold)
    root = tmp_path / 'run'
    calls = []
    actual_execute = controller.execute
    interrupt = [True]
    def execute(command, log, **kwargs):
        name = Path(log).name.split('.')[0]
        calls.append(name)
        actual_execute(command, log, **kwargs)
        if name == 'gatekeeper' and interrupt[0]:
            interrupt[0] = False
            raise RuntimeError('injected after gate decision')
    monkeypatch.setattr(controller, 'execute', execute)
    first = controller.run(root, config, 1)
    assert 'gatekeeper' not in calls  # Initial trained model establishes the baseline.
    with pytest.raises(RuntimeError, match='injected after gate'):
        controller.run(root, config, 2)
    pending = json.loads((root / 'state.json').read_text())
    assert pending['completed_rounds'] == 1 and pending['accepted_model'] == first['accepted_model']
    trained = checkpoint_progress(root / 'train/checkpoint.ckpt')
    assert trained[0] > first['total_samples']
    # Recreate the crash gap before native directory move, after decision commit.
    gate = next((root / 'rounds/000002/gatekeeper').iterdir())
    candidate_name = gate.name
    report = json.loads((gate / 'result.json').read_text())
    moved = gate / ('accepted' if report['accepted'] else 'rejected') / candidate_name
    moved.rename(gate / 'test' / candidate_name)
    count = len(calls)
    second = controller.run(root, config, 2)
    assert len(calls) == count and moved.exists()
    assert checkpoint_progress(root / 'train/checkpoint.ckpt') == trained
    expected = f'models/{candidate_name}' if threshold == .5 else first['accepted_model']
    assert second['accepted_model'] == expected
    assert (root / 'models' / candidate_name / 'model.bin').exists() or (root / 'models' / candidate_name / 'model.json').exists()
    third = controller.run(root, config, 3)
    third_report = json.loads((root / 'rounds/000003/result.json').read_text())
    assert third_report['input_model'] == Path(expected).name
    assert third_report['gatekeeper']['baseline'] == Path(expected).name
    assert third['total_samples'] > second['total_samples']
    assert checkpoint_progress(root / 'train/checkpoint.ckpt')[0] > trained[0]
