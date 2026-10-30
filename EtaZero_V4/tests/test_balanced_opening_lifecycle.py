"""Opening cancellation, explicit starts and native gating boundaries."""
import json
import os
import shutil
import signal
import subprocess
import time

import numpy as np
import pytest

from etazero import ROOT
from test_balanced_opening import configure, selfplay
from test_controller_recovery import wait_until
from test_gatekeeper import native_command
from test_match import export_models

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA')


def ordinary_models(tmp_path, game):
    ordinary, _ = export_models(tmp_path, game)
    model = tmp_path / 'models/published'; model.mkdir(parents=True)
    shutil.copyfile(ordinary, model / 'model.bin')
    return model.parent


def test_explicit_empty_sgf_is_not_replaced_by_balanced_opening(tmp_path):
    models = ordinary_models(tmp_path, 'renju')
    positions = tmp_path / 'positions'; positions.mkdir()
    (positions / 'empty.sgf').write_text('(;GM[4]RU[renju]SZ[5])')
    cfg = configure(tmp_path / 'selfplay.cfg', 'renju', startPosesFromSgfDir=positions,
                    startPosesProb=1, startPosesLoadProb=1, startPosesTurnWeightLambda=0)
    files = selfplay(tmp_path, cfg, models)
    assert files
    for file in files:
        with np.load(file) as data:
            target = data['globalTargetsNC']
            assert (target[:, 55] == 4).all()  # Source SGF-position mode.
            assert (target[:, 53] == 0).all()
    assert 'Balanced opening:' not in (tmp_path / 'native.log').read_text()


def test_signal_during_rejection_sampling_discards_unfinished_game(tmp_path):
    models = ordinary_models(tmp_path, 'hex')
    # Make acceptance negligible so SIGTERM exercises the opening loop before
    # any search or training row. This is a test condition, not a recipe.
    cfg = configure(tmp_path / 'selfplay.cfg', 'hex', balanceOpeningDropPow=1e300,
                    minAcceptRate=1e-300, numGameThreads=1)
    command = [str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
               '-models-dir', str(models), '-output-dir', str(tmp_path / 'data'), '-max-games-total', '1']
    log = tmp_path / 'native.log'
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: 'starting game on new neural net' in log.read_text(), seconds=30)
            time.sleep(.25)
            assert process.poll() is None
            process.send_signal(signal.SIGTERM)
            code = process.wait(timeout=15)
            assert code == 0, log.read_text()
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    assert 'Exited cleanly after signal' in log.read_text()
    assert 'Balanced opening:' not in log.read_text()
    assert not list((tmp_path / 'data').glob('*/tdata/*.npz'))


@pytest.mark.parametrize('game,muzero', [('renju', False), ('hex', True)])
def test_gatekeeper_uses_shared_native_opening(tmp_path, game, muzero):
    command, receipt = native_command(tmp_path, game, muzero, .5, moves=100, games=4)
    with (tmp_path / 'gate.cfg').open('a') as stream:
        stream.write('balanceOpeningProb = 1\n')
        if game == 'hex': stream.write('makeOpeningFairRate = 1\n')
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(receipt.read_text())
    assert report['planned_games'] == 4 and report['games'] > 0
    assert 'Balanced opening:' in result.stdout + result.stderr
    assert sum(report['outcomes'].values()) == report['games']
