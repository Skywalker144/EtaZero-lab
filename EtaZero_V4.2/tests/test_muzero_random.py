"""Random MuZero uses the latent domain and produces real trajectory data without NN files."""
import json
import os
import subprocess

import numpy as np
import pytest

from etazero import ROOT
from test_xiangqi_controller import selfplay_config, verify_native_output


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
def test_random_position_static_domain_and_cache(tmp_path, game):
    request = tmp_path / 'request.json'
    response = tmp_path / 'response.json'
    request.write_text(json.dumps(dict(random_game=game, canvas=12 if game == 'xiangqi' else 5)))
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-muzero-position',
                             str(tmp_path / 'absent-model'), str(request), str(response)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = json.loads(response.read_text())
    for depth, output in enumerate(outputs):
        policy = np.array(output['policy'])
        assert np.isfinite(policy).all() and policy[policy >= 0].sum() == pytest.approx(1)
        value = np.array(output['value'])
        assert np.isfinite(value).all() and value.sum() == pytest.approx(1)
        assert (value[3] > 0) == (game == 'go')
        assert output['player'] == (2 if depth % 2 == 0 else 1)
        assert output['optimism'] == 0
        assert output['auxiliary'][0] == 0
        assert output['auxiliary'][1] == pytest.approx(.5 * np.log(2))
    assert outputs[1]['policy'] != outputs[3]['policy']


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('gumbel', [False, True])
def test_empty_models_selfplay_writes_muzero_trajectories(tmp_path, game, gumbel):
    cfg = tmp_path / 'selfplay.cfg'
    options = dict(useMuZero='true', useGraphSearch='false',
                   rootSearchAlgorithm='gumbel' if gumbel else 'puct',
                   nonrootSearchAlgorithm='gumbel' if gumbel else 'puct',
                   gumbelMaxConsideredActions=4)
    if game == 'xiangqi':
        selfplay_config(cfg, **options)
    else:
        cfg.write_text(f'''@include {os.path.relpath(ROOT / 'configs/checks/source_go.cfg', tmp_path)}
game = {game}
komiMean = 0
bSizes = 3
dataBoardLen = 5
maxMovesPerGame = 12
maxVisits = 12
numSearchThreads = 2
subtreeValueBiasFactor = 0
rootSymmetryPruning = false
''' + ''.join(f'{key} = {value}\n' for key, value in options.items()))
    models = tmp_path / 'models'
    models.mkdir()
    receipt = tmp_path / 'result.json'
    result = subprocess.run([str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
                             '-models-dir', str(models), '-output-dir', str(tmp_path / 'output'),
                             '-max-games-total', '4', '-result-file', str(receipt)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'MuZero random evaluator' in result.stdout
    shards = list((tmp_path / 'output/random/tdata').glob('*.npz'))
    assert shards and not list(models.iterdir())
    games = roots = 0
    for shard in shards:
        with np.load(shard) as data:
            games += len(data['muZeroGameOffsets']) - 1
            roots += len(data['muZeroRootNC'])
            assert data['muZeroGameOffsets'][-1] == len(data['muZeroActions'])
            assert np.isfinite(data['globalTargetsNC']).all()
    assert json.loads(receipt.read_text()) == {'games_requested': 4, 'completed': True}
    # After the opening and stochastic target-weight sampling, a short game
    # may contribute no sampled roots and therefore no written trajectory.
    assert 0 < games <= 4 and roots > 0
    if game == 'xiangqi':
        verify_native_output(tmp_path / 'output', 10)
