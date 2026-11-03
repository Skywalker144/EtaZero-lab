"""Source GameRunner over MuZero search, then real trajectory shuffle/learner."""
from dataclasses import asdict
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_muzero import export_components
from katago.train.load_model import load_checkpoint
from test_muzero_model import make_model
from test_muzero_data import shuffle_source
from test_muzero_learner import run_cli
from test_muzero_cuda import ieee_reference
from test_xiangqi_controller import selfplay_config, verify_native_output

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA')


@pytest.mark.parametrize('game,root,interior,nhwc', [
    ('go', 'puct', 'puct', True), ('renju', 'gumbel', 'puct', False),
    ('hex', 'gumbel', 'gumbel', False),
    ('xiangqi', 'puct', 'puct', True), ('xiangqi', 'gumbel', 'puct', False),
    ('xiangqi', 'gumbel', 'gumbel', True),
])
def test_native_selfplay_bundle(tmp_path, game, root, interior, nhwc):
    torch.manual_seed(83)
    model = make_model(game).eval()
    models = tmp_path / 'models'
    models.mkdir()
    export_components(model, models / 'published', 'native-selfplay')
    # A newer component is not a published model. Discovery only uses manifests.
    partial = models / 'partial'
    partial.mkdir()
    (partial / 'prediction.bin').write_bytes(b'unpublished')
    cfg = tmp_path / 'selfplay.cfg'
    cfg.write_text(f'''@include {os.path.relpath(ROOT / 'configs/checks/source_go.cfg', tmp_path)}
game = {game}
useMuZero = true
komiMean = 0
bSizes = 3
dataBoardLen = 5
maxMovesPerGame = 12
maxVisits = 12
numSearchThreads = 2
useGraphSearch = false
subtreeValueBiasFactor = 0
rootSymmetryPruning = false
rootSearchAlgorithm = {root}
nonrootSearchAlgorithm = {interior}
gumbelMaxConsideredActions = 4
numNNServerThreadsPerModel = 2
gpuToUse = 1023
cudaDeviceToUseModel0Thread0 = 0
cudaDeviceToUseModel0Thread1 = 0
cudaUseFP16 = false
cudaUseFP16-0 = true
inputsUseNHWC = true
cudaInputsUseNHWC0 = {str(nhwc).lower()}
cudaUseNHWC = {str(nhwc).lower()}
maxBoardSizeForNNBuffer = 4
maxBoardSizeForNNBuffer0 = 5
nnRandSeed = selfplay-bundle
''')
    if game == 'xiangqi':
        selfplay_config(cfg, model.pos_len, useMuZero='true', useGraphSearch='false', maxVisits=12,
            rootSearchAlgorithm=root, nonrootSearchAlgorithm=interior, gumbelMaxConsideredActions=4,
            numNNServerThreadsPerModel=2, gpuToUse=1023, cudaDeviceToUseModel0Thread0=0,
            cudaDeviceToUseModel0Thread1=0, cudaUseFP16='false', inputsUseNHWC='true',
            cudaInputsUseNHWC0=str(nhwc).lower(), cudaUseNHWC=str(nhwc).lower(),
            maxBoardSizeForNNBuffer=4, maxBoardSizeForNNBuffer0=model.pos_len,
            nnRandSeed='selfplay-bundle', **{'cudaUseFP16-0': 'true'})
    command = [str(ROOT / 'build/katago'), 'selfplay', '-config', str(cfg),
               '-models-dir', str(models), '-output-dir', str(tmp_path / 'selfplay'),
               '-max-games-total', '4']
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (tmp_path / 'native.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    shards = list((tmp_path / 'selfplay/published/tdata').glob('*.npz'))
    assert shards
    games = roots = 0
    for path in shards:
        with np.load(path) as data:
            games += len(data['muZeroGameOffsets']) - 1
            roots += len(data['muZeroRootNC'])
            assert data['muZeroGameOffsets'][-1] == len(data['muZeroActions'])
            assert np.isfinite(data['globalTargetsNC']).all()
    assert games == 4 and roots > 0
    if game == 'xiangqi':
        verify_native_output(tmp_path / 'selfplay', model.pos_len)
    assert not (tmp_path / 'selfplay/partial').exists()
    bad = subprocess.run(command + ['-override-config', f'dataBoardLen={11 if game == "xiangqi" else 4}'],
                         capture_output=True, text=True, timeout=30)
    assert bad.returncode != 0
    assert 'model canvas must equal dataBoardLen' in bad.stdout + bad.stderr


@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
def test_source_game_runner_muzero_to_learner(tmp_path, game):
    torch.manual_seed(82)
    model = make_model(game).eval()
    export_components(model, tmp_path / 'model', 'game-runner')
    cfg = tmp_path / 'play.cfg'
    cfg.write_text(f'''@include {os.path.relpath(ROOT / 'configs/checks/source_go.cfg', tmp_path)}
game = {game}
testFP16 = true
komiMean = 0
bSizes = 3
numSearchThreads = 2
dataBoardLen = 5
maxMovesPerGame = 12
maxVisits = 24
useGraphSearch = false
subtreeValueBiasFactor = 0
rootSymmetryPruning = false
rootNumSymmetriesToSample = 1
initGamesWithPolicy = true
policyInitAreaProp = 0.15
handicapProb = {1 if game == 'go' else 0}
{'numExtraBlackFixed = 1' if game == 'go' else ''}
cheapSearchProb = 1
cheapSearchVisits = 8
cheapSearchTargetWeight = 0
useReanalyze = true
reanalyzeProp = 1
reanalyzePolicySurpriseWeight = 1
reanalyzeValueSurpriseWeight = 1
reanalyzeSurpriseExponent = 0
reanalyzeUseOutcomeTargets = false
sidePositionProb = 1
earlyForkGameProb = 0.5
forkGameProb = 0.5
''')
    if game == 'xiangqi':
        selfplay_config(cfg, model.pos_len, testFP16='true', useGraphSearch='false', maxVisits=24,
            initGamesWithPolicy='true', policyInitAreaProp=.15, cheapSearchProb=1,
            cheapSearchVisits=8, cheapSearchTargetWeight=0, useReanalyze='true', reanalyzeProp=1,
            reanalyzePolicySurpriseWeight=1, reanalyzeValueSurpriseWeight=1,
            reanalyzeSurpriseExponent=0, reanalyzeUseOutcomeTargets='false',
            sidePositionProb=1, earlyForkGameProb=.5, forkGameProb=.5)
    output = tmp_path / 'play'
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-muzero-play',
                             str(tmp_path / 'model'), str(cfg), str(output)],
                            capture_output=True, text=True, timeout=240)
    (tmp_path / 'native.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'report.json').read_text())
    assert {row['mode'] for row in report} == {0, 1, 2}
    assert sum(row['side'] for row in report) > 0
    assert sum(row['tree_side'] for row in report) > 0
    assert sum(row['opening'] for row in report) > 0
    assert sum(row['used_fork'] for row in report) > 0
    if game == 'go': assert max(row['handicap'] for row in report) > 0
    for path in (output / 'data').glob('*.npz'):
        with np.load(path) as data:
            assert 'muZeroRootNC' in data
            assert data['muZeroGameOffsets'][-1] == len(data['muZeroActions'])
            assert np.all(data['globalTargetsNC'][:, 25] > 0)
    # Use the actual produced trajectories; no constructed search targets.
    shuffled = shuffle_source(output / 'data', tmp_path, steps=3, rows_per_file=1000, files_per_bucket=1)
    data_dir = tmp_path / 'data'; data_dir.mkdir()
    shutil.move(shuffled, data_dir / 'train')
    shutil.move(shuffled.with_suffix('.json'), data_dir / 'train.json')
    (data_dir / 'val').mkdir()
    network = tmp_path / 'network.json'
    network.write_text(json.dumps(asdict(model.network_config)))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    run_cli([ROOT / 'python/train.py', '-traindir', tmp_path / 'train', '-datadir', data_dir,
             '-muzero-config', network, '-pos-len', str(model.pos_len), '-batch-size', '4', '-samples-per-epoch', '8',
             '-max-epochs-this-instance', '1', '-no-compile', '-use-adamw', '-use-fp16', '-lookahead-k', '2',
             '-no-export', '-no-longterm-checkpoints', '-quit-if-no-data'], tmp_path / 'learner.log', env)
    checkpoint = load_checkpoint(tmp_path / 'train/checkpoint.ckpt')
    assert checkpoint['train_state']['global_step_samples'] > 0
    assert checkpoint['learner_runtime']['scaler']
    for component in ('representation.', 'dynamics.', 'prediction.'):
        assert any(value['step'] > 0 for key, value in checkpoint['optimizer']['state'].items()
                   if key in {p for group in checkpoint['optimizer']['param_groups']
                              if group['component_name'] == component[:-1] for p in group['params']})
