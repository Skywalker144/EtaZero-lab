"""Real MuZero rounds: shared learner initialization, native data, publication and resume."""
from dataclasses import asdict
import json
import os

import numpy as np
import pytest
import torch

from etazero import ROOT, __version__
from etazero.config import RunConfig
from etazero.run import run
from export_muzero import export_components
from katago.train.load_model import load_checkpoint, load_model
from test_muzero_model import make_model, limited_cpu_threads
from test_xiangqi_controller import selfplay_config, verify_native_output


pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and learner')


@pytest.mark.parametrize('game,algorithm,interior,weights,compiled', [
    ('go', 'muzero', 'puct', 'ordinary', False),
    ('renju', 'gumbel_muzero', 'gumbel', 'swa', False),
    ('hex', 'gumbel_muzero', 'puct', 'ordinary', True),
    ('xiangqi', 'muzero', 'puct', 'ordinary', False),
    ('xiangqi', 'gumbel_muzero', 'gumbel', 'swa', False),
])
def test_muzero_two_round_publication_and_resume(tmp_path, game, algorithm, interior, weights, compiled):
    check_muzero_two_rounds(tmp_path, game, algorithm, interior, weights, compiled)


def check_muzero_two_rounds(tmp_path, game, algorithm, interior, weights, compiled, *, binary=None, fp16=True, backend_config=None, heads_only=False):
    cfg = tmp_path / 'selfplay.cfg'
    cfg.write_text(f'''@include {os.path.relpath(ROOT / 'configs/checks/source_go.cfg', tmp_path)}
game = {game}
komiMean = 0
bSizes = 3
maxMovesPerGame = 12
maxVisits = 12
numSearchThreads = 2
useGraphSearch = false
subtreeValueBiasFactor = 0
rootSymmetryPruning = false
nonrootSearchAlgorithm = {interior}
gumbelMaxConsideredActions = 4
cudaUseFP16 = {str(fp16).lower()}
''')
    canvas = (12 if weights == 'swa' else 10) if game == 'xiangqi' else 5
    if game == 'xiangqi':
        selfplay_config(cfg, canvas, useGraphSearch='false', maxVisits=12,
                        nonrootSearchAlgorithm=interior, gumbelMaxConsideredActions=4)
    if binary is not None:
        with cfg.open('a') as stream:
            options = dict(inputsUseNHWC='true', numNNServerThreadsPerModel=2, eigenUseFP16='false')
            options.update(backend_config or {})
            stream.write('\n' + ''.join(f'{key} = {value}\n' for key, value in options.items()))
    network = asdict(make_model(game, pos_len=canvas).network_config)
    if heads_only:
        network['prediction'].update(block_kind=[], trunk_num_channels=network['latent_channels'])
    config = RunConfig(selfplay_config=str(cfg), game=game, algorithm=algorithm,
        muzero_network=network, muzero_unroll_steps=3,
        publish_weights=weights, canvas=canvas, batch_size=4, games_per_round=4, samples_per_epoch=8,
        shuffle_rows_per_file=128, shuffle_keep_rows=512, shuffle_args=('-min-rows', '1'),
        train_args=('-use-adamw', *(('-use-fp16',) if fp16 else ()), '-lookahead-k', '2', '-swa-period-samples', '8',
                    '-no-longterm-checkpoints', *(() if compiled else ('-no-compile',)))).validate()
    root = tmp_path / 'run'
    first = run(root, config, 1, binary=binary)
    initial_bytes = (root / 'initial/checkpoint.ckpt').read_bytes()
    initial = load_checkpoint(root / 'initial/checkpoint.ckpt')
    first_checkpoint = load_checkpoint(root / 'train/checkpoint.ckpt')
    assert initial['train_state']['global_step_samples'] == 0
    assert initial['train_state']['export_cycle_counter'] == 0
    assert initial['swa_model']['n_averaged'] == 0
    assert not initial['optimizer']['state']
    assert first['total_samples'] > 0
    first_report = json.loads((root / 'rounds/000001/result.json').read_text())
    assert first_report['input_model'] == 'initial'
    first_files = {p.name: p.read_bytes() for p in (root / 'models/round-000001').iterdir()}
    second = run(root, config, 2, binary=binary)
    checkpoint = load_checkpoint(root / 'train/checkpoint.ckpt')
    assert second['total_samples'] > first['total_samples']
    assert second['completed_rounds'] == 2 and second['active_round'] is None
    assert checkpoint['train_state']['export_cycle_counter'] == 2
    assert checkpoint['config'] == {'algorithm': 'muzero', 'network': config.muzero_network}
    assert checkpoint['train_state']['muzero_modelnorm_baselines'] == initial['train_state']['muzero_modelnorm_baselines']
    assert checkpoint['learner_runtime']['train_rng'] != first_checkpoint['learner_runtime']['train_rng']
    assert bool(checkpoint['learner_runtime']['scaler']) == fp16
    for name in ('representation.', 'dynamics.', 'prediction.'):
        assert any(not torch.equal(initial['model'][k], v) for k, v in checkpoint['model'].items() if k.startswith(name))
    report = json.loads((root / 'rounds/000002/result.json').read_text())
    assert report['input_model'] == 'round-000001'
    assert report['samples_before'] == first['total_samples']
    assert report['published_weights'] == weights
    assert 'initialize' not in report['stage_seconds']
    assert (root / 'initial/checkpoint.ckpt').read_bytes() == initial_bytes
    assert {p.name: p.read_bytes() for p in (root / 'models/round-000001').iterdir()} == first_files
    for number, input_name in ((1, 'initial'), (2, 'round-000001')):
        frozen = root / 'rounds' / f'{number:06d}' / 'input_models' / input_name
        for file in ('representation.bin', 'dynamics.bin', 'prediction.bin', 'model.json'):
            assert (frozen / file).read_bytes() == (root / 'models' / input_name / file).read_bytes()
        shards = list((root / 'selfplay' / f'{number:06d}' / input_name / 'tdata').glob('*.npz'))
        assert shards
        if game == 'xiangqi':
            verify_native_output(root / 'selfplay' / f'{number:06d}', canvas)
        for shard in shards:
            with np.load(shard) as data:
                assert 'muZeroRootNC' in data and np.isfinite(data['globalTargetsNC']).all()
        metadata = json.loads((root / 'shuffleddata' / f'{number:06d}' / 'train.json').read_text())
        assert metadata['muzero_unroll_steps'] == 3 and metadata['muzero_weight_mean'] > 0
    # Independently select checkpoint weights and compare every published layer.
    # This catches exporting ordinary weights when SWA was requested.
    model, swa, _ = load_model(root / 'train/checkpoint.ckpt', weights == 'swa', 'cpu', pos_len=canvas)
    if weights == 'swa':
        assert checkpoint['swa_model']['n_averaged'] > 0
        assert any(not torch.equal(v, swa.state_dict()[k]) for k, v in model.state_dict().items())
    export_components(swa if weights == 'swa' else model, tmp_path / 'expected',
                      f'etazero-v{__version__.replace(".", "_")}-round-000002')
    for file in ('representation.bin', 'dynamics.bin', 'prediction.bin', 'model.json'):
        assert (root / 'models/round-000002' / file).read_bytes() == (tmp_path / 'expected' / file).read_bytes()
    modified = (root / 'train/checkpoint.ckpt').stat().st_mtime_ns
    assert run(root, config, 2, binary=binary) == second
    assert (root / 'train/checkpoint.ckpt').stat().st_mtime_ns == modified
