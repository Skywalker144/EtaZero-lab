"""Real CUDA multi-model scheduling, residency changes and partial-game recovery."""
from dataclasses import asdict
import json
import os

import pytest
import torch

from config_samples import CONFIGS
from etazero.arena import Player, identity, run_arena
from etazero.config import ROOT, load_config, write_native
from etazero.engine_config import load_engine_config
from etazero.export import export_model
from etazero.storage import load_json, sha256
from etazero.training import initialize

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA required')


@pytest.mark.parametrize('rule', ['renju', 'hex'])
def test_mixed_arena_residency_and_partial_resume(tmp_path, rule):
    torch.set_num_threads(1)
    players = []
    for index, profile in enumerate(('smoke_test', 'muzero_minimal_test', 'smoke_test')):
        config = load_config(CONFIGS/profile)
        config['network'].update(canvas=6, channels=24, blocks=1)
        if config['agent']['algorithm'] == 'muzero':
            config['muzero'].update(latent_channels=16, dynamics_channels=24, dynamics_blocks=1,
                                    prediction_channels=24, prediction_blocks=1)
        run = tmp_path/f'model_{index}'
        write_native(config, run/'config/effective.cfg')
        checkpoint = initialize(run, config)
        exported = export_model(run, config, checkpoint)
        players.append(Player(str(index), str(index), 1, index+1, str(run/exported['path']), exported['sha256']))
    config = load_engine_config(CONFIGS/'muzero', match=True, environ={})
    config['match'].update(board_size=5, rule=rule, visits=4, game_threads=7, max_batch=8,
                           server_threads=2, cache_entries=8, inference_precision='float16')
    pairs = [('0', '1'), ('1', '2'), ('0', '2')]
    binary = ROOT/'build/etazero'
    manifest = dict(players=[asdict(p) for p in players], config=config, games_per_pair=4,
                    pairs=[dict(id=identity(pair), players=list(pair)) for pair in pairs],
                    anchor='0', binary_sha256=sha256(binary), execution={'max_loaded_models': 2})
    output = tmp_path/'arena'
    result = run_arena(output, manifest, binary, 2, 2)
    assert result['games'] == 12
    files = {p: p.read_bytes() for p in output.glob('pairs/*/games/*.json')}
    openings = {p: p.read_bytes() for p in output.glob('pairs/*/openings/*.json')}
    assert len(files) == 12 and len(openings) == 6
    for path, raw in files.items():
        game = json.loads(raw)
        opening = load_json(path.parent.parent/'openings'/f"{game['opening_id']:08d}.json")
        assert game['moves'][:len(opening['moves'])] == opening['moves']
        assert all(v == 4 for v in game['root_visits'])
    events = load_json(next((output/'performance').glob('*.json')))['events']
    stats = next(e for e in events if e['type'] == 'arena_stats')
    assert stats['games'] == 12 and stats['model_windows'] == 3
    assert sum(e['requests'] for e in events if e['type'] == 'model_stats') > 0
    missing = next(iter(files)); missing.unlink()
    before = set((output/'performance').glob('*.json'))
    assert run_arena(output, manifest, binary, 2, 2)['games'] == 12
    for path, raw in {**files, **openings}.items():
        if path != missing:
            assert path.read_bytes() == raw
    events = load_json(next(iter(set((output/'performance').glob('*.json')) - before)))['events']
    assert next(e for e in events if e['type'] == 'arena_stats')['games'] == 1
    assert len((output/'tasks.txt').read_text().splitlines()) == 1
