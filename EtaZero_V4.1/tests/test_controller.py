import fcntl
import json
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest

from etazero import ROOT
from etazero.config import RunConfig
from etazero.run import run


@pytest.fixture
def config():
    return RunConfig(selfplay_config=str(ROOT / 'configs/checks/source_go.cfg'),
                     model_kind='b2c16-fson-mish', canvas=7, batch_size=8,
                     bootstrap_games=8, train_steps=4,
                     shuffle_rows_per_file=8, shuffle_keep_rows=1024,
                     min_rows=1,
                     train_args=('-no-compile', '-lookahead-k', '6', '-no-longterm-checkpoints')).validate()


@pytest.mark.parametrize('algorithm', ['gumbel', 'gumbel-muzero'])
def test_unconnected_algorithms_are_rejected(config, algorithm):
    with pytest.raises(ValueError, match='not implemented'):
        replace(config, algorithm=algorithm).validate()


@pytest.mark.parametrize('args', [('-multi-gpus', '0,1'), ('-latestdatadir=/tmp/data',),
                                 ('-max-training-samples', '1'), ('-traind=/tmp/data',),
                                 ('-initialize-only',), ('-muzero-conf=/tmp/network.json',),
                                 ('-sub-epochs', '2'), ('-epoch-files', '/tmp/plan.json')])
def test_source_arguments_cannot_override_stage_ownership(config, args):
    with pytest.raises(ValueError, match='outer controller'):
        replace(config, train_args=args).validate()


def test_muzero_settings_are_owned_by_the_selected_algorithm(config):
    from dataclasses import asdict
    from test_muzero_model import make_model
    network = asdict(make_model('go').network_config)
    mu = replace(config, algorithm='muzero', model_kind=None, muzero_network=network, muzero_unroll_steps=3)
    assert mu.validate().is_muzero
    assert replace(mu, algorithm='gumbel_muzero').validate().is_gumbel
    for changed, message in [
        (dict(muzero_network=None), 'requires muzero_network'),
        (dict(muzero_unroll_steps=0), 'positive integer'),
        (dict(muzero_unroll_steps=True), 'positive integer'),
        (dict(game='hex'), 'must match'),
        (dict(model_kind=config.model_kind), 'uses muzero_network'),
        (dict(algorithm='alphazero'), 'require a MuZero algorithm'),
        (dict(publish_weights='latest'), 'ordinary or swa'),
        (dict(shuffle_args=('-muzero-unroll', '1')), 'outer controller'),
        (dict(shuffle_args=('-input-files-stable',)), 'outer controller'),
    ]:
        with pytest.raises(ValueError, match=message):
            replace(mu, **changed).validate()


def test_shuffle_window_can_exhaust_before_the_epoch_target(config):
    assert replace(config, shuffle_keep_rows=config.batch_size,
                   train_steps=10).validate()
    with pytest.raises(ValueError, match='at least one batch'):
        replace(config, shuffle_keep_rows=config.batch_size - 1).validate()


def test_epoch_target_is_a_positive_sample_count(config):
    assert replace(config, train_steps=3).validate().samples_per_epoch == config.batch_size * 3
    with pytest.raises(ValueError, match='train_steps'):
        replace(config, train_steps=0).validate()


def test_failed_stage_preserves_artifacts_and_retries_same_round(tmp_path, config):
    root = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='see .*selfplay.log'):
        run(root, config, 1, binary='/bin/false')
    marker = root / 'selfplay/000000/preserved.txt'
    marker.write_text('partial data')
    state = json.loads((root / 'state.json').read_text())
    assert state['completed_rounds'] == -1 and state['accepted_model'] is None
    assert state['active_round'] == 0 and state['phase'] == 'selfplay'
    with pytest.raises(RuntimeError, match=r'see .*selfplay\.2\.log'):
        run(root, config, 1, binary='/bin/false')
    assert marker.read_text() == 'partial data'
    assert len(list((root / 'rounds').iterdir())) == 1
    state = json.loads((root / 'state.json').read_text())
    assert state['round_progress']['attempts'] == {'selfplay': 2}
    assert (root / 'rounds/000000/logs/selfplay.log').is_file()


def test_concurrent_controller_cannot_write_the_same_run(tmp_path, config):
    with (tmp_path / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another controller'):
            run(tmp_path, config, 1, binary='/bin/false')
    assert not (tmp_path / 'state.json').exists()


@pytest.mark.parametrize('running', ['selfplay', 'train', 'initialize', None])
def test_unobserved_stage_invalidates_timing_on_recovery(tmp_path, config, capsys, running):
    with pytest.raises(RuntimeError, match='Source stage exited'):
        run(tmp_path, config, 1, binary='/bin/false')
    path = tmp_path / 'state.json'
    state = json.loads(path.read_text())
    # This is the persisted state left by a controller killed during a child.
    state['round_progress']['running'] = running
    state['round_progress']['clock_running'] = True
    path.write_text(json.dumps(state))
    observed_seconds = state['round_progress']['stage_seconds']['selfplay']
    with pytest.raises(RuntimeError, match='Source stage exited'):
        run(tmp_path, config, 1, binary='/bin/false')
    progress = json.loads(path.read_text())['round_progress']
    assert progress['timing_valid'] is False
    assert progress['running'] is None
    if running in ('train', 'initialize'):
        assert progress['compile_seconds'] is None
        assert progress['stage_compile_seconds'][running] is None
    assert progress['stage_seconds']['selfplay'] >= observed_seconds
    assert 'elapsed stage time is incomplete' in capsys.readouterr().err


def test_unrelated_output_directory_is_not_adopted(tmp_path, config):
    (tmp_path / 'existing.txt').write_text('keep')
    with pytest.raises(ValueError, match='empty output directory'):
        run(tmp_path, config, 1, binary='/bin/false')
    assert (tmp_path / 'existing.txt').read_text() == 'keep'


def test_reduced_round_budget_does_not_abandon_an_active_round(tmp_path, config):
    with pytest.raises(RuntimeError, match='Source stage exited'):
        run(tmp_path, config, 2, binary='/bin/false')
    path = tmp_path / 'state.json'
    state = json.loads(path.read_text())
    # Simulate interruption in round 2, then lower the cumulative target to 1.
    state.update(completed_rounds=1, active_round=2)
    path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match='Source stage exited'):
        run(tmp_path, config, 1, binary='/bin/false')
    resumed = json.loads(path.read_text())
    assert resumed['active_round'] == 2
    assert resumed['round_progress']['attempts']['selfplay'] == 2
    assert (tmp_path / 'rounds/000002/logs/selfplay.2.log').is_file()


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA build and GPU')
@pytest.mark.parametrize('weights', ['ordinary', 'swa'])
def test_source_model_is_consumed_after_completed_round_restart(tmp_path, config, weights):
    selfplay_config = tmp_path / 'selfplay.cfg'
    selfplay_config.write_text(
        f'@include {os.path.relpath(config.selfplay_config, selfplay_config.parent)}\n'
        'komiMean = -2.5\ninitGamesWithPolicy = true\npolicyInitAreaProp = 0.1\n'
        'handicapProb = 0.5\nnumExtraBlackFixed = 2\nearlyForkGameProb = 0.5\n'
        'normalAsymmetricPlayoutProb = 0.25\nmaxAsymmetricRatio = 4\nmaxVisits = 24\n'
    )
    # Keep enough rows for the first epoch and a Lookahead/SWA update. With a
    # one-row window, the random final raw shard can have fewer than one batch.
    config = replace(config, selfplay_config=str(selfplay_config), publish_weights=weights, train_steps=8,
                     min_rows=2 * 8 * config.batch_size)
    run_dir = tmp_path / 'run'
    first = run(run_dir, config, 1)
    assert first['total_samples'] > 0
    second = run(run_dir, config, 2)
    assert second['completed_rounds'] == 2 and second['total_samples'] > first['total_samples']
    result = json.loads((run_dir / 'rounds/000002/result.json').read_text())
    assert result['input_model'].startswith('round-000001-s')
    assert result['samples_before'] == first['total_samples']
    assert result['training_epochs'] == 1
    assert all(Path(p).parent.parent.name == '000002' for p in result['training_plan']['files'])
    assert result['published_weights'] == weights
    assert list((run_dir / 'selfplay/000002' / result['input_model'] / 'tdata').glob('*.npz'))
    assert not (run_dir / 'selfplay/000002/random').exists()
    for round_dir in (run_dir / 'selfplay').iterdir():
        komis = [float(k) for sgf in round_dir.glob('*/sgfs/*.sgfs')
                 for k in re.findall(r'KM\[([^]]+)\]', sgf.read_text())]
        if not komis:
            commit = json.loads((run_dir / 'rounds' / round_dir.name / 'commit.json').read_text())
            assert commit['sampling']['games'] == 0
        else:
            assert set(komis) == {-2.5}
    checkpoint = run_dir / 'train/checkpoint.ckpt'
    modified = checkpoint.stat().st_mtime_ns
    assert run(run_dir, config, 2) == second
    assert checkpoint.stat().st_mtime_ns == modified
