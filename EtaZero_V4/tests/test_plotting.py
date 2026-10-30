from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path

import numpy as np
import pytest

from etazero import observations, plotting
import etazero.run as controller
from katago.train.metrics_logging import log_metrics
from test_controller_recovery import recovery_config


def test_native_learner_logs_are_read_verbatim_without_checkpoints(tmp_path):
    source = StringIO()
    log_metrics({'loss_sum': 42., 'gnorm_batch': 21., 'nsamp': 64},
                {'loss_sum': 12., 'gnorm_batch': 7., 'nsamp': 64}, {}, source)
    path = tmp_path / 'metrics_train.json'; path.write_text(source.getvalue())
    original = path.read_bytes()
    assert observations.learner_log(path) == [{'loss': 3.5, 'gnorm_batch': 3., 'nsamp': 64}]
    assert path.read_bytes() == original
    path.write_text('')
    assert observations.learner_log(path) == []


def test_nonfinite_and_partial_native_records_remain_missing(tmp_path):
    path = tmp_path / 'metrics_train.json'
    path.write_text('{"loss": Infinity, "nsamp": 100}\n{"loss":')
    with pytest.warns(RuntimeWarning):
        assert observations.learner_log(path) == [{'loss': None, 'nsamp': 100}]


def test_logged_cursors_and_retry_gaps_are_not_replaced_by_round_endpoints():
    history = [dict(total_samples=500, train_logs=[{'nsamp': 100, 'loss': 3},
                {'nsamp': 200, 'loss': 2}, {'nsamp': 100, 'loss': 4}, {'nsamp': 400, 'loss': 1}]),
               dict(total_samples=600, train_logs=[])]
    assert plotting.logged_values(history, 'loss') == ([100, 200, 100, 100, 400, 600], [3, 2, None, 4, 1, None])


def native_selfplay_log(*, games=2, moves=12, rows=10, nn_rows=0, batches=0, hits=3, seconds=2):
    values = {'Final games finished': games, 'Final moves played': moves, 'Final data rows': rows,
              'Final NN rows': nn_rows, 'Final NN batches': batches, 'Final NN cache hits': hits,
              'Total selfplay runtime (seconds)': seconds}
    return ''.join(f'2026-10-29 12:34:56+0800: {key}: {value}\n' for key, value in values.items())


def test_native_observations_combine_drained_attempts_without_random_inference(tmp_path):
    (tmp_path / 'log1.log').write_text(native_selfplay_log())
    assert 'nn_rows' not in observations.selfplay_metrics(tmp_path, 1)
    (tmp_path / 'log1.log').write_text(native_selfplay_log(nn_rows=20, batches=5))
    (tmp_path / 'log2.log').write_text(native_selfplay_log(games=1, nn_rows=30, batches=10, hits=6))
    total = observations.selfplay_metrics(tmp_path, 2)
    assert total['games'] == 3 and total['moves_played'] == 24
    assert total['nn_rows'] == 50 and total['nn_batches'] == 15 and total['cache_hits'] == 9
    assert total['cache_hit_rate'] == pytest.approx(9 / 59)
    assert 'cache_hit_rate' not in observations.selfplay_metrics(tmp_path, 2, muzero=True)
    assert not {'black_wins', 'draws', 'no_results', 'truncated', 'total_moves', 'nn_initial_rows'} & total.keys()
    with pytest.raises(ValueError, match='drained attempts'):
        observations.selfplay_metrics(tmp_path, 3)


def test_missing_native_fields_stay_missing_without_sgf_or_added_json(tmp_path):
    log = native_selfplay_log(nn_rows=30, batches=10).replace('Final data rows: 10', 'unrelated: 10')
    (tmp_path / 'log1.log').write_text(log + 'Selfplay metrics: {"games":999,"black_wins":999,"rows":999}\n')
    (tmp_path / 'games.sgfs').write_text('(;RE[B+R])\n')
    with pytest.warns(RuntimeWarning, match='leaving gaps'):
        result = observations.selfplay_metrics(tmp_path, 1)
    assert 'rows' not in result and 'black_wins' not in result
    assert result['games'] == 2 and result['nn_rows'] == 30


def test_gatekeeper_measurement_uses_final_native_log_only(tmp_path):
    (tmp_path / 'result.json').write_text('{"games":999,"candidate_points":999}')
    with pytest.warns(RuntimeWarning, match='leaving a gap'):
        assert observations.gatekeeper_metrics(tmp_path) == {}
    directory = tmp_path / 'matches/1'; directory.mkdir(parents=True)
    log = directory / 'log1.log'
    log.write_text('prefix Candidate won match, score 5.500 to 2.500 in 8 games, accepting candidate new\n')
    assert observations.gatekeeper_metrics(tmp_path) == dict(games=8, candidate_points=5.5, baseline_points=2.5)
    log.write_text(log.read_text() * 2)
    with pytest.warns(RuntimeWarning, match='leaving a gap'):
        assert observations.gatekeeper_metrics(tmp_path) == {}


def write_history(root, *, missing=False):
    root.mkdir(exist_ok=True)
    (root / 'state.json').write_text(json.dumps({'completed_rounds': 3}))
    (root / 'config.json').write_text(json.dumps({'algorithm': 'gumbel_muzero'}))
    train, validation = [], []
    for number in range(1, 5):
        directory = root / 'rounds' / f'{number:06d}'
        directory.mkdir(parents=True)
        valid = number < 3
        commit = {'state': {'completed_rounds': number, 'total_samples': number * 20,
                            'elapsed_seconds': number * 10, 'timing_valid': valid},
                  'round_timing': {'wall_seconds': 12, 'compile_seconds': 2, 'seconds': 10 if valid else None,
                                   'timing_valid': valid}}
        (directory / 'commit.json').write_text(json.dumps(commit))
        logs = directory / 'logs'; logs.mkdir()
        (logs / 'selfplay.log').write_text('native stdout capture')
        if missing and number == 2:
            continue
        data = {'rows': 10, 'train': {'loss': 3. / number, 'p0loss': 1., 'vloss': .5, 'gnorm_batch': 6.,
                                     'mu_step_0_loss': 1., 'mu_step_1_loss': 2., 'mu_recurrent_p0loss': 2.,
                                     'Ip0loss': 1.5, 'mu_root_loss': 1., 'mu_recurrent_loss': 2.},
                'validation': {'loss': 2. / number} if number != 2 else {},
                'stage_seconds': {'selfplay': 4, 'train': 6}, 'stage_compile_seconds': {'train': 2},
                'selfplay': {'games': 4, 'moves_played': 24, 'nn_rows': 30, 'nn_batches': 10, 'cache_hits': 6}}
        train.append(dict(data['train'], nsamp=number * 20, time_since_last_print=2.))
        if data['validation']:
            validation.append(dict(data['validation'], nsamp_train=number * 20))
        native = root / 'selfplay' / f'{number:06d}'; native.mkdir(parents=True)
        (native / 'log1.log').write_text(native_selfplay_log(games=4, moves=24, nn_rows=30, batches=10, hits=6))
        # An obsolete outer cache must not override the producer's logs.
        (directory / 'metrics.json').write_text(json.dumps({'train': {'loss': 999999}, 'rows': 999999}))
    native = root / 'train'; native.mkdir()
    for name, records in [('train', train), ('val', validation)]:
        (native / f'metrics_{name}.json').write_text(''.join(json.dumps(row) + '\n' for row in records))


def test_history_uses_only_committed_progress_and_does_not_fill_missing_metrics(tmp_path):
    write_history(tmp_path, missing=True)
    with pytest.warns(RuntimeWarning, match='Round 2'):
        history = plotting.run_history(tmp_path)
    assert len(history) == 3
    assert history[1]['total_samples'] == 40 and history[1]['new_samples'] == 20
    assert history[1]['train'] == {}
    assert [row['total_rows'] for row in history] == [10, None, None]
    assert not history[-1]['timing_valid']


def test_figures_preserve_missing_outcomes_and_muzero_cache_denominator(tmp_path):
    write_history(tmp_path)
    history = plotting.run_history(tmp_path)
    figure = plotting.training_figure(history, 'gumbel_muzero')
    assert len(figure.axes) == 7
    assert not figure.axes[0].lines and not figure.axes[5].lines
    assert 'Not recorded in native self-play logs' in [text.get_text() for text in figure.axes[0].texts]
    np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [6., 6., 6.])
    np.testing.assert_allclose(figure.axes[6].lines[0].get_ydata(), [1., 2.])
    np.testing.assert_allclose(figure.axes[0].child_axes[0].get_xticks(), [10, 20])
    figure.clear()
    figure = plotting.loss_figure(history)
    total = next(axis for axis in figure.axes if axis.get_title() == 'Total loss')
    assert np.isnan(total.lines[1].get_ydata()[1])
    assert total.get_yscale() == 'log'
    assert 'Intermediate Policy' in {axis.get_title() for axis in figure.axes}
    figure.clear()
    figure = plotting.performance_figure(history)
    np.testing.assert_allclose(figure.axes[2].lines[0].get_ydata(), [3., 3., 3.], equal_nan=True)
    assert not figure.axes[3].lines
    figure.clear()


def test_atomic_plot_export_is_read_only_for_run_evidence(tmp_path):
    root, output = tmp_path / 'run', tmp_path / 'pictures'
    write_history(root)
    originals = {path: path.read_bytes() for path in root.rglob('*.json')}
    paths = plotting.plot_run(root, output)
    assert {p.relative_to(output).as_posix() for p in paths} == {'training.png', 'loss.png', 'logs/performance.png'}
    assert all(p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for p in paths)
    assert not list(output.rglob('*.tmp.*'))
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_auto_plot_failure_is_visible_without_reversing_training(tmp_path, monkeypatch, capsys):
    config = recovery_config(tmp_path, 'alphazero')
    def fail(*args):
        raise OSError('injected plot output failure')
    monkeypatch.setattr(plotting, 'plot_run', fail)
    controller.refresh_plots(tmp_path, config)
    assert 'committed training remains valid' in capsys.readouterr().err
    controller.refresh_plots(tmp_path, replace(config, auto_plot=False))
    assert not capsys.readouterr().err


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA')
def test_gatekeeper_plot_reads_real_native_log_without_stdout(tmp_path):
    import subprocess
    from test_gatekeeper import native_command
    command, _ = native_command(tmp_path, 'renju', False, .5)
    (tmp_path / 'matches').mkdir()
    command[command.index('-sgf-output-dir') + 1] = str(tmp_path / 'matches/1')
    with (tmp_path / 'gate.cfg').open('a') as stream:
        stream.write('logToStdout = false\n')
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    measured = observations.gatekeeper_metrics(tmp_path)
    assert measured['games'] > 0
    # This fixture exhausts its small Renju board without a five-in-a-row.
    assert measured['candidate_points'] == measured['baseline_points'] == measured['games'] / 2


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and learner')
@pytest.mark.parametrize('algorithm', ['alphazero', 'gumbel_muzero'])
def test_real_observations_and_two_round_plots(tmp_path, monkeypatch, capsys, algorithm):
    config = replace(recovery_config(tmp_path, algorithm), epochs_per_round=1, games_per_round=80,
                     samples_per_epoch=800, shuffle_keep_rows=2048)
    with Path(config.selfplay_config).open('a') as stream:
        stream.write('logToStdout = false\n')
    root = tmp_path / 'run'
    first = controller.run(root, config, 1)
    assert (root / 'training.png').is_file()
    history = plotting.run_history(root)
    assert history[0]['selfplay']['games'] == config.games_per_round
    assert ('nn_rows' in history[0]['selfplay']) == config.is_muzero
    assert history[0]['train']['loss'] > 0
    if config.is_muzero:
        assert history[0]['train']['mu_step_0_loss'] > 0
    # Rendering failures occur after commit and cannot prevent the next model.
    actual = plotting.plot_run
    def fail(*args):
        raise OSError('injected plot failure')
    monkeypatch.setattr(plotting, 'plot_run', fail)
    second = controller.run(root, config, 2)
    assert second['total_samples'] > first['total_samples']
    assert 'plotting failed' in capsys.readouterr().err
    monkeypatch.setattr(plotting, 'plot_run', actual)
    before = (root / 'state.json').read_bytes()
    actual(root)
    assert (root / 'state.json').read_bytes() == before
    history = plotting.run_history(root)
    latest = history[-1]
    assert latest['selfplay']['nn_rows'] > 0
    assert latest['rows'] > 0 and latest['total_rows'] == sum(row['rows'] for row in history)
    assert latest['train']['nsamp'] <= second['total_samples']
    assert not (root / 'rounds/000002/metrics.json').exists()
    assert not {'black_wins', 'white_wins', 'draws', 'no_results', 'truncated'} & latest['selfplay'].keys()
    assert ('cache_hit_rate' in latest['selfplay']) == (not config.is_muzero)
    for path in (root / 'selfplay/000002').glob('log*.log'):
        text = path.read_text()
        assert 'Selfplay metrics:' not in text and 'Final moves played: ' in text
    # Independent evidence: the source SGF writer emits one line per real game.
    sgfs = [line for path in (root / 'selfplay/000002').glob('*/sgfs/*.sgfs') for line in path.read_text().splitlines()]
    assert len(sgfs) == latest['selfplay']['games']
    native = observations.learner_log(root / 'train/metrics_train.json')
    assert latest['train'] == native[-1]
    assert plotting.logged_values(history, 'loss')[1][-1] == native[-1]['loss']
