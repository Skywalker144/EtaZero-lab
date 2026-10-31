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


def learner_header(*, batch=8, version=18, muzero=False, main=None, intermediate=None):
    config = {'version': version}
    if muzero:
        config = {'algorithm': 'muzero', 'network': {'prediction': config}}
    fields = dict(soft_policy_weight_scale=3., value_loss_scale=.4,
                  td_value_loss_scales=[.2, .3, .5], main_loss_scale=main,
                  intermediate_loss_scale=intermediate)
    return (repr(['train.py', '-batch-size', str(batch)]) + '\n' + repr(config) + '\n' +
            ''.join(f'{key} {value!r}\n' for key, value in fields.items()) + 'Parameters in model:\n')


@pytest.mark.parametrize('muzero', [False, True])
def test_overview_weights_heads_and_normalizes_gradient_without_changing_detail(tmp_path, muzero):
    (tmp_path / 'train.log').write_text(learner_header(muzero=muzero, main=.25, intermediate=.75))
    scales = observations.learner_display_scales(tmp_path)
    # Independently chosen head losses: combined soft=3, policy=5, value=7.
    record = dict(nsamp=100, p0softloss=6., Ip0softloss=2., p0loss=8., Ip0loss=4.,
                  vloss=10., Ivloss=6., tdvloss1=4., Itdvloss1=4.,
                  tdvloss2=4., Itdvloss2=4., tdvloss3=4., Itdvloss3=4.,
                  p0lopt=5., Ip0lopt=5., p0sopt=5., Ip0sopt=5., gnorm_batch=24.)
    original = dict(record)
    row = dict(total_samples=100, total_rows=0, round=1, train_logs=[record], display_scales=scales)
    figure = plotting.training_figure([row], 'muzero' if muzero else 'alphazero')
    policy = {line.get_label(): line.get_ydata()[0] for line in figure.axes[2].lines}
    value = {line.get_label(): line.get_ydata()[0] for line in figure.axes[3].lines}
    assert policy == pytest.approx({'Policy': 4.65, 'Soft policy': 9., 'Long optimistic': .5, 'Short optimistic': 1.})
    assert value == pytest.approx({'Value': 2.8, 'TD long': .8, 'TD mid': 1.2, 'TD short': 2.})
    assert figure.axes[4].lines[0].get_ydata()[0] == 3.
    figure.clear()
    detail = plotting.loss_figure([row])
    soft = next(axis for axis in detail.axes if axis.get_title() == 'Soft policy')
    assert soft.lines[0].get_ydata()[0] == 6.
    assert record == original
    detail.clear()


def test_overview_scales_preserve_retry_and_missing_data_gaps(tmp_path):
    (tmp_path / 'train.log').write_text(learner_header(version=11, batch=4))
    scales = observations.learner_display_scales(tmp_path)
    assert scales['loss']['p0loss'] == 1.
    assert scales['loss']['p0lopt'] == scales['loss']['p0sopt'] == 0.
    history = [dict(total_samples=20, display_scales=scales, train_logs=[
        dict(nsamp=10, gnorm_batch=12.), dict(nsamp=10, gnorm_batch=8.), dict(nsamp=20, gnorm_batch=None)]),
        dict(total_samples=30, train_logs=[dict(nsamp=30, gnorm_batch=16.)])]
    assert plotting.logged_values(history, 'gnorm_batch', transform=plotting.overview_value) == (
        [10, 10, 10, 20, 30], [3., None, 2., None, None])
    (tmp_path / 'train.2.log').write_text(learner_header(batch=16))
    with pytest.warns(RuntimeWarning, match='conflicting'):
        assert observations.learner_display_scales(tmp_path) == {}


def test_missing_learner_settings_do_not_use_current_config(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'batch_size': 128}))
    with pytest.warns(RuntimeWarning, match='Missing'):
        assert observations.learner_display_scales(tmp_path) == {}


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


def outcome_log(black, white, draws, no_results, truncated):
    return ''.join(marker + str(value) + '\n' for marker, value in
                   zip(observations.OUTCOMES.values(), (black, white, draws, no_results, truncated)))


def test_outcomes_count_games_across_attempts_and_keep_gaps(tmp_path):
    first = native_selfplay_log(games=5, rows=0) + outcome_log(1, 1, 1, 1, 1)
    (tmp_path / 'log1.log').write_text(first)
    # Zero-row games participate; all five classes have independent meanings.
    assert all(observations.selfplay_metrics(tmp_path, 1)[key] == 1 for key in observations.OUTCOMES)
    second = native_selfplay_log(games=3) + outcome_log(0, 2, 0, 0, 1)
    (tmp_path / 'log2.log').write_text(second)
    total = observations.selfplay_metrics(tmp_path, 2)
    assert [total[key] for key in observations.OUTCOMES] == [1, 3, 1, 1, 2]
    assert sum(total[key] for key in observations.OUTCOMES) == total['games'] == 8
    (tmp_path / 'log2.log').write_text(native_selfplay_log(games=3))
    with pytest.warns(RuntimeWarning, match='Incomplete native selfplay outcomes'):
        total = observations.selfplay_metrics(tmp_path, 2)
    assert not observations.OUTCOMES.keys() & total.keys()
    (tmp_path / 'log2.log').write_text(second.replace('Final white wins: 2', 'Final white wins: 3'))
    with pytest.warns(RuntimeWarning, match='do not sum'):
        total = observations.selfplay_metrics(tmp_path, 2)
    assert not observations.OUTCOMES.keys() & total.keys()


def test_outcome_plot_uses_completed_games_not_rows(tmp_path):
    write_history(tmp_path)
    for number, counts in [(1, (1, 1, 1, 0, 1)), (3, (0, 2, 0, 1, 1))]:
        with (tmp_path / 'selfplay' / f'{number:06d}' / 'log1.log').open('a') as stream:
            stream.write(outcome_log(*counts))
    figure = plotting.training_figure(plotting.run_history(tmp_path), 'alphazero')
    assert len(figure.axes[0].lines) == 5
    np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [.25, np.nan, 0], equal_nan=True)
    np.testing.assert_allclose(figure.axes[0].lines[1].get_ydata(), [.25, np.nan, .5], equal_nan=True)
    figure.clear()


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
        (logs / 'train.log').write_text(learner_header(muzero=True))
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


def test_bootstrap_rows_and_zero_game_first_training_remain_on_history_axis(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(dict(algorithm='alphazero')))
    for number, samples in [(0, 0), (1, 8)]:
        directory = tmp_path / 'rounds' / f'{number:06d}'
        (directory / 'logs').mkdir(parents=True)
        state = dict(completed_rounds=number, total_samples=samples, elapsed_seconds=number + 1,
                     timing_valid=True)
        sampling = dict(games=2, rows=10) if number == 0 else dict(games=0, rows=0)
        (directory / 'commit.json').write_text(json.dumps(dict(state=state, sampling=sampling, round_timing={})))
    (tmp_path / 'state.json').write_text(json.dumps(state))
    (tmp_path / 'rounds/000000/logs/selfplay.log').touch()
    (tmp_path / 'rounds/000001/logs/train.log').write_text(learner_header())
    output = tmp_path / 'selfplay/000000'
    output.mkdir(parents=True)
    (output / 'log1.log').write_text(native_selfplay_log())
    (tmp_path / 'train').mkdir()
    (tmp_path / 'train/metrics_train.json').write_text('{"nsamp":8,"loss":1}\n')
    (tmp_path / 'train/metrics_val.json').touch()
    records = plotting.run_history(tmp_path)
    assert [r['round'] for r in records] == [0, 1]
    assert [r['total_rows'] for r in records] == [10, 10]
    assert records[0]['train'] == {} and records[1]['train']['loss'] == 1
    assert records[1]['selfplay'] == dict(rows=0, games=0)


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
    config = replace(recovery_config(tmp_path, algorithm), bootstrap_games=80,
                     train_steps=200, shuffle_keep_rows=2048, min_rows=512)
    # The source caps random-evaluator data at min-rows. Retain enough batches
    # to reach its native metrics cadence without changing the logger itself.
    with Path(config.selfplay_config).open('a') as stream:
        stream.write('logToStdout = false\n')
    root = tmp_path / 'run'
    first = controller.run(root, config, 1)
    assert (root / 'training.png').is_file()
    history = plotting.run_history(root)
    assert history[0]['selfplay']['games'] == config.bootstrap_games
    assert ('nn_rows' in history[0]['selfplay']) == config.is_muzero
    assert history[1]['train']['loss'] > 0
    if config.is_muzero:
        assert history[1]['train']['mu_step_0_loss'] > 0
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
    assert sum(latest['selfplay'][key] for key in observations.OUTCOMES) == latest['selfplay']['games']
    assert latest['selfplay']['draws'] == latest['selfplay']['games']
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


def test_history_reads_last_epoch_candidate_gate_logs(tmp_path):
    write_history(tmp_path)
    directory = tmp_path / 'rounds/000001'
    path = directory / 'commit.json'
    commit = json.loads(path.read_text())
    commit['candidate_model'] = 'models/round-000001-s20-d12'
    path.write_text(json.dumps(commit))
    for samples, points in ((10, 2.0), (20, 3.0)):
        logs = directory / 'gatekeeper' / f'round-000001-s{samples}-d12' / 'matches/1'
        logs.mkdir(parents=True)
        (logs / 'log1.log').write_text(f'Candidate won match, score {points} to {4-points} in 4 games, accepting candidate\n')
    history = plotting.run_history(tmp_path)
    assert history[0]['gatekeeper'] == dict(candidate_points=3., baseline_points=1., games=4)
