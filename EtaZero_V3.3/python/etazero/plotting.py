"""Rebuild figures from committed progress and optional observational metrics."""
import json
import os
import warnings
from pathlib import Path
from .storage import atomic_write, load_json

BLUE, RED, ORANGE, GREEN, GREY = '#61afef', '#e06c75', '#e8924b', '#98c379', '#9aa2b1'
THEME = {
    'figure.facecolor': '#1e2127', 'axes.facecolor': '#282c34',
    'savefig.facecolor': '#1e2127', 'text.color': '#abb2bf',
    'axes.labelcolor': '#abb2bf', 'axes.titlecolor': '#c8ccd4',
    'axes.edgecolor': '#454c5a', 'xtick.color': '#8a93a3', 'ytick.color': '#8a93a3',
    'grid.color': '#3b4250', 'legend.labelcolor': '#abb2bf', 'legend.frameon': False,
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'grid.linewidth': .7,
}
METRICS = ('loss', 'policy_loss', 'opponent_policy_loss', 'soft_policy_loss',
           'soft_opponent_policy_loss', 'value_loss', 'td_value_long_loss', 'td_value_mid_loss',
           'td_value_short_loss', 'long_optimistic_policy_loss', 'short_optimistic_policy_loss',
           'shortterm_value_error_loss', 'grad_norm')
POLICY_LOSSES = (
    ('policy_loss', 'Policy', BLUE), ('opponent_policy_loss', 'Opponent policy', RED),
    ('soft_policy_loss', 'Soft policy', ORANGE),
    ('soft_opponent_policy_loss', 'Soft opponent policy', '#c678dd'),
    ('long_optimistic_policy_loss', 'Long optimistic', '#56b6c2'),
    ('short_optimistic_policy_loss', 'Short optimistic', '#e5c07b'))
VALUE_LOSSES = (
    ('value_loss', 'Value', GREEN), ('td_value_long_loss', 'TD long', '#56b6c2'),
    ('td_value_mid_loss', 'TD mid', '#7fbf7f'), ('td_value_short_loss', 'TD short', '#e5c07b'),
    ('shortterm_value_error_loss', 'Value error', GREY))
Q_LOSS = ('q_winloss_loss', 'W-L Q', '#c678dd')


def journal_events(path, offset=0):
    """Only a trailing partial record may be ignored, as in the controller journal."""
    with Path(path).open() as file:
        file.seek(offset)
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if file.read(1):
                    raise ValueError(f'Corrupt journal before its trailing partial record: {path}')
                return


def run_history(run_dir, state=None):
    root = Path(run_dir)
    state = load_json(root/'.internal/state.json') if state is None else state
    history = []
    for iteration in range(state['iteration']):
        commit = load_json(root/'.internal/iterations'/f'{iteration:06d}'/'status.json')['commit']
        try:
            metrics = load_json(root/'logs/iterations'/f'{iteration:06d}.json')
        except (OSError, ValueError) as error:
            warnings.warn(f'Round {iteration} metrics unavailable: {error}', RuntimeWarning)
            metrics = {'metrics_error': str(error)}
        history.append({'phases': {}, **metrics, **commit})
    return history


def round_history(run_dir, iteration, final_step, offset):
    """Read only this round's journal suffix; retries replace the same step."""
    root = Path(run_dir)
    updates, statistics, completed, phases, inference, validation = {}, {}, {}, {}, {}, {}
    for event in journal_events(root/'logs/events.jsonl', offset):
        if event.get('iteration') != iteration: continue
        kind = event['event']
        if kind == 'update' and event['step'] <= final_step:
            updates[event['step']] = event
        elif kind == 'selfplay_statistics':
            statistics[iteration] = event
        elif kind == 'iteration_complete':
            completed[iteration] = event
        elif kind == 'validation':
            validation[iteration] = event
        elif kind == 'phase_end':
            totals = phases.setdefault(iteration, {})
            totals[event['phase']] = totals.get(event['phase'], 0) + event['seconds']
        elif kind == 'inference' and event.get('evaluator', 'random' if iteration < 2 else 'network') == 'network':
            # Recovery can re-emit an attempt's counters; count each worker once.
            inference[(iteration, event['attempt'], event['worker_id'])] = event
    history = {}
    for iteration, event in completed.items():
        history[iteration] = {'iteration': iteration, 'steps': 0, 'total_rows': event['unique_rows'],
                              **statistics.get(iteration, {}), 'total_samples': event['total_samples'],
                              'phases': phases.get(iteration, {})}
    if final_step:
        final = updates.get(final_step)
        if final is None or final['round_metrics']['steps'] != final_step:
            raise ValueError('Round metrics do not cover the final learner cursor')
        row = history.setdefault(iteration, {'iteration': iteration, 'phases': phases.get(iteration, {})})
        # All-batch sums are restored with the checkpoint. The latest final-cursor
        # record replaces abandoned attempts, independent of print cadence.
        row.update(final['round_metrics'])
    for row in history.values():
        # Validation has no update ID. Only completed rounds can expose the
        # latest attempt's result; a new plan clears an abandoned attempt.
        if row['iteration'] in completed and row['iteration'] in validation:
            row['validation'] = validation[row['iteration']]
        counters = [e for (iteration, _, _), e in inference.items() if iteration == row['iteration']]
        if counters:
            for key in ('requests', 'batches', 'submitted', 'cache_hits'):
                row[key] = sum(e[key] for e in counters)
            if all('queue_wait_us' in e for e in counters):
                row['queue_wait_us'] = sum(e['queue_wait_us'] for e in counters)
    return history[iteration]


def _series(axis, series, logarithmic=False, legend_columns=1, show_markers=None, legend=True):
    import numpy as np
    positive = True
    for index, (label, color, x, y) in enumerate(series):
        values = np.asarray(y, dtype=float)
        if not np.isfinite(values).any():
            continue
        positive &= bool((values[np.isfinite(values)] > 0).all())
        mark = (show_markers is None or show_markers[index]) and (
            len(values) == 1 or not np.isfinite(values).all())
        axis.plot(x, values, color=color, linewidth=1.7,
                  marker='o' if mark else None, markersize=3,
                  linestyle='-', label=label)
    if axis.lines:
        if logarithmic and positive:
            axis.set_yscale('log')
        if legend:
            axis.legend(loc='best', fontsize=8 if legend_columns > 1 else 9, ncols=legend_columns,
                        frameon=legend_columns > 1, facecolor=THEME['axes.facecolor'], edgecolor='none', framealpha=.9)
    else:
        axis.text(.5, .5, 'No measurements yet', ha='center', va='center',
                  color='#8a93a3', transform=axis.transAxes)


def _figure(titles, labels, title, columns=2, figsize=(15, 13.5), sharex=False):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import MaxNLocator
    figure = Figure(figsize=figsize, layout='constrained')
    FigureCanvasAgg(figure)
    rows = (len(titles) + columns - 1) // columns
    axes = list(figure.subplots(rows, columns, sharex=sharex, squeeze=False).flat)
    for axis in axes[len(titles):]:
        figure.delaxes(axis)
    axes = axes[:len(titles)]
    figure.suptitle(title, fontsize=17)
    for axis, name, (xlabel, ylabel) in zip(axes, titles, labels):
        axis.set(title=name, xlabel=xlabel, ylabel=ylabel)
        axis.grid(axis='y'); axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    return figure, axes


def _compact_number(value, _):
    label = f'{value:.3g}'
    if 'e' in label:
        mantissa, exponent = label.split('e')
        return f'{mantissa}e{int(exponent)}'
    return label


def _elapsed_axis(axis, history, sample_key):
    """Label measured round endpoints; no constant-throughput conversion.

    At a repeated sample count, use the latest completed round's time. This
    avoids assigning multiple time labels to one position during zero-work rounds.
    """
    import numpy as np
    points = {r[sample_key]: r['elapsed_seconds'] / 3600 for r in history
              if sample_key in r and 'elapsed_seconds' in r and r.get('timing_valid', True)}
    if not points:
        return
    samples = np.asarray(list(points), dtype=float)
    hours = np.asarray(list(points.values()), dtype=float)
    targets = np.linspace(samples[0], samples[-1], 4)
    indices = sorted({int(np.abs(samples - target).argmin()) for target in targets})
    top = axis.secondary_xaxis('top')
    top.set_xticks(samples[indices], [_compact_number(value, None) for value in hours[indices]])
    top.set_xlabel('Elapsed time (h)', fontsize=9)
    top.tick_params(labelsize=9)
    axis.set_title(axis.get_title(), pad=52)


def training_figure(history, algorithm=None):
    import matplotlib as mpl
    import numpy as np
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    if algorithm is None:
        algorithm='muzero' if any('step_losses' in r for r in history) else 'alphazero'
    muzero=algorithm=='muzero'
    titles=['Self-play outcomes', 'Self-play game length', 'Policy losses', 'Value losses',
            'Gradient norm', 'NN cache hit rate']
    labels=[('Cumulative effective self-play rows', 'Share of games'),
            ('Cumulative effective self-play rows', 'Moves per game'),
            ('Cumulative training samples', 'Mean weighted loss'), ('Cumulative training samples', 'Mean weighted loss'),
            ('Cumulative training samples', 'Mean L2 grad norm (before clipping)'),
            ('Cumulative effective self-play rows', 'Share of submitted requests')]
    if muzero:
        titles[5]='Root NN invalid policy mass'
        labels[5]=('Cumulative effective self-play rows', 'Mean probability mass on illegal actions')
        titles+=['Loss by unroll step']
        labels+=[('Unroll step k (0 = representation)', 'Mean weighted loss per step')]
    with mpl.rc_context(THEME):
        figure, axes = _figure(
            titles, labels, f'EtaZero {"MuZero" if muzero else "AlphaZero"} training progress',
            figsize=(15, 16.5) if muzero else (15, 12.5))
        selfplay = [r for r in history if 'total_rows' in r]
        x = [r['total_rows'] for r in selfplay]
        _series(axes[0], [(label, color, x, [r[key]/r['games'] if r.get('games', 0) > 0 else np.nan for r in selfplay])
                         for key, label, color in (('black_wins', 'Black win', BLUE),
                                                   ('white_wins', 'White win', RED), ('draws', 'Draw', GREY))])
        axes[0].set_ylim(0, 1.025); axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        _series(axes[1], [('Mean (including opening)', ORANGE, x, [r.get('avg_game_length', np.nan) for r in selfplay])])
        samples = selfplay[-1]['total_rows'] if selfplay else 0
        for axis in (axes[0], axes[1], axes[5]):
            axis.set_xlim(0, max(1, samples*1.025))
            axis.xaxis.set_major_formatter(FuncFormatter(_compact_number))
            _elapsed_axis(axis, history, 'total_rows')
        trained = [r for r in history if r['steps'] > 0]
        x = [r['total_samples'] for r in trained]
        _series(axes[2], [(label, color, x, [r.get(key, np.nan) for r in trained])
                          for key, label, color in POLICY_LOSSES], logarithmic=True, legend_columns=2)
        _series(axes[3], [(label, color, x, [r.get(key, np.nan) for r in trained])
                          for key, label, color in VALUE_LOSSES]+
                ([('W-L Q', '#c678dd', x, [r.get('q_winloss_loss', np.nan) for r in trained])]
                 if any('q_winloss_loss' in r for r in trained) else []),
                logarithmic=True, legend_columns=2)
        _series(axes[4], [('Network', ORANGE, x, [r.get('grad_norm', np.nan) for r in trained])], True)
        for axis in (axes[2], axes[3], axes[4]):
            axis.set_xlim(0, max(1, max(x, default=0)*1.025))
            axis.xaxis.set_major_formatter(FuncFormatter(_compact_number))
            _elapsed_axis(axis, history, 'total_samples')
        if muzero:
            measured = [r for r in history if 'total_rows' in r]
            _series(axes[5], [('Mean over roots', BLUE, [r['total_rows'] for r in measured],
                              [r.get('root_policy_invalid_mass', np.nan) for r in measured])], show_markers=[False])
        else:
            cached = [r for r in history if 'total_rows' in r]
            _series(axes[5], [('Cache hits', BLUE, [r['total_rows'] for r in cached],
                              [r['cache_hits']/r['submitted'] if r.get('submitted', 0) > 0
                               else np.nan for r in cached])])
        axes[5].set_ylim(0, min(1.025, axes[5].get_ylim()[1]) if muzero else 1.025)
        axes[5].yaxis.set_major_formatter(PercentFormatter(1))
        if muzero:
            measured=[r['step_losses'] for r in trained if 'step_losses' in r]
            series=[]
            if measured:
                matrix=np.asarray(measured,dtype=float);steps=np.arange(matrix.shape[1])
                series=[('Mean over measured iterations', ORANGE, steps, matrix.mean(axis=0))]
                if trained and 'step_losses' in trained[-1]:
                    series.append(('Latest iteration', BLUE, steps, trained[-1]['step_losses']))
                axes[6].set_xticks(steps if len(steps)<=9 else np.arange(0,len(steps),max(1,(len(steps)-1)//8)))
            _series(axes[6],series,logarithmic=True)
            for axis in axes:
                for line in axis.lines:
                    line.set_marker('None')
        games = sum(r.get('games', 0) for r in selfplay)
        figure.supxlabel(f'{len([r for r in trained if "total_rows" in r])} completed training iterations · '
                         f'{games:,} recorded self-play games · {samples:,} effective rows · '
                         f'{max(x, default=0):,} training samples\n'
                         'Loss and gradients: arithmetic means of committed updates, unsmoothed; '
                         'missing metrics are gaps; elapsed time excludes compile, plot and pauses', fontsize=10, color='#8a93a3')
    return figure


def loss_figure(history):
    """One weighted loss per panel, with round-mean train and end-of-round validation."""
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    trained = [r for r in history if r['steps'] > 0]
    metrics = [('loss', 'Total loss', ORANGE), *POLICY_LOSSES, *VALUE_LOSSES]
    if any('q_winloss_loss' in r for r in trained):
        metrics.append(Q_LOSS)
    columns = 4
    rows = (len(metrics) + columns - 1) // columns
    with mpl.rc_context(THEME):
        figure, axes = _figure(
            [label for _, label, _ in metrics],
            [('Cumulative training samples', 'Mean weighted loss') for _ in metrics],
            'EtaZero training and validation losses\n', columns=columns,
            figsize=(18, 3.3 * rows), sharex=True)
        figure.legend(handles=[Line2D([], [], color=BLUE, linewidth=1.7, label='Train'),
                               Line2D([], [], color=RED, linewidth=1.7, label='Validation')],
                      loc='upper center', bbox_to_anchor=(.5, 1 - .36 / figure.get_figheight()), ncols=2)
        x = [r['total_samples'] for r in trained]
        for axis, (key, _, _) in zip(axes, metrics):
            # Keep missing validation as a gap, rather than connecting across a
            # round that had no complete held-out batch or skipped validation.
            train = [r.get(key, float('nan')) for r in trained]
            val = [r.get('validation', {}).get(key, float('nan')) for r in trained]
            _series(axis, [('Train', BLUE, x, train), ('Validation', RED, x, val)],
                    logarithmic=True, show_markers=(True, False), legend=False)
            axis.set_xlim(0, max(1, max(x, default=0)*1.025))
            axis.xaxis.set_major_formatter(FuncFormatter(_compact_number))
        figure.supxlabel(
            'Train: round mean over committed consumed batches; validation: raw model at round end.\n'
            'Weighted sample-mean losses, unsmoothed; missing metrics are gaps, never zero.',
            fontsize=10, color='#8a93a3')
    return figure


def performance_figure(history, batch_size):
    import matplotlib as mpl
    import numpy as np
    with mpl.rc_context(THEME):
        figure, axes = _figure(
            ('Phase wall time', 'Self-play throughput', 'Learner throughput',
             'Inference mean batch', 'Inference queue wait', 'NN cache hit rate'),
            [('Iteration', 'Seconds (sum of completed attempts)'), ('Iteration', 'Effective rows / second'),
             ('Iteration', 'Committed training samples / second'), ('Iteration', 'Rows / batch'),
             ('Iteration', 'Microseconds / NN request'), ('Iteration', 'Share of submitted requests')],
            'EtaZero execution measurements')
        _series(axes[0], [(phase, color, [r['iteration'] for r in history],
                          [r['phases'].get(phase, np.nan) for r in history])
                         for phase, color in zip(('selfplay', 'shuffle', 'train', 'export'), (BLUE, GREEN, ORANGE, RED))])
        for axis, phase, count, label in ((axes[1], 'selfplay', 'rows', 'Self-play'),
                                          (axes[2], 'train', 'steps', 'Learner')):
            multiplier = batch_size if phase == 'train' else 1
            _series(axis, [(label, ORANGE, [r['iteration'] for r in history],
                            [r[count]*multiplier/r['phases'][phase]
                             if count in r and r['phases'].get(phase, 0) > 0 else np.nan for r in history])])
        for axis, numerator, denominator, label in ((axes[3], 'requests', 'batches', 'Mean batch'),
                                                    (axes[4], 'queue_wait_us', 'requests', 'Queue wait'),
                                                    (axes[5], 'cache_hits', 'submitted', 'Cache hits')):
            _series(axis, [(label, BLUE, [r['iteration'] for r in history],
                            [r[numerator]/r[denominator] if numerator in r and r.get(denominator, 0) > 0
                             else np.nan for r in history])])
        figure.supxlabel('Stage wall time includes setup, checkpoint and export costs. '
                         'Incomplete phase attempts have no duration and are excluded.', fontsize=10, color='#8a93a3')
    return figure


def plot_run(run_dir, state=None):
    root = Path(run_dir)
    os.environ.setdefault('MPLCONFIGDIR', str(root/'.internal/matplotlib'))
    history = run_history(root,state)
    config = load_json(root/'.internal/run.json')['config']
    figures = [(root/'training.png', training_figure(history, config['agent']['algorithm'])),
               (root/'loss.png', loss_figure(history)),
               (root/'logs/performance.png', performance_figure(history, config['training']['batch_size']))]
    for destination, figure in figures:
        try:
            atomic_write(destination, lambda path: figure.savefig(path, format='png', dpi=160,
                                                                  facecolor=THEME['figure.facecolor']))
        finally:
            figure.clear()
    return root/'training.png'
