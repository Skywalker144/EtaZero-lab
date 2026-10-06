"""Rebuild figures from durable journal values and checkpoint commit lineage."""
import json
import os
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
MODULE_GRADIENTS = (('representation', 'Representation h', BLUE),
                    ('dynamics', 'Dynamics g', GREEN), ('prediction', 'Prediction f', ORANGE))


def journal_events(path):
    """Only a trailing partial record may be ignored, as in the controller journal."""
    with Path(path).open() as file:
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
    reference = state['checkpoint']
    committed = set()
    seen = set()
    while reference:
        if reference['id'] in seen:
            raise ValueError('Checkpoint lineage contains a cycle')
        seen.add(reference['id'])
        sidecar = load_json((root/reference['path']).with_suffix('.json'))
        committed.update(sidecar['committed_updates'])
        reference = sidecar['parent']
    updates, statistics, completed, phases, inference, validation = {}, {}, {}, {}, {}, {}
    for event in journal_events(root/'logs/events.jsonl'):
        kind = event['event']; iteration = event.get('iteration')
        if kind == 'plan':
            # Only the latest attempt of a committed round contributes timings.
            phases.pop(iteration, None)
            validation.pop(iteration, None)
            for key in [k for k in inference if k[0] == iteration]:
                del inference[key]
        if kind == 'update' and event['update_id'] in committed:
            updates[event['update_id']] = event
        elif kind == 'selfplay_statistics' and iteration < state['iteration']:
            statistics[iteration] = event
        elif kind == 'iteration_complete' and iteration < state['iteration']:
            completed[iteration] = event
        elif kind == 'validation' and iteration < state['iteration']:
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
                              **statistics.get(iteration, {}), 'phases': phases.get(iteration, {})}
    for event in updates.values():
        row = history.setdefault(event['iteration'], {'iteration': event['iteration'], 'steps': 0,
                                                      'phases': phases.get(event['iteration'], {})})
        row['steps'] += 1
        skipped = event.get('amp_skipped', False)
        row['amp_skipped_steps'] = row.get('amp_skipped_steps', 0) + int(skipped)
        row['gradient_steps'] = row.get('gradient_steps', 0) + int(not skipped)
        for metric in METRICS:
            # Preserve overflow norms in the raw journal; they are not finite
            # gradient measurements for the round's plotted mean.
            if metric == 'grad_norm' and skipped:
                continue
            row[metric] = row.get(metric, 0) + event[metric]
        if 'q_winloss_loss' in event:
            row['q_winloss_loss']=row.get('q_winloss_loss',0)+event['q_winloss_loss']
        if 'step_losses' in event:
            measured=event['step_losses']
            total=row.setdefault('step_losses',[0.0]*len(measured))
            if len(total)!=len(measured):raise ValueError('Unroll length changed within a training round')
            row['step_losses']=[a+b for a,b in zip(total,measured)]
            row['step_loss_updates']=row.get('step_loss_updates',0)+1
        if 'grad_norms' in event and not skipped:
            totals=row.setdefault('grad_norms',dict.fromkeys((k for k,_,_ in MODULE_GRADIENTS),0.0))
            for key in totals:totals[key]+=event['grad_norms'][key]
            row['module_gradient_steps']=row.get('module_gradient_steps',0)+1
    for row in history.values():
        if row['steps']:
            for metric in METRICS:
                if metric == 'grad_norm':
                    row[metric] = row.get(metric, 0) / row['gradient_steps'] if row['gradient_steps'] else float('nan')
                else:
                    row[metric] /= row['steps']
            if 'q_winloss_loss' in row:row['q_winloss_loss']/=row['steps']
            if 'step_losses' in row:
                row['step_losses']=[v/row['step_loss_updates'] for v in row['step_losses']]
            if 'grad_norms' in row:
                row['grad_norms']={k:v/row['module_gradient_steps'] for k,v in row['grad_norms'].items()}
        # Validation has no update ID. Only completed rounds can expose the
        # latest attempt's result; a new plan clears an abandoned attempt.
        if row['iteration'] in completed and row['iteration'] in validation:
            row['validation'] = validation[row['iteration']]
        counters = [e for (iteration, _, _), e in inference.items() if iteration == row['iteration']]
        row['requests'] = sum(e['requests'] for e in counters)
        row['batches'] = sum(e['batches'] for e in counters)
        row['queue_wait_us'] = sum(e['queue_wait_us'] for e in counters)
        row['submitted'] = sum(e['submitted'] for e in counters)
        row['cache_hits'] = sum(e['cache_hits'] for e in counters)
    return [history[k] for k in sorted(history)]


def _series(axis, series, logarithmic=False, legend_columns=1, show_markers=None):
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


def training_figure(history, algorithm=None):
    import matplotlib as mpl
    import numpy as np
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    if algorithm is None:
        algorithm='muzero' if any('step_losses' in r or 'grad_norms' in r for r in history) else 'alphazero'
    muzero=algorithm=='muzero'
    titles=['Self-play outcomes', 'Self-play game length', 'Policy losses', 'Value losses',
            'Gradient norm', 'NN cache hit rate']
    labels=[('Cumulative effective self-play rows', 'Share of games'),
            ('Cumulative effective self-play rows', 'Moves per game'),
            ('Iteration (1-based)', 'Mean weighted loss'), ('Iteration (1-based)', 'Mean weighted loss'),
            ('Iteration (1-based)', 'Mean L2 grad norm (before clipping)'),
            ('Cumulative effective self-play rows', 'Share of submitted requests')]
    if muzero:
        titles+=['Loss by unroll step', 'Gradient norms by module']
        labels+=[('Unroll step k (0 = representation)', 'Mean weighted loss per step'),
                 ('Iteration (1-based)', 'Mean L2 grad norm (before clipping)')]
    with mpl.rc_context(THEME):
        figure, axes = _figure(
            titles, labels, f'EtaZero {"MuZero" if muzero else "AlphaZero"} training progress',
            figsize=(15, 14.5) if muzero else (15, 11))
        selfplay = [r for r in history if r.get('games', 0) > 0]
        x = [r['total_rows'] for r in selfplay]
        _series(axes[0], [(label, color, x, [r[key]/r['games'] for r in selfplay])
                         for key, label, color in (('black_wins', 'Black win', BLUE),
                                                   ('white_wins', 'White win', RED), ('draws', 'Draw', GREY))])
        axes[0].set_ylim(0, 1.025); axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        _series(axes[1], [('Mean (including opening)', ORANGE, x, [r['avg_game_length'] for r in selfplay])])
        samples = selfplay[-1]['total_rows'] if selfplay else 0
        for axis in (axes[0], axes[1], axes[5]):
            axis.set_xlim(0, max(1, samples*1.025))
            axis.xaxis.set_major_formatter(FuncFormatter(_compact_number))
        trained = [r for r in history if r['steps'] > 0]
        x = [r['iteration'] for r in trained]
        _series(axes[2], [(label, color, x, [r[key] for r in trained])
                          for key, label, color in POLICY_LOSSES], logarithmic=True, legend_columns=2)
        _series(axes[3], [(label, color, x, [r[key] for r in trained])
                          for key, label, color in VALUE_LOSSES]+
                ([('W-L Q', '#c678dd', x, [r['q_winloss_loss'] for r in trained])]
                 if trained and all('q_winloss_loss' in r for r in trained) else []),
                logarithmic=True, legend_columns=2)
        _series(axes[4], [('Network', ORANGE, x, [r['grad_norm'] for r in trained])], True)
        for axis in (axes[2], axes[3], axes[4]):
            axis.set_xlim(0, max(1.25, max(x, default=1)*1.025))
        cached = [r for r in history if 'total_rows' in r and r.get('submitted', 0) > 0]
        _series(axes[5], [('Cache hits', BLUE, [r['total_rows'] for r in cached],
                          [r['cache_hits']/r['submitted'] for r in cached])])
        axes[5].set_ylim(0, 1.025); axes[5].yaxis.set_major_formatter(PercentFormatter(1))
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
            _series(axes[7],[(label,color,x,[r.get('grad_norms',{}).get(key,float('nan')) for r in trained])
                              for key,label,color in MODULE_GRADIENTS],logarithmic=True)
            axes[7].set_xlim(0,max(1.25,max(x,default=1)*1.025))
        games = sum(r['games'] for r in selfplay)
        figure.supxlabel(f'{len([r for r in trained if "total_rows" in r])} completed training iterations · '
                         f'{games:,} self-play games · {samples:,} effective rows\n'
                         'Loss and gradients: arithmetic means of committed updates, unsmoothed; '
                         'bootstrap is iteration 0', fontsize=10, color='#8a93a3')
    return figure


def loss_figure(history):
    """One weighted loss per panel, with round-mean train and end-of-round validation."""
    import matplotlib as mpl
    trained = [r for r in history if r['steps'] > 0]
    metrics = [('loss', 'Total loss', ORANGE), *POLICY_LOSSES, *VALUE_LOSSES]
    if any('q_winloss_loss' in r for r in trained):
        metrics.append(Q_LOSS)
    columns = 4
    rows = (len(metrics) + columns - 1) // columns
    with mpl.rc_context(THEME):
        figure, axes = _figure(
            [label for _, label, _ in metrics],
            [('Iteration (1-based)', 'Mean weighted loss') for _ in metrics],
            'EtaZero training and validation losses', columns=columns,
            figsize=(18, 3.3 * rows), sharex=True)
        for axis, (key, _, _) in zip(axes, metrics):
            # Keep missing validation as a gap, rather than connecting across a
            # round that had no complete held-out batch or skipped validation.
            train = [r.get(key, float('nan')) for r in trained]
            val = [r.get('validation', {}).get(key, float('nan')) for r in trained]
            x = [r['iteration'] for r in trained]
            _series(axis, [('Train', BLUE, x, train), ('Validation', RED, x, val)],
                    logarithmic=True, show_markers=(True, False))
            axis.set_xlim(0, max(1.25, max(x, default=1)*1.025))
        figure.supxlabel(
            'Train: round mean over committed consumed batches; validation: raw model at round end.\n'
            'Weighted sample-mean losses, unsmoothed; missing validation is not replaced with zero.',
            fontsize=10, color='#8a93a3')
    return figure


def performance_figure(history, batch_size):
    import matplotlib as mpl
    with mpl.rc_context(THEME):
        figure, axes = _figure(
            ('Phase wall time', 'Self-play throughput', 'Learner throughput',
             'Inference mean batch', 'Inference queue wait', 'NN cache hit rate'),
            [('Iteration', 'Seconds (sum of completed attempts)'), ('Iteration', 'Effective rows / second'),
             ('Iteration', 'Committed training samples / second'), ('Iteration', 'Rows / batch'),
             ('Iteration', 'Microseconds / NN request'), ('Iteration', 'Share of submitted requests')],
            'EtaZero execution measurements')
        _series(axes[0], [(phase, color, [r['iteration'] for r in history if phase in r['phases']],
                          [r['phases'][phase] for r in history if phase in r['phases']])
                         for phase, color in zip(('selfplay', 'shuffle', 'train', 'export'), (BLUE, GREEN, ORANGE, RED))])
        for axis, phase, count, label in ((axes[1], 'selfplay', 'rows', 'Self-play'),
                                          (axes[2], 'train', 'steps', 'Learner')):
            rows = [r for r in history if 'total_rows' in r and count in r and r['phases'].get(phase, 0) > 0]
            multiplier = batch_size if phase == 'train' else 1
            _series(axis, [(label, ORANGE, [r['iteration'] for r in rows],
                            [r[count]*multiplier/r['phases'][phase] for r in rows])])
        for axis, numerator, denominator, label in ((axes[3], 'requests', 'batches', 'Mean batch'),
                                                    (axes[4], 'queue_wait_us', 'requests', 'Queue wait'),
                                                    (axes[5], 'cache_hits', 'submitted', 'Cache hits')):
            rows = [r for r in history if r.get(denominator, 0) > 0]
            _series(axis, [(label, BLUE, [r['iteration'] for r in rows], [r[numerator]/r[denominator] for r in rows])])
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
