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
METRICS = ('loss', 'policy_loss', 'value_loss', 'grad_norm')


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
    updates, statistics, completed, phases, inference = {}, {}, {}, {}, {}
    for event in journal_events(root/'logs/events.jsonl'):
        kind = event['event']; iteration = event.get('iteration')
        if kind == 'plan':
            # Only the latest attempt of a committed round contributes timings.
            phases.pop(iteration, None)
            for key in [k for k in inference if k[0] == iteration]:
                del inference[key]
        if kind == 'update' and event['update_id'] in committed:
            updates[event['update_id']] = event
        elif kind == 'selfplay_statistics' and iteration < state['iteration']:
            statistics[iteration] = event
        elif kind == 'iteration_complete' and iteration < state['iteration']:
            completed[iteration] = event
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
        for metric in METRICS:
            row[metric] = row.get(metric, 0) + event[metric]
    for row in history.values():
        if row['steps']:
            for metric in METRICS:
                row[metric] /= row['steps']
        counters = [e for (iteration, _, _), e in inference.items() if iteration == row['iteration']]
        row['requests'] = sum(e['requests'] for e in counters)
        row['batches'] = sum(e['batches'] for e in counters)
        row['queue_wait_us'] = sum(e['queue_wait_us'] for e in counters)
        row['submitted'] = sum(e['submitted'] for e in counters)
        row['cache_hits'] = sum(e['cache_hits'] for e in counters)
    return [history[k] for k in sorted(history)]


def _series(axis, series, logarithmic=False):
    import numpy as np
    positive = True
    for label, color, x, y in series:
        values = np.asarray(y, dtype=float)
        if not np.isfinite(values).any():
            continue
        positive &= bool((values[np.isfinite(values)] > 0).all())
        axis.plot(x, values, color=color, linewidth=1.7, marker='o' if len(values) == 1 else None, label=label)
    if axis.lines:
        if logarithmic and positive:
            axis.set_yscale('log')
        axis.legend(loc='best', fontsize=9)
    else:
        axis.text(.5, .5, 'No measurements yet', ha='center', va='center',
                  color='#8a93a3', transform=axis.transAxes)


def _figure(titles, labels, title):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import MaxNLocator
    figure = Figure(figsize=(15, 13.5), layout='constrained')
    FigureCanvasAgg(figure)
    axes = list(figure.subplots(3, 2).flat)
    figure.suptitle(title, fontsize=17)
    for axis, name, (xlabel, ylabel) in zip(axes, titles, labels):
        axis.set(title=name, xlabel=xlabel, ylabel=ylabel)
        axis.grid(axis='y'); axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    return figure, axes


def training_figure(history):
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    with mpl.rc_context(THEME):
        figure, axes = _figure(
            ('Self-play outcomes', 'Self-play game length', 'Total loss', 'Loss components',
             'Effective training rows per game', 'Gradient norm'),
            [('Cumulative effective self-play rows', 'Share of games'),
             ('Cumulative effective self-play rows', 'Moves per game'),
             ('Iteration (1-based)', 'Mean loss'), ('Iteration (1-based)', 'Mean loss'),
             ('Cumulative effective self-play rows', 'Effective rows per game'),
             ('Iteration (1-based)', 'Mean L2 grad norm (before clipping)')],
            'EtaZero AlphaZero training progress')
        selfplay = [r for r in history if r.get('games', 0) > 0]
        x = [r['total_rows'] for r in selfplay]
        _series(axes[0], [(label, color, x, [r[key]/r['games'] for r in selfplay])
                         for key, label, color in (('black_wins', 'Black win', BLUE),
                                                   ('white_wins', 'White win', RED), ('draws', 'Draw', GREY))])
        axes[0].set_ylim(0, 1.025); axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        _series(axes[1], [('Mean (including opening)', ORANGE, x, [r['avg_game_length'] for r in selfplay])])
        _series(axes[4], [('Mean (excluding opening)', BLUE, x, [r['avg_rows_per_game'] for r in selfplay])])
        samples = selfplay[-1]['total_rows'] if selfplay else 0
        for axis in (axes[0], axes[1], axes[4]):
            axis.set_xlim(0, max(1, samples*1.025))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3g}'))
        trained = [r for r in history if r['steps'] > 0]
        x = [r['iteration'] for r in trained]
        _series(axes[2], [('Total', ORANGE, x, [r['loss'] for r in trained])], True)
        _series(axes[3], [(label, color, x, [r[key] for r in trained]) for key, label, color in
                          (('policy_loss', 'Policy', BLUE), ('value_loss', 'Value', GREEN))], True)
        _series(axes[5], [('Network', ORANGE, x, [r['grad_norm'] for r in trained])], True)
        for axis in (axes[2], axes[3], axes[5]):
            axis.set_xlim(0, max(1.25, max(x, default=1)*1.025))
        games = sum(r['games'] for r in selfplay)
        figure.supxlabel(f'{len([r for r in trained if "total_rows" in r])} completed training iterations · '
                         f'{games:,} self-play games · {samples:,} effective rows\n'
                         'Loss and gradients: arithmetic means of committed updates, unsmoothed; '
                         'bootstrap is iteration 0', fontsize=10, color='#8a93a3')
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
    figures = [(root/'training.png', training_figure(history)),
               (root/'logs/performance.png', performance_figure(history, config['training']['batch_size']))]
    for destination, figure in figures:
        try:
            atomic_write(destination, lambda path: figure.savefig(path, format='png', dpi=160,
                                                                  facecolor=THEME['figure.facecolor']))
        finally:
            figure.clear()
    return root/'training.png'
