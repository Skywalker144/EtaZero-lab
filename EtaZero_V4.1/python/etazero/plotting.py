"""EtaZero figures from committed rounds and source-owned measurements."""
import json
import os
from pathlib import Path
import uuid
import warnings

BLUE, RED, ORANGE, GREEN, GREY = '#61afef', '#e06c75', '#e8924b', '#98c379', '#9aa2b1'
THEME = {
    'figure.facecolor': '#1e2127', 'axes.facecolor': '#282c34',
    'savefig.facecolor': '#1e2127', 'text.color': '#abb2bf',
    'axes.labelcolor': '#abb2bf', 'axes.titlecolor': '#c8ccd4',
    'axes.edgecolor': '#454c5a', 'xtick.color': '#8a93a3', 'ytick.color': '#8a93a3',
    'grid.color': '#3b4250', 'legend.labelcolor': '#abb2bf', 'legend.frameon': False,
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'grid.linewidth': .7,
}

def _series(axis, series, logarithmic=False, legend_columns=1, legend=True):
    import numpy as np
    positive = True
    for label, color, x, y in series:
        values = np.asarray(y, dtype=float)
        if not np.isfinite(values).any():
            continue
        positive &= bool((values[np.isfinite(values)] > 0).all())
        axis.plot(x, values, color=color, linewidth=1.7,
                  marker='None',
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


def _value_loss_series(axis, series):
    import numpy as np
    from matplotlib.lines import Line2D

    handles = []
    has_positive = False
    for label, color, x, y in series:
        values = np.asarray(y, dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        if (finite < 0).any():
            warnings.warn(f'{label} contains negative value losses; omitted from log scale', RuntimeWarning)
        if (finite > 0).any():
            has_positive = True
            line, = axis.plot(x, np.where(np.isfinite(values) & (values > 0), values, np.nan), color=color,
                              linewidth=1.7, marker='None', linestyle='-', label=label)
            handles.append(line)
        else:
            suffix = 'all zero' if (finite == 0).all() else 'no positive values'
            handles.append(Line2D([], [], color=color, linewidth=1.7, label=f'{label} ({suffix})'))
    axis.set_yscale('log')
    if not has_positive:
        axis.text(.5, .5, 'No positive measurements for log scale', ha='center', va='center',
                  color='#8a93a3', transform=axis.transAxes)
    if handles:
        axis.legend(handles=handles, loc='best', fontsize=8, ncols=2, frameon=True,
                    facecolor=THEME['axes.facecolor'], edgecolor='none', framealpha=.9)


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
              if r.get(sample_key) is not None and r.get('elapsed_seconds') is not None and r.get('timing_valid', False)}
    if not points:
        return
    samples = np.asarray(list(points), dtype=float)
    hours = np.asarray(list(points.values()), dtype=float)
    targets = np.linspace(samples[0], samples[-1], 4)
    indices = sorted({int(np.abs(samples - target).argmin()) for target in targets})
    top = axis.secondary_xaxis('top')
    top.set_xticks(samples[indices], [_compact_number(value, None) for value in hours[indices]])
    top.set_xlabel('EtaZero controller elapsed time (h)', fontsize=9)
    top.tick_params(labelsize=9)
    axis.set_title(axis.get_title(), pad=52)


POLICY = (('p0loss', 'Policy'), ('p1loss', 'Opponent policy'), ('p0softloss', 'Soft policy'),
          ('p1softloss', 'Soft opponent policy'), ('p0lopt', 'Long optimistic'), ('p0sopt', 'Short optimistic'))
VALUE = (('vloss', 'Value'), ('tdvloss1', 'TD long'), ('tdvloss2', 'TD mid'),
         ('tdvloss3', 'TD short'), ('evstloss', 'Value error'), ('qwlloss', 'W-L Q'))
LOSSES = (('loss', 'Total loss'), *POLICY, *VALUE, ('fploss', 'Future position'),
          ('vtimeloss', 'Variance time'), ('alogitpen', 'Attention penalty'))
COLORS = (BLUE, RED, GREEN, ORANGE, '#c678dd', '#56b6c2', '#e5c07b')


def run_history(root):
    from .observations import learner_log, learner_display_scales, selfplay_metrics, gatekeeper_metrics
    root = Path(root)
    muzero = 'muzero' in json.loads((root / 'config.json').read_text())['algorithm']
    state = json.loads((root / 'state.json').read_text())
    train = learner_log(root / 'train/metrics_train.json') if (root / 'train/metrics_train.json').exists() or state.get('total_samples', 1) else []
    validation = learner_log(root / 'train/metrics_val.json') if (root / 'train/metrics_val.json').exists() or state.get('total_samples', 1) else []
    history, previous_samples = [], 0
    first = 0 if (root / 'rounds/000000').exists() else 1
    for number in range(first, state['completed_rounds'] + 1):
        directory = root / 'rounds' / f'{number:06d}'
        commit = json.loads((directory / 'commit.json').read_text())
        current = commit['state']
        observed = {}
        for section, records, cursor in (('train', train, 'nsamp'), ('validation', validation, 'nsamp_train')):
            selected = [row for row in records if type(row.get(cursor)) in (int, float)
                        and previous_samples < row[cursor] <= current['total_samples']]
            observed[section + '_logs'] = selected
            observed[section] = selected[-1] if selected else {}
        if observed['train_logs']:
            observed['display_scales'] = learner_display_scales(directory / 'logs')
        try:
            attempts = len(list((directory / 'logs').glob('selfplay*.log')))
            if attempts == 0 and commit.get('sampling') == {'rows': 0, 'games': 0}:
                observed['selfplay'] = dict(rows=0, games=0)
            else:
                observed['selfplay'] = selfplay_metrics(root / 'selfplay' / f'{number:06d}', attempts, muzero=muzero)
            observed['rows'] = observed['selfplay'].get('rows')
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f'Round {number} native selfplay logs unavailable: {error}', RuntimeWarning)
        gatekeeper = directory / 'gatekeeper'
        if commit.get('candidate_model'):
            candidate_logs = gatekeeper / Path(commit['candidate_model']).name
            if candidate_logs.is_dir():
                gatekeeper = candidate_logs
        if gatekeeper.exists():
            try:
                observed['gatekeeper'] = gatekeeper_metrics(gatekeeper)
            except (OSError, ValueError) as error:
                warnings.warn(f'Round {number} native gatekeeper logs unavailable: {error}', RuntimeWarning)
        total_rows = current['replay']['rows']
        history.append({**observed, 'round': number, 'total_rows': total_rows,
                        'total_samples': current['total_samples'],
                        'new_samples': current['total_samples'] - previous_samples,
                        'elapsed_seconds': current['elapsed_seconds'], 'timing_valid': current['timing_valid'],
                        'round_timing': commit['round_timing']})
        previous_samples = current['total_samples']
    return history


def values(history, key, section=None):
    return [(row if section is None else row.get(section, {})).get(key, None) for row in history]


def measured_losses(history):
    present = {key for row in history for section in ('train', 'validation')
               for record in row.get(section + '_logs', []) for key in record}
    ordinary = [(key, title) for key, title in LOSSES if key in present]
    # Preserve source intermediate and MuZero recurrent metrics as distinct
    # curves: ordinary components are root-only, total includes the unroll.
    extra = []
    for prefix, title in (('I', 'Intermediate '), ('mu_recurrent_', 'Recurrent '),
                          ('mu_recurrent_I', 'Recurrent intermediate ')):
        extra.extend((prefix + key, title + label) for key, label in LOSSES if prefix + key in present)
    if 'mu_root_loss' in present:
        extra.append(('mu_root_loss', 'MuZero root total'))
    return ordinary + extra


def logged_values(history, key, section='train', *, transform=None):
    """Place every observation at its native cursor, retaining retry gaps."""
    x, y, previous = [], [], None
    cursor = 'nsamp' if section == 'train' else 'nsamp_train'
    for row in history:
        records = row.get(section + '_logs', [])
        if not records:
            x.append(row['total_samples']); y.append(None)
        for record in records:
            current = record[cursor]
            if previous is not None and current <= previous:
                x.append(current); y.append(None)
            value = record.get(key) if transform is None else transform(row, record, key)
            x.append(current); y.append(value)
            previous = current
    return x, y


def overview_value(row, record, key):
    """Rescale only the overview, preserving native observations and gaps."""
    scales = row.get('display_scales', {})
    if not scales:
        return None
    if key == 'gnorm_batch':
        value = record.get(key)
        return None if value is None else value / scales['batch_size']
    total = 0.
    for prefix, weight in scales['heads'].items():
        if weight == 0:
            continue
        value = record.get(prefix + key)
        if value is None:
            return None
        total += weight * value
    return total * scales['loss'][key]


def _progress_axes(axes, history, key):
    from matplotlib.ticker import FuncFormatter
    maximum = max((row[key] for row in history if row.get(key) is not None), default=0)
    for axis in axes:
        axis.set_xlim(0, max(1, maximum * 1.025))
        axis.xaxis.set_major_formatter(FuncFormatter(_compact_number))
        _elapsed_axis(axis, history, key)


def training_figure(history, algorithm):
    import matplotlib as mpl
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    muzero = 'muzero' in algorithm
    titles = ['Self-play outcomes', 'Search moves per game', 'Root policy losses' if muzero else 'Policy losses',
              'Root value losses' if muzero else 'Value losses', 'Gradient norm', 'Position NN cache hit rate']
    labels = [('Cumulative effective self-play rows', 'Share of recorded games'),
              ('Cumulative effective self-play rows', 'Moves played by search, excluding opening / initial turn'),
              ('Cumulative training samples', 'Weighted loss contribution per root/sample'),
              ('Cumulative training samples', 'Weighted loss contribution per root/sample'),
              ('Cumulative training samples', 'Batch-mean gradient L2 norm, pre-clip'),
              ('Cumulative effective self-play rows', 'Cache hits / position requests')]
    if muzero:
        titles.append('Loss by unroll step')
        labels.append(('Unroll step (0 = representation)', 'Source moving mean per root'))
    with mpl.rc_context(THEME):
        figure, axes = _figure(titles, labels, f'EtaZero V4 · {algorithm}',
                               figsize=(15, 16.5) if muzero else (15, 12.5))
        x = values(history, 'total_rows')
        outcomes = (('black_wins', 'Black wins', BLUE), ('white_wins', 'White wins', RED),
                    ('draws', 'Draws', GREEN), ('no_results', 'No result', '#c678dd'),
                    ('truncated', 'Truncated', GREY))
        if any('black_wins' in row.get('selfplay', {}) for row in history):
            _series(axes[0], [(label, color, x,
                    [row['selfplay'][key] / row['selfplay']['games']
                     if key in row.get('selfplay', {}) and row['selfplay'].get('games', 0) else np.nan
                     for row in history]) for key, label, color in outcomes])
        else:
            axes[0].text(.5, .5, 'Not recorded in native self-play logs', ha='center', va='center',
                         color=GREY, transform=axes[0].transAxes)
        _series(axes[1], [('Mean', ORANGE, x,
                          [row.get('selfplay', {}).get('moves_played', np.nan) / row['selfplay']['games']
                           if row.get('selfplay', {}).get('games', 0) else np.nan for row in history])])
        _series(axes[5], [('Cache hits', BLUE, x, values(history, 'cache_hit_rate', 'selfplay'))])
        for axis in (axes[0], axes[5]):
            axis.set_ylim(0, 1.025)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        _progress_axes((axes[0], axes[1], axes[5]), history, 'total_rows')
        x = values(history, 'total_samples')
        _series(axes[2], [(label, COLORS[i % len(COLORS)], *logged_values(history, key, transform=overview_value))
                          for i, (key, label) in enumerate(POLICY)], logarithmic=True, legend_columns=2)
        _value_loss_series(axes[3], [(label, COLORS[i % len(COLORS)],
                                      *logged_values(history, key, transform=overview_value))
                                     for i, (key, label) in enumerate(VALUE)])
        _series(axes[4], [('Network', ORANGE, *logged_values(history, 'gnorm_batch', transform=overview_value))], logarithmic=True)
        _progress_axes((axes[2], axes[3], axes[4]), history, 'total_samples')
        if muzero:
            measured = [row for row in history if any(key.startswith('mu_step_') for key in row.get('train', {}))]
            series = []
            if measured:
                latest = measured[-1]
                keys = sorted((key for key in latest['train'] if key.startswith('mu_step_') and key.endswith('_loss')),
                              key=lambda key: int(key.split('_')[2]))
                series.append((f'Latest measurement, round {latest["round"]}', BLUE,
                               [int(key.split('_')[2]) for key in keys], [latest['train'][key] for key in keys]))
            _series(axes[6], series, logarithmic=True)
        figure.supxlabel('Loss and gradients: native learner logs at their logged sample counts.\n'
                            'Losses include objective and head weights; MuZero policy/value panels show roots only.\n'
                            'Gradient norms are divided by training batch size; missing measurements/settings are gaps.',
                            fontsize=10, color=GREY)
    return figure


def loss_figure(history):
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    metrics = measured_losses(history) or [('loss', 'Total loss')]
    with mpl.rc_context(THEME):
        figure, axes = _figure([label for _, label in metrics],
            [('Cumulative training samples', 'Source metric per root/sample') for _ in metrics],
            'EtaZero V4 · source training and validation losses\n', columns=4,
            figsize=(18, 3.3 * ((len(metrics) + 3) // 4)), sharex=True)
        figure.legend(handles=[Line2D([], [], color=BLUE, label='Train'), Line2D([], [], color=RED, label='Validation')],
                      loc='upper center', bbox_to_anchor=(.5, 1 - .36 / figure.get_figheight()), ncols=2)
        x = values(history, 'total_samples')
        for axis, (key, _) in zip(axes, metrics):
            _series(axis, [('Train', BLUE, *logged_values(history, key)),
                           ('Validation', RED, *logged_values(history, key, 'validation'))],
                    logarithmic=True, legend=False)
            axis.set_xlim(0, max(1, max(x, default=0) * 1.025))
            axis.xaxis.set_major_formatter(FuncFormatter(_compact_number))
        figure.supxlabel('Train and validation: native metrics JSON logs, excluding updates abandoned on recovery.\n'
                            'Components retain source scaling and need not sum to total loss; no additional smoothing.',
                            fontsize=10, color=GREY)
    return figure


def performance_figure(history):
    import matplotlib as mpl
    import numpy as np
    with mpl.rc_context(THEME):
        figure, axes = _figure(
            ['Native self-play runtime', 'Native self-play throughput', 'Inference mean batch',
             'Position NN cache hit rate', 'Native learner log interval', 'Gatekeeper candidate score'],
            [('Round', 'Seconds, native self-play process'), ('Round', 'Written rows / native runtime'),
             ('Round', 'NN rows / batch'), ('Round', 'Hits / position requests'),
             ('Logged training samples', 'Seconds since previous native print'),
             ('Round', 'Candidate points / played games')], 'EtaZero V4 · native log measurements')
        x = values(history, 'round')
        _series(axes[0], [('Self-play', BLUE, x, values(history, 'seconds', 'selfplay'))])
        _series(axes[1], [('Written rows', ORANGE, x,
                          [row['selfplay']['rows'] / row['selfplay']['seconds']
                           if row.get('selfplay', {}).get('seconds', 0) > 0 and 'rows' in row['selfplay'] else np.nan for row in history])])
        _series(axes[2], [('All NN roles', BLUE, x,
                          [row['selfplay']['nn_rows'] / row['selfplay']['nn_batches']
                           if row.get('selfplay', {}).get('nn_batches', 0) else np.nan for row in history])])
        _series(axes[3], [('Position cache', GREEN, x, values(history, 'cache_hit_rate', 'selfplay'))])
        _series(axes[4], [('Native print interval', GREEN, *logged_values(history, 'time_since_last_print'))])
        _series(axes[5], [('Changing accepted opponent', RED, x,
                          [row['gatekeeper']['candidate_points'] / row['gatekeeper']['games']
                           if row.get('gatekeeper', {}).get('games', 0) else np.nan for row in history])])
        figure.supxlabel('Native self-play, learner and gatekeeper logs; missing measurements stay blank.\n'
                         'Print intervals are not pure training throughput; gatekeeper scores are not Elo.', fontsize=10, color=GREY)
    return figure


def plot_run(root, output=None):
    root = Path(root)
    output = root if output is None else Path(output)
    history = run_history(root)
    config = json.loads((root / 'config.json').read_text())
    figures = [('training.png', lambda: training_figure(history, config['algorithm'])),
               ('loss.png', lambda: loss_figure(history)), ('logs/performance.png', lambda: performance_figure(history))]
    paths = []
    for name, create in figures:
        figure = create()
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.tmp.' + uuid.uuid4().hex)
        try:
            figure.savefig(temporary, format='png', dpi=140, bbox_inches='tight', pad_inches=.15,
                           metadata={'Software': 'EtaZero V4'})
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
            figure.clear()
        paths.append(destination)
    return paths


