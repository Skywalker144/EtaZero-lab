"""Sample and elapsed-time axes retain measured progress, including idle rounds."""
import numpy as np
import pytest

from etazero.plotting import METRICS, loss_figure, training_figure
from etazero.plotting import run_history, performance_figure
from etazero.storage import save_json


def test_sample_and_time_axes_follow_round_measurements():
    history = [dict(dict.fromkeys(METRICS, 1.), iteration=i, steps=int(i in (1, 2, 3)),
                    total_rows=rows, total_samples=samples, elapsed_seconds=seconds,
                    phases={})
               for i, (rows, samples, seconds) in enumerate([
                   (1000, 0, 900), (8000, 100000, 1800),
                   (18000, 120000, 10800), (23000, 300000, 12600),
                   (23000, 300000, 14400)])]
    figure = training_figure(history)
    figure.canvas.draw()
    for axis in figure.axes[2:5]:
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [100000, 120000, 300000])
        assert axis.xaxis.get_major_formatter()(120000) == '1.2e5'
        top, = axis.child_axes
        np.testing.assert_array_equal(top.get_xticks(), [0, 100000, 120000, 300000])
        assert [label.get_text() for label in top.get_xticklabels()] == ['0.25', '0.5', '3', '4']
    top, = figure.axes[0].child_axes
    np.testing.assert_array_equal(top.get_xticks(), [1000, 8000, 18000, 23000])
    figure.clear()


def test_loss_uses_one_shared_legend_even_without_validation():
    history = [dict(dict.fromkeys(METRICS, 1.), iteration=1, steps=1, total_samples=120000)]
    for rows in ([], history):
        figure = loss_figure(rows)
        figure.canvas.draw()
        legend, = figure.legends
        assert [label.get_text() for label in legend.get_texts()] == ['Train', 'Validation']
        assert all(axis.get_legend() is None for axis in figure.axes)
        if rows:
            np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), [120000])
        figure.clear()


@pytest.mark.parametrize('missing', ['absent', 'corrupt'])
def test_missing_round_metrics_preserve_progress_and_plot_gaps(tmp_path, missing):
    save_json(tmp_path/'.internal/state.json', {'iteration': 4})
    for i in range(5):
        save_json(tmp_path/'.internal/iterations'/f'{i:06d}'/'status.json',
                  {'commit': dict(iteration=i, steps=int(i > 0), total_samples=i*16,
                                  total_rows=100+i*20, elapsed_seconds=i*10, model=None)})
        if i == 2:
            if missing == 'corrupt':
                path = tmp_path/'logs/iterations/000002.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{partial')
            continue
        save_json(tmp_path/'logs/iterations'/f'{i:06d}.json',
                  dict(dict.fromkeys(METRICS, 1.), phases={'train': 1},
                       games=2, black_wins=1, white_wins=1, draws=0, avg_game_length=8,
                       requests=10, batches=2, submitted=12, cache_hits=2))
    with pytest.warns(RuntimeWarning, match='Round 2 metrics unavailable'):
        history = run_history(tmp_path)
    assert [r['iteration'] for r in history] == [0, 1, 2, 3]
    assert history[2]['total_samples'] == 32 and 'loss' not in history[2]
    for make in (training_figure, loss_figure):
        figure = make(history)
        figure.canvas.draw()
        axis = figure.axes[2] if make is training_figure else figure.axes[0]
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [16, 32, 48])
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [1., np.nan, 1.])
        figure.clear()
    figure = performance_figure(history, 16)
    figure.canvas.draw()
    figure.clear()
