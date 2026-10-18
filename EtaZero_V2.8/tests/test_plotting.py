"""Sample and elapsed-time axes retain measured progress, including idle rounds."""
import numpy as np

from etazero.plotting import METRICS, loss_figure, training_figure


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
