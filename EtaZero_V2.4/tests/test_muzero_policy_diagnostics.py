import json
import os

import numpy as np
import pytest

from etazero.config import ROOT
from etazero.data import Catalog, metadata, read_raw, validate_raw
from etazero.plotting import METRICS, training_figure
from etazero.runtime import run_training
from etazero.storage import save_npz, sha256
from test_muzero_pipeline import small_config
from test_python import compact_search, winning_record


def test_iteration_invalid_mass_and_marker_free_missing_measurements():
    history = [dict(dict.fromkeys(METRICS, 1.), iteration=i, steps=1,
                    total_rows=10*i, total_samples=8*i, phases={},
                    step_losses=[1., 2.], grad_norms=dict(representation=1., dynamics=2., prediction=3.),
                    **measurement)
               for i, measurement in enumerate([{}, {'root_policy_invalid_mass': .25}, {},
                                                {'root_policy_invalid_mass': .1}], 1)]
    for rows in (history, history[1:2]):
        figure = training_figure(rows, algorithm='muzero')
        axis = figure.axes[5]
        assert axis.get_title() == 'Root NN invalid policy mass'
        line, = axis.lines
        if len(rows) > 1:
            np.testing.assert_array_equal(line.get_xdata(), [10, 20, 30, 40])
            np.testing.assert_equal(line.get_ydata(), [np.nan, .25, np.nan, .1])
        else:
            np.testing.assert_array_equal(line.get_ydata(), [.25])
        assert all(line.get_marker() in ('None', '', None) for axis in figure.axes for line in axis.lines)
        figure.clear()
    old = dict(history[0], submitted=10, cache_hits=4)
    figure = training_figure([old], algorithm='muzero')
    assert not figure.axes[5].lines
    figure.clear()
    figure = training_figure([old], algorithm='alphazero')
    assert figure.axes[5].get_title() == 'NN cache hit rate'
    np.testing.assert_array_equal(figure.axes[5].lines[0].get_ydata(), [.4])
    figure.clear()


def test_small_invalid_mass_remains_visible_as_percentages():
    values = [.0036322966, .0000180212, .0022969174]
    rows = [dict(dict.fromkeys(METRICS, 1.), iteration=i, steps=1, total_rows=100*i,
                 total_samples=8*i, phases={}, root_policy_invalid_mass=value)
            for i, value in enumerate(values, 1)]
    figure = training_figure(rows, algorithm='muzero')
    figure.canvas.draw()
    axis = figure.axes[5]
    bottom, top = axis.get_ylim()
    assert bottom == 0 and max(values) < top < .01
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), values)
    assert axis.yaxis.get_major_formatter()(values[1]) != '0%'
    assert axis.lines[0].get_marker() in ('None', '', None)
    figure.clear()


def test_catalog_weights_roots_and_deduplicates_fragments_after_restart(tmp_path):
    raw, meta = winning_record()
    for name, begin, rows in [('first', 0, 4), ('tail', 4, 5)]:
        fragment = dict(raw)
        info = dict(meta, shard_id=name, row_begin=begin, rows=rows,
                    root_policy_invalid_mass_sum=[.9], root_policy_invalid_mass_count=[9])
        fragment['metadata'] = np.frombuffer(json.dumps(info).encode(), np.uint8)
        save_npz(tmp_path/'selfplay'/f'{name}.npz', fragment)
    raw, meta = winning_record()
    prefix = 4
    raw['opening_moves'][0] = raw['balanced_moves'][0] = prefix
    raw['opening_status'][0] = raw['opening_attempts'][0] = 1
    raw['train_mask'][:prefix] = raw['row_repeats'][:prefix] = raw['target_weights'][:prefix] = 0
    for key in ('policies', 'visits', 'simulations', 'temperatures'):
        raw[key][:prefix] = 0
    compact_search(raw)
    meta.update(attempt_id='second', shard_id='second', rows=5,
                root_policy_invalid_mass_sum=[2.5], root_policy_invalid_mass_count=[5])
    raw['metadata'] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    save_npz(tmp_path/'selfplay/second.npz', raw)
    for restart in range(3):
        if restart == 2:
            (tmp_path/'.internal/catalog.sqlite').unlink()
        catalog = Catalog(tmp_path, 'test', 'config')
        try:
            catalog.scan({1: 'model'})
            stats = catalog.statistics(1)
            assert stats['games'] == 2
            assert stats['root_policy_invalid_mass_count'] == 14
            assert stats['root_policy_invalid_mass_sum'] == pytest.approx(3.4)
            assert stats['root_policy_invalid_mass'] == pytest.approx(3.4/14)
            assert 'root_policy_invalid_mass' not in catalog.statistics(0)
        finally:
            catalog.close()


@pytest.mark.parametrize('total,count', [([1.], [0]), ([float('nan')], [9]), ([-1.], [9]),
                                        ([10.], [9]), ([1.], [1.5]), ([1.], []), ([1.], [10]),
                                        ([1.], [8]), ([1.], None)])
def test_invalid_game_measurements_are_rejected(total, count):
    raw, meta = winning_record()
    meta.update(root_policy_invalid_mass_sum=total, root_policy_invalid_mass_count=count)
    raw['metadata'] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    with pytest.raises(ValueError, match='root policy'):
        validate_raw(raw, deep=False)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA acceptance')
def test_cuda_invalid_mass_pipeline_and_resume(tmp_path):
    import torch
    from etazero.config import validate
    from etazero.plotting import run_history
    torch.set_num_threads(1)
    config = small_config()
    config['inference']['inference_precision'] = 'float16'
    config['writer'].update(shard_rows=8, first_file_min_random_proportion=1)
    config['search']['cheap_search_probs'] = .5
    config['opening']['probability'] = 0
    config['policy_init']['policy_init'] = False
    config['side_positions']['side_position_prob'] = .5
    config['reanalysis'] = dict(use_reanalyze=True, reanalyze_prop=.5, reanalyze_policy_surprise_weight=1,
                              reanalyze_value_surprise_weight=0, reanalyze_surprise_exponent=1,
                              reanalyze_use_outcome_targets=False)
    validate(config)
    root = tmp_path/'run'
    state = run_training(root, config, ROOT/'build/etazero', max_iteration=3)
    totals = {}
    seen = set()
    cheap = reanalyzed = sides = 0
    files = {path: sha256(path) for path in (root/'selfplay').rglob('*.npz')}
    for path in files:
        raw = read_raw(path)
        meta = metadata(raw)
        for i, game_id in enumerate(raw['game_ids']):
            identity = (meta['attempt_id'], meta['worker_id'], int(game_id))
            if identity in seen:
                continue
            seen.add(identity)
            count = meta['root_policy_invalid_mass_count'][i]
            total = meta['root_policy_invalid_mass_sum'][i]
            lo, hi = raw['game_offsets'][i:i+2]
            if meta['model_id'].startswith('random:'):
                assert count == total == 0
            else:
                assert count == hi-lo-raw['opening_moves'][i]
                assert 0 <= total <= count
                sums = totals.setdefault(meta['iteration_id'], [0., 0])
                sums[0] += total
                sums[1] += count
                cheap += raw['cheap_search'][lo:hi].sum()
                reanalyzed += raw['reanalyzed'][lo:hi].sum()
                sides += np.count_nonzero(raw['side_game_indices'] == i)
    assert cheap and reanalyzed and sides
    assert len(totals) == 2
    history = run_history(root)
    for row in history:
        if row['iteration'] in totals:
            total, count = totals[row['iteration']]
            assert row['root_policy_invalid_mass_count'] == count
            assert row['root_policy_invalid_mass'] == pytest.approx(total/count)
        else:
            assert 'root_policy_invalid_mass' not in row
    assert run_training(root, config, ROOT/'build/etazero', resume=True, max_iteration=3) == state
    assert all(sha256(path) == digest for path, digest in files.items())
    rebuilt = Catalog(root, state['run_id'], '')
    try:
        for iteration, (total, count) in totals.items():
            assert rebuilt.statistics(iteration)['root_policy_invalid_mass'] == pytest.approx(total/count)
    finally:
        rebuilt.close()
