import math
import pytest
from etazero.train.training_metrics import TrainingMetrics
from etazero.evaluation.plotting import round_history
from etazero.storage import save_json


def test_all_batch_means_extrema_and_resume_are_independent_of_print_cadence():
    a = TrainingMetrics()
    a.add({'loss': 2.}, 3., False, False, [1., 1.])
    a.add({'loss': 8.}, 600., False, True, [3., 5.])
    b = TrainingMetrics(a.state)
    b.add({'loss': 5.}, float('inf'), True, True, [2., 3.])
    result = b.summary()
    assert result['steps'] == 3 and a.summary()['steps'] == 2
    assert result['loss'] == 5. and result['grad_norm'] == 301.5
    assert result['max_grad_norm'] == 600 and result['max_loss'] == 8
    assert result['clipped_steps'] == result['amp_skipped_steps'] == 1
    assert result['step_losses'] == [2., 3.]
    assert result['step_loss_updates'] == 3


def test_all_skipped_has_no_fabricated_gradient_measurement():
    a = TrainingMetrics()
    a.add({'loss': 3.}, float('inf'), True, False)
    assert math.isnan(a.summary()['grad_norm']) and a.summary()['gradient_steps'] == 0


def test_sparse_history_uses_restored_final_summary_and_rejects_missing_tail(tmp_path):
    import json
    metrics = TrainingMetrics()
    for loss in (2., 8., 5.):
        metrics.add({'loss': loss}, 3., False, False)
    events = [dict(event='update', iteration=1, step=2, round_metrics=dict(steps=2,loss=999)),
              dict(event='update', iteration=1, step=3, round_metrics=metrics.summary()),
              dict(event='update', iteration=1, step=4, round_metrics=dict(steps=4,loss=999))]
    path = tmp_path/'logs/events.jsonl'; path.parent.mkdir()
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    assert round_history(tmp_path, 1, 3, 0)['loss'] == 5.
    with pytest.raises(ValueError, match='final learner cursor'):
        round_history(tmp_path, 1, 5, 0)
