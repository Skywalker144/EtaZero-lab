"""Unroll loss using the source Metrics on each real or absorbing target row.

Trajectory batches are supplied by muzero_data; the formal learner integration
is separate. This function does
not manufacture future targets, choose samples, or advance optimizer clocks.
"""
import torch
from contextlib import nullcontext
from .metrics_pytorch import Metrics
from .muzero_model import scale_gradient


def iter_unroll_metrics(model, root_batch, target_steps, actions, *, active_steps=None,
                        hidden_gradient_scale=0.5, metrics_fn=None, component_forward=None,
                        autocast_context=nullcontext, **metric_options):
    """Forward each active step under AMP, then apply source postprocessing/loss.

    Compilation is confined to component forwards and the source metrics call;
    CPU sequence boundaries and learner clocks remain outside compiled graphs.
    """
    steps = actions.shape[1]
    if steps < 1 or len(target_steps) != steps + 1:
        raise ValueError('MuZero training requires K actions and K+1 target sets, K >= 1')
    if metrics_fn is None:
        metrics_fn = Metrics(1, model.prediction).metrics_dict_batchwise
    results = model.unroll(
            root_batch['binaryInputNCHW'], root_batch['globalInputNC'], actions,
            active_steps=active_steps, hidden_gradient_scale=hidden_gradient_scale,
            input_meta=root_batch.get('metadataInputNC'), component_forward=component_forward)
    for index in range(steps + 1):
        with autocast_context():
            result = next(results, None)
        if result is None:
            break
        rows = result.rows
        batch = {key: value[rows] for key, value in target_steps[index].items()}
        # Metrics consumes the board mask, not a future observation.
        batch['binaryInputNCHW'] = root_batch['binaryInputNCHW'][rows]
        batch['globalInputNC'] = root_batch['globalInputNC'][rows]
        if 'metadataInputNC' in root_batch:
            batch['metadataInputNC'] = root_batch['metadataInputNC'][rows]
        postprocessed = model.prediction.postprocess_output(result.outputs)
        current = metrics_fn(model.prediction, postprocessed, None, batch,
                             include_model_norms=False, **metric_options)
        yield index, result, current


def unroll_loss(model, root_batch, target_steps, actions, *, active_steps=None,
                hidden_gradient_scale=0.5, **metric_options):
    """Source batch-summed loss: root backward 1, future 1/K, unscaled logs."""
    steps = actions.shape[1]
    logged = []
    total = None
    for index, result, current in iter_unroll_metrics(
            model, root_batch, target_steps, actions, active_steps=active_steps,
            hidden_gradient_scale=hidden_gradient_scale, **metric_options):
        contribution = current['loss_sum'] if index == 0 else scale_gradient(current['loss_sum'], 1.0 / steps)
        total = contribution if total is None else total + contribution
        logged.append({key: value.detach() if isinstance(value, torch.Tensor) else value
                       for key, value in current.items()})
    if total is None:
        raise ValueError('A training batch must contain an active root')
    return total, logged
