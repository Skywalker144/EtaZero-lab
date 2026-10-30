"""MuZero model/loss boundaries for the source learner's training loop.

Optimizer steps, LR schedules, Lookahead, SWA, and file selection stay in
train.py. Independently configured components retain their own source parameter
roles, norm baselines, weight decay and gradient clipping thresholds.
"""
from dataclasses import asdict
import math
from contextlib import nullcontext

import torch
from torch import nn

from .model_pytorch import Model
from .metrics_pytorch import Metrics
from .muzero_model import MuZeroConfig, MuZeroModel, LatentBody, scale_gradient
from .muzero_training import iter_unroll_metrics
from . import trainloop_helpers


def is_muzero_config(config):
    return config.get('algorithm') == 'muzero'


def create_model(config, pos_len):
    return (MuZeroModel(MuZeroConfig(**config['network']), pos_len)
            if is_muzero_config(config) else Model(config, pos_len))


def checkpoint_config(model):
    return ({'algorithm': 'muzero', 'network': asdict(model.network_config)}
            if isinstance(model, MuZeroModel) else model.config)


def components(model):
    return model.training_components() if isinstance(model, MuZeroModel) else {'': model}


def source_models(model):
    return {key: value.trunk if isinstance(value, LatentBody) else value
            for key, value in components(model).items()}


def uses_brenorm(model):
    return any(component.get_norm_kind() in ('brenorm', 'fixbrenorm')
               for component in components(model).values())


def initialize_component_baselines(model, train_state):
    if isinstance(model, MuZeroModel) and 'muzero_modelnorm_baselines' not in train_state:
        train_state['muzero_modelnorm_baselines'] = {
            key: Metrics.get_model_norms(component) for key, component in components(model).items()}


def weight_decay_context(model, train_state, running_metrics, component_name):
    if not component_name:
        return model, train_state, running_metrics
    baseline = train_state['muzero_modelnorm_baselines'][component_name]
    state = dict(train_state, modelnorm_normal_baseline=baseline['normal'], modelnorm_input_baseline=baseline['input'])
    prefix = component_name + '_'
    metrics = {kind: {key[len(prefix):]: value for key, value in values.items() if key.startswith(prefix)}
               for kind, values in running_metrics.items()}
    return components(model)[component_name], state, metrics


def detensorify_metrics(metrics):
    """Retain source overflow handling for each component, with one device transfer."""
    result = trainloop_helpers.detensorify_metrics(metrics)
    for component in ('representation', 'dynamics', 'prediction'):
        names = {key: component + '_' + key for key in ('gnorm_batch', 'exgnorm_sum')}
        local = {key: result[name] for key, name in names.items() if name in result}
        # Values are already Python scalars. Reuse the source's exact omission
        # rule without another tensor transfer or a separate threshold policy.
        retained = trainloop_helpers.detensorify_metrics(local)
        for key in local.keys() - retained.keys():
            del result[names[key]]
    return result


class MuZeroMetrics(Metrics):
    def __init__(self, world_size, model):
        super().__init__(world_size, model.prediction)

    @staticmethod
    def get_model_norm_metrics(model):
        result = Metrics.get_model_norm_metrics(model)
        if isinstance(model, MuZeroModel):
            for name, component in components(model).items():
                result.update({name + '_' + key: value
                               for key, value in Metrics.get_model_norm_metrics(component).items()})
        return result

    def get_specific_norms_and_gradient_stats(self, model):
        result = {}
        for name, component in source_models(model).items():
            result.update({name + '_' + key: value for key, value in
                           super().get_specific_norms_and_gradient_stats(component).items()})
        return result


class MuZeroForward(nn.Module):
    """Own compiled callables outside the raw model and its SWA/export copies."""
    def __init__(self, model, no_compile):
        super().__init__()
        self.model = model
        forwards = []
        for component in components(model).values():
            if not no_compile:
                source = component.trunk if isinstance(component, LatentBody) else component
                trainloop_helpers.configure_model_for_compile(source)
                forwards.append(torch.compile(component, mode=trainloop_helpers.get_compile_mode()))
            else:
                forwards.append(component)
        # A tuple is intentionally not a ModuleList: each parameter is already
        # registered exactly once in model, and compiled names are not persisted.
        self.component_forward = tuple(forwards)
        self.attention_loss = None

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def forward(self, batch, *, metrics_fn, metric_options, hidden_gradient_scale=0.5,
                autocast_context=nullcontext):
        root, targets, actions, active = batch
        total = None
        result_metrics = {}
        self.attention_loss = None
        upper_bounds = []
        for index, step, current in iter_unroll_metrics(
                self.model, root, targets, actions, active_steps=active,
                hidden_gradient_scale=hidden_gradient_scale, metrics_fn=metrics_fn,
                component_forward=self.component_forward, autocast_context=autocast_context, **metric_options):
            if index == 0:
                result_metrics = {key: value.detach() if isinstance(value, torch.Tensor) else value
                                  for key, value in current.items()}
                result_metrics['mu_root_loss_sum'] = current['loss_sum'].detach()
                # Each step is averaged over all roots by source metric logging.
                # Inactive future steps contribute zero forward supervision;
                # the backward-only 1/K scale is not applied to this observation.
                result_metrics.update({f'mu_step_{k}_loss_sum': 0.0 for k in range(actions.shape[1] + 1)})
            else:
                for key, value in current.items():
                    # Report recurrent totals per root, with explicit names.
                    # Ordinary accuracy/weight/sample metrics keep root semantics.
                    if key.endswith('_sum'):
                        name = 'mu_recurrent_' + key
                        result_metrics[name] = result_metrics.get(name, 0) + value.detach()
                result_metrics['mu_recurrent_samples_sum'] = result_metrics.get('mu_recurrent_samples_sum', 0) + len(step.rows)
            result_metrics[f'mu_step_{index}_loss_sum'] = current['loss_sum'].detach()
            loss = current['loss_sum']
            contribution = loss if index == 0 else scale_gradient(loss, 1/actions.shape[1])
            total = contribution if total is None else total + contribution
            # Capture penalties before the next recurrent forward overwrites the
            # source modules' per-forward observations. Same backward scale as loss.
            for source in (self.model.representation.trunk if index == 0 else self.model.dynamics.trunk,
                           self.model.prediction):
                if source.attn_logit_penalty_cap is not None:
                    penalty = source.attn_logit_penalty_per_sample.mean() * len(step.rows)
                    contribution = penalty if index == 0 else scale_gradient(penalty, 1/actions.shape[1])
                    self.attention_loss = contribution if self.attention_loss is None else self.attention_loss + contribution
                    upper_bounds.append(source.attn_logit_ub_batch_max)
        if total is None:
            raise ValueError('MuZero batch has no active roots')
        result_metrics['loss_sum'] = total
        if self.attention_loss is not None:
            result_metrics['alogitpen_sum'] = self.attention_loss.detach()
            result_metrics['alogitubmax_batch'] = torch.stack(upper_bounds).max()
        return result_metrics


def clip_component_gradients(model, metrics, batch_size, *, adaptive_optimizer, clip_scale, lr_scale):
    """Apply source norm-dependent clipping independently to the three networks."""
    caps = []
    norms = []
    excess = []
    for name, component in components(model).items():
        kind = component.get_norm_kind()
        if adaptive_optimizer:
            cap = 11000.0
        elif kind in ('fixup', 'fixscale', 'fixscaleonenorm'):
            cap = 2500.0
        elif kind in ('bnorm', 'brenorm', 'fixbrenorm'):
            cap = 5500.0
        else:
            raise ValueError(f'Unsupported component normalization: {kind}')
        cap *= (1.0 if clip_scale is None else clip_scale) * math.sqrt(batch_size/256.0)
        cap /= math.sqrt(max(0.0000001, lr_scale))
        local = {}
        trainloop_helpers.clip_gradients_and_record(component, cap, local, batch_size)
        # PyTorch returns a CPU zero when this component has no gradients (a
        # side-only batch never executes dynamics). The source metric collector
        # stacks scalar tensors on the learner device before a single transfer.
        local = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value
                 for key, value in local.items()}
        metrics.update({name + '_' + key: value for key, value in local.items()})
        metrics[name + '_gnorm_cap_batch'] = cap
        caps.append(cap)
        if 'gnorm_batch' in local:
            norms.append(torch.as_tensor(local['gnorm_batch'], device=model.device))
            excess.append(torch.as_tensor(local['exgnorm_sum'], device=model.device))
    if norms:
        metrics['gnorm_batch'] = torch.stack(norms).square().sum().sqrt()
        metrics['exgnorm_sum'] = torch.stack(excess).sum()
    # The global bound is for reporting/watching, not a second clipping pass.
    return math.sqrt(sum(cap*cap for cap in caps))
