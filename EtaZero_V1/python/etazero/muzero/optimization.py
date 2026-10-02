"""MuZero parameter roles for the shared fson optimizer schedule.

Representation/dynamics normalization is internal to the latent model, so its
gamma/bias use normal_gamma/noreg. Only prediction normalization and the heads
use output roles. No AlphaZero model layout or parameter traversal is changed.
"""
import torch

from ..network import BiasMask, FixedScaleMask, MaskedBatchNorm
from ..optimization import group_settings, model_norms


def parameter_groups(model):
    groups = {name: [] for name in ('input', 'normal', 'normal_gamma', 'noreg', 'output', 'output_noreg')}
    input_layers = {model.representation.stem, model.representation.linear_global, model.dynamics.stem}
    output_layers = set(model.prediction.policy_head.modules()) | set(model.prediction.value_head.modules())
    output_layers.add(model.prediction.trunk.norm)
    for layer in model.modules():
        output = layer in output_layers
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            role = 'input' if layer in input_layers else 'output' if output else 'normal'
            groups[role].append(layer.weight)
            if layer.bias is not None:
                groups['output_noreg' if output else 'noreg'].append(layer.bias)
        elif isinstance(layer, (FixedScaleMask, MaskedBatchNorm)):
            groups['output' if output else 'normal_gamma'].append(layer.weight)
            groups['output_noreg' if output else 'noreg'].append(layer.bias)
        elif isinstance(layer, BiasMask):
            groups['output_noreg' if output else 'noreg'].append(layer.bias)
    registered = [id(p) for params in groups.values() for p in params]
    if len(registered) != len(set(registered)) or set(registered) != {id(p) for p in model.parameters()}:
        raise ValueError('MuZero optimizer groups must contain every model parameter exactly once')
    return [{'group_name': name, 'params': params} for name, params in groups.items() if params]


def optimizer_for(model, config):
    plain = config.get('muzero_training', {})
    if not plain.get('katago_optimizer', True):
        groups = [{'group_name': 'all', 'params': list(model.parameters())}]
        options = dict(lr=plain['learning_rate'], weight_decay=plain['weight_decay'])
        if config['optimizer']['kind'] == 'adamw':
            return torch.optim.AdamW(groups, **options, fused=next(model.parameters()).is_cuda)
        return torch.optim.SGD(groups, **options, momentum=0.9)
    groups = parameter_groups(model)
    options = config['optimizer']
    if options['kind'] not in ('sgd', 'adamw'):
        raise ValueError('MuZero optimizer requires sgd or adamw')
    baselines = model_norms(groups)
    for group in groups:
        group['lr'], group['weight_decay'] = group_settings(
            group['group_name'], options, config['training']['batch_size'], 0, {}, baselines, model.norm_kind)
    if options['kind'] == 'adamw':
        return torch.optim.AdamW(groups, fused=next(model.parameters()).is_cuda)
    return torch.optim.SGD(groups, momentum=0.9)


class PlainOptimization:
    """Fixed LR/decay and mean-loss gradients, without Lookahead or averaging."""
    backward_scale = 1
    counter = 0
    swa_count = 0

    def __init__(self, model, config, optimizer, state=None):
        self.batch_size = config['training']['batch_size']
        for key in ('consumed_samples', 'optimizer_steps', 'round_batches', 'subepoch', 'subepoch_batches'):
            setattr(self, key, 0 if state is None else state[key])

    def begin_round(self):
        self.round_batches = 0
        self.subepoch = -1
        self.begin_subepoch()

    def begin_subepoch(self):
        self.subepoch += 1
        self.subepoch_batches = 0

    def before_step(self):
        pass

    def gradient_cap(self, override):
        return override or float('inf')

    def after_step(self, successful=True):
        self.optimizer_steps += int(successful)
        self.consumed_samples += self.batch_size
        self.round_batches += 1
        self.subepoch_batches += 1

    def finish_round(self):
        pass

    def state_dict(self):
        return {'swa': None, **{key: getattr(self, key) for key in
                ('consumed_samples', 'optimizer_steps', 'round_batches', 'subepoch', 'subepoch_batches')}}
