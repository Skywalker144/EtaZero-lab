"""MuZero NBT/ResNet parameter roles for the shared normalized optimizer policy.

Representation/dynamics normalization is internal to the latent model, so its
gamma/bias use normal_gamma/noreg. Only prediction normalization and the heads
use output roles. No AlphaZero model layout or parameter traversal is changed.
"""
import torch

from ..network import BiasMask, FixedScaleMask, MaskedBatchNorm
from ..optimization import group_settings, optimizer_from_groups
from .resnet import MaskedNorm


def parameter_groups(model):
    groups = {name: [] for name in ('input', 'normal', 'normal_gamma', 'noreg', 'output', 'output_noreg')}
    input_layers = {model.representation.stem, model.representation.linear_global, model.dynamics.stem}
    output_layers = set(model.prediction.policy_head.modules()) | set(model.prediction.value_head.modules())
    if model.architecture == 'nbt':
        output_layers.add(model.prediction.trunk.norm)
    for layer in model.modules():
        output = layer in output_layers
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            role = 'input' if layer in input_layers else 'output' if output else 'normal'
            groups[role].append(layer.weight)
            if layer.bias is not None:
                groups['output_noreg' if output else 'noreg'].append(layer.bias)
        elif isinstance(layer, (FixedScaleMask, MaskedBatchNorm, MaskedNorm)):
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
        from ..plain_optimization import optimizer_for as plain_optimizer
        kind = config['optimizer']['kind']
        return plain_optimizer(model, kind, plain, parameter_groups(model) if kind == 'muon' else None)
    groups = parameter_groups(model)
    options = config['optimizer']
    for group in groups:
        group['lr'], group['weight_decay'] = group_settings(
            group['group_name'], options, config['training']['batch_size'], 0, {}, {}, model.norm_kind)
    return optimizer_from_groups(groups, options['kind'])
