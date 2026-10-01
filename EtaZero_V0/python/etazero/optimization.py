"""KataGo BN optimizer policy, adapted to EtaZero's residual policy/value net."""
import math
import torch
from torch.optim.swa_utils import AveragedModel


def parameter_groups(model):
    groups = {name: [] for name in ('input', 'normal', 'normal_gamma', 'noreg', 'output', 'output_noreg')}
    groups['input'] += [model.stem.weight, model.linear_global.weight]
    groups['normal_gamma'].append(model.stem_bn.weight)
    groups['noreg'].append(model.stem_bn.bias)
    for block in model.blocks:
        groups['normal'] += [block.conv1.weight, block.conv2.weight]
        groups['normal_gamma'] += [block.bn1.weight, block.bn2.weight]
        groups['noreg'] += [block.bn1.bias, block.bn2.bias]
    for name in ('policy_conv', 'policy_out', 'value_conv', 'value_hidden', 'value_out'):
        layer = getattr(model, name)
        groups['output'].append(layer.weight)
        if layer.bias is not None:
            groups['output_noreg'].append(layer.bias)
    for name in ('policy_bn', 'value_bn'):
        norm = getattr(model, name)
        groups['output'].append(norm.weight)
        groups['output_noreg'].append(norm.bias)
    registered = [id(p) for params in groups.values() for p in params]
    if len(registered) != len(set(registered)) or set(registered) != {id(p) for p in model.parameters()}:
        raise ValueError('Optimizer groups must contain every model parameter exactly once')
    return [{'group_name': name, 'params': params} for name, params in groups.items() if params]


def model_norms(groups):
    with torch.no_grad():
        return {g['group_name']: torch.sqrt(sum(p.float().square().sum() for p in g['params'])).item()
                for g in groups if g['group_name'] in ('input', 'normal')}


def warmup_factor(samples, enabled=True):
    if not enabled:
        return 1.0
    denominators = (20, 14, 10, 7, 5, 3, 2, 1.4)
    return 1 / denominators[int(samples // 250000)] if samples < 2000000 else 1.0


def group_settings(name, options, batch_size, samples, norms, baselines):
    adamw = options['kind'] == 'adamw'
    batch_scale = math.sqrt(batch_size / 256) if adamw else batch_size / 256
    warmup = warmup_factor(samples, options['lr_warmup'])
    scale = options['lr_scale']
    lr = (1.33 if adamw else 1) * 0.00003 * scale * warmup / options['lookahead_alpha']
    if adamw:
        lr *= batch_scale
    if name in ('output', 'output_noreg'):
        lr *= options['head_lr_factor']
    if name in ('noreg', 'output_noreg'):
        lr *= options['noreg_lr_factor']
    if name in ('input', 'normal', 'normal_gamma'):
        norm_key = 'input' if name == 'input' else 'normal'
        adaptive = 1.0
        if norm_key in norms and baselines[norm_key] > 0:
            ratio = norms[norm_key] / (baselines[norm_key] + 1e-30)
            adaptive = 2 ** (2 * math.tanh(math.log(ratio + 1e-30) * 3))
        factor = options['input_wd_factor'] if name == 'input' else options['normal_wd_factor'] if name == 'normal' else 1
        if name == 'input' and adamw:
            factor *= 2 / 3
        if name == 'normal_gamma':
            factor *= 0.25 if adamw else 0.125
        wd = (0.009 if adamw else 0.00125) * batch_scale * (scale * warmup) ** 0.75 * adaptive * factor
    elif name == 'output':
        wd = (0.004 if adamw else 1e-6) * batch_scale
    elif name == 'noreg':
        wd = 1e-6 * batch_scale * (scale * warmup) ** 0.75
    elif name == 'output_noreg':
        wd = (1e-6 if adamw else 1e-8) * batch_scale
    else:
        raise ValueError(f'Unknown optimizer group: {name}')
    return lr, wd


def optimizer_for(model, config):
    groups = parameter_groups(model)
    options = config['optimizer']
    baselines = model_norms(groups)
    for group in groups:
        group['lr'], group['weight_decay'] = group_settings(
            group['group_name'], options, config['training']['batch_size'], 0, {}, baselines)
    if options['kind'] == 'adamw':
        return torch.optim.AdamW(groups, fused=next(model.parameters()).is_cuda)
    return torch.optim.SGD(groups, momentum=0.9)


class Optimization:
    """Own the schedule, slow weights and inference average alongside an optimizer."""
    def __init__(self, model, config, optimizer, state=None):
        self.model, self.optimizer = model, optimizer
        self.options, self.batch_size = config['optimizer'], config['training']['batch_size']
        self.period = self.options['swa_period_samples'] or max(1, config['training']['train_steps'] * self.batch_size // 2)
        self.baselines = model_norms(optimizer.param_groups) if state is None else state['baselines']
        self.norms = {} if state is None else state['norms']
        self.counter = 0 if state is None else state['lookahead_counter']
        self.swa_samples = 0 if state is None else state['swa_samples']
        self.slow = {name: p.detach().clone() for name, p in model.named_parameters()}
        if state is not None:
            if set(state['slow']) != set(self.slow):
                raise ValueError('Lookahead state does not match model parameters')
            for name, p in self.slow.items():
                p.copy_(state['slow'][name])
        new_factor = 1 / self.options['swa_scale']
        self.swa = AveragedModel(model, avg_fn=lambda avg, current, count: avg * (1 - new_factor) + current * new_factor,
                                 use_buffers=False)
        if state is not None:
            self.swa.load_state_dict(state['swa'])

    def configure(self, total_steps):
        # The reference samples norms at its 100-batch print interval. The
        # schedule itself is refreshed each update here, including after resume.
        if (total_steps + 1) % self.options['norm_interval'] == 0:
            self.norms = model_norms(self.optimizer.param_groups)
        samples = total_steps * self.batch_size
        for group in self.optimizer.param_groups:
            group['lr'], group['weight_decay'] = group_settings(
                group['group_name'], self.options, self.batch_size, samples, self.norms, self.baselines)

    def gradient_cap(self, override):
        # EtaZero logs mean losses. Backward uses the batch sum, like KataGo.
        if override:
            return override * self.batch_size
        cap = 11000 if self.options['kind'] == 'adamw' else 5500
        return cap * math.sqrt(self.batch_size / 256) / math.sqrt(max(1e-7, self.options['lr_scale']))

    @torch.no_grad()
    def after_step(self):
        synced = True
        if self.options['lookahead_alpha'] < 1:
            self.counter += 1
            synced = self.counter == self.options['lookahead_k']
            if synced:
                for name, p in self.model.named_parameters():
                    self.slow[name].add_(p - self.slow[name], alpha=self.options['lookahead_alpha'])
                    p.copy_(self.slow[name])
                self.counter = 0
        self.swa_samples += self.batch_size
        if self.swa_samples >= self.period and synced:
            self.swa.update_parameters(self.model)
            self.swa_samples = 0

    @torch.no_grad()
    def finish_round(self):
        # Match the native epoch boundary: discard leftover fast weights while
        # retaining optimizer moments and successful-update/sample counters.
        if self.options['lookahead_alpha'] < 1:
            for name, p in self.model.named_parameters():
                p.copy_(self.slow[name])
            self.counter = 0

    def state_dict(self):
        return {'baselines': self.baselines, 'norms': self.norms,
                'lookahead_counter': self.counter, 'slow': self.slow,
                'swa_samples': self.swa_samples, 'swa': self.swa.state_dict()}


def inference_weights(checkpoint):
    swa = checkpoint['optimization']['swa']
    if swa['n_averaged'].item() == 0:
        return checkpoint['model']
    return {name.removeprefix('module.'): value for name, value in swa.items() if name.startswith('module.')}
