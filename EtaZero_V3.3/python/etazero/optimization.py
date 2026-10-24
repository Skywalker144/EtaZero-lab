"""KataGo optimizer policies, including the MuZero masked-normalization adaptation."""
import math
import torch
from torch.optim.swa_utils import AveragedModel
from .network import FixedScaleMask, BiasMask
from .lr_schedule import effective_scale


def parameter_groups(model):
    if hasattr(model, 'parameter_groups'):
        return model.parameter_groups()
    groups = {name: [] for name in ('input', 'normal', 'normal_attn', 'normal_gamma', 'noreg', 'output', 'output_noreg')}
    groups['input'] += [model.stem.weight, model.linear_global.weight]
    for block in model.blocks:
        for layer in block.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                groups[getattr(layer,'parameter_group','normal')].append(layer.weight)
            elif isinstance(layer, FixedScaleMask):
                groups['normal_gamma'].append(layer.weight)
                groups['noreg'].append(layer.bias)
            elif isinstance(layer, BiasMask):
                groups['noreg'].append(layer.bias)
            elif isinstance(layer, torch.nn.RMSNorm):
                groups['noreg'].append(layer.weight)
    # Final fson BN belongs to the output groups in KataGo, like the heads.
    if hasattr(model.trunk_norm,'weight'):
        groups['output'].append(model.trunk_norm.weight)
    groups['output_noreg'].append(model.trunk_norm.bias)
    for head in (model.policy_head, model.value_head):
        for layer in head.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                groups['output'].append(layer.weight)
                if layer.bias is not None:
                    groups['output_noreg'].append(layer.bias)
            elif isinstance(layer, BiasMask):
                groups['output_noreg'].append(layer.bias)
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


def group_settings(name, options, batch_size, samples, norms, baselines, norm_kind='fixscaleonenorm', schedule=None, completed_rounds=0):
    muon = options['kind'] in ('muon', 'normuon', 'aurora')
    adamw = options['kind'] == 'adamw' or muon
    batch_scale = math.sqrt(batch_size / 256) if adamw else batch_size / 256
    warmup = warmup_factor(samples, options['lr_warmup'])
    scale = effective_scale(options['lr_scale'], schedule or {}, samples, completed_rounds)
    lr = (1.33 if adamw else 1) * 0.00003 * scale * warmup / options['lookahead_alpha']
    if adamw:
        lr *= batch_scale
    if muon and name in ('normal', 'normal_attn', 'normal_gab', 'gab_mlp', 'tab_module'):
        lr *= 2
    if name in ('output', 'output_noreg'):
        lr *= options['head_lr_factor']
    if name in ('input_noreg', 'noreg', 'output_noreg'):
        lr *= options['noreg_lr_factor']
    if muon and name not in ('normal','normal_attn','normal_gab','gab_mlp','tab_module'):
        lr *= options.get('muon_adam_lr_factor', 1.0)
    if norm_kind in ('fixup', 'fixscale'):
        factor = (options['input_wd_factor'] if name == 'input' else options['normal_wd_factor'] if name == 'normal'
                  else options['normal_attn_wd_factor'] if name == 'normal_attn' else 1.0)
        if name in ('input','normal','normal_gamma','output','normal_attn', 'normal_gab', 'gab_mlp', 'tab_module'):
            wd = (0.005 if adamw else 1e-6) * batch_scale * factor
            if name in ('normal_attn', 'normal_gab', 'gab_mlp', 'tab_module'):
                wd *= 0.5
        elif name in ('input_noreg','noreg','output_noreg'):
            wd = 1e-8 * batch_scale
        else:
            raise ValueError(f'Unknown fixup optimizer group: {name}')
        return lr, wd
    # MuZero's per-sample masked norm uses the same affine-parameter roles and
    # normalized-trunk LR/WD policy; its forward normalization stays unchanged.
    if norm_kind not in ('fixscaleonenorm', 'bnorm', 'brenorm', 'fixbrenorm', 'masked_layernorm'):
        raise ValueError(f'Unsupported optimizer norm kind: {norm_kind}')
    if name in ('input', 'normal', 'normal_gamma', 'normal_attn', 'normal_gab', 'gab_mlp', 'tab_module'):
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
        if name in ('normal_attn', 'normal_gab', 'gab_mlp', 'tab_module'):
            factor *= {'normal_attn': 0.5, 'normal_gab': 0.3, 'gab_mlp': 0.1, 'tab_module': 0.1}[name]
            if name == 'normal_attn':
                factor *= options['normal_attn_wd_factor']
        regular = 0.020 if muon and name in ('normal', 'normal_attn', 'normal_gab', 'gab_mlp', 'tab_module') else 0.009 if adamw else 0.00125
        wd = regular * batch_scale * (scale * warmup) ** (0.70 if muon else 0.75) * adaptive * factor
    elif name == 'output':
        wd = (0.004 if adamw else 1e-6) * batch_scale
    elif name in ('input_noreg', 'noreg'):
        wd = 1e-6 * batch_scale * (scale * warmup) ** 0.75
    elif name == 'output_noreg':
        wd = (1e-6 if adamw else 1e-8) * batch_scale
    else:
        raise ValueError(f'Unknown optimizer group: {name}')
    return lr, wd


def optimizer_from_groups(groups, kind, options=None):
    if kind in ('muon', 'normuon', 'aurora'):
        from .muon import Muon
        return Muon(groups, {**(options or {}), 'kind': kind})
    if kind == 'adamw':
        return torch.optim.AdamW(groups, fused=groups[0]['params'][0].is_cuda)
    if kind == 'sgd':
        return torch.optim.SGD(groups, momentum=0.9)
    raise ValueError(f'Unsupported optimizer kind: {kind}')


def optimizer_for(model, config):
    if 'plain_optimizer' in config:
        from .plain_optimization import optimizer_for as plain_optimizer
        kind = config['optimizer']['kind']
        return plain_optimizer(model, kind, config['plain_optimizer'],
                               parameter_groups(model) if kind in ('muon','normuon','aurora') else None, config['optimizer'])
    if config['agent']['algorithm'] == 'muzero':
        from .muzero.optimization import optimizer_for as muzero_optimizer
        return muzero_optimizer(model, config)
    groups = parameter_groups(model)
    options = config['optimizer']
    for group in groups:
        group['lr'], group['weight_decay'] = group_settings(
            group['group_name'], options, config['training']['batch_size'], 0, {}, {}, model.norm_kind, config.get('lr_schedule'))
    return optimizer_from_groups(groups, options['kind'], options)


def optimization_for(model, config, optimizer, state=None, round_steps=None):
    if 'plain_optimizer' in config or not config.get('muzero_training', {}).get('katago_optimizer', True):
        from .plain_optimization import PlainOptimization
        return PlainOptimization(model, config, optimizer, state)
    return Optimization(model, config, optimizer, state, round_steps=round_steps)


class Optimization:
    """Own the schedule, slow weights and inference average alongside an optimizer."""
    def __init__(self, model, config, optimizer, state=None, round_steps=None):
        self.model, self.optimizer = model, optimizer
        self.options, self.batch_size = config['optimizer'], config['training']['batch_size']
        self.schedule = config.get('lr_schedule', {})
        self.completed_rounds = 0 if state is None else state['completed_rounds']
        from .muon import configure_floors
        self.initial_rms = configure_floors(model, optimizer, self.options, state)
        self.lookahead_k = self.options['lookahead_k']
        # Target epoch size selects whole files; actual boundaries come from
        # the source file generator and may straddle outer data rounds.
        self.epoch_steps = config['training']['train_steps']
        segments = config['training']['sub_epochs']
        if segments > self.epoch_steps:
            raise ValueError('Optimizer subepochs require at least one batch each')
        self.epoch_end = False
        self.period = self.options['swa_period_samples'] or max(1, config['training']['train_steps'] * self.batch_size // 2)
        self.baselines = model_norms(optimizer.param_groups) if state is None else state['baselines']
        self.norms = {} if state is None else state['norms']
        self.counter = 0 if state is None else state['lookahead_counter']
        self.swa_samples = 0 if state is None else state['swa_samples']
        self.consumed_samples = 0 if state is None else state['consumed_samples']
        self.optimizer_steps = 0 if state is None else state['optimizer_steps']
        self.round_batches = 0 if state is None else state['round_batches']
        self.subepoch=0 if state is None else state['subepoch']
        self.subepoch_batches=0 if state is None else state['subepoch_batches']
        from .brenorm import BoundsSchedule
        self.brenorm = BoundsSchedule(model, config.get('brenorm'), self.consumed_samples,
                                     None if state is None else state['brenorm'])
        self.norm_sums = {} if state is None else state['norm_sums'].copy()
        self.norm_weights = {} if state is None else state['norm_weights'].copy()
        self.pending_norms = None
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

    def configure(self):
        for group in self.optimizer.param_groups:
            group['lr'], group['weight_decay'] = group_settings(
                group['group_name'], self.options, self.batch_size, self.consumed_samples, self.norms, self.baselines,
                self.model.norm_kind, self.schedule, self.completed_rounds)

    @property
    def backward_scale(self):
        return self.batch_size

    @property
    def swa_count(self):
        return int(self.swa.n_averaged.item())

    def begin_round(self):
        """Resume the fixed optimizer epoch across an outer data round."""
        if self.consumed_samples == 0:
            self.configure()
            self.brenorm.update(self.consumed_samples)

    def prepare_batch(self, clock):
        """Apply boundaries from the consumed reader cursor, never prefetch."""
        self.epoch_end = clock['epoch_end']
        if clock['subepoch_batches'] == 1:
            if clock['subepoch'] == 0:
                self.round_batches = 0
                self.subepoch = -1
                self.configure()
                self.brenorm.update(self.consumed_samples)
            self.begin_subepoch()
        if self.completed_rounds != clock['epoch'] or self.subepoch != clock['subepoch']:
            raise ValueError('Optimizer and consumed file subepoch clocks differ')

    def begin_subepoch(self):
        # Source resets the counter, without copying fast weights to slow here.
        # LR/norm/SWA and fast/slow weights are not reset at this boundary.
        self.counter = 0
        self.subepoch+=1
        self.subepoch_batches=0

    def before_step(self):
        # Norm metrics describe the pre-update parameters, including on overflow.
        is_print = (self.round_batches + 1) % self.options['norm_interval'] == 0
        self.pending_norms = (model_norms(self.optimizer.param_groups)
                              if is_print or not self.options['norm_only_at_print'] else None)

    def record_norms(self):
        is_print = self.round_batches % self.options['norm_interval'] == 0
        if self.pending_norms is not None:
            if self.options['norm_only_at_print']:
                self.norm_sums = self.pending_norms.copy()
                self.norm_weights = {name: 1.0 for name in self.pending_norms}
            elif not (self.options['lookahead_print'] and self.options['lookahead_alpha'] < 1 and self.counter != 0):
                for name, norm in self.pending_norms.items():
                    self.norm_sums[name] = self.norm_sums.get(name, 0.0) + norm
                    self.norm_weights[name] = self.norm_weights.get(name, 0.0) + 1.0
            self.norms = {name: value / self.norm_weights[name] for name, value in self.norm_sums.items()}
        # metrics_logging only decays *_sum per batch. Norms are *_batch:
        # their historical sums AND weights shrink at the print point by .001.
        if is_print:
            self.norm_sums = {name: value * 0.001 for name, value in self.norm_sums.items()}
            self.norm_weights = {name: value * 0.001 for name, value in self.norm_weights.items()}
        self.pending_norms = None

    def gradient_cap(self, override):
        # EtaZero logs mean losses. Backward uses the batch sum, like KataGo.
        if override:
            return override * self.batch_size
        cap = (11000 if self.options['kind'] in ('adamw', 'muon', 'normuon', 'aurora') else
               5500 if self.model.norm_kind in ('bnorm', 'brenorm', 'fixbrenorm') else 2500)
        scale = effective_scale(self.options['lr_scale'], self.schedule, self.consumed_samples, self.completed_rounds)
        return cap * math.sqrt(self.batch_size / 256) / math.sqrt(max(1e-7, scale))

    @torch.no_grad()
    def after_step(self, successful=True):
        # GradScaler skip still consumes a batch in KataGo's scheduling clocks.
        self.optimizer_steps += int(successful)
        self.consumed_samples += self.batch_size
        self.round_batches += 1
        self.subepoch_batches += 1
        if self.round_batches % 500 == 0:
            self.brenorm.update(self.consumed_samples)
        self.record_norms()
        if ((self.consumed_samples <= 200000000 and self.round_batches % 5 == 0)
                or self.round_batches % 50 == 0):
            self.configure()
        synced = True
        if self.options['lookahead_alpha'] < 1:
            self.counter += 1
            synced = self.counter == self.lookahead_k
            if synced:
                for name, p in self.model.named_parameters():
                    self.slow[name].add_(p - self.slow[name], alpha=self.options['lookahead_alpha'])
                    p.copy_(self.slow[name])
                self.counter = 0
        self.swa_samples += self.batch_size
        if self.swa_samples >= self.period and synced:
            self.swa.update_parameters(self.model)
            self.swa_samples = 0
        if self.epoch_end:
            self.finish_round()

    @torch.no_grad()
    def finish_round(self):
        self.completed_rounds += 1
        # Match the native epoch boundary: discard leftover fast weights while
        # retaining optimizer moments and separate consumption/update counters.
        if self.options['lookahead_alpha'] < 1:
            for name, p in self.model.named_parameters():
                p.copy_(self.slow[name])
            self.counter = 0

    def state_dict(self):
        return {'baselines': self.baselines, 'norms': self.norms, 'brenorm': self.brenorm.state_dict(),
                'completed_rounds': self.completed_rounds, 'initial_rms': self.initial_rms,
                'norm_sums': self.norm_sums, 'norm_weights': self.norm_weights,
                'consumed_samples': self.consumed_samples, 'optimizer_steps': self.optimizer_steps,
                'round_batches': self.round_batches,'subepoch':self.subepoch,'subepoch_batches':self.subepoch_batches,
                'lookahead_counter': self.counter, 'slow': self.slow,
                'swa_samples': self.swa_samples, 'swa': self.swa.state_dict()}


def inference_weights(checkpoint):
    swa = checkpoint['optimization']['swa']
    if swa is None or swa['n_averaged'].item() == 0:
        return checkpoint['model']
    return {name.removeprefix('module.'): value for name, value in swa.items() if name.startswith('module.')}
