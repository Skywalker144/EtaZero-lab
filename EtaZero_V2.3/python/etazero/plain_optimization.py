"""Ordinary optimizer and mean-loss update control shared by AlphaZero/MuZero."""
import torch


def optimizer_for(model, kind, options):
    groups = [{'group_name': 'all', 'params': list(model.parameters())}]
    kwargs = dict(lr=options['learning_rate'], weight_decay=options['weight_decay'])
    if kind == 'adamw':
        return torch.optim.AdamW(groups, **kwargs, fused=next(model.parameters()).is_cuda)
    return torch.optim.SGD(groups, **kwargs, momentum=0.9)


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
