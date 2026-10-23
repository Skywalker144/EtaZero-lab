"""Ordinary optimizer and mean-loss update control shared by AlphaZero/MuZero."""
from .optimization import optimizer_from_groups


def optimizer_for(model, kind, options, groups=None, muon_options=None):
    if groups is None:
        groups = [{'group_name': 'all', 'params': list(model.parameters())}]
    for group in groups:
        group.update(lr=options['learning_rate'], weight_decay=options['weight_decay'])
    return optimizer_from_groups(groups, kind, muon_options)


class PlainOptimization:
    """Fixed LR/decay and mean-loss gradients, without Lookahead or averaging."""
    backward_scale = 1
    counter = 0
    swa_count = 0

    def __init__(self, model, config, optimizer, state=None):
        self.batch_size = config['training']['batch_size']
        from .muon import configure_floors
        self.initial_rms = configure_floors(model, optimizer, config['optimizer'], state)
        for key in ('consumed_samples', 'optimizer_steps', 'round_batches', 'subepoch', 'subepoch_batches'):
            setattr(self, key, 0 if state is None else state[key])
        from .brenorm import BoundsSchedule
        self.brenorm = BoundsSchedule(model, config.get('brenorm'), self.consumed_samples,
                                     None if state is None else state['brenorm'])

    def begin_round(self):
        self.round_batches = 0
        self.subepoch = -1
        self.begin_subepoch()
        self.brenorm.update(self.consumed_samples)

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
        if self.round_batches % 500 == 0:
            self.brenorm.update(self.consumed_samples)

    def finish_round(self):
        pass

    def state_dict(self):
        return {'swa': None, 'brenorm': self.brenorm.state_dict(), 'initial_rms': self.initial_rms, **{key: getattr(self, key) for key in
                ('consumed_samples', 'optimizer_steps', 'round_batches', 'subepoch', 'subepoch_batches')}}
