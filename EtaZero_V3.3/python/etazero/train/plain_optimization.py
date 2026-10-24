"""Ordinary optimizer and mean-loss update control shared by AlphaZero/MuZero."""
from etazero.train.optimization import optimizer_from_groups
from etazero.train.training_clock import TrainingClock


def optimizer_for(model, kind, options, groups=None, muon_options=None):
    if groups is None:
        groups = [{'group_name': 'all', 'params': list(model.parameters())}]
    for group in groups:
        group.update(lr=options['learning_rate'], weight_decay=options['weight_decay'])
    return optimizer_from_groups(groups, kind, muon_options)


class PlainOptimization(TrainingClock):
    """Fixed LR/decay and mean-loss gradients, without Lookahead or averaging."""
    backward_scale = 1
    counter = 0
    swa_count = 0

    def __init__(self, model, config, optimizer, state=None):
        super().__init__(model, config, state)
        from etazero.train.muon import configure_floors
        self.initial_rms = configure_floors(model, optimizer, config['optimizer'], state)

    def before_step(self):
        pass

    def gradient_cap(self, override):
        return override or float('inf')

    def after_step(self, successful=True):
        self.consume_batch(successful)
        if self.epoch_end:
            self.finish_round()

    def state_dict(self):
        return {'swa': None, 'initial_rms': self.initial_rms, **self.clock_state()}
