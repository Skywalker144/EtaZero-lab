"""Consumed-file epoch clock shared by every optimizer recipe.

Outer selfplay rounds may pause this clock, including within a file. Only the
consumed reader cursor advances subepochs; prefetch and publication do not.
"""
from etazero.train.brenorm import BoundsSchedule


class TrainingClock:
    fields = ('completed_rounds', 'consumed_samples', 'optimizer_steps',
              'round_batches', 'subepoch', 'subepoch_batches')

    def __init__(self, model, config, state=None):
        self.batch_size = config['training']['batch_size']
        self.epoch_steps = config['training']['train_steps']
        if config['training']['sub_epochs'] > self.epoch_steps:
            raise ValueError('Optimizer subepochs require at least one batch each')
        for name in self.fields:
            setattr(self, name, 0 if state is None else state[name])
        self.epoch_end = False
        self.brenorm = BoundsSchedule(model, config.get('brenorm'), self.consumed_samples,
                                     None if state is None else state['brenorm'])

    def configure(self):
        """Recipes with scheduled LR/WD override this hook."""

    def begin_round(self):
        if self.consumed_samples == 0:
            self.configure()
            self.brenorm.update(self.consumed_samples)

    def prepare_batch(self, cursor):
        self.epoch_end = cursor['epoch_end']
        if cursor['subepoch_batches'] == 1:
            if cursor['subepoch'] == 0:
                self.round_batches = 0
                self.subepoch = -1
                self.configure()
                self.brenorm.update(self.consumed_samples)
            self.begin_subepoch()
        if self.completed_rounds != cursor['epoch'] or self.subepoch != cursor['subepoch']:
            raise ValueError('Optimizer and consumed file subepoch clocks differ')

    def begin_subepoch(self):
        self.subepoch += 1
        self.subepoch_batches = 0

    def consume_batch(self, successful):
        # An AMP-skipped update still consumes data and advances source clocks.
        self.optimizer_steps += int(successful)
        self.consumed_samples += self.batch_size
        self.round_batches += 1
        self.subepoch_batches += 1
        if self.round_batches % 500 == 0:
            self.brenorm.update(self.consumed_samples)

    def finish_round(self):
        """Finish the internal source epoch, independent of outer rounds."""
        self.completed_rounds += 1

    def clock_state(self):
        return {**{name: getattr(self, name) for name in self.fields},
                'brenorm': self.brenorm.state_dict()}
