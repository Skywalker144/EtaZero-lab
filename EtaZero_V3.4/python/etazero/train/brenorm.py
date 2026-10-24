"""KataGo batch-renorm bounds, advanced at epoch entry and every 500 batches."""
import math


class BoundsSchedule:
    def __init__(self, model, options, samples=0, state=None):
        self.model, self.options = model, options
        self.rmax, self.dmax, self.last_samples = (1.0, 0.0, samples) if state is None else (
            state['rmax'], state['dmax'], state['last_samples'])
        self.apply()

    def apply(self):
        if hasattr(self.model, 'set_brenorm_bounds'):
            momentum = 0.001 if self.options is None else self.options['avg_momentum']
            self.model.set_brenorm_bounds(momentum, self.rmax, self.dmax)

    def update(self, samples):
        if self.options is not None:
            fraction = -math.expm1(-(samples - self.last_samples) / self.options['adjustment_scale'])
            self.rmax += fraction * (self.options['target_rmax'] - self.rmax)
            self.dmax += fraction * (self.options['target_dmax'] - self.dmax)
        self.last_samples = samples
        self.apply()

    def state_dict(self):
        return dict(rmax=self.rmax, dmax=self.dmax, last_samples=self.last_samples)
