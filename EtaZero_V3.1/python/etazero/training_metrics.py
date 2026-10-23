"""All-batch statistics, independent of journal cadence and resumable with the learner."""
import copy


class TrainingMetrics:
    def __init__(self, state=None):
        self.state = copy.deepcopy(state) if state is not None else dict(
            steps=0, amp_skipped_steps=0, clipped_steps=0, sums={}, counts={}, maxima={},
            step_sums=[], step_loss_updates=0)

    def add(self, losses, grad_norm, skipped, clipped, step_losses=None):
        s = self.state
        s['steps'] += 1
        s['amp_skipped_steps'] += int(skipped)
        s['clipped_steps'] += int(clipped and not skipped)
        measured = dict(losses)
        if not skipped:
            measured['grad_norm'] = grad_norm
        for name, value in measured.items():
            s['sums'][name] = s['sums'].get(name, 0.) + value
            s['counts'][name] = s['counts'].get(name, 0) + 1
        for name in ('loss', 'grad_norm'):
            if name in measured:
                value = measured[name]
                s['maxima'][name] = max(s['maxima'].get(name, value), value)
        if step_losses is not None:
            if not s['step_sums']:
                s['step_sums'] = [0.] * len(step_losses)
            if len(s['step_sums']) != len(step_losses):
                raise ValueError('Unroll length changed within a training round')
            s['step_sums'] = [a+b for a,b in zip(s['step_sums'], step_losses)]
            s['step_loss_updates'] += 1

    def summary(self):
        s = self.state
        if s.get('incomplete'):
            return {'steps': s['steps'], 'metrics_error': 'Incomplete learner metrics'}
        result = {k: s[k] for k in ('steps', 'amp_skipped_steps', 'clipped_steps')}
        result.update({k: v/s['counts'][k] for k,v in s['sums'].items()})
        result.update(gradient_steps=s['counts'].get('grad_norm', 0),
                      grad_norm=result.get('grad_norm', float('nan')),
                      max_grad_norm=s['maxima'].get('grad_norm', float('nan')),
                      max_loss=s['maxima'].get('loss', float('nan')))
        if s['step_loss_updates']:
            result.update(step_loss_updates=s['step_loss_updates'],
                          step_losses=[v/s['step_loss_updates'] for v in s['step_sums']])
        return result
