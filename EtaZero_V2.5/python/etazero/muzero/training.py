"""Unroll supervision with a real root and learned transitions only."""
import torch
from torch import nn

from ..network import losses, row_mean
from ..symmetry import apply_symmetry
from .network import scale_gradient


# MuZero uses the same root-based coordinate transform for K+1 targets and K actions.
from ..symmetry import augment_batch


class TrainingForward(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.steps = config['unroll']['steps']
        self.hidden_scale = config['unroll']['hidden_gradient_scale']
        self.soft_scale = config['training']['soft_policy_weight_scale']
        self.disable_optimistic = config['training']['disable_optimistic_policy']
        self.auxiliary_losses = config.get('muzero_training', {}).get('auxiliary_losses', True)

    def components(self, hidden, obs, targets, step, weight):
        logits, value, td, error = self.model.prediction(hidden)
        with torch.autocast('cuda', enabled=False):
            if not self.auxiliary_losses:
                # Only main policy and terminal WDL, with unit coefficients.
                # Keep the diagnostic tuple layout; inactive losses are exact
                # zeros disconnected from auxiliary predictions and targets.
                mask = obs[:, 0].flatten(1).bool()
                logp = torch.log_softmax(logits[:, 0].float().masked_fill(~mask, -1e9), dim=1)
                policy = targets[0][:, step]
                policy = policy / policy.sum(1, keepdim=True)
                pl = -row_mean((policy * logp).sum(1), weight)
                vl = -row_mean((targets[3][:, step] * torch.log_softmax(value.float(), dim=1)).sum(1), weight)
                zero = pl.new_zeros(())
                return (pl + vl, pl, zero, zero, zero, vl, *([zero] * 6))
            return losses(logits, value, td, error, obs,
                          *(t[:, step] for t in targets[:6]), self.soft_scale,
                          self.disable_optimistic, *(t[:, step] if t is not None else None for t in targets[6:]), row_weight=weight)

    def forward(self, obs, globals, policy, opponent_policy, opponent_policy_weight,
                target, td_target, full_game_weight, q_values, q_visits,
                actions, step_weights, sequence_mask):
        targets = (policy, opponent_policy, opponent_policy_weight, target, td_target,
                   full_game_weight, q_values, q_visits)
        hidden = self.model.representation(obs, globals)
        total = self.components(hidden, obs, targets, 0, step_weights[:, 0])
        # Diagnostics retain the weighted forward loss, before backward-only
        # 1/K scaling. Their sum equals the logged total, with no extra graph.
        step_losses = [total[0].detach()]
        # Side searches have no continuation. Exclude them from recurrent BN,
        # instead of running fictitious transitions with merely zero loss.
        indices = torch.nonzero(sequence_mask[:, 1], as_tuple=True)[0]
        hidden = hidden[indices]
        if hidden.shape[0] == 0:
            return total, torch.stack(step_losses + [step_losses[0].new_zeros(())] * self.steps)
        obs = obs[indices]; actions = actions[indices].long()
        targets = tuple(t[indices] if t is not None else None for t in targets)
        weights = step_weights[indices] * (hidden.shape[0] / step_weights.shape[0])
        for step in range(self.steps):
            # MuZero_V2 leaves the first representation->dynamics gradient intact.
            if step > 0:
                hidden = scale_gradient(hidden, self.hidden_scale)
            hidden = self.model.dynamics(hidden, actions[:, step])
            current = self.components(hidden, obs, targets, step+1, weights[:, step+1])
            step_losses.append(current[0].detach())
            total = tuple(a + scale_gradient(b, 1.0/self.steps) for a, b in zip(total, current))
        return total, torch.stack(step_losses)
