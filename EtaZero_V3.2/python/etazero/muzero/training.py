"""Unroll supervision with a real root and learned transitions only."""
import torch
from torch import nn

from ..network import losses, row_mean, action_mask
from .network import scale_gradient


# MuZero uses the same root-based coordinate transform for K+1 targets and K actions.
from ..symmetry import augment_batch


class TrainingForward(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        from ..games import spec_for_config
        self.action_layout = spec_for_config(config).action_layout
        self.steps = config['unroll']['steps']
        self.hidden_scale = config['unroll']['hidden_gradient_scale']
        self.soft_scale = config['training']['soft_policy_weight_scale']
        self.disable_optimistic = config['training']['disable_optimistic_policy']
        self.auxiliary_losses = config.get('muzero_training', {}).get('auxiliary_losses', True)
        self.options = config.get('losses', {})

    def components(self, hidden, obs, targets, step, weight):
        logits, value, td, error, future, variance = self.model.prediction.training_outputs(hidden)
        with torch.autocast('cuda', enabled=False):
            if not self.auxiliary_losses:
                # Only main policy and terminal WDL, with unit coefficients.
                # Keep the diagnostic tuple layout; inactive losses are exact
                # zeros disconnected from auxiliary predictions and targets.
                mask = action_mask(obs, logits.shape[-1], self.action_layout).bool()
                logp = torch.log_softmax(logits[:, 0].float().masked_fill(~mask, -1e9), dim=1)
                policy = targets[0][:, step]
                policy = policy / policy.sum(1, keepdim=True)
                pl = -row_mean((policy * logp).sum(1), weight)
                vl = -row_mean((targets[3][:, step] * torch.log_softmax(value.float(), dim=1)).sum(1), weight)
                zero = pl.new_zeros(())
                return (pl + vl, pl, zero, zero, zero, vl, *([zero] * 6))
            components = losses(logits, value, td, error, obs,
                          *(t[:, step] for t in targets[:6]), self.soft_scale,
                          self.disable_optimistic, *(t[:, step] if t is not None else None for t in targets[6:8]),
                          row_weight=weight, action_layout=self.action_layout,
                          future_logits=future, raw_variance_time=variance,
                          future_target=targets[8][:, step], variance_target=targets[9][:, step],
                          value_scale=self.options.get('value_scale', 0.6),
                          td_scales=tuple(self.options.get(k, 0.6) for k in ('td_long_scale','td_mid_scale','td_short_scale')),
                          variance_scale=self.options.get('variance_time_scale', 1.0))
            scale = self.options.get('main_head_scale', -1)
            return tuple(x * scale for x in components) if scale >= 0 else components

    def forward(self, obs, globals, policy, opponent_policy, opponent_policy_weight,
                target, td_target, full_game_weight, q_values, q_visits,
                future_target, variance_target, training_weight, actions, step_weights, recurrent_indices):
        targets = (policy, opponent_policy, opponent_policy_weight, target, td_target,
                   full_game_weight, q_values, q_visits, future_target, variance_target)
        if training_weight is not None:
            step_weights = step_weights * training_weight
        hidden = self.model.representation(obs, globals)
        total = self.components(hidden, obs, targets, 0, step_weights[:, 0])
        # Diagnostics retain the weighted forward loss, before backward-only
        # 1/K scaling. Their sum equals the logged total, with no extra graph.
        step_losses = [total[0].detach()]
        # Side searches have no continuation. Exclude them from recurrent BN,
        # instead of running fictitious transitions with merely zero loss.
        indices = recurrent_indices
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
