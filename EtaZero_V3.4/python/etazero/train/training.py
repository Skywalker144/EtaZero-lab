"""AlphaZero forward and supervised loss graph."""
import torch
from torch import nn
from etazero.train.losses import losses

class TrainingForward(nn.Module):
    """Compile the network and all supervision in one graph."""
    def __init__(self, model, soft_policy_weight_scale, disable_optimistic_policy, options=None, attention_options=None):
        super().__init__()
        self.model = model
        from etazero.games import SPECS
        self.action_layout = SPECS[model.game].action_layout
        self.soft_policy_weight_scale = soft_policy_weight_scale
        self.disable_optimistic_policy = disable_optimistic_policy
        self.options = options or {}
        self.attention_cap = (attention_options or {}).get('cap', -1)
        from etazero.train.attention_penalty import configure
        configure(model, attention_options or {})

    def forward(self, obs, globals, policy, opponent_policy, opponent_policy_weight, target, td_target, value_aux_weight,
                q_values=None, q_visits=None, future_target=None, variance_target=None, training_weight=None,
                policy_weight=1, value_weight=1, td_value_weight=1, future_position_weight=1, metadata=None):
        outputs = self.model.training_outputs(obs, globals, metadata)
        with torch.autocast("cuda", enabled=False):
            components = [losses(*head[:4], obs, policy, opponent_policy, opponent_policy_weight,
                                 target, td_target, value_aux_weight, self.soft_policy_weight_scale,
                                 self.disable_optimistic_policy, q_values, q_visits, row_weight=training_weight,
                                 action_layout=self.action_layout, future_logits=head[4], raw_variance_time=head[5],
                                 future_target=future_target, variance_target=variance_target,
                                 value_scale=self.options.get('value_scale', 0.6),
                                 td_scales=tuple(self.options.get(k, 0.6) for k in ('td_long_scale','td_mid_scale','td_short_scale')),
                                 variance_scale=self.options.get('variance_time_scale', 1.0),
                                 policy_weight=policy_weight, value_weight=value_weight,
                                 td_value_weight=td_value_weight, future_position_weight=future_position_weight) for head in outputs]
            weights = self.model.head_loss_weights
            if len(weights) == 1 and self.options.get('main_head_scale', -1) >= 0:
                weights = (self.options['main_head_scale'],)
            if len(weights) == 2:
                weights = (self.options.get('main_head_scale', -1), self.options.get('intermediate_head_scale', -1))
                weights = self.model.head_loss_weights if weights == (-1, -1) else (1.0 if weights[0] < 0 else weights[0], weights[1])
            result = tuple(sum(weight * value for weight, value in zip(weights, values)) for values in zip(*components))
            if self.attention_cap >= 0:
                from etazero.train.attention_penalty import metrics
                result += metrics(self.model, self.attention_cap)
            return result


