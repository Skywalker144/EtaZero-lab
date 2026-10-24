"""Shared W/D/L/NR, policy and auxiliary training losses."""
import torch
from torch.nn import functional as F

def action_mask(obs: torch.Tensor, actions: int, layout: str = 'points') -> torch.Tensor:
    mask = obs[:, 0].flatten(1)
    if layout == 'moves':
        return mask.new_ones((mask.size(0), actions))
    if layout == 'points_pass':
        return torch.cat((mask, mask.new_ones((mask.size(0),1))),1)
    return mask


def row_mean(value, weight=None):
    return value.mean(0) if weight is None else (value * weight.reshape((-1,) + (1,) * (value.ndim-1))).mean(0)


def base_losses(logits, value, obs, policy, opponent_policy, opponent_policy_weight, target, soft_policy_weight_scale, row_weight=None, action_layout='points', value_scale=0.6, policy_weight=1, value_weight=1):
    """Float targets on the logits' device: policy/opponent [N,A], weight [N], W/D/L/NR [N,4].

    Training logits are [N,heads,A], with the game's fixed action count A. Reduce every weighted component
    over the full batch, including zero-weight terminal opponent rows.
    """
    # Occupied points remain in the training softmax; outside-canvas padding does not.
    mask = action_mask(obs, logits.size(-1), action_layout)
    logp = torch.log_softmax(logits.float().masked_fill(~mask[:, None].bool(), -1e9), dim=2)
    # KataGo metrics_pytorch.py: soften over all on-board cells, including occupied ones.
    policies = torch.stack((policy, opponent_policy), dim=1)
    policies = policies / policies.sum(2, keepdim=True)
    soft = ((policies + 1e-7) * mask[:, None]).pow(0.25)
    soft = soft / soft.sum(2, keepdim=True)
    targets = torch.cat((policies, soft), dim=1)
    ce = -(targets * logp[:, :4]).sum(2)
    policy_loss = 0.930 * row_mean(policy_weight * ce[:, 0], row_weight)
    opponent_policy_loss = row_mean(0.15 * opponent_policy_weight * ce[:, 1], row_weight)
    soft_policy_loss = soft_policy_weight_scale * row_mean(policy_weight * ce[:, 2], row_weight)
    soft_opponent_policy_loss = row_mean(0.15 * soft_policy_weight_scale * opponent_policy_weight * ce[:, 3], row_weight)
    # v15 ordinary policy scale; value CE's internal 1.20 times CLI default 0.6.
    value_loss = 1.20 * value_scale * -row_mean(value_weight * (target * torch.log_softmax(value.float(), dim=1)).sum(1), row_weight)
    total = policy_loss + opponent_policy_loss + soft_policy_loss + soft_opponent_policy_loss + value_loss
    # Log weighted contributions, so all five components add up to total loss.
    return total, policy_loss, opponent_policy_loss, soft_policy_loss, soft_opponent_policy_loss, value_loss


class ErrorSoftplus(torch.autograd.Function):
    """KataGo v15/v17 squared forward with its intentionally surrogate backward."""
    @staticmethod
    def forward(ctx, raw):
        ctx.save_for_backward(raw)
        return torch.nn.functional.softplus(raw * 0.5).square()

    @staticmethod
    def backward(ctx, grad):
        raw, = ctx.saved_tensors
        return grad * (0.05 + 0.95 * torch.sigmoid(raw))


def error_variance(raw):
    return 0.25 * ErrorSoftplus.apply(raw.float())


class VarianceTimeSoftplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw):
        ctx.save_for_backward(raw)
        return F.softplus(raw)

    @staticmethod
    def backward(ctx, grad):
        raw, = ctx.saved_tensors
        return grad * (0.05 + 0.95 * torch.sigmoid(raw))


def continuation_losses(future_logits, raw_variance_time, obs, future_target, variance_target,
                        future_position_weight, value_aux_weight, row_weight=None, variance_scale=1.0):
    mask = obs[:, :1]
    error = (torch.tanh(future_logits.float()) - future_target).square() * mask
    error = error[:, 0].sum((1, 2)) + 0.25 * error[:, 1].sum((1, 2))
    future = row_mean(0.25 * future_position_weight * error / mask.sum((1, 2, 3)).sqrt(), row_weight)
    predicted = 40.0 * VarianceTimeSoftplus.apply(raw_variance_time.float())
    difference = (predicted - (variance_target + 1e-5)).abs()
    huber = torch.where(difference <= 50, 0.5 * difference.square(), 50 * (difference - 25))
    variance = row_mean(0.0003 * variance_scale * value_aux_weight * huber, row_weight)
    return future, variance


def auxiliary_losses(logits, td_logits, raw_error, obs, policy, target, td_target,
                     value_aux_weight, disable_optimistic_policy, row_weight=None, action_layout='points', td_scales=(0.6, 0.6, 0.6), policy_weight=1, td_value_weight=1):
    """Source slot 27 gates error/optimism; policy and TD have separate weights."""
    td_logp = torch.log_softmax(td_logits.float(), dim=2)
    entropy = -(td_target * torch.log(td_target + 1e-30)).sum(2)
    td = 1.20 * row_mean((-(td_target * td_logp).sum(2) - entropy) *
                         (td_value_weight[:,None] if torch.is_tensor(td_value_weight) else td_value_weight), row_weight)
    variance = error_variance(raw_error)
    predicted = torch.softmax(td_logits[:, 2].float(), dim=1)
    predicted = (predicted[:, 0] - predicted[:, 2]).detach()
    actual = td_target[:, 2, 0] - td_target[:, 2, 2]
    error_target = (predicted - actual).square() + 1e-8
    difference = (variance - error_target).abs()
    # KataGo's Huber is unnormalized, unlike torch smooth_l1_loss.
    huber = torch.where(difference <= 0.4, 0.5 * difference.square(), 0.4 * (difference - 0.2))
    error_loss = row_mean(2.0 * value_aux_weight * huber, row_weight)
    if disable_optimistic_policy:
        long_weight = torch.full_like(value_aux_weight, 0.5)
        short_weight = long_weight
    else:
        long_weight = (target[:, 0] + 0.5 * target[:, 1]).square() * value_aux_weight
        short_weight = torch.sigmoid(3.0 * ((actual - predicted) / torch.sqrt(variance.detach() + 0.0001) - 1.5)) * value_aux_weight
    mask = action_mask(obs, logits.size(-1), action_layout).bool()
    logp = torch.log_softmax(logits[:, 4:6].float().masked_fill(~mask[:, None], -1e9), dim=2)
    normalized = policy / policy.sum(1, keepdim=True)
    ce = -(normalized[:, None] * logp).sum(2)
    long_loss = row_mean(0.100 * policy_weight * long_weight * ce[:, 0], row_weight)
    short_loss = row_mean(0.200 * policy_weight * short_weight * ce[:, 1], row_weight)
    return td[0]*td_scales[0], td[1]*td_scales[1], td[2]*td_scales[2], long_loss, short_loss, error_loss


def q_winloss_loss(prediction, target, visits, row_weight=None):
    """Source v17 per-action pure W-L Q supervision, mean over output rows.

    Prediction is pre-tanh; targets are writer int16/32000, visits are capped
    child NODE visits. Side rows participate even when value_aux_weight=0.
    """
    mask=(visits!=0).float()
    weights=torch.sqrt(visits.float())
    ce=torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.float()*mask*2.0,(1.0+target.float())/2.0,reduction='none')
    return row_mean(1.5*((ce*weights).sum(1)/(weights.sum(1)+1.0)), row_weight)


def losses(logits, value, td_logits, raw_error, obs, policy, opponent_policy,
           opponent_policy_weight, target, td_target, value_aux_weight,
           soft_policy_weight_scale, disable_optimistic_policy, q_values=None, q_visits=None, row_weight=None, action_layout='points',
           future_logits=None, raw_variance_time=None, future_target=None, variance_target=None,
           value_scale=0.6, td_scales=(0.6,0.6,0.6), variance_scale=1.0,
           policy_weight=1, value_weight=1, td_value_weight=1, future_position_weight=1):
    base = base_losses(logits, value, obs, policy, opponent_policy, opponent_policy_weight, target, soft_policy_weight_scale, row_weight, action_layout, value_scale, policy_weight, value_weight)
    auxiliary = auxiliary_losses(logits, td_logits, raw_error, obs, policy, target, td_target,
                                 value_aux_weight, disable_optimistic_policy, row_weight, action_layout, td_scales, policy_weight, td_value_weight)
    extra = ()
    if logits.shape[1]==7:
        if q_values is None or q_visits is None:
            raise ValueError('Enabled Q head requires per-action Q value/node-visit targets')
        qloss=q_winloss_loss(logits[:,6],q_values,q_visits,row_weight)
        extra = (qloss,)
    if future_target is not None:
        extra += continuation_losses(future_logits, raw_variance_time, obs, future_target,
                                     variance_target, future_position_weight, value_aux_weight,
                                     row_weight, variance_scale)
    return (base[0] + sum(auxiliary) + sum(extra), *base[1:], *auxiliary, *extra)


