"""KataGo plain, NBT and nested Transformer policy/value networks.

Adapted from KataGo model_pytorch.py (MIT, attribution in THIRD_PARTY.md).
Convolution presets use fson/Mish; the bare Transformer uses fixup/ReLU.
"""
import copy
import math
import torch
from torch import nn
from .config import network_widths
from .schema import CONTRACT_ID, PLANES, GLOBALS, POLICY_HEADS


def init_weights(tensor, scale=1.0, identity=False, fan_tensor=None, activation='mish'):
    """KataGo's variance-corrected, two-sigma truncated normal initialization."""
    if activation not in ('mish','relu'):
        raise ValueError(f'Unsupported initialization activation: {activation}')
    fan_in = nn.init._calculate_fan_in_and_fan_out(tensor if fan_tensor is None else fan_tensor)[0]
    gain = 1.0 if identity else math.sqrt(2.0 if activation == 'relu' else 2.210277)
    std = scale * gain / math.sqrt(fan_in) / 0.87962566103423978
    with torch.no_grad():
        if std < 1e-10:
            tensor.zero_()
        else:
            nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)


class FixedScaleMask(nn.Module):
    """Pre-activation fixed scale; weight is a zero-centered gamma offset."""
    def __init__(self, channels, scale=1.0):
        super().__init__()
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        result = x.float() * ((self.weight + 1.0) * self.scale).view(1, -1, 1, 1)
        # Keep multiply/add separate under the TorchScript inference fuser.
        result.add_(self.bias.view(1, -1, 1, 1))
        return result * mask.float()


class BiasMask(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (x.float() + self.bias.view(1, -1, 1, 1)) * mask.float()


class MaskedBatchNorm(nn.Module):
    """KataGo fson final norm: EMA of masked population mean and std, not variance."""
    def __init__(self, channels, scale=1.0):
        super().__init__()
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.zeros(channels))  # gamma = 1 + weight
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_std", torch.ones(channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xf, mf = x.float(), mask.float()
        if self.training:
            count = mf.sum().clamp_min(1.0)
            mean = (xf * mf).sum((0, 2, 3)) / count
            centered = xf - mean.view(1, -1, 1, 1)
            var = (centered.square() * mf).sum((0, 2, 3)) / count
            std = torch.sqrt(var + 1e-4)
            with torch.no_grad():
                self.running_mean.add_((mean.detach() - self.running_mean) * 0.001)
                self.running_std.add_((std.detach() - self.running_std) * 0.001)
        else:
            std = self.running_std
            centered = xf - self.running_mean.view(1, -1, 1, 1)
        result = centered * torch.reciprocal(std).view(1, -1, 1, 1)
        result = result * ((self.weight + 1.0) * self.scale).view(1, -1, 1, 1)
        result.add_(self.bias.view(1, -1, 1, 1))
        return result * mf


class GlobalPool(nn.Module):
    """Mean, board-size-scaled mean, maximum; input [N,C,H,W], output [N,3C]."""
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        area = mask.float().sum((2, 3)).clamp_min(1.0)
        mean = (x.float() * mask).sum((2, 3)) / area
        # Match KataGo's max gradient: a tie routes to the first maximum.
        maximum = x.float().masked_fill(mask <= 0, float('-inf')).flatten(2).max(2)[0]
        return torch.cat((mean, mean * ((area.sqrt() - 14.0) / 10.0), maximum), 1)


class ValuePool(nn.Module):
    """Three size-conditioned means, without max pooling (KataGo ValueHead)."""
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        area = mask.float().sum((2, 3)).clamp_min(1.0)
        mean = (x.float() * mask).sum((2, 3)) / area
        offset = area.sqrt() - 14.0
        return torch.cat((mean, mean * (offset / 10.0), mean * (offset.square() / 100.0 - 0.1)), 1)


class ConvAndGlobalPool(nn.Module):
    def __init__(self, c_in, c_out, c_gpool):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.conv_global = nn.Conv2d(c_in, c_gpool, 3, padding=1, bias=False)
        self.norm_global = FixedScaleMask(c_gpool)
        self.act = nn.Mish()
        self.pool = GlobalPool()
        self.linear = nn.Linear(3 * c_gpool, c_out, bias=False)
        init_weights(self.conv.weight, 0.8)
        init_weights(self.conv_global.weight, math.sqrt(0.6))
        init_weights(self.linear.weight, math.sqrt(0.6))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        local = self.conv(x)
        global_features = self.act(self.norm_global(self.conv_global(x), mask))
        context = self.linear(self.pool(global_features, mask))
        return local + context.unsqueeze(-1).unsqueeze(-1)


class NormActConv(nn.Module):
    __constants__ = ['has_global_pool']
    def __init__(self, c_in, c_out, kernel, scale=1.0, c_gpool=0):
        super().__init__()
        self.norm = FixedScaleMask(c_in, scale)
        self.act = nn.Mish()
        if c_gpool:
            self.conv = ConvAndGlobalPool(c_in, c_out, c_gpool)
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2, bias=False)
            init_weights(self.conv.weight)
        self.has_global_pool = bool(c_gpool)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm(x, mask))
        if self.has_global_pool:
            return self.conv(y, mask)
        return self.conv(y)


class InnerResidualBlock(nn.Module):
    def __init__(self, channels, index, c_gpool=0):
        super().__init__()
        local_channels = channels - c_gpool
        self.pre = NormActConv(channels, local_channels, 3, 1.0 / math.sqrt(index + 1.0), c_gpool)
        self.post = NormActConv(local_channels, channels, 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x + self.post(self.pre(x, mask), mask)


class NestedBottleneckBlock(nn.Module):
    def __init__(self, channels, mid, index, c_gpool=0):
        super().__init__()
        self.pre = NormActConv(channels, mid, 1, 1.0 / math.sqrt(index + 1.0))
        self.inner = nn.ModuleList([InnerResidualBlock(mid, i, c_gpool if i == 0 else 0) for i in range(2)])
        self.post = NormActConv(mid, channels, 1, 1.0 / math.sqrt(3.0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.pre(x, mask)
        for block in self.inner:
            y = block(y, mask)
        return x + self.post(y, mask)


class PolicyHead(nn.Module):
    def __init__(self, channels, width, activation='mish', predict_q_values=False):
        super().__init__()
        self.local = nn.Conv2d(channels, width, 1, bias=False)
        self.global_conv = nn.Conv2d(channels, width, 1, bias=False)
        self.global_bias = BiasMask(width)
        self.pool = GlobalPool()
        self.linear = nn.Linear(3 * width, width, bias=False)
        self.bias = BiasMask(width)
        self.act = nn.ReLU() if activation == 'relu' else nn.Mish()
        self.out = nn.Conv2d(width, len(POLICY_HEADS)+int(predict_q_values), 1, bias=False)
        init_weights(self.local.weight, 0.8, activation=activation)
        init_weights(self.global_conv.weight, activation=activation)
        init_weights(self.linear.weight, 0.6, activation=activation)
        init_weights(self.out.weight, 0.3, identity=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        context = self.act(self.global_bias(self.global_conv(x), mask))
        context = self.linear(self.pool(context, mask)).unsqueeze(-1).unsqueeze(-1)
        # Global context must enter before the nonlinear activation; a constant
        # added to final spatial logits cancels in the policy softmax.
        local = self.act(self.bias(self.local(x) + context, mask))
        return self.out(local).flatten(2)


class ValueHead(nn.Module):
    def __init__(self, channels, width, hidden, activation='mish'):
        super().__init__()
        self.conv = nn.Conv2d(channels, width, 1, bias=False)
        self.bias = BiasMask(width)
        self.pool = ValuePool()
        self.hidden = nn.Linear(3 * width, hidden)
        self.out = nn.Linear(hidden, 13)
        self.act = nn.ReLU() if activation == 'relu' else nn.Mish()
        init_weights(self.conv.weight, activation=activation)
        init_weights(self.hidden.weight, activation=activation)
        init_weights(self.hidden.bias, 0.2, fan_tensor=self.hidden.weight, activation=activation)
        init_weights(self.out.weight, identity=True)
        init_weights(self.out.bias, 0.2, identity=True, fan_tensor=self.out.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.act(self.bias(self.conv(x), mask))
        return self.out(self.act(self.hidden(self.pool(features, mask))))


class PolicyValueNet(nn.Module):
    def __init__(self, canvas, channels, blocks, architecture, predict_q_values=False):
        super().__init__()
        widths = network_widths(channels, architecture)
        self.architecture = architecture
        if predict_q_values and architecture!='transformer':
            raise ValueError('predict_q_values requires the v17 Transformer preset')
        self.predict_q_values = bool(predict_q_values)
        self.fp32_heads = True
        if architecture == 'plain' and blocks != 10:
            raise ValueError('plain requires b10c128-fson-mish: channels=128, blocks=10')
        if architecture == 'transformer' and blocks != 5:
            raise ValueError('transformer requires b5c192h3nbttfrs: channels=192, blocks=5')
        self.norm_kind = 'fixup' if architecture == 'transformer' else 'fixscaleonenorm'
        self.model_version = 17 if architecture == 'transformer' else 15
        activation = 'relu' if architecture == 'transformer' else 'mish'
        self.canvas = canvas
        self.contract = CONTRACT_ID
        self.stem = nn.Conv2d(len(PLANES), channels, 3, padding=1, bias=False)
        self.linear_global = nn.Linear(len(GLOBALS), channels, bias=False)
        init_weights(self.stem.weight, 0.8, activation=activation)
        init_weights(self.linear_global.weight, 0.6, activation=activation)
        if architecture == 'plain':
            # Source regular blocks are the same two-convolution residual unit
            # used inside NBT, with gpool only in source blocks 5 and 8.
            self.blocks = nn.ModuleList([
                InnerResidualBlock(channels, i, widths['gpool'] if i in (4, 7) else 0)
                for i in range(blocks)
            ])
        elif architecture == 'transformer':
            from .transformer import NestedBottleneckTransformerBlock
            self.blocks = nn.ModuleList([NestedBottleneckTransformerBlock(canvas) for _ in range(blocks)])
        else:
            self.blocks = nn.ModuleList([
                NestedBottleneckBlock(channels, widths['mid'], i, widths['gpool'] if i % 2 == 1 else 0)
                for i in range(blocks)
            ])
        self.trunk_norm = (BiasMask(channels) if architecture == 'transformer' else
                           MaskedBatchNorm(channels, 1.0 / math.sqrt(blocks + 1.0)))
        self.act = nn.ReLU() if activation == 'relu' else nn.Mish()
        self.policy_head = PolicyHead(channels, widths['policy'], activation, self.predict_q_values)
        self.value_head = ValueHead(channels, widths['value'], widths['value_hidden'], activation)

    @torch.jit.export
    def metadata(self) -> tuple[int, str]:
        return self.canvas, self.contract

    @torch.jit.export
    def forward_all(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Six policies, WDL, three TD WDLs and raw error; current-player W/D/L."""
        mask = obs[:, 0:1]
        x = self.stem(obs * mask) + self.linear_global(globals).unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.act(self.trunk_norm(x, mask))
        # Learner heads stay FP32 in both training and validation. Only the
        # independent export copy follows the configured inference autocast.
        if self.fp32_heads:
            with torch.autocast("cuda", enabled=False):
                policy, value = self.policy_head(x.float(), mask.float()), self.value_head(x.float(), mask.float())
        else:
            policy, value = self.policy_head(x, mask), self.value_head(x, mask)
        return policy, value[:, :3], value[:, 3:12].reshape(-1, 3, 3), value[:, 12]

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy, value, _, _ = self.forward_all(obs, globals)
        return policy, value


def make_network(config):
    return PolicyValueNet(**config["network"])


def base_losses(logits, value, obs, policy, opponent_policy, opponent_policy_weight, target, soft_policy_weight_scale):
    """Float targets on the logits' device: policy/opponent [N,A], weight [N], WDL [N,3].

    Training logits are [N,4,A], A=canvas². Reduce every weighted component
    over the full batch, including zero-weight terminal opponent rows.
    """
    # Occupied points remain in the training softmax; outside-canvas padding does not.
    mask = obs[:, 0].flatten(1)
    logp = torch.log_softmax(logits.float().masked_fill(~mask[:, None].bool(), -1e9), dim=2)
    # KataGo metrics_pytorch.py: soften over all on-board cells, including occupied ones.
    policies = torch.stack((policy, opponent_policy), dim=1)
    policies = policies / policies.sum(2, keepdim=True)
    soft = ((policies + 1e-7) * mask[:, None]).pow(0.25)
    soft = soft / soft.sum(2, keepdim=True)
    targets = torch.cat((policies, soft), dim=1)
    ce = -(targets * logp[:, :4]).sum(2)
    policy_loss = 0.930 * ce[:, 0].mean()
    opponent_policy_loss = (0.15 * opponent_policy_weight * ce[:, 1]).mean()
    soft_policy_loss = soft_policy_weight_scale * ce[:, 2].mean()
    soft_opponent_policy_loss = (0.15 * soft_policy_weight_scale * opponent_policy_weight * ce[:, 3]).mean()
    # v15 ordinary policy scale; value CE's internal 1.20 times CLI default 0.6.
    value_loss = 1.20 * 0.6 * -(target * torch.log_softmax(value.float(), dim=1)).sum(1).mean()
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


def auxiliary_losses(logits, td_logits, raw_error, obs, policy, target, td_target,
                     full_game_weight, disable_optimistic_policy):
    """Row multiplicity is already resolved; complete-game gate is independent of TD."""
    td_logp = torch.log_softmax(td_logits.float(), dim=2)
    entropy = -(td_target * torch.log(td_target + 1e-30)).sum(2)
    td = 0.72 * (-(td_target * td_logp).sum(2) - entropy).mean(0)
    variance = error_variance(raw_error)
    predicted = torch.softmax(td_logits[:, 2].float(), dim=1)
    predicted = (predicted[:, 0] - predicted[:, 2]).detach()
    actual = td_target[:, 2, 0] - td_target[:, 2, 2]
    error_target = (predicted - actual).square() + 1e-8
    difference = (variance - error_target).abs()
    # KataGo's Huber is unnormalized, unlike torch smooth_l1_loss.
    huber = torch.where(difference <= 0.4, 0.5 * difference.square(), 0.4 * (difference - 0.2))
    error_loss = (2.0 * full_game_weight * huber).mean()
    if disable_optimistic_policy:
        long_weight = torch.full_like(full_game_weight, 0.5)
        short_weight = long_weight
    else:
        long_weight = (target[:, 0] + 0.5 * target[:, 1]).square() * full_game_weight
        short_weight = torch.sigmoid(3.0 * ((actual - predicted) / torch.sqrt(variance.detach() + 0.0001) - 1.5)) * full_game_weight
    mask = obs[:, 0].flatten(1).bool()
    logp = torch.log_softmax(logits[:, 4:6].float().masked_fill(~mask[:, None], -1e9), dim=2)
    normalized = policy / policy.sum(1, keepdim=True)
    ce = -(normalized[:, None] * logp).sum(2)
    long_loss = (0.100 * long_weight * ce[:, 0]).mean()
    short_loss = (0.200 * short_weight * ce[:, 1]).mean()
    return td[0], td[1], td[2], long_loss, short_loss, error_loss


def q_winloss_loss(prediction, target, visits):
    """Source v17 per-action pure W-L Q supervision, mean over output rows.

    Prediction is pre-tanh; targets are writer int16/32000, visits are capped
    child NODE visits. Side rows participate even when full_game_weight=0.
    """
    mask=(visits!=0).float()
    weights=torch.sqrt(visits.float())
    ce=torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.float()*mask*2.0,(1.0+target.float())/2.0,reduction='none')
    return (1.5*((ce*weights).sum(1)/(weights.sum(1)+1.0))).mean()


def losses(logits, value, td_logits, raw_error, obs, policy, opponent_policy,
           opponent_policy_weight, target, td_target, full_game_weight,
           soft_policy_weight_scale, disable_optimistic_policy, q_values=None, q_visits=None):
    base = base_losses(logits, value, obs, policy, opponent_policy, opponent_policy_weight, target, soft_policy_weight_scale)
    auxiliary = auxiliary_losses(logits, td_logits, raw_error, obs, policy, target, td_target,
                                 full_game_weight, disable_optimistic_policy)
    if logits.shape[1]==7:
        if q_values is None or q_visits is None:
            raise ValueError('Enabled Q head requires per-action Q value/node-visit targets')
        qloss=q_winloss_loss(logits[:,6],q_values,q_visits)
        return (base[0] + sum(auxiliary)+qloss, *base[1:], *auxiliary, qloss)
    return (base[0] + sum(auxiliary), *base[1:], *auxiliary)


class TrainingForward(nn.Module):
    """Compile the network and all supervision in one graph."""
    def __init__(self, model, soft_policy_weight_scale, disable_optimistic_policy):
        super().__init__()
        self.model = model
        self.soft_policy_weight_scale = soft_policy_weight_scale
        self.disable_optimistic_policy = disable_optimistic_policy

    def forward(self, obs, globals, policy, opponent_policy, opponent_policy_weight, target, td_target, full_game_weight,
                q_values=None, q_visits=None):
        logits, value, td, error = self.model.forward_all(obs, globals)
        with torch.autocast("cuda", enabled=False):
            return losses(logits, value, td, error, obs, policy, opponent_policy, opponent_policy_weight,
                          target, td_target, full_game_weight, self.soft_policy_weight_scale, self.disable_optimistic_policy,q_values,q_visits)


class InferencePolicyValueNet(nn.Module):
    """Only search outputs: ordinary/short optimistic logits, WDL and error stdev."""
    def __init__(self, model):
        super().__init__();self.model=model

    @torch.jit.export
    def metadata(self) -> tuple[int, str]:
        return self.model.metadata()

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, value, _, raw = self.model.forward_all(obs, globals)
        stdev = 0.5 * torch.nn.functional.softplus(raw.float() * 0.5)
        return policy[:, 0], value, policy[:, 5], stdev


class InferenceNormalization(nn.Module):
    def __init__(self, norm):
        super().__init__()
        for name, value in (('mean', norm.running_mean),
                            ('inv_std', torch.reciprocal(norm.running_std)),
                            ('weight', (norm.weight + 1.0) * norm.scale), ('bias', norm.bias)):
            self.register_buffer(name, value.detach().clone().view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        result = (x.float() - self.mean) * self.inv_std
        result = result * self.weight
        result.add_(self.bias)
        return result * mask.float()


def inference_network(model):
    """Cache final BN statistics on the inference device, retaining CUDA operation order."""
    if model.training:
        raise ValueError('Inference normalization conversion requires evaluation mode')
    result = copy.deepcopy(model)
    result.fp32_heads = False
    with torch.no_grad():
        if isinstance(result.trunk_norm, MaskedBatchNorm):
            result.trunk_norm = InferenceNormalization(result.trunk_norm)
    # Keep all six policy filters so CUDA selects the same convolution kernel.
    return InferencePolicyValueNet(result)
