"""KataGo-style nested bottleneck convolutional policy/value network.

Adapted from KataGo model_pytorch.py (MIT, attribution in THIRD_PARTY.md).
NBT2 + Mish + fixed scaling with one masked BatchNorm at the trunk end.
"""
import copy
import math
import torch
from torch import nn
from .config import network_widths
from .schema import CONTRACT_ID, PLANES, GLOBALS, POLICY_HEADS


def init_weights(tensor, scale=1.0, identity=False, fan_tensor=None):
    """KataGo's variance-corrected, two-sigma truncated normal initialization."""
    fan_in = nn.init._calculate_fan_in_and_fan_out(tensor if fan_tensor is None else fan_tensor)[0]
    gain = 1.0 if identity else math.sqrt(2.210277)  # Mish gain
    std = scale * gain / math.sqrt(fan_in) / 0.87962566103423978
    with torch.no_grad():
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
    def __init__(self, channels, width):
        super().__init__()
        self.local = nn.Conv2d(channels, width, 1, bias=False)
        self.global_conv = nn.Conv2d(channels, width, 1, bias=False)
        self.global_bias = BiasMask(width)
        self.pool = GlobalPool()
        self.linear = nn.Linear(3 * width, width, bias=False)
        self.bias = BiasMask(width)
        self.act = nn.Mish()
        self.out = nn.Conv2d(width, len(POLICY_HEADS), 1, bias=False)
        init_weights(self.local.weight, 0.8)
        init_weights(self.global_conv.weight)
        init_weights(self.linear.weight, 0.6)
        init_weights(self.out.weight, 0.3, identity=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        context = self.act(self.global_bias(self.global_conv(x), mask))
        context = self.linear(self.pool(context, mask)).unsqueeze(-1).unsqueeze(-1)
        # Global context must enter before the nonlinear activation; a constant
        # added to final spatial logits cancels in the policy softmax.
        local = self.act(self.bias(self.local(x) + context, mask))
        return self.out(local).flatten(2)


class ValueHead(nn.Module):
    def __init__(self, channels, width, hidden):
        super().__init__()
        self.conv = nn.Conv2d(channels, width, 1, bias=False)
        self.bias = BiasMask(width)
        self.pool = ValuePool()
        self.hidden = nn.Linear(3 * width, hidden)
        self.out = nn.Linear(hidden, 3)
        self.act = nn.Mish()
        init_weights(self.conv.weight)
        init_weights(self.hidden.weight)
        init_weights(self.hidden.bias, 0.2, fan_tensor=self.hidden.weight)
        init_weights(self.out.weight, identity=True)
        init_weights(self.out.bias, 0.2, identity=True, fan_tensor=self.out.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.act(self.bias(self.conv(x), mask))
        return self.out(self.act(self.hidden(self.pool(features, mask))))


class PolicyValueNet(nn.Module):
    def __init__(self, canvas, channels, blocks):
        super().__init__()
        widths = network_widths(channels)
        self.canvas = canvas
        self.contract = CONTRACT_ID
        self.stem = nn.Conv2d(len(PLANES), channels, 3, padding=1, bias=False)
        self.linear_global = nn.Linear(len(GLOBALS), channels, bias=False)
        init_weights(self.stem.weight, 0.8)
        init_weights(self.linear_global.weight, 0.6)
        self.blocks = nn.ModuleList([
            NestedBottleneckBlock(channels, widths['mid'], i, widths['gpool'] if i % 2 == 1 else 0)
            for i in range(blocks)
        ])
        self.trunk_norm = MaskedBatchNorm(channels, 1.0 / math.sqrt(blocks + 1.0))
        self.act = nn.Mish()
        self.policy_head = PolicyHead(channels, widths['policy'])
        self.value_head = ValueHead(channels, widths['value'], widths['value_hidden'])

    @torch.jit.export
    def metadata(self) -> tuple[int, str]:
        return self.canvas, self.contract

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 inputs [N,5,H,W], [N,4]; policy [N,4,H*W], side-to-move WDL [N,3]."""
        mask = obs[:, 0:1]
        x = self.stem(obs * mask) + self.linear_global(globals).unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.act(self.trunk_norm(x, mask))
        return self.policy_head(x, mask), self.value_head(x, mask)


def make_network(config):
    return PolicyValueNet(**config["network"])


def losses(logits, value, obs, policy, opponent_policy, opponent_policy_weight, target, soft_policy_weight_scale):
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
    ce = -(targets * logp).sum(2)
    policy_loss = ce[:, 0].mean()
    opponent_policy_loss = (0.15 * opponent_policy_weight * ce[:, 1]).mean()
    soft_policy_loss = soft_policy_weight_scale * ce[:, 2].mean()
    soft_opponent_policy_loss = (0.15 * soft_policy_weight_scale * opponent_policy_weight * ce[:, 3]).mean()
    value_loss = -(target * torch.log_softmax(value.float(), dim=1)).sum(1).mean()
    total = policy_loss + opponent_policy_loss + soft_policy_loss + soft_opponent_policy_loss + value_loss
    # Log weighted contributions, so all five components add up to total loss.
    return total, policy_loss, opponent_policy_loss, soft_policy_loss, soft_opponent_policy_loss, value_loss


class TrainingForward(nn.Module):
    """Compile the network and complete loss as one graph, keeping raw state ownership."""
    def __init__(self, model, soft_policy_weight_scale):
        super().__init__();self.model=model;self.soft_policy_weight_scale=soft_policy_weight_scale

    def forward(self, obs, globals, policy, opponent_policy, opponent_policy_weight, target):
        logits,value=self.model(obs,globals)
        return losses(logits,value,obs,policy,opponent_policy,opponent_policy_weight,target,self.soft_policy_weight_scale)


class InferencePolicyValueNet(nn.Module):
    """Expose only the primary policy to the native inference interface."""
    def __init__(self, model):
        super().__init__();self.model=model

    @torch.jit.export
    def metadata(self) -> tuple[int, str]:
        return self.model.metadata()

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy,value=self.model(obs,globals)
        return policy[:,0],value


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
    with torch.no_grad():
        result.trunk_norm = InferenceNormalization(result.trunk_norm)
    # Keep all four policy filters so CUDA selects the same convolution kernel.
    return InferencePolicyValueNet(result)
