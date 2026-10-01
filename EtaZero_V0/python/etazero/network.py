"""Masked residual policy/value network, with a scripted native inference contract."""
import copy
import torch
from torch import nn
from .schema import CONTRACT_ID, PLANES, GLOBALS


class MaskedBatchNorm(nn.Module):
    """Batch statistics include only real board cells, never padding."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Accumulate normalization in float32 even under autocast.
        xf, mf = x.float(), mask.float()
        if self.training:
            count = mf.sum().clamp_min(1.0)
            mean = (xf * mf).sum((0, 2, 3)) / count
            centered = xf - mean.view(1, -1, 1, 1)
            var = (centered.square() * mf).sum((0, 2, 3)) / count
            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(mean.detach(), alpha=0.1)
                corrected = var.detach() * count / (count - 1).clamp_min(1.0)
                self.running_var.mul_(0.9).add_(corrected, alpha=0.1)
        else:
            mean, var = self.running_mean, self.running_var
            centered = xf - mean.view(1, -1, 1, 1)
        result = centered * torch.rsqrt(var.view(1, -1, 1, 1) + 1e-5)
        result = result * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return (result * mf).to(x.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = MaskedBatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = MaskedBatchNorm(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x), mask))
        return torch.relu(x + self.bn2(self.conv2(y), mask)) * mask


class PolicyValueNet(nn.Module):
    def __init__(self, canvas, channels, blocks, value_hidden):
        super().__init__()
        self.canvas = canvas
        self.contract = CONTRACT_ID
        self.stem = nn.Conv2d(len(PLANES), channels, 3, padding=1, bias=False)
        self.linear_global = nn.Linear(len(GLOBALS), channels, bias=False)
        self.stem_bn = MaskedBatchNorm(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = MaskedBatchNorm(2)
        self.policy_out = nn.Conv2d(2, 1, 1)
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = MaskedBatchNorm(1)
        self.value_hidden = nn.Linear(1, value_hidden)
        self.value_out = nn.Linear(value_hidden, 3)

    @torch.jit.export
    def metadata(self) -> tuple[int, str]:
        return self.canvas, self.contract

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = obs[:, 0:1]
        stem = self.stem(obs * mask) + self.linear_global(globals).unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.stem_bn(stem, mask))
        for block in self.blocks:
            x = block(x, mask)
        policy = self.policy_out(torch.relu(self.policy_bn(self.policy_conv(x), mask))).flatten(1)
        value = torch.relu(self.value_bn(self.value_conv(x), mask))
        pooled = (value * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1.0)
        value = self.value_out(torch.relu(self.value_hidden(pooled))) # W/D/L logits, side to move
        return policy, value


def make_network(config):
    return PolicyValueNet(**config["network"])


def losses(logits, value, obs, policy, target):
    # Occupied points remain in the training softmax; outside-canvas padding does not.
    mask = obs[:, 0].flatten(1).bool()
    logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=1)
    policy_loss = -(policy * logp).sum(1).mean()
    value_loss = -(target * torch.log_softmax(value.float(), dim=1)).sum(1).mean()
    return policy_loss + value_loss, policy_loss, value_loss


class TrainingForward(nn.Module):
    """Compile the network and complete loss as one graph, keeping raw state ownership."""
    def __init__(self, model):
        super().__init__();self.model=model

    def forward(self, obs, globals, policy, target):
        logits,value=self.model(obs,globals)
        return losses(logits,value,obs,policy,target)


class InferenceNormalization(nn.Module):
    def __init__(self, norm):
        super().__init__()
        for name,value in (('mean',norm.running_mean),
                           ('inv_std',torch.rsqrt(norm.running_var.detach()+1e-5)),
                           ('weight',norm.weight),('bias',norm.bias)):
            self.register_buffer(name,value.detach().clone().view(1,-1,1,1))

    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        centered=x.float()-self.mean
        result=centered*self.inv_std
        # Mutate only this fresh intermediate. Separate in-place operations keep
        # TorchScript's warmed-up fuser from replacing mul + add with an FMA.
        result.mul_(self.weight)
        result.add_(self.bias)
        return (result*mask.float()).to(x.dtype)


def inference_network(model):
    """Cache inverse standard deviations on the actual inference device.

    Preserve normalization operation order: even combining scale/offset changes
    rounding enough to cross subsequent TF32 convolution quantization boundaries.
    CPU and CUDA rsqrt can also round differently, so move the source model to
    the inference device before conversion.
    """
    if model.training:
        raise ValueError('Inference normalization conversion requires evaluation mode')
    result=copy.deepcopy(model)
    norms=[(result,name) for name in ('stem_bn','policy_bn','value_bn')]
    norms += [(block,name) for block in result.blocks for name in ('bn1','bn2')]
    with torch.no_grad():
        for parent,name in norms:
            setattr(parent,name,InferenceNormalization(getattr(parent,name)))
    return result
