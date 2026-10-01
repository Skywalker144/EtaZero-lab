"""Masked residual policy/value network, with a scripted native inference contract."""
import torch
from torch import nn
from .schema import CONTRACT_ID


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
        self.stem = nn.Conv2d(6, channels, 3, padding=1, bias=False)
        self.stem_bn = MaskedBatchNorm(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = MaskedBatchNorm(2)
        self.policy_out = nn.Conv2d(2, 1, 1)
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = MaskedBatchNorm(1)
        self.value_hidden = nn.Linear(1, value_hidden)
        self.value_out = nn.Linear(value_hidden, 1)

    @torch.jit.export
    def metadata(self) -> tuple[int, str]:
        return self.canvas, self.contract

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = obs[:, 3:4]
        x = torch.relu(self.stem_bn(self.stem(obs * mask), mask))
        for block in self.blocks:
            x = block(x, mask)
        policy = self.policy_out(torch.relu(self.policy_bn(self.policy_conv(x), mask))).flatten(1)
        value = torch.relu(self.value_bn(self.value_conv(x), mask))
        pooled = (value * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1.0)
        value = torch.tanh(self.value_out(torch.relu(self.value_hidden(pooled)))).flatten()
        return policy, value


def make_network(config):
    return PolicyValueNet(**config["network"])


def losses(logits, value, obs, policy, target, model, l2):
    # Occupied points remain in the training softmax; outside-canvas padding does not.
    mask = obs[:, 3].flatten(1).bool()
    logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=1)
    policy_loss = -(policy * logp).sum(1).mean()
    value_loss = (value.float() - target).square().mean()
    regularization = sum(p.float().square().sum() for p in model.parameters()) * l2
    return policy_loss + value_loss + regularization, policy_loss, value_loss, regularization
