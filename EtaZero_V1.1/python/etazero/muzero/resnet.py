"""MuZero_V2 dense residual trunks; EtaZero supplies observations and heads.

Adapted from MuZero_V2/python/muzero/network.py (reference_sources.json).
Normalization is per sample over channels and valid cells, with no running
statistics. Reductions stay FP32 under autocast, as do EtaZero latent states.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MaskedNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, mask = x.float(), mask.float()
        count = mask.sum(dim=[2, 3], keepdim=True) * x.size(1)
        mean = (x * mask).sum(dim=[1, 2, 3], keepdim=True) / count
        variance = ((x - mean).square() * mask).sum(dim=[1, 2, 3], keepdim=True) / count
        return ((x - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias) * mask


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = MaskedNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = MaskedNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.silu(self.norm1(x, mask))) * mask
        out = self.conv2(F.silu(self.norm2(out, mask))) * mask
        return out + x


class ResidualTrunk(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(blocks)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask
        for block in self.blocks:
            x = block(x, mask)
        return x


class StemNormAct(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = MaskedNorm(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(x, mask))


class IdentityStem(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x
