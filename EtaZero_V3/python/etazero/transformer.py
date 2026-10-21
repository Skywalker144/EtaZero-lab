"""The bare KataGo b5c192h3nbttfrs v17 nested Transformer preset (MIT).

NCHW trunk, fixed interleaved 2D RoPE, key-only attention mask, RMSNorm and
SwiGLU. The source's selectable SDPA/NCHW path is used for LibTorch export;
training AMP retains its fused SwiGLU kernel and rounding behavior.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .network import BiasMask, FixedScaleMask
from . import network
from .fused_swiglu import fused_swiglu


def rope_tables(canvas, head_dim=32):
    half = head_dim // 2
    frequencies = 1.0 / (100.0 ** (torch.arange(0, half, 2).float() / half))
    positions = torch.arange(canvas, dtype=torch.float32)
    rows, cols = torch.meshgrid(positions, positions, indexing='ij')
    angles = torch.cat((rows.unsqueeze(-1) * frequencies, cols.unsqueeze(-1) * frequencies), -1)
    angles = angles.flatten(0, 1).repeat_interleave(2, dim=-1)
    return angles.cos(), angles.sin()


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    return torch.stack((-paired[...,1], paired[...,0]), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, canvas):
        super().__init__()
        self.norm = nn.RMSNorm(96, eps=1e-6)
        self.q_proj = nn.Linear(96, 96, bias=False)
        self.k_proj = nn.Linear(96, 96, bias=False)
        self.v_proj = nn.Linear(96, 96, bias=False)
        self.out_proj = nn.Linear(96, 96, bias=False)
        for layer in (self.q_proj,self.k_proj,self.v_proj,self.out_proj):
            layer.parameter_group = 'normal_attn'
        cos, sin = rope_tables(canvas)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        # Source initialize() is a no-op: preserve nn.Linear defaults.

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        seq = height * width
        tokens = x.flatten(2).transpose(1,2)
        tokens = self.norm(tokens)
        qkv = F.linear(tokens, torch.cat((self.q_proj.weight,self.k_proj.weight,self.v_proj.weight),0))
        q,k,v = qkv.split(96,dim=-1)
        q,k,v = q.reshape(batch,seq,3,32),k.reshape(batch,seq,3,32),v.reshape(batch,seq,3,32)
        cos,sin = self.cos.view(1,seq,1,32),self.sin.view(1,seq,1,32)
        # Fixed source RoPE rotates in FP32, then returns to the projection dtype.
        q_rotated, k_rotated = q*cos, k*cos
        q_rotated.add_(rotate_pairs(q)*sin)
        k_rotated.add_(rotate_pairs(k)*sin)
        q,k = q_rotated.to(q.dtype),k_rotated.to(k.dtype)
        # Keep the SDPA boundary explicitly contiguous in BHSD order.
        # Learner compilation also preserves the selected NCHW trunk layout.
        q,k,v = q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous()
        bias = torch.zeros_like(mask.reshape(batch,1,1,seq),dtype=q.dtype)
        bias = bias.masked_fill(mask.reshape(batch,1,1,seq)==0,float('-inf'))
        attended = F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.0,scale=1/math.sqrt(32.0))
        attended = attended.transpose(1,2).contiguous().reshape(batch,seq,96)
        return self.out_proj(attended).transpose(1,2).reshape(batch,channels,height,width)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(96,eps=1e-6)
        self.input = nn.Linear(96,256,bias=False)
        self.gate = nn.Linear(96,256,bias=False)
        self.output = nn.Linear(256,96,bias=False)
        # Source initialize() is a no-op here as well.

    def projected(self, tokens: torch.Tensor) -> torch.Tensor:
        both = F.linear(tokens,torch.cat((self.input.weight,self.gate.weight),0))
        value, gate = both.split(256,dim=-1)
        hidden = F.silu(value)
        # Retain the eager SiLU rounding before gating under the JIT fuser.
        hidden.mul_(gate)
        return hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = self.norm(x.flatten(2).transpose(1,2))
        if torch.jit.is_scripting():
            hidden = self.projected(tokens)
        else:
            if tokens.is_cuda and torch.is_autocast_enabled('cuda'):
                dtype = torch.get_autocast_dtype('cuda')
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    hidden = fused_swiglu(tokens.to(dtype),self.input.weight.to(dtype),self.gate.weight.to(dtype))
                else:
                    hidden = self.projected(tokens)
            else:
                hidden = self.projected(tokens)
        return self.output(hidden).transpose(1,2).reshape(batch,channels,height,width)


class FixupProjection(nn.Module):
    def __init__(self, c_in, c_out, gamma, scale):
        super().__init__()
        self.has_global_pool = False
        self.norm = FixedScaleMask(c_in) if gamma else BiasMask(c_in)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(c_in,c_out,1,bias=False)
        network.init_weights(self.conv.weight,scale,activation='relu')

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.conv(self.act(self.norm(x,mask)))


class NestedBottleneckTransformerBlock(nn.Module):
    def __init__(self, canvas):
        super().__init__()
        self.pre = FixupProjection(192,96,False,math.pow(1/math.sqrt(5.0),1/3))
        self.inner = nn.ModuleList([Attention(canvas),SwiGLU(),Attention(canvas),SwiGLU()])
        self.post = FixupProjection(96,192,True,0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.pre(x,mask)
        for block in self.inner:
            y = y + block(y,mask)
        return x + self.post(y,mask)
