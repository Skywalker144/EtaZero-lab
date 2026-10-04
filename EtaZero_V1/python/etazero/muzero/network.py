"""NBT or dense ResNet MuZero with EtaZero observations and prediction heads.

The final latent channel is the immutable board mask, not an occupancy mask.
Dynamics consumes only latent and a canvas-coordinate action. All predictions
describe the player to move at that latent step; search must flip W/L on edges.
No reward, terminal predictor, real-board transition or future observation is used.
Masked min/max normalization follows the pinned MuZero_V2 implementation.
"""
import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..config import network_widths
from ..network import (NestedBottleneckBlock, MaskedBatchNorm, PolicyHead,
                       ValueHead, InferenceNormalization, init_weights)
from ..schema import CONTRACT_ID, PLANES, GLOBALS
from .resnet import IdentityStem, ResidualTrunk, StemNormAct


@dataclass(frozen=True)
class NetworkConfig:
    """Explicit component dimensions; these are not training-run defaults."""
    canvas: int
    latent_channels: int
    representation_channels: int
    representation_blocks: int
    dynamics_channels: int
    dynamics_blocks: int
    prediction_channels: int
    prediction_blocks: int
    predict_q_values: bool = False
    architecture: str = 'nbt'

    def __post_init__(self):
        for key, value in vars(self).items():
            if key == 'architecture':
                if value not in ('nbt', 'resnet'):
                    raise ValueError('MuZero architecture must be nbt or resnet')
            elif key == 'predict_q_values':
                if type(value) is not bool:
                    raise ValueError('predict_q_values must be boolean')
            elif type(value) is not int:
                raise ValueError(f'{key} must be an integer')
        if not 5 <= self.canvas <= 25 or self.latent_channels < 1:
            raise ValueError('MuZero requires canvas in [5,25] and positive latent_channels')
        for part in ('representation', 'dynamics', 'prediction'):
            network_widths(getattr(self, part + '_channels'), 'nbt')
            if getattr(self, part + '_blocks') < 0:
                raise ValueError(f'{part}_blocks must be nonnegative')


def normalize_hidden_state(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """FP32 per-sample min/max over channels and valid cells; constant -> zero."""
    x = hidden.float()
    minimum = x.masked_fill(mask == 0, float('inf')).amin([1, 2, 3], keepdim=True)
    maximum = x.masked_fill(mask == 0, float('-inf')).amax([1, 2, 3], keepdim=True)
    span = maximum - minimum
    denominator = torch.where(span > 0, span, torch.ones_like(span))
    return ((x - minimum) / denominator).masked_fill(mask == 0, 0)


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity forward and scaled backward, as in MuZero_V2 unroll training."""
    return tensor * scale + tensor.detach() * (1.0 - scale)


class NBTTrunk(nn.Module):
    def __init__(self, channels: int, blocks: int):
        super().__init__()
        widths = network_widths(channels, 'nbt')
        self.blocks = nn.ModuleList([
            NestedBottleneckBlock(channels, widths['mid'], i,
                                  widths['gpool'] if i % 2 else 0)
            for i in range(blocks)
        ])
        self.norm = MaskedBatchNorm(channels, 1.0 / math.sqrt(blocks + 1.0))
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return self.act(self.norm(x, mask))


def projection(inputs: int, outputs: int, architecture: str = 'nbt') -> nn.Module:
    if inputs == outputs:
        return nn.Identity()
    layer = nn.Conv2d(inputs, outputs, 1, bias=False)
    if architecture == 'nbt':
        init_weights(layer.weight)
    return layer


class Representation(nn.Module):
    def __init__(self, c: NetworkConfig):
        super().__init__()
        self.stem = nn.Conv2d(len(PLANES), c.representation_channels, 3, padding=1, bias=False)
        self.linear_global = nn.Linear(len(GLOBALS), c.representation_channels, bias=False)
        if c.architecture == 'nbt':
            init_weights(self.stem.weight, 0.8)
            init_weights(self.linear_global.weight, 0.6)
        self.stem_act = StemNormAct(c.representation_channels) if c.architecture == 'resnet' else IdentityStem()
        trunk = ResidualTrunk if c.architecture == 'resnet' else NBTTrunk
        self.trunk = trunk(c.representation_channels, c.representation_blocks)
        self.hidden_projection = projection(c.representation_channels, c.latent_channels, c.architecture)

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        mask = obs[:, :1].float()
        x = self.stem(obs * mask) + self.linear_global(globals).unsqueeze(-1).unsqueeze(-1)
        x = self.stem_act(x, mask)
        x = self.hidden_projection(self.trunk(x, mask))
        return torch.cat((normalize_hidden_state(x, mask), mask), dim=1)


class Dynamics(nn.Module):
    def __init__(self, c: NetworkConfig):
        super().__init__()
        self.canvas = c.canvas
        self.stem = nn.Conv2d(c.latent_channels + 1, c.dynamics_channels, 3, padding=1, bias=False)
        if c.architecture == 'nbt':
            init_weights(self.stem.weight, 0.8)
        self.stem_act = StemNormAct(c.dynamics_channels) if c.architecture == 'resnet' else IdentityStem()
        trunk = ResidualTrunk if c.architecture == 'resnet' else NBTTrunk
        self.trunk = trunk(c.dynamics_channels, c.dynamics_blocks)
        self.hidden_projection = projection(c.dynamics_channels, c.latent_channels, c.architecture)

    def forward(self, hidden: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mask = hidden[:, -1:]
        action_plane = F.one_hot(actions, self.canvas * self.canvas).to(hidden.dtype)
        action_plane = action_plane.reshape(-1, 1, self.canvas, self.canvas)
        x = self.stem(torch.cat((hidden[:, :-1] * mask, action_plane * mask), dim=1))
        x = self.stem_act(x, mask)
        x = self.hidden_projection(self.trunk(x, mask))
        return torch.cat((normalize_hidden_state(x, mask), mask), dim=1)


class Prediction(nn.Module):
    def __init__(self, c: NetworkConfig):
        super().__init__()
        widths = network_widths(c.prediction_channels, 'nbt')
        self.projection = projection(c.latent_channels, c.prediction_channels, c.architecture)
        trunk = ResidualTrunk if c.architecture == 'resnet' else NBTTrunk
        self.trunk = trunk(c.prediction_channels, c.prediction_blocks)
        self.policy_head = PolicyHead(c.prediction_channels, widths['policy'], predict_q_values=c.predict_q_values)
        self.value_head = ValueHead(c.prediction_channels, widths['value'], widths['value_hidden'])
        self.fp32_heads = True

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = hidden[:, -1:]
        x = self.trunk(self.projection(hidden[:, :-1] * mask), mask)
        if self.fp32_heads:
            with torch.autocast('cuda', enabled=False):
                policy = self.policy_head(x.float(), mask.float())
                value = self.value_head(x.float(), mask.float())
        else:
            policy, value = self.policy_head(x, mask), self.value_head(x, mask)
        return policy, value[:, :3], value[:, 3:12].reshape(-1, 3, 3), value[:, 12]


class MuZeroNet(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.canvas = config.canvas
        self.latent_channels = config.latent_channels
        self.predict_q_values = config.predict_q_values
        self.norm_kind = 'fixscaleonenorm' if config.architecture == 'nbt' else 'masked_layernorm'
        self.architecture = config.architecture
        self.representation = Representation(config)
        self.dynamics = Dynamics(config)
        self.prediction = Prediction(config)

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.prediction(self.representation(obs, globals))


def network_config(config):
    if 'muzero' not in config or 'unroll' not in config:
        raise ValueError('MuZero requires explicit muzero and unroll sections')
    n = config['network']
    return NetworkConfig(canvas=n['canvas'], representation_channels=n['channels'],
                         representation_blocks=n['blocks'], predict_q_values=n['predict_q_values'],
                         architecture=n['architecture'], **config['muzero'])


class InferenceMuZeroNet(nn.Module):
    """initial/recurrent -> latent, ordinary logits, WDL logits, optimistic, stdev.

    latent [B,C+1,H,W] stays FP32, including during FP16 inference. Actions are
    int64 [B] on its device, with the same fixed D4 orientation as this latent.
    The backend validates nonempty binary masks and on-board action coordinates.
    Occupancy and real-game terminal status never enter recurrent inference.
    """
    def __init__(self, model: MuZeroNet):
        super().__init__()
        self.model = model
        self.contract = CONTRACT_ID

    @torch.jit.export
    def metadata(self) -> tuple[int, str, str, int]:
        return self.model.canvas, self.contract, 'muzero', self.model.latent_channels

    def predict(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, value, _, error = self.model.prediction(hidden)
        stdev = 0.5 * F.softplus(error.float() * 0.5)
        return hidden, policy[:, 0], value, policy[:, 5], stdev

    @torch.jit.export
    def initial(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.predict(self.model.representation(obs, globals))

    @torch.jit.export
    def recurrent(self, hidden: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.predict(self.model.dynamics(hidden, actions))

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.initial(obs, globals)


def inference_network(model: MuZeroNet) -> InferenceMuZeroNet:
    if model.training:
        raise ValueError('Inference normalization conversion requires evaluation mode')
    result = copy.deepcopy(model)
    result.prediction.fp32_heads = False
    if model.architecture == 'nbt':
        for part in (result.representation, result.dynamics, result.prediction):
            part.trunk.norm = InferenceNormalization(part.trunk.norm)
    return InferenceMuZeroNet(result)
