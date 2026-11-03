"""MuZero components using the source Model trunk and prediction heads.

Latents contain FP32 features followed by the immutable board mask. Recurrent
inference sees only that latent and a fixed-vocabulary action, never a successor
observation or a real-board terminal/legality result.

Normalization attribution: MiniZero (Apache-2.0), via EtaZero V3.4; see
reference_sources.json and licenses/MiniZero.txt.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import NamedTuple
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model_pytorch import Model, init_weights


@dataclass(frozen=True)
class MuZeroConfig:
    game: str
    latent_channels: int
    representation: dict
    dynamics: dict
    prediction: dict

    def validate(self):
        if self.game not in {'go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'}:
            raise ValueError(f'MuZero action layout is not connected for {self.game}')
        if type(self.latent_channels) is not int or self.latent_channels < 1:
            raise ValueError('latent_channels must be a positive integer')
        for name in ('representation', 'dynamics', 'prediction'):
            config = getattr(self, name)
            if ((config['version'] == 18) != (self.game not in {'go', 'xiangqi'}) or
                    (config['version'] == 19) != (self.game == 'xiangqi')):
                raise ValueError('Component versions must match the observation and action layout')
            if name != 'representation' and config.get('metadata_encoder') is not None:
                raise ValueError('Only representation can consume observation metadata')
        return self


def normalize_hidden_state(hidden, mask):
    """Masked FP32 min/max, adapting MiniZero's small-span guard via EtaZero V3.4."""
    x = hidden.float()
    minimum = x.masked_fill(mask == 0, float('inf')).amin((1, 2, 3), keepdim=True)
    maximum = x.masked_fill(mask == 0, float('-inf')).amax((1, 2, 3), keepdim=True)
    span = maximum - minimum
    denominator = torch.where(span < 1e-5, span + 1e-5, span)
    return ((x - minimum) / denominator).masked_fill(mask == 0, 0)


def scale_gradient(tensor, scale):
    """Identity forward; multiply only the backward contribution by scale."""
    return tensor * scale + tensor.detach() * (1.0 - scale)


class LatentBody(nn.Module):
    def __init__(self, config, pos_len, spatial_channels, global_channels, latent_channels):
        super().__init__()
        self.trunk = Model(deepcopy(config), pos_len, num_bin_features=spatial_channels,
                           num_global_features=global_channels, trunk_only=True)
        self.projection = nn.Conv2d(self.trunk.c_trunk, latent_channels, 1, bias=False)

    def initialize(self):
        self.trunk.initialize()
        with torch.no_grad():
            init_weights(self.projection.weight, 'identity', scale=1.)

    def forward(self, spatial, global_input, input_meta=None):
        mask = spatial[:, :1].float()
        hidden = self.projection(self.trunk(spatial, global_input, input_meta=input_meta))
        return torch.cat((normalize_hidden_state(hidden, mask), mask), dim=1)

    def get_norm_kind(self):
        return self.trunk.get_norm_kind()

    @property
    def config(self):
        return self.trunk.config

    def add_reg_dict(self, groups):
        groups.clear()
        local = {}
        self.trunk.add_reg_dict(local)
        for role, parameters in local.items():
            target = {'output': 'normal_gamma', 'output_noreg': 'noreg'}.get(role, role)
            groups.setdefault(target, []).extend(parameters)
        groups.setdefault('normal', []).append(self.projection.weight)


class MuZeroStep(NamedTuple):
    rows: torch.Tensor
    hidden: torch.Tensor
    outputs: tuple


class MuZeroModel(nn.Module):
    def __init__(self, config: MuZeroConfig, pos_len: int):
        super().__init__()
        self.network_config = deepcopy(config.validate())
        self.pos_len = pos_len
        self.game = config.game
        self.latent_channels = config.latent_channels
        move_squares = None
        if self.game == 'xiangqi':
            from katago.game.xiangqi import HEIGHT, MOVES
            if pos_len < HEIGHT:
                raise ValueError('Xiangqi needs a canvas that contains its 9 x 10 board')
            move_squares = torch.tensor([(y * pos_len + x, v * pos_len + u) for x, y, u, v in MOVES],
                                        dtype=torch.int64)
        self.register_buffer('move_squares', move_squares, persistent=False)
        from .modelconfigs import get_num_bin_input_features, get_num_global_input_features
        self.representation = LatentBody(config.representation, pos_len,
            get_num_bin_input_features(config.representation),
            get_num_global_input_features(config.representation), self.latent_channels)
        self.dynamics = LatentBody(config.dynamics, pos_len,
            self.latent_channels + 2 + int(self.game == 'xiangqi'), 0, self.latent_channels)
        self.prediction = Model(deepcopy(config.prediction), pos_len,
                                num_bin_features=self.latent_channels + 1, num_global_features=0,
                                heads_only=not config.prediction["block_kind"])

    def initialize(self):
        self.representation.initialize()
        self.dynamics.initialize()
        self.prediction.initialize()

    @property
    def device(self):
        return self.representation.trunk.device

    def predict(self, hidden, prediction=None):
        # Source mask is the first input channel, packed latent mask is the last.
        inputs = torch.cat((hidden[:, -1:], hidden[:, :-1]), dim=1)
        return (self.prediction if prediction is None else prediction)(inputs, inputs.new_empty((inputs.shape[0], 0)))

    def initial(self, spatial, global_input, input_meta=None, component_forward=None):
        representation = self.representation if component_forward is None else component_forward[0]
        hidden = representation(spatial, global_input, input_meta)
        return hidden, self.predict(hidden, None if component_forward is None else component_forward[2])

    def recurrent(self, hidden, actions, component_forward=None):
        if actions.ndim != 1 or actions.dtype != torch.int64 or actions.shape[0] != hidden.shape[0]:
            raise ValueError('Recurrent actions must be int64 [batch]')
        area = self.pos_len * self.pos_len
        if self.game == 'xiangqi':
            # Joint actions already use the root's fixed perspective. Do not
            # rotate alternate plies or consult a real successor board here.
            squares = self.move_squares.index_select(0, actions)
            action_planes = F.one_hot(squares, area).to(hidden.dtype).reshape(-1, 2, self.pos_len, self.pos_len)
        else:
            action_planes = F.one_hot(actions, area + int(self.game == 'go'))[:, :area]
            action_planes = action_planes.to(hidden.dtype).reshape(-1, 1, self.pos_len, self.pos_len)
        # Go pass has an all-zero plane. Occupancy never masks recurrent actions.
        spatial = torch.cat((hidden[:, -1:], hidden[:, :-1], action_planes), dim=1)
        dynamics = self.dynamics if component_forward is None else component_forward[1]
        next_hidden = dynamics(spatial, spatial.new_empty((spatial.shape[0], 0)))
        return next_hidden, self.predict(next_hidden, None if component_forward is None else component_forward[2])

    def forward(self, spatial, global_input, input_meta=None):
        return self.initial(spatial, global_input, input_meta)

    def unroll(self, spatial, global_input, actions, *, active_steps=None,
               hidden_gradient_scale=0.5, input_meta=None, component_forward=None):
        """Yield K+1 predictions with original row indices, removing inactive rows before BN.

        actions: int64 [B,K], already transformed to a fixed root orientation.
        active_steps: optional bool [B,K+1], monotonically false after an episode
        fragment ends; a side position is active only at step 0. The data reader
        owns this trajectory boundary, including absorbing states after real ends.
        """
        if actions.ndim != 2 or actions.dtype != torch.int64 or actions.shape[0] != spatial.shape[0]:
            raise ValueError('Unroll actions must be int64 [batch,steps]')
        if not math.isfinite(hidden_gradient_scale) or not 0 <= hidden_gradient_scale <= 1:
            raise ValueError('hidden_gradient_scale must be finite and in [0,1]')
        if active_steps is not None and (active_steps.dtype != torch.bool or
                active_steps.shape != (actions.shape[0], actions.shape[1] + 1)):
            raise ValueError('active_steps must be bool [batch,steps+1]')
        # The reader already knows sequence boundaries on CPU. Keep row indices
        # there when it supplies a CPU mask, rather than copying GPU indices back
        # to index that mask at every step (or running CUDA nonzero).
        row_device = spatial.device if active_steps is None else active_steps.device
        rows = torch.arange(spatial.shape[0], device=row_device)
        if active_steps is not None:
            rows = rows[active_steps[:, 0]]
        if rows.numel() == 0:
            return
        hidden, outputs = self.initial(spatial[rows], global_input[rows],
                                       None if input_meta is None else input_meta[rows], component_forward)
        yield MuZeroStep(rows, hidden, outputs)
        for step in range(actions.shape[1]):
            if active_steps is not None:
                keep = active_steps[rows, step + 1]
                rows, hidden = rows[keep], hidden[keep]
                if rows.numel() == 0:
                    break
            # The first representation -> dynamics edge keeps its full gradient.
            if step > 0:
                hidden = scale_gradient(hidden, hidden_gradient_scale)
            hidden, outputs = self.recurrent(hidden, actions[rows, step], component_forward)
            yield MuZeroStep(rows, hidden, outputs)

    def add_reg_dict(self, groups):
        groups.clear()
        for component in self.training_components().values():
            local = {}
            component.add_reg_dict(local)
            for role, parameters in local.items():
                groups.setdefault(role, []).extend(parameters)

    def training_components(self):
        return dict(representation=self.representation, dynamics=self.dynamics, prediction=self.prediction)

    def get_has_metadata_encoder(self):
        return self.representation.trunk.get_has_metadata_encoder()

    def get_has_intermediate_head(self):
        return self.prediction.get_has_intermediate_head()

    def set_brenorm_params(self, momentum, rmax, dmax):
        for component in (self.representation.trunk, self.dynamics.trunk, self.prediction):
            component.set_brenorm_params(momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper, lower, d):
        for component in (self.representation.trunk, self.dynamics.trunk, self.prediction):
            component.add_brenorm_clippage(upper, lower, d)
