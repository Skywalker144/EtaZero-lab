"""NBT or dense ResNet MuZero with EtaZero observations and prediction heads.

The final latent channel is the immutable board mask, not an occupancy mask.
Dynamics consumes only latent and a fixed-vocabulary action. All predictions
describe the player to move at that latent step; search must flip W/L on edges.
No reward, terminal predictor, real-board transition or future observation is used.
Masked min/max normalization adapts MuZero_V2 with MiniZero's small-span guard
(Apache-2.0, attribution and pinned source in THIRD_PARTY.md).
"""
import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from etazero.config import network_widths
from etazero.schema import PLANES, GLOBALS
from etazero.games import SPECS, XIANGQI_MOVES, spec_for_config


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
    game: str = 'gomoku'

    def __post_init__(self):
        for key, value in vars(self).items():
            if key == 'game':
                if value not in SPECS: raise ValueError('Unknown game')
            elif key == 'architecture':
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
    """FP32 masked per-sample min/max, with MiniZero's small-span protection."""
    x = hidden.float()
    minimum = x.masked_fill(mask == 0, float('inf')).amin([1, 2, 3], keepdim=True)
    maximum = x.masked_fill(mask == 0, float('-inf')).amax([1, 2, 3], keepdim=True)
    span = maximum - minimum
    # MiniZero adds epsilon only to small spans; ordinary ranges stay unchanged.
    denominator = torch.where(span < 1e-5, span + 1e-5, span)
    return ((x - minimum) / denominator).masked_fill(mask == 0, 0)


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity forward and scaled backward, as in MuZero_V2 unroll training."""
    return tensor * scale + tensor.detach() * (1.0 - scale)


from etazero.models.muzero.source_components import Body, Prediction, component_config


class Representation(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.body=Body(component_config(c.representation_channels,c.representation_blocks,c.architecture),
                       c.canvas,SPECS[c.game].planes,SPECS[c.game].globals,c.latent_channels)

    def forward(self,obs,globals):
        mask=obs[:,:1].float()
        return torch.cat((normalize_hidden_state(self.body(obs,globals),mask),mask),1)


class Dynamics(nn.Module):
    def __init__(self,c):
        super().__init__();self.game=c.game;self.canvas=c.canvas
        moves=[(y*c.canvas+x,v*c.canvas+u) for x,y,u,v in XIANGQI_MOVES] if c.game=='xiangqi' else []
        self.register_buffer('move_squares',torch.tensor(moves,dtype=torch.long),persistent=False)
        self.body=Body(component_config(c.dynamics_channels,c.dynamics_blocks,c.architecture),
                       c.canvas,c.latent_channels+(2 if c.game=='xiangqi' else 1)+1,0,c.latent_channels)

    def forward(self,hidden,actions):
        mask=hidden[:,-1:];area=self.canvas*self.canvas
        if self.game=='xiangqi':
            squares=self.move_squares.index_select(0,actions)
            action=F.one_hot(squares,area).to(hidden.dtype).reshape(-1,2,self.canvas,self.canvas)
        elif self.game=='go':action=F.one_hot(actions,area+1)[:,:area].to(hidden.dtype).reshape(-1,1,self.canvas,self.canvas)
        else:action=F.one_hot(actions,area).to(hidden.dtype).reshape(-1,1,self.canvas,self.canvas)
        x=self.body(torch.cat((mask,hidden[:,:-1],action),1))
        return torch.cat((normalize_hidden_state(x,mask),mask),1)


class MuZeroNet(nn.Module):
    def __init__(self, config: NetworkConfig, auxiliary_losses=True):
        super().__init__()
        self.game = config.game
        self.canvas = config.canvas
        self.latent_channels = config.latent_channels
        self.predict_q_values = config.predict_q_values
        self.auxiliary_losses = auxiliary_losses
        self.norm_kind = 'fixscaleonenorm'
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
                         architecture=n['architecture'], game=spec_for_config(config).name, **config['muzero'])


class InferenceMuZeroNet(nn.Module):
    """initial/recurrent -> latent, ordinary logits, WDL logits, optimistic, stdev.

    latent [B,C+1,H,W] stays FP32, including during FP16 inference. Actions are
    int64 [B] on its device, with the same fixed root orientation as this latent.
    The backend validates nonempty binary masks and the static action vocabulary.
    Occupancy and real-game terminal status never enter recurrent inference.
    """
    def __init__(self, model: MuZeroNet):
        super().__init__()
        self.model = model
        self.has_auxiliary = model.auxiliary_losses

    def game_name(self) -> str:
        return self.model.game

    def metadata(self) -> tuple[int, str, int]:
        return self.model.canvas, 'muzero', self.model.latent_channels

    def raw(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy, value, _, _ = self.model.prediction(self.model.representation(obs, globals))
        return policy, value

    def predict(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, value, _, error = self.model.prediction(hidden)
        stdev = 0.5 * F.softplus(error.float() * 0.5)
        return hidden, policy[:, 0], value, policy[:, 5], stdev

    def initial(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.predict(self.model.representation(obs, globals))

    def recurrent(self, hidden: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.predict(self.model.dynamics(hidden, actions))

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.initial(obs, globals)


class SerializedMuZero(nn.Module):
    def __init__(self,graph,model):
        super().__init__();self.graph=graph;self.canvas=model.canvas;self.game=model.game;self.latent_channels=model.latent_channels
        self.has_auxiliary=model.auxiliary_losses

    @torch.jit.export
    def game_name(self)->str:return self.game
    @torch.jit.export
    def metadata(self)->tuple[int,str,int]:return self.canvas,'muzero',self.latent_channels
    @torch.jit.export
    def initial(self,obs:torch.Tensor,globals:torch.Tensor):return self.graph.initial(obs,globals)
    @torch.jit.export
    def recurrent(self,hidden:torch.Tensor,actions:torch.Tensor):return self.graph.recurrent(hidden,actions)
    @torch.jit.export
    def raw(self,obs:torch.Tensor,globals:torch.Tensor):return self.graph.raw(obs,globals)
    def forward(self,obs:torch.Tensor,globals:torch.Tensor):return self.initial(obs,globals)


def inference_network(model: MuZeroNet):
    if model.training:raise ValueError('Inference export requires evaluation mode')
    result=InferenceMuZeroNet(copy.deepcopy(model))
    spec=SPECS[model.game];device=next(model.parameters()).device;n=model.canvas
    obs=torch.zeros((2,spec.planes,n,n),device=device);obs[:,0]=1
    globals=torch.zeros((2,spec.globals),device=device)
    hidden=torch.zeros((2,model.latent_channels+1,n,n),device=device);hidden[:,-1]=1
    actions=torch.zeros(2,dtype=torch.long,device=device)
    with torch.no_grad():
        graph=torch.jit.trace_module(result,{'initial':(obs,globals),'recurrent':(hidden,actions),'raw':(obs,globals)},check_trace=False)
    return SerializedMuZero(graph,model)
