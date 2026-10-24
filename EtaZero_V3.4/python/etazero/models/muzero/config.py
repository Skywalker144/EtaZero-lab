"""Pure MuZero component dimensions and configuration validation."""
from dataclasses import dataclass
from etazero.models.katago_configs import network_widths
from etazero.games import SPECS, spec_for_config


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


def network_config(config):
    if 'muzero' not in config or 'unroll' not in config:
        raise ValueError('MuZero requires explicit muzero and unroll sections')
    n = config['network']
    return NetworkConfig(canvas=n['canvas'], representation_channels=n['channels'],
                         representation_blocks=n['blocks'], predict_q_values=n['predict_q_values'],
                         architecture=n['architecture'], game=spec_for_config(config).name, **config['muzero'])


