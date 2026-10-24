"""Construct and export the active AlphaZero and MuZero models."""

def make_network(config):
    if config['agent']['algorithm'] == 'muzero':
        from etazero.models.muzero.network import MuZeroNet
        from etazero.models.muzero.config import network_config
        return MuZeroNet(network_config(config), config.get('muzero_training', {}).get('auxiliary_losses', True))
    from etazero.games import spec_for_config
    game = spec_for_config(config).name
    from etazero.models.katago_configs import source_architecture
    from etazero.models.katago_network import KataGoNet
    options = dict(config['network'])
    options['architecture'] = source_architecture(options['architecture'], options.get('channels'), options.get('blocks'))
    return KataGoNet(**options, game=game)


def inference_network(model):
    """Cache final BN statistics on the inference device, retaining CUDA operation order."""
    from etazero.models.muzero.network import MuZeroNet, inference_network as muzero_inference
    if isinstance(model, MuZeroNet):
        return muzero_inference(model)
    from etazero.models.katago_network import KataGoNet, inference_network as katago_inference
    if isinstance(model, KataGoNet):
        return katago_inference(model)
    raise TypeError(f"Unknown inference model: {type(model).__name__}")
