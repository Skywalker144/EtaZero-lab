"""Export MuZero components through KataGo's shared binary layer writer.

The checkpoint entry selects ordinary or SWA weights explicitly. Publication of
the completed bundle belongs to the outer controller.
"""
from copy import deepcopy
import json
from pathlib import Path

from export_model_pytorch import prepare_model_for_export, write_model_binary


def export_components(model, directory, model_name, **export_options):
    """Write a new directory with three independently configured source networks.

    No training tensors are modified: export optimizations operate on a CPU/eval
    copy. Partial output is retained on failure and must not be published.
    """
    model = deepcopy(model).cpu().eval()
    model.network_config.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for role, name in enumerate(('representation', 'dynamics', 'prediction'), 1):
        component = getattr(model, name)
        body = role != 3
        trunk = component.trunk if body else component
        prepare_model_for_export(trunk, **export_options)
        with (directory / f'{name}.bin').open('wb') as stream:
            write_model_binary(trunk, stream, f'{model_name}-{name}',
                muzero_component=role, latent_channels=model.latent_channels,
                projection=component.projection if body else None)
    # Written last; the outer controller will own atomic publication of a bundle.
    (directory / 'model.json').write_text(json.dumps(dict(
        game=model.game, latent_channels=model.latent_channels,
        pos_len=model.pos_len), indent=2) + '\n')


def main():
    import argparse
    from katago.train.load_model import load_model
    from katago.train.muzero_model import MuZeroModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-checkpoint', required=True)
    parser.add_argument('-export-dir', required=True)
    parser.add_argument('-model-name', required=True)
    parser.add_argument('-pos-len', type=int, required=True)
    parser.add_argument('-use-swa', action='store_true')
    parser.add_argument('-prune-dead-ffn-channels', action='store_true')
    parser.add_argument('-prune-ffn-scratch-factor', type=float, default=2.0)
    parser.add_argument('-attn-logit-bound-limit', type=float, default=2.5e4)
    parser.add_argument('-ignore-attn-logit-bound', action='store_true')
    args = parser.parse_args()
    model, swa, _ = load_model(args.checkpoint, args.use_swa, 'cpu', pos_len=args.pos_len)
    if not isinstance(model, MuZeroModel):
        raise ValueError('Expected a MuZero checkpoint; use export_model_pytorch.py for AlphaZero')
    export_components(swa if args.use_swa else model, args.export_dir, args.model_name,
                      prune_ffn=args.prune_dead_ffn_channels,
                      prune_ffn_scratch_factor=args.prune_ffn_scratch_factor,
                      attn_logit_bound_limit=args.attn_logit_bound_limit,
                      ignore_attn_logit_bound=args.ignore_attn_logit_bound)


if __name__ == '__main__':
    main()
