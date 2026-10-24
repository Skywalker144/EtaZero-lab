"""KataGo's handwritten presets and ordered suffix expansion, resolved lazily.

The source eagerly builds 848640 dictionaries. Resolve only the selected name,
with the same suffix order, restrictions, and values, without that runtime cost.
"""
import copy
from ._vendor.katago.modelconfigs import base_config_of_name

BASE_NAMES = tuple(base_config_of_name)
_BASE_PREFIXES = sorted(BASE_NAMES, key=len, reverse=True)
_NORMS = {'fs': 'fixscale', 'bn': 'bnorm', 'brn': 'brenorm',
          'fbrn': 'fixbrenorm', 'bng': 'bnorm', 'brng': 'brenorm',
          'fbrng': 'fixbrenorm', 'fson': 'fixscaleonenorm'}


def resolve(name):
    """Return an independent source configuration; reject ungenerated names."""
    base = next((key for key in _BASE_PREFIXES
                 if name == key or name.startswith(key + '-')), None)
    if base is None:
        raise ValueError(f'Unknown KataGo network preset: {name}')
    config = copy.deepcopy(base_config_of_name[base])
    config['norm_kind'] = 'fixup'
    previous = -1
    for suffix in name[len(base) + 1:].split('-') if name != base else ():
        if suffix in _NORMS:
            stage = 0
            config['norm_kind'] = _NORMS[suffix]
            if suffix in ('bng', 'brng', 'fbrng', 'fson'):
                config['bnorm_use_gamma'] = True
        elif suffix in ('elu', 'gelu', 'mish', 'silu'):
            stage = 1
            config['activation'] = suffix
        elif suffix in ('rvgi', 'rvgl', 'rvglr'):
            stage = 2
            config['use_repvgg_linear' if suffix == 'rvgl' else 'use_repvgg_init'] = True
            if suffix == 'rvglr':
                config['use_repvgg_learning_rate'] = True
        elif suffix == 'tcg':
            stage = 3
            config['use_trunk_channel_gate'] = True
        elif suffix in ('trb', 'trbnr'):
            stage = 4
            config['use_trunk_residual_backout'] = True
            if suffix == 'trbnr':
                config['trunk_residual_backout_noreg'] = True
        elif suffix in ('ih', 'bnh', 'rnh', 'rsnh', 'rgnh'):
            stage = 5
            if suffix in ('ih', 'bnh'):
                if suffix == 'bnh' and 'norm' not in config['norm_kind']:
                    raise ValueError(f'KataGo -bnh requires batch normalization: {name}')
                config['has_intermediate_head'] = True
                config['intermediate_head_blocks'] = (len(config['block_kind']) if suffix == 'bnh'
                                                     else len(config['block_kind']) // 2)
                if suffix == 'bnh':
                    config['trunk_normless'] = True
            else:
                config['trunk_final_rmsnorm'] = True
                if suffix in ('rsnh', 'rgnh'):
                    config['trunk_rmsnorm_spatial'] = True
                if suffix == 'rgnh':
                    config['rmsnorm_spatial_cgroup_size'] = 16
        elif suffix == 'meta':
            stage = 6
            config['metadata_encoder'] = {'meta_encoder_version': 1,
                                          'internal_num_channels': config['trunk_num_channels']}
        else:
            raise ValueError(f'Unknown KataGo network suffix in {name}: {suffix}')
        if stage <= previous:
            raise ValueError(f'KataGo network suffixes are duplicated or out of order: {name}')
        previous = stage
    return config


def source_architecture(name, channels=None, blocks=None):
    aliases = {'nbt': ('b5c192nbt-fson-mish', 192, 5),
               'plain': ('b10c128-fson-mish', 128, 10),
               'transformer': ('b5c192h3nbttfrs', 192, 5)}
    if name not in aliases:
        return name
    preset, width, depth = aliases[name]
    if (channels is not None and channels != width) or (blocks is not None and blocks != depth):
        raise ValueError(f'{name} is an alias for the fixed source preset {preset}; choose a named KataGo preset')
    return preset


def is_preset(name):
    return name not in ('nbt', 'plain', 'transformer', 'resnet')


def validate_dimensions(config):
    group = config.get('rmsnorm_spatial_cgroup_size')
    if group is not None and (group <= 0 or config['trunk_num_channels'] % group):
        raise ValueError('KataGo spatial RMSNorm channels must be divisible by its channel group size')


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', help='Resolve one complete source network name')
    args = parser.parse_args()
    print(json.dumps(resolve(args.name), indent=2) if args.name else '\n'.join(BASE_NAMES))
