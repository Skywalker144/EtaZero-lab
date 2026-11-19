"""Checkpoint-only imports do not expand the named network catalogue."""
import importlib.util
from pathlib import Path

import pytest


def fresh_module():
    path = Path(__file__).resolve().parents[1] / 'katago/train/modelconfigs.py'
    spec = importlib.util.spec_from_file_location('fresh_modelconfigs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_helpers_do_not_build_named_presets():
    module = fresh_module()
    assert module.get_num_bin_input_features({'version': 18}) == 5
    assert module.get_policy_size({'version': 18}, 7) == 49
    assert 'config_of_name' not in vars(module)
    with pytest.raises(AttributeError):
        module.unknown_attribute
    assert 'config_of_name' not in vars(module)


def test_named_presets_are_cached_and_keep_source_suffix_semantics():
    module = fresh_module()
    configs = module.config_of_name
    assert module.config_of_name is configs
    base = module.base_config_of_name['b2c16']
    assert configs['b2c16'] == dict(base, norm_kind='fixup')
    expected = dict(base, norm_kind='fixscaleonenorm', bnorm_use_gamma=True,
                    activation='mish', use_repvgg_init=True, use_repvgg_learning_rate=True,
                    use_trunk_channel_gate=True, use_trunk_residual_backout=True,
                    trunk_residual_backout_noreg=True, has_intermediate_head=True,
                    intermediate_head_blocks=len(base['block_kind']), trunk_normless=True,
                    metadata_encoder={'meta_encoder_version': 1,
                                      'internal_num_channels': base['trunk_num_channels']})
    assert configs['b2c16-fson-mish-rvglr-tcg-trbnr-bnh-meta'] == expected
    assert 'b2c16-bnh' not in configs  # No batchnorm in the default fixup model.
    with pytest.raises(KeyError):
        configs['b2c16-mish-fson']  # Suffix order is the source order.
