import copy
import os

import pytest
from config_samples import CONFIGS, derive_config
from etazero.config import ROOT, load_config, validate
from etazero.engine_config import load_engine_config
from etazero.run.runtime import run_training


def test_subtree_bias_resolves_through_training_and_engine_profiles(tmp_path):
    selected = derive_config(tmp_path/'profile', CONFIGS/'smoke_test')
    (selected/'selfplay.cfg').write_text(
        f'@include {CONFIGS / "smoke_test/selfplay.cfg"}\n[subtree_value_bias]\n'
        'subtree_value_bias_factor=.3\nsubtree_value_bias_free_prop=.6\n'
        'subtree_value_bias_weight_exponent=.75\nsubtree_value_bias_table_num_shards=7\n')
    c = load_config(selected, run_dir=tmp_path/'run')
    assert c['subtree_value_bias'] == dict(subtree_value_bias_factor=.3, subtree_value_bias_free_prop=.6,
                                         subtree_value_bias_weight_exponent=.75, subtree_value_bias_table_num_shards=7)
    for match in (False, True):
        mode = 'match' if match else 'analysis'
        c = load_engine_config(CONFIGS/'smoke_test', match, environ={
            'ENGINE_SUBTREE_VALUE_BIAS_FACTOR': '.3',
            mode.upper()+'_SUBTREE_VALUE_BIAS_FACTOR': '.5',
            'ENGINE_SUBTREE_VALUE_BIAS_TABLE_NUM_SHARDS': '7'})[mode]
        assert c['subtree_value_bias_factor'] == .5
        assert c['subtree_value_bias_table_num_shards'] == 7


@pytest.mark.parametrize('key,value', [
    ('subtree_value_bias_factor', -0.1), ('subtree_value_bias_factor', 1.01),
    ('subtree_value_bias_free_prop', 1.1), ('subtree_value_bias_weight_exponent', -0.1),
    ('subtree_value_bias_table_num_shards', 0), ('subtree_value_bias_table_num_shards', 2**31),
    ('subtree_value_bias_factor', float('nan'))])
def test_subtree_bias_rejects_invalid_values(key, value):
    c = load_config(CONFIGS/'smoke_test')
    c['subtree_value_bias'][key] = value
    with pytest.raises(ValueError, match='subtree_value_bias'):
        validate(c)
    with pytest.raises(ValueError, match='subtree_value_bias'):
        load_engine_config(CONFIGS/'smoke_test', environ={'ENGINE_'+key.upper(): str(value)})


def test_muzero_refuses_real_board_bias():
    from test_muzero_pipeline import small_config
    c = small_config()
    c['subtree_value_bias']['subtree_value_bias_factor'] = .3
    with pytest.raises(ValueError, match='MuZero latent search requires subtree_value_bias_factor=0'):
        validate(c)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA acceptance')
def test_subtree_bias_cuda_pipeline_and_resume(tmp_path):
    c = copy.deepcopy(load_config(CONFIGS/'smoke_test'))
    c['subtree_value_bias'].update(subtree_value_bias_factor=.45, subtree_value_bias_free_prop=.8,
                                 subtree_value_bias_weight_exponent=.85, subtree_value_bias_table_num_shards=32)
    c['devices'].update(train='cuda:0', selfplay='cuda:0')
    # Enough depth to learn a subtree residual, with real CUDA batched inference.
    c['search'].update(full_search_visits=32, cheap_search_probs=0)
    c['opening']['probability'] = 0
    c['policy_init']['policy_init'] = False
    state = run_training(tmp_path/'run', c, ROOT/'build/etazero', max_iteration=2)
    assert state['iteration'] == 3
    assert state['checkpoint']['total_steps'] > 0
    assert run_training(tmp_path/'run', c, ROOT/'build/etazero', max_iteration=2) == state
