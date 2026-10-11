"""Raw MuZero loss, ordinary optimizer, sampling and recovery semantics."""
from config_samples import CONFIGS
import copy
import json
import math
import os

import numpy as np
import pytest
import torch

from etazero.config import ROOT, load_config, validate
from etazero.engine_config import load_engine_config
from etazero.muzero.training import TrainingForward
from etazero.optimization import optimizer_for, optimization_for, inference_weights


def test_raw_profile_and_shuffle_budget():
    from etazero.shuffle import resource_plan
    c = load_config(CONFIGS / 'raw_muzero')
    assert c['training']['d4_augmentation'] and c['symmetry']['nn_randomize']
    assert not c['fpu']['use_fpu'] and not c['lcb']['use_lcb']
    assert not c['policy_target']['policy_target_pruning']
    assert c['policy_target']['chosen_move_prune'] == 0
    assert c['forced_playouts']['root_desired_per_child_visits_coeff'] == 0
    assert c['search']['cheap_search_probs'] == 0
    assert not c['reduce_visits']['reduce_visits']
    assert c['value_weighting']['value_weight_exponent'] == 0
    assert sum(c['surprise_weighting'][k] for k in ('policy_surprise_data_weight', 'value_surprise_data_weight')) == 0
    assert not c['dirichlet_noise']['shaped_dirichlet_noise'] and c['dirichlet_noise']['noise_fraction'] > 0
    assert c['replay']['min_rows'] == c['replay']['max_rows']
    assert resource_plan(150000, c['writer']['shard_rows'], 10, c)['bucket_rows'] >= 8192
    for match in (False, True):
        search = load_engine_config(CONFIGS / 'raw_muzero', match=match)['match' if match else 'analysis']
        for key in ('use_fpu', 'use_lcb', 'policy_target_pruning', 'use_uncertainty', 'use_noise_pruning',
                    'policy_optimism', 'root_policy_optimism', 'value_weight_exponent', 'chosen_move_prune'):
            assert not search[key]
        assert search['nn_randomize']
    c['network']['predict_q_values'] = True
    with pytest.raises(ValueError, match='without auxiliary losses'):
        validate(c)


def test_basic_loss_hand_calculation_and_auxiliary_independence():
    class Prediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Parameter(torch.zeros(2, 6, 4))
            self.value = torch.nn.Parameter(torch.zeros(2, 3))
            self.td = torch.nn.Parameter(torch.full((2, 3, 3), float('nan')))
            self.error = torch.nn.Parameter(torch.full((2,), float('nan')))

        def prediction(self, hidden):
            return self.policy, self.value, self.td, self.error

    model = Prediction()
    f = TrainingForward(model, load_config(CONFIGS / 'raw_muzero'))
    obs = torch.ones(2, 5, 2, 2); obs[1, 0, 1, :] = 0
    # Weighted batch mean: row 1 has four actions, row 2 has two.
    policy = torch.tensor([[[1., 0, 0, 0]], [[0., 1, 0, 0]]])
    value = torch.tensor([[[1., 0, 0]], [[0., 0, 1]]])
    ignored = torch.full((2, 1), float('nan'))
    targets = (policy, ignored, ignored, value, ignored, ignored, ignored, ignored)
    parts = f.components(None, obs, targets, 0, torch.tensor([1., .5]))
    assert parts[1].item() == pytest.approx((math.log(4) + .5*math.log(2))/2)
    assert parts[5].item() == pytest.approx(.75*math.log(3))
    assert parts[0].item() == pytest.approx(parts[1].item() + parts[5].item())
    assert all(parts[i].item() == 0 for i in (2, 3, 4, 6, 7, 8, 9, 10, 11))
    parts[0].backward()
    assert model.td.grad is None and model.error.grad is None
    assert torch.count_nonzero(model.policy.grad[:, 1:]) == 0
    assert model.policy.grad[0, 0].tolist() == pytest.approx([-.375, .125, .125, .125])
    assert model.policy.grad[1, 0].tolist() == pytest.approx([.125, -.125, 0, 0])


@pytest.mark.parametrize('kind', ['adamw', 'sgd'])
def test_plain_optimizer_matches_torch_and_resumes(kind):
    c = load_config(CONFIGS / 'raw_muzero'); c['optimizer']['kind'] = kind
    model = torch.nn.Linear(2, 1); reference = copy.deepcopy(model)
    opt = optimizer_for(model, c); control = optimization_for(model, c, opt)
    options = dict(lr=.001, weight_decay=.0003)
    refopt = (torch.optim.AdamW(reference.parameters(), **options) if kind == 'adamw'
              else torch.optim.SGD(reference.parameters(), **options, momentum=.9))
    x = torch.tensor([[1., 2.], [-2., 3.]])
    for step in range(9):
        if step in (0, 4): control.begin_round()
        control.before_step()
        opt.zero_grad(); refopt.zero_grad()
        (model(x).square().mean()*control.backward_scale).backward()
        reference(x).square().mean().backward()
        opt.step(); refopt.step(); control.after_step()
        if step == 3: control.finish_round()
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if step == 5:
            saved = copy.deepcopy(dict(model=model.state_dict(), optimizer=opt.state_dict(), optimization=control.state_dict()))
            model = copy.deepcopy(model)
            opt = optimizer_for(model, c); opt.load_state_dict(saved['optimizer'])
            control = optimization_for(model, c, opt, saved['optimization'])
            assert inference_weights(saved) is saved['model']
            assert saved['optimization']['swa'] is None and 'slow' not in saved['optimization']
    assert control.optimizer_steps == 9 and control.consumed_samples == 9*c['training']['batch_size']
    assert control.gradient_cap(0) == float('inf') and control.gradient_cap(2) == 2


def small_raw_config():
    from test_muzero_pipeline import small_config
    c = load_config(CONFIGS / 'raw_muzero'); small = small_config()
    for section in ('network', 'muzero', 'unroll', 'environment', 'devices', 'parallelism', 'selfplay', 'writer', 'shuffle'):
        c[section] = small[section]
    for key in ('train_steps', 'batch_size', 'checkpoint_every', 'compile', 'skip_validation', 'prefetch_depth', 'replay_ratio'):
        c['training'][key] = small['training'][key]
    for key in ('full_search_visits', 'cheap_search_visits'):
        c['search'][key] = small['search'][key]
    c['reduce_visits']['reduced_visits_min'] = 2
    c['replay'].update(min_rows=32, max_rows=256)
    c['inference'].update(server_threads=2, max_batch=8)
    validate(c)
    return c


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA acceptance')
def test_raw_cuda_pipeline_and_resume(tmp_path):
    from etazero.runtime import run_training
    from etazero.training import load_checkpoint
    from etazero.data import read_raw
    from test_gpu import assert_compiled_partial_resume
    from test_muzero_pipeline import assert_diagnostics
    torch.set_num_threads(1)
    c = small_raw_config(); c['training'].update(amp='float16', compile=True)
    root = tmp_path/'raw'
    state = run_training(root, c, ROOT/'build/etazero', max_iteration=2)
    saved = load_checkpoint(root, state['checkpoint'], c)
    assert saved['optimization']['optimizer_steps'] > 0
    assert saved['optimization']['swa'] is None and 'slow' not in saved['optimization']
    assert_diagnostics(root, c)
    for path in (root/'selfplay').rglob('*.npz'):
        a = read_raw(path)
        assert np.all(a['row_repeats'] == 1)
        assert np.all(a['target_weights'] == 1)
        assert len(a['side_row_repeats']) == 0
    updates = [json.loads(line) for line in (root/'logs/events.jsonl').read_text().splitlines()
               if json.loads(line)['event'] == 'update']
    for row in updates:
        assert row['loss'] == pytest.approx(row['policy_loss'] + row['value_loss'], rel=1e-6)
        assert row['learning_rates'] == {'all': .001}
        assert row['weight_decays'] == {'all': .0003}
        assert row['swa_samples'] == row['lookahead_counter'] == 0
    assert run_training(root, c, ROOT/'build/etazero', resume=True, max_iteration=2) == state
    assert_compiled_partial_resume(tmp_path, root, c)
