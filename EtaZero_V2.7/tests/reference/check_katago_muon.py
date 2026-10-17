"""Compare Muon updates and LR/WD against the registered KataGo source."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import logging
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['KATAGO_MUON_BATCHED_NS'] = '0'
    import torch
    from etazero.muon import Muon
    from etazero.optimization import group_settings
    torch.set_num_threads(1)
    reference = json.loads((ROOT/'reference_sources.json').read_text())['KataGoMuon']
    source = Path(reference['root'])
    for name, checksum in reference['sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == checksum, name
    assert (ROOT/'python/etazero/_vendor/muon.py').read_bytes() == (source/'python/muon/muon.py').read_bytes()
    spec = importlib.util.spec_from_file_location('_katago_muon_reference', source/'python/muon/muon.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(29)
    params = [torch.nn.Parameter(torch.randn(shape)) for shape in [(4, 3), (3, 2, 3, 3), (3, 4), (3,)]]
    replicas = [torch.nn.Parameter(p.detach().clone()) for p in params]
    groups = [dict(group_name='normal', params=params[:2], lr=.007, weight_decay=.02),
              dict(group_name='input', params=params[2:], lr=.003, weight_decay=.01)]
    source_groups = copy.copy(groups)
    source_groups = [dict(g, params=replicas[:2] if i == 0 else replicas[2:], use_muon=i == 0)
                     for i, g in enumerate(source_groups)]
    optimizer = Muon(groups)
    original = module.SingleDeviceMuonWithAuxAdam(source_groups)
    for _ in range(4):
        for p, replica in zip(params, replicas):
            p.grad = torch.randn_like(p); replica.grad = p.grad.clone()
        optimizer.step(); original.step()
        for p, replica in zip(params, replicas):
            torch.testing.assert_close(p, replica, rtol=0, atol=0)
            for name, value in optimizer.state[p].items():
                torch.testing.assert_close(value, original.state[replica][name], rtol=0, atol=0)

    wanted = ('get_is_muon_suitable', 'get_weight_decay', 'update_and_return_lr_and_wd')
    tree = ast.parse((source/'python/train.py').read_text())
    functions = {name: [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
                 for name in wanted}
    assert all(len(nodes) == 1 for nodes in functions.values())
    code = compile(ast.Module(body=[functions[name][0] for name in wanted], type_ignores=[]),
                   str(source/'python/train.py'), 'exec')
    options = dict(kind='muon', lr_scale=1.0, lr_warmup=True, lookahead_alpha=.5,
                   head_lr_factor=.65, noreg_lr_factor=1.2, input_wd_factor=1.3,
                   normal_wd_factor=.9, normal_attn_wd_factor=.7)
    cases = 0
    for norm_kind, samples, batch, scale, norm_ratio in itertools.product(
            ['fixup', 'fixscaleonenorm'], [0, 250000, 1250000, 2000000], [8, 64, 256], [.25, 4], [.5, 2]):
        names = ['input', 'normal', 'normal_gamma', 'noreg', 'output', 'output_noreg']
        if norm_kind == 'fixup':
            names.append('normal_attn')
        state = dict(global_step_samples=samples, modelnorm_input_baseline=1., modelnorm_normal_baseline=2.)
        metrics = dict(sums={'norm_input_batch': norm_ratio, 'norm_normal_batch': 2*norm_ratio},
                       weights={'norm_input_batch': 1., 'norm_normal_batch': 1.})
        source_optimizer = SimpleNamespace(param_groups=[dict(group_name=n, lr=0., weight_decay=0.) for n in names])
        env = dict(math=math, logging=logging, use_muon=True, use_adamw=False, world_size=1,
                   batch_size=batch, no_lr_warmup=False, lookahead_alpha=.5, muon_adam_lr_factor=1.,
                   train_state=state, running_metrics=metrics, optimizer=source_optimizer,
                   raw_model=SimpleNamespace(get_norm_kind=lambda: norm_kind),
                   get_effective_lr_scale=lambda state: scale,
                   **{key: options[key] for key in ('head_lr_factor', 'noreg_lr_factor', 'input_wd_factor',
                                                   'normal_wd_factor', 'normal_attn_wd_factor')})
        exec(code, env)
        env['update_and_return_lr_and_wd'](log_if='never')
        for group in source_optimizer.param_groups:
            lr, wd = group_settings(group['group_name'], dict(options, lr_scale=scale), batch, samples,
                                    {'input': norm_ratio, 'normal': 2*norm_ratio}, {'input': 1., 'normal': 2.}, norm_kind)
            assert math.isclose(lr, group['lr'], rel_tol=1e-14), (group, lr)
            assert math.isclose(wd, group['weight_decay'], rel_tol=1e-14), (group, wd)
            cases += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(source=reference, updates=4, lr_wd_cases=cases, passed=True), indent=2)+'\n')
    print(f'Passed: 4 source optimizer updates, {cases} source LR/WD comparisons')


if __name__ == '__main__':
    main()
