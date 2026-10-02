"""Recheck EtaZero.md scalar formulas against local KataGo source, without imports.

Only selected function bodies are compiled from AST. No training module, torch,
CUDA, optimizer update, or file-writing code is loaded. LR fixtures disable
automatic/cyclic schedules and Muon; WD fixtures supply identical norm snapshots.
"""
import argparse
import ast
import copy
import hashlib
import itertools
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace


REL_TOL = 1e-12
ABS_TOL = 1e-15
GROUPS = ('input', 'normal', 'normal_gamma', 'noreg', 'output', 'output_noreg')


class Source:
    def __init__(self, path):
        self.path = path.resolve()
        self.data = self.path.read_bytes()
        self.tree = ast.parse(self.data, filename=str(self.path))

    def function(self, name):
        matches = [node for node in ast.walk(self.tree)
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matches) != 1:
            raise ValueError(f'{self.path}: expected one function {name}, found {len(matches)}')
        node = copy.deepcopy(matches[0])
        node.decorator_list = []
        return node

    def load(self, names, namespace):
        module = ast.Module(body=[self.function(name) for name in names], type_ignores=[])
        exec(compile(module, str(self.path), 'exec'), namespace)

    def gradient_function(self):
        # Extract the source's cap branch, two scaling statements and final LR
        # adjustment, skipping unrelated gradient-stat/RepVGG tensor operations.
        def assigns_cap(statement):
            if isinstance(statement, ast.AugAssign):
                return isinstance(statement.target, ast.Name) and statement.target.id == 'gnorm_cap'
            return isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'gnorm_cap'
                for target in statement.targets)

        candidates = []
        for node in ast.walk(self.tree):
            body = getattr(node, 'body', None)
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                if not isinstance(statement, ast.If):
                    continue
                names = {item.id for item in ast.walk(statement.test) if isinstance(item, ast.Name)}
                if names != {'use_adamw', 'use_muon'} or not any(
                        assigns_cap(item) for item in statement.body):
                    continue
                statements = [statement] + [item for item in body[index+1:] if assigns_cap(item)]
                candidates.append(statements)
        if len(candidates) != 1 or len(candidates[0]) != 4:
            raise ValueError(f'{self.path}: gradient-cap block changed; recheck extraction')
        node = ast.parse('def reference_gradient_cap():\n    pass\n').body[0]
        node.body = copy.deepcopy(candidates[0]) + [ast.Return(value=ast.Name(id='gnorm_cap', ctx=ast.Load()))]
        return ast.fix_missing_locations(node)

    def evidence(self):
        return {'path': str(self.path), 'sha256': hashlib.sha256(self.data).hexdigest()}


def close(actual, expected, context):
    if not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        raise AssertionError(f'{context}: EtaZero={actual!r}, KataGo={expected!r}')


def check_replay(eta, reference):
    count = 0
    for minimum, exponent, expansion in itertools.product(
            (1, 16, 150000, 250000), (0.5, 0.65, 1.0), (0.4, 1.0, 2.0)):
        for rows in (0, minimum-1, minimum, minimum+1, 2*minimum, 10*minimum, 100*minimum):
            replay = {'min_rows': minimum, 'taper_exponent': exponent, 'expand_per_row': expansion}
            actual = eta['desired_window'](rows, replay)
            expected = reference['compute_desired_num_rows'](
                rows, minimum, 0, exponent, expansion, None, None)
            if actual != expected:
                raise AssertionError(f'replay rows={rows}, {replay}: {actual} != {expected}')
            count += 1
    return count


def check_optimizer(eta, reference):
    samples = (0,) + tuple(value for boundary in range(250000, 2000001, 250000)
                          for value in (boundary-1, boundary)) + (4000000,)
    factor_profiles = (
        {'lookahead_alpha': 0.5, 'head_lr_factor': 0.5, 'noreg_lr_factor': 1.0,
         'input_wd_factor': 1.0, 'normal_wd_factor': 1.0},
        {'lookahead_alpha': 1.0, 'head_lr_factor': 0.3, 'noreg_lr_factor': 0.7,
         'input_wd_factor': 1.4, 'normal_wd_factor': 0.6},
    )
    count = 0
    baselines = {'input': 3.0, 'normal': 2.0}
    for kind, batch, consumed, scale, ratio, factors, warmup in itertools.product(
            ('sgd', 'adamw'), (64, 128, 256, 1024), samples, (0.25, 1.0, 4.0),
            (None, 0.25, 4.0), factor_profiles, (False, True)):
        options = {'kind': kind, 'lr_scale': scale, 'lr_warmup': warmup, **factors}
        norms = {} if ratio is None else {key: value*ratio for key, value in baselines.items()}
        train_state = {'global_step_samples': consumed,
                       **{f'modelnorm_{key}_baseline': value for key, value in baselines.items()}}
        metrics = {'sums': {f'norm_{key}_batch': value for key, value in norms.items()},
                   'weights': {f'norm_{key}_batch': 1.0 for key in norms}}
        groups = [{'group_name': name, 'lr': 0.0, 'weight_decay': 0.0} for name in GROUPS]
        reference.update(
            use_adamw=kind == 'adamw', use_muon=False, world_size=1, batch_size=batch,
            lr_scale=scale, no_lr_warmup=not warmup,
            lookahead_alpha=None if factors['lookahead_alpha'] == 1 else factors['lookahead_alpha'],
            head_lr_factor=factors['head_lr_factor'], noreg_lr_factor=factors['noreg_lr_factor'],
            input_wd_factor=factors['input_wd_factor'], normal_wd_factor=factors['normal_wd_factor'],
            normal_attn_wd_factor=1.0, train_state=train_state, running_metrics=metrics,
            raw_model=SimpleNamespace(get_norm_kind=lambda: 'fixscaleonenorm'),
            optimizer=SimpleNamespace(param_groups=groups))
        reference['update_and_return_lr_and_wd'](log_if='never')
        for group in groups:
            name = group['group_name']
            actual_lr, actual_wd = eta['group_settings'](name, options, batch, consumed, norms, baselines)
            context = f'{kind}, batch={batch}, samples={consumed}, scale={scale}, ratio={ratio}, warmup={warmup}, group={name}, factors={factors}'
            close(actual_lr, group['lr'], context + ' LR')
            close(actual_wd, group['weight_decay'], context + ' WD')
            count += 1
    return count


def check_gradient_caps(eta, reference):
    counts = {'sgd_fson_difference_cases': 0, 'adamw_equal_cases': 0}
    example = None
    for kind, batch, scale in itertools.product(
            ('sgd', 'adamw'), (64, 128, 256, 1024), (0.25, 1.0, 4.0)):
        fixture = SimpleNamespace(options={'kind': kind, 'lr_scale': scale}, batch_size=batch)
        actual = eta['gradient_cap'](fixture, 0)
        reference.update(use_adamw=kind == 'adamw', use_muon=False, batch_size=batch,
                         world_size=1, model_config={'norm_kind': 'fixscaleonenorm'},
                         gnorm_clip_scale=None, lr_scale=scale, train_state={})
        expected = reference['reference_gradient_cap']()
        if kind == 'sgd':
            # This known difference is a documented result, not an equality pass.
            close(actual/expected, 2.2, f'SGD fson cap ratio, batch={batch}, scale={scale}')
            counts['sgd_fson_difference_cases'] += 1
            if batch == 128 and scale == 1:
                example = {'batch_size': batch, 'lr_scale': scale,
                           'etazero_cap': actual, 'katago_cap': expected, 'ratio': actual/expected}
        else:
            close(actual, expected, f'AdamW cap, batch={batch}, scale={scale}')
            counts['adamw_equal_cases'] += 1
    return {**counts, 'sgd_fson_example': example}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--katago-root', type=Path, default=Path.home()/'RL/SkyZero/KataGo')
    args = parser.parse_args()
    eta_root = Path(__file__).resolve().parents[1]
    eta_shuffle = Source(eta_root/'python/etazero/shuffle.py')
    eta_optimizer = Source(eta_root/'python/etazero/optimization.py')
    kg_shuffle = Source(args.katago_root/'python/shuffle.py')
    kg_train = Source(args.katago_root/'python/train.py')
    eta = {'math': math}
    reference = {'math': math, 'logging': logging, 'lr_scale_auto_factor': lambda state: 1.0}
    eta_shuffle.load(('desired_window',), eta)
    eta_optimizer.load(('warmup_factor', 'group_settings', 'gradient_cap'), eta)
    kg_shuffle.load(('compute_desired_num_rows',), reference)
    kg_train.load(('get_effective_lr_scale', 'get_is_muon_suitable', 'get_weight_decay',
                   'update_and_return_lr_and_wd'), reference)
    module = ast.Module(body=[kg_train.gradient_function()], type_ignores=[])
    exec(compile(module, str(kg_train.path), 'exec'), reference)
    result = {
        'replay_equal_cases': check_replay(eta, reference),
        'optimizer_group_equal_cases': check_optimizer(eta, reference),
        'float_tolerance': {'relative': REL_TOL, 'absolute': ABS_TOL},
        'gradient_caps': check_gradient_caps(eta, reference),
        'sources': [source.evidence() for source in (eta_shuffle, eta_optimizer, kg_shuffle, kg_train)],
        'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
