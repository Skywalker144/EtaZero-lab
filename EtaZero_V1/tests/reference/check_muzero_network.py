"""Compare normalization, gradients and dense trunks to pinned MuZero_V2 bodies."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from dataclasses import replace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from etazero.muzero.network import (normalize_hidden_state, scale_gradient, NetworkConfig,
                                   Representation, Dynamics, Prediction)
from etazero.muzero.resnet import ResBlock


def check_resnet(namespace):
    torch.manual_seed(991)
    mask = torch.ones(2, 1, 6, 6); mask[0, :, 5] = 0; mask[0, :, :, 5] = 0
    block = ResBlock(24)
    reference = namespace['ResBlock'](24); reference.load_state_dict(block.state_dict())
    x = (torch.randn(2, 24, 6, 6) * mask).requires_grad_()
    y = x.detach().clone().requires_grad_()
    actual, expected = block(x, mask), reference(y, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward(); (expected * weights).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for p, q in zip(block.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    c = NetworkConfig(6, 16, 24, 2, 32, 2, 48, 0, architecture='resnet')
    representation, dynamics = Representation(c), Dynamics(c)
    source_rep = namespace['RepresentationNet'](5, 24, 2, 16)
    source_dyn = namespace['DynamicsNet'](6, 32, 2, 16)
    for ours, source in ((representation, source_rep), (dynamics, source_dyn)):
        # EtaZero's six global features are a separate linear input adaptation.
        state = {k.replace('stem.weight', 'start_layer.weight').replace('stem_act.norm.', 'norm.'): v
                 for k, v in ours.state_dict().items() if k != 'linear_global.weight'}
        source.load_state_dict(state)
    obs = torch.randn(2, 5, 6, 6); obs[:, :1] = mask
    actual = representation(obs, torch.zeros(2, 6)); expected = source_rep(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for step in range(5):
        actions = torch.tensor([step, step+6])
        actual = dynamics(actual, actions); expected = source_dyn(expected, actions)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward(); (expected * weights).sum().backward()
    for ours, source in ((representation, source_rep), (dynamics, source_dyn)):
        reference_parameters = dict(source.named_parameters())
        for name, parameter in ours.named_parameters():
            if name == 'linear_global.weight':
                continue
            key = name.replace('stem.weight', 'start_layer.weight').replace('stem_act.norm.', 'norm.')
            torch.testing.assert_close(parameter.grad, reference_parameters[key].grad, rtol=0, atol=0)
    for blocks in (0, 2):
        prediction = Prediction(replace(c, prediction_blocks=blocks))
        source = namespace['ResidualTrunk'](16, 48, blocks)
        source.load_state_dict({**{'projection.'+k: v for k, v in prediction.projection.state_dict().items()},
                                **prediction.trunk.state_dict()})
        x = actual.detach()[:, :-1]
        torch.testing.assert_close(prediction.trunk(prediction.projection(x * mask), mask), source(x, mask),
                                   rtol=0, atol=0)
    print('MuZero_V2: dense block, representation, five recurrent steps and prediction trunk parity passed')


def main():
    reference = json.loads((ROOT / 'reference_sources.json').read_text())['MuZero_V2']
    relative = 'python/muzero/network.py'
    source = (Path(reference['root']) / relative).read_bytes()
    if hashlib.sha256(source).hexdigest() != reference['sha256'][relative]:
        raise ValueError('MuZero_V2 network source differs from the pinned reference')
    names = {'normalize_hidden_state', 'scale_gradient', 'MaskedNorm', 'ResBlock',
             'channel_projection', 'ResidualTrunk', 'RepresentationNet', 'DynamicsNet'}
    module = ast.parse(source)
    selected = ast.Module(body=[n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names], type_ignores=[])
    namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional, 'Plane': SimpleNamespace(ON_BOARD=0)}
    exec(compile(selected, str(Path(reference['root']) / relative), 'exec'), namespace)
    torch.set_num_threads(1)
    torch.manual_seed(151)
    cases = 0
    for channels in (1, 7, 24):
        for size in (5, 6):
            for constant in (False, True):
                mask = torch.zeros(2, 1, 6, 6); mask[0, :, :size, :size] = 1; mask[1] = 1
                base = torch.full((2, channels, 6, 6), 2.) if constant else torch.randn(2, channels, 6, 6)
                for symmetry in range(8):
                    from etazero.symmetry import apply_symmetry
                    current_mask = apply_symmetry(mask, symmetry)
                    x = base.clone().requires_grad_(); y = base.clone().requires_grad_()
                    actual = normalize_hidden_state(x, current_mask)
                    expected = namespace['normalize_hidden_state'](y, current_mask)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    weights = torch.randn_like(actual)
                    (actual * weights).sum().backward(); (expected * weights).sum().backward()
                    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                    cases += 1
    for factor in (0., .5, 1., .2):
        x = torch.randn(13, requires_grad=True); y = x.detach().clone().requires_grad_()
        actual = scale_gradient(x, factor); expected = namespace['scale_gradient'](y, factor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward(); expected.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    print(f'MuZero_V2: {cases} normalization forward/backward cases and 4 gradient scales passed')
    check_resnet(namespace)


if __name__ == '__main__':
    main()
