"""Check MuZero_V2 parity and the intentional MiniZero constant-span adaptation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from etazero.models.muzero.network import normalize_hidden_state, scale_gradient


def main():
    reference = json.loads((ROOT / 'reference_sources.json').read_text())['MuZero_V2']
    relative = 'python/muzero/network.py'
    source = (Path(reference['root']) / relative).read_bytes()
    if hashlib.sha256(source).hexdigest() != reference['sha256'][relative]:
        raise ValueError('MuZero_V2 network source differs from the pinned reference')
    names = {'normalize_hidden_state', 'scale_gradient'}
    module = ast.parse(source)
    selected = ast.Module(body=[n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names], type_ignores=[])
    namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional, 'Plane': SimpleNamespace(ON_BOARD=0)}
    exec(compile(selected, str(Path(reference['root']) / relative), 'exec'), namespace)
    torch.set_num_threads(1)
    torch.manual_seed(151)
    normal_cases = constant_cases = 0
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
                    if constant:
                        # MiniZero's zero-span denominator is 1e-5, versus V2's 1.
                        torch.testing.assert_close(x.grad * 1e-5, y.grad, rtol=1e-5, atol=1e-6)
                        constant_cases += 1
                    else:
                        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                        normal_cases += 1
    for factor in (0., .5, 1., .2):
        x = torch.randn(13, requires_grad=True); y = x.detach().clone().requires_grad_()
        actual = scale_gradient(x, factor); expected = namespace['scale_gradient'](y, factor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward(); expected.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    print(f'MuZero_V2: {normal_cases} normal-span forward/backward parity cases, '
          f'{constant_cases} constant forward parity / MiniZero gradient adaptations, '
          'and 4 gradient scales passed')


if __name__ == '__main__':
    main()
