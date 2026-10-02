"""Compare latent normalization and gradient scaling to pinned MuZero_V2 bodies."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from etazero.muzero.network import normalize_hidden_state, scale_gradient


def main():
    reference = json.loads((ROOT / 'reference_sources.json').read_text())['MuZero_V2']
    relative = 'python/muzero/network.py'
    source = (Path(reference['root']) / relative).read_bytes()
    if hashlib.sha256(source).hexdigest() != reference['sha256'][relative]:
        raise ValueError('MuZero_V2 network source differs from the pinned reference')
    names = {'normalize_hidden_state', 'scale_gradient'}
    module = ast.parse(source)
    selected = ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = {'torch': torch}
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


if __name__ == '__main__':
    main()
