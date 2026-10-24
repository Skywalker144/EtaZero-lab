"""Compare every handwritten architecture and generated name with fixed KataGo.

Use --all-configs for the exhaustive name/config comparison. Architecture
checks use the full original source model, adapted at its game input
projection and four-outcome output dimensions. No reduced-width or reduced-depth substitutes are constructed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from etazero.katago_configs import BASE_NAMES, resolve
from etazero.katago_network import KataGoNet
from etazero.schema import GLOBALS, PLANES


def load_source():
    record = json.loads((ROOT / 'reference_sources.json').read_text())['KataGo']
    root = Path(record['root'])
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != record['commit']:
        raise ValueError('KataGo reference revision differs')
    for name in ('modelconfigs.py', 'model_pytorch.py', 'fused_swiglu.py', 'fused_rope.py'):
        path = 'python/katago/train/' + name
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != record['sha256'][path]:
            raise ValueError('KataGo reference file differs: ' + path)
    sys.path.insert(0, str(root / 'python'))
    from katago.train import modelconfigs, model_pytorch
    return modelconfigs, model_pytorch, commit


def original_model(mp, config, canvas, device):
    model = mp.Model(copy.deepcopy(config), canvas)
    kernel = 1 if config['initial_conv_1x1'] else 3
    model.conv_spatial = torch.nn.Conv2d(len(PLANES), config['trunk_num_channels'], kernel,
                                        padding=kernel//2, bias=False)
    model.linear_global = torch.nn.Linear(len(GLOBALS), config['trunk_num_channels'], bias=False)
    # Accepted game-output adaptation; all trunk tensors still come from source.
    heads = (model.value_head,) + ((model.intermediate_value_head,) if model.has_intermediate_head else ())
    for head in heads:
        hidden = head.linear_valuehead.in_features
        head.linear_valuehead = torch.nn.Linear(hidden, 4)
        head.linear_miscvaluehead = torch.nn.Linear(hidden, 12)
        head.linear_moremiscvaluehead = torch.nn.Linear(hidden, 9)
    if device == 'cpu':
        model.use_flex_attention = False
    model.initialize()
    model.set_brenorm_params(0.001, 1.0, 0.0)
    return model.to(device)


def check_architecture(name, mc, mp, device, backward):
    torch.manual_seed(853)
    config = resolve(name)
    assert config == mc.config_of_name[name]
    eta = KataGoNet(name, 5).to(device)
    if device == 'cpu':
        eta.network.use_flex_attention = False
    torch.manual_seed(853)
    ref = original_model(mp, eta.source_config, 5, device)
    assert {key: value.shape for key, value in eta.network.state_dict().items()} == {
        key: value.shape for key, value in ref.state_dict().items()}
    for key, parameter in eta.network.state_dict().items():
        torch.testing.assert_close(parameter, ref.state_dict()[key], rtol=0, atol=0, msg=key)
    # Open zero-initialized outer projections so inner blocks participate in
    # both the output and gradient comparisons, including register readout.
    with torch.no_grad():
        for key, parameter in eta.network.named_parameters():
            if parameter.ndim >= 2 and ('normactconvq.conv.weight' in key or
                                       'normactconv2.conv.weight' in key or 'rw_reg_readout_proj2.weight' in key):
                parameter.add_(torch.randn_like(parameter) * 0.002)
    ref.load_state_dict(eta.network.state_dict(), strict=True)
    eta_roles = {id(p): group['group_name'] for group in eta.parameter_groups() for p in group['params']}
    ref_groups = {}; ref.add_reg_dict(ref_groups)
    ref_roles = {id(p): group for group, params in ref_groups.items() for p in params}
    for (key, param), (ref_key, ref_param) in zip(eta.network.named_parameters(), ref.named_parameters(), strict=True):
        assert key == ref_key and eta_roles[id(param)] == ref_roles[id(ref_param)]
    obs = torch.randn(2, len(PLANES), 5, 5, device=device)
    obs[:, 0] = 1; obs[0, 0, 4] = 0
    globals = torch.randn(2, len(GLOBALS), device=device)
    eta_obs = obs.clone().requires_grad_(backward)
    ref_obs = obs.clone().requires_grad_(backward)
    metadata = torch.randn(2, 192, device=device) if ref.metadata_encoder is not None else None
    actual = eta.select_outputs(eta.source_outputs(eta_obs, globals, metadata)[0], eta_obs[:,0:1])
    outputs = ref(ref_obs * ref_obs[:, 0:1], globals, input_meta=metadata)[0]
    p, v, misc, more = outputs[:4]
    nr = torch.full_like(v[:, 0], -10000)
    expected = (p[:, :6, :-1] * ref_obs[:, 0:1].flatten(2),
                torch.stack((v[:,0],v[:,3],v[:,1],nr),1),
                torch.stack((torch.stack((misc[:,4],misc[:,10],misc[:,5],nr),1),
                             torch.stack((misc[:,7],misc[:,11],misc[:,8],nr),1),
                             torch.stack((more[:,2],more[:,8],more[:,3],nr),1)),1), more[:,0])
    max_forward = max((a-b).abs().max().item() for a,b in zip(actual, expected))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
    max_gradient = 0.0
    if backward:
        sum(x.square().mean() for x in actual).backward()
        sum(x.square().mean() for x in expected).backward()
        torch.testing.assert_close(eta_obs.grad, ref_obs.grad, rtol=8e-5, atol=2e-5)
        for (key, param), (ref_key, ref_param) in zip(eta.network.named_parameters(), ref.named_parameters(), strict=True):
            assert (param.grad is None) == (ref_param.grad is None), key
            if param.grad is not None:
                max_gradient = max(max_gradient, (param.grad-ref_param.grad).abs().max().item())
                torch.testing.assert_close(param.grad, ref_param.grad, rtol=8e-5, atol=2e-5, msg=key)
    return {'parameters': sum(p.numel() for p in eta.parameters()), 'max_forward_abs': max_forward,
            'max_gradient_abs': max_gradient, 'backward': backward}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--architecture', nargs='+')
    parser.add_argument('--all-configs', action='store_true')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == 'cuda':
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    mc, mp, commit = load_source()
    assert set(BASE_NAMES) == set(mc.base_config_of_name)
    configurations = 0
    if args.all_configs:
        for name, expected in mc.config_of_name.items():
            assert resolve(name) == expected, name
            configurations += 1
            if configurations % 100000 == 0:
                print('configuration comparison:', configurations, flush=True)
    results = {}
    for name in args.architecture or BASE_NAMES:
        started = time.monotonic()
        results[name] = check_architecture(name, mc, mp, args.device, args.backward)
        print(name, results[name], f'{time.monotonic()-started:.2f}s', flush=True)
    payload = {'status': 'verified', 'source_commit': commit, 'device': args.device,
               'configurations_checked': configurations, 'architectures': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
