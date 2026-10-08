"""Compare mapped plain/NBT/Transformer tensors with the fixed KataGo model.

The source model computes its real trunk, policy and value heads. Only Go input
channels, pass/score outputs and W/L/no-result -> W/D/L row selection are adapted.
This checker is a development tool, never an EtaZero runtime dependency.
"""
import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from etazero.config import load_config
from etazero.network import make_network, init_weights
from etazero import network as eta_network
from etazero.schema import GLOBALS, PLANES
from etazero.optimization import parameter_groups


def load_source():
    record = json.loads((ROOT / 'reference_sources.json').read_text())['KataGo']
    root = Path(record['root'])
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == record['commit'], (commit, record['commit'])
    files = ('python/katago/train/modelconfigs.py', 'python/katago/train/model_pytorch.py',
             'python/katago/train/trainloop_helpers.py', 'python/katago/train/fused_swiglu.py')
    hashes = {}
    for name in files:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert digest == record['sha256'][name], name
        hashes[name] = digest
    sys.path.insert(0, str(root / 'python'))
    from katago.train import modelconfigs, model_pytorch
    return modelconfigs, model_pytorch, dict(commit=commit, sha256=hashes)


def parameter_mapping(eta, source, architecture):
    """Each row is (Eta parameter, source parameter, Eta index, source index)."""
    pairs = []
    def add(e, s, ei=Ellipsis, si=Ellipsis):
        assert e[ei].numel() == s[si].numel()
        pairs.append((e, s, ei, si))
    def norm(e, s):
        if hasattr(e,'weight'):
            add(e.weight, s.gamma)
        else:
            assert s.gamma is None
        add(e.bias, s.beta)
    def conv(e, s):
        norm(e.norm, s.norm)
        if e.has_global_pool:
            pool = s.convpool
            add(e.conv.conv.weight, pool.conv1r.weight)
            add(e.conv.conv_global.weight, pool.conv1g.weight)
            norm(e.conv.norm_global, pool.normg)
            add(e.conv.linear.weight, pool.linear_g.weight)
        else:
            add(e.conv.weight, s.conv.weight)
    def residual(e, s):
        conv(e.pre, s.normactconv1)
        conv(e.post, s.normactconv2)
    add(eta.stem.weight, source.conv_spatial.weight, si=(slice(None), slice(0, 5)))
    add(eta.linear_global.weight, source.linear_global.weight, si=(slice(None), slice(0, len(GLOBALS))))
    for e, s in zip(eta.blocks, source.blocks, strict=True):
        if architecture == 'plain':
            residual(e, s)
        else:
            conv(e.pre, s.normactconvp)
            for inner, ref in zip(e.inner, s.blockstack, strict=True):
                if architecture != 'transformer':
                    residual(inner, ref)
                elif hasattr(inner,'q_proj'):
                    add(inner.norm.weight,ref.norm1.weight)
                    for name in ('q_proj','k_proj','v_proj','out_proj'):
                        add(getattr(inner,name).weight,getattr(ref,name).weight)
                    torch.testing.assert_close(inner.cos,ref.cos_cached,rtol=0,atol=0)
                    torch.testing.assert_close(inner.sin,ref.sin_cached,rtol=0,atol=0)
                else:
                    add(inner.norm.weight,ref.norm.weight)
                    for name,source_name in (('input','ffn_linear1'),('gate','ffn_linear_gate'),('output','ffn_linear2')):
                        add(getattr(inner,name).weight,getattr(ref,source_name).weight)
            conv(e.post, s.normactconvq)
    norm(eta.trunk_norm, source.norm_trunkfinal)
    for name in ('running_mean', 'running_std'):
        if hasattr(eta.trunk_norm,name):
            getattr(source.norm_trunkfinal, name).copy_(getattr(eta.trunk_norm, name))
    ep, sp = eta.policy_head, source.policy_head
    for e, s in ((ep.local.weight, sp.conv1p.weight), (ep.global_conv.weight, sp.conv1g.weight),
                 (ep.global_bias.bias, sp.biasg.beta), (ep.linear.weight, sp.linear_g.weight),
                 (ep.bias.bias, sp.bias2.beta)):
        add(e, s)
    add(ep.out.weight,sp.conv2p.weight,si=slice(0,ep.out.weight.shape[0]))
    ev, sv = eta.value_head, source.value_head
    for e, s in ((ev.conv.weight, sv.conv1.weight), (ev.bias.bias, sv.bias1.beta),
                 (ev.hidden.weight, sv.linear2.weight), (ev.hidden.bias, sv.linear2.bias)):
        add(e, s)
    # Source order is win/loss/no-result; Eta uses win/draw/loss. Source Go
    # no-result logits are repurposed as the draw logits in this WDL adaptation.
    for e, s in ((ev.out.weight, sv.linear_valuehead.weight), (ev.out.bias, sv.linear_valuehead.bias)):
        add(e, s, ei=slice(0, 3), si=[0, 2, 1])
    for e, s in ((ev.out.weight, sv.linear_miscvaluehead.weight), (ev.out.bias, sv.linear_miscvaluehead.bias)):
        add(e, s, ei=slice(3, 9), si=[4, 6, 5, 7, 9, 8])
    for e, s in ((ev.out.weight, sv.linear_moremiscvaluehead.weight), (ev.out.bias, sv.linear_moremiscvaluehead.bias)):
        add(e, s, ei=slice(9, 13), si=[2, 4, 3, 0])
    assert {id(p[0]) for p in pairs} == {id(p) for p in eta.parameters()}
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
        for e, s, ei, si in pairs:
            s[si] = e[ei].reshape(s[si].shape)
    return pairs


def source_outputs(source, obs, globals):
    mask = obs[:, :1]
    # Only the input projections are adapted to EtaZero's feature contract.
    p, v, misc, more, *_ = source(obs * mask, globals)[0]
    # v17 Q adds pure W-L as output six; Go score Q (output seven) is omitted.
    return (p[:, :7 if p.shape[1]==8 else 6, :-1] * mask.flatten(2), v[:, [0, 2, 1]],
            torch.stack((misc[:, [4, 6, 5]], misc[:, [7, 9, 8]], more[:, [2, 4, 3]]), 1), more[:, 0])


def check_architecture(architecture, device, mc, mp, amp='off', compiled=False, predict_q_values=False):
    preset = {'plain':'b10c128-fson-mish','nbt':'b5c192nbt-fson-mish','transformer':'b5c192h3nbttfrs'}[architecture]
    config = load_config(ROOT / 'tests/fixtures/configs/smoke_test', run_dir='/tmp/etazero_reference_sample')
    config['network'].update(architecture=architecture, canvas=15,
                             channels=128 if architecture == 'plain' else 192,
                             blocks=10 if architecture == 'plain' else 5)
    ref_config = copy.deepcopy(mc.config_of_name[preset])
    if predict_q_values:
        assert architecture=='transformer'
        config['network']['predict_q_values']=True
        ref_config['predict_q_values']=True
    torch.manual_seed(851)
    eta_inits, source_inits = {}, {}
    original_eta, original_source = eta_network.init_weights, mp.init_weights
    def record_eta(tensor, scale=1.0, identity=False, fan_tensor=None, activation='mish'):
        eta_inits[id(tensor)] = (scale, 'identity' if identity else activation)
        return original_eta(tensor, scale, identity, fan_tensor, activation)
    def record_source(tensor, activation, scale=1.0, fan_tensor=None):
        source_inits[id(tensor)] = (scale, activation)
        return original_source(tensor, activation, scale, fan_tensor)
    try:
        eta_network.init_weights, mp.init_weights = record_eta, record_source
        eta = make_network(config).to(device)
        source = mp.Model(ref_config, pos_len=15).to(device)
        # Match EtaZero's feature contract at the input projections.
        # Keeping Go's zero-filled 22/19 widths picks different AMP GEMM/conv
        # kernels and rounding despite mathematically identical projections.
        source.conv_spatial = torch.nn.Conv2d(len(PLANES),config['network']['channels'],3,padding=1,bias=False).to(device)
        source.linear_global = torch.nn.Linear(len(GLOBALS),config['network']['channels'],bias=False).to(device)
        source.initialize()
    finally:
        eta_network.init_weights, mp.init_weights = original_eta, original_source
    if architecture == 'transformer':
        # Fixed source provides these public backend toggles. Compare its real
        # NCHW SDPA path with identical masks and AMP fusion, without rewriting
        # attention, RoPE, RMSNorm or SwiGLU functions.
        source.transformer_seq_layout=False
        source.use_flex_attention=False
        source.transformer_skip_redundant_masks=False
        for block in source.blocks:
            block.normactconvp.norm.skip_mask=False
            block.normactconvq.norm.skip_mask=False
        for block in eta.blocks:
            assert block.post.conv.weight.count_nonzero()==0
            # Exercise the whole transformer branch, rather than trivially
            # comparing its zero-initialized fixup residual and zero gradients.
            original_eta(block.post.conv.weight,0.3,activation='relu')
    pairs = parameter_mapping(eta, source, architecture)
    source_roles = {}
    reg = {}
    source.add_reg_dict(reg)
    for name, parameters in reg.items():
        for p in parameters:
            assert id(p) not in source_roles
            source_roles[id(p)] = name
    eta_roles = {id(p): g['group_name'] for g in parameter_groups(eta) for p in g['params']}
    for e, s, _, _ in pairs:
        assert eta_roles[id(e)] == source_roles[id(s)], (eta_roles[id(e)], source_roles[id(s)])
        assert eta_inits.get(id(e)) == source_inits.get(id(s)), (eta_inits.get(id(e)), source_inits.get(id(s)))
    max_forward, max_gradient = 0.0, 0.0
    cases = []
    execution_gaps = []
    forward = eta.forward_all
    reference_forward = source
    if compiled:
        from torch._inductor import config as compiler_config
        compiler_config.compile_threads = 1
        forward = torch.compile(forward,fullgraph=True,dynamic=False,
                                options={'layout_optimization':False} if architecture=='transformer' else None)
        reference_forward = torch.compile(source,fullgraph=True,dynamic=False,
                                          options={'layout_optimization':False} if architecture=='transformer' else None)
    # First two calls exercise real masked BN training and its EMA; eval consumes
    # these learned buffers. Both batches mix an 11x11 board and a 15x15 board.
    for training in (True, True, False):
        eta.train(training); source.train(training)
        obs = torch.randn(2, 5, 15, 15, device=device)
        obs[:, 0].fill_(1); obs[0, 0, 11:].zero_(); obs[0, 0, :, 11:].zero_()
        obs.requires_grad_()
        globals = torch.randn(2, len(GLOBALS), device=device, requires_grad=True)
        ref_obs = obs.detach().clone().requires_grad_()
        ref_globals = globals.detach().clone().requires_grad_()
        context = (torch.autocast('cuda',dtype=torch.float16 if amp=='float16' else torch.bfloat16,
                                  enabled=training) if amp!='off' else contextlib.nullcontext())
        with context:
            actual = forward(obs, globals)
            expected = source_outputs(reference_forward, ref_obs, ref_globals)
            if compiled and architecture=='transformer':
                with torch.no_grad():
                    eager_actual = eta.forward_all(obs, globals)
                    eager_expected = source_outputs(source, ref_obs, ref_globals)
                execution_gaps.append(dict(
                    eta_compiled_vs_eager_max_abs=max((a-b).abs().max().item() for a,b in zip(actual,eager_actual)),
                    source_compiled_vs_eager_max_abs=max((a-b).abs().max().item() for a,b in zip(expected,eager_expected))))
        f_rtol,f_atol = (3e-5,3e-6) if amp=='off' else (3e-3,3e-4) if amp=='float16' else (3e-2,3e-3)
        g_rtol,g_atol = (8e-5,2e-5) if amp=='off' else (5e-3,3e-3) if amp=='float16' else (5e-2,3e-2)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=f_rtol, atol=f_atol)
            max_forward = max(max_forward, (a-b).abs().max().item())
        eta.zero_grad(set_to_none=True); source.zero_grad(set_to_none=True)
        cotangents = [torch.randn_like(a) for a in actual]
        sum((a*c).sum() for a,c in zip(actual,cotangents)).backward()
        sum((b*c).sum() for b,c in zip(expected,cotangents)).backward()
        # Plane 0 is the discrete, binary board geometry, not a trainable input.
        # The source's -5000 logit padding is absent in Eta (loss/search mask it).
        # Compare gradients of all features, globals and trainable parameters.
        gradient_pairs = [(obs.grad[:,1:], ref_obs.grad[:,1:]), (globals.grad, ref_globals.grad)]
        gradient_pairs += [(e.grad[ei], s.grad[si].reshape(e.grad[ei].shape)) for e,s,ei,si in pairs]
        for a, b in gradient_pairs:
            torch.testing.assert_close(a, b, rtol=g_rtol, atol=g_atol)
            max_gradient = max(max_gradient, (a-b).abs().max().item())
        assert obs.grad[0, 1:, 11:].count_nonzero() == 0
        assert obs.grad[0, 1:, :, 11:].count_nonzero() == 0
        for name in ('running_mean', 'running_std'):
            if hasattr(eta.trunk_norm,name):
                torch.testing.assert_close(getattr(eta.trunk_norm,name), getattr(source.norm_trunkfinal,name), rtol=1e-6, atol=1e-7)
        if architecture=='transformer':
            assert all(p.grad.abs().sum()>0 for block in eta.blocks for inner in block.inner for p in inner.parameters())
        cases.append(dict(training=training, mapped_gradient_tensors=len(gradient_pairs)))
    return dict(preset=preset, parameters=sum(p.numel() for p in eta.parameters()),
                source_config=ref_config, max_forward_abs=max_forward, max_gradient_abs=max_gradient,
                mapped_parameter_roles=len(pairs), amp=amp, compiled=compiled,
                oracle_execution='compiled' if compiled else 'eager', source_input_projection_shapes=[len(PLANES),len(GLOBALS)],
                predict_q_values=predict_q_values,
                execution_gaps=execution_gaps,
                tolerances=dict(forward=[f_rtol,f_atol],gradient=[g_rtol,g_atol]),cases=cases)


def check_initialization(mp):
    cases = 0
    # Replay the source initializer with identical seeds, actual parameter shapes,
    # and all scales used by these fson models (including fan_tensor biases).
    for activation in ('mish','relu'):
      for shape in ((128,128,3,3), (32,128,3,3), (96,96), (6,32,1,1), (13,80)):
        for scale in (1.0, 0.8, 0.6, 0.6**0.5, 0.3, 0.2, 0.0, (1/5**0.5)**(1/3)):
            for identity in (False, True):
                a, b = torch.empty(shape), torch.empty(shape)
                torch.manual_seed(103); init_weights(a, scale, identity, activation=activation)
                torch.manual_seed(103); mp.init_weights(b, 'identity' if identity else activation, scale=scale)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                cases += 1
        a, b = torch.empty(shape[0]), torch.empty(shape[0])
        fan = torch.empty(shape)
        torch.manual_seed(104); init_weights(a, 0.2, fan_tensor=fan, activation=activation)
        torch.manual_seed(104); mp.init_weights(b, activation, scale=0.2, fan_tensor=fan)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        cases += 1
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--architecture',choices=('plain','nbt','transformer'),nargs='+',default=['plain','nbt','transformer'])
    parser.add_argument('--amp',choices=('off','float16','bfloat16'),default='off')
    parser.add_argument('--compile',action='store_true',help='Compare Eta and original source under the same full-graph compilation settings')
    parser.add_argument('--predict-q-values',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert args.amp=='off' or args.device=='cuda'
    if args.device == 'cuda':
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    mc, mp, evidence = load_source()
    result = dict(status='verified', source=evidence, device=args.device,
                  initialization_cases=check_initialization(mp),
                  architectures={kind: check_architecture(kind,args.device,mc,mp,args.amp,args.compile,args.predict_q_values) for kind in args.architecture})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='architectures'}))
    for kind,value in result['architectures'].items():
        print(kind, {k:v for k,v in value.items() if k not in ('source_config','cases')})


if __name__ == '__main__':
    main()
