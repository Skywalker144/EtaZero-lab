"""Host-CUDA tests for named KataGo model training, export and native inference."""
import json
import os
import subprocess
import numpy as np
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import ROOT, load_config, validate
from etazero.katago_configs import BASE_NAMES, resolve
from etazero.katago_network import KataGoNet, inference_network
from etazero.runtime import run_training
from etazero.storage import sha256
from etazero.training import load_checkpoint

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA required')


_HALF_NAMES = ('b1c6nbt', 'b20c384bt', 'b15c384lbt', 'b10c480nb3t', 'b18c384dnbt1',
               'b7c96h6kv3qk32v16tflrs', 'b5c48h3tfr', 'b4c256h4nbttflrs',
               'b10c512h8nbt3tflrs', 'b6c384h6nbttflrs-qkn-ireg16',
               'b2b10c96h3tfrs', 'b5c384h6nbttfgabs', 'b5c384h6nbttftabs',
               'b5c384h6nbttflrtab2cheaps', 'b5c384h6nbttflrtab2cheaps-fson-silu',
               'b6c384h6nbttflrs-dwc')


@pytest.mark.parametrize('name,precision', [(name, 'float32') for name in BASE_NAMES] +
                                         [(name, 'float16') for name in _HALF_NAMES])
def test_every_handwritten_architecture_native_cuda_graph(tmp_path, name, precision):
    torch.set_num_threads(2)
    torch.manual_seed(853)
    # Match LibTorch's default execution settings explicitly. Deep trunks can
    # amplify TF32 convolution/JIT fusion rounding. This native test is stricter
    # than inference_probe's 3e-3 guard; exact architecture/source arithmetic is
    # checked separately without TF32 in check_katago_architectures.py.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model = KataGoNet(name, 6).cuda().eval()
    with torch.no_grad():
        for key, parameter in model.network.named_parameters():
            if parameter.ndim >= 2 and ('normactconvq.conv.weight' in key or 'rw_reg_readout_proj2.weight' in key):
                parameter.add_(torch.randn_like(parameter) * 0.002)
    scripted = torch.jit.script(inference_network(model))
    path = tmp_path / 'model.pt'; scripted.save(str(path))
    try:
        result = subprocess.run([str(ROOT / 'build/inference_probe'), str(path), 'cuda:0', precision],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        actual = json.loads(result.stdout)
        spatial = torch.zeros(5, 5, 6, 6, device='cuda')
        globals = torch.zeros(5, 8, device='cuda')
        for i in range(5):
            size = 5 + i % 2; spatial[i, 0, :size, :size] = 1
            spatial[i, 1, 0, i] = 1; spatial[i, 2, 1, i] = 1
            globals[i, :6] = torch.tensor([i % 2, i % 3 == 0, -1 if i % 3 == 0 else 0,
                                           i % 3 == 0, i % 2, .25 * i], device='cuda')
        with torch.inference_mode(), torch.autocast('cuda', enabled=precision=='float16', dtype=torch.float16):
            for turn, count in enumerate((1, 3, 5, 2, 1, 4, 3)):
                indices = [(turn + i) % 5 for i in range(count)]
                # Native Graph capture rounds to its pre-captured batch size,
                # filling spare rows with the first valid request.
                bucket = next(size for size in (1, 2, 4, 5) if size >= count)
                indices += [indices[0]] * (bucket - count)
                for _ in range(3):
                    scripted(spatial[indices], globals[indices])
                logits, values, optimistic, error = scripted(spatial[indices], globals[indices])
                expected = torch.cat((logits.float(), values.float().softmax(1), optimistic.float(), error.float()[:, None]), 1)
                rtol, atol = (3e-3, 3e-3) if precision == 'float16' else (1e-3, 1e-4)
                np.testing.assert_allclose(actual[turn], expected[:count].cpu().numpy(), rtol=rtol, atol=atol)
    finally:
        path.unlink()


@pytest.mark.parametrize('name', ['b7c96h6kv3qk32v16tflrs-fson-silu',
                                 'b6c384h6nbttflrs-qkn-ireg16-fson-silu',
                                 'b5c384h6nbttflrtab2cheaps',
                                 'b5c384h6nbttflrtab2cheaps-fson-silu'])
def test_export_moves_from_cuda_to_cpu_and_preserves_dynamic_batches(tmp_path, name):
    model = KataGoNet(name, 6).cuda().eval()
    exported = torch.jit.script(inference_network(model))
    path = tmp_path / 'model.pt'; exported.save(str(path))
    restored = torch.jit.load(str(path), map_location='cpu')
    model.cpu()
    with torch.inference_mode():
        for batch in (1, 3):
            spatial = torch.randn(batch, 5, 6, 6); spatial[:, 0] = 1
            spatial[0, 0, 5] = 0; globals = torch.randn(batch, 8)
            p, v, _, raw = model.forward_all(spatial, globals)
            expected = p[:, 0], v, p[:, 5], .5 * torch.nn.functional.softplus(raw * .5)
            for actual, reference in zip(restored(spatial, globals), expected):
                torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_tab2_export_preserves_eager_amp_and_training_state(dtype):
    torch.set_num_threads(2)
    torch.manual_seed(853)
    model = KataGoNet('b5c384h6nbttflrtab2cheaps', 6).cuda().eval()
    with torch.no_grad():
        for name, parameter in model.network.named_parameters():
            if 'normactconvq.conv.weight' in name:
                parameter.add_(torch.randn_like(parameter) * .002)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    exported = torch.jit.script(inference_network(model))
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert not any('real_kernel' in name or 'imag_kernel' in name for name, _ in exported.named_parameters())
    # Eager source heads stay FP32; the inference graph uses AMP heads. Match
    # the source oracle's existing BF16 tolerance for that precision difference.
    rtol = 3e-3 if dtype == torch.float16 else 3e-2
    with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
        for batch in (1, 3):
            obs = torch.randn(batch, 5, 6, 6, device='cuda'); obs[:, 0] = 1
            obs[0, 0, 5] = 0
            globals = torch.randn(batch, 8, device='cuda')
            policy, value, _, raw = model.forward_all(obs, globals)
            expected = policy[:, 0], value, policy[:, 5], .5 * torch.nn.functional.softplus(raw.float() * .5)
            for _ in range(3):
                exported(obs, globals)
            for actual, reference in zip(exported(obs, globals), expected):
                torch.testing.assert_close(actual.float(), reference.float(), rtol=rtol, atol=3e-3)


@pytest.mark.parametrize('name,amp,compiled,q,kind', [
    ('b2c16-fson-mish-rvgl-bnh', 'float16', False, False, None),
    ('b4c256h4nbttflrs-fson-silu-rsnh', 'bfloat16', True, False, None),
    ('b2c16-brng-gelu-tcg-trbnr-ih-meta', 'bfloat16', True, False, None),
    ('b7c96h6kv3qk32v16tflrs-fson-silu', 'bfloat16', False, True, 'muon'),
    ('b9c96h3tgabs-fson-silu', 'bfloat16', True, False, 'adamw'),
    ('b5c384h6nbttflrtab2cheaps-fson-silu', 'bfloat16', True, False, 'muon'),
])
def test_named_architecture_real_cuda_pipeline_and_resume(tmp_path, name, amp, compiled, q, kind):
    assert torch.cuda.is_available()
    torch.set_num_threads(2); torch._dynamo.reset()
    config = load_config(CONFIGS / 'smoke_test')
    preset = resolve(name)
    config['network'].update(architecture=name, channels=preset['trunk_num_channels'],
                             blocks=len(preset['block_kind']), predict_q_values=q)
    if kind:
        config['optimizer']['kind'] = kind
    config['training'].update(amp=amp, compile=compiled, train_steps=2)
    config['devices'].update(train='cuda:0', selfplay='cuda:0')
    if preset['norm_kind'] in ('brenorm', 'fixbrenorm'):
        config['brenorm'] = dict(avg_momentum=.01, target_rmax=2., target_dmax=1., adjustment_scale=10.)
    validate(config)
    root = tmp_path / 'run'
    state = run_training(root, config, ROOT / 'build/etazero', max_iteration=2)
    assert state['iteration'] == 3 and state['checkpoint']['optimizer_steps'] > 0
    saved = load_checkpoint(root, state['checkpoint'], config)
    model = KataGoNet(name, 6, predict_q_values=q).cuda().eval(); model.load_state_dict(saved['model'])
    path = root / state['model']['path']; digest = sha256(path)
    scripted = torch.jit.load(str(path), map_location='cuda:0')
    spatial = torch.zeros(2, 5, 6, 6, device='cuda'); spatial[:, 0, :5, :5] = 1
    globals = torch.zeros(2, 8, device='cuda')
    with torch.inference_mode():
        for actual in scripted(spatial, globals):
            assert torch.isfinite(actual).all()
    resumed = run_training(root, config, ROOT / 'build/etazero', max_iteration=3)
    assert resumed['iteration'] == 4 and resumed['run_id'] == state['run_id']
    assert resumed['checkpoint']['total_steps'] > state['checkpoint']['total_steps']
    assert sha256(path) == digest
    events = [json.loads(line) for line in (root / 'logs/events.jsonl').read_text().splitlines()]
    updates = [event for event in events if event['event'] == 'update']
    assert updates and all(np.isfinite(event['loss']) for event in updates)
    # Initial FP16 loss-scale overflow is a legitimate GradScaler event. Check
    # that skipped updates and the restored scale/sample clocks are accounted.
    assert resumed['checkpoint']['optimizer_steps'] == resumed['checkpoint']['total_steps'] - sum(
        event['amp_skipped_steps'] for event in updates)
    if preset.get('has_intermediate_head'):
        for key in ('intermediate_policy_head.conv2p.weight', 'intermediate_value_head.linear_valuehead.weight'):
            assert 'network.' + key in saved['model']
    torch._dynamo.reset()
