"""Real CUDA acceptance for deferred synchronization and sparse learner journals."""
import copy
import math
import os
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import load_config, resume_config, validate
from etazero.storage import load_json
from etazero.train.learner import finish_optimizer_step, initialize, load_checkpoint, train_iteration
from test_replay import id_snapshot

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA acceptance')


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_state(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b):
            equal_state(x,y)
    else:
        assert a == b


@pytest.mark.parametrize('kind', ['sgd', 'adamw'])
@pytest.mark.parametrize('amp', ['off', 'float16', 'bfloat16'])
def test_deferred_sync_is_bitwise_equal_to_pre_update_sync(kind, amp):
    torch.manual_seed(41)
    initial = torch.nn.Linear(16, 8).cuda()
    x = torch.randn(32, 16, device='cuda')
    y = torch.randn(32, 8, device='cuda')
    results = []
    for deferred in (False, True):
        model = copy.deepcopy(initial)
        if kind == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, fused=True)
        scaler = torch.amp.GradScaler('cuda', enabled=amp=='float16', init_scale=8)
        params = list(model.parameters())
        values_seen = []
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=amp!='off', dtype=torch.float16 if amp=='float16' else torch.bfloat16):
                prediction = model(x)
                loss = (prediction.float()-y).square().mean()
            scaler.scale(loss).backward()
            if amp == 'float16' and step in (0, 2):
                params[0].grad.fill_(float('inf'))
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.get_total_norm([p.grad for p in params])
            status = ([scaler._get_scale_async(), sum(scaler._found_inf_per_device(optimizer).values())]
                      if scaler.is_enabled() else [])
            packed = torch.stack([loss.detach(), norm, *status])
            if deferred:
                values, skipped = finish_optimizer_step(params, optimizer, scaler, norm, .3, packed, 1)
            else:
                values = packed.tolist()
                skipped = bool(values[-1]) if status else False
                if math.isfinite(values[1]):
                    torch.nn.utils.clip_grads_with_norm_(params, .3, norm)
                scaler.step(optimizer)
                scaler.update()
            values_seen.append((values, skipped))
        results.append((model.state_dict(), optimizer.state_dict(), scaler.state_dict(), values_seen))
    equal_state(*results)


def learner_config():
    torch.set_num_threads(1)
    c = load_config(CONFIGS/'smoke_test')
    c['training'].update(compile=False, checkpoint_every=0, log_every=2, skip_validation=True, amp='off')
    validate(c)
    return c


def prepare(root, c):
    id_snapshot(root, [8*c['training']['batch_size']])
    plan = dict(iteration=1, train_steps=4, batch_size=c['training']['batch_size'], snapshot_id='ids')
    return plan, initialize(root, c)


def test_sparse_logs_tail_checkpoint_and_resume_keep_every_batch(tmp_path, monkeypatch):
    import etazero.train.learner as training
    c = learner_config()
    dirs = [tmp_path/'full', tmp_path/'resume']
    plans = [prepare(root, c) for root in dirs]
    records = []
    def log(event, **fields):
        records.append(dict(event=event, **fields))
    full, done = train_iteration(dirs[0], c, *plans[0], log)
    assert done
    updates = [r for r in records if r['event']=='update']
    assert [r['step'] for r in updates] == [2,4]
    assert [r['steps'] for r in updates] == [2,2]
    assert len(load_json(dirs[0]/'.internal/checkpoints.json')) == 2
    count = 0
    original = training.finish_optimizer_step
    def counted(*args):
        nonlocal count
        result = original(*args)
        count += 1
        return result
    monkeypatch.setattr(training, 'finish_optimizer_step', counted)
    records.clear()
    partial, done = train_iteration(dirs[1], c, *plans[1], log, lambda: count>=3)
    assert not done and partial['step']==3
    updates = [r for r in records if r['event']=='update']
    assert [(r['step'],r['steps']) for r in updates] == [(2,2),(3,1)]
    changed = copy.deepcopy(c); changed['training']['log_every'] = 100
    assert resume_config(c) == resume_config(changed)
    resumed, done = train_iteration(dirs[1], changed, *plans[1], log)
    assert done
    a = load_checkpoint(dirs[0], full, c)
    b = load_checkpoint(dirs[1], resumed, changed)
    for key in ('model', 'optimizer', 'optimization', 'scaler', 'reader', 'training_metrics'):
        equal_state(a[key], b[key])
    assert records[-2]['round_metrics']['steps'] == 4


@pytest.mark.parametrize('failure', ['nan_loss', 'inf_loss', 'inf_grad', 'norm_overflow', 'bad_wdl'])
def test_fatal_batch_never_publishes_checkpoint_or_continues(tmp_path, monkeypatch, failure):
    import etazero.train.learner as training
    c = learner_config()
    if failure == 'norm_overflow':
        c['training']['amp'] = 'float16'
        original_init = torch.amp.GradScaler.__init__
        def small_scale(self, *args, **kwargs):
            kwargs['init_scale'] = 1.
            original_init(self, *args, **kwargs)
        monkeypatch.setattr(torch.amp.GradScaler, '__init__', small_scale)
    root = tmp_path/'run'
    plan, base = prepare(root, c)
    original = training.forward_batch
    calls = 0
    def corrupt(forward, tensors):
        nonlocal calls
        calls += 1
        if failure == 'bad_wdl' and calls==2:
            tensors['td_value'][0,0] = tensors['td_value'].new_tensor([-5e-11,1.,7e-9])
        components, steps = original(forward, tensors)
        if calls==2 and failure in ('nan_loss','inf_loss'):
            components = (components[0] * float('nan' if failure=='nan_loss' else 'inf'), *components[1:])
        return components, steps
    monkeypatch.setattr(training, 'forward_batch', corrupt)
    original_unscale = torch.amp.GradScaler.unscale_
    def unscale(scaler, optimizer):
        # Finite 1e30 elements overflow only the norm reduction, after unscaling.
        original_unscale(scaler, optimizer)
        if calls==2 and failure in ('inf_grad', 'norm_overflow'):
            p = next(p for group in optimizer.param_groups for p in group['params'] if p.grad is not None)
            p.grad.fill_(1e30 if failure=='norm_overflow' else float('inf'))
    monkeypatch.setattr(torch.amp.GradScaler, 'unscale_', unscale)
    events = []
    with pytest.raises(FloatingPointError, match='iteration 1, step 2'):
        train_iteration(root, c, plan, base, lambda event, **fields: events.append(dict(event=event, **fields)))
    assert calls == 2
    assert load_json(root/'.internal/checkpoints.json') == [base]
    assert not (root/'.internal/iterations/000001/learner.json').exists()
    assert events[-1]['event'] == 'training_failure'
    assert not any(e['event']=='update' for e in events)
