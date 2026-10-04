"""MuZero sequence semantics and native/CUDA end-to-end acceptance."""
import os
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch

from etazero.config import ROOT, load_config, validate
from etazero.data import read_raw, metadata, training_view
from etazero.network import make_network
from etazero.runtime import run_training
from etazero.storage import load_json, sha256
from etazero.training import load_checkpoint


def assert_diagnostics(root, config):
    from etazero.plotting import run_history, training_figure
    events=[json.loads(line) for line in (root/'logs/events.jsonl').read_text().splitlines()]
    updates=[e for e in events if e['event']=='update']
    assert updates
    for row in updates:
        assert len(row['step_losses']) == config['unroll']['steps']+1
        assert sum(row['step_losses']) == pytest.approx(row['loss'],rel=2e-6)
        assert set(row['grad_norms']) == {'representation','dynamics','prediction'}
        if not row['amp_skipped']:
            assert np.linalg.norm(list(row['grad_norms'].values())) == pytest.approx(row['grad_norm'],rel=2e-6)
    figure=training_figure(run_history(root))
    assert len(figure.axes)==8 and len(figure.axes[6].lines)==2 and len(figure.axes[7].lines)==3
    figure.clear()


def small_config():
    c = load_config(ROOT/'configs/smoke_test')
    defaults = load_config(ROOT/'configs/muzero')
    c['agent'] = defaults['agent']
    c['muzero'] = dict(latent_channels=16, dynamics_channels=24, dynamics_blocks=1,
                       prediction_channels=24, prediction_blocks=1)
    c['unroll'] = dict(steps=3, hidden_gradient_scale=.5)
    c['search']['reuse_tree'] = False
    c['graph_search']['use_graph_search'] = False
    c['symmetry']['root_num_symmetries_to_sample'] = 1
    c['inference'].update(cache_entries=8, server_threads=2)
    c['training']['skip_validation'] = True
    validate(c)
    return c


def test_config_inherits_baseline_and_disables_invalid_tree_features(tmp_path):
    c = load_config(ROOT/'configs/muzero')
    base = load_config(ROOT/'configs/baseline')
    assert c['training']['train_steps'] == base['training']['train_steps']
    assert c['value_weighting'] == base['value_weighting']
    from etazero.experiment import initialization_key, write_arm_config
    assert initialization_key(c) != initialization_key(base)
    c['value_weighting']['value_weight_exponent'] = 0
    validate(c)
    write_arm_config(c, tmp_path/'resolved')
    assert load_config(tmp_path/'resolved') == c
    assert make_network(small_config()).canvas == 6
    for section, key, value in [('search','reuse_tree',True), ('graph_search','use_graph_search',True),
                                 ('symmetry','root_num_symmetries_to_sample',2)]:
        invalid = small_config(); invalid[section][key] = value
        with pytest.raises(ValueError, match='MuZero requires'):
            validate(invalid)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA acceptance')
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('architecture', ['nbt', 'resnet'])
def test_cuda_complete_pipeline_and_resume(tmp_path, compiled, architecture):
    torch.set_num_threads(1); torch._dynamo.reset()
    c = small_config(); c['training']['compile'] = compiled
    c['network']['architecture'] = architecture
    if architecture == 'resnet':
        c['muzero_training'] = dict(auxiliary_losses=True, katago_optimizer=False,
                                    learning_rate=.001, weight_decay=.0003)
        c['optimizer']['kind'] = 'adamw'
    validate(c)
    c['network']['predict_q_values'] = True
    root = tmp_path/'run'
    state = run_training(root, c, ROOT/'build/etazero', max_iteration=2)
    assert_diagnostics(root,c)
    saved = load_checkpoint(root, state['checkpoint'], c)
    assert state['checkpoint']['total_steps'] == 8
    assert saved['algorithm'] == 'muzero' and saved['muzero_config'] == c['muzero']
    initial = load_checkpoint(root, load_json(root/'.internal/iterations/000001/plan.json')['input_checkpoint'], c)
    for key in ('representation.stem.weight','dynamics.stem.weight','prediction.policy_head.out.weight'):
        assert not torch.equal(initial['model'][key], saved['model'][key])
    files = {p: sha256(p) for p in (root/'selfplay').rglob('*.npz')}
    absorbing = side = cheap = 0
    for path in files:
        a = read_raw(path); view = training_view(a)
        assert metadata(a)['algorithm'] == 'muzero'
        absorbing += view['absorbing'].sum()
        side += (view['sequence_mask'][:, 1] == 0).sum()
        cheap += ((a['row_repeats'] == 0) & a['train_mask'].astype(bool)).sum()
        assert np.isfinite(view['td_value']).all()
        mask = view['absorbing'].astype(bool)
        assert np.allclose(view['value'][mask].sum(-1), 1)
    assert absorbing > 0 and cheap > 0
    assert run_training(root, c, ROOT/'build/etazero', resume=True, max_iteration=2) == state
    assert all(sha256(p) == digest for p, digest in files.items())
    manifest = load_json(root/Path(state['model']['path']).parent/'manifest.json')
    assert manifest['algorithm'] == 'muzero'
    assert manifest['normalization'] == ('masked_layernorm' if architecture == 'resnet' else 'precomputed_inv_std')
    # Same start rows, absorbing-action RNG, optimizer and normalization state
    # must lead to the identical next updates after a mid-round interruption.
    from test_gpu import assert_compiled_partial_resume
    assert_compiled_partial_resume(tmp_path, root, c)
    from etazero.evaluation import evaluate
    from etazero.eval_config import load_evaluation_config
    evaluation = load_evaluation_config(ROOT/'configs/muzero')
    evaluation['evaluation'].update(visits=12, search_threads=2)
    _, result = evaluate(evaluation, ROOT/'build/etazero', root, size=5, rule='renju', moves='0,5')
    assert result['result']['root_visits'] == 12 and result['result']['initial_visits'] == 0


def test_unroll_gradient_boundaries_and_side_mask():
    from etazero.muzero.training import TrainingForward
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(2.))
            self.d = torch.nn.Parameter(torch.tensor(3.))
            self.q = torch.nn.Parameter(torch.tensor(5.))
        def representation(self, obs, globals):
            return self.p.expand(len(obs), 1, 1, 1)
        def dynamics(self, hidden, actions):
            return hidden * self.d
    class Forward(TrainingForward):
        def components(self, hidden, obs, targets, step, weight):
            return ((hidden.flatten(1).mean(1) * self.model.q * weight).mean(),)
    model = Model(); forward = Forward(model, small_config())
    targets = [torch.ones(2, 4)] * 8
    inputs = [torch.ones(2, 1), torch.ones(2, 1), *targets,
              torch.zeros(2, 3), torch.ones(2, 4), torch.ones(2, 4)]
    (value,), step_losses = forward(*inputs); value.backward()
    assert step_losses.tolist() == [10,30,90,270]
    assert not step_losses.requires_grad
    assert value.item() == 400
    assert model.p.grad.item() == pytest.approx(28.75)
    assert model.d.grad.item() == pytest.approx(70+5/6)
    assert model.q.grad.item() == pytest.approx(28)
    model.zero_grad(); inputs[-1][1, 1:] = 0
    (value,), step_losses = forward(*inputs); value.backward()
    assert step_losses.tolist() == [10,15,45,135]
    assert value.item() == 205
    assert model.p.grad.item() == pytest.approx(5+(28.75-5)/2)
    model.zero_grad(); inputs[-1][:, 1:] = 0
    (value,), step_losses = forward(*inputs); value.backward()
    assert value.item() == 10 and step_losses.tolist() == [10,0,0,0]
    assert model.p.grad.item() == 5 and model.d.grad is None


@pytest.mark.parametrize('symmetry', range(8))
def test_sequence_d4_actions_match_all_spatial_targets(symmetry):
    from etazero.muzero.training import augment_batch
    from etazero.symmetry import apply_symmetry
    actions = torch.tensor([[0,7,14], [35,28,21]])
    obs = torch.zeros(2, 5, 6, 6); obs[:, 0, :5, :5] = 1
    policy = torch.nn.functional.one_hot(actions, 36).float()
    batch = dict(obs=obs, actions=actions, policy=policy, opponent_policy=policy, q_values=policy, q_visits=policy)
    result = augment_batch(batch, symmetry)
    assert result['policy'].shape == policy.shape
    assert torch.equal(result['actions'], result['policy'].argmax(-1))
    assert torch.equal(result['obs'], apply_symmetry(obs, symmetry))


def test_native_sequences_survive_zero_repeat_split_and_reader_resume(tmp_path):
    from etazero.config import write_native
    from etazero.shuffle import build_snapshot
    from etazero.reader import BatchReader
    c = small_config(); c['training']['batch_size'] = 4
    c['writer'].update(shard_rows=8, first_file_min_random_proportion=.15)
    c['parallelism'].update(game_threads=1, search_threads=1)
    c['inference']['queue_capacity'] = 1
    c['side_positions']['side_position_prob'] = .6
    c['reanalysis']=dict(use_reanalyze=True,reanalyze_prop=.5,reanalyze_policy_surprise_weight=1,
                        reanalyze_value_surprise_weight=0,reanalyze_surprise_exponent=1,reanalyze_use_outcome_targets=False)
    c['opening']['probability'] = 0; c['policy_init']['policy_init'] = False
    c['replay']['keep_target_rows'] = 'all'
    write_native(c, tmp_path/'effective.cfg')
    directory = tmp_path/'selfplay/raw'
    subprocess.run([str(ROOT/'build/etazero'), 'selfplay', '--config', str(tmp_path/'effective.cfg'),
                    '--evaluator','random','--model','random','--model-id','random','--device','cpu',
                    '--games','3','--output',str(directory),'--run-id','test','--attempt-id','test',
                    '--config-id','test','--source-id','test','--iteration','1','--worker','0','--seed','734'],
                   check=True, capture_output=True, text=True, timeout=60)
    entries=[]; sides=absorbing=zero=0
    for path in sorted(directory.glob('*.npz')):
        a = read_raw(path); m = metadata(a); v = training_view(a)
        entries.append(dict(path=str(path.relative_to(tmp_path)), sha256=sha256(path), metadata=m))
        starts=[]
        for g in range(m['games']):
            lo, hi=a['game_offsets'][g:g+2]
            starts += [(g, int(i)) for i in np.repeat(np.arange(lo, hi), a['row_repeats'][lo:hi])]
            starts += [(g, -1)] * int(a['side_row_repeats'][a['side_game_indices']==g].sum())
        starts=starts[m['row_begin']:m['row_begin']+m['rows']]
        for row, (g, start) in enumerate(starts):
            if start < 0:
                sides+=1; assert not v['sequence_mask'][row,1:].any(); continue
            lo, hi=a['game_offsets'][g:g+2]; ol=a['observation_offsets'][g]
            for step in range(1, 4):
                index=start+step
                if index < hi:
                    np.testing.assert_array_equal(v['policy'][row,step], a['trajectory_policy'][index])
                    assert v['step_weights'][row,step] == a['target_weights'][index]
                    zero += a['row_repeats'][index] == 0
                else:
                    absorbing+=1; assert v['absorbing'][row,step] and v['step_weights'][row,step]==1
                outcome=int(a['winners'][g])*int(a['players'][ol+start-lo])*(-1)**step
                assert v['value'][row,step].tolist() == [float(outcome==1),float(outcome==0),float(outcome==-1)]
    assert sides and absorbing and zero
    snapshot=tmp_path/'snapshots'/build_snapshot(tmp_path,1,entries,c)
    reader=BatchReader(snapshot,4,1,17); reader.next(); cursor=reader.state(); expected=reader.next(); reader.close()
    restored=BatchReader(snapshot,4,1,17,cursor); actual=restored.next(); restored.close()
    for key in expected: np.testing.assert_array_equal(actual[key], expected[key])
    assert np.all(actual['actions'] >= 0)
    original={p.name:sha256(p) for p in (snapshot/'data').glob('*.npz')}
    shutil.rmtree(snapshot/'data')
    restored=BatchReader(snapshot,4,1,17,cursor); restored.next(); restored.close()
    assert original == {p.name:sha256(p) for p in (snapshot/'data').glob('*.npz')}
    # Exercise the shared validation consumer with sequence targets and side rows.
    from etazero.storage import save_json
    from etazero.training import training_forward, validate_epoch
    manifest=load_json(snapshot/'manifest.json')
    manifest['validation_files']=manifest['files'][:1]
    manifest['validation_rows']=sum(f['rows'] for f in manifest['validation_files'])
    (snapshot/'data/validation').mkdir()
    for info in manifest['validation_files']:
        shutil.copy(snapshot/'data'/info['path'],snapshot/'data/validation'/info['path'])
    save_json(snapshot/'manifest.json',manifest)
    c['training'].update(skip_validation=False,max_validation_samples=1)
    model=make_network(c).train(); before={k:v.clone() for k,v in model.state_dict().items()}
    logs=[]
    validate_epoch(snapshot,model,training_forward(model,c),c,1,'cpu',lambda e,**f:logs.append((e,f)))
    assert logs[0][0]=='validation' and logs[0][1]['samples']==4
    assert np.isfinite(logs[0][1]['loss']) and model.training
    for key,value in model.state_dict().items(): assert torch.equal(value,before[key])


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA acceptance')
@pytest.mark.parametrize('amp', ['float16', 'bfloat16'])
def test_cuda_amp_pipeline_and_mixed_match(tmp_path, amp):
    from etazero.evaluation import evaluate
    from etazero.eval_config import load_evaluation_config
    from etazero.export import export_model
    from etazero.training import initialize
    from etazero.config import write_native
    torch.set_num_threads(1)
    c=small_config(); c['training']['amp']=amp
    c['inference']['inference_precision']='float16'
    root=tmp_path/'muzero'
    state=run_training(root,c,ROOT/'build/etazero',max_iteration=2)
    assert_diagnostics(root,c)
    assert state['checkpoint']['optimizer_steps'] > 0
    alpha=tmp_path/'alphazero'; alpha.mkdir()
    ac=load_config(ROOT/'configs/smoke_test'); write_native(ac,alpha/'config/effective.cfg')
    initial=initialize(alpha,ac)
    model=export_model(alpha,ac,initial)
    match=load_evaluation_config(ROOT/'configs/muzero',match=True)
    match['match'].update(visits=9,game_threads=2,search_threads=2,max_batch=8,
                          inference_precision='float16')
    _, result=evaluate(match,ROOT/'build/etazero',root,model_b=alpha/model['path'],
                       size=5,rule='renju',games=4,output=tmp_path/'mixed_match')
    assert result['result']['complete'] and len(result['result']['games'])==4
