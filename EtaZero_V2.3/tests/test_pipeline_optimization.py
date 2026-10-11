"""Replay read cost, explicit validation and stage-resume behavior."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import ROOT, load_config, resume_config, validate
from etazero.data import Catalog, read_raw, validate_raw, training_view, training_targets
from etazero.muzero.data import replay_weight_mean
from etazero.reader import BatchReader
from etazero.runtime import Controller, run_training
from etazero.storage import load_json, save_npz, sha256
from etazero.training import load_checkpoint
from test_python import winning_record
from test_replay import id_snapshot

HOST_CUDA = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA acceptance')


def test_catalog_materializes_each_npz_member_once(tmp_path, monkeypatch):
    raw, _ = winning_record()
    save_npz(tmp_path/'selfplay/game.npz', raw)
    counts = Counter()
    from numpy.lib.npyio import NpzFile
    getitem = NpzFile.__getitem__
    def counted(self, key):
        counts[key] += 1
        return getitem(self, key)
    monkeypatch.setattr(NpzFile, '__getitem__', counted)
    catalog = Catalog(tmp_path, 'test', 'config')
    try:
        catalog.scan({1: 'model'})
        assert counts and set(counts.values()) == {1}
        first = counts.copy()
        catalog.scan({1: 'model'})
        assert counts == first and catalog.counts() == (9, 1)
    finally:
        catalog.close()


def test_reader_repeated_passes_do_not_hash_files(tmp_path, monkeypatch):
    import etazero.storage as storage
    snapshot = id_snapshot(tmp_path, [4, 4])
    monkeypatch.setattr(storage, 'sha256', lambda p: pytest.fail('Reader must not hash internal shards'))
    reader = BatchReader(snapshot, 4, 1, 13)
    try:
        for _ in range(6):
            assert len(reader.next()['value']) == 4
    finally:
        reader.close()


def test_deep_trajectory_validation_is_explicit(tmp_path):
    raw, _ = winning_record()
    raw['observations'][1, 1, 0] ^= 128
    path = tmp_path/'raw.npz'
    save_npz(path, raw)
    read_raw(path, deep=False)
    with pytest.raises(ValueError, match='observation/transition'):
        read_raw(path)
    raw['globals'] = raw['globals'].astype(np.float64)
    with pytest.raises(ValueError, match='dtype'):
        validate_raw(raw, deep=False)


def test_weight_mean_deduplicates_games_across_partial_shards(tmp_path):
    a = {'id': 'game-a', 'sum': 6., 'count': 3}
    b = {'id': 'game-b', 'sum': 3., 'count': 1}
    sources = [{'metadata': {'algorithm': 'muzero'}, 'weight_stats': stats}
               for stats in ([a], [a, b], [b])]
    # Weights (1,2,3) and (3) give 9/4; repeated game fragments add nothing.
    assert replay_weight_mean(tmp_path, sources) == 2.25


@pytest.mark.parametrize('total', [0., float('nan'), float('inf')])
def test_weight_mean_rejects_invalid_normalization(tmp_path, total):
    sources = [{'metadata': {'algorithm': 'muzero'},
                'weight_stats': [{'id': 'a', 'sum': total, 'count': 1}]}]
    with pytest.raises(ValueError, match='positive main-position weight'):
        replay_weight_mean(tmp_path, sources)


def test_execution_changes_preserve_resume_requirements():
    original = load_config(CONFIGS/'smoke_test')
    changed = copy.deepcopy(original)
    changed['run'].update(cpu_threads=3, max_iteration=9, max_seconds=99, run_dir='/tmp/relocated')
    changed['training'].update(checkpoint_every=1, checkpoint_keep=2, prefetch_depth=0,
                               cuda_prefetch=False, compile=True)
    changed['devices'].update(selfplay='cuda:1', train='cuda:1')
    assert resume_config(original) == resume_config(changed)
    changed['training']['batch_size'] *= 2
    assert resume_config(original) != resume_config(changed)


def test_disabled_options_and_superlinear_window_are_valid():
    c = load_config(CONFIGS/'smoke_test')
    c['search'].update(cheap_search_probs=0, cheap_search_visits=1000, cheap_search_target_weight=2)
    c['reduce_visits'].update(reduce_visits=False, reduced_visits_min=1000)
    c['uncertainty'].update(use_uncertainty=False, uncertainty_coeff=0)
    c['noise_pruning'].update(use_noise_pruning=False, noise_prune_utility_scale=0)
    c['replay']['taper_exponent'] = 1.5
    validate(c)
    c['search']['cheap_search_probs'] = .5
    with pytest.raises(ValueError, match='cheap search'):
        validate(c)


@pytest.mark.parametrize('algorithm', ['alphazero', 'muzero', 'raw_muzero'])
def test_inactive_targets_preserve_rows_actions_losses_and_gradients(algorithm):
    from etazero.schema import unpack_observations
    from etazero.muzero.data import prepare_batch
    from etazero.training import training_forward, forward_batch, augment_training_batch
    from etazero.network import make_network
    import random
    from test_muzero_pipeline import small_config
    from test_raw_muzero import small_raw_config
    c = (load_config(CONFIGS/'smoke_test') if algorithm=='alphazero'
         else small_raw_config() if algorithm=='raw_muzero' else small_config())
    raw, meta = winning_record()
    if algorithm != 'alphazero':
        meta.update(algorithm='muzero',unroll_steps=c['unroll']['steps'])
        raw['metadata']=np.frombuffer(json.dumps(meta).encode(),np.uint8)
        raw['trajectory_policy']=raw['policies'].copy()
        raw['trajectory_q_values']=raw['q_values'].copy()
        raw['trajectory_q_visits']=raw['q_visits'].copy()
    full=training_view(raw);lean=training_view(raw,training_targets(c))
    assert 'q_values' not in lean and 'q_visits' not in lean
    if algorithm=='raw_muzero':
        assert not {'td_value','opponent_policy','full_game_weight'} & lean.keys()
    for key in lean:
        np.testing.assert_array_equal(lean[key],full[key])
    # Absorbing action draws, D4, forward targets and backward gradients all
    # agree. Omitted targets must also remain unused through recurrent steps.
    models=[make_network(c).train() for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    measured=[]
    for model,view in zip(models,(full,lean)):
        view['obs']=unpack_observations(view['obs'],c['network']['canvas'])
        if algorithm!='alphazero':
            view=prepare_batch(view,1.,random.Random(37))
        tensors={k:torch.from_numpy(v).to(dtype=torch.int64 if k=='actions' else torch.float32)
                 for k,v in view.items()}
        tensors=augment_training_batch(tensors,3)
        components,steps=forward_batch(training_forward(model,c),tensors)
        components[0].backward()
        measured.append((components,steps,tensors))
    for a,b in zip(measured[0][0],measured[1][0]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    if algorithm!='alphazero':
        torch.testing.assert_close(measured[0][1],measured[1][1],rtol=0,atol=0)
        assert torch.equal(measured[0][2]['actions'],measured[1][2]['actions'])
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_indexed_replay_selection_matches_full_history(tmp_path):
    from etazero.shuffle import window_sources,desired_window
    c=load_config(CONFIGS/'smoke_test')
    c['replay'].update(min_rows=9,max_rows=18,taper_exponent=1,expand_per_row=1)
    catalog=Catalog(tmp_path,'test','config');models={}
    try:
        for index in range(8):
            raw,meta=winning_record()
            model='random:1' if index<3 else 'model'
            meta.update(iteration_id=index,shard_id=str(index),attempt_id=str(index),model_id=model)
            models[index]=model
            raw['metadata']=np.frombuffer(json.dumps(meta).encode(),np.uint8)
            path=tmp_path/'selfplay'/str(index)/'game.npz';save_npz(path,raw)
            os.utime(path,ns=(100+index//2,100+index//2)) # Include tied mtimes.
            catalog.scan(models,[path.parent])
        full,reference=window_sources(catalog.entries(),c['replay'])
        counts=catalog.replay_counts(c['replay']['min_rows'])
        assert counts==dict(raw_rows=72,random_rows=27,postrandom_rows=45,usable_rows=54)
        selected,actual=window_sources(catalog.entries(desired_window(counts['usable_rows'],c['replay'])),c['replay'],counts)
        assert actual==reference and selected==full and len(selected)==2
        reopened=Catalog(tmp_path,'test','config')
        try:
            assert reopened.replay_counts(c['replay']['min_rows'])==counts
        finally:
            reopened.close()
    finally:
        catalog.close()


@HOST_CUDA
def test_multiple_rounds_plot_after_each_commit_and_on_resume(tmp_path,monkeypatch):
    c=load_config(CONFIGS/'smoke_test');calls=[]
    monkeypatch.setattr(Controller,'plot',lambda self:calls.append(load_json(self.root/'.internal/state.json')['iteration']))
    state=run_training(tmp_path,c,ROOT/'build/etazero',max_iteration=2)
    assert calls==[1,2,3] and state['iteration']==3
    calls.clear()
    assert run_training(tmp_path,c,ROOT/'build/etazero',max_iteration=2)==state
    assert calls==[3]


@HOST_CUDA
@pytest.mark.parametrize('algorithm', ['alphazero', 'muzero'])
def test_controller_resumes_saved_batch_without_repeating_selfplay(tmp_path, monkeypatch, algorithm):
    import etazero.training as training
    from test_muzero_pipeline import small_config
    torch.set_num_threads(1)
    c = small_config() if algorithm == 'muzero' else load_config(CONFIGS/'smoke_test')
    c['training'].update(compile=False, checkpoint_every=1)
    c['inference']['inference_precision'] = 'float16'
    monkeypatch.setattr(Controller, 'plot', lambda *a, **k: None)
    parent = tmp_path/'parent'
    run_training(parent, c, ROOT/'build/etazero', max_iteration=1)
    # Fix the exact selfplay data and shuffle snapshot before branching the
    # comparison. Independently scheduled GPU selfplay is not bitwise stable.
    with monkeypatch.context() as patch:
        def stop_before_learner(*args, **kwargs):
            raise RuntimeError('fixed snapshot ready')
        patch.setattr('etazero.runtime.train_iteration', stop_before_learner)
        with pytest.raises(RuntimeError, match='fixed snapshot ready'):
            run_training(parent, c, ROOT/'build/etazero', max_iteration=2)
    roots = [tmp_path/'continuous', tmp_path/'resumed']
    for root in roots:
        shutil.copytree(parent, root)
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        full = run_training(roots[0], c, ROOT/'build/etazero', max_iteration=2)
        original = training.save_json
        with monkeypatch.context() as patch:
            def interrupt(path, value, *args, **kwargs):
                result = original(path, value, *args, **kwargs)
                if (Path(path) == roots[1]/'.internal/iterations/000002/learner.json'
                        and value['checkpoint']['step'] == 1):
                    raise RuntimeError('injected saved batch interruption')
                return result
            patch.setattr(training, 'save_json', interrupt)
            with pytest.raises(RuntimeError, match='saved batch'):
                run_training(roots[1], c, ROOT/'build/etazero', max_iteration=2)
        progress = load_json(roots[1]/'.internal/iterations/000002/learner.json')
        assert progress['checkpoint']['step'] == 1
        files = {p: sha256(p) for p in (roots[1]/'selfplay').rglob('*.npz')}
        changed = copy.deepcopy(c)
        changed['run'].update(cpu_threads=2, max_iteration=2, run_dir=str(roots[1]))
        changed['training'].update(checkpoint_every=3, prefetch_depth=2)
        resumed = run_training(roots[1], changed, ROOT/'build/etazero', max_iteration=2)
        assert files == {p: sha256(p) for p in (roots[1]/'selfplay').rglob('*.npz')}
        assert load_json(roots[1]/'.internal/iterations/000002/status.json')['snapshot_id'] == progress['snapshot_id']
        left = load_checkpoint(roots[0], full['checkpoint'], c)
        right = load_checkpoint(roots[1], resumed['checkpoint'], changed)
        def equal(a, b):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            elif isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    equal(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                assert a == b
        for key in ('model', 'optimizer', 'optimization', 'scaler', 'reader', 'rng'):
            equal(left[key], right[key])
        events = [json.loads(line) for line in (roots[1]/'logs/events.jsonl').read_text().splitlines()]
        assert [e['step'] for e in events if e['event'] == 'update' and e['iteration'] == 2] == [1, 2, 3, 4]
        assert load_json(roots[1]/'config/effective.json') == changed
    finally:
        torch.use_deterministic_algorithms(deterministic)


@HOST_CUDA
def test_plot_failure_cannot_undo_training_commit(tmp_path, monkeypatch):
    import etazero.plotting as plotting
    torch.set_num_threads(1)
    c = load_config(CONFIGS/'smoke_test')
    def fail(*a, **k):
        raise OSError('injected PNG failure')
    monkeypatch.setattr(plotting, 'plot_run', fail)
    with pytest.warns(RuntimeWarning, match='plotting failed'):
        state = run_training(tmp_path, c, ROOT/'build/etazero', max_iteration=1)
    assert state['iteration'] == 2 and state['checkpoint']['total_steps'] == 4
    assert load_json(tmp_path/'models/current.json')['model'] == state['model']
    assert [r['steps'] for r in plotting.run_history(tmp_path)] == [0, 4]
    raw = {p: sha256(p) for p in (tmp_path/'selfplay').rglob('*.npz')}
    with pytest.warns(RuntimeWarning, match='plotting failed'):
        assert run_training(tmp_path, c, ROOT/'build/etazero', max_iteration=1) == state
    assert raw == {p: sha256(p) for p in (tmp_path/'selfplay').rglob('*.npz')}
