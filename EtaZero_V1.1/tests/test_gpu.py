"""Real-CUDA integration checks. Enable only in an execution context with host device access."""
from config_samples import CONFIGS
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from etazero.config import ROOT, load_config, write_native
from etazero.data import read_raw, metadata, game_row_ranges
from etazero.evaluation import evaluate
from etazero.eval_config import load_evaluation_config
from etazero.export import example_inputs, export_model, verify_export
from etazero.network import make_network, inference_network, MaskedBatchNorm
from etazero.optimization import Optimization, inference_weights
from etazero.runtime import run_training
from etazero.storage import load_json, save_json, sha256
from etazero.training import initialize, load_checkpoint, train_iteration, optimizer_for, commit_checkpoint, restore_rng

pytestmark=pytest.mark.skipif(os.environ.get("ETAZERO_GPU_TESTS")!="1",reason="Enable ETAZERO_GPU_TESTS=1 with host CUDA access")
BINARY=ROOT/"build"/"etazero"


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    # Cases intentionally change network/precision and install instrumentation.
    # Give each case its own controller cache budget, retaining caches across
    # all rounds and resume calls within that case. Production limits stay intact.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def assert_preserved(root, files):
    for path, digest in files.items():
        if path.exists():
            assert sha256(path)==digest
        else:
            archived=list((root/'.internal/discarded').glob('*/'+str(path.relative_to(root))))
            assert len(archived)==1 and sha256(archived[0])==digest


@pytest.fixture(scope="module")
def gpu_config():
    assert torch.cuda.is_available(),"GPU tests require actual host CUDA access"
    torch.set_num_threads(1)
    return load_config(CONFIGS/"smoke_test")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory,gpu_config):
    root=tmp_path_factory.mktemp("pipeline")
    state=run_training(root,gpu_config,BINARY,max_iteration=2)
    return root,state


def test_script_starts_and_auto_resumes_training(tmp_path):
    configs=tmp_path/"configs";shutil.copytree(CONFIGS,configs)
    root=tmp_path/"selfplay"/"smoke_test"
    selected=configs/"smoke_test"
    (selected/"run.cfg").write_text(f"[run]\nextends = baseline\nrun_dir = {root}\nmax_iteration = 1\ncpu_threads = 1\n")
    env={**os.environ,"CONFIG_DIR":str(selected)}
    command=["bash",str(ROOT/"scripts"/"run.sh")]
    def launch(*args):
        result=subprocess.run(command+list(args),cwd=tmp_path,env=env,text=True,capture_output=True,timeout=180)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        return load_json(root/".internal/state.json")
    first=launch()
    assert first["iteration"]==2 and first["checkpoint"]["total_steps"]==4
    run_id=first["run_id"]
    shards={p:sha256(p) for p in (root/"selfplay").rglob("*.npz")}
    assert shards
    second=launch("--iterations","2")
    assert second["run_id"]==run_id and second["iteration"]==3
    assert second["checkpoint"]["total_steps"]==8
    assert all(sha256(path)==checksum for path,checksum in shards.items())
    assert launch("--plot")==second
    assert (root/"training.png").is_file()


def test_unlimited_iterations_stop_and_auto_resume(tmp_path):
    configs=tmp_path/"configs";shutil.copytree(CONFIGS,configs)
    root=tmp_path/"run"
    selected=configs/"smoke_test"
    (selected/"run.cfg").write_text(f"[run]\nextends = baseline\nrun_dir = {root}\nmax_iteration = 0\ncpu_threads = 1\n")
    env={**os.environ,"CONFIG_DIR":str(selected)}
    command=["bash",str(ROOT/"scripts"/"run.sh")]
    stdout=tmp_path/"stdout";stderr=tmp_path/"stderr"
    with stdout.open('w') as out,stderr.open('w') as err:
        process=subprocess.Popen(command,cwd=tmp_path,env=env,stdout=out,stderr=err,start_new_session=True)
        deadline=time.monotonic()+90
        try:
            while time.monotonic()<deadline:
                state_path=root/".internal/state.json"
                if state_path.exists() and load_json(state_path)["iteration"]>=3:
                    process.send_signal(signal.SIGINT)
                    break
                if process.poll() is not None:
                    pytest.fail("Unlimited training exited before two iterations: "+stderr.read_text())
                time.sleep(0.01)
            else:
                pytest.fail("Timed out waiting for unlimited training to complete two iterations")
            assert process.wait(timeout=30)==0,stdout.read_text()+'\n'+stderr.read_text()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
    first=load_json(root/".internal/state.json")
    assert first["iteration"]>=3
    assert load_json(root/"config/effective.json")["run"]["max_iteration"]==0
    events=[json.loads(line) for line in (root/"logs/events.jsonl").read_text().splitlines()]
    assert any(e["event"]=="run_stopped" and e["completed_iterations"]>=2 for e in events)
    shards={p:sha256(p) for p in (root/"selfplay").rglob("*.npz")}
    target=first["iteration"]
    result=subprocess.run(command+["--iterations",str(target)],cwd=tmp_path,env=env,
                          text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    resumed=load_json(root/".internal/state.json")
    assert resumed["run_id"]==first["run_id"] and resumed["iteration"]==target+1
    assert resumed["checkpoint"]["total_steps"]==target*4
    assert_preserved(root,shards)


def test_default_nested_run_directory_and_explicit_resume(tmp_path,monkeypatch,gpu_config):
    from etazero.__main__ import main
    shutil.copytree(CONFIGS,tmp_path/'configs')
    selected=tmp_path/'configs/experiment/100v';selected.mkdir(parents=True)
    (selected/'run.cfg').write_text('[run]\nextends=smoke_test\n')
    monkeypatch.setattr('etazero.config.ROOT',tmp_path)
    caller=tmp_path/'caller';caller.mkdir();monkeypatch.chdir(caller)
    command=['etazero','run','--config-dir',str(selected),'--binary',str(BINARY)]
    monkeypatch.setattr(sys,'argv',command+['--iterations','1'])
    main()
    root=tmp_path/'data/experiment/100v'
    first=load_json(root/'.internal/state.json')
    effective=load_json(root/'config/effective.json')
    assert effective['run']['run_dir']==str(root)
    assert first['iteration']==2 and first['checkpoint']['total_steps']==4
    shards={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')}
    assert shards and not (caller/'data').exists()
    monkeypatch.setattr(sys,'argv',command+['--iterations','2','--run-dir',str(root)])
    main()
    second=load_json(root/'.internal/state.json')
    assert second['run_id']==first['run_id'] and second['iteration']==3
    assert second['checkpoint']['total_steps']==8
    assert load_json(root/'config/effective.json')==effective
    assert all(sha256(p)==digest for p,digest in shards.items())


def test_random_pda_bootstrap_and_cuda_training(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config)
    c['pda']['normal_asymmetric_playout_prob']=1
    c['pda']['max_asymmetric_ratio']=2
    c['search']['full_search_visits']=24
    c['search']['cheap_search_visits']=12
    c['reduce_visits']['reduced_visits_min']=12
    state=run_training(tmp_path,c,BINARY,max_iteration=2)
    assert state['checkpoint']['total_steps']==2*c['training']['train_steps']
    assert state['model'] is not None
    for iteration in range(3):
        shards=list((tmp_path/'selfplay'/f'iteration_{iteration:06d}').rglob('*.npz'))
        assert shards
        globals_=np.concatenate([read_raw(path)['globals'] for path in shards])
        assert np.any((globals_[:,-1]!=0) & (globals_[:,-1]!=np.floor(globals_[:,-1])))


def test_gpu_pipeline_and_idempotent_completed_resume(pipeline,gpu_config):
    root,state=pipeline
    assert state["checkpoint"]["total_steps"]==8
    assert state["checkpoint"]["total_samples"]==64
    shards={p:sha256(p) for p in (root/"selfplay").rglob("*.npz")}
    assert shards
    for path in shards:
        raw=read_raw(path)
        np.testing.assert_array_equal(raw['sample_indices'],np.flatnonzero(raw['row_repeats']))
        assert len(raw['policies'])==len(raw['visits'])==len(raw['sample_indices'])
        assert len(raw['opponent_policies'])==len(raw['opponent_policy_weights'])==len(raw['policies'])
    assert any(len(read_raw(path)['sample_indices'])<len(read_raw(path)['actions']) for path in shards)
    events=[json.loads(line) for line in (root/"logs/events.jsonl").read_text().splitlines()]
    for event in (e for e in events if e['event']=='update'):
        components=[event[key] for key in ('policy_loss','opponent_policy_loss','soft_policy_loss',
                                          'soft_opponent_policy_loss','value_loss')]
        assert all(np.isfinite(components)) and min(components)>0
        components += [event[key] for key in ('td_value_long_loss','td_value_mid_loss','td_value_short_loss',
                                              'long_optimistic_policy_loss','short_optimistic_policy_loss','shortterm_value_error_loss')]
        assert all(np.isfinite(components))
        assert sum(components)==pytest.approx(event['loss'],rel=2e-6)
    assert max(e["max_batch"] for e in events if e["event"]=="inference")>1
    starts=[e for e in events if e['event']=='worker_start']
    assert len(starts)>1 and len({e['pid'] for e in starts})==1
    for entry in starts:
        attempt=load_json(root/'.internal/iterations'/f"{entry['iteration']:06d}"/f"attempt_{entry['attempt']}.json")
        assert attempt['model_id']==load_json(root/'.internal/iterations'/f"{entry['iteration']:06d}"/'plan.json')['input_model']['id']
    before=len(list((root/"checkpoints").glob("*.pt")))
    resumed=run_training(root,gpu_config,BINARY,resume=True,max_iteration=2)
    assert state==resumed and len(list((root/"checkpoints").glob("*.pt")))==before
    assert shards=={p:sha256(p) for p in shards}


@pytest.mark.parametrize('amp',['off','float16','bfloat16'])
def test_b5c192nbt_compiled_11x11_pipeline(tmp_path,gpu_config,amp):
    c=copy.deepcopy(gpu_config)
    c['network'].update(canvas=11,channels=192,blocks=5)
    c['environment'].update(sizes='11',size_weights='1')
    c['training'].update(compile=True,amp=amp)
    c['shuffle']['group_rows']=512  # One complete 11x11 game can exceed smoke's 128-row cap.
    state=run_training(tmp_path,c,BINARY,max_iteration=2)
    assert state['checkpoint']['total_steps']==8 and state['checkpoint']['total_samples']==64
    saved=load_checkpoint(tmp_path,state['checkpoint'],c)
    assert saved['model']['value_head.hidden.weight'].shape==(80,96)
    assert saved['model']['policy_head.out.weight'].shape==(6,32,1,1)
    updates=[json.loads(line) for line in (tmp_path/'logs/events.jsonl').read_text().splitlines()]
    updates=[e for e in updates if e['event']=='update']
    assert len(updates)==8 and all(np.isfinite(e['loss']) and (e['amp_skipped'] or np.isfinite(e['grad_norm'])) for e in updates)
    assert saved['optimizer_steps']==8-sum(e['amp_skipped'] for e in updates)
    info=load_json(tmp_path/Path(state['model']['path']).parent/'manifest.json')
    assert info['weights']=='swa'
    assert verify_export(tmp_path,info,c['network']['canvas'])==info
    assert run_training(tmp_path,c,BINARY,resume=True,max_iteration=2)==state


@pytest.mark.parametrize('amp',['off','float16','bfloat16'])
def test_b10c128_plain_mixed_sizes_compiled_pipeline_and_resume(tmp_path,gpu_config,amp):
    c=copy.deepcopy(gpu_config)
    c['network'].update(architecture='plain',canvas=15,channels=128,blocks=10)
    c['environment'].update(sizes='11,15',size_weights='1,1')
    c['training'].update(compile=True,amp=amp)
    c['shuffle']['group_rows']=512
    root=tmp_path/'pipeline'
    state=run_training(root,c,BINARY,max_iteration=2)
    saved=load_checkpoint(root,state['checkpoint'],c)
    assert state['checkpoint']['total_steps']==8 and state['checkpoint']['total_samples']==64
    assert saved['model']['blocks.4.pre.conv.conv.weight'].shape==(96,128,3,3)
    assert saved['model']['blocks.7.pre.conv.conv_global.weight'].shape==(32,128,3,3)
    assert saved['model']['value_head.hidden.weight'].shape==(80,96)
    assert saved['model']['policy_head.out.weight'].shape==(6,32,1,1)
    initial=load_checkpoint(root,load_json(root/'.internal/iterations/000001/plan.json')['input_checkpoint'],c)
    assert not torch.equal(initial['model']['stem.weight'],saved['model']['stem.weight'])
    assert not torch.equal(initial['model']['blocks.4.pre.conv.linear.weight'],saved['model']['blocks.4.pre.conv.linear.weight'])
    updates=[json.loads(line) for line in (root/'logs/events.jsonl').read_text().splitlines()]
    updates=[e for e in updates if e['event']=='update']
    assert len(updates)==8 and all(np.isfinite(e['loss']) and (e['amp_skipped'] or np.isfinite(e['grad_norm'])) for e in updates)
    assert saved['optimizer_steps']==8-sum(e['amp_skipped'] for e in updates)
    manifest=load_json(root/Path(state['model']['path']).parent/'manifest.json')
    assert manifest['weights']=='swa'
    assert verify_export(root,manifest,c['network']['canvas'])==manifest
    assert run_training(root,c,BINARY,resume=True,max_iteration=2)==state
    # Resume after a real compiled update, comparing against uninterrupted
    # consumption of the same native/shuffled snapshot and initialization.
    assert_compiled_partial_resume(tmp_path,root,c)


@pytest.mark.parametrize('predict_q_values',[False,True])
@pytest.mark.parametrize('kind,amp',[('sgd','off'),('sgd','float16'),('sgd','bfloat16'),('adamw','float16')])
def test_b5c192_transformer_compiled_pipeline_and_resume(tmp_path,gpu_config,kind,amp,monkeypatch,predict_q_values):
    original=make_network
    attention_gradients=[]
    def instrumented(config):
        model=original(config)
        def record(grad):
            attention_gradients.append(grad.detach().abs().max())
            return grad
        model.blocks[0].inner[0].q_proj.weight.register_hook(record)
        return model
    monkeypatch.setattr('etazero.training.make_network',instrumented)
    c=copy.deepcopy(gpu_config)
    c['network'].update(architecture='transformer',canvas=15,channels=192,blocks=5,predict_q_values=predict_q_values)
    c['environment'].update(sizes='11,15',size_weights='1,1')
    c['training'].update(compile=True,amp=amp)
    c['optimizer']['kind']=kind
    c['shuffle']['group_rows']=512
    root=tmp_path/'pipeline'
    state=run_training(root,c,BINARY,max_iteration=2)
    saved=load_checkpoint(root,state['checkpoint'],c)
    assert state['checkpoint']['total_steps']==8 and state['checkpoint']['total_samples']==64
    assert saved['model']['value_head.hidden.weight'].shape==(64,96)
    assert saved['model']['policy_head.out.weight'].shape==(6+int(predict_q_values),32,1,1)
    assert not any('running_' in name for name in saved['model'])
    initial=load_checkpoint(root,load_json(root/'.internal/iterations/000001/plan.json')['input_checkpoint'],c)
    assert not torch.equal(saved['model']['stem.weight'],initial['model']['stem.weight'])
    assert saved['model']['blocks.0.post.conv.weight'].count_nonzero()>0
    # Fixup starts with a zero outer projection. At source warmup LR, an inner
    # SGD update in this tiny run can be smaller than a stored weight's ULP.
    # Require actual attention gradients rather than inventing a weight-change
    # threshold, increasing the LR or replacing the source initialization.
    assert len(attention_gradients)==8
    assert any(torch.isfinite(g) and g>0 for g in attention_gradients[1:])
    groups={g['group_name']:g for g in saved['optimizer']['param_groups']}
    assert set(groups)=={'input','normal','normal_attn','normal_gamma','noreg','output','output_noreg'}
    scale=(8/256)**0.5 if kind=='adamw' else 8/256
    assert groups['normal_attn']['weight_decay']==pytest.approx((0.005 if kind=='adamw' else 1e-6)*0.5*scale)
    events=[json.loads(l) for l in (root/'logs/events.jsonl').read_text().splitlines()]
    updates=[x for x in events if x['event']=='update']
    assert len(updates)==8 and all(np.isfinite(e['loss']) and (e['amp_skipped'] or np.isfinite(e['grad_norm'])) for e in updates)
    assert saved['optimizer_steps']==8-sum(e['amp_skipped'] for e in updates)
    assert all(('q_winloss_loss' in e)==predict_q_values for e in updates)
    if predict_q_values:
        assert all(e['q_winloss_loss']>0 for e in updates)
        assert not torch.equal(saved['model']['policy_head.out.weight'][6],initial['model']['policy_head.out.weight'][6])
        from etazero.data import read_raw,training_view
        for shard in (root/'selfplay').rglob('*.npz'):
            raw=read_raw(shard);view=training_view(raw)
            assert raw['q_visits'].max()>0
            assert raw['q_values'].dtype==np.int16 and abs(raw['q_values']).max()<=32000
            # Raw context includes complete games; shards select a range of the
            # repeated main and side rows, ordered main-then-side per game.
            expected=[];main_start=side_start=0
            for g in range(len(raw['game_ids'])):
                lo,hi=raw['game_offsets'][g:g+2]
                main_count=int(raw['row_repeats'][lo:hi].sum())
                side_count=int(raw['side_row_repeats'][raw['side_game_indices']==g].sum())
                expected.append(raw['q_values'][main_start:main_start+main_count])
                expected.append(raw['side_q_values'][side_start:side_start+side_count])
                main_start+=main_count;side_start+=side_count
            m=metadata(raw)
            expected=np.concatenate(expected)[m['row_begin']:m['row_begin']+m['rows']]
            np.testing.assert_array_equal(view['q_values'],expected/np.float32(32000))
    manifest=load_json(root/Path(state['model']['path']).parent/'manifest.json')
    assert manifest['weights']=='swa'
    assert verify_export(root,manifest,c['network']['canvas'])==manifest
    assert manifest['normalization']=='masked_fixup_bias'
    assert run_training(root,c,BINARY,resume=True,max_iteration=2)==state
    # Audit exact state recovery under a deterministic attention backend.
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        assert_compiled_partial_resume(tmp_path,root,c)


def test_checkpoint_retention_resume_history_and_old_model(tmp_path,gpu_config):
    from etazero.plotting import run_history, plot_run
    c=copy.deepcopy(gpu_config)
    c['training'].update(checkpoint_keep=2,replay_ratio=1000)
    state=run_training(tmp_path,c,BINARY,max_iteration=3)
    statuses=[load_json(tmp_path/'.internal/iterations'/f'{i:06d}'/'status.json') for i in range(1,4)]
    expected={tmp_path/s['checkpoint']['path'] for s in statuses[-2:]}
    assert set((tmp_path/'checkpoints').glob('*.pt'))==expected
    assert not (tmp_path/statuses[0]['checkpoint']['path']).exists()
    sidecars={p:sha256(p) for p in (tmp_path/'checkpoints').glob('*.json')}
    assert len(sidecars)==7  # Initialization plus two checkpoints per training round.
    history=run_history(tmp_path)
    assert [row['steps'] for row in history if row['iteration']>0]==[4,4,4]
    models={p:sha256(p) for p in (tmp_path/'models').rglob('model.pt')}
    raw={p:sha256(p) for p in (tmp_path/'selfplay').rglob('*.npz')}
    assert run_training(tmp_path,c,BINARY,resume=True,max_iteration=3)==state
    plot_run(tmp_path)
    assert run_history(tmp_path)==history
    assert (tmp_path/'training.png').stat().st_size>0
    old_model=tmp_path/statuses[0]['model']['path']
    _,evaluation=evaluate(load_evaluation_config(CONFIGS / 'smoke_test'),BINARY,
                          tmp_path,model=old_model,size=5,rule='renju')
    assert evaluation['result']['root_visits']==100
    resumed=run_training(tmp_path,c,BINARY,resume=True,max_iteration=4)
    assert resumed['checkpoint']['total_steps']==16
    assert set((tmp_path/'checkpoints').glob('*.pt'))=={
        tmp_path/state['checkpoint']['path'],tmp_path/resumed['checkpoint']['path']}
    assert [row['steps'] for row in run_history(tmp_path) if row['iteration']>0]==[4,4,4,4]
    assert all(sha256(p)==digest for p,digest in {**sidecars,**models,**raw}.items())


def test_resume_finishes_cleanup_after_round_commit(tmp_path,gpu_config,monkeypatch):
    import etazero.runtime as runtime
    from etazero.plotting import run_history
    c=copy.deepcopy(gpu_config);c['training']['checkpoint_keep']=1
    prune=runtime.prune_checkpoints
    def interrupted(root,keep):
        if load_json(root/'.internal/state.json')['iteration']==2:
            raise RuntimeError('injected interruption before checkpoint cleanup')
        return prune(root,keep)
    with monkeypatch.context() as patch:
        patch.setattr(runtime,'prune_checkpoints',interrupted)
        with pytest.raises(RuntimeError,match='before checkpoint cleanup'):
            run_training(tmp_path,c,BINARY,max_iteration=1)
    state=load_json(tmp_path/'.internal/state.json')
    assert state['checkpoint']['total_steps']==4
    assert len(list((tmp_path/'checkpoints').glob('*.pt')))==3
    history=run_history(tmp_path)
    preserved={p:sha256(p) for folder,pattern in [('checkpoints','*.json'),
               ('models','model.pt'),('selfplay','*.npz')] for p in (tmp_path/folder).rglob(pattern)}
    resumed=run_training(tmp_path,c,BINARY,resume=True,max_iteration=1)
    assert resumed==state
    assert set((tmp_path/'checkpoints').glob('*.pt'))=={tmp_path/state['checkpoint']['path']}
    assert run_history(tmp_path)==history
    assert all(sha256(p)==digest for p,digest in preserved.items())
    resumed=run_training(tmp_path,c,BINARY,resume=True,max_iteration=2)
    assert resumed['checkpoint']['total_steps']==8


def test_unbounded_replay_window_pipeline(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config)
    c['replay'].update(keep_target_rows='all',taper_exponent=1,expand_per_row=1)
    state=run_training(tmp_path,c,BINARY,max_iteration=2)
    status=load_json(tmp_path/'.internal/iterations'/'000002'/'status.json')
    snapshot=load_json(tmp_path/'snapshots'/status['snapshot_id']/'manifest.json')
    raw=[read_raw(path) for path in (tmp_path/'selfplay').rglob('*.npz')]
    generated_rows=sum(int(metadata(a)['rows']) for a in raw)
    random_rows=sum(int(metadata(a)['rows']) for a in raw if metadata(a)['model_id'].startswith('random:'))
    usable=min(random_rows,c['replay']['min_rows'])+generated_rows-random_rows
    assert snapshot['selection']['raw_rows']==generated_rows
    assert snapshot['desired_rows']==usable and usable>c['replay']['min_rows']
    selected=sum(source['metadata']['rows'] for source in snapshot['sources'])
    assert snapshot['rows']==snapshot['selection']['window_rows']==selected>=usable
    assert selected-snapshot['sources'][-1]['metadata']['rows']<usable
    assert state['checkpoint']['total_steps']==8


def test_sampled_replay_pipeline_and_resume(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config)
    c['replay'].update(min_rows=128,keep_target_rows=48)
    c['shuffle'].update(waves=3,bucket_rows=16,training_shard_rows=16)
    state=run_training(tmp_path,c,BINARY,max_iteration=2)
    for iteration in (1,2):
        status=load_json(tmp_path/'.internal/iterations'/f'{iteration:06d}'/'status.json')
        snapshot=load_json(tmp_path/'snapshots'/status['snapshot_id']/'manifest.json')
        assert snapshot['keep_prob']<1 and 0<snapshot['rows']<snapshot['window_rows']
    assert state['checkpoint']['total_steps']==8
    assert run_training(tmp_path,c,BINARY,resume=True,max_iteration=2)==state


def test_native_parity_all_sizes_rules_and_match(pipeline,gpu_config):
    root,state=pipeline
    checkpoint=load_checkpoint(root,state["checkpoint"],gpu_config)
    model=make_network(gpu_config).cuda().eval();model.load_state_dict(inference_weights(checkpoint))
    canvas=gpu_config["network"]["canvas"]
    for size in (5,6):
        for rule in ("freestyle","standard","renju"):
            command=[str(BINARY),"infer","--config",str(root/"config/effective.cfg"),"--model",str(root/state["model"]["path"]),
                     "--model-id",state["model"]["id"],"--device","cuda:0","--size",str(size),"--rule",rule,
                     "--moves",f"0,{canvas},1"]
            native=json.loads(subprocess.run(command,text=True,capture_output=True,check=True).stdout)
            with torch.inference_mode():
                p,v=model(*(torch.from_numpy(x).unsqueeze(0).cuda() for x in example_inputs(canvas,size,rule,(0,canvas,1))))
            np.testing.assert_allclose(native["raw_logits"],p[0,0].cpu(),rtol=2e-4,atol=2e-5)
            np.testing.assert_allclose(native["raw_wdl"],v[0].float().softmax(0).cpu(),rtol=2e-4,atol=2e-5)
    model_b=root/load_json(root/".internal/iterations"/"000001"/"status.json")["model"]["path"]
    _,result=evaluate(load_evaluation_config(CONFIGS / 'smoke_test',match=True),BINARY,root,model_b=model_b,size=5,rule="renju",games=4)
    assert result["result"]["complete"] and len(result["result"]["games"])==4
    assert {g["winner"] for g in result["result"]["games"]}<={-1,0,1}
    assert all(set(g["root_visits"])=={100} for g in result["result"]["games"])
    _,result=evaluate(load_evaluation_config(CONFIGS / 'smoke_test'),BINARY,root,size=5,rule="freestyle",moves="0,5,1,6,2,7,3,8,4")
    assert result["result"]["terminal"] and result["result"]["value"]==-1


def test_root_d4_probabilities_and_policy_temperatures(pipeline,gpu_config):
    from etazero.symmetry import apply_symmetry
    root,state=pipeline
    checkpoint=load_checkpoint(root,state['checkpoint'],gpu_config)
    model=make_network(gpu_config).cuda().eval();model.load_state_dict(inference_weights(checkpoint))
    canvas=gpu_config['network']['canvas']
    obs,globals=example_inputs(canvas,5,'renju',(0,canvas))
    tensor=torch.from_numpy(obs).unsqueeze(0).cuda()
    global_tensor=torch.from_numpy(globals).unsqueeze(0).cuda()
    inverse=(0,3,2,1,4,5,6,7)
    legal=torch.from_numpy(((obs[0]>0)&(obs[1]==0)&(obs[2]==0)).flatten()).cuda()
    policies,values=[],[]
    with torch.inference_mode():
        for symmetry in range(8):
            logits,value=model(apply_symmetry(tensor,symmetry).contiguous(),global_tensor)
            restored=apply_symmetry((logits[:,0]+.2*(logits[:,5]-logits[:,0])).reshape(1,canvas,canvas),inverse[symmetry]).flatten(1)
            policies.append(torch.softmax((restored/1.7).masked_fill(~legal,-torch.inf),dim=1))
            values.append(value.float().softmax(1))
    expected_policy=torch.stack(policies).mean(0)[0].cpu().numpy()
    expected_wdl=torch.stack(values).mean(0)[0].cpu().numpy()
    config=load_evaluation_config(CONFIGS / 'smoke_test')
    config['evaluation'].update(root_num_symmetries_to_sample=8,nn_policy_temperature=1.7,
                               root_policy_temperature_early=2,root_policy_temperature=1.5,
                               temperature_halflife=7,temperature_early=0.8,temperature=0.1)
    _,payload=evaluate(config,BINARY,root,size=5,rule='renju',moves='0,5')
    result=payload['result'];crop=[y*canvas+x for y in range(5) for x in range(5)]
    np.testing.assert_allclose(result['network_policy'],expected_policy[crop],rtol=3e-4,atol=2e-5)
    np.testing.assert_allclose(result['network_wdl'],expected_wdl,rtol=3e-4,atol=2e-5)
    root_temperature=1.5+0.5*0.5**(2/7*19/5)
    expected_search=expected_policy**(1/root_temperature);expected_search/=expected_search.sum()
    np.testing.assert_allclose(result['search_policy'],expected_search[crop],rtol=3e-4,atol=2e-5)
    assert result['root_visits']==100 and result['simulations']==99
    assert sum(result['visits'])==99 and result['nn_requests']>=9


@pytest.mark.parametrize('precision', ['float32','auto'])
def test_single_d4_root_only_and_inference_precision(tmp_path,pipeline,gpu_config,precision):
    from etazero.symmetry import apply_symmetry
    root,state=pipeline;canvas=gpu_config['network']['canvas']
    # LibTorch runs the published scripted artifact; FP16 graph fusion can differ from eager.
    model=torch.jit.load(str(root/state['model']['path'])).cuda().eval()
    obs,globals=example_inputs(canvas,5,'renju',(1,canvas))
    tensor=torch.from_numpy(obs).unsqueeze(0).cuda();global_tensor=torch.from_numpy(globals).unsqueeze(0).cuda()
    legal=((obs[0]>0)&(obs[1]==0)&(obs[2]==0)).flatten()
    crop=[y*canvas+x for y in range(5) for x in range(5)]
    config=load_evaluation_config(CONFIGS / 'smoke_test')
    config['evaluation'].update(visits=1,max_playouts=1,inference_precision=precision,nn_randomize=False)
    for symmetry in (0,1,5,7):
        config['evaluation']['nn_symmetry']=symmetry
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=precision=='auto'):
            logits,wdl,optimistic,_=model(apply_symmetry(tensor,symmetry).contiguous(),global_tensor)
            logits=logits.float()+.2*(optimistic.float()-logits.float())
        restored=apply_symmetry(logits.reshape(1,canvas,canvas),(0,3,2,1,4,5,6,7)[symmetry]).flatten().float()
        expected=restored.masked_fill(~torch.from_numpy(legal).cuda(),-torch.inf).softmax(0).cpu().numpy()
        _,payload=evaluate(config,BINARY,root,size=5,rule='renju',moves='1,5')
        result=payload['result']
        np.testing.assert_allclose(result['network_policy'],expected[crop],rtol=4e-4,atol=3e-5)
        np.testing.assert_allclose(result['network_wdl'],wdl[0].float().softmax(0).cpu(),rtol=4e-4,atol=3e-5)
        assert result['inference_precision']==('float16' if precision=='auto' else 'float32')
        assert result['initial_visits']==0 and result['root_visits']==result['new_playouts']==1
        assert result['simulations']==sum(result['visits'])==0 and result['nn_requests']==2
    config['evaluation'].update(visits=10,max_playouts=100,max_time=0)
    _,timed=evaluate(config,BINARY,root,size=5,rule='renju',moves='1,5')
    assert timed['result']['root_visits']==timed['result']['new_playouts']==2
    assert timed['result']['simulations']==1 and timed['result']['stopped_early']
    config['evaluation']['max_playouts']=0
    _,empty=evaluate(config,BINARY,root,size=5,rule='renju',moves='1,5')
    assert empty['result']['action']==-1 and empty['result']['new_playouts']==empty['result']['root_visits']==0
    assert len(empty['result']['policy'])==25 and not any(empty['result']['policy'])
    assert empty['result']['nn_requests']==1 # Only the explicitly requested raw diagnostic.


def test_match_same_bot_clear_and_different_bot_reuse(tmp_path,pipeline):
    from etazero.arena import single_match
    root,state=pipeline;a=root/state['model']['path']
    b=root/load_json(root/'logs/iterations/000001.json')['model']['path']
    config=load_evaluation_config(CONFIGS / 'smoke_test',True)
    config['match'].update(inference_precision='auto',search_threads=4,server_threads=2)
    _,same=single_match(config,BINARY,a,a,tmp_path/'same')
    assert len(same)==4
    assert all(g['same_bot'] and not any(g['initial_visits']) for g in same)
    assert all(g['inference_precision']=='float16' and set(g['new_playouts'])=={100} for g in same)
    _,different=single_match(config,BINARY,a,b,tmp_path/'different')
    assert len(different)==4 and all(not g['same_bot'] for g in different)
    assert any(v>0 for g in different for v in g['initial_visits'])
    for g in different:
        assert g['inference_precision']=='float16' and set(g['root_visits'])=={100}
        assert all(initial+new==visits for initial,new,visits in zip(g['initial_visits'],g['new_playouts'],g['root_visits']))


class LeafFailureModel(torch.nn.Module):
    def __init__(self,canvas):
        super().__init__()
        from etazero.schema import CONTRACT_ID
        self.canvas=canvas;self.contract=CONTRACT_ID

    @torch.jit.export
    def metadata(self) -> tuple[int,str]:
        return self.canvas,self.contract

    def forward(self,x,globals):
        torch._assert(x[:,1:3].sum()==0,'injected CUDA leaf failure')
        return x[:,0].flatten(1)*0,x[:,0].flatten(1)[:,:3]*0,x[:,0].flatten(1)*0,x[:,0].sum((1,2))*0


def test_real_cuda_parallel_leaf_failure_releases_waiters(tmp_path,gpu_config):
    canvas=gpu_config['network']['canvas'];model=tmp_path/'failure.pt'
    torch.jit.script(LeafFailureModel(canvas).cuda().eval()).save(str(model))
    config=load_evaluation_config(CONFIGS / 'smoke_test')
    config['evaluation'].update(search_threads=8,server_threads=2,cache_entries=0)
    native={**config,'network':{'canvas':canvas}};write_native(native,tmp_path/'effective.cfg')
    command=[str(BINARY),'evaluate','--config',str(tmp_path/'effective.cfg'),'--model',str(model),
             '--model-id','leaf-failure','--device','cuda:0','--size','5','--rule','freestyle']
    result=subprocess.run(command,text=True,capture_output=True,timeout=30)
    (tmp_path/'stdout.log').write_text(result.stdout);(tmp_path/'stderr.log').write_text(result.stderr)
    assert result.returncode==1 and 'injected CUDA leaf failure' in result.stderr
    assert 'leaked pending visits' not in result.stderr


def test_learner_resume_matches_uninterrupted_updates(tmp_path,pipeline,gpu_config):
    root,_=pipeline
    plan=load_json(root/".internal/iterations"/"000001"/"plan.json")
    status=load_json(root/".internal/iterations"/"000001"/"status.json")
    plan["snapshot_id"]=status["snapshot_id"]
    base=plan["input_checkpoint"]
    dirs=[tmp_path/"full",tmp_path/"resumed"]
    for dest in dirs:
        shutil.copytree(root/"checkpoints",dest/"checkpoints")
        shutil.copytree(root/"snapshots"/plan["snapshot_id"],dest/"snapshots"/plan["snapshot_id"])
    logs=[]
    def log(event,**fields):
        if event=="update":
            logs.append(fields)
    deterministic=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        full,done=train_iteration(dirs[0],gpu_config,plan,base,log)
        assert done
        logs.clear()
        partial,done=train_iteration(dirs[1],gpu_config,plan,base,log,lambda:len(logs)>=1)
        assert not done and partial['step']==1
        midway=load_checkpoint(dirs[1],partial,gpu_config)
        assert midway['optimization']['lookahead_counter']==1
        resumed,done=train_iteration(dirs[1],gpu_config,plan,base,log)
        assert done and resumed["total_steps"]==4
        a=load_checkpoint(dirs[0],full,gpu_config);b=load_checkpoint(dirs[1],resumed,gpu_config)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key],b["model"][key],rtol=0,atol=0)
        for key in a["optimizer"]["state"]:
            for field in a['optimizer']['state'][key]:
                torch.testing.assert_close(a["optimizer"]["state"][key][field],
                                           b["optimizer"]["state"][key][field],rtol=0,atol=0)
        assert a["reader"]==b["reader"]
        for key in a['optimization']['swa']:
            torch.testing.assert_close(a['optimization']['swa'][key],b['optimization']['swa'][key],rtol=0,atol=0)
        for key in a['optimization']['slow']:
            torch.testing.assert_close(a['optimization']['slow'][key],b['optimization']['slow'][key],rtol=0,atol=0)
        assert a['optimization']['norms']==b['optimization']['norms']
        torch.testing.assert_close(a["rng"]["torch"],b["rng"]["torch"],rtol=0,atol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize('kind', ['sgd', 'adamw'])
def test_optimizer_checkpoint_d4_and_swa_export(tmp_path,pipeline,gpu_config,kind):
    root,_=pipeline
    c=copy.deepcopy(gpu_config);c['optimizer']['kind']=kind
    plan=load_json(root/'.internal/iterations'/'000001'/'plan.json')
    plan['snapshot_id']=load_json(root/'.internal/iterations'/'000001'/'status.json')['snapshot_id']
    plan['train_steps']=5
    destinations=[tmp_path/'full',tmp_path/'resumed']
    for destination in destinations:
        shutil.copytree(root/'snapshots'/plan['snapshot_id'],destination/'snapshots'/plan['snapshot_id'])
    bases=[initialize(destination,c,root/plan['input_checkpoint']['path']) for destination in destinations]
    updates=[]
    def log(event,**fields):
        if event=='update':updates.append(fields)
    full,done=train_iteration(destinations[0],c,plan,bases[0],log);assert done
    uninterrupted=[u['symmetry'] for u in updates]
    updates.clear()
    partial,done=train_iteration(destinations[1],c,plan,bases[1],log,lambda:len(updates)>=1)
    assert not done
    resumed,done=train_iteration(destinations[1],c,plan,bases[1],log);assert done
    assert uninterrupted==[u['symmetry'] for u in updates]
    a=load_checkpoint(destinations[0],full,c);b=load_checkpoint(destinations[1],resumed,c)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    for key in a['optimization']['swa']:
        torch.testing.assert_close(a['optimization']['swa'][key],b['optimization']['swa'][key],rtol=0,atol=0)
    for key in a['optimizer']['state']:
        for field in a['optimizer']['state'][key]:
            torch.testing.assert_close(a['optimizer']['state'][key][field],b['optimizer']['state'][key][field],rtol=0,atol=0)
    assert a['optimization']['lookahead_counter']==0 and a['optimization']['swa']['n_averaged']==2
    assert a['optimization']['swa_samples']==c['training']['batch_size']
    for key in a['optimization']['slow']:
        torch.testing.assert_close(a['model'][key],a['optimization']['slow'][key],rtol=0,atol=0)
    assert any(not torch.equal(a['model'][k],v) for k,v in inference_weights(a).items())
    (destinations[0]/'config').mkdir()
    write_native(c,destinations[0]/'config/effective.cfg')
    info=export_model(destinations[0],c,full)
    assert info['weights']=='swa' and info['swa_samples']==2
    assert verify_export(destinations[0],info,c['network']['canvas'])==info


@pytest.mark.parametrize('amp',['float16','bfloat16'])
@pytest.mark.parametrize('kind',['sgd','adamw'])
@pytest.mark.parametrize('compiled',[False,True])
def test_real_amp_updates(tmp_path,pipeline,gpu_config,amp,kind,compiled,monkeypatch):
    from torch._functorch import config as autograd_config
    donation_before=autograd_config.donated_buffer
    root,_=pipeline;c=copy.deepcopy(gpu_config);c['training'].update(amp=amp,compile=compiled)
    c['optimizer']['kind']=kind
    plan=load_json(root/".internal/iterations"/"000001"/"plan.json");plan["train_steps"]=2
    plan["snapshot_id"]=load_json(root/".internal/iterations"/"000001"/"status.json")["snapshot_id"]
    shutil.copytree(root/"snapshots"/plan["snapshot_id"],tmp_path/"snapshots"/plan["snapshot_id"])
    base=initialize(tmp_path,c)
    if amp=="float16":
        saved=load_checkpoint(tmp_path,base,c)
        model=make_network(c).cuda();model.load_state_dict(saved["model"])
        optimizer=optimizer_for(model,c);optimizer.load_state_dict(saved["optimizer"])
        scaler=torch.amp.GradScaler("cuda",init_scale=2**32)
        restore_rng(saved["rng"])
        optimization=Optimization(model,c,optimizer,saved['optimization'])
        base=commit_checkpoint(tmp_path,c,model,optimizer,scaler,0,0,0,None,base,[],optimization)
        del optimization,model,optimizer,scaler,saved
    forwards=[];events=[]
    original=make_network
    def instrumented(config):
        model=original(config)
        model.policy_head.register_forward_hook(lambda *a:forwards.append(1))
        return model
    monkeypatch.setattr("etazero.training.make_network",instrumented)
    result,done=train_iteration(tmp_path,c,plan,base,lambda event,**fields:events.append((event,fields)))
    assert autograd_config.donated_buffer==donation_before
    assert done and result["total_steps"]==2
    assert len(forwards)==2
    if amp=="float16":
        assert any(event=="amp_overflow" for event,_ in events)
    checkpoint=load_checkpoint(tmp_path,result,c)
    skipped=sum(fields['amp_skipped'] for event,fields in events if event=='update')
    assert checkpoint['optimizer_steps']==2-skipped
    assert checkpoint['total_samples']==2*plan['batch_size']
    assert checkpoint['optimization']['round_batches']==2
    assert checkpoint['optimization']['swa']['n_averaged']==1
    assert checkpoint['optimization']['lookahead_counter']==0
    assert all(torch.isfinite(v).all() for v in checkpoint["model"].values())


@pytest.mark.parametrize('amp', ['float16', 'bfloat16'])
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('training', [False, True])
def test_learner_autocast_heads_are_fp32(gpu_config, amp, compiled, training):
    from etazero.network import TrainingForward
    c=copy.deepcopy(gpu_config)
    model=make_network(c).cuda().train(training)
    with torch.no_grad():model.value_head.out.bias[0]=1e5
    observations,globals=example_inputs(6,5,'renju',(0,6))
    obs=torch.from_numpy(observations).unsqueeze(0).cuda()
    global_tensor=torch.from_numpy(globals).unsqueeze(0).cuda()
    policy=obs[:,0].flatten(1)/25
    dtypes=[]
    for head in (model.policy_head,model.value_head):
        for layer in head.modules():
            if isinstance(layer,(torch.nn.Conv2d,torch.nn.Linear)):
                layer.register_forward_hook(lambda module,args,out:dtypes.append(out.dtype))
    trunk=[]
    model.stem.register_forward_hook(lambda module,args,out:trunk.append(out.dtype))
    forward=TrainingForward(model,8,False)
    if compiled:
        forward=torch.compile(forward,fullgraph=True,dynamic=False)
    dtype=torch.float16 if amp=='float16' else torch.bfloat16
    with torch.autocast('cuda',dtype=dtype):
        outputs=forward(obs,global_tensor,policy,policy,torch.ones(1,device='cuda'),
                        torch.tensor([[1.,0.,0.]],device='cuda'),torch.tensor([[[1.,0.,0.]]*3],device='cuda'),torch.ones(1,device='cuda'))
    assert trunk==[dtype]
    assert len(dtypes)==7 and all(x==torch.float32 for x in dtypes)
    assert all(x.dtype==torch.float32 for x in outputs)
    assert all(torch.isfinite(x).all() for x in outputs)
    outputs[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('compiled', [False, True])
def test_amp_skip_resume_preserves_consumption_and_next_update(tmp_path,pipeline,gpu_config,compiled,monkeypatch):
    root,_=pipeline;c=copy.deepcopy(gpu_config)
    c['training'].update(amp='float16',compile=compiled,train_steps=5)
    c['optimizer'].update(norm_only_at_print=False,lookahead_print=True)
    plan=load_json(root/'.internal/iterations/000001/plan.json');plan['train_steps']=5
    plan['snapshot_id']=load_json(root/'.internal/iterations/000001/status.json')['snapshot_id']
    dirs=[tmp_path/'full',tmp_path/'resumed']
    bases=[]
    for dest in dirs:
        shutil.copytree(root/'snapshots'/plan['snapshot_id'],dest/'snapshots'/plan['snapshot_id'])
        base=initialize(dest,c)
        saved=load_checkpoint(dest,base,c)
        model=make_network(c).cuda();model.load_state_dict(saved['model'])
        optimizer=optimizer_for(model,c);optimizer.load_state_dict(saved['optimizer'])
        optimization=Optimization(model,c,optimizer,saved['optimization'])
        scaler=torch.amp.GradScaler('cuda',init_scale=1)
        restore_rng(saved['rng'])
        bases.append(commit_checkpoint(dest,c,model,optimizer,scaler,0,0,0,None,base,[],optimization))
    original=make_network
    inject={'remaining':1}
    def instrumented(config):
        model=original(config)
        def overflow_once(grad):
            if inject['remaining']:
                inject['remaining']-=1
                return torch.full_like(grad,float('inf'))
            return grad
        model.stem.weight.register_hook(overflow_once)
        return model
    monkeypatch.setattr('etazero.training.make_network',instrumented)
    events=[]
    def log(event,**fields):
        if event=='update': events.append(fields)
    deterministic=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        full,done=train_iteration(dirs[0],c,plan,bases[0],log);assert done
        assert [e['amp_skipped'] for e in events]==[True,False,False,False,False]
        inject['remaining']=1;events.clear()
        partial,done=train_iteration(dirs[1],c,plan,bases[1],log,lambda:len(events)==1)
        assert not done and partial['optimizer_steps']==0 and partial['total_samples']==8
        mid=load_checkpoint(dirs[1],partial,c)
        assert mid['optimization']['lookahead_counter']==1 and mid['optimization']['swa_samples']==8
        resumed,done=train_iteration(dirs[1],c,plan,bases[1],log);assert done
        assert [e['amp_skipped'] for e in events]==[True,False,False,False,False]
        a=load_checkpoint(dirs[0],full,c);b=load_checkpoint(dirs[1],resumed,c)
        assert a['reader']==b['reader'] and a['scaler']==b['scaler']
        assert a['total_samples']==b['total_samples']==40 and a['optimizer_steps']==b['optimizer_steps']==4
        for name in a['model']:
            torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
        for name in ('norms','norm_sums','norm_weights','consumed_samples','optimizer_steps',
                     'round_batches','lookahead_counter','swa_samples'):
            assert a['optimization'][name]==b['optimization'][name]
        for name in a['optimization']['swa']:
            torch.testing.assert_close(a['optimization']['swa'][name],b['optimization']['swa'][name],rtol=0,atol=0)
        for name in a['optimization']['slow']:
            torch.testing.assert_close(a['optimization']['slow'][name],b['optimization']['slow'][name],rtol=0,atol=0)
        for group in a['optimizer']['state']:
            for name in a['optimizer']['state'][group]:
                torch.testing.assert_close(a['optimizer']['state'][group][name],b['optimizer']['state'][group][name],rtol=0,atol=0)
        torch.testing.assert_close(a['rng']['torch'],b['rng']['torch'],rtol=0,atol=0)
        for x,y in zip(a['rng']['cuda'],b['rng']['cuda']):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("phase",["shuffle","export","after_publish"])
def test_stage_failure_recovery(tmp_path,gpu_config,monkeypatch,phase):
    import etazero.runtime as runtime
    c=copy.deepcopy(gpu_config);c["training"]["replay_ratio"]=1000
    with monkeypatch.context() as patch:
        if phase=="shuffle":
            patch.setattr(runtime,"build_snapshot",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("injected stage failure")))
        elif phase=="export":
            original=runtime.export_model
            def failed_export(root,config,checkpoint):
                if checkpoint["iteration"]:
                    raise RuntimeError("injected stage failure")
                return original(root,config,checkpoint)
            patch.setattr(runtime,"export_model",failed_export)
        else:
            original=runtime.Controller.publish
            def failed_commit(self,model,elapsed):
                original(self,model,elapsed)
                if model["checkpoint"]["iteration"]:
                    raise RuntimeError("injected stage failure")
            patch.setattr(runtime.Controller,"publish",failed_commit)
        with pytest.raises(RuntimeError,match="injected stage failure"):
            run_training(tmp_path,c,BINARY,max_iteration=2)
    shards={p:sha256(p) for p in (tmp_path/"selfplay").rglob("*.npz")}
    plan_hash=sha256(tmp_path/".internal/iterations"/"000001"/"plan.json")
    checkpoint=None
    progress=tmp_path/".internal/iterations"/"000001"/"learner.json"
    if progress.exists():
        checkpoint=load_json(progress)["checkpoint"]
    state=run_training(tmp_path,c,BINARY,resume=True,max_iteration=2)
    assert state["checkpoint"]["total_steps"]==8 and state["target_rows"]==pytest.approx(state["replay_origin_rows"]+0.032)
    assert sha256(tmp_path/".internal/iterations"/"000001"/"plan.json")==plan_hash
    assert_preserved(tmp_path,shards)
    assert list((tmp_path/"selfplay"/"iteration_000002").rglob("*.npz"))
    if checkpoint:
        current=load_json(tmp_path/".internal/iterations"/"000001"/"status.json")["checkpoint"]
        assert (current==checkpoint)==(phase=="after_publish")


def test_forced_process_kill_during_training(tmp_path,gpu_config):
    configs=tmp_path/"configs";shutil.copytree(CONFIGS,configs)
    (configs/"smoke_test"/"train.cfg.local").write_text("[training]\ntrain_steps=100\ncheckpoint_every=2\nreplay_ratio=8\n")
    root=tmp_path/"run";c=load_config(configs/"smoke_test",run_dir=root)
    env=os.environ.copy();env["PYTHONPATH"]=str(ROOT/"python")
    command=[sys.executable,"-m","etazero","run","--config-dir",str(configs/"smoke_test"),
             "--run-dir",str(root),"--iterations","1"]
    with (tmp_path/"stdout").open("w") as out,(tmp_path/"stderr").open("w") as err:
        process=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
        progress=root/".internal/iterations"/"000001"/"learner.json";deadline=time.monotonic()+60
        try:
            while time.monotonic()<deadline:
                if progress.exists() and load_json(progress)["checkpoint"]["step"]>=2:
                    process.kill();break
                if process.poll() is not None:
                    pytest.fail("Controller exited before the forced-kill point: "+(tmp_path/"stderr").read_text())
                time.sleep(0.005)
            else:
                pytest.fail("Timed out waiting for a committed training checkpoint")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    persisted=load_json(progress)["checkpoint"]
    assert 2<=persisted["step"]<100
    plan_hash=sha256(root/".internal/iterations"/"000001"/"plan.json")
    raw={p:sha256(p) for p in (root/"selfplay").rglob("*.npz")}
    state=run_training(root,c,BINARY,resume=True,max_iteration=1)
    assert state["checkpoint"]["step"]==100 and state["checkpoint"]["total_samples"]==800
    assert state["target_rows"]==state["replay_origin_rows"] and sha256(root/".internal/iterations"/"000001"/"plan.json")==plan_hash
    assert_preserved(root,raw)
    assert not (root/persisted["path"]).exists()


def test_multiple_selfplay_workers(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config);c["devices"]["selfplay"]="cuda:0,cuda:0"
    c["training"]["replay_ratio"]=1000
    root=tmp_path/"first";repeat=tmp_path/"repeat"
    run_training(root,c,BINARY,max_iteration=1)
    run_training(repeat,c,BINARY,max_iteration=1)
    def seeds(path):
        events=[json.loads(line) for line in (path/"logs/events.jsonl").read_text().splitlines()]
        # Compare the same bootstrap attempt. Stochastic row counts may change later backfill attempts.
        starts=[e for e in events if e['event']=='worker_start' and e['iteration']==0]
        first_attempt=min(e['attempt'] for e in starts)
        return {e['worker']['id']:e['worker']['seed'] for e in starts if e['attempt']==first_attempt}
    assert seeds(root)==seeds(repeat) and len(set(seeds(root).values()))==2
    sources=[];games=set()
    for path in (root/"selfplay").rglob("*.npz"):
        a=read_raw(path);m=json.loads(a["metadata"].tobytes());sources.append(m)
        _,starts,_=game_row_ranges(a)
        for game_id in a["game_ids"][starts==0]:
            key=(m["attempt_id"],m["worker_id"],int(game_id))
            assert key not in games;games.add(key)
    assert {m["worker_id"] for m in sources}=={0,1}
    assert all(m["source_id"] for m in sources)


def test_native_selfplay_signal_flushes_complete_games(tmp_path,pipeline,gpu_config):
    root,state=pipeline;output=tmp_path/"selfplay"
    command=[str(BINARY),"selfplay","--config",str(root/"config/effective.cfg"),"--model",str(root/state["model"]["path"]),
             "--model-id",state["model"]["id"],"--device","cuda:0","--games","100",
             "--output",str(output),"--run-id","signal","--attempt-id","a","--config-id","test",
             "--source-id","test","--iteration","1","--worker","0","--seed","1"]
    with (tmp_path/"stdout").open("w") as out,(tmp_path/"stderr").open("w") as err:
        process=subprocess.Popen(command,stdout=out,stderr=err)
        deadline=time.monotonic()+30
        try:
            while time.monotonic()<deadline:
                if list(output.glob("*.npz")):
                    process.send_signal(signal.SIGINT);break
                assert process.poll() is None,(tmp_path/"stderr").read_text()
                time.sleep(0.005)
            else:
                pytest.fail("No complete shard before the signal deadline")
            assert process.wait(timeout=10)==2
        finally:
            if process.poll() is None:
                process.kill();process.wait()
    games=set()
    for path in output.glob("*.npz"):
        games.update(map(int,read_raw(path)["game_ids"]))
    assert 0<len(games)<100


def test_multiple_servers_fp16_packed_waves_and_prefetch(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config)
    c["parallelism"].update(game_threads=4,search_threads=3)
    c["inference"].update(server_threads=2,inference_precision="float16",cache_entries=64)
    c["shuffle"].update(waves=3,temp_dir=str(tmp_path/"scratch"))
    root=tmp_path/"run"
    state=run_training(root,c,BINARY,max_iteration=2)
    assert state["checkpoint"]["total_steps"]==8
    events=[json.loads(line) for line in (root/"logs/events.jsonl").read_text().splitlines()]
    services=[e for e in events if e["event"]=="inference"]
    assert services and all(len(e["rows_by_server"])==2 for e in services)
    assert any(min(e["rows_by_server"])>0 for e in services if e["iteration"]>=2)
    assert all(e["requests"]+e["cache_hits"]==e["submitted"] for e in services)
    evaluation_config=load_evaluation_config(CONFIGS / 'smoke_test',environ={
        'EVAL_SERVER_THREADS':'2','EVAL_INFERENCE_PRECISION':'float16','EVAL_CACHE_ENTRIES':'64'})
    _,evaluation=evaluate(evaluation_config,BINARY,root,size=5,rule="freestyle",moves="0,5,1")
    assert evaluation['result']['inference_precision']=='float16'
    # Repeated analysis of the same board reuses the first random orientation.
    # Raw diagnostics deliberately bypass cache and cannot establish this property.
    evaluation_config['evaluation'].update(visits=2,cache_entries=65536)
    effective=tmp_path/'cache.cfg';write_native({**evaluation_config,'network':{'canvas':6}},effective)
    command=[str(BINARY),'serve','--config',str(effective),'--model',str(root/state['model']['path']),
             '--model-id',state['model']['id'],'--device','cuda:0']
    session=subprocess.run(command,input='new 5 freestyle\nanalyze 2\nanalyze 2\nquit\n',
                           text=True,capture_output=True,check=True,timeout=30)
    (tmp_path/'cache_session.jsonl').write_text(session.stdout)
    analyses=[json.loads(line)['analysis'] for line in session.stdout.splitlines() if '"analysis"' in line]
    assert len(analyses)==2 and analyses[0]['requests']==2 and analyses[1]['requests']==0
    assert analyses[0]['network_wdl']==analyses[1]['network_wdl']
    assert analyses[0]['candidates']==analyses[1]['candidates']
    for path in (root/"selfplay").rglob("*.npz"):
        assert read_raw(path)["observations"].shape[1:]==(5,5) # ceil(6*6/8)
    assert not list((tmp_path/"scratch").iterdir())
    scripted=torch.jit.load(str(root/state["model"]["path"])).cuda().eval()
    assert state["model"]["inference_precision"]=="float16"
    previous_autocast=torch._C._jit_set_autocast_mode(False)
    try:
        for size in (5,6):
            for rule in ("freestyle","standard","renju"):
                moves=(0,6,1)
                command=[str(BINARY),"infer","--config",str(root/"config/effective.cfg"),"--model",str(root/state["model"]["path"]),
                         "--model-id",state["model"]["id"],"--device","cuda:0","--size",str(size),"--rule",rule,"--moves","0,6,1"]
                native=json.loads(subprocess.run(command,text=True,capture_output=True,check=True).stdout)
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                    p,v,_,_=scripted(*(torch.from_numpy(x).unsqueeze(0).cuda() for x in example_inputs(6,size,rule,moves)))
                assert p.dtype==v.dtype==torch.float16
                np.testing.assert_allclose(native["raw_logits"],p[0].cpu(),rtol=3e-3,atol=3e-3)
                np.testing.assert_allclose(native["raw_wdl"],v[0].float().softmax(0).cpu(),rtol=3e-3,atol=3e-3)
    finally:
        torch._C._jit_set_autocast_mode(previous_autocast)


@pytest.mark.parametrize('amp',['off','float16','bfloat16'])
def test_compiled_training_resume(tmp_path,pipeline,gpu_config,amp):
    root,_=pipeline;c=copy.deepcopy(gpu_config);c['training'].update(compile=True,amp=amp)
    assert_compiled_partial_resume(tmp_path,root,c)


def assert_compiled_partial_resume(tmp_path,root,c):
    # CUDA convolution backward may choose nondeterministic kernels at these
    # full preset widths. Exact recovery is audited with deterministic kernels,
    # as in the eager recovery test, without altering production execution.
    deterministic=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        _assert_compiled_partial_resume(tmp_path,root,c)
    finally:
        torch.use_deterministic_algorithms(deterministic)


def _assert_compiled_partial_resume(tmp_path,root,c):
    plan=load_json(root/'.internal/iterations'/'000001'/'plan.json')
    plan['snapshot_id']=load_json(root/'.internal/iterations'/'000001'/'status.json')['snapshot_id']
    destinations=[tmp_path/'full',tmp_path/'resumed']
    bases=[]
    for destination in destinations:
        shutil.copytree(root/'snapshots'/plan['snapshot_id'],destination/'snapshots'/plan['snapshot_id'])
        bases.append(initialize(destination,c,root/plan['input_checkpoint']['path']))
    logs=[]
    def log(event,**fields):
        if event=='update':logs.append(fields)
    full,done=train_iteration(destinations[0],c,plan,bases[0],log);assert done
    logs.clear()
    partial,done=train_iteration(destinations[1],c,plan,bases[1],log,lambda:len(logs)>=1)
    assert not done and partial['step']==1
    resumed,done=train_iteration(destinations[1],c,plan,bases[1],log);assert done
    a=load_checkpoint(destinations[0],full,c);b=load_checkpoint(destinations[1],resumed,c)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert a['reader']==b['reader'] and a['scaler']==b['scaler']


@pytest.mark.parametrize('amp',['off','float16'])
def test_inference_normalization_preserves_convolution_precision(gpu_config,amp):
    c=copy.deepcopy(gpu_config);c['network'].update(canvas=15,channels=192,blocks=5)
    with torch.random.fork_rng():
        torch.manual_seed(53)
        model=make_network(c).cuda().eval()
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer,MaskedBatchNorm):
                    layer.running_mean.uniform_(-0.5,0.5);layer.running_std.uniform_(0.1,2)
                    layer.weight.uniform_(-0.5,0.5);layer.bias.uniform_(-0.2,0.2)
        eager=inference_network(model)
        inference=torch.jit.script(eager)
        probes=[example_inputs(15,size,rule,(0,15,1)) for size in (9,15) for rule in ('freestyle','standard','renju')]
        obs=tuple(torch.from_numpy(np.stack(x)).cuda() for x in zip(*probes))
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=True),torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=amp=='float16'):
            for inputs in (obs,tuple(x[:1] for x in obs)):
                expected=eager(*inputs)
                assert model.fp32_heads and not eager.model.fp32_heads
                assert expected[0].dtype==(torch.float16 if amp=='float16' else torch.float32)
                # Include the optimized graph selected after JIT profiling.
                for _ in range(6):
                    for reference,actual in zip(expected,inference(*inputs)):
                        rtol,atol=(3e-3,3e-3) if amp=='float16' else (0,0)
                        torch.testing.assert_close(actual,reference,rtol=rtol,atol=atol)


@pytest.mark.parametrize('precision',['float32','float16'])
@pytest.mark.parametrize('architecture,predict_q_values',[('nbt',False),('plain',False),('transformer',False),('transformer',True)])
def test_export_tf32_with_nontrivial_normalization(tmp_path,gpu_config,precision,architecture,predict_q_values):
    c=copy.deepcopy(gpu_config)
    c['network'].update(architecture=architecture,canvas=15,channels=128 if architecture=='plain' else 192,
                        blocks=10 if architecture=='plain' else 5,predict_q_values=predict_q_values)
    c['environment'].update(sizes='11,15',size_weights='1,1')
    c['inference']['inference_precision']=precision
    (tmp_path/'models').mkdir();(tmp_path/'checkpoints').mkdir()
    write_native(c,tmp_path/'config/effective.cfg')
    with torch.random.fork_rng(),torch.backends.cudnn.flags(enabled=True,allow_tf32=True):
        torch.manual_seed(53)
        model=make_network(c).cuda().eval()
        with torch.no_grad():
            if architecture=='transformer':
                from etazero.network import init_weights
                for block in model.blocks:
                    init_weights(block.post.conv.weight,0.3,activation='relu')
                for layer in model.modules():
                    if isinstance(layer,torch.nn.RMSNorm):
                        layer.weight.uniform_(0.5,1.5)
            for layer in model.modules():
                if isinstance(layer,MaskedBatchNorm):
                    layer.running_mean.uniform_(-2,2);layer.running_std.uniform_(0.3,2)
                    layer.weight.uniform_(-0.5,0.5);layer.bias.uniform_(-0.2,0.2)
        optimizer=optimizer_for(model,c)
        checkpoint=commit_checkpoint(tmp_path,c,model,optimizer,
                                     torch.amp.GradScaler('cuda',enabled=False),0,0,0,None,None,[],
                                     Optimization(model,c,optimizer))
        info=export_model(tmp_path,c,checkpoint)
        # Numerical parity belongs to this independent test, not model publication.
        inference=inference_network(model)
        scripted=torch.jit.load(str(tmp_path/info['path']))
        probes=[example_inputs(15,size,rule,(0,15)) for size in (11,15)
                for rule in ('freestyle','standard','renju')]
        inputs=tuple(torch.from_numpy(np.stack(x)).cuda() for x in zip(*probes))
        with torch.inference_mode():
            expected=inference(*inputs)
            for _ in range(3):
                for actual,reference in zip(scripted(*inputs),expected):
                    torch.testing.assert_close(actual,reference,rtol=2e-4,atol=2e-5)
                    assert torch.isfinite(actual).all()
    assert verify_export(tmp_path,info,c['network']['canvas'])==info
    assert info['normalization']==('masked_fixup_bias' if architecture=='transformer' else 'precomputed_inv_std')
    assert info['inference_precision']==precision
    assert sha256(tmp_path/checkpoint['path'])==checkpoint['sha256']


def test_persistent_native_protocol_and_signal(tmp_path,pipeline,gpu_config):
    from etazero.native import NativeWorker
    root,state=pipeline;model=state['model'];config=root/'config/effective.cfg'
    command=[str(BINARY),'worker','--config',str(config),'--device','cuda:0',
             '--run-id','protocol','--config-id','test','--source-id','test','--worker','0']
    service=NativeWorker(command,tmp_path/'stderr')
    def wait_result():
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            result=service.take()
            if result is not None:return result
            time.sleep(0.005)
        pytest.fail('Native worker response timeout')
    try:
        output=tmp_path/'带空格 data\nfirst'
        service.submit([str(root/model['path']),model['id'],1,str(output),'first',1,5,'network'],tmp_path/'first.jsonl')
        assert wait_result()['code']==0
        assert len({int(g) for p in output.glob('*.npz') for g in read_raw(p)['game_ids']})==1
        service.release()
        output=tmp_path/'second'
        service.submit([str(root/model['path']),model['id'],100,str(output),'second',2,6,'network'],tmp_path/'second.jsonl')
        deadline=time.monotonic()+30
        while not list(output.glob('*.npz')):
            assert time.monotonic()<deadline and service.process.poll() is None
            time.sleep(0.005)
        service.process.send_signal(signal.SIGINT)
        assert wait_result()['code']==2
        assert list(output.glob('*.npz'))
        for path in output.glob('*.npz'):read_raw(path)
    finally:
        service.close()
    assert service.process.returncode==2
    idle=NativeWorker(command,tmp_path/'idle.stderr');idle.close()
    assert idle.process.returncode==0


def test_snapshot_retention_and_restore(tmp_path,gpu_config):
    from etazero.reader import BatchReader
    c=copy.deepcopy(gpu_config);c['shuffle']['snapshot_keep']=1;c['training']['replay_ratio']=1000
    state=run_training(tmp_path,c,BINARY,max_iteration=3)
    statuses=[load_json(tmp_path/'.internal/iterations'/f'{i:06d}'/'status.json') for i in range(1,4)]
    snapshots=[tmp_path/'snapshots'/s['snapshot_id'] for s in statuses]
    assert [p.joinpath('data').is_dir() for p in snapshots]==[False,False,True]
    raw={p:sha256(p) for p in (tmp_path/'selfplay').rglob('*.npz')}
    reader=BatchReader(snapshots[0],8,1,1)
    try:
        assert len(reader.next()['value'])==8
    finally:reader.close()
    assert raw=={p:sha256(p) for p in raw}
    assert state['checkpoint']['total_steps']==12
    resumed=run_training(tmp_path,c,BINARY,resume=True,max_iteration=4)
    assert resumed['checkpoint']['total_steps']==16


def test_bootstrap_backfill_balance_and_replay_reset(tmp_path,gpu_config,monkeypatch):
    import etazero.runtime as runtime
    from etazero.data import Catalog, metadata, training_view
    c=copy.deepcopy(gpu_config)
    c['selfplay']['bootstrap_games']=2
    c['replay']['min_rows']=150
    c['training']['replay_ratio']=8
    c['opening']['probability']=1
    root=tmp_path/'run'
    first_launch=runtime.Controller.launch
    def partial_bootstrap(self,plan,games):
        if plan['iteration']==0:
            first_launch(self,plan,1)
            raise RuntimeError('injected partial bootstrap')
        first_launch(self,plan,games)
    with monkeypatch.context() as patch:
        patch.setattr(runtime.Controller,'launch',partial_bootstrap)
        with pytest.raises(RuntimeError,match='partial bootstrap'):
            run_training(root,c,BINARY,max_iteration=3)
    bootstrap_before={p:sha256(p) for p in (root/'selfplay'/'iteration_000000').rglob('*.npz')}
    assert len({int(g) for p in bootstrap_before for g in read_raw(p)['game_ids']})==1
    # Fail immediately after the durable reset, then verify it is never reset again.
    commit=runtime.Controller.state
    def fail_after_reset(self,state):
        commit(self,state)
        if state['iteration']==2:
            raise RuntimeError('injected after replay reset')
    with monkeypatch.context() as patch:
        patch.setattr(runtime.Controller,'state',fail_after_reset)
        with pytest.raises(RuntimeError,match='after replay reset'):
            run_training(root,c,BINARY,resume=True,max_iteration=3)
    origin=load_json(root/'.internal/state.json')['replay_origin_rows']
    assert origin>=150
    state=run_training(root,c,BINARY,resume=True,max_iteration=3)
    assert state['replay_origin_rows']==origin and state['target_rows']==origin+8
    assert state['checkpoint']['total_steps']==12
    bootstrap=set();balanced=0
    for path in (root/'selfplay').rglob('*.npz'):
        a=read_raw(path);m=metadata(a)
        if m['iteration_id']<=1:
            assert m['model_id'].startswith('random:') and a['opening_moves'].sum()==0
            if m['iteration_id']==0:bootstrap.update((m['attempt_id'],m['worker_id'],int(g)) for g in a['game_ids'])
        else:
            assert m['model_id'].startswith('iteration_')
            assert (a['opening_status']==1).all() and (a['opening_moves']>0).all()
            assert len(training_view(a)['value'])==m['rows']<m['plies']
            _,starts,_=game_row_ranges(a)
            balanced+=int((starts==0).sum())
    assert len(bootstrap)==2 and balanced>=2
    assert_preserved(root,bootstrap_before)
    assert all(not p.exists() for p in bootstrap_before)
    catalog=Catalog(root,state['run_id'],runtime.fingerprint(c))
    try:
        assert catalog.iteration_counts(0)[1]==2
        assert catalog.iteration_counts(2)[1]>0 and catalog.iteration_counts(3)[1]>0
    finally:
        catalog.close()
    plans={i:load_json(root/'.internal/iterations'/f'{i:06d}'/'plan.json') for i in range(4)}
    assert plans[0]['train_steps']==0 and plans[1]['target_rows']==150
    assert plans[2]['target_rows']==origin+4 and plans[3]['target_rows']==origin+8
    hashes={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')}
    assert run_training(root,c,BINARY,max_iteration=3)==state
    assert hashes=={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')}


def test_autoexp_real_cuda_shared_weights_queue_resume_and_plots(tmp_path):
    from etazero.experiment import experiment_plan, run_experiment
    from etazero.plotting import run_history
    from etazero.schema import CONTRACT_ID
    umbrella = tmp_path/'umbrella'; umbrella.mkdir()
    (umbrella/'exp.cfg').write_text('[experiment]\nmax_iteration = 2\nmax_seconds = 0\narm_gpus = 0\nshared_init = true\n')
    for name in ('a', 'b'):
        arm = umbrella/name; arm.mkdir()
        (arm/'run.cfg').write_text(f'[run]\nextends = smoke_test\nrun_dir = {tmp_path/name}\n')
    plan = experiment_plan(umbrella, environ={}, work_dir=tmp_path/'scheduler')
    assert run_experiment(plan, BINARY) == 0
    records = [load_json(tmp_path/name/'.internal/run.json') for name in ('a', 'b')]
    assert records[0]['id'] != records[1]['id']
    assert records[0]['weights_initialization'] == records[1]['weights_initialization']
    checkpoints = []
    hashes = {}
    for name in ('a', 'b'):
        root = tmp_path/name
        state = load_json(root/'.internal/state.json')
        assert state['iteration'] == 3 and state['checkpoint']['total_steps'] == 8
        initial = next(p for p in (root/'checkpoints').glob('iteration_000000*.pt'))
        checkpoints.append(torch.load(initial, map_location='cpu', weights_only=False))
        assert (root/'training.png').stat().st_size > 10000
        assert (root/'logs/performance.png').stat().st_size > 10000
        history = run_history(root)
        assert [r['iteration'] for r in history] == [0, 1, 2]
        assert history[0]['steps'] == 0 and history[2]['steps'] == 4
        assert history[2]['avg_game_length'] >= history[2]['avg_rows_per_game']
        hashes.update({p: sha256(p) for p in (root/'selfplay').rglob('*.npz')})
    assert all(torch.equal(value, checkpoints[1]['model'][key]) for key, value in checkpoints[0]['model'].items())
    assert all(c['contract'] == CONTRACT_ID and c['total_steps'] == 0 for c in checkpoints)
    assert run_experiment(plan, BINARY) == 0
    assert all(sha256(path) == digest for path, digest in hashes.items())
    # Increase only the cumulative invocation budget, keeping effective arm config unchanged.
    plan['settings']['max_iteration'] = 3
    assert run_experiment(plan, BINARY) == 0
    for name in ('a', 'b'):
        state = load_json(tmp_path/name/'.internal/state.json')
        assert state['iteration'] == 4 and state['checkpoint']['total_steps'] == 12
    assert all(sha256(path) == digest for path, digest in hashes.items())


def test_autoexp_signal_and_real_cuda_learner_restore(tmp_path):
    umbrella = tmp_path/'umbrella'; arm = umbrella/'a'; arm.mkdir(parents=True)
    root = tmp_path/'run'; work = tmp_path/'scheduler'
    (umbrella/'exp.cfg').write_text('[experiment]\nmax_iteration = 1\nmax_seconds = 0\narm_gpus = 0\nshared_init = false\n')
    (arm/'run.cfg').write_text(f'[run]\nextends = smoke_test\nrun_dir = {root}\n')
    (arm/'train.cfg').write_text('[training]\ntrain_steps = 200\ncheckpoint_every = 20\n')
    command = [sys.executable, '-m', 'etazero.experiment', '--config-dir', str(umbrella),
               '--work-dir', str(work), '--binary', str(BINARY)]
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'python')}
    log = tmp_path/'runner.log'
    with log.open('w') as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+90
        try:
            progress = root/'.internal/iterations/000001/learner.json'
            while time.monotonic() < deadline:
                if progress.exists():
                    process.send_signal(signal.SIGTERM); break
                if process.poll() is not None:
                    pytest.fail('Scheduler exited before learner checkpoint: '+log.read_text())
                time.sleep(.01)
            else:
                pytest.fail('Timed out waiting for learner checkpoint')
            assert process.wait(timeout=30) == 130, log.read_text()
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=30)
    old = load_json(progress)['checkpoint']
    old_hash = sha256(root/old['path'])
    assert old['total_steps'] >= 20
    assert load_json(work/'.internal/status.json')['arms']['a']['status'] == 'interrupted'
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    state = load_json(root/'.internal/state.json')
    assert state['iteration'] == 2 and state['checkpoint']['total_steps'] == 200
    assert not (root/old['path']).exists()
    assert_preserved(root,{root/old['path']:old_hash})
    from etazero.plotting import run_history
    assert run_history(root)[-1]['steps'] == 200
    assert (root/'training.png').is_file()


def test_autoexp_cumulative_time_budget_and_iteration_extension(tmp_path):
    from etazero.experiment import experiment_plan, run_experiment
    umbrella = tmp_path/'umbrella'; arm = umbrella/'a'; arm.mkdir(parents=True)
    root = tmp_path/'run'
    (umbrella/'exp.cfg').write_text('[experiment]\nmax_iteration = 0\nmax_seconds = 1\narm_gpus = 0\nshared_init = false\n')
    (arm/'run.cfg').write_text(f'[run]\nextends = smoke_test\nrun_dir = {root}\n')
    plan = experiment_plan(umbrella, environ={}, work_dir=tmp_path/'scheduler')
    assert run_experiment(plan, BINARY) == 0
    events = [json.loads(line) for line in (root/'logs/events.jsonl').read_text().splitlines()]
    assert load_json(root/'.internal/state.json')['elapsed_seconds'] >= 1
    info = load_json(root/'.internal/run.json')
    checksum = sha256(root/'.internal/state.json')
    assert run_experiment(plan, BINARY) == 0
    assert sha256(root/'.internal/state.json') == checksum
    plan['settings'].update(max_seconds=0, max_iteration=1)
    assert run_experiment(plan, BINARY) == 0
    state = load_json(root/'.internal/state.json')
    assert state['iteration'] == 2 and state['checkpoint']['total_steps'] == 4
    assert load_json(root/'.internal/run.json') == info


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_compiler_timing_and_sigint_rollback(tmp_path,gpu_config,monkeypatch,algorithm):
    import etazero.runtime as runtime
    from etazero.compiler import compilation_seconds
    from etazero.experiment import arm_progress
    if algorithm=='muzero':
        from test_muzero_pipeline import small_config
        c=small_config()
    else:
        c=copy.deepcopy(gpu_config)
    c['training']['compile']=True
    root=tmp_path/'run'
    before_compile=compilation_seconds()
    base=run_training(root,c,BINARY,max_iteration=1)
    def verify_time(state):
        rows=[load_json(p) for p in sorted((root/'logs/iterations').glob('*.json'))]
        assert len(rows)==state['iteration']
        assert state['elapsed_seconds']==pytest.approx(sum(r['seconds'] for r in rows))
        for row in rows:
            assert row['seconds']>0 and row['compile_seconds']>=0
            assert row['wall_seconds']==pytest.approx(row['seconds']+row['compile_seconds'])
        assert load_json(root/'models/current.json')['elapsed_seconds']==state['elapsed_seconds']
        return rows
    rows=verify_time(base)
    measured=compilation_seconds()-before_compile
    assert measured>0 and sum(r['compile_seconds'] for r in rows)==pytest.approx(measured,abs=.001)
    assert rows[0]['compile_seconds']==0  # Bootstrap work is retained.
    assert rows[1]['compile_seconds']>0
    original=runtime.train_iteration
    def interrupted(root,config,plan,checkpoint,log,stopping):
        def signal_after_update(event,**fields):
            log(event,**fields)
            if event=='update':os.kill(os.getpid(),signal.SIGINT)
        return original(root,config,plan,checkpoint,signal_after_update,stopping)
    with monkeypatch.context() as patch:
        patch.setattr(runtime,'train_iteration',interrupted)
        assert run_training(root,c,BINARY,resume=True,max_iteration=2)==base
    pending=load_json(root/'.internal/iterations/000002/learner.json')['checkpoint']
    assert pending['step']==1 and pending!=base['checkpoint']
    assert load_json(root/'.internal/state.json')==base
    assert not (root/'logs/iterations/000002.json').exists()
    # Simulate a fresh interpreter's in-memory compiler cache on resume.
    torch._dynamo.reset()
    resumed=run_training(root,c,BINARY,resume=True,max_iteration=2)
    rows=verify_time(resumed)
    assert resumed['checkpoint']['total_steps']==8
    assert rows[-1]['compile_seconds']>0
    assert load_json(root/'.internal/iterations/000002/plan.json')['input_checkpoint']==base['checkpoint']
    assert not (root/pending['path']).exists()
    assert list((root/'.internal/discarded').glob('*/'+pending['path']))
    assert rows[-1]['seconds']==pytest.approx(resumed['elapsed_seconds']-base['elapsed_seconds'])
    events=[json.loads(line) for line in (root/'logs/events.jsonl').read_text().splitlines()]
    assert events[-1]['active_seconds']>resumed['elapsed_seconds']
    train=[e for e in events if e['event']=='phase_end' and e['phase']=='train']
    assert all(e['seconds']>0 and e['wall_seconds']>=e['seconds'] for e in train)
    assert arm_progress({'name':'test','run_dir':str(root),'config':c},
                      dict(max_iteration=0,max_seconds=resumed['elapsed_seconds']),None)=='complete'


def test_compiler_timer_covers_lazy_backward_and_recompile(gpu_config):
    from etazero.compiler import WorkTimer,compilation_seconds
    from torch._dynamo.utils import calculate_time_spent
    from torch._inductor import config as inductor_config
    # Cold compilation makes lazy backward observable even on a warm machine.
    with inductor_config.patch(force_disable_caches=True):
        def objective(x):return (x.sin()*x).sum()
        forward=torch.compile(objective,fullgraph=True,dynamic=False)
        for size in (17,33):
            x=torch.randn(size,device='cuda',requires_grad=True)
            torch.cuda.synchronize()
            before=calculate_time_spent()
            timer=WorkTimer(True)
            loss=forward(x)
            after_forward=calculate_time_spent()
            loss.backward();torch.cuda.synchronize()
            after_backward=calculate_time_spent()
            elapsed=timer.finish()
            assert after_forward['entire_frame_compile']>before.get('entire_frame_compile',0)
            assert after_backward['entire_backward_compile']>after_forward.get('entire_backward_compile',0)
            assert elapsed['compile_seconds']==pytest.approx(after_backward['total_wall_time']-before['total_wall_time'])
            assert elapsed['seconds']>0
            torch.testing.assert_close(x.grad,x.detach().sin()+x.detach()*x.detach().cos())
            warm_start=compilation_seconds()
            forward(x).backward();torch.cuda.synchronize()
            assert compilation_seconds()==warm_start


def test_eval_100_visits_and_arena_resume_elo(tmp_path,pipeline,gpu_config,monkeypatch):
    from etazero.arena import single_match, PairStore
    from etazero.elo import write_ratings
    root,state=pipeline
    ec=load_evaluation_config(CONFIGS / 'smoke_test')
    _,result=evaluate(ec,BINARY,root,size=5,rule='renju',moves='0,6')
    assert result['result']['root_visits']==100
    assert result['result']['simulations']==99 and sum(result['result']['visits'])==99
    assert 0<=result['result']['action']<25 and len(result['result']['policy'])==25
    mc=load_evaluation_config(CONFIGS / 'smoke_test',True)
    a=root/state['model']['path']
    b=root/load_json(root/'logs/iterations/000001.json')['model']['path']
    output=tmp_path/'match'
    original=PairStore.accept
    interrupted=False
    def stop_after_game(self,event):
        nonlocal interrupted
        original(self,event)
        if event['type']=='game' and not interrupted:
            interrupted=True
            raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr(PairStore,'accept',stop_after_game)
        with pytest.raises(KeyboardInterrupt):single_match(mc,BINARY,a,b,output)
    before={p:sha256(p) for p in (output/'pairs').rglob('*.json')}
    assert before
    _,games=single_match(mc,BINARY,a,b,output)
    assert len(games)==4 and all(set(g['root_visits'])=={100} for g in games)
    assert all(sha256(p)==digest for p,digest in before.items())
    preserved={p:sha256(p) for p in (output/'pairs').rglob('*.json')}
    _,extended=single_match(mc,BINARY,a,b,output,games=8)
    assert len(extended)==8 and all(sha256(p)==v for p,v in preserved.items())
    ratings=write_ratings(output,10)
    assert ratings['games']==8
    for ext in ('json','csv','png','svg'):assert (output/f'elo.{ext}').stat().st_size>100
    # CLI discovery uses committed history and preserves result identity on restart.
    out=tmp_path/'elo_history'
    command=['bash',str(ROOT/'scripts/run.sh'),'arena','--data',str(root),'--output',str(out),
             '--config-dir',str(CONFIGS / 'smoke_test'),'--games','4','--bootstrap-samples','10']
    first=subprocess.run(command,env={**os.environ,'PYTHONPATH':str(ROOT/'python')},text=True,capture_output=True,timeout=120)
    assert first.returncode==0,first.stdout+first.stderr
    hashes={p:sha256(p) for p in (out/'pairs').rglob('*.json')}
    second=subprocess.run(command,text=True,capture_output=True,timeout=120)
    assert second.returncode==0,second.stdout+second.stderr
    assert all(sha256(p)==digest for p,digest in hashes.items())


def test_mixed_rule_openings_and_training_feature_dropout(tmp_path,pipeline,gpu_config):
    from etazero.data import training_view
    from etazero.schema import unpack_observations
    root,state=pipeline
    c=copy.deepcopy(gpu_config)
    c['parallelism'].update(game_threads=1,search_threads=1)
    c['inference']['server_threads']=1
    c['opening']['probability']=1
    c['symmetry']['nn_randomize']=False
    # Stress final multiplicity using existing valid surprise-weighting settings.
    c['search']['cheap_search_probs']=0
    c['reduce_visits']['reduce_visits']=False
    c['surprise_weighting'].update(policy_surprise_data_weight=1,value_surprise_data_weight=0)
    datasets=[]
    for dropout in (0.,.5,1.):
        c['environment']['forbidden_feature_dropout_prob']=dropout
        cfg=tmp_path/f'{dropout}.cfg';write_native(c,cfg)
        output=tmp_path/f'rows_{dropout}'
        command=[str(BINARY),'selfplay','--config',str(cfg),'--model',str(root/state['model']['path']),
                 '--model-id',state['model']['id'],'--device','cuda:0','--games','18','--output',str(output),
                 '--run-id','test','--attempt-id','features','--config-id','test','--source-id','test',
                 '--iteration','2','--worker','0','--seed','123']
        subprocess.run(command,text=True,capture_output=True,check=True,timeout=180)
        records=sorted((read_raw(p) for p in output.glob('*.npz')),key=lambda a:int(a['game_ids'][0]))
        assert set(np.concatenate([a['rules'] for a in records]))=={0,1,2}
        assert set(np.concatenate([a['sizes'] for a in records]))=={5,6}
        flags=[];independent_repeat=False
        for a in records:
            assert (a['opening_status']==1).all() and (a['opening_moves']>0).all()
            view=training_view(a)
            flags.extend(view['globals'][view['globals'][:,1]==1,3])
            start=0
            for repeats in a['row_repeats'][a['sample_indices']]:
                block=a['forbidden_input'][start:start+repeats]
                if len(set(block))==2:independent_repeat=True
                start+=repeats
            assert (a['globals'][a['globals'][:,1]==1,3]==1).all()
            if dropout==1:
                obs=unpack_observations(view['obs'],6)
                assert not obs[:,3:5].any()
        assert flags
        if dropout==.5:
            assert set(flags)=={0,1}
            assert independent_repeat
        else: assert set(flags)=={1-dropout}
        datasets.append(records)
    # With one search/game thread and identical model/seed, writing augmented rows
    # must not alter the openings, search targets, sampled sizes/rules or results.
    for other in datasets[1:]:
        assert len(other)==len(datasets[0])
        for a,b in zip(datasets[0],other):
            for key in a.keys()-{'forbidden_input','side_forbidden_input','metadata'}:
                np.testing.assert_array_equal(a[key],b[key])


@pytest.mark.parametrize('lookback',[1,3])
def test_reduce_visits_pcr_native_training_and_resume(tmp_path,gpu_config,lookback):
    c=copy.deepcopy(gpu_config)
    # Exercise extreme-value/PCR transitions in random-evaluator trajectories,
    # independent of the randomly initialized learner's value-head confidence.
    c['selfplay']['bootstrap_games']=32
    c['opening']['probability']=0
    c['policy_init'].update(policy_init=False,policy_after=False,policy_on_failure=False)
    c['parallelism'].update(game_threads=1,search_threads=1)
    c['surprise_weighting'].update(policy_surprise_data_weight=0,value_surprise_data_weight=0)
    c['search'].update(full_search_visits=41,cheap_search_probs=0.5,cheap_search_visits=4)
    c['reduce_visits'].update(reduce_visits=True,reduce_visits_threshold=0,
                             reduce_visits_threshold_lookback=lookback,reduced_visits_min=2,reduced_visits_weight=0.1)
    root=tmp_path/'run'
    state=run_training(root,c,BINARY,max_iteration=2)
    assert state['checkpoint']['total_steps']==8 and state['checkpoint']['total_samples']==64
    files={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')}
    assert run_training(root,c,BINARY,resume=True,max_iteration=2)==state
    state=run_training(root,c,BINARY,resume=True,max_iteration=3)
    assert state['checkpoint']['total_steps']==12 and state['checkpoint']['total_samples']==96
    assert all(sha256(p)==digest for p,digest in files.items())
    cheap_count=reduced_count=reduced_weight_count=reduced_after_cheap=cheap_reply_count=0
    for path in sorted((root/'selfplay').rglob('*.npz')):
        a=read_raw(path)
        sampled={int(index):row for row,index in enumerate(a['sample_indices'])}
        for game,(lo,hi) in enumerate(zip(a['game_offsets'][:-1],a['game_offsets'][1:])):
            history=[];previous_cheap=False
            ol=a['observation_offsets'][game]
            assert a['opening_moves'][game]==0
            for index in range(lo,hi):
                cheap=bool(a['cheap_search'][index])
                if cheap:
                    cheap_count+=1
                    assert a['target_weights'][index]==0
                    assert a['row_repeats'][index]==0
                    assert 0 <= a['simulations'][index] <= 3
                    assert index not in sampled
                    if index>lo and index-1 in sampled:
                        prior=sampled[index-1]
                        assert a['opponent_policy_weights'][prior]==1
                        # Native policy targets are quantized int16 weights;
                        # cheap replies still provide a nonempty opponent target.
                        reply=a['opponent_policies'][prior]
                        assert reply.dtype==np.int16 and (reply>=0).all() and reply.sum()>0
                        cheap_reply_count+=1
                else:
                    # Independently reconstruct certainty from stored completed WDL,
                    # rather than importing or translating the C++ limits helper.
                    certain=0
                    if len(history)>=lookback:
                        window=np.asarray(history[-lookback:])
                        if (window>0).all() or (window<0).all():
                            certain=float(np.min(np.abs(window)))
                    reduction=certain**2
                    expected_weight=1-0.9*reduction
                    unrounded_visits=41-39*reduction
                    # NPZ WDL is float32; allow only its rounding uncertainty at
                    # half-integer boundaries, not an extra search simulation.
                    root_visits=int(a['simulations'][index])+1
                    if index in sampled:
                        assert int(a['visits'][sampled[index]].sum())+1==root_visits
                    assert abs(root_visits-unrounded_visits)<=0.5002
                    assert a['simulations'][index]==root_visits-1
                    assert a['target_weights'][index]==pytest.approx(expected_weight,abs=2e-6)
                    if a['target_weights'][index]<1-1e-6:
                        reduced_weight_count+=1
                    if root_visits<41:
                        reduced_count+=1
                        reduced_after_cheap+=int(previous_cheap)
                player=int(a['players'][ol+index-lo])
                wdl=a['search_wdl'][index].astype(np.float64)
                history.append(player*(wdl[0]-wdl[2]))
                previous_cheap=cheap
    assert cheap_count>0 and reduced_weight_count>0 and cheap_reply_count>0
    if lookback==1:
        assert reduced_count>0 and reduced_after_cheap>0
    # For lookback=3, a small but real weight reduction may round the visit cap
    # back to 41. The per-turn checks above still verify both formulas exactly.


class OpeningChoiceModel(torch.nn.Module):
    def __init__(self,canvas,ascending,value_logit):
        super().__init__()
        from etazero.schema import CONTRACT_ID
        self.canvas=canvas;self.contract=CONTRACT_ID
        self.ascending=ascending;self.value_logit=value_logit

    @torch.jit.export
    def metadata(self) -> tuple[int,str]:
        return self.canvas,self.contract

    def forward(self,x,globals):
        order=torch.arange(self.canvas*self.canvas,device=x.device,dtype=x.dtype)
        logits=order[None,:].expand(x.shape[0],-1)*(100. if self.ascending else -100.)
        wdl=torch.zeros((x.shape[0],3),device=x.device,dtype=x.dtype)
        wdl[:,0]=self.value_logit
        return logits,wdl,logits,x[:,0].sum((1,2))*0


def test_match_opening_uses_both_models_by_source_roles(tmp_path,gpu_config):
    canvas=gpu_config['network']['canvas']
    paths=[]
    for name,ascending,value_logit in [('a',False,0.),('b',True,1.3862943611198906)]:
        model=tmp_path/(name+'.pt')
        torch.jit.script(OpeningChoiceModel(canvas,ascending,value_logit).cuda().eval()).save(str(model))
        paths.append(model)
    c=load_evaluation_config(CONFIGS / 'smoke_test',True,environ={})
    c['match'].update(visits=2,game_threads=1,nn_randomize=False,cache_entries=0,inference_precision='float32')
    c['opening'].update(policy_init=True,policy_init_mean=20,policy_temperature=1,rejection_probability=0)
    write_native({**c,'network':{'canvas':canvas}},tmp_path/'effective.cfg')
    # generator is reference Black assignment: half A Black, half B Black.
    tasks=tmp_path/'tasks.txt'
    tasks.write_text(''.join(f'{i} {123+i*7919} {i%2} 3 -1\n' for i in range(8)))
    command=[str(BINARY),'match','--config',str(tmp_path/'effective.cfg'),'--model',str(paths[0]),
             '--model-b',str(paths[1]),'--model-id','a','--model-b-id','b','--device','cuda:0',
             '--size','5','--rule','freestyle','--tasks',str(tasks)]
    run=subprocess.run(command,text=True,capture_output=True,timeout=120)
    (tmp_path/'stdout.log').write_text(run.stdout);(tmp_path/'stderr.log').write_text(run.stderr)
    assert run.returncode==0,run.stderr
    events=[json.loads(line) for line in run.stdout.splitlines()]
    openings=[event for event in events if event.get('type')=='opening']
    assert len(openings)==8 and len([e for e in events if e.get('type')=='game'])==16
    assert set(i for e in openings for i in e['balance_evaluators'])=={0,1}
    assert sum(e['policy_moves'] for e in openings)>4
    for e in openings:
        assert e['reference_black_a']==(e['generator']==0)
        selected_a=(e['balance_evaluators'][-1]==e['generator'])
        # A emits uniform WDL => W-L=0; B emits [4,1,1]/6 => W-L=.5.
        assert e['value']==pytest.approx(0. if selected_a else .5,abs=1e-6)
        balanced=e['balanced_moves'];moves=e['moves'];occupied=set(moves[:balanced])
        assert e['policy_evaluators']==[(balanced+i)%2 for i in range(e['policy_moves'])]
        for i,move in enumerate(moves[balanced:]):
            bot_is_a=e['policy_evaluators'][i]==e['generator']
            # Models deliberately prefer opposite legal board corners.
            legal=[a for a in range(25) if a not in occupied]
            assert move==(min(legal) if bot_is_a else max(legal))
            occupied.add(move)
    (tmp_path/'opening_events.json').write_text(json.dumps(openings,indent=2))


def test_target_five_sizes_three_rules_cuda_and_unsupported_requests(tmp_path,gpu_config):
    c=copy.deepcopy(gpu_config);c['network']['canvas']=15
    c['environment'].update(sizes='15,14,13,12,11',size_weights='100,10,5,3,1')
    c['inference'].update(inference_precision='float32',server_threads=1)
    model=make_network(c).cuda().eval()
    path=tmp_path/'model.pt';torch.jit.script(inference_network(model)).save(str(path))
    config=tmp_path/'effective.cfg';write_native(c,config)
    command=[str(BINARY),'infer','--config',str(config),'--model',str(path),
             '--model-id','sizes','--device','cuda:0','--size','15','--rule','renju','--moves','0,15,1']
    evidence=[]
    for size in (15,14,13,12,11):
        for rule in ('freestyle','standard','renju'):
            current=command.copy();current[current.index('--size')+1]=str(size);current[current.index('--rule')+1]=rule
            result=subprocess.run(current,text=True,capture_output=True,check=True,timeout=30)
            native=json.loads(result.stdout)
            obs,globals=example_inputs(15,size,rule,(0,15,1))
            with torch.inference_mode():
                policy,value=model(torch.from_numpy(obs)[None].cuda(),torch.from_numpy(globals)[None].cuda())
            np.testing.assert_allclose(native['raw_logits'],policy[0,0].cpu(),rtol=2e-4,atol=2e-5)
            np.testing.assert_allclose(native['raw_wdl'],value[0].float().softmax(0).cpu(),rtol=2e-4,atol=2e-5)
            evidence.append({'size':size,'rule':rule,'wdl':native['raw_wdl']})
    # Native command parsing must reject rectangular strings, rather than stoi
    # silently accepting the initial integer and running on a square board.
    for flag,value,error in [('--size','15x14','Invalid integer'),('--rule','vc1_b','Unknown Gomoku rule'),('--size','16','Invalid canvas')]:
        current=command.copy();current[current.index(flag)+1]=value
        result=subprocess.run(current,text=True,capture_output=True,timeout=30)
        assert result.returncode==1 and error in result.stderr
    (tmp_path/'size_rule_results.json').write_text(json.dumps(evidence,indent=2))


def test_auxiliary_outputs_train_export_native_cuda(tmp_path,pipeline,gpu_config):
    root,state=pipeline
    c=copy.deepcopy(gpu_config)
    saved=load_checkpoint(root,state['checkpoint'],c)
    model=make_network(c).cuda().eval();model.load_state_dict(inference_weights(saved))
    assert saved['model']['value_head.out.weight'].shape[0]==13
    path=root/state['model']['path']
    scripted=torch.jit.load(str(path),map_location='cuda')
    for size in (5,6):
        for rule in ('freestyle','standard','renju'):
            obs,globals=example_inputs(6,size,rule,(0,6,1))
            inputs=torch.from_numpy(obs)[None].cuda(),torch.from_numpy(globals)[None].cuda()
            with torch.inference_mode():
                policy,value,td,raw=model.forward_all(*inputs)
                exported=scripted(*inputs)
            expected=(policy[:,0],value,policy[:,5],.5*torch.nn.functional.softplus(raw/2))
            for a,b in zip(exported,expected): torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
            command=[str(BINARY),'infer','--config',str(root/'config/effective.cfg'),'--model',str(path),
                     '--model-id',state['model']['id'],'--device','cuda:0','--size',str(size),
                     '--rule',rule,'--moves','0,6,1']
            native=json.loads(subprocess.run(command,check=True,text=True,capture_output=True).stdout)
            np.testing.assert_allclose(native['raw_optimistic_logits'],policy[0,5].cpu(),rtol=2e-4,atol=2e-5)
            assert native['raw_shortterm_value_stdev']==pytest.approx(float(expected[3][0]),rel=2e-4,abs=2e-5)
    events=[json.loads(x) for x in (root/'logs/events.jsonl').read_text().splitlines()]
    updates=[e for e in events if e['event']=='update']
    names=('policy_loss','opponent_policy_loss','soft_policy_loss','soft_opponent_policy_loss','value_loss',
           'td_value_long_loss','td_value_mid_loss','td_value_short_loss','long_optimistic_policy_loss',
           'short_optimistic_policy_loss','shortterm_value_error_loss')
    assert len(updates)==8
    for e in updates:
        assert e['loss']==pytest.approx(sum(e[n] for n in names),rel=2e-6)
        assert all(np.isfinite(e[n]) for n in names)
    assert any(e['shortterm_value_error_loss']>0 for e in updates)


def test_native_side_supervision_cuda_training_and_export(tmp_path,gpu_config):
    from etazero.data import metadata,training_view
    from etazero.shuffle import build_snapshot
    from etazero.storage import save_json
    c=copy.deepcopy(gpu_config)
    c['replay'].update(min_rows=11,keep_target_rows='all')
    c['training'].update(batch_size=11,train_steps=2)
    c['shuffle'].update(bucket_rows=11,training_shard_rows=11)
    root=tmp_path/'run';directory=root/'selfplay/worker'
    subprocess.run([str(ROOT/'build/record_contract_test'),str(directory)],check=True,capture_output=True,text=True)
    path=next(directory.glob('*.npz'));raw=read_raw(path);view=training_view(raw)
    assert not view['full_game_weight'][-2:].any()
    snapshot=build_snapshot(root,1,[{'path':str(path.relative_to(root)), 'sha256':sha256(path),'metadata':metadata(raw)}],c)
    initial=initialize(root,c)
    plan={'iteration':1,'snapshot_id':snapshot,'train_steps':2,'batch_size':11}
    events=[]
    reference,done=train_iteration(root,c,plan,initial,lambda event,**fields:events.append({'event':event,**fields}))
    assert done and reference['total_samples']==22 and reference['optimizer_steps']==2
    assert all(np.isfinite(e['loss']) for e in events if e['event']=='update')
    assert all(e['td_value_short_loss']>0 for e in events if e['event']=='update')
    write_native(c,root/'config/effective.cfg')
    info=export_model(root,c,reference)
    assert verify_export(root,info,c['network']['canvas'])==info
    save_json(root/'side_validation.json',{'rows':len(view['value']), 'side_rows':int((view['full_game_weight']==0).sum()),
                                        'checkpoint':reference,'export':info,'events':events})


class DiamondCudaModel(torch.nn.Module):
    def __init__(self,canvas,fail_after=-1):
        super().__init__()
        from etazero.schema import CONTRACT_ID
        self.canvas=canvas;self.contract=CONTRACT_ID;self.fail_after=fail_after
        black=torch.full((canvas*canvas,),-1000.0);white=black.clone()
        black[0]=black[1]=0;white[canvas+1]=white[canvas+2]=0
        self.register_buffer('black',black);self.register_buffer('white',white)

    @torch.jit.export
    def metadata(self) -> tuple[int,str]:
        return self.canvas,self.contract

    def forward(self,x,globals):
        if self.fail_after>=0:
            torch._assert(torch.all(x[:,1:3].sum((1,2,3))<self.fail_after),'injected CUDA shared graph failure')
        black=(x[:,1].sum((1,2))==x[:,2].sum((1,2))).unsqueeze(1)
        policy=torch.where(black,self.black.unsqueeze(0),self.white.unsqueeze(0))
        value=torch.zeros((x.shape[0],3),device=x.device,dtype=x.dtype)
        return policy,value,policy,x[:,0].sum((1,2))*0+.5


def test_graph_real_cuda_shared_nodes_leak_and_cache(tmp_path,gpu_config):
    canvas=gpu_config['network']['canvas'];model=tmp_path/'diamond.pt'
    torch.jit.script(DiamondCudaModel(canvas).cuda().eval()).save(str(model))
    config=load_evaluation_config(CONFIGS / 'smoke_test')
    config['evaluation'].update(visits=500,search_threads=1,server_threads=1,cache_entries=0,
        nn_randomize=False,nn_symmetry=0,root_num_symmetries_to_sample=1,inference_precision='float32')
    cases={}
    for name,graph,leak,threads,cache in (
        ('tree',False,0,1,0),('tree_cached',False,0,1,4096),('graph',True,0,1,0),('leak',True,1,1,0),
        ('parallel',True,0,8,0),('cached',True,0,1,4096)):
        cfg=copy.deepcopy(config);cfg['evaluation'].update(use_graph_search=graph,
            graph_search_catch_up_leak_prob=leak,search_threads=threads,server_threads=2 if threads>1 else 1,cache_entries=cache)
        native={**cfg,'network':{'canvas':canvas}};path=tmp_path/(name+'.cfg');write_native(native,path)
        command=[str(BINARY),'evaluate','--config',str(path),'--model',str(model),'--model-id','diamond',
                 '--device','cuda:0','--size','5','--rule','renju','--seed','23']
        run=subprocess.run(command,text=True,capture_output=True,timeout=60)
        (tmp_path/(name+'.stdout.log')).write_text(run.stdout);(tmp_path/(name+'.stderr.log')).write_text(run.stderr)
        assert run.returncode==0,run.stderr
        result=json.loads(run.stdout);cases[name]=result
        assert result['root_visits']==result['new_playouts']==500 and sum(result['visits'])==499
        assert result['graph_cycles']==0
        if not cache:
            assert result['cache_hits']==0
        if graph:
            assert result['graph_hits']>0
        else:
            assert result['graph_hits']==result['graph_catch_ups']==0
    assert cases['graph']['graph_catch_ups']>0 and cases['leak']['graph_catch_ups']==0
    assert cases['graph']['nn_requests']<cases['tree']['nn_requests']
    assert cases['graph']['graph_nodes']<cases['tree']['graph_nodes']
    assert cases['tree_cached']['cache_hits']>0 and cases['tree_cached']['nn_requests']<cases['tree']['nn_requests']
    assert cases['tree_cached']['graph_nodes']==cases['tree']['graph_nodes']
    assert cases['tree_cached']['visits']==cases['tree']['visits']
    # Identical one-thread graph decisions with caching on/off show that sharing is in Search itself.
    for key in ('visits','policy','wdl','graph_hits','graph_catch_ups','graph_nodes'):
        assert cases['graph'][key]==cases['cached'][key]
    (tmp_path/'cases.json').write_text(json.dumps(cases,indent=2))



def test_graph_shared_cuda_failure_releases_all_parents(tmp_path,gpu_config):
    canvas=gpu_config['network']['canvas'];model=tmp_path/'shared_failure.pt'
    torch.jit.script(DiamondCudaModel(canvas,4).cuda().eval()).save(str(model))
    config=load_evaluation_config(CONFIGS / 'smoke_test')
    config['evaluation'].update(visits=500,search_threads=8,server_threads=2,cache_entries=0,
        nn_randomize=False,nn_symmetry=0,root_num_symmetries_to_sample=1,use_graph_search=True)
    native={**config,'network':{'canvas':canvas}};write_native(native,tmp_path/'effective.cfg')
    command=[str(BINARY),'evaluate','--config',str(tmp_path/'effective.cfg'),'--model',str(model),
             '--model-id','shared-failure','--device','cuda:0','--size','5','--rule','renju']
    result=subprocess.run(command,text=True,capture_output=True,timeout=30)
    (tmp_path/'stdout.log').write_text(result.stdout);(tmp_path/'stderr.log').write_text(result.stderr)
    assert result.returncode==1 and 'injected CUDA shared graph failure' in result.stderr
    assert 'leaked pending visits' not in result.stderr


class SearchCorrectionsCudaModel(torch.nn.Module):
    def __init__(self,canvas,error):
        super().__init__()
        from etazero.schema import CONTRACT_ID
        self.canvas=canvas;self.contract=CONTRACT_ID;self.error=error
        ordinary=torch.zeros(canvas*canvas);optimistic=ordinary.clone()
        ordinary[4]=np.log(9);optimistic[1]=np.log(9)
        self.register_buffer('ordinary',ordinary);self.register_buffer('optimistic',optimistic)
        self.register_buffer('value',torch.tensor([np.log(.6),np.log(.2),np.log(.2)],dtype=torch.float32))

    @torch.jit.export
    def metadata(self) -> tuple[int,str]:
        return self.canvas,self.contract

    def forward(self,x,globals):
        zero=x[:,0].sum((1,2))*0
        return self.ordinary.unsqueeze(0)+zero.unsqueeze(1),self.value.unsqueeze(0)+zero.unsqueeze(1),self.optimistic.unsqueeze(0)+zero.unsqueeze(1),zero+self.error


def test_cuda_search_corrections_logits_error_and_terminal_stats(tmp_path,gpu_config):
    canvas=gpu_config['network']['canvas'];models={}
    for error in (0.,.5,2.):
        path=tmp_path/f'error_{error}.pt';torch.jit.script(SearchCorrectionsCudaModel(canvas,error).cuda().eval()).save(str(path));models[error]=path
    config=load_evaluation_config(CONFIGS / 'smoke_test');config['evaluation'].update(visits=1,max_playouts=1,
        inference_precision='float32',nn_randomize=False,nn_symmetry=0,search_threads=1,server_threads=1,nn_policy_temperature=1.7)
    cases={}
    def run(name,c,error=.5,moves=''):
        native={**c,'network':{'canvas':canvas}};path=tmp_path/(name+'.cfg');write_native(native,path)
        command=[str(BINARY),'evaluate','--config',str(path),'--model',str(models[error]),'--model-id',f'aux-{error}',
                 '--device','cuda:0','--size','5','--rule','renju','--moves',moves,'--seed','4']
        result=subprocess.run(command,text=True,capture_output=True,timeout=30)
        (tmp_path/(name+'.stdout.log')).write_text(result.stdout);(tmp_path/(name+'.stderr.log')).write_text(result.stderr)
        assert result.returncode==0,result.stderr
        output=json.loads(result.stdout);cases[name]=output;return output
    for error in (0.,.5,2.):
        for optimism in (0.,.2,1.):
            c=copy.deepcopy(config);c['evaluation']['root_policy_optimism']=optimism
            r=run(f'root_{error}_{optimism}',c,error)
            ordinary=np.zeros(canvas*canvas,np.float32);ordinary[4]=np.log(9)
            short=np.zeros_like(ordinary);short[1]=np.log(9)
            mixed=ordinary+(short-ordinary)*np.float32(optimism)
            legal=np.array([y*canvas+x for y in range(5) for x in range(5)])
            weights=np.exp(mixed[legal].astype(np.float64)/1.7);weights/=weights.sum()
            np.testing.assert_allclose(np.array(r['network_policy'])[legal],weights,rtol=2e-7,atol=1e-8)
            expected=.25/(error+.25/8)
            assert r['root_visits']==r['new_playouts']==1
            assert r['network_value_stdev']==error
            assert r['network_sample_weight']==pytest.approx(expected,rel=1e-12)
            assert r['search_weight']==pytest.approx(expected,rel=1e-12)
            assert r['search_weight_sq']==pytest.approx(expected*expected,rel=1e-12)
    c=copy.deepcopy(config);c['evaluation'].update(visits=2,max_playouts=2,root_policy_optimism=0,nn_policy_temperature=1,
        use_noise_pruning=False)
    # Black's four-in-a-row: the highest ordinary logit at (0,4) wins immediately.
    moves=','.join(str(v) for v in (0,canvas,1,canvas+1,2,canvas+2,3,canvas+3))
    for uncertainty in (False,True):
        c['evaluation']['use_uncertainty']=uncertainty;r=run('terminal_'+str(uncertainty),c,moves=moves)
        own=.25/(.5+.25/8) if uncertainty else 1.;leaf=8 if uncertainty else 1.
        predicted=r['network_wdl'];value=(own*(predicted[0]-predicted[2])+leaf)/(own+leaf);draw=own*predicted[1]/(own+leaf)
        expected=np.array([(1-draw+value)/2,draw,(1-draw-value)/2])
        np.testing.assert_allclose(r['wdl'],expected,rtol=1e-12,atol=1e-12)
        assert r['root_visits']==2 and sum(r['visits'])==1 and r['nn_requests']==2 # raw + root; terminal has no NN call
        assert r['search_weight']==pytest.approx(own+leaf,rel=1e-12)
        assert r['search_weight_sq']==pytest.approx(own*own+leaf*leaf,rel=1e-12)
    # Combined production settings exercise all three corrections with a weighted shared graph.
    c=copy.deepcopy(config);c['evaluation'].update(visits=500,max_playouts=500,search_threads=8,server_threads=2,
        root_num_symmetries_to_sample=8,cache_entries=1024)
    combined=run('combined_graph',c)
    assert combined['root_visits']==500 and sum(combined['visits'])==499
    assert combined['graph_hits']>0 and combined['search_weight']>0 and combined['search_weight_sq']>0
    assert all(np.isfinite(combined[k]) for k in ('value','search_weight','search_weight_sq'))
    (tmp_path/'cases.json').write_text(json.dumps(cases,indent=2))


def test_selfplay_nondefault_search_corrections_cuda(tmp_path,gpu_config):
    canvas=gpu_config['network']['canvas'];model=tmp_path/'aux.pt'
    torch.jit.script(SearchCorrectionsCudaModel(canvas,.5).cuda().eval()).save(str(model))
    for enabled in (False,True):
        c=copy.deepcopy(gpu_config);c['environment'].update(sizes='5',size_weights='1',rules='renju',rule_weights='1')
        c['opening']['probability']=0;c['policy_init']['policy_init']=False
        c['uncertainty']['use_uncertainty']=enabled;c['noise_pruning']['use_noise_pruning']=enabled
        c['optimistic_policy'].update(root_policy_optimism=.2 if enabled else 0,policy_optimism=1 if enabled else 0)
        c['parallelism'].update(game_threads=2,search_threads=4);c['inference'].update(server_threads=2,inference_precision='float32')
        path=tmp_path/str(enabled);path.mkdir();write_native(c,path/'effective.cfg')
        command=[str(BINARY),'selfplay','--config',str(path/'effective.cfg'),'--model',str(model),'--model-id','aux',
                 '--device','cuda:0','--games','2','--output',str(path/'selfplay'),'--run-id','corrections',
                 '--attempt-id',str(enabled),'--config-id',str(enabled),'--source-id','test','--iteration','2','--worker','0','--seed','6']
        result=subprocess.run(command,text=True,capture_output=True,timeout=60)
        (path/'stdout.log').write_text(result.stdout);(path/'stderr.log').write_text(result.stderr)
        assert result.returncode==0,result.stderr
        games=set();rows=0
        for shard in (path/'selfplay').glob('*.npz'):
            raw=read_raw(shard);games.update(map(int,raw['game_ids']));rows+=metadata(raw)['rows']
            assert np.isfinite(raw['network_wdl']).all() and np.isfinite(raw['search_wdl']).all()
        assert len(games)==2 and rows>0


class PdaCudaModel(torch.nn.Module):
    def __init__(self,canvas):
        super().__init__()
        from etazero.schema import CONTRACT_ID
        self.canvas=canvas;self.contract=CONTRACT_ID
        self.register_buffer('grid',torch.linspace(-.2,.2,canvas*canvas))

    @torch.jit.export
    def metadata(self) -> tuple[int,str]:
        return self.canvas,self.contract

    def forward(self,x,globals):
        advantage=globals[:,5].float()
        ordinary=self.grid[None].expand(x.size(0),-1)+advantage[:,None]*self.grid[None]
        value=torch.stack((advantage*.2,advantage*0,-advantage*.2),1)
        return ordinary,value,ordinary.flip(1),torch.full_like(advantage,.5)


@pytest.mark.parametrize('predict_q_values',[False,True])
@pytest.mark.parametrize('direct',[False,True])
def test_pda_side_reanalysis_cuda_and_real_training(tmp_path,gpu_config,direct,predict_q_values):
    from etazero.data import metadata,training_view
    from etazero.shuffle import build_snapshot
    c=copy.deepcopy(gpu_config)
    if predict_q_values:c['network'].update(architecture='transformer',channels=192,blocks=5,predict_q_values=True)
    c['environment'].update(sizes='5',size_weights='1',rules='renju',rule_weights='1')
    c['opening']['probability']=0;c['policy_init']['policy_init']=False
    c['search'].update(full_search_visits=40,cheap_search_visits=30,cheap_search_probs=.6)
    c['reduce_visits']['reduced_visits_min']=30
    c['pda'].update(normal_asymmetric_playout_prob=1,max_asymmetric_ratio=8)
    c['side_positions']['side_position_prob']=1
    c['reanalysis']=dict(use_reanalyze=True,reanalyze_prop=1,reanalyze_policy_surprise_weight=1,
                         reanalyze_value_surprise_weight=1,reanalyze_surprise_exponent=.5,
                         reanalyze_use_outcome_targets=False)
    c['surprise_weighting']['use_search_value_surprise']=direct
    c['parallelism'].update(game_threads=2,search_threads=4)
    c['inference'].update(server_threads=2,inference_precision='float32')
    c['shuffle'].update(group_rows=1024,bucket_rows=32)
    c['training'].update(batch_size=8,train_steps=2)
    c['replay'].update(min_rows=8,keep_target_rows='all',taper_exponent=1,expand_per_row=1)
    c['training']['skip_validation']=True
    from etazero.config import validate
    validate(c)
    model=tmp_path/'conditional.pt'
    scripted=torch.jit.script(PdaCudaModel(c['network']['canvas']).cuda().eval());scripted.save(str(model))
    # Fresh one-visit roots isolate actual conditional NN values from terminal
    # outcomes in deeper search, using both side-to-move signs and neutral input.
    evaluations=[]
    e=load_evaluation_config(CONFIGS / 'smoke_test')
    e['evaluation'].update(visits=1,inference_precision='float32',nn_randomize=False)
    for d,moves in ((0,''),(1,''),(1,'0'),(3,''),(3,'0')):
        e['evaluation']['playout_doubling_advantage']=d
        write_native({**c,**e},tmp_path/'evaluation.cfg')
        output=subprocess.run([str(BINARY),'evaluate','--config',str(tmp_path/'evaluation.cfg'),
             '--model',str(model),'--model-id','conditional','--device','cuda:0','--size','5','--rule','renju','--moves',moves],
             text=True,capture_output=True,check=True,timeout=30)
        value=json.loads(output.stdout);half=.5*d*(-1 if moves else 1)
        logits=np.array([.2*half,0,-.2*half]);expected=np.exp(logits);expected/=expected.sum()
        np.testing.assert_allclose(value['raw_wdl'],expected,rtol=0,atol=8e-8)
        np.testing.assert_allclose(value['wdl'],expected,rtol=0,atol=8e-8)
        evaluations.append({'doublings':d,'moves':moves,'native':value})
    (tmp_path/'conditional_evaluations.json').write_text(json.dumps(evaluations,indent=2))
    root=tmp_path/'run';root.mkdir();write_native(c,root/'effective.cfg')
    directory=root/'selfplay/worker'
    command=[str(BINARY),'selfplay','--config',str(root/'effective.cfg'),'--model',str(model),'--model-id','conditional',
             '--device','cuda:0','--games','2','--output',str(directory),'--run-id','sampling',
             '--attempt-id',str(direct),'--config-id',str(direct),'--source-id','test','--iteration','2','--worker','0','--seed','716']
    result=subprocess.run(command,text=True,capture_output=True,timeout=120)
    (root/'stdout.log').write_text(result.stdout);(root/'stderr.log').write_text(result.stderr)
    assert result.returncode==0,result.stderr
    # This case checks conservation of all generated supervision. A smaller
    # source-style replay window legitimately omits older training-row shards.
    entries=[];main_count=side_count=reanalyzed=0
    expected_gate_counts=np.zeros(2,np.int64)
    gated_main_rows=0
    for path in directory.glob('*.npz'):
        raw=read_raw(path);view=training_view(raw)
        assert (raw['globals'][:,4]==1).all() and not (raw['side_globals'][:,4:6]).any()
        for game_index,(lo,hi) in enumerate(zip(raw['game_offsets'][:-1],raw['game_offsets'][1:])):
            offset=raw['observation_offsets'][game_index]
            half=raw['globals'][offset:offset+hi-lo,5].astype(np.float64)
            logits=np.stack((.2*half,np.zeros_like(half),-.2*half),1)
            expected=np.exp(logits);expected/=expected.sum(1,keepdims=True)
            np.testing.assert_allclose(raw['network_wdl'][lo:hi],expected,rtol=0,atol=2e-7)
        reanalyzed+=int(raw['reanalyzed'].sum());side_count+=len(raw['side_players']);main_count+=len(raw['actions'])
        np.testing.assert_array_equal(raw['reanalyzed'],raw['cheap_search'])
        assert not raw['reanalysis_used_outcome'].any()
        ix=np.flatnonzero(raw['reanalyzed'])
        for turn in ix:
            game=int(np.searchsorted(raw['game_offsets'][1:],turn,side='right'))
            obs_index=int(raw['observation_offsets'][game]+turn-raw['game_offsets'][game])
            d=2*float(raw['globals'][obs_index,5]);factor=2*(2**d)/(1+2**d)
            assert raw['reanalysis_original_visits'][turn]==int(np.floor(30*factor+.5))
            sample=np.searchsorted(raw['sample_indices'],turn)
            if sample<len(raw['sample_indices']) and raw['sample_indices'][sample]==turn:
                assert raw['visits'][sample].sum()+1==int(np.floor(40*factor+.5))
                assert raw['opponent_policy_weights'][sample]==0
                assert raw['q_visits'][sample].max()>0
        assert (raw['side_q_visits'].max(1)>0).all()
        assert raw['side_q_values'].dtype==np.int16 and abs(raw['side_q_values']).max()<=32000
        assert np.isfinite(raw['side_wdl']).all() and (raw['side_wdl']>=0).all()
        np.testing.assert_allclose(raw['side_wdl'].sum(1),1,rtol=0,atol=2e-7)
        # Outcome-disabled reanalysis suppresses auxiliary complete-game
        # supervision, while unreanalyzed main rows keep it after repetition.
        game_gates=[]
        for gi,(lo,hi) in enumerate(zip(raw['game_offsets'][:-1],raw['game_offsets'][1:])):
            main_gate=np.repeat(1-raw['reanalyzed'][lo:hi],raw['row_repeats'][lo:hi]).astype(np.float32)
            gated_main_rows+=int((main_gate==0).sum())
            side_rows=int(raw['side_row_repeats'][raw['side_game_indices']==gi].sum())
            game_gates.extend((main_gate,np.zeros(side_rows,np.float32)))
        all_gates=np.concatenate(game_gates);m=metadata(raw)
        np.testing.assert_array_equal(view['full_game_weight'],all_gates[m['row_begin']:m['row_begin']+m['rows']])
        expected_gate_counts+=np.bincount(view['full_game_weight'].astype(np.int64),minlength=2)
        assert np.isfinite(view['td_value']).all()
        entries.append({'path':str(path.relative_to(root)), 'sha256':sha256(path),'metadata':metadata(raw)})
    assert reanalyzed>0 and side_count>0 and main_count>0
    assert gated_main_rows>0 and expected_gate_counts[1]>0
    # Actual generated PDA/side/reanalysis rows reach the GPU learner and exporter.
    snapshot=build_snapshot(root,1,entries,c)
    manifest=load_json(root/'snapshots'/snapshot/'manifest.json')
    actual_gate_counts=np.zeros(2,np.int64)
    for file in manifest['files']:
        with np.load(root/'snapshots'/snapshot/'data'/file['path'],allow_pickle=False) as batch:
            actual_gate_counts+=np.bincount(batch['full_game_weight'].astype(np.int64),minlength=2)
    np.testing.assert_array_equal(actual_gate_counts,expected_gate_counts)
    initial=initialize(root,c);original=load_checkpoint(root,initial,c)['model']['linear_global.weight'].clone()
    events=[]
    reference,done=train_iteration(root,c,{'iteration':1,'snapshot_id':snapshot,'train_steps':2,'batch_size':8},initial,
                                   lambda event,**fields:events.append({'event':event,**fields}))
    assert done and reference['total_samples']==16 and reference['optimizer_steps']==2
    learned=load_checkpoint(root,reference,c)['model']['linear_global.weight']
    assert learned.shape[1]==6 and not torch.equal(learned[:,4:6],original[:,4:6])
    write_native(c,root/'config/effective.cfg');info=export_model(root,c,reference)
    assert verify_export(root,info,c['network']['canvas'])==info and all(np.isfinite(e['loss']) for e in events if e['event']=='update')
    assert all(('q_winloss_loss' in e)==predict_q_values for e in events if e['event']=='update')
    if predict_q_values:assert all(e['q_winloss_loss']>0 for e in events if e['event']=='update')
    (root/'validation.json').write_text(json.dumps({'main_positions':main_count,'side_positions':side_count,
                                                 'reanalyzed':reanalyzed,'checkpoint':reference,'export':info,'events':events},indent=2))


@pytest.mark.parametrize('kind',['early','late','hint'])
def test_hint_game_fork_cuda_actual_prefix_and_training(tmp_path,gpu_config,kind):
    from etazero.data import metadata,training_view
    from etazero.shuffle import build_snapshot
    from etazero.config import validate
    c=copy.deepcopy(gpu_config)
    c['environment'].update(sizes='5',size_weights='1',rules='renju',rule_weights='1')
    c['opening']['probability']=0;c['policy_init']['policy_init']=False
    c['search'].update(full_search_visits=40,cheap_search_visits=20,cheap_search_probs=.8)
    c['reduce_visits'].update(reduce_visits=False,reduced_visits_min=20)
    c['pda'].update(normal_asymmetric_playout_prob=1,max_asymmetric_ratio=2)
    c['side_positions']['side_position_prob']=.2
    c['reanalysis']=dict(use_reanalyze=True,reanalyze_prop=1,reanalyze_policy_surprise_weight=1,
                         reanalyze_value_surprise_weight=1,reanalyze_surprise_exponent=1,reanalyze_use_outcome_targets=True)
    c['parallelism'].update(game_threads=1,search_threads=4)
    c['inference'].update(server_threads=2,inference_precision='float32')
    c['game_forks'].update(early_fork_game_prob=float(kind=='early'),fork_game_prob=float(kind=='late'),
                           early_fork_game_expected_move_prop=0)
    c['training'].update(batch_size=8,train_steps=2);c['replay'].update(min_rows=8,keep_target_rows='all')
    c['shuffle'].update(group_rows=1024,bucket_rows=32)
    if kind=='hint':
        positions=tmp_path/'positions.txt';positions.write_text('5 renju 1 12 2 0 5\n')
        c['hint_positions'].update(hint_positions_prob=1,positions_file=str(positions))
        c['hint_positions']['positions_sha256']=sha256(positions)
    validate(c)
    scripted=torch.jit.script(PdaCudaModel(c['network']['canvas']).cuda().eval());model=tmp_path/'model.pt';scripted.save(str(model))
    root=tmp_path/'run';root.mkdir();write_native(c,root/'effective.cfg');directory=root/'selfplay/worker'
    result=subprocess.run([str(BINARY),'selfplay','--config',str(root/'effective.cfg'),'--model',str(model),'--model-id','conditional',
         '--device','cuda:0','--games','6','--output',str(directory),'--run-id','forks','--attempt-id',kind,'--config-id',kind,
         '--source-id','test','--iteration','2','--worker','0','--seed','734'],text=True,capture_output=True,timeout=120)
    (root/'stdout.log').write_text(result.stdout);(root/'stderr.log').write_text(result.stderr);assert result.returncode==0,result.stderr
    if kind=='hint':
        positions.write_text('5 renju 1 13 2 0 5\n')
        rejected_directory=root/'selfplay/rejected'
        command=result.args.copy();command[command.index('--output')+1]=str(rejected_directory)
        rejected=subprocess.run(command,text=True,capture_output=True,timeout=30)
        assert rejected.returncode!=0 and 'Hint positions checksum mismatch' in rejected.stderr
        assert not rejected_directory.exists()
        positions.write_text('5 renju 1 12 2 0 5\n')
    from etazero.data import game_row_ranges
    entries=[];seen=[];games=[]
    for path in directory.glob('*.npz'):
        raw=read_raw(path);view=training_view(raw);assert np.isfinite(view['td_value']).all()
        _,fragment_starts,_=game_row_ranges(raw)
        for gi,(lo,hi) in enumerate(zip(raw['game_offsets'][:-1],raw['game_offsets'][1:])):
            prefix=int(raw['opening_moves'][gi]);tag=int(raw['initial_position_kind'][gi])
            if fragment_starts[gi]==0:seen.append(tag)
            assert raw['balanced_moves'][gi]==0 and raw['policy_moves'][gi]==0
            assert prefix==raw['initial_position_moves'][gi]
            assert not raw['row_repeats'][lo:lo+prefix].any() and not raw['simulations'][lo:lo+prefix].any()
            offset=raw['observation_offsets'][gi];condition=raw['globals'][offset:offset+hi-lo+1,4:6]
            if tag>=2:assert not condition.any() and raw['hint_actions'][gi]==-1 and prefix>0
            else:assert (condition[:,0]==1).all()
            if tag==1:
                assert prefix==2 and raw['hint_actions'][gi]==2*c['network']['canvas']+2
                assert not raw['cheap_search'][lo+prefix] and not raw['reanalyzed'][lo+prefix]
                sample=np.searchsorted(raw['sample_indices'],lo+prefix)
                if sample<len(raw['sample_indices']) and raw['sample_indices'][sample]==lo+prefix:
                    signed_d=2*float(condition[prefix,1]);factor=2*2**signed_d/(1+2**signed_d)
                    assert raw['visits'][sample].sum()+1==int(np.floor(160*factor+.5))
                np.testing.assert_allclose(raw['search_wdl'][lo+prefix],raw['search_wdl'][lo+prefix+1][::-1],rtol=0,atol=1e-7)
            np.testing.assert_array_equal(raw['reanalyzed'][lo:hi],raw['cheap_search'][lo:hi])
            if fragment_starts[gi]==0:
                games.append(dict(id=int(raw['game_ids'][gi]),kind=tag,prefix=prefix,rows=int(raw['row_repeats'][lo:hi].sum())))
        entries.append({'path':str(path.relative_to(root)),'sha256':sha256(path),'metadata':metadata(raw)})
    assert len(seen)==6
    if kind=='early':assert next(g['kind'] for g in games if g['id']==0)==0 and 2 in seen
    if kind=='late':assert next(g['kind'] for g in games if g['id']==0)==0 and 3 in seen
    if kind=='hint':assert 1 in seen
    if kind=='early':
        # The worker's shared pool survives round requests and evaluator release/model changes.
        from etazero.native import NativeWorker
        worker=NativeWorker([str(BINARY),'worker','--config',str(root/'effective.cfg'),'--device','cuda:0',
             '--run-id','worker-fork','--config-id',kind,'--source-id','test','--worker','0'],root/'worker.stderr')
        tags=[]
        try:
            for round_index in range(2):
                directory=root/f'worker_round_{round_index}'
                worker.submit([str(model),f'model-{round_index}',1,str(directory),f'fork-{round_index}',round_index+1,734+round_index,'network'],root/f'worker_{round_index}.stdout')
                deadline=time.monotonic()+60
                while (completed:=worker.take()) is None:
                    assert time.monotonic()<deadline,'Fork worker timed out'
                    time.sleep(.01)
                assert completed['code']==0
                records=[read_raw(path) for path in directory.glob('*.npz')]
                assert len({int(g) for raw in records for g in raw['game_ids']})==1
                tags.append(int(records[0]['initial_position_kind'][0]))
                worker.release()
        finally:worker.close()
        assert tags==[0,2]
        (root/'worker_fork_validation.json').write_text(json.dumps({'round_start_kinds':tags}))
    snapshot=build_snapshot(root,1,entries,c);initial=initialize(root,c);events=[]
    reference,done=train_iteration(root,c,{'iteration':1,'snapshot_id':snapshot,'train_steps':2,'batch_size':8},initial,
        lambda event,**fields:events.append({'event':event,**fields}))
    assert done and reference['optimizer_steps']==2 and reference['total_samples']==16
    write_native(c,root/'config/effective.cfg');info=export_model(root,c,reference);assert verify_export(root,info,c['network']['canvas'])==info
    (root/'validation.json').write_text(json.dumps(dict(games=games,checkpoint=reference,export=info,events=events),indent=2))


@pytest.mark.parametrize('architecture,amp,compile_enabled,no_repeat',[
    ('nbt','off',False,False),('nbt','float16',False,True),
    ('nbt','bfloat16',True,True),('transformer','bfloat16',True,True)])
def test_replay_holdout_real_cuda_epoch_validation_and_file_cursor(tmp_path,gpu_config,architecture,amp,compile_enabled,no_repeat):
    from test_replay import holdout_entries
    from etazero.shuffle import build_snapshot
    c=copy.deepcopy(gpu_config);c['training'].update(skip_validation=False,d4_augmentation=False,max_validation_samples=8,
                                                    train_steps=4,amp=amp,compile=compile_enabled,no_repeat_files=no_repeat)
    if architecture=='transformer':c['network'].update(architecture='transformer',channels=192,blocks=5,predict_q_values=True)
    c['replay'].update(min_rows=9,taper_exponent=1,expand_per_row=1,keep_target_rows='all')
    root=tmp_path/'run';entries=holdout_entries(root);snapshot=build_snapshot(root,1,entries,c)
    initial=initialize(root,c);plan=dict(iteration=1,snapshot_id=snapshot,train_steps=4,batch_size=8);events=[]
    reference,complete=train_iteration(root,c,plan,initial,lambda event,**fields:events.append((event,fields)))
    assert complete and reference['total_samples']==32 and reference['total_steps']==4
    val=[f for e,f in events if e=='validation'];updates=[f for e,f in events if e=='update']
    assert len(val)==1 and val[0]['samples']==16 and val[0]['batches']==2 and val[0]['model']=='raw'
    assert sum(val[0]['symmetry_counts'])==2 and val[0]['symmetry_counts'][0]!=2
    assert all(f['symmetry']==0 for f in updates) # validation D4 is independently enabled.
    checkpoint=load_checkpoint(root,reference,c,torch.device('cuda:0'))
    assert checkpoint['reader']['no_repeat_files']==no_repeat and checkpoint['reader']['split']=='train'
    assert checkpoint['optimization']['consumed_samples']==32
    assert (('q_winloss_loss' in val[0])==c['network']['predict_q_values'])
    assert all(np.isfinite(f['loss']) for f in updates+val)
    save_json(root/'validation_checks.json',dict(configuration=c,snapshot=snapshot,checkpoint=reference,events=events))


def test_no_repeat_insufficient_snapshot_fails_before_cuda_update(tmp_path,gpu_config):
    from test_replay import holdout_entries
    from etazero.shuffle import build_snapshot
    c=copy.deepcopy(gpu_config);c['training'].update(no_repeat_files=True,train_steps=100)
    # Include both train and holdout files before testing the learner's quota.
    c['replay'].update(min_rows=9,keep_target_rows='all',taper_exponent=1,expand_per_row=1)
    root=tmp_path/'run';entries=holdout_entries(root,2);snapshot=build_snapshot(root,1,entries,c)
    initial=initialize(root,c);events=[]
    with pytest.raises(ValueError,match='insufficient complete file batches'):
        train_iteration(root,c,dict(iteration=1,snapshot_id=snapshot,train_steps=100,batch_size=8),initial,
                        lambda event,**fields:events.append((event,fields)))
    assert not events and len(list((root/'checkpoints').glob('*.pt')))==1


def test_fixed_quota_actual_cuda_short_first_attempt_and_fixed_models(tmp_path,gpu_config,monkeypatch):
    import etazero.runtime as runtime
    from etazero.data import metadata
    c=copy.deepcopy(gpu_config);c['replay']['min_rows']=64;c['training']['sub_epochs']=2
    c['selfplay']['bootstrap_games']=2;c['training']['replay_ratio']=.25
    launch=runtime.Controller.launch;shortened=set()
    def short_first(self,plan,games):
        if plan['iteration']>=2 and plan['iteration'] not in shortened:
            shortened.add(plan['iteration']);assert games>1;return launch(self,plan,1)
        return launch(self,plan,games)
    monkeypatch.setattr(runtime.Controller,'launch',short_first)
    root=tmp_path/'run';state=run_training(root,c,BINARY,max_iteration=3)
    assert state['checkpoint']['total_samples']==96 and state['checkpoint']['total_steps']==12
    origin=state['replay_origin_rows'];assert origin>=64 and state['target_rows']==origin+256
    events=[json.loads(line) for line in (root/'logs/events.jsonl').read_text().splitlines()]
    backfills=[e for e in events if e['event']=='selfplay_backfill'];assert {e['iteration'] for e in backfills}>={2,3}
    cumulative=0;identities=[];worker_pids=set()
    for iteration in range(4):
        plan=load_json(root/'.internal/iterations'/f'{iteration:06d}'/'plan.json');round_rows=0
        for path in (root/'selfplay'/f'iteration_{iteration:06d}').rglob('*.npz'):
            m=metadata(read_raw(path));assert m['model_id']==plan['input_model']['id'];round_rows+=m['rows']
        cumulative+=round_rows
        if iteration>=1:
            assert cumulative>=plan['target_rows']
            status=load_json(root/'.internal/iterations'/f'{iteration:06d}'/'status.json')
            m=load_json(root/'snapshots'/status['snapshot_id']/'manifest.json')
            assert m['selection']['raw_rows']==cumulative
            assert m['selection']['random_rows']==origin if iteration>=2 else m['selection']['random_rows']==cumulative
            assert m['selection']['usable_rows']==64+(cumulative-origin) if iteration>=2 else m['selection']['usable_rows']==64
            starts=[e for e in events if e['event']=='subepoch_start' and e['iteration']==iteration]
            assert [e['consumed_step'] for e in starts]==[0,2]
            identities.append(plan['input_model']['id'])
        phases=[e for e in events if e.get('iteration')==iteration]
        end=next(i for i,e in enumerate(phases) if e['event']=='phase_end' and e['phase']=='selfplay')
        releases=[i for i,e in enumerate(phases) if e['event']=='worker_release'];assert releases and max(releases)<end
        worker_pids|={e['pid'] for e in phases if e['event']=='worker_start'}
    assert len(set(identities))==3 and len(worker_pids)==1
    hashes={p:sha256(p) for p in (root/'selfplay').rglob('*.npz')}
    assert run_training(root,c,BINARY,resume=True,max_iteration=3)==state
    assert hashes=={p:sha256(p) for p in hashes}
    save_json(root/'quota_checks.json',dict(origin=origin,state=state,backfills=backfills,input_models=identities,worker_pids=sorted(worker_pids)))


@pytest.mark.parametrize('compiled', [False,True])
@pytest.mark.parametrize('stop_at',[2,3])
def test_segmented_amp_skip_cuda_resume_preserves_all_clocks(tmp_path,gpu_config,monkeypatch,compiled,stop_at):
    from test_replay import holdout_entries
    from etazero.shuffle import build_snapshot
    c=copy.deepcopy(gpu_config);c['training'].update(amp='float16',compile=compiled,train_steps=7,sub_epochs=3)
    c['optimizer'].update(lookahead_k=3,swa_period_samples=16,norm_only_at_print=False,lookahead_print=True)
    # Include the full fixture instead of selecting only its newest holdout files.
    c['replay'].update(min_rows=9,keep_target_rows='all',taper_exponent=1,expand_per_row=1)
    source=tmp_path/'source';entries=holdout_entries(source);snapshot=build_snapshot(source,1,entries,c)
    roots=[tmp_path/'full',tmp_path/'resumed'];bases=[]
    for root in roots:
        shutil.copytree(source/'snapshots'/snapshot,root/'snapshots'/snapshot)
        base=initialize(root,c);value=load_checkpoint(root,base,c)
        model=make_network(c).cuda();model.load_state_dict(value['model']);optimizer=optimizer_for(model,c);optimizer.load_state_dict(value['optimizer'])
        o=Optimization(model,c,optimizer,value['optimization']);scaler=torch.amp.GradScaler('cuda',init_scale=1)
        restore_rng(value['rng']);bases.append(commit_checkpoint(root,c,model,optimizer,scaler,0,0,0,None,base,[],o))
    original=make_network;injection={'remaining':1}
    def instrumented(configuration):
        model=original(configuration)
        def overflow(grad):
            if injection['remaining']:
                injection['remaining']-=1;return torch.full_like(grad,float('inf'))
            return grad
        model.stem.weight.register_hook(overflow);return model
    monkeypatch.setattr('etazero.training.make_network',instrumented)
    plan=dict(iteration=1,snapshot_id=snapshot,train_steps=7,batch_size=8)
    full_events=[];resumed_events=[]
    deterministic=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
    try:
        full,done=train_iteration(roots[0],c,plan,bases[0],lambda e,**f:full_events.append((e,f)));assert done
        injection['remaining']=1
        updates=lambda:sum(e=='update' for e,_ in resumed_events)
        mid,done=train_iteration(roots[1],c,plan,bases[1],lambda e,**f:resumed_events.append((e,f)),lambda:updates()>=stop_at)
        assert not done and mid['step']==stop_at
        saved=load_checkpoint(roots[1],mid,c)
        assert saved['optimization']['subepoch']==(0 if stop_at==2 else 1)
        assert saved['optimization']['subepoch_batches']==(2 if stop_at==2 else 1)
        resumed,done=train_iteration(roots[1],c,plan,bases[1],lambda e,**f:resumed_events.append((e,f)));assert done
        a=load_checkpoint(roots[0],full,c);b=load_checkpoint(roots[1],resumed,c)
        def equal(left,right):
            if isinstance(left,torch.Tensor):torch.testing.assert_close(left,right,rtol=0,atol=0)
            elif isinstance(left,dict):
                assert left.keys()==right.keys()
                for key in left:equal(left[key],right[key])
            elif isinstance(left,(list,tuple)):
                assert len(left)==len(right)
                for x,y in zip(left,right):equal(x,y)
            else:assert left==right
        for key in ('model','optimizer','scaler','reader','optimization'):equal(a[key],b[key])
        torch.testing.assert_close(a['rng']['torch'],b['rng']['torch'],rtol=0,atol=0)
        assert a['total_samples']==56 and a['optimizer_steps']==6
        rows=[f for e,f in full_events if e=='update'];assert [f['amp_skipped'] for f in rows]==[True]+[False]*6
        assert [f['subepoch'] for f in rows]==[0,0,1,1,2,2,2]
        assert [f['subepoch_batches'] for f in rows]==[1,2,1,2,1,2,3]
        assert [f['lookahead_counter'] for f in rows]==[1,2,1,2,1,2,0]
        assert [f['swa_samples'] for f in rows]==[0,0,0,0,0,0,1]
        assert a['optimization']['swa_samples']==0 and a['optimization']['subepoch']==2
        save_json(roots[1]/'segment_checks.json',dict(configuration=c,full=full,resumed=resumed,stop_at=stop_at,events=full_events))
    finally:torch.use_deterministic_algorithms(deterministic)


def test_native_worker_same_id_different_model_path_clears_cuda_cache(tmp_path,gpu_config):
    from etazero.native import NativeWorker
    c=copy.deepcopy(gpu_config);c['inference']['inference_precision']='float32'
    c['opening']['probability']=0;c['policy_init']['policy_init']=False
    c['side_positions']['side_position_prob']=0;c['game_forks'].update(early_fork_game_prob=0,fork_game_prob=0)
    c['hint_positions']['hint_positions_prob']=0
    write_native(c,tmp_path/'effective.cfg');models=[]
    for label,probabilities in [('a',[.8,.1,.1]),('b',[.1,.2,.7])]:
        model=SearchCorrectionsCudaModel(c['network']['canvas'],.5).cuda().eval()
        model.value.copy_(torch.tensor(probabilities,device='cuda').log());path=tmp_path/f'{label}.pt'
        torch.jit.script(model).save(str(path));models.append((path,probabilities))
    command=[str(BINARY),'worker','--config',str(tmp_path/'effective.cfg'),'--model',str(models[0][0]),'--model-id','fixed-id',
             '--device','cuda:0','--games','1','--output',str(tmp_path/'unused'),'--run-id','test',
             '--attempt-id','initial','--config-id','config','--source-id','source','--iteration','2','--worker','0','--seed','17']
    worker=NativeWorker(command,tmp_path/'stderr.log');pid=worker.process.pid
    try:
        for index,(path,expected) in enumerate([models[0],models[1],models[1]]):
            output=tmp_path/f'raw_{index}';worker.submit([str(path),'fixed-id',1,str(output),str(index),2,17,'network'],tmp_path/f'stdout_{index}.jsonl')
            deadline=time.monotonic()+30
            while True:
                result=worker.take()
                if result is not None:break
                assert time.monotonic()<deadline,'Native worker timed out';time.sleep(.005)
            assert result['code']==0 and worker.process.pid==pid
            raw=read_raw(next(output.glob('*.npz')))
            np.testing.assert_allclose(raw['network_wdl'],np.tile(expected,(len(raw['actions']),1)),rtol=1e-5,atol=1e-6)
            if index==1:worker.release()
    finally:worker.close()


@pytest.fixture(scope='module')
def recovery_parent(tmp_path_factory,gpu_config):
    root=tmp_path_factory.mktemp('recovery_parent')
    state=run_training(root,gpu_config,BINARY,max_iteration=1)
    return root,state


@pytest.mark.parametrize('boundary',['selfplay_finish','shuffle_finish','checkpoint_payload','learner_pointer',
                                      'export_stage','export_rename','metrics','state_commit','publication'])
def test_controller_transaction_boundaries_real_cuda(tmp_path,recovery_parent,gpu_config,monkeypatch,boundary):
    import etazero.runtime as runtime
    import etazero.training as training
    import etazero.export as exporting
    from etazero.data import Catalog
    from etazero.plotting import run_history
    parent,base=recovery_parent;root=tmp_path/'run';shutil.copytree(parent,root)
    preserved={p:sha256(p) for name,pattern in [('selfplay','*.npz'),('models','model.pt'),('checkpoints','*.pt')]
               for p in (root/name).rglob(pattern)}
    def fail():raise RuntimeError('injected transaction boundary')
    with monkeypatch.context() as patch:
        if boundary in ('selfplay_finish','shuffle_finish'):
            original=runtime.Controller.phase;selected=boundary.split('_')[0]
            def phase(self,name,iteration,function):
                value=original(self,name,iteration,function)
                if name==selected and iteration==2:fail()
                return value
            patch.setattr(runtime.Controller,'phase',phase)
        elif boundary=='checkpoint_payload':
            original=training.atomic_write
            def write(path,*args,**kwargs):
                value=original(path,*args,**kwargs)
                if Path(path).suffix=='.pt' and 'iteration_000002' in str(path):fail()
                return value
            patch.setattr(training,'atomic_write',write)
        elif boundary=='learner_pointer':
            original=training.save_json
            def write(path,*args,**kwargs):
                value=original(path,*args,**kwargs)
                if Path(path).name=='learner.json':fail()
                return value
            patch.setattr(training,'save_json',write)
        elif boundary=='export_stage':
            original=exporting.os.fsync
            def fsync(fd):
                value=original(fd)
                path=os.readlink(f'/proc/self/fd/{fd}')
                if '/models/.tmp_' in path and path.endswith('/model.pt'):fail()
                return value
            patch.setattr(exporting.os,'fsync',fsync)
        elif boundary=='export_rename':
            original=exporting.os.rename
            def rename(src,dst):
                value=original(src,dst)
                if Path(dst).parent==root/'models':fail()
                return value
            patch.setattr(exporting.os,'rename',rename)
        elif boundary=='metrics':
            original=runtime.save_json
            def write(path,*args,**kwargs):
                value=original(path,*args,**kwargs)
                if Path(path)==root/'logs/iterations/000002.json':fail()
                return value
            patch.setattr(runtime,'save_json',write)
        elif boundary=='state_commit':
            original=runtime.Controller.state
            def state(self,value):
                original(self,value)
                if value['iteration']==3:fail()
            patch.setattr(runtime.Controller,'state',state)
        else:
            original=runtime.Controller.publish
            def publish(self,model,elapsed):
                original(self,model,elapsed)
                if model['checkpoint']['iteration']==2:fail()
            patch.setattr(runtime.Controller,'publish',publish)
        with pytest.raises(RuntimeError,match='transaction boundary'):
            run_training(root,gpu_config,BINARY,resume=True,max_iteration=2)
    failed_state=load_json(root/'.internal/state.json')
    committed=boundary in ('state_commit','publication')
    assert failed_state['iteration']==(3 if committed else 2)
    assert all(sha256(path)==digest for path,digest in preserved.items())
    pending={p:sha256(p) for folder in ('selfplay/iteration_000002','checkpoints','models') for p in (root/folder).rglob('*')
             if p.is_file() and ('iteration_000002' in str(p) or '.tmp_' in str(p)) and
             # After commit, configured pruning may remove intermediate payloads;
             # lineage JSON, the final checkpoint, raw games and models remain.
             not (committed and p.parent==root/'checkpoints' and p.suffix=='.pt' and p!=root/failed_state['checkpoint']['path'])}
    assert pending
    resumed=run_training(root,gpu_config,BINARY,resume=True,max_iteration=2)
    assert resumed['iteration']==3 and resumed['checkpoint']['total_steps']==8 and resumed['checkpoint']['total_samples']==64
    assert resumed['replay_origin_rows']==base['replay_origin_rows']
    quota=gpu_config['training']['train_steps']*gpu_config['training']['batch_size']/gpu_config['training']['replay_ratio']
    assert resumed['target_rows']==pytest.approx(base['target_rows']+quota)
    assert_preserved(root,pending)
    assert not list((root/'models').glob('.tmp_*'))
    assert load_json(root/'models/current.json')['model']==resumed['model']
    info=load_json(root/'.internal/run.json')
    catalog=Catalog(root,info['id'],info['config_id'])
    try:
        rows=catalog.counts(gpu_config['selfplay']['recent_games'])[0]
    finally:catalog.close()
    raw_rows=sum(metadata(read_raw(p))['rows'] for p in (root/'selfplay').rglob('*.npz'))
    assert rows==raw_rows
    history=run_history(root);assert [event['iteration'] for event in history]==[0,1,2]
    before={p:sha256(p) for folder in ('models','selfplay','checkpoints','logs/iterations') for p in (root/folder).rglob('*') if p.is_file()}
    assert run_training(root,gpu_config,BINARY,resume=True,max_iteration=2)==resumed
    assert before=={p:sha256(p) for p in before}
    save_json(root/'transaction_checks.json',{'boundary':boundary,'committed_before_recovery':committed,
              'final_steps':8,'final_samples':64,'active_rows':rows,'preserved_old_artifacts':len(preserved),
              'preserved_pending_artifacts':len(pending),'history_iterations':[0,1,2],
              'resume_policy':'whole round rollback before state commit; no rerun after commit'})


@pytest.mark.parametrize('corruption',['model','manifest','checkpoint'])
def test_committed_artifact_corruption_rejected_before_cuda_workers(tmp_path,recovery_parent,gpu_config,corruption):
    parent,state=recovery_parent;root=tmp_path/'run';shutil.copytree(parent,root)
    if corruption=='model':path=root/state['model']['path']
    elif corruption=='manifest':path=(root/state['model']['path']).parent/'manifest.json'
    else:path=root/state['checkpoint']['path']
    path.write_bytes(b'injected partial artifact')
    before=sha256(root/'.internal/state.json')
    starts=sum(json.loads(line)['event']=='worker_start' for line in (root/'logs/events.jsonl').read_text().splitlines())
    with pytest.raises((ValueError,json.JSONDecodeError),match='checksum|Expecting'):
        run_training(root,gpu_config,BINARY,resume=True,max_iteration=2)
    assert sha256(root/'.internal/state.json')==before
    assert starts==sum(json.loads(line)['event']=='worker_start' for line in (root/'logs/events.jsonl').read_text().splitlines())


def test_native_final_row_shards_cuda_training_and_export(tmp_path,gpu_config):
    from etazero.config import validate
    from etazero.data import Catalog,metadata,training_view
    from etazero.shuffle import build_snapshot
    c=copy.deepcopy(gpu_config)
    c['environment'].update(sizes='5',size_weights='1',rules='renju',rule_weights='1')
    c['opening']['probability']=0;c['policy_init']['policy_init']=False
    c['search'].update(full_search_visits=40,cheap_search_visits=20,cheap_search_probs=.5)
    c['reduce_visits'].update(reduce_visits=False,reduced_visits_min=20)
    c['pda'].update(normal_asymmetric_playout_prob=1,max_asymmetric_ratio=2)
    c['side_positions']['side_position_prob']=.4
    c['parallelism'].update(game_threads=1,search_threads=2)
    c['inference'].update(server_threads=2,max_batch=8,queue_capacity=1,inference_precision='float32')
    c['writer'].update(shard_rows=8,first_file_min_random_proportion=.15)
    c['training'].update(train_steps=2,batch_size=8,skip_validation=True)
    c['replay'].update(min_rows=8,keep_target_rows='all',taper_exponent=1,expand_per_row=1)
    c['shuffle'].update(group_rows=64,bucket_rows=32,training_shard_rows=32)
    validate(c)
    root=tmp_path/'run';root.mkdir();write_native(c,root/'effective.cfg')
    model=root/'conditional.pt';torch.jit.script(PdaCudaModel(c['network']['canvas']).cuda().eval()).save(str(model))
    directory=root/'selfplay/worker'
    result=subprocess.run([str(BINARY),'selfplay','--config',str(root/'effective.cfg'),'--model',str(model),
        '--model-id','conditional','--device','cuda:0','--games','4','--output',str(directory),'--run-id','shards',
        '--attempt-id','rows','--config-id','test','--source-id','test','--iteration','2','--worker','0','--seed','734'],
        text=True,capture_output=True,timeout=120)
    (root/'stdout.log').write_text(result.stdout);(root/'stderr.log').write_text(result.stderr)
    assert result.returncode==0,result.stderr
    entries=[];rows=0;represented=[];side_rows=0
    for path in directory.glob('*.npz'):
        raw=read_raw(path);m=metadata(raw);view=training_view(raw)
        assert 0<m['rows']<=8 and len(view['value'])==m['rows']
        assert np.isfinite(view['td_value']).all() and np.isfinite(view['q_values']).all()
        rows+=m['rows'];represented.extend(map(int,raw['game_ids']))
        side_rows+=int((view['opponent_policy_weight']==0).sum())
        entries.append({'path':str(path.relative_to(root)),'sha256':sha256(path),'metadata':m})
    assert rows>8 and len(represented)>4 and set(represented)==set(range(4)) and side_rows>0
    catalog=Catalog(root,'shards','test')
    try:
        catalog.scan({2:'conditional'})
        assert catalog.counts(10)[:2]==(rows,4) and catalog.iteration_counts(2)==(rows,4)
        assert catalog.statistics(2)['games']==4
    finally:catalog.close()
    snapshot=build_snapshot(root,2,entries,c)
    assert load_json(root/'snapshots'/snapshot/'manifest.json')['rows']==rows
    initial=initialize(root,c);events=[]
    reference,done=train_iteration(root,c,{'iteration':2,'snapshot_id':snapshot,'train_steps':2,'batch_size':8},initial,
        lambda event,**fields:events.append({'event':event,**fields}))
    assert done and reference['total_samples']==16 and reference['optimizer_steps']==2
    assert all(np.isfinite(e['loss']) for e in events if e['event']=='update')
    write_native(c,root/'config/effective.cfg');info=export_model(root,c,reference)
    assert verify_export(root,info,c['network']['canvas'])==info
    save_json(root/'writer_validation.json',{'shards':len(entries),'rows':rows,'games':4,'represented_trajectories':len(represented),
        'checkpoint':reference,'export':info})
