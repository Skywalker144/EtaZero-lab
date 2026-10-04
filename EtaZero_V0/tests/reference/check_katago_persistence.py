"""Execute the pinned, unchanged source save function and inspect local saved state.

This compares persisted scope, not a claim that the two recovery protocols match.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from etazero.config import ROOT,load_config
from etazero.network import make_network
from etazero.optimization import Optimization,optimizer_for
from etazero.training import commit_checkpoint,load_checkpoint


def check():
    registration=json.loads((ROOT/'reference_sources.json').read_text())['KataGo']
    source=Path(registration['root'])/'python/train.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==registration['sha256']['python/train.py']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source.parent,text=True).strip()==registration['commit']
    tree=ast.parse(source.read_text());save=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='save')
    captures=[];renames=[]
    namespace={'gnorm_stats_debug':False,'rank':0,'logging':SimpleNamespace(info=lambda *a:None,warning=lambda *a:None),
               'torch':SimpleNamespace(save=lambda value,path:captures.append(copy.deepcopy(value))),
               'os':SimpleNamespace(replace=lambda a,b:renames.append((a,b))),
               'time':SimpleNamespace(sleep=lambda *a:None),'defaultdict':__import__('collections').defaultdict,
               'model_config':{'source':'config'}}
    exec(compile(ast.Module(body=[save],type_ignores=[]),str(source),'exec'),namespace)
    state=lambda value:SimpleNamespace(state_dict=lambda:value)
    for swa in (None,state({'average':7})):
        for skip in (False,True):
            namespace['save'](state({'fast':1}),swa,state({'optimizer':2}),state({'metrics':3}),
                              {'running':{'value':4}},{'global_step_samples':8,'swa_sample_accum':16},
                              {'validation':5},path='/independent/checkpoint',skip_optimizer=skip)
            keys=set(captures[-1]);expected={'model','metrics','running_metrics','train_state','last_val_metrics','config'}
            if not skip:expected.add('optimizer')
            if swa:expected.add('swa_model')
            assert keys==expected
            assert renames[-1]==('/independent/checkpoint.tmp','/independent/checkpoint')
    config=load_config(ROOT / 'tests/fixtures/configs/smoke_test', run_dir='/tmp/etazero_reference_sample');config['devices']['train']='cpu'
    model=make_network(config);optimizer=optimizer_for(model,config);optimization=Optimization(model,config,optimizer)
    optimization.begin_round();optimization.after_step(successful=False)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    cursor={'independent':'consumed cursor'}
    with tempfile.TemporaryDirectory() as directory:
        reference=commit_checkpoint(directory,config,model,optimizer,scaler,1,1,1,cursor,None,['independent_update'],optimization)
        saved=load_checkpoint(directory,reference,config)
        expected={'id','contract','config_id','network_config','model','optimizer','optimization','scaler','rng','reader',
                  'iteration','step','total_steps','total_samples','optimizer_steps','parent','committed_updates','source_id'}
        assert set(saved)==expected and saved['reader']==cursor and saved['optimizer_steps']==0 and saved['total_samples']==8
        assert set(saved['rng'])=={'python','numpy','torch','cuda'}
        assert set(saved['optimization'])>={'slow','lookahead_counter','swa','swa_samples','consumed_samples','optimizer_steps',
               'round_batches','subepoch','subepoch_batches'}
    return dict(status='verified',source_commit=registration['commit'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_save_lines=[save.lineno,save.end_lineno],source_save_cases=4,
                source_checkpoint_fields=sorted(set().union(*(set(x) for x in captures))),local_checkpoint_fields=sorted(expected),
                local_rng_fields=sorted(saved['rng']),local_optimization_fields=sorted(saved['optimization']),
                scope='Original save function executed; local real CPU checkpoint roundtrip; field-scope comparison, not recovery equivalence.',
                differences=['Source saves model/optimizer/metrics/train_state/SWA; save function does not capture Python/NumPy/Torch/CUDA RNG, AMP scaler or Lookahead local cache/counter.',
                             'Source train_state owns SWA accumulated samples and file usage, not Eta consumed file-row cursor and whole-round quota/commit authority.',
                             'Eta exact learner resume has extra persisted state; controller resumes last committed whole round and archives pending work.'])


if __name__=='__main__':print(json.dumps(check(),indent=2))
