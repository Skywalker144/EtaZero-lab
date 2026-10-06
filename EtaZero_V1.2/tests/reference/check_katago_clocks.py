"""Run pinned train.py LR/Lookahead/SWA/reset blocks unchanged, on scalar weights."""
import argparse,ast,copy,hashlib,itertools,json,subprocess,sys,logging
from pathlib import Path
import torch
from torch.optim.swa_utils import AveragedModel
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'tests'))
from etazero.config import load_config
from etazero.training import subepoch_ends
from test_optimization import small_optimization


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    ref=json.loads((ROOT/'reference_sources.json').read_text())['KataGo'];path=Path(ref['root'])/'python/train.py';data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==ref['sha256']['python/train.py']
    assert subprocess.check_output(['git','-C',ref['root'],'rev-parse','HEAD'],text=True).strip()==ref['commit']
    tree=ast.parse(data);nodes=list(ast.walk(tree))
    def unique(predicate):
        selected=[n for n in nodes if predicate(n)];assert len(selected)==1,len(selected);return selected[0]
    lr=unique(lambda n:isinstance(n,ast.If) and '200000000' in ast.unparse(n.test) and 'batch_count_this_epoch' in ast.unparse(n.test))
    lookahead=unique(lambda n:isinstance(n,ast.If) and ast.unparse(n.test)=='lookahead_k is not None' and 'lookahead_counter += 1' in ast.unparse(n))
    swa=unique(lambda n:isinstance(n,ast.If) and ast.unparse(n.test)=='swa_model is not None and swa_scale is not None' and 'swa_sample_accum' in ast.unparse(n))
    reset=unique(lambda n:isinstance(n,ast.For) and ast.unparse(n.target)=='i' and ast.unparse(n.iter)=='range(sub_epochs)')
    reset=next(n for n in reset.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='lookahead_counter')
    finish=unique(lambda n:isinstance(n,ast.If) and ast.unparse(n.test)=='lookahead_k is not None' and 'lookahead_counter' not in ast.unparse(n) and 'param.data.copy_(slow_param_data)' in ast.unparse(n))
    code=compile(ast.Module(body=[copy.deepcopy(lr),ast.parse('in_between_lookaheads = False').body[0],copy.deepcopy(lookahead),copy.deepcopy(swa)],type_ignores=[]),str(path),'exec')
    reset_code=compile(ast.Module(body=[copy.deepcopy(reset)],type_ignores=[]),str(path),'exec');finish_code=compile(ast.Module(body=[copy.deepcopy(finish)],type_ignores=[]),str(path),'exec')
    cases=0
    for steps,nsegments,k,alpha,start_samples,period in itertools.product([7,17,53],[1,3],[2,6],[.5,1.],[0,199999976,200000008],[8,32]):
        config=load_config(ROOT / 'tests/fixtures/configs/smoke_test', run_dir='/tmp/etazero_reference_sample');config['training'].update(train_steps=steps,sub_epochs=nsegments)
        config['optimizer'].update(lookahead_k=k,lookahead_alpha=alpha,swa_period_samples=period,norm_interval=10000)
        model,optimizer,o=small_optimization(config);o.consumed_samples=start_samples;o.begin_round()
        original=copy.deepcopy(model);source_optimizer=torch.optim.SGD(original.parameters(),lr=1)
        slow={p:p.detach().clone() for p in original.parameters()}
        source_swa=AveragedModel(original,avg_fn=lambda avg,current,count:avg*(1-1/8)+current/8,use_buffers=False)
        source_calls=[];target_calls=[];real_configure=o.configure
        def configure():target_calls.append((o.round_batches,o.consumed_samples));real_configure()
        o.configure=configure
        ns=dict(logging=logging,train_state={'global_step_samples':start_samples,'swa_sample_accum':0},batch_count_this_epoch=0,
                lookahead_k=k if alpha<1 else None,lookahead_alpha=alpha,lookahead_cache=slow,optimizer=source_optimizer,
                swa_model=source_swa,swa_scale=8,swa_period_samples=period,batch_size=8,world_size=1,raw_model=original)
        def update(log_if):source_calls.append((ns['batch_count_this_epoch'],ns['train_state']['global_step_samples']));return 0,0
        ns['update_and_return_lr_and_wd']=update;exec(reset_code,ns);ends=subepoch_ends(steps,nsegments)
        for step in range(steps):
            if step in ends[:-1]:o.begin_subepoch();exec(reset_code,ns)
            successful=step%4!=1
            if successful:
                with torch.no_grad():
                    for a,b in zip(model.parameters(),original.parameters()):a.add_((step+1)/13);b.add_((step+1)/13)
            o.before_step();o.after_step(successful)
            ns['batch_count_this_epoch']+=1;ns['train_state']['global_step_samples']+=8;exec(code,ns)
            for a,b in zip(model.parameters(),original.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip(o.slow.values(),slow.values()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            assert o.counter==ns['lookahead_counter'] and o.swa_samples==ns['train_state']['swa_sample_accum']
            for name,value in o.swa.state_dict().items():torch.testing.assert_close(value,source_swa.state_dict()[name],rtol=0,atol=0)
            assert target_calls==source_calls
            assert o.consumed_samples==start_samples+(step+1)*8 and o.optimizer_steps==sum(j%4!=1 for j in range(step+1))
            cases+=1
        o.finish_round();exec(finish_code,ns)
        for a,b in zip(model.parameters(),original.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    result=dict(status='verified',source_commit=ref['commit'],source_sha256=hashlib.sha256(data).hexdigest(),step_cases=cases,
                source_lines=dict(lr=lr.lineno,lookahead=lookahead.lineno,swa=swa.lineno,subepoch_reset=reset.lineno,epoch_finish=finish.lineno),
                scope='Unmodified AST LR/WD refresh, Lookahead, SWA, segment-reset and end-of-epoch blocks; scalar fast/slow/SWA states, source float32 operation order, skips, 200M boundary and reset clock ordering.',
                adaptations='Local fixed budget splits into floor boundaries, not source probabilistic file-selection budgets. Successful step schedule supplied identically; actual CUDA GradScaler skips separately tested. Local epoch-end counter normalized to zero; source resets it at the next segment. No training bucket.')
    assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
