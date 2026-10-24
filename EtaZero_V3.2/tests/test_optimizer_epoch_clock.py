"""Outer data boundaries cannot change the source optimizer trajectory."""
import copy
import torch
from config_samples import CONFIGS
from etazero.config import load_config
from etazero.network import make_network
from etazero.optimization import Optimization, optimizer_for


def run(boundaries=(), resume=None):
    torch.set_num_threads(1);torch.manual_seed(410)
    config=load_config(CONFIGS/'smoke_test')
    config['training'].update(train_steps=24,sub_epochs=3)
    config['optimizer'].update(lookahead_k=6,swa_period_samples=32)
    model=make_network(config);optimizer=optimizer_for(model,config)
    clock=Optimization(model,config,optimizer,round_steps=1)
    assert clock.lookahead_k==6
    clock.begin_round()
    counters=[]
    for step in range(29):
        if step in boundaries:clock.begin_round()
        if step==resume:
            state=copy.deepcopy((model.state_dict(),optimizer.state_dict(),clock.state_dict()))
            model=make_network(config);model.load_state_dict(state[0])
            optimizer=optimizer_for(model,config);optimizer.load_state_dict(state[1])
            clock=Optimization(model,config,optimizer,state[2],round_steps=1)
        clock.prepare_batch(dict(epoch=step//24,subepoch=(step%24)//8,subepoch_batches=step%8+1,epoch_end=step%24==23));optimizer.zero_grad(set_to_none=True)
        # Smooth, deterministic nonzero gradients through each optimizer role.
        loss=sum((p.square().sum()+.01*p.sum()) for p in model.parameters())
        loss.backward();clock.before_step();optimizer.step();clock.after_step()
        counters.append((clock.completed_rounds,clock.round_batches,clock.subepoch,clock.counter,clock.swa_count))
    return model.state_dict(),optimizer.state_dict(),clock.state_dict(),counters


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_round_partition_and_resume_preserve_source_optimizer_clock():
    whole=run()
    split=run(boundaries=(1,4,5,9,13,18,24,25),resume=11)
    equal(whole,split)
    history=whole[-1]
    assert history[5][3]==0  # First synchronization is batch 6, never shortened.
    assert history[8][2:4]==(1,1)  # Source resets Lookahead at the fixed subepoch.
    assert history[23][:2]==(1,24)
    assert history[24][:3]==(1,1,0)
