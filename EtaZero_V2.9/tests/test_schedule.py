"""Single-pass budgets, earned training credit and round/segment/learner clocks."""
from config_samples import CONFIGS
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from etazero.config import ROOT,load_config,validate
from etazero.runtime import Controller,iteration_plan,training_budget
from etazero.training import subepoch_ends
from etazero.shuffle import replay_counts
from etazero.data import Catalog, validate_raw
from etazero.storage import save_npz
from test_optimization import small_optimization
from test_python import winning_record, compact_search


def controller(config,rows=100,games=1):
    class Counts:
        def __init__(self):self.rows=rows;self.games=games;self.round_games=0
        def counts(self):return self.rows,self.games
        def previous_rows_per_game(self,iteration):return rows/games if games else None
        def iteration_counts(self,iteration):return 0,self.round_games
    value=object.__new__(Controller);value.config=config;value.catalog=Counts();value.stop=False;value.services={}
    value.stopping=lambda:False;value.scan=lambda:None;value.events=[]
    value.journal=lambda event,**fields:value.events.append((event,fields));return value


def test_two_iteration_estimate_uses_actual_rows_and_weights_by_games(tmp_path):
    catalog=Catalog(tmp_path,'test','config')
    try:
        assert catalog.previous_rows_per_game(1) is None
        for iteration,counts in ((0,[90]),(1,[2]),(2,[8,0,5]),(3,[90])):
            for game,rows in enumerate(counts):
                raw,m=winning_record()
                raw['row_repeats'][:]=0;raw['row_repeats'][0]=rows
                raw['target_weights']=raw['row_repeats'].astype(np.float32)
                compact_search(raw)
                m.update(iteration_id=iteration,attempt_id=str(iteration),shard_id=f'{iteration}:{game}',rows=rows)
                raw['game_ids'][0]=game
                raw['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
                validate_raw(raw)
                save_npz(tmp_path/'selfplay'/f'{iteration}_{game}.npz',raw)
        catalog.scan({i:'model' for i in range(4)})
        assert catalog.previous_rows_per_game(1)==90  # Bootstrap only.
        assert catalog.previous_rows_per_game(2)==46  # Bootstrap and iteration 1.
        assert catalog.previous_rows_per_game(3)==3.75  # (2 + 8 + 0 + 5) / (1 + 3).
        assert catalog.previous_rows_per_game(4)==25.75  # Drops iteration 1.
        assert catalog.counts()==(195,6)
        assert catalog.statistics(2)['avg_game_length']==9
        assert catalog.previous_rows_per_game(0) is None
    finally:catalog.close()


def test_short_games_finish_estimated_quota_without_backfill():
    c=load_config(CONFIGS / 'smoke_test');driver=controller(c,100,20);quotas=[]
    def launch(plan,games):
        quotas.append(games)
        driver.catalog.rows+=games*2;driver.catalog.games+=games;driver.catalog.round_games+=games
    driver.launch=launch
    plan=dict(iteration=3,start_rows=100,target_rows=115)
    driver.produce(plan)
    assert quotas==[3] and driver.catalog.rows==106
    assert not any(e=='selfplay_backfill' for e,_ in driver.events)
    driver.produce(plan)
    assert quotas==[3]  # Completed quota is not replayed after a stage interruption.


def test_interrupted_game_quota_completes_only_remaining_games():
    c=load_config(CONFIGS / 'smoke_test');driver=controller(c,106,20)
    driver.catalog.previous_rows_per_game=lambda i:5
    driver.catalog.round_games=3;quotas=[]
    def launch(plan,games):
        quotas.append(games)
        driver.catalog.rows+=games;driver.catalog.games+=games;driver.catalog.round_games+=games
    driver.launch=launch
    driver.produce(dict(iteration=2,start_rows=100,target_rows=125))
    assert quotas==[2] and driver.catalog.round_games==5 and driver.catalog.rows==108


def test_minimum_game_and_no_progress():
    c=load_config(CONFIGS / 'smoke_test');driver=controller(c,200,20);quotas=[]
    def launch(plan,games):
        quotas.append(games);driver.catalog.rows+=games*9;driver.catalog.games+=games;driver.catalog.round_games+=games
    driver.launch=launch
    plan=dict(iteration=2,start_rows=200,target_rows=200)
    driver.produce(plan)
    assert quotas==[1] and driver.catalog.rows==209
    driver.produce(plan);assert quotas==[1]
    empty=controller(c,100,0);empty.launch=lambda p,g:pytest.fail('No estimate must fail before launching')
    with pytest.raises(RuntimeError,match='rows-per-game'):empty.produce(plan)
    stuck=controller(c);stuck.launch=lambda p,g:None
    with pytest.raises(RuntimeError,match='complete games'):stuck.produce(plan)
    bootstrap=controller(c,0,0);bootstrap.launch=lambda p,g:None
    with pytest.raises(RuntimeError,match='Bootstrap'):bootstrap.produce(dict(iteration=0,target_rows=0))


def test_completed_zero_row_games_finish_normal_selfplay_quota():
    c=load_config(CONFIGS / 'smoke_test');driver=controller(c,100,20);quotas=[]
    def launch(plan,games):
        quotas.append(games);driver.catalog.games+=games;driver.catalog.round_games+=games
    driver.launch=launch;driver.produce(dict(iteration=2,start_rows=100,target_rows=110))
    assert quotas==[2] and driver.catalog.rows==100


def test_baseline_plans_use_actual_rows_without_cold_start_credit():
    c=load_config(CONFIGS / 'smoke_test')
    c['training'].update(train_steps=2000,batch_size=128,replay_ratio=8)
    state=dict(iteration=2,model={'id':'net'},checkpoint={'id':'trained'},replay_rows=300123,train_credit=0)
    first=iteration_plan(state,c)
    assert first['target_rows']==332123 and first['start_rows']==300123 and first['train_credit']==0
    # A short round does not leave an accumulating production deficit.
    state.update(iteration=3,replay_rows=316123,train_credit=128)
    second=iteration_plan(state,c)
    assert second['target_rows']==348123 and second['train_credit']==128
    entries=[{'metadata':{'model_id':'random:1','rows':300123}}, {'metadata':{'model_id':'iteration_2','rows':16000}}]
    assert replay_counts(entries,250000)==dict(raw_rows=316123,random_rows=300123,postrandom_rows=16000,usable_rows=266000)


def budget_plan(iteration=2,credit=0):
    return dict(iteration=iteration,train_steps=2000,batch_size=128,replay_ratio=8,start_rows=150000,train_credit=credit)


def test_half_production_halves_training_and_snapshot_pass_caps_consumption():
    snapshot={'resource_plan':{'complete_batches_per_pass':3000}}
    budget=training_budget(budget_plan(),snapshot,166000)
    assert budget==dict(train_steps=1000,snapshot_batches=3000,available_credit=128000,new_rows=16000)
    snapshot['resource_plan']['complete_batches_per_pass']=700
    capped=training_budget(budget_plan(),snapshot,166000)
    assert capped['train_steps']==700 and capped['available_credit']-700*128==38400
    assert training_budget(budget_plan(credit=38400),snapshot,150000)['train_steps']==300
    snapshot['resource_plan']['complete_batches_per_pass']=3000
    assert training_budget(budget_plan(),snapshot,214000)['train_steps']==2000


def test_cold_start_single_pass_and_fractional_credit_can_make_zero_step_round():
    snapshot={'resource_plan':{'complete_batches_per_pass':1170}}
    assert training_budget(budget_plan(iteration=1),snapshot,150000)['train_steps']==1170
    assert training_budget(budget_plan(iteration=1,credit=1000),snapshot,150000)['available_credit']==0
    plan=budget_plan();plan['replay_ratio']=.25
    assert training_budget(plan,snapshot,150001)['train_steps']==0
    plan['train_credit']=127.75
    assert training_budget(plan,snapshot,150001)['train_steps']==1
    with pytest.raises(ValueError,match='decreased'):training_budget(plan,snapshot,149999)
    snapshot['resource_plan']['complete_batches_per_pass']=0
    with pytest.raises(ValueError,match='no complete'):training_budget(plan,snapshot,150001)


def test_local_segment_boundaries_are_nonempty_and_keep_exact_fixed_budget():
    for steps in range(1,31):
        for n in range(1,steps+1):
            ends=subepoch_ends(steps,n);lengths=np.diff([0]+ends)
            assert ends[-1]==steps and len(ends)==n and min(lengths)>0 and max(lengths)-min(lengths)<=1
    assert subepoch_ends(7,3)==[2,4,7]
    for steps,n in [(0,1),(2,0),(2,3)]:
        with pytest.raises(ValueError):subepoch_ends(steps,n)
    c=load_config(CONFIGS / 'smoke_test');c['training']['sub_epochs']=5
    with pytest.raises(ValueError,match='sub_epochs'):validate(c)


def test_segment_reset_preserves_fast_slow_swa_norm_and_round_lr_clocks():
    c=load_config(CONFIGS / 'smoke_test');c['optimizer'].update(lookahead_k=3,swa_period_samples=16,norm_only_at_print=False,norm_interval=5)
    model,optimizer,o=small_optimization(c);o.begin_round();calls=[];configure=o.configure
    def track():calls.append(o.round_batches);configure()
    o.configure=track
    with torch.no_grad():model.weight.fill_(5)
    for successful in [True,False]:o.before_step();o.after_step(successful)
    slow=o.slow['weight'].clone();norms=(copy.deepcopy(o.norm_sums),copy.deepcopy(o.norm_weights));swa=o.swa_samples
    o.begin_subepoch()
    assert o.subepoch==1 and o.subepoch_batches==0 and o.counter==0
    assert model.weight.item()==5 and torch.equal(slow,o.slow['weight']) and o.swa_samples==swa
    assert (o.norm_sums,o.norm_weights)==norms and o.round_batches==2 and o.consumed_samples==16 and o.optimizer_steps==1
    for _ in range(3):o.before_step();o.after_step()
    assert calls==[5] and o.subepoch_batches==3 and o.round_batches==5 and o.consumed_samples==40
    assert o.counter==0 and o.swa.n_averaged==1 and o.swa_samples==0
    assert o.optimizer_steps==4
    state=o.state_dict();other=type(o)(model,c,optimizer,state)
    assert (other.subepoch,other.subepoch_batches,other.round_batches,other.counter)==(1,3,5,0)
