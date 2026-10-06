"""Fixed-round production quota and independent round/segment/learner clocks."""
from config_samples import CONFIGS
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from etazero.config import ROOT,load_config,validate
from etazero.runtime import Controller,iteration_plan
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


def test_planning_and_backfill_keep_previous_iterations_estimate():
    c=load_config(CONFIGS / 'smoke_test');driver=controller(c,100,20);quotas=[]
    def launch(plan,games):
        quotas.append(games)
        driver.catalog.rows+=games*2;driver.catalog.games+=games;driver.catalog.round_games+=games
    driver.launch=launch;driver.produce(dict(iteration=3,target_rows=115))
    assert quotas==[3,2,1,1,1] and driver.catalog.rows==116
    events=[fields for event,fields in driver.events if event in ('selfplay_target','selfplay_backfill')]
    assert [f['rows_per_game'] for f in events]==[5]*5


def test_postbootstrap_backfills_entire_quota_using_actual_short_games():
    c=load_config(CONFIGS / 'smoke_test');c['replay']['min_rows']=32;driver=controller(c);quotas=[]
    def launch(plan,games):
        quotas.append(games);driver.catalog.rows+=games*3;driver.catalog.games+=games;driver.catalog.round_games+=games
    driver.launch=launch;driver.produce(dict(iteration=2,target_rows=150))
    # Initial estimate100 rows/game is deliberately too high. min_rows is
    # already satisfied; source quota150 must still be filled before shuffle.
    assert quotas[0]==1 and len(quotas)>1 and sum(quotas)==17 and driver.catalog.rows==151
    backfills=[fields for event,fields in driver.events if event=='selfplay_backfill']
    assert backfills and all(f['deficit']==150-f['rows']>0 for f in backfills)


def test_surplus_carry_over_minimum_game_and_no_progress():
    c=load_config(CONFIGS / 'smoke_test');driver=controller(c,200,20);quotas=[]
    def launch(plan,games):
        quotas.append(games);driver.catalog.rows+=games*9;driver.catalog.games+=games;driver.catalog.round_games+=games
    driver.launch=launch;driver.produce(dict(iteration=2,target_rows=150))
    assert quotas==[1] and driver.catalog.rows==209
    driver.produce(dict(iteration=2,target_rows=150));assert quotas==[1] # No duplicate completed game.
    empty=controller(c,100,0);empty.launch=lambda p,g:pytest.fail('No estimate must fail before launching')
    with pytest.raises(RuntimeError,match='rows-per-game'):empty.produce(dict(iteration=2,target_rows=150))
    stuck=controller(c);stuck.launch=lambda p,g:None
    with pytest.raises(RuntimeError,match='valid training rows'):stuck.produce(dict(iteration=2,target_rows=150))
    bootstrap=controller(c,0,0);bootstrap.launch=lambda p,g:None
    with pytest.raises(RuntimeError,match='Bootstrap'):bootstrap.produce(dict(iteration=0,target_rows=0))


def test_completed_zero_row_games_do_not_abort_quota_backfill():
    c=load_config(CONFIGS / 'smoke_test');driver=controller(c,100,20);quotas=[]
    def launch(plan,games):
        quotas.append(games)
        driver.catalog.games+=games;driver.catalog.round_games+=games
        # Legal stochastic rounding gives the first completed batch zero rows.
        if len(quotas)>1:driver.catalog.rows+=games*5
    driver.launch=launch;driver.produce(dict(iteration=2,target_rows=110))
    assert quotas==[2,2] and driver.catalog.rows==110
    estimates=[f['rows_per_game'] for e,f in driver.events if e in ('selfplay_target','selfplay_backfill')]
    assert estimates==[5,5]


def test_cold_anchor_fixed_consumption_quota_and_random_id_are_separate():
    c=load_config(CONFIGS / 'baseline');assert c['training']['sub_epochs']==1
    state=dict(iteration=2,model={'id':'net'},checkpoint={'id':'trained'},target_rows=300123,replay_origin_rows=300123)
    for i in range(2,8):
        state['iteration']=i;plan=iteration_plan(state,c)
        assert plan['target_rows']==300123+(i-1)*16000 and plan['train_steps']*plan['batch_size']==128000
        state['target_rows']=plan['target_rows']
    entries=[{'metadata':{'model_id':'random:1','rows':300123}}, {'metadata':{'model_id':'iteration_2','rows':16000}}]
    assert replay_counts(entries,250000)==dict(raw_rows=316123,random_rows=300123,postrandom_rows=16000,usable_rows=266000)
    assert state['replay_origin_rows']==300123


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
