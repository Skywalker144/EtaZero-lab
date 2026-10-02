import copy
import json
from pathlib import Path
import shutil
import pytest
from etazero.config import ROOT, fingerprint, load_config
from etazero.eval_config import load_evaluation_config
from etazero.arena import Player, build_schedule, discover_players, PairStore, save_manifest
from etazero.runtime import recover_iteration
from etazero.storage import save_json, load_json, sha256
from etazero.schema import CONTRACT_ID


def test_eval_profiles_are_independent(tmp_path,monkeypatch):
    shutil.copytree(ROOT/'configs',tmp_path/'configs')
    monkeypatch.setattr('etazero.config.ROOT',tmp_path)
    selected=tmp_path/'configs/smoke_test'
    before=load_config(selected)
    (selected/'eval.cfg.local').write_text('[evaluation]\nvisits = 71\n')
    assert load_config(selected)==before
    assert fingerprint(load_config(selected))==fingerprint(before)
    assert load_evaluation_config(selected,environ={})['evaluation']['visits']==71
    assert load_evaluation_config(selected,environ={'EVAL_VISITS':'100','MATCH_VISITS':'3'})['evaluation']['visits']==100
    assert load_evaluation_config(selected,match=True,environ={'EVAL_VISITS':'3'})['match']['visits']==100
    # Changing training search/inference does not alter either evaluator.
    evaluation=load_evaluation_config(selected,environ={})
    match=load_evaluation_config(selected,True,environ={})
    (selected/'selfplay.cfg.local').write_text('[puct]\nc_puct=7\n[inference]\ninference_precision=float16\n')
    assert load_evaluation_config(selected,environ={})==evaluation
    assert load_evaluation_config(selected,True,environ={})==match
    (selected/'eval.cfg.local').write_text('[evaluation]\nvisits=oops\n')
    assert load_config(selected)['puct']['c_puct']==7
    with pytest.raises(ValueError):load_evaluation_config(selected,environ={})
    for field,value in [('VISITS','0'),('BOARD_SIZE','4'),('RULE','oops'),('OPENING_PROBABILITY','0'),
                        ('ROOT_NUM_SYMMETRIES_TO_SAMPLE','9'),('NN_POLICY_TEMPERATURE','0'),
                        ('ROOT_POLICY_TEMPERATURE','0'),('TEMPERATURE_HALFLIFE','0'),('TEMPERATURE_ONLY_BELOW_PROB','1.1')]:
        with pytest.raises(ValueError):load_evaluation_config(ROOT/'configs/baseline',True,environ={'MATCH_'+field:value})


def test_manifest_and_partial_pair_resume(tmp_path):
    manifest={'games_per_pair':4,'config':{'visits':100},'pairs':[]}
    save_manifest(tmp_path/'manifest.json',manifest)
    save_manifest(tmp_path/'manifest.json',{**manifest,'games_per_pair':8})
    with pytest.raises(ValueError):save_manifest(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError):save_manifest(tmp_path/'manifest.json',{**manifest,'games_per_pair':8,'config':{'visits':200}})
    store=PairStore(tmp_path/'pair',4)
    opening={'type':'opening','id':0,'generator':0,'seed':71,'moves':[12],'attempts':1,'value':.1}
    game={'type':'game','id':0,'opening_id':0,'black_a':True,'winner':1,'moves':[12,0],'seconds':.2}
    store.accept(opening);store.accept(game)
    tasks=store.tasks(71)
    assert tasks[0]['mask']==2 and tasks[0]['moves']==[12]
    assert len(store.games())==1
    with pytest.raises(ValueError):store.accept({**game,'winner':-1})
    with pytest.raises(ValueError):PairStore(tmp_path,6)
    assert PairStore(tmp_path/'pair',4).tasks(71)==tasks


def test_committed_selection_and_schedule(tmp_path):
    arm=tmp_path/'a'
    save_json(arm/'.internal/state.json',{'iteration':6,'elapsed_seconds':25})
    for i in range(7):
        model=None
        if i:
            path=arm/'models'/str(i)/'model.pt';path.parent.mkdir(parents=True);path.write_bytes(str(i).encode())
            model={'id':str(i),'path':str(path.relative_to(arm)),'sha256':sha256(path),'contract':CONTRACT_ID,'canvas':6}
            save_json(path.parent/'manifest.json',model)
        save_json(arm/'logs/iterations'/f'{i:06d}.json',{'iteration':i,'elapsed_seconds':i*i,'model':model})
    players=discover_players(tmp_path,2)
    assert [p.iteration for p in players]==[1,2,4,5]
    assert [p.seconds for p in players]==[1,4,16,25]
    assert ('a:00000001','a:00000004') in build_schedule(players,2)
    cross=players+[Player('b:1','b',1,9,'/b','hash')]
    assert any('b:1' in pair for pair in build_schedule(cross,2))
    (arm/'models/2/model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):discover_players(tmp_path,2)


def test_rollback_preserves_committed_data_and_archives_uncommitted_time(tmp_path):
    state={'iteration':2,'elapsed_seconds':17.,'checkpoint':{'path':'checkpoints/iteration_000001_done.pt'},'model':None}
    committed=tmp_path/'selfplay/iteration_000001/raw.npz';committed.parent.mkdir(parents=True);committed.write_bytes(b'committed')
    paths=['selfplay/iteration_000002/raw.npz','.internal/iterations/000002/learner.json',
           'checkpoints/iteration_000002_step_01.pt','models/iteration_000002_a/model.pt',
           'snapshots/iteration_000002_a/data.npz','logs/iterations/000002.json','.internal/catalog.sqlite']
    for relative in paths:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'uncommitted')
    recover_iteration(tmp_path,state)
    assert committed.read_bytes()==b'committed' and state['elapsed_seconds']==17
    for relative in paths:
        assert not (tmp_path/relative).exists()
        saved=list((tmp_path/'.internal/discarded').glob('*/'+relative))
        assert len(saved)==1 and saved[0].read_bytes()==b'uncommitted'
    archives=list((tmp_path/'.internal/discarded').iterdir())
    recover_iteration(tmp_path,state)
    assert list((tmp_path/'.internal/discarded').iterdir())==archives


def test_cli_requires_explicit_model_source():
    import subprocess
    result=subprocess.run(['bash',str(ROOT/'scripts/run.sh'),'evaluate','--config-dir',str(ROOT/'configs/smoke_test')],
                          text=True,capture_output=True)
    assert result.returncode!=0 and 'requires --model or --run-dir' in result.stderr
    help_result=subprocess.run(['bash',str(ROOT/'scripts/run.sh'),'--help'],text=True,capture_output=True)
    assert help_result.returncode==0 and 'arena' in help_result.stdout
