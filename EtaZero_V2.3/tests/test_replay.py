"""Source replay semantics, complete-file row IDs, holdout and immutable rebuild."""
from config_samples import CONFIGS
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from etazero.config import ROOT,load_config,validate
from etazero.data import Catalog,training_view
from etazero.reader import BatchReader,BatchPrefetcher
from etazero.schema import CONTRACT_ID
from etazero.shuffle import build_snapshot,desired_window,window_sources,_groups,resource_plan,validation_file,prune_derived,restore_snapshot
from etazero.storage import save_npz,save_json,sha256
from etazero.network import make_network,TrainingForward
from etazero.training import validate_epoch
from test_python import winning_record


def id_snapshot(root,counts,validation=False):
    snapshot=Path(root)/'snapshots/ids';directory=snapshot/'data';directory.mkdir(parents=True)
    base=training_view(winning_record()[0]);files=[];val_files=[]
    for index,n in enumerate(counts):
        arrays={k:v[np.arange(n)%len(v)].copy() for k,v in base.items()}
        ids=np.arange(n)+index*100
        arrays['globals'][:,5]=ids
        arrays['q_values'][:,0]=ids
        filename=f'{index:03}.npz';save_npz(directory/filename,arrays)
        info={'path':filename,'rows':n,'sha256':sha256(directory/filename)};files.append(info)
        if validation:
            save_npz(directory/'validation'/filename,arrays);val_files.append(info.copy())
    save_json(snapshot/'manifest.json',{'id':'ids','contract':CONTRACT_ID,'canvas':6,'rows':sum(counts),'files':files,
                                      'validation_rows':sum(counts) if validation else 0,'validation_files':val_files})
    return snapshot


def test_complete_file_prefixes_no_repeat_exhaustion_resume_and_prefetch(tmp_path):
    counts=[0,1,3,4,5,9,17];snapshot=id_snapshot(tmp_path,counts)
    expected=[[i*100+j for j in range(start,start+4)] for i,n in enumerate(counts) for start in range(0,n//4*4,4)]
    reader=BatchReader(snapshot,4,3,17,no_repeat_files=True)
    prepared=BatchPrefetcher(reader,2);actual=[]
    try:
        first,state=prepared.next();actual.append(first['globals'][:,5].tolist())
        resumed=BatchReader(snapshot,4,1,999,state,no_repeat_files=True)
        try:
            while True:
                try:batch,cursor=prepared.next()
                except StopIteration:break
                other=resumed.next()
                for key in batch:np.testing.assert_array_equal(batch[key],other[key])
                np.testing.assert_array_equal(batch['globals'][:,5],batch['q_values'][:,0])
                actual.append(batch['globals'][:,5].tolist())
            for _ in range(2):
                with pytest.raises(StopIteration):resumed.next()
            exhausted=resumed.state()
        finally:resumed.close()
        again=BatchReader(snapshot,4,1,1,exhausted,no_repeat_files=True)
        try:
            with pytest.raises(StopIteration):again.next()
        finally:again.close()
    finally:prepared.close();reader.close()
    assert sorted(actual)==sorted(expected)
    assert sum(map(len,actual))==sum(n//4*4 for n in counts)==32
    with pytest.raises(ValueError,match='complete per-file'):
        BatchReader(snapshot,100,1,1)


def test_repeat_gap_order_retains_each_file_and_drops_tails(tmp_path):
    snapshot=id_snapshot(tmp_path,[5]*9);reader=BatchReader(snapshot,4,2,11)
    try:
        orders=[]
        for _ in range(5):
            orders.append([int(reader.next()['globals'][0,5]//100) for _ in range(9)])
            assert sorted(orders[-1])==list(range(9))
        for prev,current in zip(orders,orders[1:]):
            assert min(9+current.index(i)-prev.index(i) for i in range(9))>=3
    finally:reader.close()


def test_capped_random_and_actual_mtime_window_range():
    c=load_config(CONFIGS / 'smoke_test');r=c['replay'];r.update(min_rows=100,taper_exponent=1,expand_per_row=1)
    entries=[{'path':'newrandom','metadata':{'model_id':'random:1','rows':180,'iteration_id':0,'created_ns':0},'mtime_ns':30},
             {'path':'oldpost','metadata':{'model_id':'model','rows':70,'iteration_id':9,'created_ns':900},'mtime_ns':10},
             {'path':'recentpost','metadata':{'model_id':'model','rows':20,'iteration_id':1,'created_ns':1},'mtime_ns':20}]
    chosen,selection=window_sources(entries,r)
    assert selection==dict(raw_rows=270,random_rows=180,postrandom_rows=90,usable_rows=190,desired_rows=190,window_rows=200,range=[70,270])
    assert [e['path'] for e in chosen]==['newrandom','recentpost']
    r.update(taper_scale=200,add_to_data_rows=10.5,max_rows=150)
    assert desired_window(190,r)==150
    chosen,selection=window_sources(entries,r);assert selection['range']==[100,280]
    r['add_to_data_rows']=-200
    with pytest.raises(ValueError,match='taper_scale'):validate(c)
    r['add_to_data_rows']=0;r['max_rows']=99
    with pytest.raises(ValueError,match='max_rows'):validate(c)
    assert _groups([('a','A',6),('empty','E',0),('b','B',6),('c','C',1)],10)==[[('a','A'),('b','B')],[('c','C')]]


def holdout_entries(root,copies=12):
    entries=[];train_names=[];val_names=[]
    for i in range(5000):
        name=f'raw_{i:05}.npz';value=int(hashlib.md5(name.encode()).hexdigest()[:13],16)/2**52
        (val_names if value>=.99 else train_names).append(name)
    for index,name in enumerate(train_names[:copies]+val_names[:copies]):
        raw,meta=winning_record(index%3);meta.update(shard_id=name,attempt_id=name)
        raw['metadata']=np.frombuffer(json.dumps(meta).encode(),np.uint8)
        path=Path(root)/'selfplay'/name;save_npz(path,raw)
        entries.append({'path':str(path.relative_to(root)),'sha256':sha256(path),'metadata':meta})
    return entries


def test_validation_md5_holdout_shared_probability_across_snapshots_and_rebuild(tmp_path):
    c=load_config(CONFIGS / 'smoke_test');c['training']['skip_validation']=False
    c['replay'].update(min_rows=9,taper_exponent=1,expand_per_row=1,keep_target_rows=108)
    c['shuffle'].update(group_rows=18,bucket_rows=16,training_shard_rows=8,waves=3)
    entries=holdout_entries(tmp_path);all_train=set();all_val=set()
    class RecordingPool:
        def __init__(self):self.streams=[]
        def map(self,function,tasks):
            for task in tasks:
                if function.__name__=='_scatter':self.streams.append(tuple(task[5])+(task[0],))
                elif function.__name__=='_merge_bucket':self.streams.append(tuple(task[6]))
                yield function(task)
    for iteration in (1,2):
        pool=RecordingPool()
        identity=build_snapshot(tmp_path,iteration,entries,c,pool=pool);snapshot=tmp_path/'snapshots'/identity
        assert len(pool.streams)==len(set(pool.streams)) # Both partitions, every wave/stage/group/bucket.
        m=json.loads((snapshot/'manifest.json').read_text());train={e['path'] for e in m['sources']};val={e['path'] for e in m['validation_sources']}
        assert len(train)==len(val)==12 and not train&val and train|val=={e['path'] for e in entries}
        assert all(not validation_file(p) for p in train) and all(validation_file(p) for p in val)
        assert m['keep_prob']==m['validation_resource_plan']['keep_prob']==.5
        assert m['rows']==m['validation_rows']==54
        assert not train&all_val and not val&all_train;all_train|=train;all_val|=val
        hashes={str(p.relative_to(snapshot)):sha256(p) for p in (snapshot/'data').rglob('*.npz')}
        prune_derived(tmp_path,[identity],set());restore_snapshot(snapshot)
        assert hashes=={str(p.relative_to(snapshot)):sha256(p) for p in (snapshot/'data').rglob('*.npz')}
    assert all(sha256(tmp_path/e['path'])==e['sha256'] for e in entries)
    c['training']['skip_validation']=True
    identity=build_snapshot(tmp_path,3,entries,c);m=json.loads((tmp_path/'snapshots'/identity/'manifest.json').read_text())
    assert len(m['sources'])==24 and m['validation_rows']==0 and not m['validation_files']


def test_catalog_refreshes_mtime_and_rejects_changed_known_raw(tmp_path):
    entries=holdout_entries(tmp_path,2);catalog=Catalog(tmp_path,'test','config')
    try:
        catalog.scan({1:'model'});path=tmp_path/entries[0]['path'];stamp=path.stat().st_mtime_ns
        os.utime(path,ns=(stamp,stamp+10**12));catalog.scan({1:'model'})
        assert catalog.entries()[-1]['path']==entries[0]['path']
        assert catalog.entries()[-1]['mtime_ns']==stamp+10**12
        with path.open('ab') as f:f.write(b'changed')
        with pytest.raises(ValueError,match='changed'):catalog.scan({1:'model'})
    finally:catalog.close()


def test_validation_raw_eval_d4_cap_and_training_state_unchanged(tmp_path):
    c=load_config(CONFIGS / 'smoke_test');c['training'].update(skip_validation=False,d4_augmentation=False,max_validation_samples=8)
    snapshot=id_snapshot(tmp_path,[17,9],True)
    # IDs identify synthetic rows, but are not valid numerical Q targets.
    for info in json.loads((snapshot/'manifest.json').read_text())['validation_files']:
        path=snapshot/'data/validation'/info['path']
        with np.load(path) as f:arrays={k:f[k] for k in f.files}
        arrays['q_values'].fill(0);save_npz(path.with_suffix('.new.npz'),arrays);os.replace(path.with_suffix('.new.npz'),path)
    m=json.loads((snapshot/'manifest.json').read_text())
    for info in m['validation_files']:info['sha256']=sha256(snapshot/'data/validation'/info['path'])
    (snapshot/'manifest.json').write_text(json.dumps(m))
    model=make_network(c).train();forward=TrainingForward(model,8,False);before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state();events=[]
    validate_epoch(snapshot,model,forward,c,1,torch.device('cpu'),lambda event,**fields:events.append((event,fields)))
    assert model.training and torch.equal(rng,torch.get_rng_state())
    for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())
    event,fields=events[0];assert event=='validation' and fields['samples']==16 and fields['batches']==2 and fields['model']=='raw'
    assert sum(fields['symmetry_counts'])==2 and fields['symmetry_counts'][0]!=2
    assert np.isfinite(fields['loss'])


def test_scatter_row_ids_pairing_rounding_and_independent_wave_streams(tmp_path):
    from etazero.shuffle import _scatter,_merge_bucket
    source=tmp_path/'input.npz';ids=np.arange(12,dtype=np.float32)
    save_npz(source,dict(value=np.stack([ids,ids+100,ids+200],axis=1),marker=ids[:,None]))
    # Source round tie: 12*0.375=4.5 rounds to 4 (ties-to-even).
    root=tmp_path/'scatter';outputs=_scatter((0,[(str(source),None)],False,str(root),4,(17,2,0),str(tmp_path),.375,True,()))
    selected=[]
    for bucket,path,n in outputs:
        with np.load(path) as a:
            np.testing.assert_array_equal(a['marker'][:,0],a['value'][:,0]);selected.extend(a['marker'][:,0].tolist())
    assert len(outputs)==4 and len(selected)==len(set(selected))==4
    # Equal source groups assigned to different waves must not reuse permutations.
    orders=[]
    for wave in range(3):
        path=tmp_path/f'merge_input_{wave}.npz'
        save_npz(path,dict(value=np.stack([ids,ids+100,ids+200],axis=1),marker=ids[:,None]))
        files=_merge_bucket(([str(path)],12,str(tmp_path),str(tmp_path/f'out_{wave}'),100,1,(17,3,wave,0),str(wave),True))
        with np.load(tmp_path/f'out_{wave}'/files[0]['path']) as a:
            orders.append(a['marker'][:,0].tolist());np.testing.assert_array_equal(a['marker'][:,0],a['value'][:,0])
        assert sorted(orders[-1])==ids.tolist()
    assert len({tuple(order) for order in orders})==3
    with pytest.raises(ValueError,match='exceeds shuffle array memory'):
        _merge_bucket(([],13,str(tmp_path),str(tmp_path),12,1,(17,3,0,0),'overflow',True))
