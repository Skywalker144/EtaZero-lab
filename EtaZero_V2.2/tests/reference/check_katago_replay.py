"""Pinned, unmodified source window/groups/file order and whole-batch reader oracle."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'tests'))
from etazero.config import load_config
from etazero.shuffle import desired_window,_groups,resource_plan,validation_file
from etazero.reader import BatchReader
from test_replay import id_snapshot


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    reference=json.loads((ROOT/'reference_sources.json').read_text())['KataGo'];kg=Path(reference['root'])
    assert subprocess.check_output(['git','-C',str(kg),'rev-parse','HEAD'],text=True).strip()==reference['commit']
    paths=['python/shuffle.py','python/katago/utils/training_data_generator.py','python/katago/train/data_processing_pytorch.py','python/katago/train/modelconfigs.py','python/selfplay/shuffle.sh','python/selfplay/synchronous_loop.sh']
    hashes={path:hashlib.sha256((kg/path).read_bytes()).hexdigest() for path in paths}
    assert all(hashes[p]==reference['sha256'][p] for p in paths)
    source=module(kg/'python/shuffle.py','source_shuffle');generator=module(kg/paths[1],'source_generator')
    sys.path.insert(0,str(kg/'python'))
    from katago.train import data_processing_pytorch,modelconfigs
    c=load_config(ROOT / 'tests/fixtures/configs/smoke_test', run_dir='/tmp/etazero_reference_sample');r=c['replay'];formula=groups=plans=orders=batches=0
    for minimum,p,a,scale,offset,maximum,extra in itertools.product([9,100,250000],[.3,.65,1,1.4],[0,.4,1],
                    [0,13,300000],[-1.5,0,123.75],['all',500000],[0,1,1000,20000000]):
        r.update(min_rows=minimum,taper_exponent=p,expand_per_row=a,taper_scale=scale,add_to_data_rows=offset,max_rows=maximum)
        expected=source.compute_desired_num_rows(minimum+extra,minimum,offset,p,a,scale or None,None if maximum=='all' else maximum)
        assert desired_window(minimum+extra,r)==expected;formula+=1
    r.update(min_rows=9,taper_exponent=1,expand_per_row=1,taper_scale=0,add_to_data_rows=0,max_rows='all',keep_target_rows='all')
    for counts,limit in itertools.product([[6,6,1],[0,12,1],[1]*17,[0,0],[9]*11],[1,9,10,18,52,100]):
        items=[(str(i),'hash',n) for i,n in enumerate(counts)]
        actual=[[p for p,_ in group] for group in _groups(items,limit)]
        assert actual==source.group_files_by_rows([(p,n) for p,_,n in items],limit);groups+=1
    for rows,waves,bucket,nominal,q in itertools.product([0,1,9,31,32,48,80,99,5000],[1,3],[16,32,96],[1,8],[.1,.5,1]):
        c['shuffle'].update(bucket_rows=bucket,training_shard_rows=nominal,waves=waves)
        plan=resource_plan(rows,9,3,c,q)
        assert (plan['buckets_per_wave'],plan['files_per_bucket'])==source.compute_buckets_and_out_files(rows*q/waves,bucket,nominal);plans+=1
    # The source's basename MD5 expression, checked independently over 10K names.
    text=(kg/'python/shuffle.py').read_text();assert 'hashlib.md5' in text and '[:13]' in text
    md5=0
    for i in range(10000):
        name=f'{i:016x}.npz';fraction=int(hashlib.md5(name.encode()).hexdigest()[:13],16)/2**52
        assert validation_file('/first/'+name)==validation_file('/different/'+name)==(fraction>=.99);md5+=1
    with tempfile.TemporaryDirectory(prefix='etazero_source_replay_') as temporary:
        root=Path(temporary)
        for count in [1,2,3,4,9,31]:
            snapshot=id_snapshot(root/str(count),[5]*count)
            for seed in [0,1,17,999]:
                reader=BatchReader(snapshot,4,1,seed)
                original_random=generator.random;original_listdir=generator.os.listdir
                generator.random=random.Random(seed)
                names=[f['path'] for f in reader.files]
                generator.os.listdir=lambda path:names
                state={};oracle=generator.TrainingDataGenerator(state,False)
                try:
                    assert oracle.set_data_dir_if_has_remaining_files(str(snapshot/'data'))
                    expected=[names.index(Path(oracle.pop()).name) for _ in names]
                    assert reader.order==expected and reader.random.getstate()==generator.random.getstate();orders+=1
                    for _ in range(10):
                        reader.used=reader.order[:];reader.index=len(reader.order);reader._new_pass()
                        expected=oracle._reshuffle_for_new_epoch()
                        expected=[names.index(Path(p).name) for p in expected]
                        assert reader.order==expected and reader.random.getstate()==generator.random.getstate();orders+=1
                        state['data_files_used']=[str(snapshot/'data'/names[i]) for i in expected]
                finally:
                    generator.random=original_random;generator.os.listdir=original_listdir;reader.close()
        # Execute the ORIGINAL Go reader with minimal Go-shaped fixtures; compare
        # only whole-file row identity, retaining all required source array layouts.
        counts=[0,1,3,4,5,9,17];snapshot=id_snapshot(root/'prefix',counts)
        paths=[];mc=modelconfigs.config_of_name['b5c192nbt-fson-mish'];nb=modelconfigs.get_num_bin_input_features(mc);ng=modelconfigs.get_num_global_input_features(mc)
        for i,n in enumerate(counts):
            globals=np.zeros((n,ng),np.float32);globals[:,5]=np.arange(n)+100*i
            arrays=dict(binaryInputNCHWPacked=np.zeros((n,nb,4),np.uint8),globalInputNC=globals,
                        policyTargetsNCMove=np.zeros((n,2,26),np.int16),globalTargetsNC=np.zeros((n,80),np.float32),
                        scoreDistrN=np.zeros((n,52),np.int8),valueTargetsNCHW=np.zeros((n,5,5,5),np.int8))
            path=root/f'go_{i}.npz';np.savez_compressed(path,**arrays);paths.append(str(path))
        for size in [1,2,4,8,16]:
            reader=BatchReader(snapshot,size,2,1,no_repeat_files=True,shuffle_files=False)
            oracle=data_processing_pytorch.read_npz_training_data(paths,size,1,0,5,torch.device('cpu'),False,False,mc,prefetch_depth=2)
            try:
                for expected in oracle:
                    actual=reader.next();np.testing.assert_array_equal(actual['globals'][:,5],expected['globalInputNC'][:,5].numpy());batches+=1
                try:reader.next()
                except StopIteration:pass
                else:raise AssertionError('no-repeat implicitly reused files')
            finally:reader.close()
    result=dict(status='verified',source_commit=reference['commit'],source_sha256=hashes,
                cases=dict(window_formula=formula,group_threshold=groups,output_plan=plans,basename_md5=md5,file_orders=orders,source_reader_batches=batches),
                scope='Unmodified pinned source helpers, TrainingDataGenerator and full NPZ reader; identical RNG file orders and RNG state; original Go-shaped fixtures and EtaZero row-ID prefixes; whole-file tails and explicit exhaustion.',
                adaptations='Go random/tdata identity maps to model_id=random; single learner world_size=1; immutable per-round snapshots instead of live directory reconciliation; private deterministic shuffle/validation streams, no source gameplay RNG sequence equivalence; resource limits are explicit and do not recursively change planned bucket/file counts.')
    assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
