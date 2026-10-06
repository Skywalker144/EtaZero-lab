"""Native training-row boundaries preserve full-horizon labels and row RNG."""
import json
import sqlite3
import subprocess
import numpy as np
import pytest
from etazero.config import ROOT
from etazero.data import Catalog,metadata,read_raw,training_view
from etazero.storage import save_npz


def produce(directory,limit,minimum=1,games=1):
    result=subprocess.run([str(ROOT/'build/record_contract_test'),str(directory),'shards',str(limit),str(minimum),str(games)],
                          check=True,capture_output=True,text=True)
    return [read_raw(json.loads(line)['file']) for line in result.stdout.splitlines()]


@pytest.mark.parametrize('limit,minimum',[(1,1),(6,1),(6,.15),(8,.15),(6,0)])
@pytest.mark.parametrize('games',[1,2])
def test_native_final_row_shards_preserve_all_targets_and_catalog(tmp_path,limit,minimum,games):
    whole=produce(tmp_path/'reference',1000,games=games)[0]
    expected=training_view(whole)
    directory=tmp_path/'selfplay/worker'
    fragments=produce(directory,limit,minimum,games=games)
    counts=[metadata(a)['rows'] for a in fragments]
    assert sum(counts)==len(expected['value'])==19*games
    assert 1<=counts[0]<=limit and all(n==limit for n in counts[1:-1])
    assert 1<=counts[-1]<=limit
    if games==1:
        assert [metadata(a)['row_begin'] for a in fragments]==np.r_[0,np.cumsum(counts[:-1])].tolist()
    views=[training_view(a) for a in fragments]
    for key in expected:
        np.testing.assert_array_equal(np.concatenate([v[key] for v in views]),expected[key],err_msg=key)
    # Include a split inside one repeated main position, the zero-repeat next
    # search supplying opponent supervision, and split repeated side positions.
    assert len(fragments)>1 and expected['opponent_policy_weight'][7:9].all()
    assert not expected['full_game_weight'][-5:].any()
    assert np.unique(expected['q_values'][:7]).size>=3
    catalog=Catalog(tmp_path,'test','config')
    try:
        catalog.scan({1:'model'})
        assert catalog.counts()==(19*games,games)
        assert catalog.previous_rows_per_game(2)==19.
        assert catalog.iteration_counts(1)==(19*games,games)
        stats=catalog.statistics(1)
        assert (stats['games'],stats['plies'],stats['rows'],stats['black_wins'])==(games,9*games,19*games,games)
        catalog.scan({1:'model'})
        assert catalog.counts()==(19*games,games)
        assert catalog.previous_rows_per_game(2)==19.
        # Rebuild from randomized filenames: fragments may be discovered out of order.
        catalog.close()
        (tmp_path/'.internal/catalog.sqlite').unlink()
        catalog=Catalog(tmp_path,'test','config');catalog.scan({1:'model'})
        assert catalog.counts()==(19*games,games)
        assert catalog.previous_rows_per_game(2)==19.
    finally:catalog.close()


@pytest.mark.parametrize('corruption',['overlap','duplicate','trajectory'])
def test_catalog_rejects_bad_fragments_without_counting_them(tmp_path,corruption):
    directory=tmp_path/'selfplay/worker'
    fragments=produce(directory,6)
    catalog=Catalog(tmp_path,'test','config')
    try:
        catalog.scan({1:'model'})
        bad={key:value.copy() for key,value in fragments[0].items()}
        m=metadata(bad);m.update(shard_id='corrupt',row_begin=2,rows=2)
        if corruption=='duplicate':m.update(row_begin=0,rows=6)
        if corruption=='trajectory':bad['search_wdl'][0]=[.5,0,.5]
        bad['metadata']=np.frombuffer(json.dumps(m).encode(),np.uint8)
        save_npz(directory/'corrupt.npz',bad)
        with pytest.raises((ValueError,sqlite3.IntegrityError)):
            catalog.scan({1:'model'})
        assert catalog.counts()==(19,1)
    finally:catalog.close()


def test_native_first_file_threshold_source_formula():
    cases=[(rows,minimum,u) for rows in (1,2,6,64,16384,20000)
           for minimum in (0,.15,.5,1) for u in (0,.125,.5,.999999999)]
    query='\n'.join(f'{r} {m} {u}' for r,m,u in cases)+'\n'
    actual=subprocess.check_output([str(ROOT/'build/record_contract_test'),'--first-limit'],input=query,text=True)
    # Source trainingwrite.cpp firstFileMaxRows formula, including truncation.
    assert list(map(int,actual.split()))==[r-int(r*(1-m)*u) for r,m,u in cases]
