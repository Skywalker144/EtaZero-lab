"""Whole-file subepochs compared directly with the pinned source selector."""
import ast
import copy
import importlib.util
from pathlib import Path
import random
import numpy as np
import pytest
from etazero.data.source_reader import SourceBatchReader
from test_replay import id_snapshot


def test_file_selection_and_variable_subepoch_lengths_match_source(tmp_path):
    source=Path('/home/sky/RL/SkyZero-lab/KataGo/python')
    if not (source/'train.py').is_file():
        pytest.skip('Pinned KataGo checkout required for direct source comparison')
    module_spec=importlib.util.spec_from_file_location('reference_generator',source/'katago/utils/training_data_generator.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    module.random=random.Random(377)
    snapshot=id_snapshot(tmp_path,[3,7,11,19])
    generator=module.TrainingDataGenerator({},False)
    generator.set_data_dir_if_has_remaining_files(str(snapshot/'data'))
    env=dict(rank=0,samples_per_epoch=13*4,batch_size=4,sub_epochs=3,
             trainingdatagenerator=generator,random=module.random,npz_num_rows_index_cache={},
             get_npz_num_rows=lambda filename,_:len(np.load(filename)['globalTargetsNC']))
    node=next(n for n in ast.walk(ast.parse((source/'train.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='get_files_for_subepoch')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'source','exec'),env)
    reader=SourceBatchReader(snapshot,4,2,377,13,3)
    lengths=[]
    try:
        for i in range(18):
            expected=env['get_files_for_subepoch']()
            count=sum(env['get_npz_num_rows'](f,None)//4 for f in expected)
            lengths.append(count)
            for b in range(count):
                reader.next();state=reader.state()
                assert state['selected']==tuple(expected)
                assert (state['epoch'],state['subepoch'],state['subepoch_batches'])==(i//3,i%3,b+1)
                assert state['epoch_end']==(i%3==2 and b==count-1)
    finally:reader.close()
    assert len(set(lengths))>1 and any(n!=13//3 for n in lengths)


def test_pause_and_resume_preserve_file_queue_batch_cursor_and_rng(tmp_path):
    snapshot=id_snapshot(tmp_path,[7,11,19])
    reader=SourceBatchReader(snapshot,4,2,19,13,3)
    whole=[]
    for _ in range(70):
        batch=reader.next();whole.append((batch,copy.deepcopy(reader.state())))
    reader.close()
    state=None
    for lo,hi in zip([0,1,4,5,9,18,37,65],[1,4,5,9,18,37,65,70]):
        reader=SourceBatchReader(snapshot,4,2,19,13,3,state,max_batches=hi-lo)
        try:
            for step in range(lo,hi):
                actual=reader.next();state=copy.deepcopy(reader.state())
                for key in actual:np.testing.assert_array_equal(actual[key],whole[step][0][key])
                assert state==whole[step][1]
        finally:reader.close()
