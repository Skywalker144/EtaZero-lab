"""Native source fields must agree with an independent trajectory calculation."""
import numpy as np
import pytest
from test_writer import produce
from etazero.data import training_view,TRAIN_TARGETS
from etazero.source_data import decode_view,encode_view


def test_public_rows_without_private_metadata(tmp_path):
    from etazero.data import read_raw,open_raw
    from etazero.source_data import DTYPES,source_view
    original=produce(tmp_path,1000)[0]
    public={key:original[key] for key in DTYPES}
    path=tmp_path/'public.npz';np.savez_compressed(path,**public)
    actual=training_view(read_raw(path))
    expected=decode_view(original,TRAIN_TARGETS)
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
    with open_raw(path) as raw:
        assert source_view(raw,TRAIN_TARGETS).keys()==public.keys()


@pytest.mark.parametrize('limit',[1,6,1000])
def test_native_source_targets_and_quantization(tmp_path,limit):
    for raw in produce(tmp_path,limit,games=2):
        oracle=training_view(raw)
        actual=decode_view(raw,TRAIN_TARGETS)
        assert oracle.keys()==actual.keys()
        for key in oracle:
            np.testing.assert_allclose(actual[key],oracle[key],rtol=1e-6,atol=1e-7,err_msg=key)
        global_targets=raw['globalTargetsNC']
        assert global_targets.shape[1]==80
        assert global_targets[:,41:47].any()  # Stable packed game identity.
        assert ((global_targets[:,36:41]==0)|(global_targets[:,36:41]==1)).all()
        assert (np.diff(global_targets[:,36:41],axis=1)<=0).all()  # Prefix history dropout.
        assert (global_targets[:,31:33]>=0).all() and (global_targets[:,59]>=0).all()
        assert (global_targets[:,60]>=0).all()
        assert not global_targets[:,[20,21,27,29,34]].any()  # Excluded Go supervision.

        assert raw['policyTargetsNCMove'].dtype==np.int16
        assert raw['qValueTargetsNCMove'].dtype==np.int16
        assert raw['valueTargetsNCHW'].dtype==np.int8
        assert not raw['globalTargetsNC'][:,[2,6,10,14]].any()


def test_draw_and_no_result_have_separate_disk_slots(tmp_path):
    view=training_view(produce(tmp_path,1000)[0])
    view['value'][:2]=[[0,1,0,0],[0,0,0,1]]
    view['td_value'][:2]=view['value'][:2,None]
    encoded=encode_view(view)
    np.testing.assert_array_equal(encoded['globalTargetsNC'][:2,:3],[[0,0,0],[0,0,1]])
    np.testing.assert_array_equal(encoded['drawTargetsNC'][:2,0],[1,0])
    decoded=decode_view(encoded,TRAIN_TARGETS)
    np.testing.assert_array_equal(decoded['value'],view['value'])
    np.testing.assert_array_equal(decoded['td_value'],view['td_value'])


def test_real_selfplay_writes_entropy_visits_and_game_metadata(tmp_path):
    import subprocess
    from pathlib import Path
    from etazero.config import load_config,write_native
    from config_samples import CONFIGS
    from etazero.data import read_raw
    c=load_config(CONFIGS/'smoke_test')
    c['parallelism'].update(game_threads=1,search_threads=1)
    c['opening']['probability']=0;c['policy_init']['policy_init']=False
    write_native(c,tmp_path/'native.cfg')
    subprocess.run([str(Path(__file__).resolve().parents[1]/'build/etazero'),'selfplay','--config',str(tmp_path/'native.cfg'),
        '--evaluator','random','--model','random','--model-id','random','--device','cpu','--games','2',
        '--output',str(tmp_path/'raw'),'--run-id','test','--attempt-id','test','--config-id','test',
        '--source-id','test','--iteration','1','--worker','0','--seed','734'],check=True,capture_output=True,text=True,timeout=60)
    values=np.concatenate([read_raw(p)['globalTargetsNC'] for p in (tmp_path/'raw').glob('*.npz')])
    assert (values[:,59]>0).any() and (values[:,31]>0).any() and (values[:,32]>0).any()
    assert (values[:,60]>=1).all() and (values[:,51]>0).any()


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
@pytest.mark.parametrize('research',[False,True])
@pytest.mark.parametrize('dimensions',['5','3:5','5:3'])
def test_go_sgf_setup_survives_selfplay_record(tmp_path,research,dimensions,algorithm):
    import json
    import subprocess
    from etazero.config import write_native
    from etazero.data import read_raw,metadata
    from test_multigame import small_config,BINARY
    c=small_config('go',algorithm)
    if algorithm=='muzero':c['unroll']['steps']=40
    c['lcb']['use_non_buggy_lcb']=True
    c['writer'].update(research_records=research,shard_rows=1000)
    c['parallelism'].update(game_threads=1,search_threads=1)
    c['opening']['probability']=0;c['policy_init']['policy_init']=False
    c['start_positions'].update(probability=1,files=str(tmp_path/'setup.sgf'),load_probability=1,turn_weight_lambda=0)
    (tmp_path/'setup.sgf').write_text(f'(;SZ[{dimensions}]AB[aa][bc]AW[bb]PL[W];W[cc](;B[ab])(;B[ac]))')
    write_native(c,tmp_path/'native.cfg')
    run=subprocess.run([str(BINARY),'selfplay','--config',str(tmp_path/'native.cfg'),'--evaluator','random',
        '--model','random','--model-id','random','--device','cpu','--games','2','--output',str(tmp_path/'raw'),
        '--run-id','test','--attempt-id','test','--config-id','test','--source-id','test','--iteration','1',
        '--worker','0','--seed','734'],capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr
    arrays=[read_raw(p) for p in (tmp_path/'raw').glob('*.npz')]
    assert arrays
    for raw in arrays:
        assert ('observations' in raw)==(research or algorithm=='muzero')
        view=training_view(raw);assert len(view['value'])==metadata(raw)['rows']
        if research or algorithm=='muzero':
            assert (raw['initial_position_kind']==5).all()
            assert all(options['go_position'] for options in metadata(raw)['rule_options'])
            width,height=map(int,dimensions.split(':') if ':' in dimensions else [dimensions,dimensions])
            assert all((options['go_width'],options['go_height'])==(width,height)
                       for options in metadata(raw)['rule_options'])

        if algorithm=='muzero':
            roots=np.unpackbits(view['obs'][:,0],axis=1,bitorder='big')[:,:metadata(raw)['canvas']**2]
            expected=np.concatenate((roots,np.ones((len(roots),1),np.uint8)),axis=1)
            assert view['absorbing'].any()
            for row,step in zip(*np.nonzero(view['absorbing'])):
                np.testing.assert_array_equal(view['policy'][row,step],expected[row])
