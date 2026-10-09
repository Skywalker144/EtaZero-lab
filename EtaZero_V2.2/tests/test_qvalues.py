from etazero.schema import GLOBALS
"""Pure W-L Q semantics: independent calculus and immutable per-writer-row data."""
import copy
import json
import math
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
import torch
from etazero.config import ROOT,validate
from etazero.network import q_winloss_loss,losses,make_network
from etazero.data import validate_raw,training_view
from etazero.reader import BatchReader
from etazero.shuffle import build_snapshot
from etazero.storage import save_npz,sha256
from etazero.symmetry import augment_batch
from test_python import winning_record,compact_search,config


def test_q_loss_calculus_zero_visits_and_real_source():
    x=torch.tensor([[.8,-.3,9.],[-.2,.4,.1],[4.,-8.,3.]],requires_grad=True)
    target=torch.tensor([[1.,-.25,0.],[-1.,0.,.5],[0.,0.,0.]])
    visits=torch.tensor([[1.,9.,0.],[4.,16.,25.],[0.,0.,0.]])
    expected=0.;gradient=np.zeros((3,3))
    for n in range(3):
        denominator=1+sum(math.sqrt(float(v)) for v in visits[n])
        for a in range(3):
            weight=math.sqrt(float(visits[n,a]));z=2*float(x.detach()[n,a]) if weight else 0
            p=(1+float(target[n,a]))/2
            expected+=1.5*weight*(max(z,0)+math.log1p(math.exp(-abs(z)))-p*z)/denominator/3
            gradient[n,a]=1.5*weight*2*(1/(1+math.exp(-z))-p)/denominator/3
    actual=q_winloss_loss(x,target,visits)
    assert float(actual.detach())==pytest.approx(expected,rel=2e-7)
    actual.backward();np.testing.assert_allclose(x.grad,gradient,rtol=5e-7,atol=2e-8)
    assert not x.grad[2].any() and x.grad[0,2]==0
    ref=json.loads((ROOT/'reference_sources.json').read_text())['KataGo']
    source=Path(ref['root'])/'python/katago/train/metrics_pytorch.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==ref['sha256']['python/katago/train/metrics_pytorch.py']
    sys.path.insert(0,str(Path(ref['root'])/'python'))
    from katago.train.metrics_pytorch import Metrics
    probe=x.detach().clone().requires_grad_()
    value,_=Metrics.loss_qvalues_samplewise(SimpleNamespace(policy_len=3,scoremean_multiplier=20),
             probe,torch.zeros_like(probe),target,torch.zeros_like(target),visits,torch.ones(3))
    torch.testing.assert_close(actual.detach(),value.mean(),rtol=0,atol=0)
    value.mean().backward();torch.testing.assert_close(x.grad,probe.grad,rtol=0,atol=0)


def test_side_q_is_not_gated_by_game_outcome():
    p=torch.zeros(2,7,4,requires_grad=True);obs=torch.zeros(2,5,2,2);obs[:,0]=1
    policy=torch.ones(2,4);wdl=torch.tensor([[.2,.3,.5]]*2)
    result=losses(p,torch.zeros(2,3),torch.zeros(2,3,3),torch.zeros(2),obs,
                  policy,policy,torch.zeros(2),wdl,wdl[:,None].expand(-1,3,-1),
                  torch.zeros(2),8.,False,torch.tensor([[1.,-1.,0.,0.]]*2),torch.tensor([[9.,4.,0.,0.]]*2))
    assert len(result)==13 and result[-1]>0
    result[-1].backward();assert p.grad[:,6,:2].count_nonzero()==4
    assert not p.grad[:,:6].any() and not p.grad[:,6,2:].any()
    with pytest.raises(ValueError,match='requires'):
        losses(p,*[torch.zeros(2,3),torch.zeros(2,3,3),torch.zeros(2),obs,policy,policy,
                   torch.zeros(2),wdl,wdl[:,None].expand(-1,3,-1),torch.zeros(2),8.,False])


@pytest.mark.parametrize('symmetry',range(8))
def test_q_d4_independent_coordinate_permutation(symmetry):
    values=torch.arange(9).reshape(1,9).float();visits=100+values
    batch={'globals':torch.zeros(1,len(GLOBALS)),'obs':torch.ones(1,5,3,3),'policy':values,'opponent_policy':values,
           'q_values':values,'q_visits':visits}
    actual=augment_batch(batch,symmetry)
    expected=np.empty((3,3))
    for y in range(3):
        for x in range(3):
            yy,xx=[(y,x),(2-x,y),(2-y,2-x),(x,2-y),(x,y),(y,2-x),(2-x,2-y),(2-y,x)][symmetry]
            expected[yy,xx]=3*y+x
    np.testing.assert_array_equal(actual['q_values'].reshape(3,3),expected)
    np.testing.assert_array_equal(actual['q_visits'].reshape(3,3),expected+100)


def test_q_per_repeat_values_survive_shuffle_reader_and_resume(tmp_path,config):
    config['network'].update(architecture='transformer',blocks=5,channels=192,predict_q_values=True)
    raw,meta=winning_record(2)
    raw['row_repeats'][0]=3;raw['target_weights'][0]=3;compact_search(raw)
    meta['rows']=11;raw['metadata']=np.frombuffer(json.dumps(meta).encode(),np.uint8)
    raw['q_values'][:3,0]=[-8001,-8000,-8001]
    raw['q_visits'][0,0]=32000
    validate_raw(raw);view=training_view(raw)
    np.testing.assert_array_equal(view['q_values'][:3,0],np.array([-8001,-8000,-8001],np.float32)/32000)
    assert view['q_visits'][:3,0].tolist()==[32000]*3
    path=tmp_path/'selfplay/q.npz';save_npz(path,raw)
    config['replay'].update(min_rows=11,taper_exponent=1,expand_per_row=1,keep_target_rows='all')
    config['shuffle'].update(bucket_rows=2,training_shard_rows=2,temp_dir=str(tmp_path/'scratch'))
    identity=build_snapshot(tmp_path,1,[{'path':str(path.relative_to(tmp_path)),'sha256':sha256(path),'metadata':meta}],config)
    snapshot=tmp_path/'snapshots'/identity
    manifest=json.loads((snapshot/'manifest.json').read_text())
    expected=Counter(zip(map(bytes,view['q_values']),map(bytes,view['q_visits'])))
    actual=Counter()
    for item in manifest['files']:
        with np.load(snapshot/'data'/item['path']) as data:
            actual.update(zip(map(bytes,data['q_values']),map(bytes,data['q_visits'])))
    assert expected==actual
    first=BatchReader(snapshot,1,2,1);first.next()
    second=BatchReader(snapshot,1,2,999,first.state())
    try:
        for _ in range(3):
            a,b=first.next(),second.next()
            for key in a:np.testing.assert_array_equal(a[key],b[key])
    finally:first.close();second.close()
    for key,value in [('q_visits',-1),('q_values',32001)]:
        invalid=copy.deepcopy(raw);invalid[key][0,0]=value
        with pytest.raises(ValueError,match='Q'):validate_raw(invalid)
    invalid=copy.deepcopy(raw);invalid['q_values'][0,1]=1
    with pytest.raises(ValueError,match='Q'):validate_raw(invalid)


def test_q_configuration_and_head_contract(config):
    for architecture in ('plain','nbt'):
        c=copy.deepcopy(config);c['network']['predict_q_values']=True;c['network']['architecture']=architecture
        if architecture=='plain':c['network'].update(channels=128,blocks=10)
        with pytest.raises(ValueError,match='predict_q_values'):validate(c)
    c=copy.deepcopy(config);c['network'].update(architecture='transformer',predict_q_values=True,channels=192,blocks=5)
    validate(c);model=make_network(c)
    assert model.model_version==17 and model.policy_head.out.out_channels==7


def test_native_writer_independently_quantizes_repeats_and_caps(tmp_path):
    import subprocess
    from etazero.data import read_raw
    subprocess.run([str(ROOT/'build/record_contract_test'),str(tmp_path/'raw'),'q','fraction'],check=True)
    data=read_raw(next((tmp_path/'raw').glob('*.npz')))
    assert data['q_visits'][0,0]==32000
    assert set(data['q_values'][:64,0])=={-8001,-8000}
    assert data['q_values'].shape==(72,36)
    action=int(data['side_visits'][0].argmax())
    assert data['side_q_visits'][0,action]==32000
    assert set(data['side_q_values'][:,action])=={-8001,-8000}
    assert data['side_q_values'].shape==(64,36)
    view=training_view(data)
    assert not view['full_game_weight'][-64:].any()
    np.testing.assert_array_equal(view['q_values'][-64:],data['side_q_values']/np.float32(32000))
