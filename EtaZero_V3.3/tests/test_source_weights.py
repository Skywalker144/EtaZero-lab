"""Public source masks reach each learner without coupling unrelated losses."""
import copy
import numpy as np
import pytest
import torch
from torch import nn

from etazero.data.targets import TRAIN_TARGETS
from etazero.data.source_data import encode_view, decode_view, encode_targets, decode_targets
from etazero.models.network import TrainingForward
from etazero.train.muzero.training import TrainingForward as MuZeroForward
from etazero.train.learner import forward_batch


class Predictions(nn.Module):
    game = 'gomoku'

    def __init__(self):
        super().__init__()
        self.outputs = nn.ParameterList(nn.Parameter(torch.zeros(shape)) for shape in
            ((2,7,25),(2,4),(2,3,4),(2,),(2,2,5,5),(2,)))

    def training_outputs(self, *inputs):
        return (tuple(self.outputs),)


class LatentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Predictions()

    @property
    def prediction(self):
        return self

    def representation(self, obs, globals):
        return obs

    def dynamics(self, hidden, actions):
        return hidden

    def training_outputs(self, hidden):
        return tuple(self.head.outputs)


def public_rows():
    policy=np.zeros((2,25),np.int16);policy[:,0]=100
    value=np.tile([1,0,0,0],(2,1)).astype(np.float32)
    view=dict(obs=np.full((2,5,4),255,np.uint8),globals=np.zeros((2,8),np.float32),
              policy=policy,opponent_policy=policy.copy(),value=value,
              td_value=np.repeat(value[:,None],3,axis=1),
              future_positions=np.ones((2,2,5,5),np.float32),variance_time=np.zeros(2,np.float32),
              q_values=np.full((2,25),.5,np.float32),q_visits=np.ones((2,25),np.float32))
    for key in ('training_weight','policy_weight','opponent_policy_weight','value_weight',
                'td_value_weight','value_aux_weight','future_position_weight'):
        view[key]=np.ones(2,np.float32)
    return encode_view(view)


def gradients(arrays, algorithm, disable):
    targets=TRAIN_TARGETS if algorithm!='muzero_main' else ('policy','value','training_weight','policy_weight','value_weight')
    arrays=copy.deepcopy(arrays)
    if algorithm!='alphazero':
        arrays.update({'unroll_'+key:val[:,None] for key,val in list(arrays.items())
                       if key not in ('binaryInputNCHWPacked','globalInputNC')})
        arrays.update(muzero_actions=np.zeros((2,1),np.int64),muzero_step_weights=np.ones((2,2),np.float32),
                      muzero_sequence_mask=np.ones((2,2),np.float32),muzero_absorbing=np.zeros((2,2),np.uint8))
    batch={key:torch.from_numpy(value.copy()).float() for key,value in decode_view(arrays,targets).items()}
    batch['obs']=torch.ones(2,5,5,5)
    if algorithm=='alphazero':
        model=Predictions();forward=TrainingForward(model,.6,disable)
    else:
        model=LatentModel()
        forward=MuZeroForward(model,dict(environment={'rules':'freestyle'},unroll={'steps':1,'hidden_gradient_scale':1},
            training={'soft_policy_weight_scale':.6,'disable_optimistic_policy':disable},
            muzero_training={'auxiliary_losses':algorithm!='muzero_main'}))
        batch['recurrent_indices']=torch.arange(2)
    parts,_=forward_batch(forward,batch)
    parts[0].backward()
    head=model if algorithm=='alphazero' else model.head
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in head.outputs]


@pytest.mark.parametrize('algorithm',['alphazero','muzero','muzero_main'])
@pytest.mark.parametrize('disable',[False,True])
@pytest.mark.parametrize('slot',[26,35,24,33,27,25])
@pytest.mark.parametrize('weight',[0.,.25])
def test_public_masks_scale_only_their_loss_gradients(algorithm,disable,slot,weight):
    arrays=public_rows()
    baseline=gradients(arrays,algorithm,disable)
    arrays['globalTargetsNC'][0,slot]=1-weight if slot in (24,35) else weight
    actual=gradients(arrays,algorithm,disable)
    expected=[x.clone() for x in baseline]
    if slot==26:expected[0][0,[0,2,4,5]]*=weight
    if slot==35:expected[1][0]*=weight
    if slot==24:expected[2][0]*=weight
    if slot==33:expected[4][0]*=weight
    if slot==27:
        expected[3][0]*=weight;expected[5][0]*=weight
        if not disable:expected[0][0,4:6]*=weight
    if slot==25:
        for x in expected:x[0]*=weight
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b)


@pytest.mark.parametrize('algorithm',['alphazero','muzero','muzero_main'])
def test_finished_game_metadata_does_not_gate_losses(algorithm):
    arrays=public_rows()
    baseline=gradients(arrays,algorithm,False)
    arrays['globalTargetsNC'][:,62]=1
    for a,b in zip(gradients(arrays,algorithm,False),baseline):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('unroll',[False,True])
def test_source_mask_encoding_is_lossless_for_root_and_unroll(unroll):
    arrays=public_rows()
    arrays['globalTargetsNC'][:,[24,25,26,27,28,33,35]]=[.2,.3,.4,.5,.6,.7,.8]
    view=decode_view(arrays,TRAIN_TARGETS)
    if unroll:
        view={key:np.stack((value,value),axis=1) for key,value in view.items() if key in TRAIN_TARGETS}
        for key in ('policy_weight','value_weight','td_value_weight','future_position_weight','value_aux_weight'):
            view[key][:,1]=1-view[key][:,0]
    actual=encode_targets(view)
    decoded=decode_targets(actual,TRAIN_TARGETS)
    for key in TRAIN_TARGETS:np.testing.assert_allclose(decoded[key],view[key],atol=1e-7)
