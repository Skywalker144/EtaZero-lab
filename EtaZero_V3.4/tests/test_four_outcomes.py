"""Distinct rule outcomes survive targets, perspective changes and value learning."""
import numpy as np
import pytest
import torch
from etazero.outcomes import Reason, terminal_wdl
from etazero.data.targets import trajectory_td_targets, training_view
from etazero.models.network import base_losses, losses
from test_python import winning_record


def test_rule_draw_and_no_result_are_distinct_one_hot_classes():
    winner = np.array([1, 0, -1, 0])
    reason = np.array([Reason.FIVE, Reason.DRAW, Reason.FIVE, Reason.NO_RESULT])
    np.testing.assert_array_equal(terminal_wdl(winner, reason), np.eye(4))
    np.testing.assert_array_equal(terminal_wdl(winner, reason, -1), np.eye(4)[[2,1,0,3]])


def test_td_preserves_neutral_outcome_identity_across_players():
    players = np.array([1, -1, 1, -1])
    for index in (1, 3):
        target = np.eye(4)[index]
        actual = trajectory_td_targets(np.tile(target, (4, 1)), players, target, 25)
        np.testing.assert_array_equal(actual, np.broadcast_to(target, (4, 3, 4)))


@pytest.mark.parametrize('algorithm',['alphazero','muzero'])
def test_no_result_main_trajectory_reaches_value_and_td_training(algorithm):
    raw, meta = winning_record()
    if algorithm=='muzero':
        import json
        meta.update(algorithm='muzero',unroll_steps=12)
        raw['metadata']=np.frombuffer(json.dumps(meta).encode(),np.uint8)
        raw['trajectory_policy']=raw['policies'].copy()
        raw['trajectory_q_values']=raw['q_values'].copy()
        raw['trajectory_q_visits']=raw['q_visits'].copy()
    raw['winners'][0] = 0
    raw['reasons'][0] = Reason.NO_RESULT
    raw['search_wdl'][:] = [0, 0, 0, 1]
    view = training_view(raw)
    assert len(view['value']) == 9
    np.testing.assert_array_equal(view['value'], np.broadcast_to([0,0,0,1],view['value'].shape))
    np.testing.assert_array_equal(view['td_value'], np.broadcast_to([0,0,0,1],view['td_value'].shape))
    assert not view['value_aux_weight'].any()
    assert (view['future_position_weight']==1).all()
    if algorithm=='muzero':assert view['absorbing'].any()


@pytest.mark.parametrize('disable_optimistic', [False, True])
def test_nr_auxiliary_gates_preserve_draw_td_future_and_disabled_optimism(disable_optimistic):
    # Complete win/draw/NR, then a side row and a reanalysis without outcome.
    target=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],
                         [.2,.1,.3,.4],[1.,0.,0.,0.]])
    gate=torch.tensor([1.,1.,0.,0.,0.]);n=len(gate)
    obs=torch.ones(n,5,2,2);policy=torch.zeros(n,4);policy[:,0]=1
    logits=torch.zeros(n,6,4,requires_grad=True)
    value=torch.zeros(n,4,requires_grad=True);td=torch.zeros(n,3,4,requires_grad=True)
    error=torch.zeros(n,requires_grad=True);variance=torch.zeros(n,requires_grad=True)
    future=torch.zeros(n,2,2,2,requires_grad=True)
    result=losses(logits,value,td,error,obs,policy,policy,torch.zeros(n),target,
                  target[:,None].expand(-1,3,-1),gate,8,disable_optimistic,
                  future_logits=future,raw_variance_time=variance,
                  future_target=torch.ones_like(future),variance_target=torch.zeros(n),
                  future_position_weight=torch.tensor([1.,1.,1.,0.,0.]))
    result[0].backward()
    assert error.grad[:2].abs().min()>0 and variance.grad[:2].abs().min()>0
    assert not error.grad[2:].any() and not variance.grad[2:].any()
    assert value.grad[2,3]<0 and (td.grad[2,:,3]<0).all()
    assert future.grad[2].abs().sum()>0 and not future.grad[3:].any()
    if disable_optimistic:
        assert (logits.grad[2:,4:6].abs().sum((1,2))>0).all()
    else:
        assert not logits.grad[2:,4:6].any()
        assert logits.grad[1,4:6].abs().sum()>0  # Ordinary draw keeps these targets.


def test_equal_search_utility_does_not_merge_loss_targets():
    targets = torch.tensor([[0.,1.,0.,0.], [0.,0.,0.,1.]])
    values = torch.zeros(2,4,requires_grad=True)
    obs = torch.ones(2,5,1,1)
    policy = torch.ones(2,1)
    loss = base_losses(torch.zeros(2,6,1), values, obs, policy, policy,
                       torch.zeros(2), targets, 8)[-1]
    loss.backward()
    expected = .72 * (torch.full_like(targets, .25)-targets)/2
    torch.testing.assert_close(values.grad, expected)
    assert values.grad[0,1] < 0 and values.grad[0,3] > 0
    assert values.grad[1,1] > 0 and values.grad[1,3] < 0


def test_only_go_activates_no_result_logit_in_all_value_heads():
    from etazero.models.katago_network import KataGoNet
    from etazero.games import SPECS
    for game in ('gomoku','hex','go','xiangqi'):
        n=10 if game=='xiangqi' else 6
        model=KataGoNet('b2c16-fson-mish',n,game=game).eval()
        obs=torch.zeros(2,SPECS[game].planes,n,n);obs[:,0]=1
        with torch.no_grad():_,value,td,_=model.forward_all(obs,torch.zeros(2,SPECS[game].globals))
        if game=='go':
            assert (value.softmax(-1)[:,-1]>0).all() and (td.softmax(-1)[:,:,-1]>0).all()
        else:
            assert (value.softmax(-1)[:,-1]==0).all() and (td.softmax(-1)[:,:,-1]==0).all()
