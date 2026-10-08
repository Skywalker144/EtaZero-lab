from etazero.schema import GLOBALS
"""NBT structure, masking and global-context semantics."""
from config_samples import CONFIGS
import math

import pytest
import torch

from etazero.config import ROOT, load_config, network_widths, validate
from etazero.network import GlobalPool, ValuePool, MaskedBatchNorm, PolicyHead, make_network
from etazero.optimization import parameter_groups
from etazero.symmetry import apply_symmetry


@pytest.mark.parametrize('channels,mid,head,hidden', [(24,12,8,16), (64,32,16,32),
                                                   (192,96,32,80), (256,128,48,112), (384,192,64,160)])
def test_scaled_nbt_and_optimizer_coverage(channels,mid,head,hidden):
    c=load_config(CONFIGS / 'smoke_test')
    c['network'].update(channels=channels,blocks=5)
    model=make_network(c)
    assert network_widths(channels)==dict(mid=mid,gpool=head,policy=head,value=head,value_hidden=hidden)
    assert model.policy_head.local.weight.shape==(head,channels,1,1)
    assert model.value_head.hidden.weight.shape==(hidden,3*head)
    assert [b.inner[0].pre.has_global_pool for b in model.blocks]==[False,True,False,True,False]
    assert len([m for m in model.modules() if isinstance(m,MaskedBatchNorm)])==1
    assert model.trunk_norm.scale==pytest.approx(1/math.sqrt(6))
    for index,block in enumerate(model.blocks):
        assert block.pre.conv.weight.shape==(mid,channels,1,1)
        assert len(block.inner)==2
        assert block.pre.norm.scale==pytest.approx(1/math.sqrt(index+1))
        assert block.inner[1].pre.norm.scale==pytest.approx(1/math.sqrt(2))
        assert block.post.norm.scale==pytest.approx(1/math.sqrt(3))
    groups=parameter_groups(model)
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))
    # A width change must not omit newly introduced global branch parameters.
    normal={id(p) for g in groups if g['group_name']=='normal' for p in g['params']}
    assert id(model.blocks[1].inner[0].pre.conv.linear.weight) in normal


@pytest.mark.parametrize('channels',[8,22,25,193])
def test_reject_unsupported_nbt_width(channels):
    c=load_config(CONFIGS / 'smoke_test');c['network']['channels']=channels
    with pytest.raises(ValueError,match='even and >= 24'):
        validate(c)


def test_plain_source_preset_and_optimizer_roles():
    c=load_config(CONFIGS / 'smoke_test')
    c['network'].update(architecture='plain',channels=128,blocks=10)
    validate(c)
    model=make_network(c)
    assert [b.pre.has_global_pool for b in model.blocks]==[False]*4+[True,False,False,True,False,False]
    assert model.policy_head.out.weight.shape==(6,32,1,1)
    assert model.value_head.hidden.weight.shape==(80,96)
    assert len([m for m in model.modules() if isinstance(m,MaskedBatchNorm)])==1
    assert model.trunk_norm.scale==pytest.approx(1/math.sqrt(11))
    for i,block in enumerate(model.blocks):
        assert block.pre.norm.scale==pytest.approx(1/math.sqrt(i+1))
        assert block.post.norm.scale==1
        local=96 if i in (4,7) else 128
        assert block.post.conv.weight.shape==(128,local,3,3)
    roles={id(p):g['group_name'] for g in parameter_groups(model) for p in g['params']}
    assert len(roles)==len(list(model.parameters()))
    assert roles[id(model.blocks[4].pre.conv.linear.weight)]=='normal'
    assert roles[id(model.blocks[4].pre.conv.norm_global.weight)]=='normal_gamma'
    assert roles[id(model.trunk_norm.weight)]=='output'


@pytest.mark.parametrize('architecture,channels,blocks', [('plain',192,10),('plain',128,5),('transformer',192,4),('transformer',128,5),('unknown',128,10)])
def test_reject_unimplemented_or_nonpreset_architecture(architecture,channels,blocks):
    c=load_config(CONFIGS / 'smoke_test')
    c['network'].update(architecture=architecture,channels=channels,blocks=blocks)
    with pytest.raises(ValueError):
        validate(c)
    with pytest.raises(ValueError):
        make_network(c)


def test_transformer_bare_source_structure_initialization_and_roles():
    from etazero.network import BiasMask,FixedScaleMask
    from etazero.transformer import Attention,SwiGLU
    c=load_config(CONFIGS / 'smoke_test')
    c['network'].update(architecture='transformer',channels=192,blocks=5)
    validate(c); model=make_network(c)
    assert model.model_version==17 and model.norm_kind=='fixup'
    assert isinstance(model.act,torch.nn.ReLU) and isinstance(model.trunk_norm,BiasMask)
    assert not any(isinstance(m,MaskedBatchNorm) for m in model.modules())
    assert model.policy_head.out.weight.shape==(6,32,1,1)
    assert model.value_head.hidden.weight.shape==(64,96)
    roles={id(p):g['group_name'] for g in parameter_groups(model) for p in g['params']}
    for block in model.blocks:
        assert isinstance(block.pre.norm,BiasMask)
        assert isinstance(block.post.norm,FixedScaleMask)
        assert block.post.conv.weight.count_nonzero()==0
        assert [type(m) for m in block.inner]==[Attention,SwiGLU,Attention,SwiGLU]
        assert roles[id(block.inner[0].q_proj.weight)]=='normal_attn'
        assert roles[id(block.inner[1].input.weight)]=='normal'
        assert roles[id(block.inner[0].norm.weight)]=='noreg'
        assert roles[id(block.post.norm.weight)]=='normal_gamma'
        x=torch.randn(2,192,6,6);mask=torch.ones(2,1,6,6)
        torch.testing.assert_close(block(x,mask),x,rtol=0,atol=0)
    assert len(roles)==len(list(model.parameters()))


def test_transformer_attention_excludes_keys_with_manual_uniform_attention():
    from etazero.transformer import Attention
    layer=Attention(2)
    with torch.no_grad():
        layer.q_proj.weight.zero_();layer.k_proj.weight.zero_()
        layer.v_proj.weight.copy_(torch.eye(96));layer.out_proj.weight.copy_(torch.eye(96))
    x=torch.zeros(1,96,2,2);x[0,0,0,0]=2;x[0,1,0,1]=4;x[:,:,1,:]=100000
    mask=torch.tensor([[[[1.,1.],[0.,0.]]]])
    actual=layer(x,mask)
    expected=torch.zeros_like(x)
    expected[:,0]=1/math.sqrt(4/96+1e-6)
    expected[:,1]=2/math.sqrt(16/96+1e-6)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    # Keys are excluded, while invalid queries may receive nonzero context; the
    # trunk-final mask prevents those query outputs from reaching a head.
    changed=x.clone();changed[:,:,1,:]=-123456
    torch.testing.assert_close(layer(changed,mask),actual,rtol=0,atol=0)


def test_global_and_value_pool_hand_calculated_negative_features():
    # Large positive padding must not affect mean or maximum; all real values
    # are negative, so replacing padding with zero would be an incorrect max.
    x=torch.tensor([[[[-2.,-4.],[1000.,1000.]]],[[[-1.,-3.],[-5.,-7.]]]])
    mask=torch.tensor([[[[1.,1.],[0.,0.]]],[[[1.,1.],[1.,1.]]]])
    offsets=[math.sqrt(2)-14,2-14]
    global_expected=torch.tensor([[-3.,-3.*offsets[0]/10,-2.],[-4.,-4.*offsets[1]/10,-1.]])
    value_expected=torch.tensor([[-3.,-3.*offsets[0]/10,-3.*(offsets[0]**2/100-.1)],
                                 [-4.,-4.*offsets[1]/10,-4.*(offsets[1]**2/100-.1)]])
    for symmetry in range(8):
        torch.testing.assert_close(GlobalPool()(apply_symmetry(x,symmetry),apply_symmetry(mask,symmetry)),global_expected)
        torch.testing.assert_close(ValuePool()(apply_symmetry(x,symmetry),apply_symmetry(mask,symmetry)),value_expected)


def test_policy_global_context_changes_relative_logits():
    head=PolicyHead(2,4)
    with torch.no_grad():
        for p in head.parameters(): p.zero_()
        head.local.weight[0,0]=1
        head.global_conv.weight[0,1]=1
        head.linear.weight[0,0]=1  # mean of global channel zero
        head.out.weight[0,0]=1
    # Identical local inputs. Context in a separate channel changes the relative
    # logits via Mish, rather than becoming a softmax-cancelled constant.
    x=torch.zeros(2,2,1,3);x[:,0,0]=torch.tensor([-1.,0.,1.]);x[1,1]=2
    logits=head(x,torch.ones(2,1,1,3))[:,0]
    shifts=logits[1]-logits[0]
    assert (shifts.max()-shifts.min()).item()>0.1
    assert not torch.allclose(logits.softmax(1)[0],logits.softmax(1)[1])
    logits.sum().backward()
    assert head.linear.weight.grad[0,0].abs()>0


def test_global_max_gradient_uses_first_tie_and_excludes_padding():
    x=torch.tensor([[[[2.,2.,1000.]]]],requires_grad=True)
    pool=GlobalPool()(x,torch.tensor([[[[1.,1.,0.]]]]))
    pool[0,2].backward()
    torch.testing.assert_close(x.grad,torch.tensor([[[[1.,0.,0.]]]]),rtol=0,atol=0)


def test_nbt_skip_paths_and_padding_invariance():
    c=load_config(CONFIGS / 'smoke_test');c['network']['blocks']=2
    model=make_network(c).eval()
    block=model.blocks[0]
    x=torch.randn(2,24,6,6);mask=torch.ones(2,1,6,6)
    with torch.no_grad():
        # Zeroing the outer projection makes the entire nested branch identity.
        block.post.conv.weight.zero_()
        torch.testing.assert_close(block(x,mask),x,rtol=0,atol=0)
        inner=model.blocks[1].inner[0]
        inner.post.conv.weight.zero_()
        inner_x=torch.randn(2,12,6,6)
        torch.testing.assert_close(inner(inner_x,mask),inner_x,rtol=0,atol=0)
        obs=torch.zeros(2,5,6,6);obs[:,0,:5,:5]=1
        obs[:,1,2,2]=1;obs[:,2,1,2]=1
        obs[1,:,5,:]=1000;obs[1,:,:,5]=1000;obs[1,0,5,:]=0;obs[1,0,:,5]=0
        p,v=model(obs,torch.zeros(2,len(GLOBALS)))
        torch.testing.assert_close(p[0],p[1],rtol=0,atol=0)
        torch.testing.assert_close(v[0],v[1],rtol=0,atol=0)
        assert p[:, :, ~obs[0,0].flatten().bool()].eq(0).all()
