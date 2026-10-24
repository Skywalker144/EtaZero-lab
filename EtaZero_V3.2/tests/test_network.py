"""Named source network routing, parameter coverage and mask/pooling semantics."""
import math
import pytest
import torch
from config_samples import CONFIGS
from etazero.config import load_config,validate
from etazero.network import make_network
from etazero.katago_configs import resolve
from etazero._vendor.katago import model_pytorch as source
from etazero.optimization import parameter_groups
from etazero.symmetry import apply_symmetry


@pytest.mark.parametrize('alias,name,channels,blocks',[
    ('nbt','b5c192nbt-fson-mish',192,5),
    ('plain','b10c128-fson-mish',128,10),
    ('transformer','b5c192h3nbttfrs',192,5)])
def test_aliases_select_source_presets_and_parameter_roles(alias,name,channels,blocks):
    c=load_config(CONFIGS/'smoke_test')
    c['network'].update(architecture=alias,channels=channels,blocks=blocks)
    validate(c);model=make_network(c)
    assert isinstance(model.network,source.Model)
    expected=resolve(name)
    for key in ('block_kind','trunk_num_channels','mid_num_channels','gpool_num_channels','norm_kind','activation'):
        assert model.source_config.get(key)==expected.get(key)
    groups=parameter_groups(model)
    roles={id(p):g['group_name'] for g in groups for p in g['params']}
    assert sum(len(g['params']) for g in groups)==len(roles)==len(list(model.parameters()))
    assert roles[id(model.network.conv_spatial.weight)]=='input'
    assert roles[id(model.network.value_head.linear_valuehead.weight)]=='output'
    assert model.network.value_head.linear_valuehead.out_features==4
    assert model.network.value_head.linear_miscvaluehead.out_features==12
    assert model.network.value_head.linear_moremiscvaluehead.out_features==9


@pytest.mark.parametrize('architecture,channels,blocks',[
    ('nbt',24,1),('nbt',256,5),('plain',192,10),('plain',128,5),
    ('transformer',192,4),('transformer',128,5),('b2c16-fson-mish',24,2),('unknown',128,10)])
def test_reject_independent_rescaling_of_source_presets(architecture,channels,blocks):
    c=load_config(CONFIGS/'smoke_test');c['network'].update(architecture=architecture,channels=channels,blocks=blocks)
    with pytest.raises(ValueError):validate(c)
    with pytest.raises(ValueError):make_network(c)


def test_source_pooling_manual_values_and_masked_max_gradient():
    # Source pooling consumes already masked activation values > -1 (Mish/SiLU).
    mask=torch.tensor([[[[1.,1.],[0.,0.]]],[[[1.,1.],[1.,1.]]]])
    x=torch.tensor([[[[-.2,-.4],[0.,0.]]],[[[-.1,-.3],[-.5,-.7]]]])
    offsets=[math.sqrt(2)-14,2-14]
    expected=torch.tensor([[-.3,-.3*offsets[0]/10,-.2],[-.4,-.4*offsets[1]/10,-.1]])
    value_expected=torch.tensor([[-.3,-.3*offsets[0]/10,-.3*(offsets[0]**2/100-.1)],
                                 [-.4,-.4*offsets[1]/10,-.4*(offsets[1]**2/100-.1)]])
    for symmetry in range(8):
        xx=apply_symmetry(x,symmetry).contiguous();mm=apply_symmetry(mask,symmetry).contiguous();count=mm.sum((2,3),keepdim=True)
        torch.testing.assert_close(source.KataGPool()(xx,mm,count).flatten(1),expected)
        torch.testing.assert_close(source.KataValueHeadGPool()(xx,mm,count).flatten(1),value_expected)
    tied=torch.tensor([[[[.2,.2,0.]]]],requires_grad=True);mm=torch.tensor([[[[1.,1.,0.]]]])
    source.KataGPool()(tied,mm,mm.sum((2,3),keepdim=True))[0,2].backward()
    torch.testing.assert_close(tied.grad,torch.tensor([[[[1.,0.,0.]]]]),rtol=0,atol=0)


@pytest.mark.parametrize('name,channels,blocks',[('b2c16-fson-mish',16,2),('b1c6nbt',6,1)])
def test_source_trunk_padding_invariance(name,channels,blocks):
    torch.set_num_threads(1)
    c=load_config(CONFIGS/'smoke_test');c['network'].update(architecture=name,channels=channels,blocks=blocks)
    model=make_network(c).eval()
    obs=torch.zeros(2,5,6,6);obs[:,0,:5,:5]=1;obs[:,1,2,2]=1;obs[:,2,1,2]=1
    obs[1,1:,5,:]=1000;obs[1,1:,:,5]=-1000
    with torch.no_grad():p,v=model(obs,torch.zeros(2,8))
    torch.testing.assert_close(p[0],p[1],rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(v[0],v[1],rtol=1e-6,atol=1e-7)
    assert p[:,:,~obs[0,0].flatten().bool()].eq(0).all()
