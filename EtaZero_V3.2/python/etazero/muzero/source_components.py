"""MuZero components built from the pinned KataGo convolutional blocks.

Component widths/depths and latent projection are MuZero dimensions. Blocks,
normalization, initialization, pooling and parameter roles use source classes.
"""
import math
import torch
from torch import nn
from .._vendor.katago import model_pytorch as source
from ..katago_configs import resolve
from ..katago_network import XiangqiPolicyHead,KataGoNet
from ..config import network_widths
from ..games import SPECS


def component_config(channels,blocks,architecture,predict_q=False):
    config=resolve('b5c192nbt-fson-mish' if architecture=='nbt' else 'b10c128-fson-mish')
    widths=network_widths(channels,'nbt')
    config.update(trunk_num_channels=channels,mid_num_channels=widths['mid'] if architecture=='nbt' else channels,
                  gpool_num_channels=widths['gpool'],p1_num_channels=widths['policy'],g1_num_channels=widths['policy'],
                  v1_num_channels=widths['value'],v2_size=widths['value_hidden'],sbv2_num_channels=widths['value_hidden'],
                  version=17 if predict_q else 15,predict_q_values=predict_q)
    kind='bottlenest2' if architecture=='nbt' else 'regular'
    config['block_kind']=[[f'rconv{i+1}',kind+('gpool' if i%2 else '')] for i in range(blocks)]
    return config


class ZeroGlobal(nn.Module):
    """Format-required zero affine term for latent-only, non-global inputs."""
    def __init__(self,width):
        super().__init__();self.register_buffer('weight',torch.zeros(width,1));self.bias=None
        self.in_features=1;self.out_features=width


class Body(source.Model):
    def __init__(self,config,canvas,inputs,globals,latent_channels=0,kernel=3):
        super().__init__(config,canvas)
        self.conv_spatial=nn.Conv2d(inputs,self.c_trunk,kernel,padding=kernel//2,bias=False)
        self.linear_global=nn.Linear(max(1,globals),self.c_trunk,bias=False)
        self.initialize()
        self.set_brenorm_params(.001,1.,0.)
        del self.policy_head,self.value_head
        self.has_globals=bool(globals)
        if not globals:self.linear_global=ZeroGlobal(self.c_trunk)
        self.latent_channels=latent_channels
        if latent_channels:
            self.latent_projection=nn.Conv2d(self.c_trunk,latent_channels,1,bias=False)
            source.init_weights(self.latent_projection.weight,'identity',scale=1.)
        self.config['initial_conv_1x1']=kernel==1

    def forward(self,inputs,globals=None):
        mask=inputs[:,:1].float();count=mask.sum((2,3),keepdim=True);total=mask.sum()
        x=self.conv_spatial(inputs*mask)
        if self.has_globals:x=x+self.linear_global(globals).unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:x=x+block(x,mask=mask,mask_sum_hw=count,mask_sum=total,extra_outputs=None)
        x=self.act_trunkfinal(self.norm_trunkfinal(x,mask=mask,mask_sum_hw=count,mask_sum=total))
        return self.latent_projection(x) if self.latent_channels else x

    def roles(self,groups,prediction=False):
        groups['input'].append(self.conv_spatial.weight)
        if self.has_globals:groups['input'].append(self.linear_global.weight)
        for block in self.blocks:block.add_reg_dict(groups)
        local={k:[] for k in groups};self.norm_trunkfinal.add_reg_dict(local)
        for key,params in local.items():
            role=key if prediction else {'output':'normal_gamma','output_noreg':'noreg'}.get(key,key)
            groups[role].extend(params)
        if self.latent_channels:groups['normal'].append(self.latent_projection.weight)


class Prediction(nn.Module):
    def __init__(self,c):
        super().__init__();self.game=c.game;self.canvas=c.canvas;self.predict_q_values=c.predict_q_values
        config=component_config(c.prediction_channels,c.prediction_blocks,c.architecture,c.predict_q_values)
        self.body=Body(config,c.canvas,c.latent_channels+1,0,kernel=1)
        self.policy_head=(XiangqiPolicyHead(self.body,c.canvas) if c.game=='xiangqi' else
            source.PolicyHead(self.body.c_trunk,self.body.c_p1,self.body.c_g1,config,self.body.activation))
        self.value_head=source.ValueHead(self.body.c_trunk,self.body.c_v1,self.body.c_v2,self.body.c_sv2,
                                       self.body.num_scorebeliefs,config,self.body.activation,c.canvas)
        hidden=self.value_head.linear_valuehead.in_features
        self.value_head.linear_valuehead=nn.Linear(hidden,4)
        self.value_head.linear_miscvaluehead=nn.Linear(hidden,12)
        self.value_head.linear_moremiscvaluehead=nn.Linear(hidden,9)
        self.policy_head.initialize();self.value_head.initialize()

    def source_outputs(self,hidden):
        mask=hidden[:,-1:];x=self.body(torch.cat((mask,hidden[:,:-1]),1))
        count=mask.sum((2,3),keepdim=True);total=mask.sum()
        with torch.autocast('cuda',enabled=False):
            x=x.float();mask=mask.float()
            policy=self.policy_head(x,mask=mask,mask_sum_hw=count,mask_sum=total,extra_outputs=None)
            value=self.value_head(x,mask=mask,mask_sum_hw=count,mask_sum=total,
                                  input_global=x.new_zeros((x.shape[0],1)),extra_outputs=None)
        return (policy,*value)

    def forward(self,hidden):
        return KataGoNet.select_outputs(self,self.source_outputs(hidden),hidden[:,-1:])

    def training_outputs(self,hidden):
        outputs=self.source_outputs(hidden)
        return (*KataGoNet.select_outputs(self,outputs,hidden[:,-1:]),outputs[6],outputs[2][:,3])

    def export_model(self):
        import copy
        body=copy.deepcopy(self.body)
        body.policy_head=copy.deepcopy(self.policy_head);body.value_head=copy.deepcopy(self.value_head)
        return body
