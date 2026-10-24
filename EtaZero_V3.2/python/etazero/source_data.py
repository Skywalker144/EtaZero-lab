"""Pinned trainingwrite.cpp fields, with explicit draw and MuZero extensions.

Global W/L/NR slots retain KataGo's meaning; draws live in drawTargetsNC,
never split into half wins/losses. Go score/ownership/seki losses are absent.
"""
import numpy as np

DTYPES = dict(binaryInputNCHWPacked='uint8', globalInputNC='float32',
              policyTargetsNCMove='int16', globalTargetsNC='float32',
              valueTargetsNCHW='int8', qValueTargetsNCMove='int16', drawTargetsNC='float32')


def encode_targets(view):
    """Pack the common supervised slots; also used for MuZero latent steps."""
    shape=view['value'].shape[:-1]
    area=view['policy'].shape[-1]
    gt=np.zeros((*shape,80),np.float32)
    draws=np.zeros((*shape,5),np.float32)
    def value(slot,v):
        gt[...,4*slot:4*slot+3]=v[...,[0,2,3]]
        draws[...,slot]=v[...,1]
    value(0,view['value'])
    if 'td_value' in view:
        for i in range(3):value(i+1,view['td_value'][...,i,:])
    gt[...,22]=view.get('variance_time',0)
    gt[...,25]=view.get('training_weight',1)
    gt[...,26]=1
    gt[...,28]=view.get('opponent_policy_weight',0)
    gt[...,62]=view.get('full_game_weight',0)
    gt[...,63]=3
    policy=np.stack((view['policy'],view.get('opponent_policy',np.ones_like(view['policy']))),-2).astype(np.int16)
    result=dict(policyTargetsNCMove=policy,globalTargetsNC=gt,drawTargetsNC=draws)
    if 'future_positions' in view:
        future=view['future_positions']
        spatial=np.zeros((*shape,5,*future.shape[-2:]),np.int8)
        spatial[...,2:4,:,:]=future
        result['valueTargetsNCHW']=spatial
        gt[...,33]=1
    if 'q_values' in view:
        q=np.zeros((*shape,3,area),np.int16)
        # Targets were already independently stochastically quantized by writer.
        q[...,0,:]=np.rint(view['q_values']*32000).astype(np.int16)
        q[...,2,:]=view['q_visits']
        result['qValueTargetsNCMove']=q
    return result


def encode_view(view):
    result=dict(binaryInputNCHWPacked=view['obs'],globalInputNC=view['globals'])
    result.update(encode_targets(view))
    return result


def decode_targets(arrays,targets):
    gt=arrays['globalTargetsNC'];draws=arrays['drawTargetsNC'];policy=arrays['policyTargetsNCMove']
    def value(slot):
        return np.stack((gt[...,slot*4],draws[...,slot],gt[...,slot*4+1],gt[...,slot*4+2]),-1)
    result={'policy':policy[...,0,:], 'value':value(0)}
    if 'opponent_policy' in targets:
        result.update(opponent_policy=policy[...,1,:],opponent_policy_weight=gt[...,28])
    if 'td_value' in targets:
        result.update(td_value=np.stack([value(i) for i in (1,2,3)],-2),full_game_weight=gt[...,62])
    if 'training_weight' in targets:result['training_weight']=gt[...,25]
    if 'future_positions' in targets:
        result.update(future_positions=arrays['valueTargetsNCHW'][...,2:4,:,:].astype(np.float32),variance_time=gt[...,22])
    if 'q_values' in targets:
        q=arrays['qValueTargetsNCMove']
        result.update(q_values=q[...,0,:].astype(np.float32)/32000,q_visits=q[...,2,:].astype(np.float32))
    return result


def decode_view(arrays, targets):
    result=dict(obs=arrays['binaryInputNCHWPacked'],globals=arrays['globalInputNC'])
    result.update(decode_targets(arrays,targets))
    if 'unroll_globalTargetsNC' in arrays:
        unroll=decode_targets({k.removeprefix('unroll_'):v for k,v in arrays.items() if k.startswith('unroll_')},targets)
        for key in targets:result[key]=np.concatenate((result[key][:,None],unroll[key]),axis=1)
        for key in ('actions','step_weights','sequence_mask','absorbing'):result[key]=arrays['muzero_'+key]
    return result


def source_view(raw, targets):
    """Select native writer fields. Trajectories are only needed for MZ unrolls."""
    from .data import metadata,training_view
    result={key:raw[key] for key in DTYPES}
    if metadata(raw).get('algorithm')=='muzero':
        sequence=training_view(raw,targets)
        future=encode_targets({key:sequence[key][:,1:] for key in targets})
        result.update({'unroll_'+key:value for key,value in future.items()})
        for key in ('actions','step_weights','sequence_mask','absorbing'):result['muzero_'+key]=sequence[key]
    return result


def validate(arrays,rows,canvas,spec):
    expected=dict(binaryInputNCHWPacked=(rows,spec.planes,(canvas*canvas+7)//8),
                  globalInputNC=(rows,spec.globals),policyTargetsNCMove=(rows,2,spec.actions(canvas)),
                  globalTargetsNC=(rows,80),drawTargetsNC=(rows,5),valueTargetsNCHW=(rows,5,canvas,canvas),
                  qValueTargetsNCMove=(rows,3,spec.actions(canvas)))
    for key,shape in expected.items():
        if key not in arrays or arrays[key].shape!=shape or arrays[key].dtype!=np.dtype(DTYPES[key]):
            raise ValueError(f'Invalid source training field {key}: expected {shape}/{DTYPES[key]}')


def source_layout(arrays):
    """Validate a public AZ file by its tensor layout, without run bookkeeping."""
    from .games import SPECS
    spatial=arrays['valueTargetsNCHW'];packed=arrays['binaryInputNCHWPacked']
    if spatial.ndim!=4 or spatial.shape[-1]!=spatial.shape[-2] or packed.ndim!=3:
        raise ValueError('Invalid source input/spatial shape')
    canvas=spatial.shape[-1]
    for spec in SPECS.values():
        if (packed.shape[1]==spec.planes and arrays['globalInputNC'].shape==(len(packed),spec.globals)
                and arrays['policyTargetsNCMove'].shape==(len(packed),2,spec.actions(canvas))):
            return len(packed),canvas,spec
    raise ValueError('Unsupported source input/action layout')
