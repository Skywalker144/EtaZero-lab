"""MuZero NBT/ResNet parameter roles for the shared normalized optimizer policy.

Representation/dynamics normalization is internal to the latent model, so its
gamma/bias use normal_gamma/noreg. Only prediction normalization and the heads
use output roles. No AlphaZero model layout or parameter traversal is changed.
"""
from etazero.train.optimization import group_settings, optimizer_from_groups


def parameter_groups(model):
    groups={name:[] for name in ('input','input_noreg','normal','normal_attn','normal_gab','gab_mlp','tab_module','normal_gamma','noreg','output','output_noreg')}
    model.representation.body.roles(groups)
    model.dynamics.body.roles(groups)
    model.prediction.body.roles(groups,prediction=True)
    model.prediction.policy_head.add_reg_dict(groups)
    model.prediction.value_head.add_reg_dict(groups)
    ids=[id(p) for params in groups.values() for p in params]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters()}:
        raise ValueError('MuZero source roles must contain every parameter exactly once')
    return [{'group_name':name,'params':params} for name,params in groups.items() if params]


def optimizer_for(model, config):
    plain = config.get('muzero_training', {})
    if not plain.get('katago_optimizer', True):
        from etazero.train.plain_optimization import optimizer_for as plain_optimizer
        kind = config['optimizer']['kind']
        return plain_optimizer(model, kind, plain, parameter_groups(model) if kind in ('muon','normuon','aurora') else None, config['optimizer'])
    groups = parameter_groups(model)
    options = config['optimizer']
    for group in groups:
        group['lr'], group['weight_decay'] = group_settings(
            group['group_name'], options, config['training']['batch_size'], 0, {}, {}, model.norm_kind, config.get('lr_schedule'))
    return optimizer_from_groups(groups, options['kind'], options)
