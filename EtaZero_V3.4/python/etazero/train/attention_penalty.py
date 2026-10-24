"""Training-only attention-bound hinge; inference outputs and losses stay separate."""


def configure(model, options):
    cap = options.get('cap', -1)
    model.has_attention_penalty = cap >= 0
    if cap < 0: return
    if hasattr(model, 'network'):
        from etazero.models.katago.model_pytorch import TransformerAttentionBlock
        if not any(isinstance(module, TransformerAttentionBlock) for module in model.network.modules()):
            raise ValueError('Attention penalty requires attention layers')
        model.network.attn_logit_penalty_cap = cap
        model.network.attn_logit_penalty_batch_frac = options['batch_fraction']
    else:
        raise ValueError('Attention penalty requires attention layers')


def metrics(model, cap):
    penalty = model.network.attn_logit_penalty_per_sample.mean()
    bound = model.network.attn_logit_ub_batch_max
    # The graph is returned to the caller; don't retain it across deepcopy,
    # checkpoint publication or an unrelated inference forward.
    model.network.attn_logit_penalty_per_sample = None
    model.network.attn_logit_ub_batch_max = None
    return penalty, bound
