"""Training-only attention-bound hinge; inference outputs and losses stay separate."""
import torch


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
        from etazero.models.transformer import Attention
        modules = [module for module in model.modules() if isinstance(module, Attention)]
        if not modules: raise ValueError('Attention penalty requires attention layers')
        for module in modules:
            module.penalty_fraction = options['batch_fraction']


def metrics(model, cap):
    if hasattr(model, 'network'):
        penalty = model.network.attn_logit_penalty_per_sample.mean()
        bound = model.network.attn_logit_ub_batch_max
        # The graph is returned to the caller; don't retain it across deepcopy,
        # checkpoint publication or an unrelated inference forward.
        model.network.attn_logit_penalty_per_sample = None
        model.network.attn_logit_ub_batch_max = None
    else:
        from etazero.models.transformer import Attention
        modules = [module for module in model.modules() if isinstance(module, Attention)]
        bounds = torch.stack([module.logit_bound for module in modules])
        penalty = torch.relu(bounds - cap).mean(2).sum(0).mean()
        bound = bounds.detach().amax()
        for module in modules:
            module.logit_bound = torch.empty(0)
    return penalty, bound
