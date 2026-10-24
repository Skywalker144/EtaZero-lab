"""KataGo single-device Muon with auxiliary AdamW and strict state restoration.

The numerical kernels are vendored unchanged; provenance and licenses are in
reference_sources.json and THIRD_PARTY.md. Only trunk matrices use Muon.
"""
import torch

from ._vendor.muon import SingleDeviceMuonWithAuxAdam


def muon_suitable(group_name):
    if group_name in ('normal', 'normal_attn', 'normal_gab', 'gab_mlp', 'tab_module'):
        return True
    if group_name in ('input', 'input_noreg', 'normal_gamma', 'noreg', 'output', 'output_noreg'):
        return False
    raise ValueError(f'Unknown Muon parameter group: {group_name}')


class Muon(SingleDeviceMuonWithAuxAdam):
    def __init__(self, groups, options=None):
        options = options or {}
        for group in groups:
            group['use_muon'] = muon_suitable(group['group_name'])
            if group['use_muon'] and any(p.ndim not in (2, 4) for p in group['params']):
                raise ValueError('Muon trunk weights must be matrices or Conv2d filters')
        super().__init__(groups, adjust_lr_fn='match_rms_adamw',
                         use_normuon=options.get('kind') == 'normuon', use_aurora=options.get('kind') == 'aurora',
                         aurora_pp_iterations=options.get('aurora_pp_iterations', 2),
                         aurora_pp_beta=options.get('aurora_pp_beta', 0.5),
                         ns_steps=options.get('ns_steps', 5), use_polar_express=options.get('use_polar_express', False))

    def load_state_dict(self, state_dict):
        # The source drops mismatched legacy state with a warning. EtaZero
        # requires exact restoration and must fail instead of restarting moments.
        saved_groups = state_dict['param_groups']
        if len(saved_groups) != len(self.param_groups):
            raise ValueError('Muon optimizer checkpoint parameter groups differ')
        parameters = {}
        for saved, group in zip(saved_groups, self.param_groups):
            if (saved['group_name'] != group['group_name'] or saved['use_muon'] != group['use_muon']
                    or len(saved['params']) != len(group['params'])):
                raise ValueError('Muon optimizer checkpoint parameter groups differ')
            parameters.update(zip(saved['params'], group['params']))
        for key, state in state_dict['state'].items():
            if key not in parameters:
                raise ValueError(f'Muon optimizer checkpoint has unknown parameter {key}')
            for name, value in state.items():
                expected = (parameters[key].shape[0],) if name == 'normuon_v' else parameters[key].shape
                if isinstance(value, torch.Tensor) and value.shape != expected:
                    raise ValueError(f'Muon optimizer checkpoint {name} shape differs for parameter {key}')
        torch.optim.Optimizer.load_state_dict(self, state_dict)


def initialization_rms(model):
    with torch.no_grad():
        return {name: float(p.float().square().mean().sqrt()) for name, p in model.named_parameters()}


def configure_floors(model, optimizer, options, state=None):
    fraction = options.get('wd_floor_frac', 0)
    if not fraction: return {}
    initial = (getattr(model, 'initial_rms', None) or initialization_rms(model)) if state is None else state['initial_rms']
    optimizer.set_weight_decay_floor_norms({p: fraction * initial[name] * p[0].numel()**0.5
                                          for name, p in model.named_parameters() if p.ndim >= 2})
    return initial
