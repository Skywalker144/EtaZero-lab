"""Source KataGo architectures with the EtaZero game input/output contract.

All source heads remain present. Go-only outputs are not supervised by EtaZero.
Metadata encoders accept explicit Python inputs; the game pipeline uses the
neutral all-zero metadata vector because it has no SGF/player metadata.
"""
import copy
import math
import torch
from torch import nn
from .katago_configs import resolve, validate_dimensions
from .schema import PLANES, GLOBALS
from ._vendor.katago import model_pytorch as source
from .games import SPECS, policy_indices, XIANGQI_DELTAS


class XiangqiPolicyHead(source.PolicyHead):
    def __init__(self, model, canvas):
        super().__init__(model.c_trunk, model.c_p1, model.c_g1, model.config, model.activation)
        self.conv2p = nn.Conv2d(model.c_p1, self.num_policy_outputs*len(XIANGQI_DELTAS), 1, bias=False)
        self.register_buffer('action_indices', torch.tensor(policy_indices(canvas)), persistent=False)

    def forward(self, x, mask, mask_sum_hw, mask_sum, extra_outputs, block_shared_data=None):
        p = self.conv1p(x)
        g = self.actg(self.biasg(self.conv1g(x), mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum))
        g = self.gpool(g, mask=mask, mask_sum_hw=mask_sum_hw).flatten(1)
        p = p + self.linear_g(g).unsqueeze(-1).unsqueeze(-1)
        p = self.act2(self.bias2(p, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum))
        p = self.conv2p(p).reshape(x.shape[0], self.num_policy_outputs, -1)
        return p.index_select(2, self.action_indices)


class KataGoNet(nn.Module):
    def __init__(self, architecture, canvas, predict_q_values=False, channels=None, blocks=None, game='gomoku'):
        super().__init__()
        config = resolve(architecture)
        validate_dimensions(config)
        for key, given, actual in (('channels', channels, config['trunk_num_channels']),
                                   ('blocks', blocks, len(config['block_kind']))):
            if given is not None and given != actual:
                raise ValueError(f'{architecture} fixes network.{key}={actual}, got {given}')
        if predict_q_values and config['version'] < 17:
            raise ValueError('predict_q_values requires a KataGo v17 preset')
        config['predict_q_values'] = bool(predict_q_values)
        self.architecture, self.canvas = architecture, canvas
        self.game = game
        self.input_planes, self.global_features = SPECS[game].planes, SPECS[game].globals
        self.norm_kind, self.model_version = config['norm_kind'], config['version']
        self.predict_q_values = bool(predict_q_values)
        self.fp32_heads = True
        self.source_config = config
        self.head_loss_weights = ((0.2, 0.8) if config.get('trunk_normless', False)
                                  else (0.5, 0.5)) if config.get('has_intermediate_head', False) else (1.0,)
        self.network = source.Model(config, canvas)
        kernel = 1 if config['initial_conv_1x1'] else 3
        self.network.conv_spatial = nn.Conv2d(self.input_planes, config['trunk_num_channels'],
                                             kernel, padding=kernel//2, bias=False)
        self.network.linear_global = nn.Linear(self.global_features, config['trunk_num_channels'], bias=False)
        self.network.bin_input_shape = [self.input_planes, canvas, canvas]
        self.network.global_input_shape = [self.global_features]
        if game == 'xiangqi':
            self.network.policy_head = XiangqiPolicyHead(self.network, canvas)
            if self.network.has_intermediate_head:
                self.network.intermediate_policy_head = XiangqiPolicyHead(self.network, canvas)
        # Select the source's public NCHW/SDPA execution path, shared by eager,
        # compiled training and LibTorch export. Keep architecture and masks.
        self.network.transformer_seq_layout = False
        self.network.transformer_skip_redundant_masks = False
        self.network.use_flex_attention = False
        for module in self.network.modules():
            if isinstance(module, source.NormMask):
                module.skip_mask = False
        self.network.initialize()
        self.network.set_brenorm_params(0.001, 1.0, 0.0)

    def metadata(self) -> tuple[int, str]:
        return self.canvas, 'alphazero'

    def parameter_groups(self):
        groups = {}
        self.network.add_reg_dict(groups)
        ids = [id(p) for values in groups.values() for p in values]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters()}:
            raise ValueError('KataGo optimizer roles must contain every parameter exactly once')
        return [{'group_name': name, 'params': values} for name, values in groups.items() if values]

    def source_outputs(self, obs, globals, metadata=None):
        if self.network.metadata_encoder is not None and metadata is None:
            metadata = obs.new_zeros((obs.shape[0], 192))
        return self.network(obs * obs[:, 0:1], globals, input_meta=metadata)

    def select_outputs(self, outputs, mask):
        policy, value, misc, more = outputs[:4]
        count = 7 if self.predict_q_values else 6
        # Scalar selects avoid CPU index-tensor constants in traced CUDA Graphs.
        selected = policy[:, :count] if self.game in ('go', 'xiangqi') else policy[:, :count, :-1] * mask.flatten(2)
        return (selected,
                torch.stack((value[:, 0], value[:, 2], value[:, 1]), 1),
                torch.stack((torch.stack((misc[:, 4], misc[:, 6], misc[:, 5]), 1),
                             torch.stack((misc[:, 7], misc[:, 9], misc[:, 8]), 1),
                             torch.stack((more[:, 2], more[:, 4], more[:, 3]), 1)), 1),
                more[:, 0])

    def training_outputs(self, obs, globals):
        mask = obs[:, 0:1]
        return tuple(self.select_outputs(head, mask) for head in self.source_outputs(obs, globals))

    def forward_all(self, obs, globals):
        return self.select_outputs(self.source_outputs(obs, globals)[0], obs[:, 0:1])

    def forward(self, obs, globals):
        policy, value, _, _ = self.forward_all(obs, globals)
        return policy, value

    def prepare_compile(self):
        # Source configure_model_for_compile makes this same choice for one graph.
        self.network.disable_fused_attention_kernels()

    def prepare_brenorm(self, samples, options):
        if self.norm_kind not in ('brenorm', 'fixbrenorm'):
            return
        # Source starts at rmax=1,dmax=0. Omitted schedule keeps these initial
        # bounds; configured targets follow its exponential sample clock.
        if options is None:
            self.network.set_brenorm_params(0.001, 1.0, 0.0)
        else:
            fraction = -math.expm1(-samples / options['adjustment_scale'])
            self.network.set_brenorm_params(options['avg_momentum'],
                                           1 + fraction * (options['target_rmax'] - 1),
                                           fraction * options['target_dmax'])

    def scale_repvgg_gradients(self):
        if self.source_config.get('use_repvgg_learning_rate', False):
            for name, parameter in self.network.named_parameters():
                if ('normactconv' in name and '.conv.weight' in name and parameter.ndim == 4
                        and parameter.shape[2:] == (3, 3) and parameter.grad is not None):
                    parameter.grad[:, :, 1, 1].mul_(2.0)


class SearchOutputs(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, globals):
        policy, value, _, raw = self.model.forward_all(obs, globals)
        # Source policy masking promotes half logits to the FP32 mask dtype.
        # Restore the head dtype without changing any on-board logit value.
        return (policy[:, 0].type_as(value), value, policy[:, 5].type_as(value),
                0.5 * torch.nn.functional.softplus(raw.float() * 0.5))

    def raw(self, obs, globals):
        policy, value, _, _ = self.model.forward_all(obs, globals)
        return policy.type_as(value), value


class SerializedSearch(nn.Module):
    def __init__(self, graph, canvas, game):
        super().__init__()
        self.graph, self.canvas = graph, canvas
        self.game = game

    @torch.jit.export
    def game_name(self) -> str:
        return self.game

    @torch.jit.export
    def metadata(self) -> tuple[int, str]:
        return self.canvas, 'alphazero'

    @torch.jit.export
    def raw(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.graph.raw(obs, globals)

    def forward(self, obs: torch.Tensor, globals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.graph(obs, globals)


class ExportComplexDepthwiseConv2d(nn.Module):
    """Preassemble the source's two real kernels on the export copy.

    Runtime kernel concatenation followed by AMP casts can trigger the
    TorchScript TensorExpr OptimizeCat output-stride assertion. Conv2d weights
    avoid that rewrite and use the backend's ordinary FP16 weight preparation.
    The two convolutions and real/imaginary output arithmetic stay unchanged.
    """
    def __init__(self, module):
        super().__init__()
        self.c = module.c
        for name, weight in (
                ('conv_ri', torch.cat((module.real_kernel, module.imag_kernel), 0)),
                ('conv_neg_ir', torch.cat((-module.imag_kernel, module.real_kernel), 0))):
            # Allocate no random weights: exporting must not advance the learner RNG.
            conv = nn.Conv2d(2 * self.c, 2 * self.c, module.kernel_size,
                             padding=module.dilation * (module.kernel_size // 2),
                             dilation=module.dilation, groups=2 * self.c,
                             bias=False, device='meta')
            conv.weight = nn.Parameter(weight.detach(), requires_grad=False)
            setattr(self, name, conv)

    def forward(self, x):
        x_ri = torch.cat((x[..., :self.c, :, :], x[..., self.c:, :, :]), -3)
        conv1, conv2 = self.conv_ri(x_ri), self.conv_neg_ir(x_ri)
        real = conv1[..., :self.c, :, :] - conv1[..., self.c:, :, :]
        imag = conv2[..., self.c:, :, :] - conv2[..., :self.c, :, :]
        return torch.cat((real, imag), -3)


def inference_network(model):
    if model.training:
        raise ValueError('KataGo inference export requires evaluation mode')
    result = copy.deepcopy(model)
    result.prepare_compile()
    result.network.has_intermediate_head = False
    result.network.use_flex_attention = False
    for module in result.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, source.ComplexDepthwiseConv2d):
                setattr(module, name, ExportComplexDepthwiseConv2d(child))
        if isinstance(module, source.TransformerFFNBlock):
            module.fused_swiglu_kernel = False
        if isinstance(module, source.NormActConv) and module.conv1x1 is not None:
            # Source export_model_pytorch.write_normactconv folds RepVGG's
            # parallel 1x1 branch into the center of the spatial convolution.
            with torch.no_grad():
                height, width = module.conv.weight.shape[2:]
                module.conv.weight[:, :, height//2:height//2+1, width//2:width//2+1].add_(module.conv1x1.weight)
            module.conv1x1 = None
    device = next(result.parameters()).device
    obs = torch.zeros(2, result.input_planes, result.canvas, result.canvas, device=device)
    obs[:, 0] = 1
    globals = torch.zeros(2, result.global_features, device=device)
    # Source uses Python shared-data objects for registers and GAB/TAB. Trace
    # their tensor operations at fixed canvas; batch dimensions remain dynamic.
    with torch.no_grad():
        graph = torch.jit.trace_module(SearchOutputs(result).eval(),
                                       {'forward': (obs, globals), 'raw': (obs, globals)}, check_trace=False)
    # Tracing records .device as the tracing device. Recover a tensor-derived
    # device so the saved graph can be moved to CPU or another CUDA device.
    for method in (graph.forward, graph.raw):
        forward_graph = method.graph
        torch._C._jit_pass_inline(forward_graph)
        spatial = list(forward_graph.inputs())[1]
        device_node = forward_graph.create('prim::device', [spatial])
        device_node.output().setType(torch._C.DeviceObjType.get())
        device_node.insertBefore(next(forward_graph.nodes()))
        for node in list(forward_graph.nodes()):
            if node.kind() == 'prim::Constant' and node.output().type().kind() == 'DeviceObjType':
                node.output().replaceAllUsesWith(device_node.output())
                node.destroy()
        torch._C._jit_pass_lint(forward_graph)
    return SerializedSearch(graph, result.canvas, result.game).eval()
