# KataGo 网络配置

AlphaZero 的 `network.architecture` 可直接填写固定 KataGo 来源的网络名称。名称与架构来自 [手写预设](../python/etazero/_vendor/katago/modelconfigs.py)；[解析器](../python/etazero/katago_configs.py) 按来源相同的阶段和约束解析后缀，不在启动时展开整张派生配置表。来源版本及文件校验值见 [reference_sources.json](../reference_sources.json)。

输入／输出适配取决于 `environment.rules`：Go 保留来源 22／19 输入和 pass head；象棋采用 113／8 输入及 50 位移通道 gather 2086 动作；其余棋类保留原有布局。主干结构不因棋类更换。六项 policy、纯 WDL、TD 和误差目标共用；Go 的分数／所有权 head 不进入目标。native 读取导出的 `game_name()`，依据共享的 [棋类定义](../python/etazero/games.py) 分配输入和动作缓冲。

## 选择网络

例如，配置目录的 `net.cfg` 可以直接写为：

```ini
[network]
architecture = b28c512nbt-fson-mish-rvglr-bnh
canvas = 15
predict_q_values = false
```

未填写 `channels`、`blocks` 时从预设推导。若当前文件或被引用的父配置已经指定这两个字段，必须与预设完全一致，否则拒绝配置。`blocks` 对应来源 `block_kind` 的条目数；普通 Transformer 的 attention 与 FFN 分别计一条，名称中的 `b7` 不一定意味着此字段为 7。主干宽度、瓶颈宽度、global pooling 位置、Policy 的局部/全局宽度、Value 的卷积/隐藏宽度均采用来源值，不能通过同名字段缩放预设。

查看全部手写名称与一个名称的实际配置：

```bash
PYTHONPATH=python conda run -n pytorch python -m etazero.katago_configs
PYTHONPATH=python conda run -n pytorch python -m etazero.katago_configs --name b10c384h6nbttflrs-fson-silu-rsnh
```

手写预设覆盖 regular、bottleneck、长 bottleneck、NBT2/NBT3、空洞 NBT、普通/嵌套 Transformer、卷积与 Transformer 混合主干，以及 GQA、独立 Q/K/V head 维度、可学习 RoPE、QK normalization、inline registers、GAB、TAB、frequency mixing TAB 和 depthwise FFN。层定义由 [来源网络](../python/etazero/_vendor/katago/model_pytorch.py) 构建，没有按网络大小替换成其他实现。

后缀按来源顺序选择，可跳过阶段，每阶段最多一个：

| 阶段 | 可选后缀 |
|---|---|
| 归一化 | `fs`、`bn`、`brn`、`fbrn`、`bng`、`brng`、`fbrng`、`fson`；缺省为 Fixup |
| 激活 | `elu`、`gelu`、`mish`、`silu`；缺省为 ReLU |
| RepVGG | `rvgi`、`rvgl`、`rvglr` |
| 主干通道门 | `tcg` |
| 主干 residual backout | `trb`、`trbnr` |
| 输出头 | `ih`、`bnh`、`rnh`、`rsnh`、`rgnh` |
| 元数据编码器 | `meta` |

`bnh` 只适用于来源允许的 batch-normalization 配置。名称存在于派生表不保证任意维度组合都可实例化，例如分组 spatial RMSNorm 的通道数仍须满足其分组约束；不会通过改变通道数或归一化来绕过来源约束。实验预设名称本身带有 `qkn`、`ireg16`、`dwc` 等字段时，按最长手写名称优先匹配。

## 架构与棋类适配

[适配器](../python/etazero/katago_network.py) 保留来源全部主干、归一化、初始化、共享 GAB/TAB 模块及输出头。输入投影按棋类采用 5／8、22／19 或 113／8 的空间／全局特征；padding 在投影前按有效棋盘 mask 清零。来源三分类 head 从头按 EtaZero 的 W/D/L 语义训练，并调整通道顺序；这不是原生 KataGo no-result 标签或预训练权重的语义转换。Go 保留 pass，Gomoku／Hex 去除 pass，象棋替换为固定走法 head。训练选取六项 policy、主 WDL、三个 TD WDL 和短期价值误差；v17 的 `predict_q_values=true` 额外监督逐动作纯 W−L Q。来源的 score、ownership、seki 等输出仍在训练模型中保留，EtaZero 不为它们构造监督；Go pass 则参与所有对应策略损失。导出图只保留搜索所需输出。

`ih` 的两组头使用来源缺省的 0.5/0.5 loss 权重；`bnh` 使用主头 0.2、BN 头 0.8，推理读取主头。各头均参与其对应 loss，checkpoint 与 SWA 保存完整参数和统计。

`meta` 保留来源的 192 维编码器。EtaZero 对局没有 SGF/玩家元数据，游戏链路显式使用全零元数据向量。Python 的 `source_outputs(obs, globals, metadata=...)` 可输入真实的 192 维向量；此接口不意味着自对弈 writer 已有玩家元数据。元数据编码器的缺省激活与主干一致。

保留的 `nbt`、`plain`、`transformer` 是 EtaZero 原有架构入口；它们的缩放规则或固定小预设不代表完整 KataGo 预设。精确选择来源架构时使用来源名称。MuZero 的 representation/dynamics/prediction 仍使用其独立 `nbt` 或 `resnet` 配置，不把 AlphaZero 的来源模型当成 MuZero 网络。

## 训练与推理

参数分组采用来源 `add_reg_dict`，包含 metadata、attention、GAB/TAB、门控和寄存器参数；SGD、AdamW、Muon 均沿用各组角色。所有来源 norm kind 接入 LR/WD 分支，`rvglr` 在反缩放之后、裁剪之前将指定 3×3 卷积的中心梯度乘二。

`rvgl` 的训练模型保留并行 1×1 分支；导出副本按来源 exporter 将其权重并入空间卷积的中心，移除该分支。原训练参数和 checkpoint 不受影响，合并可能改变浮点舍入。

Frequency-mixing TAB 的复数 depthwise 卷积在导出副本中预先拼装为来源所用的两个实数 depthwise `Conv2d`，保留原有卷积拆分及实部/虚部输出运算。固定权重不再在推理时拼接，避免 TorchScript TensorExpr 对 AMP 与拼接的输出 stride 重写问题；训练参数和计算保持来源实现。

Batch renorm 默认保持来源初始界限 `rmax=1,dmax=0`，EMA momentum 为 0.001。需要随样本数调整界限时，在 `train.cfg` 显式设置 `[brenorm]` 的 `avg_momentum`、`target_rmax`、`target_dmax`、`adjustment_scale`。界限按来源的指数更新公式由累计消费样本数推导；模型保存 running/renorm statistics，续训恢复样本时钟，不重新开始调整。

执行采用来源公开的 NCHW/SDPA 路径并保留显式 mask。单图编译按来源要求关闭 fused RoPE/QKV attention 路径；支持形状的 CUDA AMP SwiGLU 保留来源融合内核。导出关闭训练融合内核，以固定 canvas 的 tensor trace 封装成 TorchScript，并保留动态 batch 和输入设备；C++ 使用已有 LibTorch/CUDA Graph 后端。此执行选择不代表与来源默认 NHWC/flex attention 的性能或浮点运算顺序完全等价。

## 检查入口

### 原生推理测速

[推理测速脚本](../scripts/benchmark_inference.py) 为所选 KataGo 名称生成同一随机种子初始化的模型，通过生产 `TorchBackend` 在同一进程中交替测量。先按 README 构建；将 `ARCH_A`、`ARCH_B` 设为需要比较的完整架构名称，在本版本目录运行：

```bash
conda run --no-capture-output -n pytorch python scripts/benchmark_inference.py \
  --architectures "$ARCH_A" "$ARCH_B" --canvas 15 --precision float16 \
  --batches 1 4 16 64 --output data/my_inference_benchmark
```

也可以用 `--config-dir` 加入现有 AlphaZero 配置目录的网络，与 `--architectures` 的命名预设一起比较。它通过训练使用的 `make_network` 构造实际网络，可测原有 `nbt` 入口；`--canvas` 须与配置中的 canvas 相同。测速不修改配置或读取训练 checkpoint。

`--game gomoku|hex|go|xiangqi` 指定命名预设的棋类；未提供时取配置中的棋类，无配置时使用 Gomoku。与配置同时使用须一致，象棋要求 `--canvas 10`。输入由该棋类的规则状态生成固定两手合法前缀，生成过程在计时外；不同棋类的输入和动作维度不同，比较时应固定棋类。

每组先预热，再按 `--repeats` 重复计时，交替模型顺序与 batch 遍历方向。耗时包含 CPU 输入打包、H2D、CUDA Graph 重放、同步 D2H 和结果构造；加载、图捕获与预热单列，不计入吞吐。结果不包含组批排队、缓存和搜索，也不直接等于自对弈速度。非图桶大小的 batch 会补到下一个桶，原始记录和汇总均记录 `graph_batch`。

测速须在可访问宿主 CUDA 的环境执行。输出目录必须不存在，保留源码快照、架构配置、种子、模型与二进制哈希、构建配置、软硬件信息、逐轮 `raw.jsonl` 及 `summary.json/csv`。汇总采用各轮平均 batch 延迟与吞吐的中位数，并列吞吐范围；这不是单次请求的尾延迟统计。

### 架构正确性

[架构测试](../tests/test_katago_architectures.py) 检查全部手写预设的真实模块构建与参数角色、名称/字段校验、输出头和归一化组合；[CUDA 测试](../tests/test_katago_gpu.py) 检查原生可变 batch 推理、跨设备导出、混合精度训练和续训。

[来源对照](../tests/reference/check_katago_architectures.py) 核对固定 commit 和文件哈希，使用来源完整模型逐个比较初始化、参数形状/角色、前向和梯度；`--all-configs` 遍历来源全部生成名称并逐项比较配置。完整网络保留原宽度和深度，检查只缩小输入棋盘。来源检查不随普通 pytest 自动执行：

```bash
conda run -n pytorch python tests/reference/check_katago_architectures.py \
  --device cuda --all-configs --backward --output /tmp/katago_architectures_check.json
```

这些检查验证实现与执行链路，不证明所有大网络的训练收敛、棋力或等时间性能收益。
