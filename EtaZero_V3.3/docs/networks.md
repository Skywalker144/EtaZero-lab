# KataGo 网络配置

AlphaZero 的 `network.architecture` 可直接填写固定 KataGo 来源的网络名称。名称与架构来自 [手写预设](../python/etazero/models/katago/modelconfigs.py)；[解析器](../python/etazero/models/katago_configs.py) 按来源相同的阶段和约束解析后缀，不在启动时展开整张派生配置表。来源版本及文件校验值见 [reference_sources.json](../reference_sources.json)。

输入／输出适配取决于 `environment.rules`：Go 保留来源 22／19 输入和 pass head；象棋采用 113／8 输入及 50 位移通道 gather 2086 动作；其余棋类保留原有布局。主干结构不因棋类更换。六项 policy、纯 WDL、TD 和误差目标共用；Go 的分数／所有权 head 不进入目标。native 读取模型 manifest（`.pt` 使用 `game_name()`），依据共享的 [棋类定义](../python/etazero/games.py) 分配输入和动作缓冲。

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
PYTHONPATH=python conda run -n pytorch python -m etazero.models.katago_configs
PYTHONPATH=python conda run -n pytorch python -m etazero.models.katago_configs --name b10c384h6nbttflrs-fson-silu-rsnh
```

手写预设覆盖 regular、bottleneck、长 bottleneck、NBT2/NBT3、空洞 NBT、普通/嵌套 Transformer、卷积与 Transformer 混合主干，以及 GQA、独立 Q/K/V head 维度、可学习 RoPE、QK normalization、inline registers、GAB、TAB、frequency mixing TAB 和 depthwise FFN。层定义由 [来源网络](../python/etazero/models/katago/model_pytorch.py) 构建，没有按网络大小替换成其他实现。

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

`bnh` 只适用于来源允许的 batch-normalization 配置。名称存在于派生表不保证任意维度组合都可实例化，例如分组 spatial RMSNorm 的通道数仍须满足其分组约束；不会通过改变通道数或归一化来绕过来源约束。原生导出同时受固定来源 exporter 支持范围约束，例如 grouped spatial RMSNorm 的导出仍会明确拒绝；不能由训练模型可构造推定所有实验结构都能原生推理。实验预设名称本身带有 `qkn`、`ireg16`、`dwc` 等字段时，按最长手写名称优先匹配。

## 架构与棋类适配

[适配器](../python/etazero/models/katago_network.py) 保留固定来源的主干、归一化、初始化、attention/GAB/TAB 与参数分组。输入按棋类适配，Go 保留 pass，象棋使用 2086 动作 gather。主价值头及三个 TD 头公开顺序为 W/D/L/NR；来源文件保留 W/L/NR 位置，新增 draw 输出。非围棋屏蔽 NR；围棋在非简单劫且非数目规则下同样屏蔽 NR，屏蔽发生在推理 softmax 前，其余 W/D/L 重新归一化。MuZero 从真实根输入保存规则条件，后续 latent 推理沿用该条件。普通和棋不拆成半胜半负。六项策略、价值、短期误差、future-position、variance-time 和可选 v17 纯 W−L Q 进入监督；score、ownership、seki 不进入监督。

`ih` 的两组头使用来源缺省的 0.5/0.5 loss 权重；`bnh` 使用主头 0.2、BN 头 0.8，推理读取主头。各头均参与其对应 loss，checkpoint 与 SWA 保存完整参数和统计。

`meta` 保留来源的 192 维编码器，但普通 RL 配置必须选择无 metadata 的预设。HumanSL 第二模型通过 `human_sl_profile` 生成来源 SGFMetadata，包含实际执色、等级和棋盘面积，并进入缓存身份；缺少必要 metadata 时明确报错，不填零向量。

AlphaZero 的 `nbt`、`plain`、`transformer` 仅是固定来源名称的别名，分别解析为 `b5c192nbt-fson-mish`、`b10c128-fson-mish` 和 `b5c192h3nbttfrs`。不允许按别名任意缩放主干。MuZero 的三段宽度、深度和 latent 投影是必要的算法适配；其卷积块、归一化、初始化、池化、heads 与参数角色复用来源组件。

深度卷积 FFN 预设（`-dwc`）支持默认 NSC 序列布局：仅在 depthwise 卷积处将棋盘 token 和 mask 恢复为空间布局，再转换回序列。参数和卷积计算与来源 NCHW 路径一致，无需关闭 `KATAGO_TRANSFORMER_NHWC`。该预设可使用 PyTorch／CUDA 训练及 TorchScript 参考导出；固定来源的原生二进制 exporter 不支持 depthwise FFN，因此不支持原生 CUDA 自对弈，可显式配置 CPU 自对弈配合 CUDA learner。

## 训练与推理

参数分组采用来源 `add_reg_dict`，包含 metadata、attention、GAB/TAB、门控和寄存器参数；SGD、AdamW、Muon／NorMuon／Aurora 均沿用各组角色。所有来源 norm kind 接入 LR/WD 分支，`rvglr` 在反缩放之后、裁剪之前将指定 3×3 卷积的中心梯度乘二。

`rvgl` 的训练模型保留并行 1×1 分支；导出副本按来源 exporter 将其权重并入空间卷积的中心，移除该分支。原训练参数和 checkpoint 不受影响，合并可能改变浮点舍入。

Frequency-mixing TAB 的复数 depthwise 卷积在导出副本中预先拼装为来源所用的两个实数 depthwise `Conv2d`，保留原有卷积拆分及实部/虚部输出运算。固定权重不再在推理时拼接，避免 TorchScript TensorExpr 对 AMP 与拼接的输出 stride 重写问题；训练参数和计算保持来源实现。

Batch renorm 的参数与初始界限以配置为准。界限在优化器 epoch 开始及该 epoch 每 500 个已消费 batch 更新，epoch 由完整文件选择决定，独立于外层产样轮次。保存统计、当前界限和更新时间，恢复不重启递推。

训练保留来源的布局、flex attention、适用的融合内核和 FP32 heads；单图编译按来源关闭 fused RoPE/QKV。CUDA 开启 cuDNN benchmark，TF32 matmul 由 `training.use_tf32_matmul` 显式控制。CPU 参考导出关闭训练融合路径并生成 TorchScript；生产 CUDA 导出来源二进制，由专用 CUDA/cuDNN 后端执行，不经过 LibTorch 前向。

## 检查入口

### 原生推理测速

[推理测速脚本](../scripts/benchmark_inference.py) 对相同种子和权重支持 `--backends cuda torchscript`，在同进程中交替比较专用 CUDA/cuDNN 与 LibTorch/CUDA Graph。可同时比较命名架构，先按 README 构建：

```bash
conda run --no-capture-output -n pytorch python scripts/benchmark_inference.py \
  --backends cuda torchscript --architectures "$ARCH_A" "$ARCH_B" --canvas 15 --precision float16 \
  --batches 1 4 16 64 --output data/my_inference_benchmark
```

也可以用 `--config-dir` 加入现有 AlphaZero 配置目录的网络，与 `--architectures` 的命名预设一起比较。它通过训练使用的 `make_network` 构造实际网络，可测原有 `nbt` 入口；`--canvas` 须与配置中的 canvas 相同。测速不修改配置或读取训练 checkpoint。

`--game gomoku|hex|go|xiangqi` 指定命名预设的棋类；未提供时取配置中的棋类，无配置时使用 Gomoku。与配置同时使用须一致，象棋要求 `--canvas 10`。输入由该棋类的规则状态生成固定两手合法前缀，生成过程在计时外；不同棋类的输入和动作维度不同，比较时应固定棋类。

每组预热后按 repeats 交替执行。耗时含输入打包、H2D、前向、同步 D2H 和结果构造，不含模型加载、预热、排队、缓存与搜索。原生 CUDA 使用实际 batch；TorchScript 路径补到图 bucket，`executed_batch` 记录实际执行批量。局部吞吐不能直接换算成整轮训练加速。

测速须在可访问宿主 CUDA 的环境执行。输出目录必须不存在，保留源码快照、架构配置、种子、模型与二进制哈希、构建配置、软硬件信息、逐轮 `raw.jsonl` 及 `summary.json/csv`。汇总采用各轮平均 batch 延迟与吞吐的中位数，并列吞吐范围；这不是单次请求的尾延迟统计。

### 架构正确性

[架构测试](../tests/test_katago_architectures.py) 检查全部手写预设的真实模块构建与参数角色、名称/字段校验、输出头和归一化组合；[CUDA 测试](../tests/test_katago_gpu.py) 检查原生可变 batch 推理、跨设备导出、混合精度训练和续训。

[来源对照](../tests/reference/check_katago_architectures.py) 核对固定 commit 和文件哈希，使用来源完整模型逐个比较初始化、参数形状/角色、前向和梯度；`--all-configs` 遍历来源全部生成名称并逐项比较配置。完整网络保留原宽度和深度，检查只缩小输入棋盘。来源检查不随普通 pytest 自动执行：

```bash
conda run -n pytorch python tests/reference/check_katago_architectures.py \
  --device cuda --all-configs --backward --output /tmp/katago_architectures_check.json
```

这些检查验证实现与执行链路，不证明所有大网络的训练收敛、棋力或等时间性能收益。
