# 配置目录与启动入口

V4 按环境、网络、搜索、自对弈、训练和运行调度拆分配置。外围转换层位于 `python/etazero/translation.py`；它连接既有 KataGo 引擎和 learner，不改核心解析器，不提供另一套搜索公式。

## 文件职责

| 文件 | 内容 |
|---|---|
| `env.cfg` / `[environment]` | `game`、`canvas`，以及原生棋盘尺寸、尺寸权重、棋规和贴目字段。 |
| `net.cfg` / `[network]`、`[muzero]` | 普通网络预设；MuZero representation、dynamics、prediction 预设、latent 宽度、展开长度。 |
| `selfplay.cfg` | 算法、搜索预算与参数、自对弈采样、fork、写盘、并行和 NN 服务参数，按职责分组。 |
| `train.cfg` / `[train]` | 原生 learner 选项名，省略命令行开头的 `-`。 |
| `shuffle.cfg` / `[shuffle]` | 原生 shuffler 选项名，省略命令行开头的 `-`。 |
| `run.cfg` / `[run]` | 产样／训练额度、模型发布、绘图、总预算和输出路径。 |

`configs/baseline_az/` 完整列出采用的公共参数与 AlphaZero 配方，其余算法 baseline 显式继承公共文件，再定义算法差异。四个 baseline 均使用围棋环境，也可直接运行。各棋类目录继承对应算法 baseline，只覆盖环境与该配方的差异；目录名称不参与算法判断，实际选择来自 `env.cfg` 和 `selfplay.cfg`。

| 算法 | 公共入口 | 围棋 | 五子棋 | Hex | 中国象棋 |
|---|---|---|---|---|---|
| AlphaZero | `baseline_az` | `az_go` | `az_gomoku` | `az_hex` | `az_xiangqi` |
| Gumbel AlphaZero | `baseline_gaz` | `gaz_go` | `gaz_gomoku` | `gaz_hex` | `gaz_xiangqi` |
| MuZero | `baseline_mz` | `mz_go` | `mz_gomoku` | `mz_hex` | `mz_xiangqi` |
| Gumbel MuZero | `baseline_gmz` | `gmz_go` | `gmz_gomoku` | `gmz_hex` | `gmz_xiangqi` |

`baseline_gaz` 继承 `baseline_az` 并引入共享的 [Gumbel 搜索配置](../configs/algorithms/gumbel.cfg)。`baseline_mz` 继承公共参数，独立定义 MuZero 网络、搜索和展开数据文件大小；`baseline_gmz` 继承 `baseline_mz`，再引入同一份 Gumbel 搜索配置。这样不会用普通网络或普通样本文件大小覆盖 MuZero。普通 baseline 不保留未启用的 MuZero 网络或 Gumbel 模板。

四个 `*_go` 完全继承各自 baseline。四个 `*_gomoku` 共用 [Freestyle 环境](../configs/games/gomoku/env.cfg)：15 路画布，11–15 路棋盘分布；`az_gomoku` 另保留已经确认的五子棋搜索、开局和回放参数。四个 `*_hex` 共用方盘环境；Hex 和象棋的 `replay_ratio` 均为 2。象棋的搜索预算、初始化、fork、side positions、写盘、并行、训练 batch 和 bootstrap 均继承对应算法 baseline；独立的小规模检查只使用 `configs/checks/`。

Hex 的 `chosenMoveTemperatureHalflife=15`，象棋为 40；实际半衰期为该值乘以 `sqrt(棋盘面积)/19`，象棋约为 20 个单方着手。该参数也控制根 policy 温度的衰减。普通 AZ/MuZero 自对弈启用根 Dirichlet 噪声，总浓度 10.83、混合权重 0.25；总浓度是各有效动作 α 的和，不是逐动作 α。继承的 KataGo 机制会在零训练权重的 cheap search 中临时关闭根噪声，保留完整训练搜索的噪声。当前 Gumbel 直接按自身规则出手，且两端 chosen 温度均为 0，继承的 half-life 不影响其出手。

所有算法的根多对称采样上限统一继承 `rootNumSymmetriesToSample=4`；实际数量是 `min(4, 该棋类支持的对称数)`。围棋／五子棋取 4，Hex／象棋取 2。MuZero 平均多对称的根预测输出，后续 latent 搜索保留其中一个采样朝向，不对 latent 张量直接求平均。

Gumbel 共用 `maxVisits=33`，即一次根初始化加名义上的 32 次动作访问；cheap search、reduce visits、LCB 和 forced playouts 关闭。`rootNoiseEnabled=true` 在这里启用 Gumbel 噪声，不添加 Dirichlet 噪声。非对称搜索预算仍可能缩放某一方的实际访问数，33 并不承诺每步严格执行 32 次动作访问。算法组合边界见 [Gumbel 搜索](gumbel.md)。

MuZero 两种算法保留 shuffle 输出文件约 8192 行、桶约 65536 行。MuZero 需要连续未来位置，shuffle 会从轨迹构造展开训练样本；增大文件行数会同时增加物化和读取展开字段的内存，不能直接照搬普通 AlphaZero 的文件大小。数据语义见 [MuZero 网络与展开训练](muzero.md)。

`baseline_az` 的目标是按机制语义接近 KataGo，排除 V4 已移除的围棋专用机制，并非与 main18b 完全一致。完整参数对照见 [V4／KataGo／KataGomo 参数表](config-parameter-audit-2026-10-31.md)。网络并非 KataGo 统一指定的默认值；当前配置也不代表棋力或吞吐已调优。`configs/checks/` 保留独立的小规模原生接口检查。

## 文件内分组

参数按 V3.4 的阅读习惯分组，键名仍使用 KataGo 原生拼写：

- `selfplay.cfg`：`[logging]`、`[writer]`、`[game_forks]`、`[policy_init]`、`[side_positions]`、`[reduce_visits]`、`[pda]`、`[surprise_weighting]`、`[parallelism]`、`[handicap]`、`[result_utility]`、`[inference]`。
- `selfplay.cfg` 的搜索分组：`[agent]`、`[search]`、`[puct]`、`[fpu]`、`[temperature]`、`[dirichlet_noise]`、`[forced_playouts]`、`[policy_target]`、`[lcb]`、`[symmetry]`、`[utility]`、`[value_weighting]`、`[subtree_value_bias]`、`[graph_search]`、`[gumbel]`。
- `train.cfg`：`[training]`、`[optimizer]`、`[learning_rate]`、`[weight_decay]`、`[lookahead]`、`[swa]`、`[precision]`、`[loss]`、`[validation]`、`[checkpoints]`、`[data]`、`[brenorm]`、`[attention]`。
- `shuffle.cfg`：`[replay]`、`[workers]`、`[output]`、`[filtering]`。

`[search]` 集中放置 `maxVisits`、`cheapSearchVisits`、`cheapSearchProb`、`cheapSearchTargetWeight` 和 `numSearchThreads`，完整搜索与低预算搜索可以在同一处修改。

这些分组只组织阅读，仍分别合并到该文件的原生参数集合，原来的 `[selfplay]`、`[search]`、`[train]`、`[shuffle]` 也可以用于覆盖。include 和 `.local` 的覆盖顺序不变，同一文件跨分组重复定义同一个原生键也会报错。`[agent]` 和仅在 Gumbel 算法生效的 `[gumbel]` 保留各自的语义，不参与普通分组合并。

## 配置环境和算法

例如五子棋环境：

```ini
[environment]
game = freestyle
canvas = 15
bSizes = 15,13,11
bSizeRelProbs = 10,3,1
allowRectangleProb = 0
komiMean = 0
```

`game` 接受 `go`、`freestyle`、`standard`、`renju`、`hex`、`xiangqi`。目前一臂固定一种棋类／五子棋规则，不能把 V3.4 的混合 `rules/rule_weights` 原样搬来。尺寸分布保留原生 `bSizes/bSizeRelProbs`，没有新增别名。象棋使用 `canvas=10`、`bSizesXY=9-10`，不带 `bSizes` 或 `allowRectangleProb`。

```ini
[agent]
algorithm = muzero
root_search_algo = gumbel
nonroot_search_algo = gumbel

[search]
maxVisits = 600
numSearchThreads = 1

[gumbel]
gumbelMaxConsideredActions = 4
```

算法族是 `alphazero` 或 `muzero`，根和非根是 `puct` 或 `gumbel`；Gumbel 非根要求 Gumbel 根。转换层组合成引擎／控制器已有的四种算法。根为 PUCT 时 `[gumbel]` 不参与原生配置。

```ini
[network]
architecture = b2c16-fson-mish

[muzero]
representation = b2c16-fson-mish
dynamics = b2c16-fson-mish
prediction = b2c16-fson-mish
prediction_blocks = 0
latent_channels = 6
unroll_steps = 3
```

组件可以选择不同的 `modelconfigs.py` 预设。其中 `b5c128nbt` 是 V4 按 V3.4 的宽度比例增加的 NBT 预设，使用 KataGo 原有块和 heads；其余名称沿用来源。普通算法只使用 `[network]`，MuZero 使用三个组件。转换层从环境推导棋类、输入族、Go／落子棋／象棋组件版本，并启用当前 MuZero 的 Q 输出。无需手写或复制每种棋类的三组件 JSON；实际 bundle 配置仍保存到运行目录的 `muzero_network.json`。

`prediction_blocks` 默认 0：hidden 的特征通道直接进入 policy/value heads，mask 仅用于掩码和池化；不创建输入卷积、残差块或 trunk 末端归一化。此时 heads 的输入宽度由 `latent_channels` 决定，head 内部宽度仍来自 `prediction` 预设。设为正整数后，保留该预设的输入卷积和末端归一化，并按预设块序列循环取出指定数量的块；可使用不同于 latent 的主干宽度。representation 和 dynamics 的深度、宽度由各自预设决定。所有组件仍保留当前 KataGo 的适用 heads 和 loss。

## 转换和原生边界

`game` 和 `canvas` 只在 `env.cfg` 填写；模型和 unroll 只在 `net.cfg` 填写；算法只在 `selfplay.cfg` 填写。新配置目录不能再在 run/train/shuffle 中重复设置这些内容，避免两个配置源互相覆盖。

转换后普通网络使用原生 `-model-kind`，MuZero 使用 `-muzero-config`；训练画布转为 `-pos-len`，展开长度转为 shuffler 的 `-muzero-unroll-steps`。搜索、自对弈和环境中的原生字段生成运行目录内的 `selfplay.resolved.cfg`，可直接查看实际引擎输入；解析后的全部调度设置保存在 `config.json`，`--dry-run` 也能查看。

必要的算法适配由转换层统一处理：MuZero 关闭真实棋盘 graph、SVB 和局部重复模式；Gumbel 关闭 graph 和训练根对称剪枝；非围棋不使用贴目或让子；逐轮运行固定输入模型。修改这些能力应修改对应实现，不能靠冲突配置重新启用。原生字段的数值范围由实际消费者检查。

`selfplay.cfg` 中搜索与采样参数共享原生键空间，按 include 和 `.local` 顺序覆盖；`env.cfg` 最后决定环境。其他原生参数可直接写入对应节，不需要在转换层注册别名。额外 SGF、hint 等原生资源路径仍遵循引擎工作目录语义（逐轮执行位于本版本 `python/`）；建议使用绝对路径。

独立 match、GTP、Web 等仍接收原生引擎配置，不直接读取带节的训练 `selfplay.cfg`。各算法 baseline 与棋类入口另有独立 `gtp.cfg`，以原生 include 继承公共推理参数、算法约束和棋类差异；它不参与训练配置解析，也不自动作为 match 配置。推理参数来源与启动方式见 [Web 工作台](web.md)。`configs/checks/` 使用原生配置和原生命令行参数文件，可独立检查底层接口；没有 `env.cfg/net.cfg` 的此类入口继续通过原有加载路径执行。

## Include、覆盖和启动

配置文件支持 `@include path` 或 `@include = path`，路径相对包含它的文件，按出现顺序展开。后出现的值覆盖之前的值，同一文件重复写同一节的键会报错。被包含文件独立声明自己的节。所有带节文件支持同名 `.local`，在该文件及其 include 后加载；不会隐式继承父目录。

原生 CLI 参数文件中的裸键启用 flag，空赋值不传该选项，例如 `no-compile` 禁用编译，`no-compile =` 恢复原生编译默认。网络／checkpoint 推导的可选项可以留空。原生引擎字段保持原生拼写，不转换成另一套 snake_case 参数。

```bash
CONFIG_DIR=configs/az_gomoku bash scripts/run.sh --iterations 2 --dry-run
CONFIG_DIR=configs/mz_hex bash scripts/run.sh --iterations 2 --run-dir /absolute/path/to/new-run
CONFIG_DIR=configs/autoexp_example bash scripts/autoexp.sh --dry-run
```

单臂运行可使用最终 iteration 编号 `iterations`（0 为 bootstrap）或累计 `max_hours` 限制总预算；CLI 优先于文件。配置和命令行都未指定这两项时默认无限运行，直到手动中断。配置位于本版本 `configs/` 时，默认输出映射到 `data/` 下相同相对路径；外部目录需显式指定输出路径。

V4 使用 V3.4 自适应逐轮调度：`run.cfg` 的 `bootstrap_games` 控制 iteration 0 探测局数，`train_steps` 是训练步数上限，`replay_ratio` 控制新增样本额度；`bootstrap_game_multiplier` 用于 iteration 1 的补满估算。首轮训练门槛由 `shuffle.cfg` 的 `min-rows` 指定。`train.cfg` 保留来源优化器、SWA 与 Lookahead 参数，内部 epoch 样本目标由 `train_steps × batch_size` 推导。逐轮控制器拥有精确暂停与候选发布边界，详见 [调度与恢复](controller.md)。

多臂 `exp.cfg` 使用 `[experiment]` 管预算、`gpus` 和共享初始化，`[arms]` 将臂名映射到配置目录；`[evaluation]` 可指定原生 match 配置和 autoelo 设置。`configs/autoexp_example/` 展示连接方法；独立 `scripts/autoelo.sh` 转发既有 autoelo 命令。评估缓存和评分口径见 [比赛与 Elo](evaluation.md)。
