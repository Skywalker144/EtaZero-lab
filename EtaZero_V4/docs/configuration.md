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

`configs/baseline/` 是公共参数入口，完整列出采用的参数。环境差异集中在 `configs/games/`，算法差异集中在 `configs/algorithms/`。各运行目录显式 include 所需公共文件，再写自己的覆盖项。目录后缀 `g`、`h`、`go`、`xiangqi` 分别选择 Renju、Hex、围棋和中国象棋；修改棋类以 `env.cfg` 为准，目录名不参与算法判断。

baseline 自对弈采用 KataGo selfplay1 的参数起点，排除已移除的围棋专用机制。训练和 shuffle 数值取自固定来源的单机同步示例。网络不是 KataGo 统一指定的默认值，应按实验选择。`configs/checks/` 保留独立的小规模原生接口检查，普通算法目录不代表快速检查预算。

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
game = renju
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

独立 match、GTP、Web 等仍接收原生引擎配置，不直接读取带节的训练 `selfplay.cfg`。`configs/checks/` 使用原生配置和原生命令行参数文件，可独立检查底层接口；没有 `env.cfg/net.cfg` 的此类入口继续通过原有加载路径执行。

## Include、覆盖和启动

配置文件支持 `@include path` 或 `@include = path`，路径相对包含它的文件，按出现顺序展开。后出现的值覆盖之前的值，同一文件重复写同一节的键会报错。被包含文件独立声明自己的节。所有带节文件支持同名 `.local`，在该文件及其 include 后加载；不会隐式继承父目录。

原生 CLI 参数文件中的裸键启用 flag，空赋值不传该选项，例如 `no-compile` 禁用编译，`no-compile =` 恢复原生编译默认。网络／checkpoint 推导的可选项可以留空。原生引擎字段保持原生拼写，不转换成另一套 snake_case 参数。

```bash
CONFIG_DIR=configs/az_g bash scripts/run.sh --iterations 2 --dry-run
CONFIG_DIR=configs/mz_h bash scripts/run.sh --iterations 2 --run-dir /absolute/path/to/new-run
CONFIG_DIR=configs/autoexp_example bash scripts/autoexp.sh --dry-run
```

总预算使用累计 `iterations` 或 `max_hours`，至少指定一项；CLI 优先于文件。配置位于本版本 `configs/` 时，默认输出映射到 `data/` 下相同相对路径；外部目录需显式指定输出路径。

V4 保留外围 replay 调度：复用比例通过 `replay_ratio` 设置，单轮训练由完整 epoch 上限控制，冷启动先收集 `bootstrap_min_rows`。它动态规划后续局数、按实际消费结转额度，并允许原生 reader 重复文件；与原生同步脚本的固定局数、learner train bucket 和 `no-repeat-files` 不同。当前 baseline 使用 SWA 发布，总预算由调用方指定，详见 [调度与恢复](controller.md)。

多臂 `exp.cfg` 使用 `[experiment]` 管预算、`gpus` 和共享初始化，`[arms]` 将臂名映射到配置目录；`[evaluation]` 可指定原生 match 配置和 autoelo 设置。`configs/autoexp_example/` 展示连接方法；独立 `scripts/autoelo.sh` 转发既有 autoelo 命令。评估缓存和评分口径见 [比赛与 Elo](evaluation.md)。
