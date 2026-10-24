# EtaZero V3.3 定位与 KataGo 对齐目标

EtaZero 以 KataGo 为通用搜索、训练和工程机制的对齐基准，扩展多棋类、Gumbel 和 MuZero，排除约定的围棋专属增强，并保留自己的实验配方、外层训练调度和研究工具。

本文定义**期望的 EtaZero**，作为后续逐项审查的标准。表中的“对齐”表示目标要求，不表示当前实现已经通过审查。现有代码、配置和功能文档用于确认实际行为；当前实现与本文不同的地方，应记录为待核对的差异，不能反过来用现状定义目标。

## 对照基准与原则

KataGo 基准固定为 [`d91ea855110dae533f0aada947b2b7d78cc8a4e1`](https://github.com/lightvector/KataGo/tree/d91ea855110dae533f0aada947b2b7d78cc8a4e1)。来源文件身份见 [reference_sources.json](../reference_sources.json)，许可证与其他算法来源见 [THIRD_PARTY.md](../THIRD_PARTY.md)。后续审查使用该来源，不混用其他提交的行为或训练配方。

- **共有部分默认对齐。** 对照机制、数学与数据语义、执行位置、更新频率、边界条件和依赖路径；不能只因公式或函数名称相同就认定对齐。
- **差异限于下面明确列出的范围。** 多棋类、Gumbel、MuZero 的适配只改变其实际需要的部分，不扩大成搜索、训练或工程体系的整体豁免。
- **机制与实验取值分开。** 实验可以选择不同参数和可选开关组合；机制本身及其开关、缺省和边界行为仍需审查。不能把缺失实现解释成实验配方不同。
- **允许调整语言和内部组织。** 若影响算法、数值、采样、调度、并行或性能语义，必须显式核对；现有私有接口、目录和文件格式本身不构成保留理由。

## 期望差异表

| 方面 | KataGo 基准 | 期望的 EtaZero V3.3 |
|---|---|---|
| 支持棋类 | 围棋规则、输入、动作及对称。 | 保留围棋，增加 Freestyle／Standard／Renju 五子棋、Hex、中国象棋；按各自棋规适配输入、动作、合法性、历史和有效对称，保留棋类专用的平衡开局。 |
| 结果分类与基础效用 | W／L／NR 输出；胜负效用和可配置的 NR 效用。 | W／D／L／NR；普通和棋独立监督，非围棋屏蔽 NR，围棋区分和棋与无结果。基础结果效用统一为 W−L，D、NR 均贡献 0；适用的通用搜索修正仍按来源对齐。 |
| 围棋专属增强 | 目差效用（score utility）、分数／归属／双活监督，以及贴目、让子、收官和防镜像相关增强。 | 排除 score utility、score／ownership／seki 监督、随机贴目、让子补偿，以及完整 passing／dame／anti-mirror 增强行为。保留规则必需的贴目与计分、pass、超级劫、合法性和终局判定。 |
| Gumbel | 来源搜索路径以 PUCT 为基础。 | 增加 Gumbel 根搜索、completed Q、sequential halving 和改进策略目标；非根可选 PUCT 或 Gumbel。采用 tree，不要求图共享。根预算保留严格配额；与 Gumbel 选点和目标冲突的 PUCT 根机制不叠加，其余共有机制仍需核对。 |
| MuZero | 使用真实棋盘进行搜索。 | 增加 representation／dynamics／prediction、latent 搜索和展开训练。真实规则负责环境与根合法性；latent 节点不读取真实后继棋盘，不使用真实棋盘 graph、重复模式或 SVB，不跨真实落子复用旧树。 |
| 网络适配 | 围棋输入、动作、输出头及来源网络预设。 | AlphaZero 主干采用固定来源预设，保留其布局、归一化、初始化和参数角色，按棋类与四结果适配输入和输出。MuZero 三组件使用来源模块，保留三段结构、latent、动作编码和展开所需的结构变化。 |
| 外层训练调度 | 可按新增数据向训练 bucket 补充额度，额度不足时等待或停止。 | 按计划训练量和 replay ratio 反推产样量，再按实际新增数据结算训练额度；同步执行自对弈 → shuffle → 训练 → 发布，在阶段边界换网。内部数据选择与优化器时钟继续对齐来源。 |
| 单卡训练与多卡范围 | 保留单卡训练路径，多卡时可使用 DDP 同步训练同一个模型。 | 保留当前单卡 learner 和同步逐轮机制，对齐适用的单卡数学、数据与内部训练时钟。DDP 及新增多卡／跨机器调度暂不纳入当前对齐范围，后续单独设计。 |
| 实验配方 | 不同训练／评估 profile 各有取值和开关组合。 | visits、温度、SVB 等取值及实验组合由 EtaZero 实验决定，不统一为某个 KataGo profile，也不自动启用全部可选技巧。 |
| 工程接口与数据格式 | 来源的数据、模型、配置、运行组织和交互协议。 | 共有部分以来源为对齐目标；仅为多棋类、四结果、MuZero 及 Gumbel 的实际功能增加必要扩展。现有 `.patterns`、起点格式、JSON 包装、运行目录等均不享有单独豁免。 |
| 研究工具与展示 | 来源提供的分析和可视化工具。 | 训练图、损失图、性能图、等时间 Elo 展示、自动实验工具和 Web 工作台按 EtaZero 的研究与使用需求维护。共有统计和评估数学仍按来源对齐。 |

### 结果效用与围棋增强

`W`、`D`、`L`、`NR` 是四类结果的概率，基础结果效用为 `W - L`。D 与 NR 的效用相同，不代表二者标签、监督或规则语义相同；普通和棋不拆成半胜半负。

score utility 是将预测的围棋终局目差转换后加入搜索的额外效用，使搜索除了胜负也考虑赢多少目。它属于上表的围棋专属排除项。围棋仍按真实规则计分、应用所选贴目并判定胜负；统一胜负目标不意味着删除规则计分。

“基础结果效用为 W−L”不排除 SVB、重复模式惩罚等适用的通用搜索修正。基础概率、修正后的搜索效用及其报告应保持可区分，不能把效用修正写回结果概率。未来棋盘等空间辅助目标也不能仅因是空间输出，就归入 ownership／seki 的排除范围。

### Gumbel 与 MuZero 的适配范围

Gumbel 的根选点、预算分配和策略目标按其算法来源实现，不能为了与普通 PUCT 相同而放宽 sequential-halving 配额，或用访问次数分布替代改进策略目标。根噪声、forced playout、目标剪枝、LCB 等机制是否适用，须按它们与 Gumbel 所在步骤的关系判断；不能把 PUCT 根流程整体叠加进去。算法来源与当前实现见 [Gumbel](gumbel.md)。

Gumbel 采用 tree，不要求合并不同路径到达的局面。图共享需要额外处理共享节点统计、各父边访问数与 sequential-halving 配额之间的关系，当前不纳入实现目标；这不表示图搜索在数学上不可行。tree 与跨真实落子的树复用是独立问题：当前实现使用 fresh tree，跨手复用的必要性和预算语义另行评估，不将“使用 tree”解释成必须开启或必须禁止复用。

MuZero 的必要差异包括三组件推理、动作与 latent 编码、轨迹展开、逐步监督、终局／吸收态处理及梯度边界。可共用的 PUCT 数学、价值聚合、训练损失、优化器、数据调度和发布机制仍应逐项对齐；“采用 MuZero”不意味着这些部分可以任意简化。latent 节点与真实棋盘机制的边界见 [MuZero](muzero.md)。

### 单卡训练与多卡范围

当前保留单卡 learner，不为对齐而引入 DDP。DDP（DistributedDataParallel）让多个训练进程各持一份模型，处理不同样本，并在反向传播时同步梯度，以联合更新同一个模型。它与多个 GPU 分别运行独立实验、selfplay worker 或推理服务的调度是不同职责。

DDP、分布式训练的数据分配和各 rank 状态恢复，以及新增多卡／跨机器资源调度，暂缓到单独设计。现有 selfplay 多 worker 与实验 GPU 分配功能继续保留；单卡内部的文件选择、更新频率、优化器与恢复语义仍按来源核对，不能将多卡暂缓扩大为这些机制的豁免。

### 工程接口与研究工具

工程对齐覆盖公共字段及其含义、配置解析、文件组织、模型导出／加载、起点与模式语料输入、GTP／JSON 命令及执行行为。扩展应落实到具体需要：例如象棋动作、多棋类规则标识、D 通道、MuZero 连续轨迹或 Gumbel 策略目标；不能以“EtaZero 自有格式”作为整体偏离的理由。

围棋接口范围包括矩形棋盘、来源 SGF／PositionSample，以及适用的完整 GTP 扩展。依赖被排除预测头的未终局分数／死活估计明确拒绝；规则计分、摆子与让子命令仍保留。

Web 页面和图表可以保留自己的交互、布局与呈现，后端共有协议仍按来源核对。接口调整时可同步适配工具，不要求保留旧接口或增加历史兼容层。

等时间实验的预算、模型采样与横轴属于 EtaZero 的研究口径，按既定实验要求维护；当前计时定义见 [运行观测与证据](implementation.md#运行观测与证据)，评估入口见 [评估与 Elo](elo.md)。保留这种研究口径，不豁免对比赛组织、赛果计分和 Elo 拟合数学的核对；改变口径时必须说明差异并另存结果。

## 共有机制审查表

以下按职责列出审查入口，均为**待依据实际代码核对的目标**。覆盖上表的明确差异后，其余适用部分应对齐 KataGo；表内没有“已完成”结论。来源路径均相对于上述固定 KataGo 仓库。

| 范围 | 需要核对的共有行为 | KataGo 来源入口 | EtaZero 当前入口 |
|---|---|---|---|
| PUCT、展开与回传 | 选点、FPU、并列处理、child 分配、virtual loss、终局与价值视角、子树聚合和权重。 | `cpp/search/searchexplorehelpers.cpp`、`searchupdatehelpers.cpp` | [搜索](../cpp/src/search/search.cpp)、[公共数学](../cpp/src/search/search_math.cpp)、[MuZero 搜索](../cpp/src/muzero/search.cpp) |
| 根探索、目标与落子 | 根对称、噪声和温度、forced playout、target pruning、LCB、chosen-move 处理及训练／评估入口差异；Gumbel 替换步骤单独核对。 | `cpp/search/searchhelpers.cpp`、`searchresults.cpp` | [算法](algorithms.md)、[Gumbel](gumbel.md) |
| 价值与探索修正 | uncertainty weighting、optimistic policy、noise pruning、SVB、重复模式／外部语料 bonus、root explore、futile 剪枝、capped-side 配额。 | `cpp/search/` 中的搜索辅助、SVB 与 pattern 实现 | [搜索控制](search-controls.md)、[算法](algorithms.md) |
| 图、缓存与跨手状态 | 图共享及合法性、NN cache 与搜索统计 cache 的区别、键的有效语义、替换／衰减／失效、换网、换服务方和跨手树复用；按真实棋盘与 latent 分别判断适用性。 | `cpp/search/search.cpp`、`searchnnhelpers.cpp`、`cpp/game/graphhash.cpp`、`cpp/neuralnet/nneval.cpp` | [搜索](../cpp/src/search/search.cpp)、[推理组批](../cpp/src/inference/batcher.cpp)、[执行架构](implementation.md#执行架构) |
| 搜索并行与预算 | 对局线程、搜索线程、最小 playout 分工、在途任务、计数与停止、节点／锁分片、错误传播；普通 PUCT 的完成计数和 Gumbel 严格配额分别核对。 | `cpp/search/`、`cpp/program/play.cpp` | [执行架构](implementation.md#执行架构)、[搜索预算](algorithms.md#puct-与搜索预算) |
| 自对弈采样与权重 | PCR／cheap、reduce visits、surprise、policy init、fork／side／reanalysis、hint／外部起点、training weight 与采样频率、scale-data-weight 及随机取整。 | `cpp/program/play.cpp`、`playsettings.cpp`、`cpp/dataio/trainingwrite.cpp` | [开局与采样](algorithms.md)、[数据链路](implementation.md#数据链路) |
| 监督目标与增强 | 主／对手／soft／optimistic policy、value、TD、误差、Q、variance-time、future-position；目标构造、mask、权重、loss 系数与归约、历史 dropout 和棋类有效对称。 | `cpp/dataio/trainingwrite.cpp`、`python/katago/train/metrics_pytorch.py`、`data_processing_pytorch.py` | [训练目标](algorithms.md)、[数据](../python/etazero/data.py)、[MuZero 展开](muzero.md#连续数据与展开训练) |
| 网络、metadata 与数值路径 | 固定预设、卷积／attention 模块、池化、归一化、初始化、heads、参数角色、真实 metadata、融合算子、布局、编译、AMP／FP32／TF32。 | `python/katago/train/modelconfigs.py`、`model_pytorch.py`、`cpp/neuralnet/sgfmetadata.cpp` | [网络](networks.md)、[MuZero 组件](../python/etazero/muzero/source_components.py) |
| 优化器与训练时钟 | 参数分组、LR／WD、warmup／cyclic、范数与裁剪、Batch renorm、Lookahead、SWA、Muon／NorMuon／Aurora、scaler；各项实际更新位置和频率、epoch／subepoch 跨外层轮次延续。 | `python/train.py`、`python/katago/train/trainloop_helpers.py` 及优化器模块 | [learner](../python/etazero/training.py)、[优化器](../python/etazero/optimization.py)、[归一化与训练控制](algorithms.md#归一化与可选训练控制) |
| 公共数据与 shuffle | 字段语义、shape／dtype、打包与量化、有效行与重复行、元信息、留出分区、回放窗口、两阶段 shuffle、目录合并、文件选择、预取与额度消费。 | `cpp/dataio/trainingwrite.cpp`、`python/shuffle.py`、`python/katago/utils/training_data_generator.py` | [来源数据适配](../python/etazero/source_data.py)、[reader](../python/etazero/source_reader.py)、[数据链路](implementation.md#数据链路) |
| 推理与模型交换 | 来源导出语义、权重变换、模型加载、专用 CUDA／cuDNN、真实 batch、缓存、精度和并发生命周期；MuZero 的三组件和设备 latent 单独适配。 | `python/export_model_pytorch.py`、`cpp/neuralnet/` | [导出](../python/etazero/cuda_export.py)、[推理后端](../cpp/src/inference/cuda_backend.cpp)、[MuZero 推理](muzero.md#网络与推理) |
| 发布与恢复 | learner、候选与 accepted 状态、直接接受／可选 gatekeeper、拒绝候选后的行为；模型、优化器、调度、数据消费进度、随机状态及回放的恢复范围。 | `cpp/command/gatekeeper.cpp`、`cpp/program/selfplaymanager.cpp`、`python/train.py` | [发布与恢复](implementation.md#模型发布与恢复)、[gatekeeper](../python/etazero/gatekeeper.py) |
| 比赛与 Elo | 独立游戏、双方执色轮次、认输、赛果计分、gatekeeper 提前判定和通过规则、Elo likelihood／prior／协方差／LOS。 | `cpp/command/match.cpp`、`gatekeeper.cpp`、`python/katago/utils/elo.py` | [比赛](../cpp/src/commands/match.cpp)、[Elo](elo.md) |
| 交互、棋钟与 HumanSL | GTP／JSON 公共命令和字段、任务调度／停止／持续分析、avoid／allow、focus、棋钟、ponder；第二策略模型、metadata、探索与行为混合及训练目标的区别。 | `cpp/command/gtp.cpp`、`analysis.cpp`、`cpp/search/timecontrols.cpp`、`searchtimehelpers.cpp` 及搜索路径 | [交互引擎](engine.md)、[搜索控制](search-controls.md) |
| 配置、文件与运行组织 | 参数解析、缺省与合法边界、未知字段处理、原始数据／checkpoint／发布文件组织、起点和模式语料格式；逐项核对必要扩展，不能整体放行自有体系。 | `cpp/core/config_parser.cpp`、`cpp/program/setup.cpp`、`cpp/configs/`、`SelfplayTraining.md` | [配置组织](implementation.md#配置组织)、[运行目录](implementation.md#运行目录)、[配置解析](../python/etazero/config.py) |

## 当前限制与审查结论

Gumbel 使用 tree、不做图共享，以及保留单卡 learner、暂缓 DDP 和新增多卡调度，属于上表明确约定的当前范围，不再列为本轮必须补齐的缺项。Gumbel fresh-tree 的跨手复用问题单独评估；公共 JSON 分析的优先级和多工作线程按来源实现，Web 可变棋局仍顺序执行。外层阶段同步是已明确保留的调度选择，与阶段内部能否并行分开判断。

后续审查对每个具体差异作出以下一种结论：

| 结论 | 判断依据 |
|---|---|
| 已对齐 | 固定来源与当前实现的适用行为一致，有相称的代码对照与验证证据。 |
| 约定差异 | 能明确对应期望差异表中的范围，并解释为何需要改变该处行为。 |
| 未对齐 | 共有机制缺失、简化或行为不同，且不属于约定差异。 |
| 待核实 | 来源路径、适用条件或验证证据不足，暂不能作出上述判断。 |

记录具体的来源行为、EtaZero 行为、影响、对应边界及验证依据即可。审查应覆盖实际配置到执行路径，包含启用／关闭与边界情况；对有 KataGo 对应实现的路径，也要核对额外校验、同步和持久化措施的职责与执行频率，避免增加影响算法或性能的私有机制。无效或未支持的配置应明确拒绝，不用占位、静默忽略或切换算法模拟支持。

参数取值不同不直接等于未对齐；缺少某项机制、接受配置却未执行，或把外层轮次当成内部训练时钟，也不能用“配方不同”解释。随机数序列与线程调度不要求逐位复现，但分布、计数和并行语义仍需核对；浮点与性能差异必须说明条件，不能只凭短测试推断等价。

后续审查结果应与目标分开记录，不覆盖原始日志和实验产物。工程可运行、语义对齐、恢复正确、端到端性能和棋力收益分别需要相应证据。
