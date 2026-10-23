# EtaZero 与 KataGo：差异与对齐审查清单

本文用于审查 EtaZero_V3.1：先明确有意保留的差异，再核对借鉴 KataGo 的共有机制。**“已有实现”“存在来源对照测试”和“当前生产链路已验证对齐”是三个不同结论。** V3.1 已接入原审查 A01–F03、G01 共 26 项机制；G02 实验配方按用户要求保留。逐项实现、适用边界和分版本验证证据见 [对齐审查报告](katago-alignment-audit.md)。下方检查项不因局部测试通过而整体勾选。

## 比较基准

- EtaZero：`EtaZero_V3.1`，由 V3 基线创建，版本提交 `15d5fc9`；当前实现改动见对齐审查。原始 V3 审查和 V3.1 新验证分别记录。
- KataGo：本地固定提交 `d91ea855110dae533f0aada947b2b7d78cc8a4e1`，以 [reference_sources.json](../reference_sources.json) 登记的来源为准，不泛指论文版、其他历史版本或官方某次训练。
- 工作约定中的 `~/RL/SkyZero/KataGo` 在当前机器不存在；实际来源是 [/home/sky/RL/SkyZero-lab/KataGo](/home/sky/RL/SkyZero-lab/KataGo)。整理时核对了 HEAD 与登记的 61 个 KataGo 文件 SHA-256，全部一致。这只确认参考源码身份。
- 对照配置必须单独指定。来源的 `selfplay1.cfg`、`selfplay8mainb18.cfg`、解析器缺省值和 EtaZero `baseline` 不是同一套参数。算法存在、配置启用、默认取值相同应分别核对。
- 本文维护边界与检查入口；V3.1 的 CPU/CUDA 验证和原始 V3 数值对照分别见 [审查报告](katago-alignment-audit.md)。没有把工程通过表述为棋力或性能收益。

## 对最初四点的准确表述

1. **多算法**：支持 AlphaZero、Gumbel AlphaZero、MuZero、Gumbel MuZero 四类；Gumbel 可选 PUCT 或 Gumbel 非根策略，因此实际有六个合法搜索组合。算法身份应记录 `(algorithm, root_search_algo, nonroot_search_algo)`，见 [算法组合](algorithms.md#算法与搜索组合)。
2. **训练量驱动的同步轮次**：固定基准 `train_steps` 和 `batch_size`，通过 `replay_ratio` 反推计划新增数据量及对局数；实际步数受额度和快照单遍限制，并非每轮强制执行固定步数。KataGo 本身也有训练额度桶和同步脚本，区别在具体编排与记账语义。
3. **多棋类**：支持 Gomoku（Freestyle／Standard／Renju）、Go、Hex、Xiangqi。共用算法与工程实现，各自保留规则、输入、动作和对称适配；默认分别训练，不意味着一个权重文件可直接跨全部棋类使用。
4. **“KataZero”的定位**：可以理解为“多棋类、多算法的 AlphaZero／MuZero 框架，吸收 KataGo 中可泛化的搜索、训练与数据机制”。更精确的边界是：不采用完整围棋专用监督和分数效用体系；仍有围棋规则、围棋特征等专用适配。**“只选通用 trick”不等于“所有通用 trick 都已移植”，也不等于“已逐项严格对齐”。**

## 还需要记录的差异

### 算法、状态与监督

| 项目 | EtaZero_V3.1 的行为与审查含义 |
|---|---|
| 搜索效用 | 基础目标是纯 `W−L`，没有 KataGo 的 static／dynamic score utility。SubtreeValueBias 可以修正搜索效用，因此也不能把所有内部 utility 都当成未经修正的 `W−L`。与来源比较 uncertainty、optimistic policy、Q、LCB 时须先消除 score 分量并匹配价值视角。 |
| 和棋与无结果 | EtaZero 三分类是普通胜／和／负；KataGo 的第三类是 no-result，普通和棋通过 `drawEquivalentWinsForWhite` 分到胜负目标。EtaZero Go 的 no-result 单独记录，主轨迹不训练，搜索中性回传，Elo 排除；不能只交换 head 通道就宣称标签等价。 |
| 辅助训练目标 | 保留六项 policy、主 WDL、三个 TD WDL、短期价值误差、未来棋盘占据、variance-time 及可选 v17 逐动作纯 W−L Q；不监督 score、ownership、seki 等围棋专用输出。命名 KataGo 网络中保留这些输出参数，不代表训练了这些任务。 |
| MuZero | 使用 representation／dynamics／prediction、latent 搜索与 unroll 训练；树内不调用真实规则判断未来终局，没有真实棋盘图共享、跨手树复用或 SubtreeValueBias。真实环境仍负责产出轨迹与监督。不能直接把 AlphaZero 路径的通过结论扩展到 MuZero。 |
| Gumbel | 根节点使用 Sequential Halving、completed-Q 改进策略；训练目标不等于访问分布。根不叠加 PUCT 的 Dirichlet、根 policy 温度、forced playout、target pruning、LCB 等步骤；当前禁止图共享和跨手复用。非根选 PUCT 不会使根恢复为 KataGo 搜索。 |
| 网络选择与输入输出 | AlphaZero 可选固定来源的命名网络，也保留 EtaZero 的 `nbt/plain/transformer` 入口；可缩放 `nbt` 的宽度规则不代表全部 KataGo 预设。Gomoku／Hex、Go、Xiangqi 的输入维数和动作布局不同；象棋使用联合走法 head。`meta` 在游戏链路使用零元数据。MuZero 的三段网络另行定义。 |
| 开局与数据分布 | 五子棋平衡开局和 policy init 直接参考 KataGomo，Hex 平衡开局参考 KataGomo Hex2024；Go／象棋从自身标准初始状态或合法前缀进入，不套用五子棋骨架。EtaZero Go 没有照搬来源完整的随机贴目、让子与公平贴目训练分布。它们是数据分布差异，不能全部归为 KataGo 通用 trick。 |
| 图身份和局部模式 | Go 图身份按来源 GraphHash，保留完整路径规则状态及下降前合法性复核；其他棋类采用自己的状态身份。SubtreeValueBias 的统计更新可共用，局部模式键随棋类适配。必须把“数学更新相同”和“哪些状态共享统计相同”分开核对。 |

主要依据：[游戏与价值语义](algorithms.md#围棋与中国象棋)、[Go 适配](../cpp/src/game/go.cpp)、[网络适配器](../python/etazero/katago_network.py)、[网络选择](networks.md)、[MuZero 边界](muzero.md#katago-工程机制的适用边界)、[Gumbel](gumbel.md)。来源对应 [SearchParams](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchparams.h)、[trainingwrite.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/dataio/trainingwrite.cpp)、[metrics_pytorch.py](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/metrics_pytorch.py) 及 [mainb18 自对弈配置](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg)。

### 调度、执行与可复现性

| 项目 | EtaZero_V3.1 的行为与审查含义 |
|---|---|
| 轮次与模型更新 | 完整轮次为 selfplay → shuffle → learner → 发布；一轮产样使用固定模型。无跨阶段异步流水线、局内换网或 DDP learner。KataGo 同时提供同步脚本及常驻异步运行方式，不能简单写成“Eta 同步、KataGo 异步”。 |
| 数据额度和冷启动 | EtaZero 根据训练预算反推本轮对局数，剩余额度在整轮提交时结转；冷启动单独处理。来源同步示例按固定局数产样，learner 使用自身有容量上限的训练桶、epoch／subepoch 与文件消费规则。二者有共同思想，但不是同一个控制器。 |
| 推理与数值执行 | EtaZero 使用 TorchScript／LibTorch 与 CUDA Graph；没有移植 KataGo 专用 CUDA/cuDNN 推理后端。命名网络采用来源公开的 NCHW／SDPA 路径及显式 mask。架构、数学输出、精度、浮点顺序与端到端吞吐应分别比较。 |
| 多线程搜索预算 | EtaZero 严格预约 visit／playout 额度；KataGo 的并发搜索可能因在途线程超过上限。相同配置 visits 不保证相同完成模拟数、NN 调用数或耗时。 |
| 数据格式与留出单位 | 使用自身的完整轨迹、稀疏采样位置、重复行与 MuZero 序列布局，非原生 KataGo 训练文件协议。启用 validation 时按棋局身份留出 1%，同局分片、side 和 unroll 不跨分区；来源脚本按文件路径哈希留出。 |
| 优化配方的选择与时钟 | 支持 KataGo 分组优化配方，也支持显式普通优化模式。普通模式不启用分组调度、warmup、Lookahead／SWA。使用 KataGo 配方时，本地分段由实际轮次预算决定，短轮还会调整 Lookahead 周期；不能仅比较 LR、momentum 两个数。 |
| 恢复范围 | EtaZero 保存 learner RNG、AMP scaler、Lookahead fast／slow／counter、reader 文件内消费游标及阶段状态；固定来源 `train.py` 的 save 不保存上述全部状态。模型、优化器、SWA 等共有部分可核对，但整个恢复协议不要求相同。selfplay 在途半局和 fork 池也不能声称逐位恢复。 |
| 发布和评估 | EtaZero 正常训练直接发布，有 SWA 时优先导出 SWA；棋力比赛独立进行。KataGo 可选择 gatekeeper 或直接发布，不能写成“来源必有 gating”。EtaZero 的成对比赛、Elo 和完整轮次计时是自身实验协议，须独立核对预算与统计含义。 |
| 随机流与日志 | 开局、shuffle、增强等采用本地随机流，不承诺与来源同 seed 逐位相同。部分日志尺度也不同，例如本地 soft-policy loss 记录乘系数后的值。恢复确定性、采样分布等价、来源逐位一致不能混用。 |

调度的最小核对式（正常阶段，`iteration >= 2`）：

```text
S = 基准 train_steps；B = batch_size；R = replay_ratio
计划新增行数 = S * B / R
可用额度 = 上轮剩余额度 + 本轮实际新增有效行数 * R
实际步数 = min(S, floor(可用额度 / B), 快照内逐文件完整 batch 总数)
结转额度 = 可用额度 - 实际步数 * B
```

这里的“行”是经过采样权重随机取整后的训练行，包含 side 和重复行，不是独特局面数或真实环境步数。MuZero 消费的是起点行，不是 `K+1` 个展开状态。bootstrap 不训练，首次训练另按冷启动规则，不套用上述正常阶段公式。

依据：[iteration_plan／training_budget](../python/etazero/runtime.py)、[数据链路与额度](implementation.md#训练额度与自对弈产量)、[发布与恢复](implementation.md#模型发布与恢复)、[来源同步脚本](/home/sky/RL/SkyZero-lab/KataGo/python/selfplay/synchronous_loop.sh)、[来源 train.py](/home/sky/RL/SkyZero-lab/KataGo/python/train.py)、[来源 shuffle.sh](/home/sky/RL/SkyZero-lab/KataGo/python/selfplay/shuffle.sh)、[本地评估协议](elo.md)。

## 容易混淆的通用搜索机制

以下能力均已接入 V3.1。它们与原有机制作用不同，审查配置与调用链时仍需分开核对。使用方法见 [通用搜索控制](search-controls.md)。

| 来源能力 | 需要避免的混淆 |
|---|---|
| `rootSymmetryPruning` | 对根对称等价动作剪枝，不等于根多对称 NN 平均，也不等于训练 D4 增强。 |
| `useEvalCache`／`evalCacheMinVisits` | 搜索层的评估统计缓存，不等于 EtaZero 已有的原始 NN 输出缓存或当前搜索图共享。 |
| `wideRootNoise` | 额外的根探索机制，不等于 shaped Dirichlet，也不等于 Gumbel 根噪声。 |
| `futileVisitsThreshold` | 根据剩余预算判断追赶无望动作的搜索剪枝，不等于生成训练目标时的 policy target pruning。 |
| `visitCapContempt` | 对一方在节点处的访问分布建模，不等于 PDA 给双方分配不对称搜索预算。 |

依据：[来源参数定义](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchparams.h)、[搜索生命周期](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.cpp)、[根对称处理](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchhelpers.cpp)、[探索与缓存使用](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchexplorehelpers.cpp)，对照 [本地配置解析器](../python/etazero/config.py) 与 [本地搜索](../cpp/src/search/search.cpp)。这五项与其余机制的实现位置和验证证据统一维护在 [审查报告](katago-alignment-audit.md)。

## 共有机制的逐项审查

以下以 **AlphaZero＋PUCT** 为最直接的 KataGo 对照对象；Gumbel 和 MuZero 只继承各自适用的部分。比较损失和搜索统计前，先明确纯 W−L、普通和棋、动作域等适配，不能要求已经声明不同的语义输出仍与完整 Go 模型相同。

每项通过至少需要回答：**来源公式／分支是什么，EtaZero 在哪里调用、调用多频繁，哪些配置生效，输入与统计口径是否相同，证据覆盖哪些边界。** 现有测试只是复核入口，需检查它是否真正调用当前生产路径。

### S：搜索

- [ ] **S01 PUCT 与 FPU**：动态 cpuct、父价值标准差缩放、visited-policy 插值、root／nonroot FPU、loss proportion、未展开动作以及 virtual loss；核对用的是 visits、weight 还是 ESS。
- [ ] **S02 根先验变换**：普通／optimistic logits 混合 → 全树温度 → 各对称 softmax 与平均 → 根温度 → shaped Dirichlet 的顺序；核对合法域、零先验、浓度总和及模型缓存条件。
- [ ] **S03 forced playout 与 target pruning**：配额含 pending 的方式、稳定子边选择、反解权重与取整、LCB 资格、chosen-move subtract／prune 的先后；实际落子与训练 target 分开检查。
- [ ] **S04 LCB 与落子温度**：加权二阶矩、ESS、方差先验、门槛、首 child／并列情况、温度按面积与手数衰减；来源 `useNonBuggyLcb` 分支须与本地约定一致。
- [ ] **S05 子树统计**：value weighting 的 t(3) 表、noise pruning 的 child 顺序及剩余权重、NN 初始样本、uncertainty 的误差尺度与真实终局权重、weight_sq 缩放；不能只验最终均值。
- [ ] **S06 SubtreeValueBias**：key 生成时机与状态共享范围、误差累积、更新频率、移根／清树／释放衰减、原始 WDL 与修正 utility 分离。Go 与其他棋类的 key 单独验证。
- [ ] **S07 图搜索与树复用**：node visits 与 edge visits、catch-up／leak、并发回传、共享子节点生命周期、晋升根 NN 刷新；Go 有界历史共享、循环和完整历史合法性复核独立覆盖。
- [ ] **S08 预算与停止**：根初次 NN、根多对称、复用访问、new playout、根边模拟的计数；零预算、时间／取消、在途尾部。严格并行额度作为已声明差异保留，不强求调度序列相同。

EtaZero 入口：[search.cpp](../cpp/src/search/search.cpp)、[search_math.cpp](../cpp/src/search/search_math.cpp)、[subtree_value_bias.cpp](../cpp/src/search/subtree_value_bias.cpp)、[search.h](../cpp/include/etazero/search.h)。

KataGo 对应：[searchexplorehelpers.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchexplorehelpers.cpp)（S01／S03）、[searchhelpers.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchhelpers.cpp) 与 [searchresults.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchresults.cpp)（S02–S04）、[searchupdatehelpers.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchupdatehelpers.cpp)（S05／S06）、[search.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.cpp)、[searchnnhelpers.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchnnhelpers.cpp)、[graphhash.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/game/graphhash.cpp)（S06–S08）。

现有复核入口：[reference formulas](../tests/reference/check_reference_formulas.py)、[search corrections](../tests/reference/check_katago_search_corrections.py)、[SVB](../tests/reference/check_katago_subtree_value_bias.py)、[Go SVB](../tests/reference/check_katago_go_bias.py)、[graph](../tests/reference/check_katago_graph.py)、[Go graph](../tests/reference/check_katago_go_graph.py)，以及 [search_features_test.cpp](../cpp/tests/search_features_test.cpp)。公式对照不能代替对 selfplay／analysis／match 调用条件的检查。

### D：自对弈与训练数据

- [ ] **D01 PCR 与 Reduce Visits**：先后与互斥、历史窗口价值视角、cheap 零权重时清树／噪声／根 FPU／对称数改变、缩减预算与基础训练权重的一致性。
- [ ] **D02 PDA**：抽样概率、优势正负号、双方预算倍增、输入条件、fork／side／reanalysis 中保留或清除的时机；与来源普通非让子分支比较。
- [ ] **D03 hint、early/game/hint fork**：抽样池、完整初始历史、强制 hint 的预算／先验／探索、后续六手 cheap 概率、首值替换与终局边界；不只验 fork 能启动。
- [ ] **D04 side 与 reanalysis**：候选分布、递归队列、独立 full search、无主局 outcome 时的 loss mask；重分析选点、重算 surprise 与保留实际行动历史的顺序。
- [ ] **D05 surprise weighting**：policy KL 的先验是否含噪声、value 的平滑与视角、基础权重、低 surprise 分支、cheap excess 恢复条件、开局排除、最终随机取整；避免 writer 与 learner 重复乘权重。
- [ ] **D06 目标落盘**：policy 选择权重缩放和 int16 量化、opponent 使用下一实际手搜索、零采样后继仍有效、末手占位 mask；Q 取 child node visits 与纯 W−L，逐重复行随机量化；主局与 side 分别验证。
- [ ] **D07 replay／shuffle／reader**：power-law 窗口与随机冷启动封顶、mtime 选整片、组内采样取整、两阶段分桶、multi-wave、字段共同排列、每文件丢弃不完整 batch、单遍消费。验证切分和 RNG 是已声明适配，测试中存在的 repeat-order 辅助实现不代表生产支持多遍。

EtaZero 入口：[search_limits.cpp](../cpp/src/selfplay/search_limits.cpp)、[sampling.cpp](../cpp/src/selfplay/sampling.cpp)、[record.cpp](../cpp/src/selfplay/record.cpp)、[data.py](../python/etazero/data.py)、[shuffle.py](../python/etazero/shuffle.py)、[reader.py](../python/etazero/reader.py)。

KataGo 对应：[play.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/play.cpp)（D01–D05）、[trainingwrite.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/dataio/trainingwrite.cpp)（D05／D06）、[shuffle.py](/home/sky/RL/SkyZero-lab/KataGo/python/shuffle.py)、[training_data_generator.py](/home/sky/RL/SkyZero-lab/KataGo/python/katago/utils/training_data_generator.py)、[data_processing_pytorch.py](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/data_processing_pytorch.py)（D07）。

现有复核入口：[sampling](../tests/reference/check_katago_sampling.py)、[sampling weights](../tests/reference/check_katago_sampling_weights.py)、[forks](../tests/reference/check_katago_forks.py)、[Q](../tests/reference/check_katago_q.py)、[replay](../tests/reference/check_katago_replay.py)、[sampling_test.cpp](../cpp/tests/sampling_test.cpp)、[writer](../tests/test_writer.py)。开局及禁手特征 dropout 应另外对照 KataGomo，不把对应测试当成 KataGo 一致性证据。

### N：网络、损失与优化

- [ ] **N01 网络结构**：完整预设名、宽度、block 排列、gpool、归一化、初始化、mask、RepVGG、Transformer 选项与参数分组；原有可缩放 `nbt` 和固定来源命名预设分别核查。
- [ ] **N02 policy 损失**：主／opponent／两个 soft／长短 optimistic 的 head 对应、系数、合法域与 on-board 平滑域区别、Go pass、stop-gradient、side／末手／无结果 gate；对比前向数值和反向梯度。
- [ ] **N03 value／TD／error／Q**：当前玩家与固定黑方视角转换、普通和棋映射、完整轨迹 TD 时间尺度、target entropy、短期误差 surrogate backward、optimistic 权重、Q mask 与归一化。score 删除后的退化式单独写清。
- [ ] **N04 SGD／AdamW／Muon／NorMuon／Aurora**：每个参数恰好归入正确组、batch 求和或均值、LR／WD scaling、warmup、范数自适应、裁剪尺度、耦合或解耦衰减、Muon 矩阵与辅助 Adam 分工；普通优化模式单独标识。
- [ ] **N05 Lookahead／SWA／BN 时钟**：subepoch 起点、短轮周期调整、轮末 fast→slow、SWA 样本时钟、统计 buffer、batch renorm、AMP skip 时哪些计数推进；核对执行频率而非仅更新公式。
- [ ] **N06 精度与导出**：训练主干 AMP、FP32 heads／loss、推理 head 精度、归一化统计及导出折叠、raw 与 SWA 选择、TorchScript／C++ 一致性；用匹配精度的真实后端检查。
- [ ] **N07 增强与多头适配**：Gomoku／Go D4、Hex 有效两对称、象棋镜像，观测／动作／policy／Q 同变换；来源双 head 的 loss 权重、推理 head 选择和 `meta` 零输入。规则适配不要求与 Go 原始输入张量相同。

EtaZero 入口：[network.py](../python/etazero/network.py)、[katago_network.py](../python/etazero/katago_network.py)、[training.py](../python/etazero/training.py)、[optimization.py](../python/etazero/optimization.py)、[muon.py](../python/etazero/muon.py)、[symmetry.py](../python/etazero/symmetry.py)、[export.py](../python/etazero/export.py)。

KataGo 对应：[modelconfigs.py](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/modelconfigs.py)、[model_pytorch.py](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/model_pytorch.py)（N01／N06／N07）、[metrics_pytorch.py](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/metrics_pytorch.py) 与 [trainingwrite.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/dataio/trainingwrite.cpp)（N02／N03）、[train.py](/home/sky/RL/SkyZero-lab/KataGo/python/train.py) 与 [muon.py](/home/sky/RL/SkyZero-lab/KataGo/python/muon/muon.py)（N04／N05）、[export_model_pytorch.py](/home/sky/RL/SkyZero-lab/KataGo/python/export_model_pytorch.py)（N06）。

现有复核入口：[network](../tests/reference/check_katago_network.py)、[architectures](../tests/reference/check_katago_architectures.py)、[Q](../tests/reference/check_katago_q.py)、[clocks](../tests/reference/check_katago_clocks.py)、[Muon](../tests/reference/check_katago_muon.py)、[auxiliary](../tests/test_auxiliary.py)、[optimization](../tests/test_optimization.py)、[CUDA 架构链路](../tests/test_katago_gpu.py)。特别注意 clocks 对照本身不覆盖训练额度桶，也不能替代实际 GradScaler 的 CUDA 跳步检查。

### E：并行、缓存与运行边界

- [ ] **E01 对局／搜索／推理并行**：对局线程共享 evaluator、搜索线程统计、virtual loss、mutex pool、队列入队与组批、服务数、背压、关闭排空和失败唤醒；区分公共架构与具体执行时序差异。
- [ ] **E02 NN cache**：key 是否覆盖真实输入、模型、PDA、对称与 optimism 条件；根多对称绕缓存、raw 输出不可变、模型切换清理。不要将该缓存与已接入的搜索 eval cache 混为一谈。
- [ ] **E03 writer 与发布**：完整对局交接、首分片随机阈值、满片／尾片、bit packing 位序、原子可见边界、未完成局处理；对比数据内容，不要求原生 NPZ schema 相同。
- [ ] **E04 训练消费与恢复**：区分已消费 batch 和已成功 optimizer step；预取不应提前推进恢复游标，验证不推进训练状态。共有优化器／SWA 状态可以对照，额外 RNG／scaler／阶段恢复按本地协议验证。
- [ ] **E05 配置与评估落地**：最终 include 后配置、解析器缺省、selfplay 与 analysis／match 的覆盖、实际模型与精度、实际 visits／playouts／耗时；有实现但关闭的项不得算成该运行已经使用。

EtaZero 入口：[batcher.cpp](../cpp/src/inference/batcher.cpp)、[torch_backend.cpp](../cpp/src/inference/torch_backend.cpp)、[record.cpp](../cpp/src/selfplay/record.cpp)、[training.py](../python/etazero/training.py)、[runtime.py](../python/etazero/runtime.py)、[config.py](../python/etazero/config.py)。

KataGo 对应：[nneval.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/neuralnet/nneval.cpp)、[selfplaymanager.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/selfplaymanager.cpp)、[selfplay.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/command/selfplay.cpp)、[trainingwrite.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/dataio/trainingwrite.cpp)、[train.py](/home/sky/RL/SkyZero-lab/KataGo/python/train.py)、[setup.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp)。

现有复核入口：[persistence](../tests/reference/check_katago_persistence.py)、[recovery](../tests/test_recovery.py)、[checkpoints](../tests/test_checkpoints.py)、[GPU](../tests/test_gpu.py)、[model reload](../tests/test_inference_reload.py)、[schedule](../tests/test_schedule.py)。persistence 来源对照主要比较保存范围，不证明双方恢复协议相同；吞吐与等时间表现需要独立测量。

## 算法适用性速查

| 机制 | AZ＋PUCT | AZ＋Gumbel | MZ＋PUCT | MZ＋Gumbel |
|---|---|---|---|---|
| PUCT 根探索／pruning／LCB | 按配置核对 | 根不使用 | 按配置核对 | 根不使用 |
| 非根 PUCT／FPU | 使用 | 仅 nonroot=puct | 使用 | 仅 nonroot=puct |
| 真实规则终局、图共享、跨手树复用 | 按配置核对 | 有真实终局；图／复用禁用 | latent 树内不使用 | latent 树内不使用 |
| SubtreeValueBias | 按配置与棋类核对 | 按配置与棋类核对 | 禁用 | 禁用 |
| 根多对称与叶随机对称 | 按棋类核对 | 按棋类核对 | 一棵 latent 树固定朝向，根对称数为 1 | 同左 |
| uncertainty／noise pruning／value weighting | 按配置核对 | 按配置核对 | 按 latent 语义核对 | 按 latent 语义核对 |
| 采样、辅助目标与优化配方 | 按配置核对 | 改用 Gumbel policy target | 按 unroll／side mask 适配 | 同时适配 Gumbel target 与 unroll |

共用名称不自动意味着每个组合的启用条件相同。例如 Gumbel 不执行 PUCT 根 forced 配额，而 MuZero 没有真实终局叶的 uncertainty 特殊权重；应核对相应边界，不能把它们登记成遗漏。

## 审查顺序与结果记录

建议先核对数据意义再看性能，减少后续测试使用错误目标的风险：

1. **固定比较条件**：棋类、完整算法三元组、网络名称、来源配置、纯 W−L／普通和棋适配及启用项。
2. **目标链路**：搜索统计 → 实际落子／policy target → writer → shuffle／reader → loss／梯度，优先覆盖 S03–S05、D05–D07、N02–N03。
3. **搜索与采样调用链**：核对公式以外的调用位置、顺序和频率，覆盖 cheap、side、hint、fork、reanalysis、graph／reuse 等分支。
4. **训练时钟与恢复**：额度、单遍边界、batch 求和、优化器、Lookahead／SWA、AMP skip 和续训。
5. **真实执行条件**：CUDA 推理数值、并行计数、缓存与实际预算；需要判断效率时再采用用户确定的性能／棋力比较口径。

每项审查结果使用下列字段即可，不必把整套算法说明复制进记录：

| 项目编号 | 结论 | 适用配置／分支 | 来源与本地位置 | 验证证据／未覆盖边界 |
|---|---|---|---|---|
| 例如 S03 | 待审／已对齐／有意差异／待修复 | 精确到算法、棋类和开关 | 固定来源与当前实现 | 测试命令、结果位置及限制 |

“有意差异”需要对应本文已明确的适配或已有授权；审查中发现的新差异不能仅因实现已存在就自动接受。“已对齐”应限定到经过核对的分支和统计语义，不扩展成整套 KataGo 训练效果或性能已经复现。
