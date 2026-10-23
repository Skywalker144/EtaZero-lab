# EtaZero_V3.1 与 KataGo 通用机制对齐审查

目标：保留已经明确的多算法、多棋类、纯胜负目标、调度与执行适配，对齐本次审查列出的全部通用机制；**G02 配方选择按用户要求保留**。A01–F03 和 G01 共 26 项已接入 V3.1 的生产路径，下面列出实现位置、适用边界和验证入口。

此处的完成状态针对这 26 项机制。来源默认关闭的能力仍由配置开启；它不表示现有实验配置开启了所有功能，也不表示棋力、吞吐或两套完整引擎的所有并发轨迹已经证明相同。

## 基准与边界

- 当前目标：`EtaZero_V3.1`，版本创建提交 `15d5fc9`；实现改动位于工作区。V3 代码及 V3.1 的实验配置未修改。
- 固定来源：`/home/sky/RL/SkyZero-lab/KataGo`，提交 `d91ea855110dae533f0aada947b2b7d78cc8a4e1`。[来源清单](../reference_sources.json) 中 KataGo 核心条目现有 61 个文件，重新计算的 SHA-256 全部匹配。
- 原审查对象：V3，提交 `8726295fba463caa8be864a9257676ac76e072d7`，2026-10-23。下方保留其原始发现；[证据 JSON](katago-alignment-audit-evidence.json) 中的原始命令、数值和 V3 路径保持原样，V3.1 验证另记。
- 保留的差异见 [对照原则](katago-comparison.md)：四类算法／六种搜索组合、四类棋、纯 W−L 与普通和棋、棋类特征与开局、MuZero latent 与 Gumbel 根、replay ratio 同步轮次、固定模型、单卡 learner、自有数据格式与留出、LibTorch、严格并行预算、自有恢复与评估协议。

最直接的搜索对照对象是 AlphaZero＋PUCT。MuZero 共用适用的训练、预算和交互机制，不给 latent 节点伪造真实棋盘；Gumbel 保留自己的根搜索和改进策略目标。

## V3.1 逐项实现

下表中的“实现”包含配置、实际调用和必要状态；来源依据和发现时的复现输入可在本文后半部分按同一 ID 查找。用法见 [训练机制](algorithms.md#历史遮蔽与完整轨迹辅助目标)、[搜索控制](search-controls.md)、[交互协议](engine.md)。

| ID | 实现与核对重点 | 当前入口／验证 |
|---|---|---|
| A01 | PUCT 保留已分配 child 的严格大于扫描；新着先取最高先验，先验并列按动作顺序。覆盖 forced 配额的高分并列。 | [AZ 搜索](../cpp/src/search/search.cpp)、[MZ 搜索](../cpp/src/muzero/search.cpp)；原生搜索测试。 |
| A02 | 保存 rmax/dmax 及上次消费样本数，在轮入口和每 500 个已消费 batch 递推更新；恢复后沿用该状态。 | [BoundsSchedule](../python/etazero/brenorm.py)、[learner](../python/etazero/training.py)；边界／恢复测试，真实 BRN GPU 训练。 |
| A03 | 每个历史步以 2% 概率停止保留更早历史；Go 同步处理五步落点、征子回退和 globals，象棋按历史棋盘适配。 | [mask_history](../python/etazero/symmetry.py)；10,000 行直接对照固定来源 reader。 |
| A04 | 完整真实轨迹生成 t+8/t+32 的占据目标，超终局取终盘；两时间尺度、面积归一化、continuation gate、输出及梯度接通。 | [数据](../python/etazero/data.py)、[网络与 loss](../python/etazero/network.py)、[MZ 数据](../python/etazero/muzero/data.py)；手算与反向测试。 |
| A05 | 完整 W−L 序列生成 variance-time；40×softplus、梯度下限、Huber、scale 与完整结果 gate 接通。 | 同 A04；包含真实终局、side 和 MuZero 吸收态。 |
| B01 | 根对称剪枝检查规则、历史和动作限制；Go 使用来源规则，其他棋类使用各自有效对称。analysis/GTP 与 selfplay 各自保留来源入口缺省。 | [Game](../cpp/src/game/game.cpp)、[Go](../cpp/src/game/go.cpp)、[搜索](../cpp/src/search/search.cpp)；四棋类代表着测试。 |
| B02 | 搜索统计缓存与原始 NN cache 分离；实现写入门槛、刷新替换、首次探索与聚合衰减、新局清理。依赖真实棋盘 graph。 | [缓存与搜索](../cpp/src/search/search.cpp)；跨清树命中及 reset 测试。 |
| B03 | 根探索先验幂变换、半概率半正态 utility 奖励；只用于 PUCT 根。 | [root_explore_score](../cpp/src/search/search_math.cpp)；分布概率／均值检查。 |
| B04 | 剩余访问／时间估计进入 futile 剪枝，区分 graph child visits 和 edge visits；棋钟也使用追赶条件提前停止。 | [探索](../cpp/src/search/search_math.cpp)、[时间预算](../cpp/src/search/time_budget.cpp)；手算及原生测试。 |
| B05 | 指定方达到门槛后冻结子边权重比例，后续追随配额差；默认作用于服务方对手。两种非根策略均支持。 | [AZ](../cpp/src/search/search.cpp)、[MZ](../cpp/src/muzero/search.cpp)；冻结前后比例检查。 |
| B06 | 当前对局重复模式及外部语料 bonus 接入 utility；Go 用来源 LocalPatternHasher，其他棋类适配；包含玩家筛选、跳过标记、文件衰减和去重。 | [模式表](../cpp/src/search/pattern_bonus.cpp)；外部文件加载、utility／WDL 分离测试。自有 `.patterns` 格式。 |
| B07 | ignore-pre-root／ignore-all-history 仅改变 NN 输入，缓存身份随实际输入变化；完整规则历史保留。 | [历史特征](../cpp/include/etazero/history_features.h)、AZ/MZ 推理；按深度检查。 |
| C01 | Transformer attention bound 的子批采样、head/layer 归约和 0.001 正则系数进入 backward；无 attention 主干拒绝配置。 | [attention_penalty](../python/etazero/attention_penalty.py)、[Transformer](../python/etazero/transformer.py)；梯度、命名网络及真实 GPU 训练。 |
| C02 | NorMuon／Aurora 使用固定来源 kernel；接通 PP iterations/beta、参数角色和恢复状态。 | [Muon wrapper](../python/etazero/muon.py)；恢复后下一次更新逐值比较，GPU 小训练。 |
| C03 | Muon auxiliary Adam LR、WD floor、NS 次数与 Polar Express 接入分组和 checkpoint；floor 基准随状态保存。 | [优化器](../python/etazero/optimization.py)、[Muon](../python/etazero/muon.py)；分组、floor、两种迭代方法与恢复测试。 |
| C04 | auto／auto2、样本分段、平方根插值 cyclic LR；映射已消费样本和发布轮次，保存周期状态。普通优化消融继续固定 LR。 | [lr_schedule](../python/etazero/lr_schedule.py)；阈值与周期手算、来源时钟对照。 |
| C05 | value、三个 TD、variance-time、main/intermediate head scale 接入真实 loss；默认配方和显式 override 分开。 | [loss](../python/etazero/network.py)、[命名适配](../python/etazero/katago_network.py)；AZ/MZ 梯度测试。 |
| D01 | scale-data-weight 在 surprise 权重之后、float32 随机取整之前统一作用于主局和 side。 | [writer](../cpp/src/selfplay/record.cpp)、[selfplay](../cpp/src/commands/main.cpp)；原生与数据回归。 |
| D02 | 无 hint 外部起点：加载概率、抽样权重、手数衰减、排除／去重、独立 policy init，保留完整合法前缀。 | [起点加载](../cpp/src/selfplay/sampling.cpp)；实际生成对局并读取训练视图。 |
| D03 | training-weight 独立于抽样频率；起点→整局→fork／side→writer→reader→loss 全链路传递。 | [数据视图](../python/etazero/data.py)、[reader](../python/etazero/reader.py)；端到端权重检查。 |
| E01 | min-playouts-per-thread 按本次剩余新访问限制活动线程，六种搜索组合共用。 | [AZ](../cpp/src/search/search.cpp)、[Gumbel](../cpp/src/search/gumbel_search.cpp)、[MZ](../cpp/src/muzero/search.cpp)；并行搜索回归。 |
| E02 | 图节点分片和 NN cache 锁池独立可配；AZ/MZ initial cache 均接通，−1 禁用 NN cache。 | [SearchSettings](../cpp/include/etazero/search.h)、[AZ batcher](../cpp/src/inference/batcher.cpp)、[MZ batcher](../cpp/src/muzero/batcher.cpp)。 |
| F01 | 深度 avoid/allow、root focus 和运行中 set_focus 接通；focus 子树更新、根以上不计权重，严格预算预约退款，实际落子清除 focus。 | [session](../cpp/src/commands/session.cpp)、各搜索；原生六组合、JSON/GTP 测试。 |
| F02 | absolute／Fischer／Canadian／byo-yomi 时钟、lag、阶段与明显招法调时、复用折算、独立 ponder caps、停止及 GTP/JSON 入口。 | [TimeControls](../cpp/src/search/timecontrols.cpp)、[时间预算](../cpp/src/search/time_budget.cpp)、session；手算、真实模型命令与 ponder 测试。 |
| F03 | 第二策略模型实际加载；根／双方非根 weightless、weightful 探索，独立 cpuct、PIKL、最终混合、分析及棋钟共用选择策略。 | [工厂](../cpp/src/commands/main.cpp)、AZ/MZ 搜索；CPU/CUDA 双模型加载与搜索，零权重子边回退测试。 |
| G01 | Python 和 C++ 接受来源边界：cpuct=0、lookback=0、reduced-min=1、64 位预算至 2^50；限制 fork 概率和 root temperature 合法域。 | [配置](../python/etazero/config.py)、[预算](../cpp/src/selfplay/search_limits.cpp)、原生命令；边界／大预算取消测试。 |
| G02 | **保留差异，不改实验配方。** 未把任何来源训练 profile 选为统一基线，也未改 full/cheap visits、温度、SVB 等实验取值。 | `configs/` 与创建 V3.1 时一致。 |

## 适配细节与使用边界

- 历史遮蔽只用于网络可见输入。Go 的规则历史、象棋的重复局面裁决和真实后续轨迹不删除。Gomoku／Hex 没有相同历史通道时不伪造遮蔽任务。
- Future-position 和 variance-time 使用完整实际轨迹；side 或明确不用 outcome 的重分析行关闭这两项 gate。MuZero 对真实展开位置取目标，吸收态保留最终棋盘及零 variance-time。
- Go 重复模式采用来源哈希；其他棋类模式、外部起点／语料格式属于多棋类和自有格式适配。MuZero 明确拒绝真实棋盘 SVB、graph eval cache 和重复模式配置。其 HumanSL 非根探索要求第二个 MuZero latent 模型，第二个 AlphaZero 模型只可用于真实根。
- Gumbel 根保留 sequential halving，不执行 PUCT 根 forced quota、wide-root-noise 或 visit-cap 配额。共用动作限制、focus、时间预算和第二策略的适用部分；非根仍按所选 PUCT／Gumbel 算法运行。
- LR cycle 以 EtaZero 发布轮次映射来源周期，不更改 replay ratio 或每轮预算。BRN 的边界递推状态与普通优化器状态一起恢复。
- 来源允许 lookback=0、PIKL lambda=0、endgame decay=0，但个别来源计算在端点可出现断言或 0/0；本地分别采用合法概率裁剪或明确的有限极限。这些端点处理不宣称逐位重现来源未定义数值。
- 根对称剪枝减少实际搜索候选；分析协议使用 EtaZero 的动作／PV 格式。没有宣称原样复制 KataGo 所有报告字段、SGF 协议或人类等级 metadata。

## V3.1 验证

验证命令和摘要记录于 [证据 JSON](katago-alignment-audit-evidence.json) 的 `v3_1_implementation`，与原始 V3 证据分开。关键新增检查是 [原生机制测试](../cpp/tests/alignment_search_test.cpp)、[MuZero 搜索测试](../cpp/tests/muzero_search_test.cpp) 和 [训练链路测试](../tests/test_alignment_training.py)。

- C++ 构建成功，原生 CTest **17/17 通过**。
- 完整快速 Python 回归：**789 passed、204 skipped**。跳过项没有记为通过；真实 CUDA 检查另行执行。
- 最后搜索边界调整后，交互／Gumbel／新增机制相关检查 **28 passed、28 skipped**；训练／辅助目标来源梯度检查 **61 passed、4 skipped**，优化器最终导入调整复验 **40 passed**。这些是交叠的针对性检查，不相加当成独立用例总数。
- 真实 RTX 5080 上，AlphaZero、MuZero BF16 及四个 Gumbel 组合的小训练、导出、恢复和相关分析／比赛检查通过；NorMuon BRN、Aurora Transformer 的可选配方小训练 **2/2 通过**。最后的第二策略模型／棋钟／约束／ponder 检查 **4/4 通过**，其中 2 个 CPU、2 个 CUDA。
- 固定来源对照：搜索 logit／noise pruning／uncertainty 分别 48／240／504 例；Lookahead／SWA／LR 时钟 3,696 例；Muon 4 次更新和 624 组 LR/WD；标量分组公式 31,104＋36,288 例、范数历史 5,760 例。历史遮蔽直接对照 10,000 行；future-position／variance-time 直接执行来源函数并比较 loss 和梯度。
- `git diff --check` 通过；V3 和 V3.1 `configs/` 没有改动。来源核心 61 个文件及 Muon 3 个登记文件的校验值匹配。

这些结果验证机制、数值、梯度、调用链及小规模实际训练；没有开展正式棋力实验、等时间性能比较、全部网络×棋类×算法的完整矩阵，或两套完整引擎的并发逐步轨迹对照。

## V3 原始审查记录

**以下记录的是修改前 V3 的发现和当时的验证结果。** “缺少／不同／未实现”描述 V3 基线；当前 V3.1 状态以上表为准。历史 Eta 源码链接指回 V3，避免把旧行号误指到修改后的实现。原始测试不是 V3.1 的复验结果。

### A：已经实现但行为不同，或基础训练目标缺失

#### A01：PUCT 并列选点改成了均匀随机

**触发**：两个或更多动作的探索分数完全相同，或未展开动作的最大先验并列。低精度输出、对称局面、均匀网络以及 forced-playout 的统一高分均可能产生并列。

KataGo 在 [已有 child 扫描](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchexplorehelpers.cpp:601) 使用严格 `>` 更新最大值；[未展开动作扫描](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchexplorehelpers.cpp:695) 同样保留遇到的第一个最大先验，最后只有新动作分数严格更高时才替换已有 child。EtaZero 在 [候选收集与选择](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/src/search/search.cpp:440) 收集全部并列动作，再排序、去重并均匀抽样（第 494 行）。

**影响**：选择分布、访问分配及训练 policy target 可不同。这不是“同分布、不同 RNG 序列”的已保留差异。

**复现**：直接链接当前 `libetazero_core.a`，单线程、两个合法动作、相同 logits、无噪声、一次根边模拟；100 个种子中首次访问动作 0／1 的次数为 **43／57**。来源扫描逻辑在这个条件下选择首个动作。这里运行了 Eta 搜索，来源一侧依据实际控制流；没有声称运行了两套完整引擎对局。

**对齐要求**：对应来源的已有 child 顺序、未展开动作扫描顺序，以及已有／新 child 并列优先级；单独覆盖 forced 配额并列。不能只改最终落子的零温度 tie-break。

#### A02：batch renorm 的公式相近，调用频率不同

**触发**：选择 `brn/fbrn` 命名网络，并设置会变化的 rmax/dmax 目标。默认使用其他 norm 的运行不受此项影响。

来源 [更新函数](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:1321) 按距上次更新的已消费样本数递推保存的 rmax/dmax；在 [epoch 入口](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:1639) 和 [每 500 个 batch](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:1877) 更新。Eta [prepare_brenorm](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/katago_network.py:119) 从累计样本数计算闭式值，并在 [每批 forward 前](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/training.py:353) 设置。

**复现**：batch=128、adjustment_scale=1,000,000、target_rmax=3、target_dmax=5；从新 epoch 开始：

| 已消费样本数 | 来源 rmax / dmax | Eta rmax / dmax |
|---:|---|---|
| 0 | 1 / 0 | 1 / 0 |
| 128 | 1 / 0 | 1.000255984 / 0.000639959 |
| 63,872 | 1 / 0 | 1.123749856 / 0.309374641 |
| 64,000，来源完成第 500 批更新 | 1.123990001 / 0.309975002 | 1.123990001 / 0.309975002 |

闭式解只在相同更新时点等价；其间训练 forward 的裁剪边界不同。现有网络测试固定 renorm 边界，无法发现此项。对齐需要落实更新时点、所用样本时钟和保存的边界状态，不能只校验指数公式。Eta 未配置 brenorm section 时还会固定为 `(1,0)`，不能将这种运行描述成来源有目标参数的调度。

#### A03：缺少训练输入的随机历史遮蔽

来源 reader 在 [每批输入处理](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/data_processing_pytorch.py:151) 调用 [apply_history_matrices](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/data_processing_pytorch.py:299)：每个历史步有 2% 概率停止保留更早历史，同时处理最近五步落点、历史征子特征回退和前五个 global history 特征。至少遮蔽一段历史的概率为 `1−0.98^5≈9.61%`。这是该 reader 的训练／验证输入处理，不是 D4。

Eta 的 Go 输入使用相应历史特征，但 [训练增强](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/training.py:349) 只进入 [坐标对称变换](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/symmetry.py:54)，没有历史遮蔽。Gomoku 的禁手特征 dropout 也不是此机制。

**复现**：对 10,000 行具有五步历史的合成输入，直接执行来源函数，固定 seed=853；来源遮蔽 **926 行**，Eta 同一输入经 symmetry=0 的生产增强后遮蔽 **0 行**。按历史步分别为 `[212,404,573,758,926]`。

**对齐要求**：Go 可直接核对来源的概率和矩阵；其他具有历史输入的棋类须按其特征语义适配。不可直接照抄 Go 通道号，更不可删除用于规则合法性的真实历史。

#### A04：缺少 future-position 辅助监督

来源 writer [生成 t+8、t+32 棋盘目标](/home/sky/RL/SkyZero-lab/KataGo/cpp/dataio/trainingwrite.cpp:777)，超过终局取最终棋盘，按当前玩家记录己方占据 `+1`、对方 `−1`、空点 `0`。这与终局 territory ownership 是不同目标。

来源 [futurepos loss](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/metrics_pytorch.py:178) 使用 tanh 平方误差、两时间尺度权重 `[1,0.25]`、有效面积平方根归一化及总系数 `0.25`，并有独立 continuation gate。

Eta [训练字段](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/data.py:345)、[网络输出筛选](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/katago_network.py:92)、[loss 汇总](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/network.py:411) 均未接入该任务。命名网络内部保留了来源 head，也没有获得这项监督。

**影响与要求**：缺少会回传到 backbone 的辅助学习信号。应从完整真实轨迹生成目标，适配棋盘表示及侧局面／重分析／终局 mask。MuZero unroll 不能自动等同于这项明确的棋盘预测 loss。

#### A05：缺少 variance-time 辅助监督

来源 [目标生成](/home/sky/RL/SkyZero-lab/KataGo/cpp/dataio/trainingwrite.cpp:603) 为：

```text
variance_time(t) = Σ(i>t) (i−t) × ((win_i−loss_i) − (win_(i−1)−loss_(i−1)))²
```

来源 [loss](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/metrics_pytorch.py:300) 使用 `Huber(pred,target+1e−5,delta=50) × 0.0003`，默认 `variance_time_loss_scale=1`。来源使用完整局面结果 gate；它的目标公式只依赖胜负预测，不需要 score。

Eta 与 A04 相同的字段／输出筛选／loss 链路没有此目标。`shortterm_value_error`、TD value 和 uncertainty weighting 都不是 variance-time。

**对齐要求**：保留完整价值序列及时间索引，正确处理 side／无实际后续的重分析 gate，并接入输出与梯度；不能因来源 gate 的名称包含 ownership 就把整个任务判为围棋专用。

### B：搜索通用能力遗漏

以下均在固定来源的参数或调用链中可用。Eta [SearchSettings](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/include/etazero/search.h:9)、[配置字段](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/config.py:80) 和 [搜索入口](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/src/search/search.cpp:584) 没有对应能力；这组并非已有开关被 baseline 关掉。

| ID | 缺失机制 | 来源与作用 | 对齐时的关键区别 |
|---|---|---|---|
| B01 | `rootSymmetryPruning` | [setup:662](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp:662)、[search:787](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.cpp:787)：根处只搜索一个对称等价动作。来源 analysis/GTP 默认开启。 | 根 NN 多对称平均和训练 D4 不提供这个剪枝。等价关系必须遵守棋类规则、历史和动作限制。 |
| B02 | `useEvalCache / evalCacheMinVisits` | [cache 生命周期](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.cpp:763)、[未展开动作使用缓存](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchexplorehelpers.cpp:643)：利用以往搜索统计替代部分首次探索 FPU，依赖图搜索。来源默认关闭。 | 原始 NN 缓存、当前树复用、图节点共享、SVB 都不是该缓存。需覆盖写入门槛、刷新和清理。 |
| B03 | `wideRootNoise` | [探索加成](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchexplorehelpers.cpp:89)：扩大根探索的额外随机机制。 | 与 Dirichlet、root temperature、Gumbel 分别是不同机制。 |
| B04 | `futileVisitsThreshold` | [剩余预算剪枝](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchexplorehelpers.cpp:157)、[配置](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp:835)：剪掉剩余时间／访问量内难以追赶的动作。默认关闭。 | policy target pruning 发生在生成目标时，不能代替搜索过程中预算剪枝。 |
| B05 | `visitCapContempt / Pla` | [参数语义](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchparams.h:104)、[分配实现](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.cpp:1175)：对指定方达到门槛的节点冻结 child 分布比例，后续访问追随该快照。默认关闭。 | PDA 改双方预算，不等于这种节点内的对手建模。 |
| B06 | `avoidRepeatedPatternUtility` 与外部 pattern bonus | [局内模式表](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.cpp:768)、[外部模式载入](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp:1089)、[utility 修正](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.cpp:1137)。默认不启用惩罚。 | SVB 是学习搜索误差的统计修正；这里是主动避免重复局部模式。模式键可按棋类适配，机制本身不依赖计目。 |
| B07 | `ignorePreRootHistory / ignoreAllHistory` | [NN 输入设置](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchnnhelpers.cpp:89)、[配置](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp:779)：控制推理可见的历史深度。 | 与 A03 的训练随机遮蔽不同。对齐包含缓存身份及移根刷新，不能改变真实合法性历史。 |

### C：learner 可选能力未接通

Eta [训练／优化字段](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/config.py:98) 与实际 learner 调用链缺少下列控制。固定来源中存在对应生产入口，不能用“基础 SGD/AdamW/Muon 数值通过”覆盖这些分支。

| ID | 缺口 | 来源与 Eta 当前情况 | 影响／验收重点 |
|---|---|---|---|
| C01 | attention logit bound penalty | 来源 [参数](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:152)、[backward 前加损失](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:1740)；Eta wrapper 不设置 cap/batch fraction，也不把该 penalty 加入总 loss。 | 通用 Transformer 正则化。来源默认 cap=None，故不是默认网络一运行就错。需核对计算子批、head/layer 归约及系数。 |
| C02 | NorMuon、Aurora | 来源 [选项](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:113) 和 [构造](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:951) 可用；Eta optimizer.kind 只有 sgd/adamw/muon。 | vendored 文件含代码不代表生产可选。包括 Aurora 的 PP iterations/beta。 |
| C03 | Muon 的非默认控制 | 来源 `muon-adam-lr-factor`、`wd-floor-frac`、`ns-steps`、`use-polar-express`；见 [选项](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:98)、[WD floor 设置](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:858)。Eta [Muon wrapper](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/muon.py:21) 只传分组和固定 adjust_lr_fn。 | 默认 kernel 可一致，辅助 Adam 的独立 LR、衰减下界、迭代方法与次数仍缺失。需接通配置、分组和恢复；不能只增加字段。 |
| C04 | 自动、分段及 cyclic LR 配方 | 来源 [lr-scale-auto/auto2、lr-schedule、lr-cycle](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:88)；Eta [group_settings](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/optimization.py:55) 只有固定 lr_scale 与 warmup。 | replay ratio 控制产样／训练额度，不提供这些 LR 变化规律。应映射样本时钟和发布周期，明确边界更新时点。 |
| C05 | 可配置 value／TD／主副 head loss scales | 来源 [value-loss-scale、td-value-loss-scales、main/intermediate-loss-scale](/home/sky/RL/SkyZero-lab/KataGo/python/train.py:158)。Eta [value loss](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/network.py:345) 和 [TD loss](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/network.py:373) 固定 `.72`，命名网络的主／中间 head 比例按来源缺省自动选定，但没有相同 override。 | 默认值可以对齐，完整配方控制尚不齐。soft-policy scale 与 disable-optimistic 已有，不应重复列为缺失。variance-time scale 随 A05 一起补齐。 |

### D：数据生产与权重的缺口

| ID | 缺口 | 来源与 Eta 当前情况 | 为什么不由已有差异覆盖 |
|---|---|---|---|
| D01 | `scaleDataWeight` | 来源 [配置](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/playsettings.cpp:114)、[最终随机取整前统一缩放](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/play.cpp:2269) 同时作用于主轨迹和 side。Eta [apply_training_weights](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/src/selfplay/record.cpp:74) 没有此全局尺度。 | 控制写出数据的频率／重复行数；replay ratio 控制 learner 消费额度，作用位置和统计意义不同。来源默认 1 时差异不触发。 |
| D02 | 无 hint 的外部起始局面语料抽样 | 来源 [startPosesFromSgfDir 一组设置](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/play.cpp:244)、[起点选择](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/play.cpp:533) 支持局面抽样、加载概率、手数权重、排除／去重、起点后的独立 policy-init 配方。Eta [外部 loader](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/src/selfplay/sampling.cpp:28) 要求合法 hint 动作，未提供普通外部起点数据集的等价链路。 | 保留各棋类标准／平衡开局和自有文件格式，可以适配该功能；强制 hint 的外部起点不能冒充无 hint 起点。无需要求所有棋类使用 SGF。 |
| D03 | 起始局面／整局的独立 loss 权重 `trainingWeight` | 来源 [起点传递](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/play.cpp:592)、[writer 传入行权重](/home/sky/RL/SkyZero-lab/KataGo/cpp/dataio/trainingwrite.cpp:1214)、[learner global_weight](/home/sky/RL/SkyZero-lab/KataGo/python/katago/train/metrics_pytorch.py:620)。Eta 的 hint `weight` 只用于抽样概率；训练视图没有外部整局 loss 权重。 | 不是 surprise/PCR 行频率，也不是 D01。仅改变被抽中的概率不等于来源抽中之后仍按 trainingWeight 缩放梯度；还要保留 fork 继承。 |

`network.py` 的内部 row_weight 参数不构成 D03 的生产支持：需要起点→局→writer→shuffle→reader→loss 全链路传递。来源默认整局权重为 1 时，普通自对弈不受这项遗漏影响。

### E：并行执行的通用控制缺口

| ID | 缺口 | 来源与 Eta 当前情况 | 判定范围 |
|---|---|---|---|
| E01 | `minPlayoutsPerThread` | 来源 [按剩余新 playout 限制线程数](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.cpp:574)，analysis/GTP 缺省为 8；Eta 按配置固定 worker 数。 | 小预算下的并发度与执行开销不同。这不是已保留的“严格预约预算、不允许超额”。没有据此推断哪方吞吐更高。 |
| E02 | 图节点表与 NN cache mutex pool 的分片控制 | 来源 [nodeTableShardsPowerOfTwo](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp:804) 缺省 16；[NN mutex pool](/home/sky/RL/SkyZero-lab/KataGo/cpp/neuralnet/nneval.cpp:1508) 可独立配置。Eta [node_table_](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/include/etazero/search.h:205) 和 [cache_locks_](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/include/etazero/inference.h:81) 固定 64 片。 | 属于并发结构／可调能力差异，未证明输出有误或吞吐下降。SVB 分片已经可配，不能拿它代替图表和 NN cache 的分片。 |

LibTorch 与来源专用 NN kernel 的差异已保留；上述控制并不依赖必须使用哪种 NN kernel，因此仍需单列。

### F：交互引擎通用能力缺口

这些主要影响 analysis/GTP／人机对弈，而非固定预算自对弈训练。完整对齐清单仍需保留，不能让“训练能跑”代表引擎能力一致。

| ID | 缺口 | 来源与 Eta 当前情况 |
|---|---|---|
| F01 | 按深度限制／允许动作、root focus | 来源 [setAvoidMoveUntilByLoc、setRootFocus](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.h:267) 及 [选点调用](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchexplorehelpers.cpp:418)。Eta [交互命令](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/src/commands/session.cpp:227) 没有等价入口；maxmoves 只限制报告项，不限制搜索动作。hint 是另一种根引导，不能代替 weightless focus。 |
| F02 | 棋钟管理与专门的 pondering 预算 | 来源 [TimeControls／搜索限时](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/search.cpp:583)、[时间预算参数](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchparams.h:150) 包括 pondering 上限、lagBuffer、树复用折算、局面阶段和明显招法的时间调整。Eta 有单次 max_time、连续分析和停止，但没有这套棋钟／pondering 管理，GTP 也无 time_settings/time_left。固定预算比赛协议仍可保留。 |
| F03 | 第二策略网络／Human-SL 对手建模 | 来源 [humanSL 参数](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchparams.h:174) 和 [第二网络推理](/home/sky/RL/SkyZero-lab/KataGo/cpp/search/searchnnhelpers.cpp:104) 可按不同概率作 weightless／weightful 探索及最终落子混合。Eta 没有第二策略网络入口。该搜索机制不依赖围棋计目；来源人类等级 metadata 的实际内容则需按已有零 metadata／多棋类边界单独适配。不能声称仅有 meta 架构就支持这套功能。 |

F03 不等于要求立即引入人类数据集或改变四类核心算法；它记录固定来源中一个未移植的可选通用搜索组件。是否有可用的第二策略模型，是后续运行该能力的前提。

### G：配置边界与配方

#### G01：已有机制的合法参数范围不同

配置解析也会删掉来源支持的边界行为，不能仅检查字段是否存在。已直接调用 Eta `load_config/validate` 验证以下四种设置被拒绝；来源允许范围来自实际 loader：

| 参数 | 固定来源 | Eta 当前行为 |
|---|---|---|
| `cpuctExploration / c_puct` | [setup:550](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp:550) 允许 0 | [配置](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/config.py:311)、[C++ 构造](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/src/search/search.cpp:164) 要求大于 0；不能表达关闭常数探索项、只保留其他项的设置。 |
| `reduceVisitsThresholdLookback` | [playsettings:108](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/playsettings.cpp:108) 允许 0，执行循环也处理空窗口 | [config:442](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/config.py:442) 要求至少 1。 |
| `reducedVisitsMin` | [playsettings:109](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/playsettings.cpp:109) 允许 1 | [config:443](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/config.py:443) 要求至少 2，并要求不超过 full visits。 |
| `maxVisits / maxPlayouts` | [setup:493](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp:493) 使用 int64，配置可达 `2^50` | [SearchSettings](/home/sky/RL/EtaZero-lab/EtaZero_V3/cpp/include/etazero/search.h:9) 使用 int；Python cap 为 `2^31−1`。这是表示范围差异，不只是严格限制并发超额。 |

另有反向差异：来源 early/game fork 概率配置上限为 0.5，Eta 接受到 1；来源 root-policy temperature 限在 `[0.01,100]`，Eta 仅检查正值。依据：[来源 fork loader](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/playsettings.cpp:97)、[来源温度 loader](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp:647)、[Eta 校验](/home/sky/RL/EtaZero-lab/EtaZero_V3/python/etazero/config.py:57)。这些不是运行数值偏差的证明，但说明当前配置接口并不具有相同合法域。

对齐时应核对 Python 与 C++ 两层，区分来源合法边界、棋类适配所需约束和额外限制；不能只把 Python 条件放宽而让底层溢出或拒绝。

#### G02：配方尚未指定统一的来源 profile

**机制对齐与配置对齐应分别验收。** 本次没有擅自把某套 KataGo 官方／本地训练配置选为用户实验基线。当前 Eta baseline 与来源 `selfplay8mainb18.cfg` 至少有以下不同：

| 参数 | Eta baseline/selfplay.cfg | KataGo selfplay8mainb18.cfg |
|---|---:|---:|
| full visits | 375 | 2000 |
| cheap visits | 75 | 350 |
| chosen-move temperature halflife | 15 | 19 |
| subtree-value-bias factor | 0.45 | 0.30 |
| subtree-value-bias weight exponent | 0.85 | 0.8 |

依据：[Eta 配置](/home/sky/RL/EtaZero-lab/EtaZero_V3/configs/baseline/selfplay.cfg)、[来源配置](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg)。来源解析器缺省与这份配置也不总相同。例如 Eta 的 SVB 数值与来源解析器常用缺省相同，不能仅因与 mainb18 不同就判定公式有 bug。

本表是已确认的例子，**不是全部配置差异表**。在确定 selfplay／analysis／match 各自来源 profile 后，仍需对最终 include、override 后的有效参数逐项核对，包括开关、预算、温度、优化器、数据窗口和精度。用户可调参数应保持可调；默认关闭不等于机制缺失。

### 已检查且未发现新增差异的范围

原 [27 项检查清单](katago-comparison.md#共有机制的逐项审查) 不整体勾选；下表记录本次覆盖和例外。

| 原清单 | 本次证据／结果 | 尚不能推出的结论 |
|---|---|---|
| S01–S05：PUCT/FPU、先验、forced/pruning、LCB、统计聚合 | 原生测试及来源公式对照通过；logit 混合 48 例、noise pruning 240 例、uncertainty 504 例。发现 A01。 | 不能说所有选点路径和非默认来源能力都一致；B 项未实现。 |
| S06–S07：SVB、graph | 通用 SVB／graph、Go SVB／GraphHash 来源检查通过，包含源码独立计算和本地测试。 | 未做完整双引擎并发轨迹逐步比较；MuZero/Gumbel 的明确边界仍保留。 |
| S08：预算与停止 | 原生检查覆盖访问／playout、取消和复用；严格预约上限是保留差异。 | E01、F02 未对齐，不能宣称所有时间管理相同。 |
| D01–D06：PCR、reduce visits、PDA、fork/side/reanalysis、surprise、量化及 Q | sampling、weights、forks、Q 来源对照通过；对应本地测试通过。 | D01–D03 本报告中的新增能力缺口，以及 A04/A05 辅助目标不被这些测试覆盖。 |
| D07：replay/shuffle/reader | 窗口 2,592 例、分组阈值 30 例、输出计划 324 例、文件顺序 264 例、来源 reader 68 例通过。 | reader 行序／批边界一致不证明输入增强一致，A03 正是反例。 |
| N01：网络结构、初始化和参数分组 | 全部 **848,640 个名称／配置定义**一致；3 个命名架构 CPU 前向／梯度一致；原有 plain/nbt/transformer 对照在指定容差内通过，初始化 170 例。 | 没有运行 848,640 个网络。命名架构只抽查 3 个完整前后向；只比较筛选出的保留输出，不能据此认可 A04/A05 的删除。 |
| N02–N03：policy/value/TD/error/Q | 本地 auxiliary、WDL、Q、multigame 测试通过，来源映射及纯 W−L 退化式已检查。 | 本次未运行所有网络／棋类／算法 loss 的完整来源端到端梯度矩阵；遗漏的目标仍欠账。 |
| N04–N05：优化分组、Lookahead/SWA/norm 时钟 | 分组公式 31,104 例、fixup 36,288 例、norm 历史 5,760 例、时钟 3,696 例及默认 Muon 对照通过。 | 比较固定配方／已提供范数和成功 step 序列；不覆盖 C 项全部选项，不覆盖 A02 的 renorm 调用频率，也不覆盖整个来源训练桶。 |
| N06–N07、E01–E05：执行、恢复、配置落地等 | CPU 原生与 Python 检查、persistence 字段对照、局部多棋类短训练测试通过。 | CUDA/AMP/Graph、真实高并发、完整崩溃恢复、等时间性能与棋力没有在本次重新验收。恢复协议已有明确适配，不能宣称来源完全相同。 |

本次命令结果：

- V3 构建成功；`ctest --test-dir build --output-on-failure -j 2`：**16/16 通过**。
- 15 个来源检查入口全部退出 0：sampling、sampling_weights、forks、search_corrections、graph、subtree_value_bias、go_bias、go_graph、clocks、replay、q、muon、persistence、formulas、network。
- `check_katago_architectures.py --device cpu --architecture b5c192nbt-fson-mish b10c128-brn-mish b10c128-fson-mish-ih --all-configs --backward` 通过。
- 8 个 Python 测试文件：optimization、auxiliary、wdl_targets、replay、qvalues、schedule、multigame、katago_architectures：**266 passed，8 skipped**。存在 PyTorch deprecated/trace 警告；CUDA 相关测试未用 CPU 结果代替通过。
- A01–A03 单独复现，结果如上；临时程序只在 `/tmp/etazero-v3-katago-audit-20261023`，未加入实现或正式测试目录。

### 没有计为通用机制遗漏的项目

- **围棋专用 score／ownership／seki 目标、计目效用、随机贴目／让子补偿、passing/dame/anti-mirror 等围棋行为**：按已保留边界处理；不要求复制这些能力。A04/A05 的定义不同，不能一起排除。
- **MuZero 真实终局／图共享、Gumbel 根 PUCT 配额、跨阶段异步、局内换网、DDP、原生 NN kernel、来源文件格式和原样恢复协议**：已经明确保留的差异，不在本次建议中偷偷恢复。
- **旧的 buggy LCB 分支**：Eta 对齐修正后的语义；不要求重现来源历史 bug。
- **`recordTreePositions` / `noResolveTargetWeights`**：来源确有字段和实现分支，但当前生产 loader 未接通前者，后者赋值主要在测试。没有将其冒充来源当前自对弈默认启用的遗漏。若以后要求连休眠内部 API 都复制，应另列范围。
- **selfplay resignation**：不能把来源 match/gatekeeper 的配置误写成当前 selfplay 的默认机制；Eta 完整对局评估是已保留协议。

