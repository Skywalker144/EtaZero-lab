# 算法与搜索设计

本文描述当前 AlphaZero 的算法语义和后续算法边界。运行架构、数据保存与验收见 [运行实现](implementation.md)。AlphaZero / PUCT 已实现，MuZero / Gumbel 仍属后续设计，网络规模、训练参数与正式实验方案由用户确定。

## 算法与搜索组合

`algorithm` 决定搜索使用的状态转移、网络结构和训练方式；`root_search_algo` 与 `nonroot_search_algo` 分别决定根节点和非根节点的搜索策略。算法身份需要记录完整三元组，不能仅用一个“Gumbel”标签概括所有配置。

| algorithm | root_search_algo | nonroot_search_algo | 目标能力 | 第一轮支持目标 |
|---|---|---|---|---|
| alphazero | puct | puct | AlphaZero | 已实现 |
| alphazero | gumbel | puct | Gumbel 根搜索 + PUCT 非根搜索 | 后续实现 |
| alphazero | gumbel | gumbel | Gumbel 根搜索 + 改进策略非根搜索 | 后续实现 |
| muzero | puct | puct | MuZero | 后续实现 |
| muzero | gumbel | puct | Gumbel 根搜索 + PUCT 非根搜索 | 后续实现 |
| muzero | gumbel | gumbel | Gumbel 根搜索 + 改进策略非根搜索 | 后续实现 |
| 任一 | puct | gumbel | 本项目不允许的组合 | 始终拒绝 |

最后一行是用户确定的项目组合约束，不表述为数学上不可能。合法但尚未实现的组合，应报告未实现能力；非法组合应报告组合错误。两者均在启动 worker、分配大块显存或创建训练产物前拦截。

配置格式的定义位置见 [配置组织](implementation.md#配置组织)。选择 Gumbel 根搜索同时选择其根动作决策及训练策略目标，不能只把选点公式换掉而仍隐式沿用 AlphaZero 的目标生成。

## 第一轮 AlphaZero

### 来源与适配边界

采用 AlphaZero 的策略价值网络、真实规则树搜索和自对弈监督闭环；网络预测落子概率与当前玩家的期望终局结果，训练结合策略交叉熵、价值误差与正则化。模型持续更新并供自对弈使用，固定模型评估独立运行，不默认加入“胜过旧模型才发布”的门控。[AlphaZero 原文，算法描述及式 1](https://arxiv.org/pdf/1712.01815)

本项目将其适配到三种五子棋规则和可配置硬件规模。以下明确的是 EtaZero 的实现约定，不声称复现原论文的棋类任务、网络规模或实验预算。KataGo 的 FPU reduction、LCB、强制探索与目标剪枝、playout cap randomization、辅助头和开局生成不是这些约定的隐含组成。

### 游戏状态与棋规

真实状态包括棋盘、当前行棋方、规则、棋盘尺寸和终局状态。无提子、无 pass；坐标落在有效棋盘内的空点才可提交，终局后拒绝继续走子。对局从空棋盘开始，第一轮不引入比赛交换开局或开局库。

| 规则 | 胜负语义 |
|---|---|
| Freestyle | 任一方连成五子或更长即胜 |
| Standard | 任一方恰好五子即胜；长连本身不判胜 |
| Renju | 白方五子或更长胜；黑方按参考棋规处理恰好五子与长连、双四、双三 |
| 满盘 | 先判本手胜负；无胜负时为和棋 |

Renju 沿用现有 MuZero / SkyZero 的棋盘局部规则，不扩展为完整比赛开局协议。参考中的禁手点允许实际落子，落下后判黑负，不在动作 mask 中提前删除；恰好五子与其他方向形状同时出现时按来源中的优先级判定。以 [MuZero Game](../../../MuZero/MuZero_V2/cpp/include/muzero/game.h)、[RenjuAnalyzer](../../../MuZero/MuZero_V2/cpp/src/rules.cpp) 和 [SkyZero Renju](/home/sky/RL/SkyZero/SkyZero_V8.1/katago/cpp/game/renju.cpp) 核对具体规则。

移植棋规要覆盖递归活三、边界与多方向交叉等情况，不能用局部字符串或简单形状计数代替来源语义。动作可提交性、落子后禁手判负和给网络的输入特征分别定义，不能混为一个“合法性”开关。

### 观测、动作与网络

当前实现以单模型固定画布承载配置中的棋盘尺寸。画布边长由该模型支持的最大尺寸决定，网络动作编号统一使用画布上的行优先编号；实际棋盘编号只在外部接口做明确转换，落盘时不得混用。

基础观测使用当前玩家视角的己方棋子、对方棋子、黑方行棋、有效棋盘、Standard 和 Renju 六个平面；规则与颜色信息用于消除不同棋规和执棋方的歧义。这是本方案的输入设计，未直接照搬参考工程的禁手特征或辅助输入；如需移植其输入体系，应以完整通道契约替换并核对。

策略价值网络使用可配置的残差卷积主体、逐点 policy 输出与 masked 全局池化价值头，输出每个画布落点的原始 policy logits 和一个 `tanh` 标量价值。BatchNorm 统计、卷积中间特征与价值池化均处理有效棋盘 mask；实现见 [network.py](../python/etazero/network.py)。搜索接口消费统一的标量期望结果，未来采用 WDL 等表示时由模型适配层完成转换。

搜索时再将已占用点和画布外位置 mask 掉，对可提交动作归一化先验。训练策略 softmax 的域为有效棋盘上的全部落点：已占用点目标为零，画布外点不参与归一化。这个搜索与训练的区别需要固定样例验证，不能由不同调用方各自选择。

第一轮不隐式开启对称增强或搜索时对称集成。若用户指定 D4 增强，观测、动作、mask 与策略目标必须使用同一变换；整个 MuZero 展开轨迹也须保持一致，不能只旋转输入棋盘。

### 搜索状态与价值视角

- AlphaZero 在每个搜索深度使用真实棋盘转移和真实规则判定，不使用随机 rollout 替代网络叶节点评估。
- `v(s)` 始终表示状态 `s` 的当前行棋方的期望结果，胜 / 和 / 负对应 `+1 / 0 / -1`。
- 边的 `Q(s,a)`、`W(s,a)` 使用父状态 `s` 的行棋方视角。沿路径回传时按实际玩家转换符号；测试必须覆盖一手获胜、禁手判负和多层回传。
- 真实终局直接返回精确结果，不继续调用网络。AlphaZero 的本轮备份使用终局结果监督，不再叠加一份相同的终局奖励。

### PUCT 与搜索预算

每条边保存先验 `P`、完成访问数 `N` 和价值和 `W`，已访问边有 `Q = W / N`。单线程基准采用以下选择分数：

```text
Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
```

未访问边的基准 `Q` 为零；首次没有完成子访问时按最大先验选择，同分使用该局可复现的随机流。`c_puct` 取值由配置给出。KataGo 式 FPU 或其他探索常数调度若纳入移植，应有独立明确的语义，不能顺手替换本定义。

根节点网络展开与模拟次数分开计数。配置的模拟预算表示需要完成的选边、叶评估或终局判定及回传次数；网络调用数、树访问数和模拟数分别记录。预算要求至少产生一个可用根访问，取消或失败不以零访问分布冒充正常策略。

多线程可使用在途访问和 virtual loss 调度请求，但正式访问数与训练目标只能使用已完成结果。搜索返回前回收或取消全部在途工作，不能为了填满 batch 超出预算后又隐瞒实际完成量。

AlphaZero 可在同一模型代次内保留实际落子对应的子树。需要分别记录本次新增模拟量与复用后的总访问量，明确训练目标使用后者；新根的原始先验与探索先验分开保存，避免对已混噪声的先验再次混入噪声。规则、编码或模型变化后使树失效。跨手复用开关属于实验条件，应保存在生效配置中。

### 根探索、落子与训练目标

训练 selfplay 在根节点使用 Dirichlet 探索先验：

```text
P_search = (1 - epsilon) * P_network + epsilon * Dirichlet(alpha)
```

噪声仅在当前可提交动作上采样和归一化。`alpha`、`epsilon` 由 selfplay 配置确定；固定模型评估默认关闭训练根噪声。非根 PUCT 使用网络先验，不重复注入根噪声。

本方案分开保存训练目标和实际落子分布：

- AlphaZero 策略训练目标为根节点已完成访问数的归一化分布。
- 实际落子按访问数的温度变换分布选择，零温度时选择最大访问数动作；温度及按手数变化的计划由用户配置。
- 训练目标生成器与落子选择器是两个明确步骤，不因修改落子温度而隐式改变监督目标。该分离可参考 [Mctx 的 `muzero_policy`](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)，是这里明确选择的实现口径。

不能用 one-hot 实际动作、原网络先验、启发式局面评分或不完整对局的猜测结果替代规定目标。终局后，为每个搜索前状态生成该状态行棋方视角的 `z_t`。

### 学习与数据使用

每个训练样本对应 `(observation_t, policy_target_t, z_t)`，来源为 [原始对局分片](implementation.md#原始对局分片)。基础损失约定为：

```text
L = mean_t[(value_t - z_t)^2 - sum_a policy_target_t[a] * log p_t[a]]
    + c * sum_j theta[j]^2
```

正则化参数集合、系数、优化器和学习率由实际配置明确。显式 L2、优化器中的耦合衰减和 AdamW 的解耦衰减不能不加说明地互换，也不能重复计算正则项。价值目标来自真实终局，未完成对局不会凭空补零价值。

learner 从该轮固定窗口快照中采样，完成 `train.cfg` 规定的训练步数和 batch size，保存完整训练状态并发布推理模型。自对弈产量按 [固定训练量与 replay ratio](implementation.md#固定训练量与自对弈产量) 规划。没有用户指定时，不加入 WDL 头、surprise 加权、辅助策略头、reanalyse 或额外一致性损失。需要这些能力时，按其完整输入、目标、权重及状态恢复要求接入。

## 后续算法接入

### 公共边界

| 边界 | AlphaZero | 后续 MuZero |
|---|---|---|
| 根输入 | 真实观测 | 真实观测经过 representation |
| 树内状态 | 真实棋盘状态 | 学习得到的 latent state |
| 转移与叶评估 | 棋规转移，再调用策略价值网络 | dynamics / recurrent 推理及 prediction |
| 树内终局与 mask | 真实规则给出 | 按所移植 MuZero 的定义，不从真实棋盘偷取树内信息 |
| 样本使用 | 单状态监督 | 连续动作序列与多步展开目标 |

公共搜索执行器负责访问统计、并行、预算和回传调用；[SearchState / AlphaZeroState](../cpp/include/etazero/algorithm.h) 负责状态、转移、叶推理、奖励 / 折扣与玩家视角。数据服务保留完整真实轨迹；训练目标由算法构造器生成。第一轮只实现实际使用的 AlphaZero 适配层，不创建返回假值的 MuZero 类。

`SearchResult` 应区分实际动作、训练策略目标、根价值、访问统计和实际预算。Gumbel 接入后允许其训练目标与访问频率不同，不能把公共接口命名为“visit policy”后强制所有算法共用。

### MuZero

后续实现优先核对现有 [MuZero V2 算法定义](/home/sky/RL/MuZero/MuZero_V2/docs/algorithm.md)、[网络](/home/sky/RL/MuZero/MuZero_V2/python/muzero/network.py)、[搜索](/home/sky/RL/MuZero/MuZero_V2/cpp/src/search.cpp) 和 [展开目标](/home/sky/RL/MuZero/MuZero_V2/python/muzero/replay.py)。该工程是特定的棋类 MuZero，区分其基础预设与增强预设，不把二者混在一起移植。

完整接入必须覆盖 representation、dynamics、prediction、初始与 recurrent 组批、latent 生命周期、搜索备份、展开长度、价值 / 策略目标、梯度缩放以及终局与展开末端处理。真实环境只负责实际动作执行和监督来源，不能在 latent 树内调用 AlphaZero 的棋规转移代替 dynamics。

通用 MuZero 定义包含对奖励、价值和策略的学习；具体棋类适配是否使用 reward 分支、采用何种目标和 mask，以用户指定的移植来源为准。框架接口不能因第一轮棋类 AlphaZero 而永久写死零奖励或真实终局检测。[MuZero 原文](https://arxiv.org/pdf/1911.08265)

### Gumbel 根搜索与非根搜索

Gumbel 的根机制覆盖 Gumbel-Top-k 候选、Sequential Halving 预算分配、动作决策、未访问动作的 completed Q 以及由改进策略生成的训练目标。非根 `gumbel` 指论文式 14 的确定性选点，不是每层重新加噪声。[Gumbel 论文 §§3–5](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)

根为 Gumbel、非根保留 PUCT 是本项目支持的组合；论文附录 Figure 7 比较了保留原非根选点和使用新非根选点的版本。切换非根策略不能把根训练目标退回访问频率目标。[论文附录 Figure 7](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf#page=16)

后续实现参考作者团队的 [Mctx action selection](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/action_selection.py)、[Q transforms](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py) 和 [policies](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)，实施时固定所用提交。候选不足、极小预算、轮次余数、并列分数和非法动作的处理都属于完整移植范围，不能仅增加一个 Gumbel 采样步骤就宣称算法完成。

实现阶段分别验收根调度、Q completion、改进策略目标、非根策略及其与 AlphaZero / MuZero 模型的组合。实验收益和各组合的比较方案由用户安排。
