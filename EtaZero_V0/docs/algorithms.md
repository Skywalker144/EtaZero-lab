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

本项目将其适配到三种五子棋规则和可配置硬件规模。以下明确的是 EtaZero 的实现约定，不声称复现原论文的棋类任务、网络规模或实验预算。搜索接入 KataGo 的 FPU、shaped Dirichlet noise、playout cap randomization、forced playout / policy target pruning、加权价值统计、LCB、根多对称与分层温度；对局采样使用 policy/value surprise weighting。五子棋无 score utility、pass、提子和 no-result，WDL 的 D 表示真实和棋。辅助头、图搜索和 reanalysis 未接入。

### 游戏状态与棋规

真实状态包括棋盘、当前行棋方、规则、棋盘尺寸和终局状态。无提子、无 pass；坐标落在有效棋盘内的空点才可提交，终局后拒绝继续走子。完整对局从空棋盘开始；网络阶段可先执行下述开局前缀，正常搜索监督从前缀后开始。自对弈不引入比赛交换开局或开局库；独立比赛采用成对平衡开局，见 [评估协议](elo.md)。

| 规则 | 胜负语义 |
|---|---|
| Freestyle | 任一方连成五子或更长即胜 |
| Standard | 任一方恰好五子即胜；长连本身不判胜 |
| Renju | 白方五子或更长胜；黑方按参考棋规处理恰好五子与长连、双四、双三 |
| 满盘 | 先判本手胜负；无胜负时为和棋 |

Renju 沿用现有 MuZero / SkyZero 的棋盘局部规则，不扩展为完整比赛开局协议。参考中的禁手点允许实际落子，落下后判黑负，不在动作 mask 中提前删除；恰好五子与其他方向形状同时出现时按来源中的优先级判定。以 [MuZero Game](../../../MuZero/MuZero_V2/cpp/include/muzero/game.h)、[RenjuAnalyzer](../../../MuZero/MuZero_V2/cpp/src/rules.cpp) 和 [SkyZero Renju](/home/sky/RL/SkyZero/SkyZero_V8.1/katago/cpp/game/renju.cpp) 核对具体规则。

移植棋规要覆盖递归活三、边界与多方向交叉等情况，不能用局部字符串或简单形状计数代替来源语义。动作可提交性、落子后禁手判负和给网络的输入特征分别定义，不能混为一个“合法性”开关。

### 平衡开局与 policy init

平衡开局以 [KataGomo 原始机制](/home/sky/RL/SkyZero/KataGomo/cpp/game/randomopening.cpp) 和 [原始调用边界](/home/sky/RL/SkyZero/KataGomo/cpp/program/play.cpp) 为依据，实现位于 [opening.cpp](../cpp/src/selfplay/opening.cpp)。基础参数与配置入口在 [selfplay.cfg](../configs/baseline/selfplay.cfg)。只移植本环境的无 VCN 模式，不引入来源的 VCN 棋规。

1. 按 `{10,30,50,80,60,40,20,10,5,1,0,0}` 权重采样随机骨架落子数。首手采用截断高斯中心分布；以后空点权重累加每个已有棋子的 `(1+中心奖励)/(距离平方+avg_dist²)²`，`avg_dist` 为指数分布乘配置系数。
2. 从当前行棋方与反事实对手行棋方分别评估根局面。负根价值 v 按 `1-exp(-3*v²)` 与当前拒绝率共同决定是否重试；仅交换观察视角，不改变真实棋盘或落子方。
3. 当反事实对手根价值为正且已有棋子时，只枚举距已有棋子 Chebyshev 半径 3 内的空点。评估候选落子后的对手视角价值 v，排除已终局候选，按 `(1-v²)^balance_exponent` 采样最后一手平衡着；另以 `1-max_candidate_weight` 与当前拒绝率决定整次拒绝。
4. 骨架提前终局或拒绝时从原始空盘重试。连续失败次数 **大于** `max_tries` 后切换到 fallback 拒绝率，继续重试直到成功或取消；`max_tries` 不表示耗尽后接受失衡局面。保存实际尝试数和最后一手后的对手视角价值。

KataGomo 原生用 WDL 转换为行棋方标量，EtaZero 使用相同视角的 `W-L`；黑白共用同一已发布模型。随机数引擎沿用 EtaZero 的每局 `mt19937_64`，不承诺与 KataGomo 原始种子逐位相同。棋盘候选按来源的 x 外层／y 内层次序采样；移除了原始累计概率抽样中的负 epsilon，避免极少数端点抽中零权重非法位置。非法骨架、零总权重继续重试，非有限评估显式记录失败；配置决定失败后是否执行 policy init，不静默宣称平衡成功。

`policy_init` 独立于平衡开局，参考 [KataGomo policy initialization](/home/sky/RL/SkyZero/KataGomo/cpp/program/playutils.cpp)：开局局数为 `max(0,floor(Exp(1)*policy_init_mean - 2*已有手数))`，按可提交动作的网络 policy 温度分布逐手采样，保留 0.0002 的均匀动作分支。`policy_after` 和 `policy_on_failure` 分别控制平衡成功／失败后的 policy init，未尝试平衡时由 `policy_init` 决定。此过程可能结束整局，轨迹仍保存，监督行数为零。

随机冷启动不执行两种网络开局。开局前缀保留所有动作、观测与奖励，`train_mask=0`；其余搜索状态保存 policy/WDL 搜索结果，按 cap 与 surprise 决定写入次数。总落子数与实际采样训练行数分别计数，replay ratio、窗口与 shuffle 使用后者。基础温度手数阈值仍按整个真实对局的手数判断，包含开局前缀。

### 观测、动作与网络

当前实现以单模型固定画布承载配置中的棋盘尺寸。画布边长由该模型支持的最大尺寸决定，网络动作编号统一使用画布上的行优先编号；实际棋盘编号只在外部接口做明确转换，落盘时不得混用。

输入依据 SkyZero_V7.19 的 `gomoku.h::encode_state/compute_global_features`，采用 5 个空间平面与前 4 个全局特征；契约的唯一代码定义在 [schema.py](../python/etazero/schema.py)。

| 空间通道 | 定义 |
|---|---|
| 0 | 有效棋盘 mask |
| 1、2 | 当前视角的己方、对方棋子 |
| 3 | 当前视角为黑方时，黑方的禁手点 |
| 4 | 当前视角为白方时，黑方的禁手点 |

禁手平面仅 Renju 非零；两个平面都表示黑方禁手，白方本身没有禁手。占用点与 padding 均为零，禁手判定复用真实落子使用的递归 RenjuAnalyzer。全局输入为 `float32[N,4]`：`is_standard`、`is_renju`、Renju 执色（黑 -1、白 +1，其他规则 0）、禁手特征可用标志（仅完整 Renju 输入为 1）。Freestyle 由前两项均为 0 表示。draw utility/PDA 尚未接入，因此不包含 SkyZero 的最后 3 个全局量。全局量经无 bias 的线性层投影到 trunk 通道，广播加到首层空间卷积输出。

`selfplay.forbidden_feature_dropout_prob` 沿用 SkyZero 的训练行独立抽样方式：被选中的 Renju 行将空间 3、4 和全局 3 同时置零；其余规则、执色、棋子和监督目标保持完整。随机流由局种子独立派生，不消耗尺寸、规则、开局或动作采样的 RNG。搜索、平衡开局的两个根视角及候选评估、固定模型评估均保留完整特征。原始轨迹保存实际训练输入，开局前缀和末状态不 dropout，shuffle 不再次随机化。

尺寸与规则每局分别按配置权重采样，零权重不参与采样。平衡开局的候选 Game 继承该局的尺寸与规则，反事实根评估同时交换己敌方平面、禁手平面和 Renju 执色符号；实际棋盘和行棋方保持不变。

策略价值网络使用可配置的残差卷积主体、逐点 policy 输出与 masked 全局池化价值头，输出每个画布落点的原始 policy logits 和当前玩家视角的 W/D/L 三分类 logits `[N,3]`。训练用终局 WDL one-hot 的交叉熵；native 推理在 FP32 中 softmax 得到概率，`v = W-L`。BatchNorm 统计、卷积中间特征与价值池化均处理有效棋盘 mask；实现见 [network.py](../python/etazero/network.py)。搜索保留网络原始 WDL、回传的 WDL 均值和 `W-L` 二阶矩；和棋概率在换视角时保持，胜负概率互换。

搜索时将已占用点和画布外位置 mask 掉，每个网络评估先将 logits 除以 `nn_policy_temperature`，再对可提交动作归一化先验；这个全树 policy 温度同时作用于根和所有非根。训练策略 softmax 的域为有效棋盘上的全部落点：已占用点目标为零，画布外点不参与归一化。这个搜索与训练的区别需要固定样例验证，不能由不同调用方各自选择。

`training.d4_augmentation` 控制训练 D4，baseline 开启。沿用 SkyZero_V8.1 的 batch 级均匀八对称，按整个画布变换全部空间通道（包括有效棋盘和禁手 mask）与 policy 目标；小棋盘的有效区域可以随变换移动到其他角。全局特征和 WDL 目标不变，原始轨迹不重写。变换发生在 learner 消费 batch 后、联合编译前，结果保持 contiguous；随机流纳入 Torch CPU RNG checkpoint，AMP 重试复用同一变换与前向图。实现见 [symmetry.py](../python/etazero/symmetry.py)。

搜索根由 `root_num_symmetries_to_sample` 控制 D4 集成，取值 1–8；大于 1 时从八种对称无放回均匀抽样。变换覆盖整个画布的全部空间输入，全局输入不变，输出 policy 还原到原始动作坐标。每个对称先独立执行全树温度及合法域 softmax，随后算术平均 policy 概率和 WDL 概率；不平均 logits，多个评估仍只构成一个根初始访问。根从非根子树推进而来时，启用多对称的正常搜索重新评估根，访问数不增加；零权重 cheap search 使用已有单对称评估，或只作一次身份对称评估。当前单对称根及非根使用身份对称，不启用来源可选的全树随机单对称。NN 缓存键包含实际变换后的完整输入，因此各朝向不会误用原朝向 logits；与来源的朝向无关缓存不同，此处可安全复用逐朝向确定性输出。实现见 [AlphaZeroState](../cpp/include/etazero/algorithm.h)。

### 搜索状态与价值视角

- AlphaZero 在每个搜索深度使用真实棋盘转移和真实规则判定，网络阶段不使用随机 rollout 替代叶节点评估；冷启动仅将 evaluator 替换为高斯随机 policy/WDL logits（WDL 经 softmax），仍保留真实规则搜索、精确模拟预算与终局监督。
- `v(s)` 始终表示状态 `s` 的当前行棋方的期望结果，胜 / 和 / 负对应 `+1 / 0 / -1`。
- 边的 `Q(s,a)`、价值和及二阶矩使用父状态 `s` 的行棋方视角。沿路径回传时按实际玩家转换符号；测试必须覆盖一手获胜、禁手判负和多层回传。
- 真实终局直接返回精确结果，不继续调用网络。AlphaZero 的本轮备份使用终局结果监督，不再叠加一份相同的终局奖励。

### PUCT 与搜索预算

节点保留完成访问数、W−L 与和棋概率的加权均值、价值二阶矩、权重和与权重平方和；边保留完成访问数和在途数。子统计按边访问数／子节点访问数换算。PUCT 使用完成子权重总和 `T` 和对应子权重 `W_a`：

```text
Q(s, a) + explore_scaling(T, parent_stats) * P(s, a) / (1 + W_a)
explore_scaling = (c_puct + c_puct_log * log((T+c_puct_base)/c_puct_base))
                  * sqrt(T+0.01) * parent_utility_stdev_factor
```

未访问边用 FPU，关闭 `use_fpu` 时为零。FPU 沿用 KataGo 的 visited-policy 插值：令 `m` 为已分配子节点的先验质量，`a=min(1,m^power)`，则 `FPU = a * parent_Q + (1-a) * NN_Q - reduction * sqrt(m)`，随后按 `fpu_loss_prop` 向 −1 插值。parent_Q 包含节点初始网络样本；根与非根 reduction、loss proportion 独立，零权重 cheap search 根使用非根参数。动态 cpuct 的 log 项与父价值标准差项沿用来源公式，可由配置关闭。价值尺度为 W−L 的 [-1,1]，无 score utility。参数见 [selfplay.cfg](../configs/baseline/selfplay.cfg)、[eval.cfg](../configs/baseline/eval.cfg) 与 [match.cfg](../configs/baseline/match.cfg)。

节点更新使用 KataGo 的 `value_weight_exponent`：先按原始权重计算简单子价值均值，对每个子树用 `sqrt(1e-8 + 1/(1.5*sqrt(weight)))` 估计标准差，以自由度 3 的 t 分布 CDF（[-50,50] 上 2000 点线性插值表）加 0.0001 后取配置指数，偏重较好的子价值。根有噪声时先执行 configured chosen-move subtract/prune；重分配后归一化回原始权重总和，再加入权重 1 的本节点网络样本。W−L、draw 和二阶矩使用同一重分配，权重平方和按缩放平方更新。指数为零是显式单位权重对照，不是默认搜索。实现见 [search_math.cpp](../cpp/src/search/search_math.cpp)。

根节点网络展开与模拟次数分开计数。`search.simulations + 1` 为自对弈 full-search 根访问上限，cheap-search 使用独立访问上限；评估直接配置 visits。上限计入根初始网络样本和复用访问，只补足缺口。网络调用数、总树访问数和本手新增模拟数分别记录。新树至少完成一条根边；复用访问已达到上限时可新增零次模拟，仍使用真实已完成统计。取消或失败不以零访问分布冒充正常策略。

多线程的在途访问以 `pending * virtual_loss` 作为虚拟样本权重，价值向 −1 插值，PUCT 分母与 forced 配额使用同一虚拟权重；探索分子的总子权重只计完成统计。正式访问数、统计与训练目标只使用已完成结果。搜索返回前回收或取消全部在途工作，不能为了填满 batch 超出预算后又隐瞒实际完成量。

AlphaZero 可在同一模型代次内保留实际落子对应的子树。需要分别记录本次新增模拟量与复用后的总访问量，明确训练目标使用后者；新根的原始先验与探索先验分开保存，避免对已混噪声的先验再次混入噪声。规则、编码或模型变化后使树失效。跨手复用开关属于实验条件，应保存在生效配置中。

### 根探索、落子与训练目标

根先验依次执行全树温度、逐对称概率平均、根 policy 温度、Dirichlet 噪声。根 policy 温度按下述半衰期公式从 `root_policy_temperature_early` 衰减到 `root_policy_temperature`，概率取 `1/T_root` 次幂并重新归一化。训练 full search 再混入 `P_search = (1-epsilon) * P_tempered + epsilon * Dirichlet(alpha)`。噪声只覆盖可提交动作，`sum(alpha)` 为固定 total concentration；shaped 模式基于根温度后的先验，将一半浓度均匀分配，另一半按 `max(0, log(min(P,0.01)+1e-20)-mean_log)` 归一化分配；全零形状退化为均匀分配。关闭 shaped 模式仍使用相同总浓度的均匀 Dirichlet。每次从原始根先验生成温度与噪声，不累积变换，非根不加根噪声，评估和比赛不使用训练根噪声。

Playout cap randomization 按每手独立 Bernoulli 选择 full / cheap cap。正常搜索按 `clear_before_search` 清树；cheap target weight 为零时保留已推进到实际落子后的子树，关闭根噪声、根 policy 温度、多对称评估和 forced playout，使用非根 FPU 参数；全树温度继续生效。访问上限包含复用量，因此 cheap search 可不新增模拟。下一次 full search 重新清树。`reuse_tree=false` 显式关闭跨手复用，cheap 权重大于零时仍按正常清树与探索配置执行。开局和模型换代均重置搜索。默认开启清树；训练预算仍由实际采样行数规划。

Reduce Visits 与 PCR 的预算选择集中在 [search_limits.cpp](../cpp/src/selfplay/search_limits.cpp)，按 KataGo `play.cpp` 的分支顺序执行：命中 PCR cheap 时不再执行 Reduce Visits，不叠加预算缩减或目标权重。非 cheap 手在历史足够时，取最近 `reduce_visits_threshold_lookback` 手的完成搜索 W−L；历史统一为黑方视角，包括 cheap 与已缩减搜索，排除开局前缀和当前手，每局重新开始。令 `e=max(min(history),-max(history))`；只有同一方连续占优且 `e>reduce_visits_threshold` 才缩减，不能逐手取绝对值后把交替占优当作稳定局势。缩减比例为 `r=((min(e,1)-threshold)/(1-threshold))²`，访问 cap 为 `round(full_cap+r*(reduced_visits_min-full_cap))`，基础目标权重为 `1+r*(reduced_visits_weight-1)`。`round` 沿用 C++ 正半整数向上取整；cap 包含初始根评估，`full_cap=search.simulations+1`。

缩减的 full search 保留正常清树、根噪声、根温度、根 FPU、多对称、forced playout 和训练 target LCB 设置，不标为 cheap；即使缩减权重为零，也不切换到 cheap 的清树与探索行为。默认清树时，它丢弃前一手 PCR 保留的树并使用自己的 cap；显式关闭清树时，cap 仍包含复用访问。参数属于自对弈配置，baseline 沿用 SkyZero_V8.1 的阈值、lookback、最低访问数和目标权重；smoke_test 明确缩小最低访问数。当前搜索要求 cap 至少为 2、lookback 至少为 1，保证存在已完成根边及非空历史窗口。评估和比赛保持独立固定预算，不启用此机制。五子棋没有动态 score utility center，无须移植来源为该中心执行的 10v 预搜索。

Forced playout 作用于已分配、先验为正的根子边：若其完成权重加虚拟样本权重小于 `sqrt(P_search * total_completed_child_weight * coeff)`，以强制优先级继续探索；虚拟权重防止线程重复追赶同一份配额。弱着这些额外访问不能直接当作训练目标。

Policy target pruning 先按 `child_weight*max(0,N-1)/max(1,N)+2*P_search` 选稳定子边，再用同一 `explore_scaling` 的 PUCT 分数反解其他子边所需权重：`ceil(min(child_weight, max(0, explore_scaling*P_search/(best_score-Q)-1)))`，仅在分母为正时减少。稳定边保留；LCB 后统一执行 `chosen_move_prune` 与 `chosen_move_subtract`，两者各自上限为最大选择权重／64。原始 visits 始终独立保存，policy target 不要求等于 visits 归一化。

LCB 使用完成样本的 W−L 加权均值、二阶矩和 `ESS = weight_sum² / weight_sq_sum`。按 KataGo 加入最大方差先验，先验权重为 `weight_sum / ESS³`，更新两种权重和后重新计算 ESS，再计算 `LCB = Q-lcb_stdevs*sqrt(variance/ESS)`。仅剪枝后权重大于零、达到稳定边原始权重比例门槛的边可成为 LCB 最优边；将最优边权重提升到至少 `other_weight * ((radius+excess)/(radius+0.2*excess))²`。此处采用修正后的 LCB 索引行为，首个子边同样有效。

自对弈实际落子使用剪枝后、LCB 前的分布；训练 policy target 再应用 LCB。评估和比赛在落子分布和输出策略中都应用 LCB，分别由各自 profile 控制。落子温度为 `late + (early-late)*0.5^(turn/temperature_halflife*19/sqrt(board_area))`，手数包含开局。自对弈的 early/late 分别为 `exploration.temperature` / `final_temperature`，eval/match 为 `temperature_early` / `temperature`。温度只改变行为抽样，不改变监督；小于等于 1e-4 且无概率保护时，按来源选择首个最大权重子边；其余情况对权重执行稳定幂变换，`temperature_only_below_prob` 可保护超过概率阈值的部分，仅变换低概率尾部。根 WDL 和 Q 是包含初始网络样本的完成加权搜索均值，与训练用的真实终局 WDL 目标分别保存。评估输出 `network_policy` / `network_wdl` 为根集成结果，`search_policy` 为根温度和噪声后的先验，`policy` 为最终监督／选择策略。

这些步骤直接核对 [KataGo FPU/forced/pruning](/home/sky/RL/SkyZero/KataGo/cpp/search/searchexplorehelpers.cpp)、[noise/LCB](/home/sky/RL/SkyZero/KataGo/cpp/search/searchhelpers.cpp)、[selection](/home/sky/RL/SkyZero/KataGo/cpp/search/searchresults.cpp)、[子树价值加权](/home/sky/RL/SkyZero/SkyZero_V8.1/katago/cpp/search/searchupdatehelpers.cpp) 和 [selfplay 调用边界](/home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp)，并以 SkyZero_V8.1、MuZero_V2 交叉核对。EtaZero 使用 W−L 子树价值加权，未引入 Go score utility、uncertainty weighting、noise pruning、subtree value bias、图转置或专用推理算子。随机引擎、并行锁调度和单对称身份评估沿用 EtaZero，不承诺来源逐位复现；t(3) CDF 使用数学等价闭式求值生成同网格插值表。

### 学习与数据使用

每个训练样本对应 `(spatial_t, globals_t, policy_target_t, WDL_target_t)`，来源为 [原始对局分片](implementation.md#原始对局分片)。基础损失约定为：

```text
L = mean_t[-sum_k WDL_target_t[k] * log WDL_pred_t[k]
           -sum_a policy_target_t[a] * log policy_pred_t[a]]
```

优化器参照 [SkyZero_V8.1 native train.py](/home/sky/RL/SkyZero/SkyZero_V8.1/katago/python/train.py) 的 BN 分支，默认 SGD（momentum 0.9），支持 AdamW（CUDA fused）。输入权重、残差权重、残差/输入 BN gamma、相应 bias、head 权重及 head bias 分组，所有参数必须恰好归组一次；head 的 BN gamma/bias 按输出组处理。具体 LR、WD 系数、batch scaling、warmup 和范数自适应公式集中在 [optimization.py](../python/etazero/optimization.py)，不在文档维护第二套常量表。当前没有 attention 等参数，未接入 Muon/NorMuon/Aurora、自动或自定义 LR schedule。

上式定义日志中的平均 loss；实际反向使用 `batch_size * L`，与来源的 batch 求和尺度一致。默认梯度上限按优化器、batch 与 LR scale 计算；`training.gradient_clip > 0` 则以平均梯度尺度覆盖上限。梯度日志仍为裁剪前平均 loss 的梯度范数。SGD 的衰减加入优化器梯度，AdamW 的衰减解耦施加，均不重复加入 loss。价值目标来自真实终局，未完成对局不会凭空补零价值。

Lookahead 同步时将 fast 权重向 slow 权重平均，LR 按 alpha 补偿；完整训练轮结束丢弃未同步的 fast 权重，优化器状态与成功更新计数保留。中断 learner 保存 fast、slow 和同步计数，恢复继续同一周期。SWA 使用来源的指数平均规则，仅在 Lookahead 同步后采样，首个样本直接复制；BN buffers 在采样时直接复制，不参与参数平均。采样累积跨轮保留，未得到 SWA 样本时发布当前模型。

适配差异：范数按配置的 batch 间隔取 snapshot（默认对应来源的 100-batch print interval），LR/WD 每次更新刷新，而非来源的 5/50-batch 刷新；D4 使用可恢复的 Torch CPU RNG，不复现来源 NumPy 种子的逐位随机序列。网络和监督头仍是 EtaZero 的基础残差 policy/value，不宣称完整复现 KataGo 训练目标。

learner 从该轮固定窗口快照中采样，完成 `train.cfg` 规定的训练步数和 batch size，保存完整训练状态并发布推理模型。自对弈产量按 [固定训练量与 replay ratio](implementation.md#固定训练量与自对弈产量) 规划。辅助策略头、reanalysis 和额外一致性损失未接入。

### Policy / value surprise weighting

每局结束后计算采样期望权重，来源是 KataGo `play.cpp` 的非 reanalysis 路径。Policy surprise 为最终 policy target 相对于实际搜索先验（含根噪声）的 KL。Value surprise 从真实终局 WDL 开始，以 `now=1/(1+board_area*0.016)` 向前平滑后续搜索 WDL，再与该手原始网络 WDL 计算 KL，限制到 [0,1]；计算采用统一黑方视角，存储采用当前玩家视角。

初始普通 full 权重为 1，PCR cheap 权重来自配置，Reduce Visits full 权重来自预算缩减的同一比例。以这些初始权重计算平均 surprise，policy 重分配量为 `weight*policy_surprise + (1-weight)*max(0,policy_surprise-1.5*weighted_mean_policy_surprise)`；value 量为 `weight*value_surprise`。两者分别归一化到初始总权重，按配置比例与原始权重混合；平均 value surprise 小于 0.010 时同比减小 value 混合比例。分母下限沿用来源 1e-10，完全零 surprise 的对应份额会衰减，不伪造均匀 surprise。开局前缀始终排除，不能因 surprise 获得监督。

期望权重可大于 1，按 `floor(weight)+Bernoulli(frac(weight))` 随机取整，只执行一次并保存 `row_repeats`。训练视图按该次数重复样本，零次数的搜索行仍保留完整轨迹。shuffle 不重新抽样这些权重，训练 loss 不再额外乘同一权重。replay、窗口和产样预算按实际重复后的行数计数，因此不会把大量 cheap 行当作完整监督。随机取整使用每局 RNG；原始权重、两种 surprise、网络/搜索 WDL、cheap 标记及实际次数均保存供核对。

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
