# 算法与搜索设计

本文描述当前 AlphaZero 的算法语义和后续算法边界。运行架构、数据保存与验收见 [运行实现](implementation.md)。AlphaZero / PUCT 和 MuZero / PUCT 已实现，Gumbel 仍属后续设计，网络规模、训练参数与正式实验方案由用户确定。

## 算法与搜索组合

`algorithm` 决定搜索使用的状态转移、网络结构和训练方式；`root_search_algo` 与 `nonroot_search_algo` 分别决定根节点和非根节点的搜索策略。算法身份需要记录完整三元组，不能仅用一个“Gumbel”标签概括所有配置。

| algorithm | root_search_algo | nonroot_search_algo | 目标能力 | 当前状态 |
|---|---|---|---|---|
| alphazero | puct | puct | AlphaZero | 已实现 |
| alphazero | gumbel | puct | Gumbel 根搜索 + PUCT 非根搜索 | 后续实现 |
| alphazero | gumbel | gumbel | Gumbel 根搜索 + 改进策略非根搜索 | 后续实现 |
| muzero | puct | puct | MuZero | 已实现，见 [MuZero](muzero.md) |
| muzero | gumbel | puct | Gumbel 根搜索 + PUCT 非根搜索 | 后续实现 |
| muzero | gumbel | gumbel | Gumbel 根搜索 + 改进策略非根搜索 | 后续实现 |
| 任一 | puct | gumbel | 本项目不允许的组合 | 始终拒绝 |

最后一行是用户确定的项目组合约束，不表述为数学上不可能。合法但尚未实现的组合，应报告未实现能力；非法组合应报告组合错误。两者均在启动 worker、分配大块显存或创建训练产物前拦截。

配置格式的定义位置见 [配置组织](implementation.md#配置组织)。选择 Gumbel 根搜索同时选择其根动作决策及训练策略目标，不能只把选点公式换掉而仍隐式沿用 AlphaZero 的目标生成。

## AlphaZero

### 来源与适配边界

采用 AlphaZero 的策略价值网络、真实规则图搜索和自对弈监督闭环；网络预测落子概率与当前玩家的期望终局结果，训练结合策略交叉熵、价值误差与正则化。模型持续更新并供自对弈使用，固定模型评估独立运行，不默认加入“胜过旧模型才发布”的门控。[AlphaZero 原文，算法描述及式 1](https://arxiv.org/pdf/1712.01815)

本项目将其适配到三种五子棋规则和可配置硬件规模。以下明确的是 EtaZero 的实现约定，不声称复现原论文的棋类任务、网络规模或实验预算。搜索接入 KataGo 的 FPU、shaped Dirichlet noise、playout cap randomization、forced playout / policy target pruning、加权价值统计、LCB、根多对称与分层温度；对局采样使用 policy/value surprise weighting，学习接入六项 policy、终局/TD WDL 和短期价值误差监督。五子棋无 score utility、pass、提子和 no-result，WDL 的 D 表示真实和棋。v17支持可选逐动作纯W−L Q监督；hint/hintFork 和 early/game fork 贯通独立启动前缀与预算；真实规则搜索支持图转置。

### 游戏状态与棋规

真实状态包括棋盘、当前行棋方、规则、棋盘尺寸和终局状态。无提子、无 pass；坐标落在有效棋盘内的空点才可提交，终局后拒绝继续走子。完整对局从空棋盘开始；网络阶段可先执行下述开局前缀，正常搜索监督从前缀后开始。自对弈可从下述hint/fork的可重放前缀开始，不引入比赛交换开局；独立比赛采用成对平衡开局，见 [评估协议](elo.md)。

| 规则 | 胜负语义 |
|---|---|
| Freestyle | 任一方连成五子或更长即胜 |
| Standard | 任一方恰好五子即胜；长连本身不判胜 |
| Renju | 白方五子或更长胜；黑方按参考棋规处理恰好五子与长连、双四、双三 |
| 满盘 | 先判本手胜负；无胜负时为和棋 |

Renju 沿用现有 MuZero / SkyZero 的棋盘局部规则，不扩展为完整比赛开局协议。参考中的禁手点允许实际落子，落下后判黑负，不在动作 mask 中提前删除；恰好五子与其他方向形状同时出现时按来源中的优先级判定。以固定版本 [KataGomo 禁手检测器](/home/sky/RL/SkyZero/KataGomo/cpp/forbiddenPoint/ForbiddenPointFinder.cpp) 和 [真实终局](/home/sky/RL/SkyZero/KataGomo/cpp/game/gamelogic.cpp) 核对具体规则；本地递归实现最初来自 MuZero / SkyZero。

棋规对照入口为 [check_katagomo_rules.py](../tests/reference/check_katagomo_rules.py)，直接编译固定commit的KataGomo `CForbiddenPointFinder` 与本地 `RenjuAnalyzer`，覆盖五种目标尺寸的邻域穷举、交叉方向、边界、同方向双四、恰五优先和假三递归检查。有限局面语料不能证明所有递归状态等价。当前范围为方形Freestyle/Standard/Renju、无pass；VCN、矩形和对应额外终局语义不支持，配置或尺寸解析明确拒绝。

移植棋规要覆盖递归活三、边界与多方向交叉等情况，不能用局部字符串或简单形状计数代替来源语义。动作可提交性、落子后禁手判负和给网络的输入特征分别定义，不能混为一个“合法性”开关。

### 平衡开局与 policy init

平衡开局以 [KataGomo 原始机制](/home/sky/RL/SkyZero/KataGomo/cpp/game/randomopening.cpp) 和 [原始调用边界](/home/sky/RL/SkyZero/KataGomo/cpp/program/play.cpp) 为依据，实现位于 [opening.cpp](../cpp/src/selfplay/opening.cpp)。基础参数与配置入口在 [selfplay.cfg](../configs/baseline/selfplay.cfg)。只移植本环境的无 VCN 模式，不引入来源的 VCN 棋规。

1. 按 `{10,30,50,80,60,40,20,10,5,1,0,0}` 权重采样随机骨架落子数。首手采用截断高斯中心分布；以后空点权重累加每个已有棋子的 `(1+中心奖励)/(距离平方+avg_dist²)²`，`avg_dist` 为指数分布乘配置系数。
2. 从当前行棋方与反事实对手行棋方分别评估根局面。负根价值 v 按 `1-exp(-3*v²)` 与当前拒绝率共同决定是否重试；仅交换观察视角，不改变真实棋盘或落子方。
3. 当反事实对手根价值为正且已有棋子时，只枚举距已有棋子 Chebyshev 半径 3 内的空点。评估候选落子后的对手视角价值 v，排除已终局候选，按 `(1-v²)^balance_exponent` 采样最后一手平衡着；另以 `1-max_candidate_weight` 与当前拒绝率决定整次拒绝。
4. 骨架提前终局或拒绝时从原始空盘重试。连续失败次数 **大于** `max_tries` 后切换到 fallback 拒绝率，继续重试直到成功或取消；`max_tries` 不表示耗尽后接受失衡局面。保存实际尝试数和最后一手后的对手视角价值。

KataGomo 原生用 WDL 转换为行棋方标量，EtaZero 使用相同视角的 `W-L`；SP黑白共用同一已发布模型；Match每次有效骨架后的balance尝试以0.5概率选参考botB或botW，同一尝试的两个根视角及全部候选使用该模型。随机数引擎沿用 EtaZero 的每局 `mt19937_64`，不承诺与 KataGomo 原始种子逐位相同。棋盘候选按来源的 x 外层／y 内层次序采样；移除了原始累计概率抽样中的负 epsilon，避免极少数端点抽中零权重非法位置。非法骨架、零总权重继续重试，非有限评估显式记录失败；配置决定失败后是否执行 policy init，不静默宣称平衡成功。

`policy_init` 独立于平衡开局，参考 [KataGomo policy initialization](/home/sky/RL/SkyZero/KataGomo/cpp/program/playutils.cpp)：开局局数为 `max(0,floor(Exp(1)*policy_init_mean - 2*已有手数))`，按可提交动作的网络 policy 温度分布逐手采样，保留 0.0002 的均匀动作分支。`policy_after` 和 `policy_on_failure` 分别控制平衡成功／失败后的 policy init，未尝试平衡时由 `policy_init` 决定。SP显式开关必填，mean缺省12、温度缺省1；具体启用状态、mean 和温度以所选配置为准。Match开关缺省false，开启时mean必须显式给出，温度缺省1；每手按参考黑白执色调用对应模型。`policy_after`/`policy_on_failure`是可配置消融开关，默认均true。此过程可能结束整局，轨迹仍保存，监督行数为零。

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

禁手平面仅 Renju 非零；两个平面都表示黑方禁手，白方本身没有禁手。占用点与 padding 均为零，禁手判定复用真实落子使用的递归 RenjuAnalyzer。全局输入为 `float32[N,6]`：`is_standard`、`is_renju`、Renju 执色（黑 -1、白 +1，其他规则 0）、禁手特征可用标志（仅完整 Renju 输入为 1）。Freestyle 由前两项均为 0 表示。最后两个全局量为 PDA 启用标志及按当前执色变号的 `0.5*d`；五子棋无 draw utility 输入。全局量经无 bias 的线性层投影到 trunk 通道，广播加到首层空间卷积输出。

`environment.forbidden_feature_dropout_prob` 按 KataGomo writer 在最终输出行独立抽样：被选中的 Renju 行将空间 3、4 和全局 3 同时置零；其余规则、执色、棋子、PDA 条件和监督目标保持完整。随机流由局种子独立派生，不消耗尺寸、规则、开局或动作采样的 RNG。搜索、平衡开局的两个根视角及候选评估、固定模型评估均保留完整特征。原始轨迹始终保存完整特征；writer在surprise次数确定后，为每个最终重复行抽样并保存 `forbidden_input:uint8[sum(row_repeats)]`。训练视图展开时应用这些已保存的决定，重复行可不同；shuffle与重启不会再次随机化。开局前缀和末状态不进入这些训练行。

尺寸与规则每局分别按配置权重采样，零权重不参与采样。平衡开局的候选 Game 继承该局的尺寸与规则，反事实根评估同时交换己敌方平面、禁手平面和 Renju 执色符号；实际棋盘和行棋方保持不变。

`network.architecture` 选择 `nbt`、`plain` 或 `transformer`，两个卷积预设均使用 Mish、fixed scaling 和末端 masked BatchNorm，输出六policy、主WDL、三个TD WDL和短期价值误差。默认 `nbt` 对照 KataGo 的 `b5c192nbt-fson-mish`，`network.blocks` 是外层嵌套瓶颈块数，`network.channels` 是主干宽度；内部宽度由 [network_widths](../python/etazero/config.py) 唯一推导。NBT主干通道须为不小于24的偶数，瓶颈为主干的一半，主干全局分支及Policy/Value卷积宽度按主干的1/6向上取整到8的倍数，Value隐藏层按主干的5/12同样取整；192通道还原源配置的96、32、80宽度。NBT宽度缩放是EtaZero的可配置约定，并非KataGo全部预设的统一比例。

`plain` 对照 `b10c128-fson-mish`（v15），配置须为128通道、10块。每块直接在主干上执行两层前激活3×3卷积并加残差；第5、8块的首层将128通道分为96局部通道和32 gpool通道，全局上下文广播加入局部结果，第二层从96投影回128。没有NBT的外层瓶颈投影。Policy/Value卷积宽度为32，Value隐藏层80；每块首层缩放为 `1/sqrt(i+1)`，第二层缩放1，末端BN缩放 `1/sqrt(11)`。其他plain宽度/深度明确拒绝。

每个 NBT 块先用前激活 1×1 卷积将主干投影到瓶颈，依次执行两个内层残差块（每个包含两层前激活 3×3 卷积），再通过前激活 1×1 投影回主干并加上外层残差。按源 b5 排列，第 2、4、6……个外层块的第一个内层块带全局池化；其首层将瓶颈宽度分为局部和全局卷积分支，全局分支经激活、池化和线性投影后广播加到局部结果。池化使用每张棋盘的均值、按尺寸缩放的均值、有效区域最大值；padding 不参与。

卷积预设的激活使用 Mish，卷积及线性层使用 KataGo 的方差校正截断正态初始化。内部归一化为 `(1+gamma_offset) × fixed_scale × x + bias` 后应用 mask，gamma_offset 初始化为零并向零衰减。外层第 i 块入口缩放为 `1/sqrt(i+1)`（i 从零开始），两个内层入口分别为 1 和 `1/sqrt(2)`，瓶颈出口为 `1/sqrt(3)`。仅主干末端使用 masked BatchNorm，再乘 `1/sqrt(blocks+1)`；训练按有效点数计算总体均值和标准差，epsilon 为 1e-4，running mean/std 的 EMA 动量为 0.001，沿用源 fson 语义。导出在实际推理设备预计算倒数标准差，保持缩放与加法的运算顺序。

`transformer` 对照bare `b5c192h3nbttfrs`（v17，缺省无Q），严格要求192通道、5块。每个外块先用bias/mask、ReLU和1×1卷积投影到96通道，再依次执行attention、SwiGLU、attention、SwiGLU四个内残差，最后经带gamma offset的fixup归一化、ReLU和1×1卷积回到192通道并加外残差。入口卷积使用ReLU增益和 `(1/sqrt(5))^(1/3)` 初始化尺度，出口卷积初始化为零；内层线性保留来源的PyTorch默认初始化，RMSNorm权重为1、epsilon为1e-6。末端只有masked bias/ReLU，无最终BN或gamma。Policy/Value卷积宽度32、Value隐藏层64。实现见 [transformer.py](../python/etazero/transformer.py)。

Attention将96通道分为3个32维头，Q/K/V由无bias线性映射，固定2D RoPE的theta=100，每头前16维编码行、后16维编码列，成对交错旋转在FP32执行后恢复投影dtype。只有padding key被加−inf，padding query仍可读取有效key，后续投影mask控制其贡献。SwiGLU为 `SiLU(W_input·RMSNorm(x)) × (W_gate·RMSNorm(x))`，FFN宽度256，再线性投影回96；CUDA训练FP16/BF16沿用来源的融合Triton前向/反向，见 [fused_swiglu.py](../python/etazero/fused_swiglu.py)。推理使用可序列化的分解计算，保留RoPE乘加及SiLU/gating的中间舍入，以维持TF32条件下eager与优化后TorchScript的一致性。

本实现选择来源公开可选的NCHW/SDPA路径，显式保留投影mask；来源默认NHWC/flex attention及省略冗余mask的执行方式不同，不宣称执行性能等价。Transformer的learner保持full-graph编译，但关闭自动channels-last布局改写：PyTorch2.12在这一BF16路径会生成错误的注意力梯度。此约束仅作用于该架构，不改变精度、算法、初始化或优化预算。

Transformer使用fixup七参数组，attention投影单列 `normal_attn`，RMSNorm和内部bias归入noreg，head归入output/output_noreg。fixup SGD的基本WD为1e-6乘batch比例，AdamW为0.005乘batch比例的平方根；attention再乘0.5和 `normal_attn_wd_factor`，input/normal沿各自factor，noreg/output_noreg为1e-8乘相同batch尺度。该分支不使用fson的运行范数自适应WD，LR/warmup、Lookahead和SWA仍由 [optimization.py](../python/etazero/optimization.py) 管理。

PolicyHead 将主干分别投影到局部和全局特征。全局特征经过 bias、Mish、上述三项池化和线性层，广播加入局部特征，再经过 bias、Mish 和 1×1 输出；全局信息进入最终非线性之前，因此不会成为被 softmax 抵消的统一偏移。ValueHead 使用 1×1、bias、Mish 后的均值，拼接 `mean`、`mean × (sqrt(area)-14)/10`、`mean × ((sqrt(area)-14)²/100-0.1)`，经隐藏层、Mish 和线性输出 WDL；这三个均值保留源尺寸条件化形式，不使用最大值。固定尺寸下它们携带相同空间信息，混合尺寸时为价值层提供尺寸条件。Transformer的上述head激活使用ReLU；两个卷积预设使用Mish。KataGo 的 Go pass、score/ownership 等辅助头、RepVGG 额外卷积及双输出头训练不属于此实现；`network.predict_q_values=true` 仅在Transformer v17添加第七个纯W−L Q输出，Go score Q不适用；缺省false及v15保留六输出。

训练 policy logits 为 `[N,6,canvas²]`，顺序为 policy、opponent policy、soft policy、soft opponent policy、长期 optimistic、短期 optimistic，定义见 [schema.py](../python/etazero/schema.py)；价值为当前玩家视角的 W/D/L 三分类 logits `[N,3]`。导出保留六平面的卷积计算形状，并向 native 返回主/短期 optimistic logits `[N,canvas²]`、主 WDL logits `[N,3]` 和误差标准差 `[N]`。训练用终局 WDL one-hot 的交叉熵；native 推理在 FP32 中 softmax 得到概率，`v = W-L`。网络在各前激活处应用有效棋盘 mask，BatchNorm 和池化排除 padding；实现见 [network.py](../python/etazero/network.py)。搜索保留网络原始 WDL、回传的 WDL 均值和 `W-L` 二阶矩；和棋概率在换视角时保持，胜负概率互换。

搜索时将已占用点和画布外位置 mask 掉，每个网络评估先将 logits 除以 `nn_policy_temperature`，再对可提交动作归一化先验；这个全树 policy 温度同时作用于根和所有非根。六项训练策略 softmax 的域均为有效棋盘上的全部落点：硬 policy 的已占用点目标为零，soft policy 经 epsilon 处理后占用点也有正目标；画布外点不参与归一化。这个搜索与训练的区别需要固定样例验证，不能由不同调用方各自选择。

`training.d4_augmentation` 控制训练 D4，baseline 开启。沿用 SkyZero_V8.1 的 batch 级均匀八对称，按整个画布变换全部空间通道（包括有效棋盘和禁手 mask）与 policy / opponent policy 目标；小棋盘的有效区域可以随变换移动到其他角。全局特征和 WDL 目标不变，原始轨迹不重写。变换发生在 learner 消费 batch 后、联合编译前，结果保持 contiguous；随机流纳入 Torch CPU RNG checkpoint，AMP overflow 消费当前 batch 后继续下一 batch，不重试该变换或前向图。实现见 [symmetry.py](../python/etazero/symmetry.py)。

搜索根由 `root_num_symmetries_to_sample` 控制 D4 集成，取值 1–8；大于 1 时从八种对称无放回均匀抽样，并绕过 NN cache 的读取和写入。变换覆盖整个画布的全部空间输入，全局输入不变，输出 policy 还原到原始动作坐标。每个对称先独立混合 ordinary/短期 optimistic logits，再执行全树温度及合法域 softmax，随后算术平均 policy 概率和 WDL 概率；不平均 logits，多个评估仍只构成一个根初始访问。根从非根子树推进而来时，多对称或 root/leaf optimism 不同时重新评估根，访问数不增加；新增预算为零的复用根也刷新条件。

单对称根、叶节点、cheap 根及 Match 由 `nn_randomize` 控制随机 D4；关闭时使用 `nn_symmetry` 指定未命中请求的朝向；已有 cache 仍复用首次输出，强制真实指定朝向须绕 cache。原生 NN 服务只在 cache 未命中后抽朝向，命中直接复用首次推理已还原的输出，不消耗 NN 朝向 RNG。根集成 RNG、NN 服务 RNG 与落子 RNG 分开；线程调度会影响请求次序，不承诺与来源随机序列逐位一致。cache 按未变换的完整输入、全局量、NN policy 温度及有效 optimism 分键，朝向不入 key；raw logits 存储在 canonical 坐标，温度由搜索 softmax 消费。raw 诊断明确指定 identity 并绕 cache，避免诊断调用固定搜索根的首个缓存朝向。实现见 [搜索](../cpp/src/search/search.cpp)、[AlphaZeroState](../cpp/include/etazero/algorithm.h) 与 [NN 服务](../cpp/src/inference/batcher.cpp)。

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

未访问边用 FPU，关闭 `use_fpu` 时为零。FPU 沿用 KataGo 的 visited-policy 插值：令 `m` 为已分配子节点的先验质量，`a=min(1,m^power)`，则 `FPU = a * parent_Q + (1-a) * NN_Q - reduction * sqrt(m)`，随后按 `fpu_loss_prop` 向 −1 插值。parent_Q 包含节点初始网络样本；根与非根 reduction、loss proportion 独立，零权重 cheap search 根使用非根参数。关闭 `fpu_parent_weight_by_visited_policy` 时，来源的固定分支为 `fpu_parent_weight * NN_Q + (1-fpu_parent_weight) * parent_Q`，之后仍扣 reduction 并应用 loss proportion；默认固定权重为零，即纯 parent_Q。power 允许零。动态 cpuct 的 log 项与父价值标准差项沿用来源公式，可由配置关闭。价值尺度为 W−L 的 [-1,1]，无 score utility。参数见 [selfplay.cfg](../configs/baseline/selfplay.cfg)、[eval.cfg](../configs/baseline/eval.cfg) 与 [match.cfg](../configs/baseline/match.cfg)。

节点更新使用 KataGo 的 `value_weight_exponent`：先执行适用的 noise pruning，按其剩余权重计算简单子价值均值，对每个子树用 `sqrt(1e-8 + 1/(1.5*sqrt(weight)))` 估计标准差，以自由度 3 的 t 分布 CDF（[-50,50] 上 2000 点线性插值表）加 0.0001 后取配置指数，偏重较好的子价值。noise pruning 关闭且根有噪声时，执行 configured chosen-move subtract/prune；noise pruning 开启时覆盖这个分支，即使 cap 为零。重分配后归一化到 noise pruning 后的剩余总权重，再加入本节点带 uncertainty 权重的网络样本。W−L、draw 和二阶矩使用同一重分配，权重平方和按缩放平方更新。指数为零是显式单位权重对照，不是默认搜索。实现见 [search_math.cpp](../cpp/src/search/search_math.cpp)。

根节点网络展开与边模拟次数分开计数。`search.full_search_visits` 为自对弈 full-search 根访问上限，cheap-search 使用独立访问上限；评估直接配置 visits。上限计入根初始网络样本和复用访问，只补足缺口。`initial_visits` 记录搜索前的保留量，`new_playouts` 包含首次根初始化，`simulations` 只计本手经过根边的新增模拟；`root_visits = initial_visits + new_playouts`。根集成多次 NN 请求仍只消耗一个首次根 playout；复用根重新集成不消耗 playout。终局子节点使用精确结果，不调用 NN。已结束五子棋根返回精确终局值和 action=-1，不照搬 Go 的 forceNonTerminal。

`max_playouts` 限制本手新 playout，`max_time` 是包含根推理的秒数；默认分别为 int32 最大值与 1e20，即无额外约束。访问／playout 限制可以让本手新增零次；时间限制仅在本手至少完成两个新 playout 后生效，显式停止可以在首次推理前生效。在途模拟完成回传后再返回，失败释放全部 pending。fresh 的零 playout 上限或立即停止返回 action=-1、零访问；只有初始根访问时按真实 NN 先验选择，不运行依赖子统计的 LCB。正常自对弈 cap 仍至少为 2，调用方拒绝没有落子的采样请求。

EtaZero 保留原有的严格并行 visit／playout 发放上限；KataGo 的并发循环可因线程在途越过计数上限。时间／显式停止均等待已在途路径完成，因此会有线程级尾部工作。此实现差异影响严格预算与等时间比较，不宣称并行访问序列或吞吐等价。

多线程的在途访问以 `pending * virtual_loss` 作为虚拟样本权重，价值向 −1 插值，PUCT 分母与 forced 配额使用同一虚拟权重；探索分子的总子权重只计完成统计。正式访问数、统计与训练目标只使用已完成结果。搜索返回前回收或取消全部在途工作，不能为了填满 batch 超出预算后又隐瞒实际完成量。

AlphaZero 可在同一模型代次内保留实际落子对应的子树。需要分别记录本次新增模拟量与复用后的总访问量，明确训练目标使用后者；新根的原始先验与探索先验分开保存，避免对已混噪声的先验再次混入噪声。规则、编码或模型变化后使树失效。跨手复用开关属于实验条件，应保存在生效配置中。不同模型身份的 Match 使用两棵树、默认保留命中子树；同一身份、同一模型路径与统一 Match profile 对应同 bot，双方共享一棵树且每手强制清树。固定局面 eval 默认不跨调用复用。`Search.set_evaluator` 更换模型时清除整棵树及线程状态；常驻 SP worker 按模型身份重建 evaluator，每轮固定模型的用户适配保持不变。

### 根探索、落子与训练目标

根先验依次执行全树温度、逐对称概率平均、根 policy 温度、Dirichlet 噪声。根 policy 温度按下述半衰期公式从 `root_policy_temperature_early` 衰减到 `root_policy_temperature`，概率取 `1/T_root` 次幂并重新归一化。训练 full search 再混入 `P_search = (1-epsilon) * P_tempered + epsilon * Dirichlet(alpha)`。噪声只覆盖可提交动作，`sum(alpha)` 为固定 total concentration；shaped 模式基于根温度后的先验，将一半浓度均匀分配，另一半按 `max(0, log(min(P,0.01)+1e-20)-mean_log)` 归一化分配；全零形状退化为均匀分配。关闭 shaped 模式仍使用相同总浓度的均匀 Dirichlet。每次从原始根先验生成温度与噪声，不累积变换，非根不加根噪声，评估和比赛不使用训练根噪声。

Playout cap randomization 按每手独立 Bernoulli 选择 full / cheap cap。正常搜索按 `clear_before_search` 清树；cheap target weight 为零时保留已推进到实际落子后的子树，关闭根噪声、根 policy 温度、多对称评估和 forced playout，使用非根 FPU 参数；全树温度继续生效。访问上限包含复用量，因此 cheap search 可不新增模拟。下一次 full search 重新清树。`reuse_tree=false` 显式关闭跨手复用，cheap 权重大于零时仍按正常清树与探索配置执行。开局和模型换代均重置搜索。默认开启清树；训练预算仍由实际采样行数规划。

Reduce Visits 与 PCR 的预算选择集中在 [search_limits.cpp](../cpp/src/selfplay/search_limits.cpp)，按 KataGo `play.cpp` 的分支顺序执行：命中 PCR cheap 时不再执行 Reduce Visits，不叠加预算缩减或目标权重。非 cheap 手在历史足够时，取最近 `reduce_visits_threshold_lookback` 手的完成搜索 W−L；历史统一为黑方视角，包括 cheap 与已缩减搜索，排除开局前缀和当前手，每局重新开始。令 `e=max(min(history),-max(history))`；只有同一方连续占优且 `e>reduce_visits_threshold` 才缩减，不能逐手取绝对值后把交替占优当作稳定局势。缩减比例为 `r=((min(e,1)-threshold)/(1-threshold))²`，访问 cap 为 `round(full_cap+r*(reduced_visits_min-full_cap))`，基础目标权重为 `1+r*(reduced_visits_weight-1)`。`round` 沿用 C++ 正半整数向上取整；cap 包含初始根评估，`full_cap=search.full_search_visits`。

缩减的 full search 保留正常清树、根噪声、根温度、根 FPU、多对称、forced playout 和训练 target LCB 设置，不标为 cheap；即使缩减权重为零，也不切换到 cheap 的清树与探索行为。默认清树时，它丢弃前一手 PCR 保留的树并使用自己的 cap；显式关闭清树时，cap 仍包含复用访问。参数属于自对弈配置；最低访问数独立于 cheap 预算，并须符合所选 full cap。当前自对弈配置要求 cap 至少为 2、lookback 至少为 1，保证存在已完成根边及非空历史窗口。评估和比赛保持独立固定预算，不启用此机制。五子棋没有动态 score utility center，无须移植来源为该中心执行的 10v 预搜索。

Forced playout 作用于已分配、先验为正的根子边：若其完成权重加虚拟样本权重小于 `sqrt(P_search * total_completed_child_weight * coeff)`，以强制优先级继续探索；虚拟权重防止线程重复追赶同一份配额。弱着这些额外访问不能直接当作训练目标。

Policy target pruning 先按 `child_weight*max(0,N-1)/max(1,N)+2*P_search` 选稳定子边，再用同一 `explore_scaling` 的 PUCT 分数反解其他子边所需权重：`ceil(min(child_weight, max(0, explore_scaling*P_search/(best_score-Q)-1)))`。分母非正时反解保留原权重，但所有非稳定边仍执行 `ceil`，包括 uncertainty 或图共享产生的非整数权重。稳定边保留原权重、不取整；非稳定边取整后再判断 LCB 资格。LCB 后统一执行 `chosen_move_prune` 与 `chosen_move_subtract`，两者各自上限为最大选择权重／64。原始 visits 始终独立保存，policy target 不要求等于 visits 归一化。

LCB 使用完成样本的 W−L 加权均值、二阶矩和 `ESS = weight_sum² / weight_sq_sum`。按 KataGo 加入最大方差先验，先验权重为 `weight_sum / ESS³`，更新两种权重和后重新计算 ESS，再计算 `LCB = Q-lcb_stdevs*sqrt(variance/ESS)`。仅剪枝后权重大于零、达到稳定边原始权重比例门槛的边可成为 LCB 最优边；将最优边权重提升到至少 `other_weight * ((radius+excess)/(radius+0.2*excess))²`。此处采用修正后的 LCB 索引行为，首个子边同样有效。

自对弈实际落子使用剪枝后、LCB 前的分布；训练 policy target 再应用 LCB。评估和比赛在落子分布和输出策略中都应用 LCB，分别由各自 profile 控制。落子温度为 `late + (early-late)*0.5^(turn/temperature_halflife*19/sqrt(board_area))`，手数包含开局。自对弈的 early/late 分别为 `temperature.temperature` / `final_temperature`，eval/match 为 `temperature_early` / `temperature`。温度只改变行为抽样，不改变监督；小于等于 1e-4 且无概率保护时，按来源选择首个最大权重子边；其余情况对权重执行稳定幂变换，`temperature_only_below_prob` 可保护超过概率阈值的部分，仅变换低概率尾部。根 WDL 和 Q 是包含初始网络样本的完成加权搜索均值，与训练用的真实终局 WDL 目标分别保存。评估输出 `network_policy` / `network_wdl` 为根集成结果，`search_policy` 为根温度和噪声后的先验，`policy` 为最终监督／选择策略。

这些步骤直接核对 [KataGo FPU/forced/pruning](/home/sky/RL/SkyZero/KataGo/cpp/search/searchexplorehelpers.cpp)、[noise/LCB](/home/sky/RL/SkyZero/KataGo/cpp/search/searchhelpers.cpp)、[selection](/home/sky/RL/SkyZero/KataGo/cpp/search/searchresults.cpp)、[子树价值加权](/home/sky/RL/SkyZero/KataGo/cpp/search/searchupdatehelpers.cpp) 和 [selfplay 调用边界](/home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp)，并以 SkyZero_V8.1、MuZero_V2 交叉核对。EtaZero 使用 W−L 子树价值加权，未引入 Go score utility、subtree value bias或专用推理算子；uncertainty weighting 与 noise pruning 见下节。随机引擎、并行锁调度和单对称身份评估沿用 EtaZero，不承诺来源逐位复现；t(3) CDF 使用数学等价闭式求值生成同网格插值表。

### 误差加权、optimistic policy 与 noise pruning

三项机制由 [SearchSettings](../cpp/include/etazero/search.h)、[共享配置校验](../python/etazero/config.py) 和 [search_math.cpp](../cpp/src/search/search_math.cpp) 定义。SP、Eval 与 Match 分别配置 uncertainty、noise pruning 和 root/leaf optimism，具体启用状态和数值以所选配置为准。所有参数进入生效配置及模型运行记录；可显式消融，不把 `policy_target_pruning` 作为 noise pruning 的开关。

设真实 short-term W−L 误差标准差为 `u`，NN 样本权重为 `c/(u^p+c/wmax)`。纯 W−L utility 的 Go score 导数为零，省去 score 误差项。关闭 uncertainty 或 evaluator 明确无辅助能力时，权重为 1；支持辅助能力的 evaluator 必须返回真实有限非负标准差，不用常数或静默回退替代。多个 D4 先平均标准差，再计算一次权重。初始 NN 一次访问贡献 `weight=w`、`weight_sq=w²`；真实终局不用 NN，在支持 uncertainty 的搜索中每次访问贡献 `wmax` 和 `wmax²`，N 次访问的二阶权重为 `N*wmax²`。因此访问、权重和 ESS 是不同统计。virtual loss 仍按 `pending*virtual_loss`，不乘 uncertainty 权重。

optimism 混合普通与短期 optimistic head 的 logits：`main+(short-main)*optimism`，使用固定来源后端的 float 运算顺序。混合在温度及合法域 softmax 前完成，每个 D4 独立处理后平均概率。主 WDL 保持普通价值预测。root/leaf 系数不同时，晋升根重新推理并替换初始 NN 样本；已有正式访问和 child 统计保留，不把条件刷新算作新增 playout。根多对称绕 cache，单对称按有效 optimism 分键。缓存保留不可变 raw 输出，混合在搜索侧执行，不污染其他条件请求。无辅助能力时有效 optimism 归零；同一个 batched evaluator 的所有 backend 能力声明及实际输出必须一致。当前发布契约要求四个真实输出，历史不符契约的模型直接拒绝。

noise pruning 按 child 激活顺序处理，不排序。以前缀已剪权重及 utility 累积得到均值；若当前 utility 较差、差值 `gap>0`，且 `W>2*Wprefix*P/Pprefix`，扣除 `min((W-2*Wprefix*P/Pprefix)*(1-exp(-gap/scale)),cap)`。先验下限、单 child 和近零总权重分支沿用来源。之后 value weighting 的均值、标准差和归一化基准均使用剩余权重，W−L/draw/二阶矩共用缩放；不能把减少的总权重重新补回。图父边先按 edge/child 访问比例换算，再执行该聚合；LCB 与父聚合各自保留不同 weight-square 缩放。

[check_katago_search_corrections.py](../tests/reference/check_katago_search_corrections.py) 编译执行固定 KataGo 的 uncertainty、noise pruning 和后端 float logit 混合原函数/语句，与本地结果独立对照。C++ 手算覆盖非单位 NN/终局权重、D4 非线性顺序、晋升根零预算刷新、无辅助能力、noise cap 与根 chosen-prune 竞争。真实 CUDA 验收覆盖真实误差、ordinary/optimistic head、图并发和显式开启的 SP。输入 cache 使用 NN 温度和 optimism 的 exact double 字节；来源 NN hash 分别按 1/2048 与 1/1024 离散这些条件。EtaZero 沿用完整输入精确缓存而更细分条件，明确不宣称 cache 命中率、随机序列、并发调度或性能等价。上述检查不代表棋力或训练效果。

### 图搜索与局面共享

[Search](../cpp/src/search/search.cpp) 以分段加锁的局面表拥有节点，根和父边借用节点指针；多个父边可以指向同一个子节点。SP 的 `[graph_search]` 及 Eval/Match 自身字段 `use_graph_search` 默认开启，`graph_search_catch_up_leak_prob` 默认零、允许 [0,1]。关闭图共享时，每条分支仍创建独立节点，NN cache 可另外开关。

[AlphaZeroState.graph_key](../cpp/include/etazero/algorithm.h) 比较完整局面字节而非仅比较哈希：包含实际尺寸、画布、执子、手数、棋规、终局标志、胜者、终局原因和棋子；哈希只决定表分段。搜索固定模型和参数，换模型会清图。NOVC 五子棋每手只增加棋子，没有 pass、提子、ko 或历史判重；不适用来源 Go 的 recent-history 折叠及 `graphSearchRepBound`，不提供不起作用的配置。当前搜索始终提供完整禁手特征，其输入由局面与棋规确定。未来增加影响 continuation 或 NN 输入的条件时须同步扩充 key；不支持身份契约的状态适配器会明确拒绝图搜索。

节点访问数与父边访问数独立。令 `r=edgeVisits/max(childVisits,1)`，父边完成权重为 `child.weight*r`；选择/LCB 的二阶权重为 `child.weight_sq*r`，父节点聚合则以 `r²` 缩放原始子节点二阶权重，再应用价值重加权的比例平方。这两个分支直接对应固定来源 `searchnode.h` 与 `recomputeNodeStats`，不能相互替代。访问追赶在下降前读取子节点访问：若边落后且未命中泄漏概率，用 CAS 将边访问加一，更新父节点而不新增子节点访问；命中泄漏则继续下降。每次只加一，未启用来源已注释的追赶比例分支。

virtual loss 使用共享子节点的全部在途预约；完成策略及价值只计正式访问，异常释放路径的边、节点和全局 pending。路径重复时按来源增加该边访问并终止本次 counted playout，不假设 childVisits 总大于 edgeVisits。NOVC 实际对局不会重复局面，通用状态适配器的循环分支仍受保护。每手的严格并行发放预算保留 EtaZero 定义；来源竞争发布重试、分字段原子统计与 dirty-counter 聚合，和此处展开等待、串行聚合/锁快照有调度差异，不宣称并行序列或等时间性能等价。

根始终是独立节点，推进时复制命中子节点的统计、先验和父边，保留其共享后继；终局或未完成展开的节点不作为可复用根。所有搜索线程静止后从新根标记可达节点，删除每个未标记节点一次，大图使用常驻线程分摊删除；循环和多父不会递归析构或重复释放。不兼容局面、reset、完整清树及模型切换会清除旧图。只读 `inspect_graph()` 在同步调用返回后提供节点/父边统计和稳定ID，用于检查共享与回收。

独立检查入口为 [check_katago_graph.py](../tests/reference/check_katago_graph.py)：执行固定来源的 child 权重、追赶函数和聚合循环，与逐步 diamond 统计对照。C++ 图测试另覆盖实际五子棋不同落子顺序、关闭共享、泄漏率、多线程、循环、终局、故障和大图推进/回收。真实 CUDA 的图共享验收关闭 NN cache，因而不会将 NN 输出复用误记为节点共享；这些检查不证明训练效果或吞吐改善。

### 学习与数据使用

每个训练样本包含当前输入、当前手搜索 policy、下一实际手搜索的 opponent policy、opponent 可用权重与终局 WDL，来源为 [原始对局分片](implementation.md#原始对局分片)。四项 policy 的机制对齐 [KataGo metrics_pytorch.py](/home/sky/RL/SkyZero/KataGo/python/katago/train/metrics_pytorch.py) 与 [trainingwrite.cpp](/home/sky/RL/SkyZero/KataGo/cpp/dataio/trainingwrite.cpp)：opponent 直接预测实际落子后下一手的搜索分布，不翻转执色或空间坐标，也不使用当前局面的对手视角网络先验。下一手即使是零采样权重的 cheap search，目标仍有效；末手没有后继搜索时，保存可归一化的均匀占位目标，令 opponent 权重为零。当前手的重复采样次数作用于整行，不额外乘下一手的采样权重。

先分别归一化两项硬目标，再按 `soft(p) = normalize(((p + 1e-7) * on_board_mask)^0.25)` 生成 soft 目标；平滑域包含有效棋盘内已占用点，不含 padding。五子棋没有 Go 的 pass 动作。opponent 固定系数为 0.15，soft 系数 `s` 由 `training.soft_policy_weight_scale` 指定，默认采用 KataGo `train.py` 的 8。记 `w_next` 为后继搜索可用的 0/1 权重、`CE` 为交叉熵：

```text
L_base = mean_t[0.72 * CE(WDL_target, WDL_pred)
           + 0.930 * CE(policy_target, policy_pred)
           + 0.15 * w_next * CE(opponent_target, opponent_pred)
           + s * CE(soft(policy_target), soft_policy_pred)
           + 0.15 * s * w_next * CE(soft(opponent_target), soft_opponent_pred)]
```

可选v17 Q预测当前玩家视角的逐动作纯W−L。搜索遍历全部已分配root children，取正visits/weight的child **node visits**与纯价值均值（包含终局子节点），按edge的玩家转换符号；不使用edge visits、policy target或包含score的utility；并发graph catch-up与在途回传也可令node/edge计数不同，刚清树不保证相等。side独立搜索与reanalysis同样提取；不改变真实轨迹或hint首值复制规则。writer按每个最终重复输出行独立将float32 Q×32000进行无偏随机量化并限于±32000；node visits限于[0,32000]，无数据点两个目标均为0。learner只还原int16/32000，不重抽量化或重复乘频率。

令 `z` 为第七输出pre-tanh，`n` 为量化后的node visits，`w=sqrt(n)`、`m=(n!=0)`、`p=(1+Q_target)/2`。来源纯W−L分量为 `L_Q = mean[1.5 * sum_a(w_a * BCEWithLogits(2*m_a*z_a,p_a)) / (1+sum_a w_a)]`；无数据行loss与Q梯度为0，side行即使完整主局标志为0也参与。W−L的tanh解释不在loss前显式计算。score Q因五子棋无对应语义而去除；Q不供native搜索消费。量化使用每局独立固定种子流，避免影响对局/禁手增强随机流；删去Go score量化及其Rand后不宣称来源随机序列相同。独立对照入口为 [check_katago_q.py](../tests/reference/check_katago_q.py) 与 [Q测试](../tests/test_qvalues.py)。

优化器参照固定来源 [KataGo train.py](/home/sky/RL/SkyZero/KataGo/python/train.py) 的 fson 分支，默认 SGD（momentum 0.9），支持 AdamW（CUDA fused）。输入权重、残差权重、残差/输入 BN gamma、相应 bias、head 权重及 head bias 分组，所有参数必须恰好归组一次；最终 masked BN 的 gamma/bias 按输出组处理。具体 LR、WD 系数、batch scaling、warmup 和范数自适应公式集中在 [optimization.py](../python/etazero/optimization.py)，不在文档维护第二套常量表。Transformer fixup/ReLU按七组规则单独处理attention衰减与RMSNorm；未接入 Muon/NorMuon/Aurora、自动或自定义 LR schedule。

总 loss 为上式加三个 TD、长期/短期 optimistic 和短期价值误差分量的平均值，启用Q时再加上述Q分量；实际反向使用 `batch_size * L`，与来源的 batch 求和尺度一致。默认梯度上限按优化器、batch 与 LR scale 计算；`training.gradient_clip > 0` 则以平均梯度尺度覆盖上限。梯度日志仍为裁剪前平均 loss 的梯度范数。SGD 的衰减加入优化器梯度，AdamW 的衰减解耦施加，均不重复加入 loss。价值目标来自真实终局，未完成对局不会凭空补零价值。

Lookahead 同步时将 fast 权重向 slow 权重平均，LR 按 alpha 补偿；每个分段开始重置同步计数但不复制权重；完整训练轮结束丢弃未同步的 fast 权重，保留优化器状态、消费计数和成功更新计数。`training.sub_epochs`默认1，可将实际batch预算按floor等分为非空分段，实际步数较少时减少段数；分段计数与整轮batch计数分别保存，LR/WD和范数采样仍沿整轮时钟。该单遍与训练额度预算是用户适配，不照抄来源文件概率预算。来源epoch末复制slow后到下个subepoch才重置counter，本地轮末立即将counter归零；下一段的优化执行一致。中断 learner 保存 fast、slow 和同步计数，恢复继续同一周期。SWA 使用来源的指数平均规则，仅在 Lookahead 同步后采样，首个样本直接复制；BN buffers 在采样时直接复制，不参与参数平均。采样累积跨轮保留，未得到 SWA 样本时发布当前模型。

LR/WD 在轮开始刷新，之后按本轮消费 batch 数每 5 批刷新，累计消费超过 2 亿样本后改为每 50 批；刷新使用消费后的计数，作用于下一 batch。范数在更新前测量，默认每 100 batch 取 snapshot，先进入统计再刷新 WD、执行 Lookahead 与 SWA。关闭 `optimizer.norm_only_at_print` 时逐 batch 累积均值，在打印点把历史和及权重同时缩为 0.001；来源的 `norm_*_batch` 没有逐 batch 的 0.995 EMA 衰减。`optimizer.lookahead_print` 只在 Lookahead 周期起点累积该分支的范数，原始 loss 日志仍逐 batch 保存。

FP16 overflow 时 GradScaler 跳过 optimizer 更新并降低 scale，消费样本、Lookahead 同步计数与 SWA 累积仍推进；因此同步可能发生在跳步 batch，成功更新数另存。learner的训练与validation autocast只用于主干，heads与loss的实际运算保持FP32；`model.eval()`只切换归一化等训练行为，不改变heads精度。独立导出副本的heads跟随推理精度配置。

主局 TD 在完整轨迹上构造，使用实际棋盘面积的三种指数时间尺度；先把搜索 WDL 转成固定黑方视角，累加到真实终局，再还原每个训练位置的玩家视角。每个 TD loss 为有效系数 0.72 的交叉熵减目标熵。误差预测和 source surrogate backward、无 score optimistic 权重及完整主局有效性定义见 [辅助监督契约](../../EtaZero.md#alignment-targets) 与 [network.py](../python/etazero/network.py)。短期误差 target 和 optimistic 权重中的预测量均 stop-gradient；禁手失利仍是正常真实终局。

搜索保留归一化前的最终选择权重，将最大值至少放大到 10、超出 30000 时统一缩小，然后按 C++ round 写入 int16。learner 读取后除行和，再生成 soft 目标；不从概率乘固定整数推回原权重。末手无效 opponent 保存全 1 占位，两个对应 loss 权重为零。重复行已在 view 展开，不再乘 surprise 频率。

Side 数据使用独立观测、policy、visits 和搜索 WDL，不伪造该分支终局。主 value 和三个 TD 都监督其自身搜索概率，opponent 及完整主局标志为零；默认 optimistic 和 error loss 因完整数据 gate 关闭。来源 disable_optimistic_policy 分支改用 0.5 的普通 policy 权重，仍保留两个头。writer/view/shuffle/reader/learner 均支持实际生成的 side。

适配差异：D4 使用可恢复的 Torch CPU RNG，不复现来源 NumPy 种子的逐位随机序列。所有 loss 日志记录乘系数后的 batch 均值，十一项之和为 total；KataGo 的 soft loss 日志在乘 soft 系数之前记录。没有 pass、元数据来源筛选或 Go 专有监督，不宣称复现完整 KataGo 网络与训练。

learner 从该轮窗口快照中最多读取一遍，训练步数受 `train.cfg` 的基准上限、新增数据额度和完整 batch 数限制，保存完整训练状态并发布推理模型；零步轮次保留当前模型。自对弈产量与训练预算见 [训练额度与自对弈产量](implementation.md#训练额度与自对弈产量)。额外一致性损失未接入。

### PDA、侧分支与重分析

Hint与game fork的生产入口为 [sampling.cpp](../cpp/src/selfplay/sampling.cpp)，预算入口为 [search_limits.cpp](../cpp/src/selfplay/search_limits.cpp)。`[hint_positions]` 无外部语料时概率0；开启须指定有效文件。每个非注释行格式为 `size rule sampling_weight hint_local_index prefix_count prefix_local_indices...`；索引采用该行实际尺寸的零起始行优先编号，载入时转成canvas编号。支持三种当前NOVC棋规；从空盘完整重放prefix，拒绝非法/终局前缀、占用提示、无正抽样质量和多余字段。抽位置按非负weight（来源hint的lambda=0）；没有SGF、设置子、Go komi/seki或额外的corpus训练权重，本范围每局有效性保持默认1。相对文件相对于所选配置目录，内容SHA256进入生效配置fingerprint；native每次产样请求读取文件后校验该哈希，解析已校验的同一份字节。运行中改动语料会明确失败，改语料不能沿用原续训配置。

精确hint要求提示起始turn、棋盘、执子与棋规匹配。该手visits和显式有限playouts乘4，跳过PCR和Reduce Visits，然后再施加PDA；无限playout保留无额外上限，int32溢出明确拒绝。根清图，先执行根policy温度/噪声，再按来源float操作从所有合法prior移出2%质量给hint。根探索在其边权重加下一次预期权重仍小于最重其他边的0.8倍时强制探索，含pending virtual weight；移除hint时清图。后6手、hintFork起点后6手cheap概率减半，六手边界恢复原概率；局后force-full reanalysis跳过hint/PCR，保留原历史的Reduce Visits和PDA。hint强制探索的首个训练search WDL由下一手正确翻转视角的值替换，只有一手则用真实终局；在reanalysis前和替换后刷新，原始行动及历史预算仍保留。

Early/game fork按来源先抽early，失败才抽late；baseline概率及候选范围以 [selfplay.cfg](../configs/baseline/selfplay.cfg) 为事实源。early位置为 `floor(Exp(1)*expected_move_prop*area)`，late从完整历史（含前缀）均匀抽位置；非空历史的索引统一截到 `min(index,历史长度-1)`，从空盘重放该数量的着法，保留最后一手落子前的局面；空历史从空盘开始。early指数尾部因此仍进入候选评估；重放后已终局则丢弃。候选数量在配置范围均匀抽，按来源`chooseRandomLegalMoves`从实际合法空点有放回选择；候选可以重复，空点少于请求数时仍抽满请求数，无合法点则丢弃fork。以对手视角的普通NN `-(W-L)` 排序选择当前方最好的一着，这是纯WDL对来源`whiteScoreMean`的环境映射；普通NN温度1、无optimism、无PDA，终局候选仍可被排名但被选为终局后拒绝fork。未选hint则另行走hint着，非终局产生hintFork。冷启动使用其当前random evaluator，网络阶段使用该轮固定模型。

对局线程共享随机取出并移除的fork池，池跨同一native worker的轮次及模型释放/更换保留，进程重启重建；不保证跨并发调度或重启复现同一fork顺序。池中位置优先于外部hint抽样。Fork跳过平衡开局、policy init和PDA抽样；外部hint跳过平衡/policy init，仍按普通局抽PDA。完整历史按独立initial前缀保存，前缀不训练；原始记录分开保存balanced/policy/initial计数、起点kind与hint action，不能将initial计成平衡或policy开局。hintFork没有再次强制hint，仅执行六手cheap减半。来源完整预算函数3072组合与C++独立prior/访问/续接检查见 [check_katago_forks.py](../tests/reference/check_katago_forks.py)、[sampling_test.cpp](../cpp/tests/sampling_test.cpp)。

PDA 在普通局以配置概率抽样：优势方均匀黑/白，doubling 值 `d` 均匀于 `[0,log2(max_ratio))`。令 `r=2^d`，先执行 PCR 或 Reduce Visits，再把优势方 visits/playouts 乘 `2r/(1+r)`，另一方乘 `2/(1+r)` 并 round；结果低于来源下限 5 或超出可表示范围明确失败。EtaZero 的 int32-max playout 字段表示无额外上限，仍保持无额外上限；显式有限值执行同一倍数。PDA 每手强制清图，输入随真实玩家翻转符号，在所有搜索深度、D4、cache、数据和模型导出中保持。当前 `Game` 保存条件，完整图身份包含它。默认 Eval/Match 的 `playout_doubling_advantage=0`；显式设置是固定搜索条件，预算仍由该 profile 决定，用于条件评估。五子棋无 handicap/komi，不加补偿或 PDA 后再次平衡的研究方案。

Side 在主局搜索后，以配置概率排除实际着，按来源 70% 网络温度 1、25% 温度 2、5% 均匀空点抽替代着；提前胜/禁手失利/满盘的候选丢弃。主局结束、重分析完成后处理侧分支队列：清除 PDA，独立 full search，保存自己的观测、policy、visits 与 WDL，初始频率 1。分支选择回复着使用恢复后的 LCB；以 0.25 概率续走回复着，再用普通 NN 温度 1 抽另一替代着，无 ban，非终局加入队尾。每次递归至少加两子，有限 NOVC 棋盘不需要截断深度。侧数据无主局终局、opponent/error/默认 optimistic 监督，主 value/TD 都取分支搜索 WDL；没有冒充完整对局或新增 plies/games。

`reanalysis.use_reanalyze` 默认 false；开启必须提供 proportion、policy/value surprise 权重、指数和 outcome-target 开关。先对 cheap 位置逐项 Bernoulli 得到选取数，再按加权 surprise 的幂无放回选同样数量；全零权重均匀抽样，按时间排序处理。零比例不消耗选择 RNG。每个位置从原始真实局面开始，force-full 跳过 hint 与 PCR，Reduce Visits 只用该手之前的原始完成搜索历史，PDA 保持该局条件；替换监督及 NN/search 统计并恢复 counterfactual full 的基础权重，不改变真实 action/reward/终局。重分析行的 `used_outcome_targets=false` 按来源关闭下一实际手 opponent policy，并令该行及所有重复输出行的 `full_game_weight=0`，关闭 short-term error 和默认 long/short optimistic 监督。主 value、三个 TD 仍按主局完整轨迹及已替换搜索值生成，Q 保持有效；`disable_optimistic_policy=true` 分支仍使用来源固定 0.5 权重训练两个 policy head。使用 outcome targets 的重分析行及普通主局行保持 `full_game_weight=1`。重分析的 original visits、原选择 policy/value surprise 及目标开关随原分片保存。

生产入口为 [sampling.cpp](../cpp/src/selfplay/sampling.cpp)、[预算](../cpp/src/selfplay/search_limits.cpp) 与 [writer](../cpp/src/selfplay/record.cpp)，独立来源公式检查为 [check_katago_sampling.py](../tests/reference/check_katago_sampling.py)。完整hint/PCR/reduced/PDA优先级检查见 [check_katago_forks.py](../tests/reference/check_katago_forks.py)，真实core频率重分配对照见 [check_katago_sampling_weights.py](../tests/reference/check_katago_sampling_weights.py)。采样使用 EtaZero 每局 RNG，不承诺来源种子逐位一致，也不声明学习效果或等时间吞吐。

### Policy / value surprise weighting

每局结束后计算采样期望权重，来源是 KataGo `play.cpp` 的主局路径，可在局后重分析后重算。Policy surprise 为最终 policy target 相对于实际搜索先验（含根噪声）的 KL。Value surprise 从真实终局 WDL 开始，以 `now=1/(1+board_area*0.016)` 向前平滑后续搜索 WDL，再与该手原始网络 WDL 计算 KL，限制到 [0,1]；计算采用统一黑方视角，存储采用当前玩家视角。

`use_search_value_surprise=true` 改为当前手 search WDL 对 raw NN 的直接 KL，不使用真实终局或未来平滑。重分析先按旧 surprise 选择位置，再替换 policy/NN/search/基础权重，之后重算 surprise；原始选择值单独保存。

初始普通 full 权重为 1，PCR cheap 权重来自配置，Reduce Visits full 权重来自预算缩减的同一比例。以这些初始权重计算平均 surprise，policy 重分配量为 `weight*policy_surprise + (1-weight)*max(0,policy_surprise-1.5*weighted_mean_policy_surprise)`；value 量为 `weight*value_surprise`。两者分别归一化到初始总权重，按配置比例与原始权重混合；平均 value surprise 小于 0.010 时同比减小 value 混合比例。启用 reanalysis 时，尚未重分析的 cheap 行关闭 excess-policy 恢复项；已重分析且被 Reduce Visits 降权的行仍可恢复。分母下限沿用来源 1e-10，完全零 surprise 的对应份额会衰减，不伪造均匀 surprise。开局前缀始终排除，不能因 surprise 获得监督。

期望权重可大于 1，按 `floor(weight)+Bernoulli(frac(weight))` 随机取整，只执行一次并保存 `row_repeats`。训练视图按该次数重复样本；当前 policy、opponent policy 与 visits 仅保存正次数位置，每个位置一份，通过 `sample_indices` 对应完整轨迹。零次数位置保留轨迹与轻量诊断；其 policy 若作为采中前一手的 opponent 目标，仍随前一手保存。shuffle 不重新抽样这些权重，训练 loss 不再额外乘同一权重。replay、窗口和产样预算按实际重复后的行数计数，因此不会把大量 cheap 行当作完整监督。随机取整使用每局 RNG；原始权重、两种 surprise、网络/搜索 WDL、cheap 标记及实际次数均保存供核对。

## 算法接口与后续设计

### 公共边界

| 边界 | AlphaZero | MuZero |
|---|---|---|
| 根输入 | 真实观测 | 真实观测经过 representation |
| 树内状态 | 真实棋盘状态 | 学习得到的 latent state |
| 转移与叶评估 | 棋规转移，再调用策略价值网络 | dynamics / recurrent 推理及 prediction |
| 树内终局与 mask | 真实规则给出 | 按所移植 MuZero 的定义，不从真实棋盘偷取树内信息 |
| 样本使用 | 单状态监督 | 连续动作序列与多步展开目标 |

当前 AlphaZero 搜索执行器负责访问统计、并行、预算和回传调用；[SearchState / AlphaZeroState](../cpp/include/etazero/algorithm.h) 负责真实状态、转移和叶推理。MuZero 使用独立的 latent 搜索与推理请求实现，按语义复用数学函数，不将 latent 塞入 AlphaZero 节点。数据服务保留真实轨迹，但 AlphaZero 的采样搜索数组压缩与单行训练视图不适用于 MuZero 连续展开。MuZero 为连续展开保留所有后续目标，具体行为见 [MuZero](muzero.md)。

`SearchResult` 应区分实际动作、训练策略目标、根价值、访问统计和实际预算。Gumbel 接入后允许其训练目标与访问频率不同，不能把公共接口命名为“visit policy”后强制所有算法共用。

### MuZero

网络、latent 搜索、组批、轨迹展开、完整运行链路及 KataGo 工程机制适用性统一维护在 [MuZero](muzero.md)。

实现依据包括 [MuZero V2 算法定义](/home/sky/RL/MuZero/MuZero_V2/docs/algorithm.md)、[网络](/home/sky/RL/MuZero/MuZero_V2/python/muzero/network.py)、[搜索](/home/sky/RL/MuZero/MuZero_V2/cpp/src/search.cpp) 和 [展开目标](/home/sky/RL/MuZero/MuZero_V2/python/muzero/replay.py)。该工程是特定的棋类 MuZero，区分其基础预设与增强预设，不把二者混在一起移植。

完整接入必须覆盖 representation、dynamics、prediction、初始与 recurrent 组批、latent 生命周期、搜索备份、展开长度、价值 / 策略目标、梯度缩放以及终局与展开末端处理。真实环境只负责实际动作执行和监督来源，不能在 latent 树内调用 AlphaZero 的棋规转移代替 dynamics。

通用 MuZero 定义包含对奖励、价值和策略的学习；具体棋类适配是否使用 reward 分支、采用何种目标和 mask，以用户指定的移植来源为准。框架接口不能因第一轮棋类 AlphaZero 而永久写死零奖励或真实终局检测。[MuZero 原文](https://arxiv.org/pdf/1911.08265)

### Gumbel 根搜索与非根搜索

Gumbel 的根机制覆盖 Gumbel-Top-k 候选、Sequential Halving 预算分配、动作决策、未访问动作的 completed Q 以及由改进策略生成的训练目标。非根 `gumbel` 指论文式 14 的确定性选点，不是每层重新加噪声。[Gumbel 论文 §§3–5](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)

根为 Gumbel、非根保留 PUCT 是本项目支持的组合；论文附录 Figure 7 比较了保留原非根选点和使用新非根选点的版本。切换非根策略不能把根训练目标退回访问频率目标。[论文附录 Figure 7](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf#page=16)

后续实现参考作者团队的 [Mctx action selection](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/action_selection.py)、[Q transforms](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py) 和 [policies](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)，实施时固定所用提交。候选不足、极小预算、轮次余数、并列分数和非法动作的处理都属于完整移植范围，不能仅增加一个 Gumbel 采样步骤就宣称算法完成。

实现阶段分别验收根调度、Q completion、改进策略目标、非根策略及其与 AlphaZero / MuZero 模型的组合。实验收益和各组合的比较方案由用户安排。
