# EtaZero 算法与训练核查表

## 基准与读法

本表先从来源源码填写公式、分支与 train/eval 行为，再逐行对照 `EtaZero_V0`。来源配置与 EtaZero 当前配置分别记录；公式相同不代表整个调用链相同。静态核查包含目标工作区的未提交改动。

- **KataGo 源码**：`/home/sky/RL/SkyZero/KataGo`，commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`；读取时工作区干净。
- **统一 selfplay 参数基准**：[selfplay8mainb18.cfg][KG-SP]。[training/README.md][KG-CONFIG] 将它定义为 public distributed 主线切换到 b18c384nbt 后的参考配置；`selfplay1.cfg` 是早期小网络示例，不混入其数值。这是仓库的主线参考配置，不能宣称是线上服务器此刻的实时配置。
- **加载后默认值**：`selfplay` 使用 `SETUP_FOR_OTHER`，`match` 使用 `SETUP_FOR_MATCH`，追到 [setup.cpp][KG-SETUP]，不采用 `SearchParams` 构造函数初始值。在线 `contribute` 使用 `SETUP_FOR_DISTRIBUTED` 与服务器任务配置，是另一条入口。
- **统一 Eval/Match 参数基准**：[match_example.cfg][KG-MATCH] 加 `SETUP_FOR_MATCH` 的默认值。它是比赛示例 profile；analysis/GTP 的额外默认行为不混入。来源比赛温度非零，不能写成“所有eval默认零温度”。
- **Learner / shuffle**：该 commit 的 `train.py`、`model_pytorch.py`、`metrics_pytorch.py`、`shuffle.py` 为机制来源；解析器和派生默认值为参数来源。selfplay配置不定义网络架构、LR或replay窗口；没有线上训练命令，不能推断线上learner参数。shuffle脚本示例值单独注明。
- **KataGomo 专有机制**：`/home/sky/RL/SkyZero/KataGomo`，commit `df152116e3787c75c6a3de099d261ca092b7dfc1`；读取时工作区干净。平衡开局、禁手输入dropout、五子棋policy init使用它的源码；KataGo的Go对应行为另外说明。
- **目标配置**：V0 [baseline selfplay][EZ-SP]、[eval][EZ-EVAL]、[match][EZ-MATCH]、[train][EZ-TRAIN]、[env][EZ-ENV]、[net][EZ-NET]。数值不代表其他profile或已有实验的历史配置。

目标核查快照：EtaZero Git HEAD `be587ad32d71ccdf29c2ebd9b4f93e9bf94bd8ce` 加当前工作区；本表引用的26个目标实现/配置文件（不含独立核查脚本）集合SHA256为 `e83d53b8012b67dc67dfa8901f28eaa9174f258c442bedbe047b901045c7b4b1`（按仓库相对路径排序，对每行`path + NUL + 文件SHA256十六进制 + LF`求SHA256）。源码改变后应重核相关行，不能仅凭HEAD沿用结论。

Train分为 **SP（自对弈产数据）**、**L（learner更新）**；Eval分为固定局面搜索与Match。公式转换到行棋方视角；KataGo白方正视角、score/no-result与五子棋WDL的差异见A01。

结论标签：**一致（静态）**、**参数差异**、**语义差异**、**环境适配**、**部分实现**、**未实现**、**明确不实现**。静态一致表示已核对源码及调用点，不表示完整运行验证或训练效果通过；不同性质的结论可并列。

## 审查结论与使用边界

57行均已对照来源和目标代码填写；独立核查脚本与复跑范围见“核查证据”。优先处理的区别如下，编号可定位详细行及源码：

| 类别 | 已确认的区别 | 对结果的含义 |
|---|---|---|
| 当前实现的数值偏离 | N11：fson+SGD自动裁剪基数5500，来源2500；当前batch=128时3889.09 vs1767.77 | 梯度超过来源阈值时，实际更新不同；没有证据说明当前运行触发频率 |
| 训练目标选择 | L01/L02：ordinary policy系数1 vs现代版本0.930；基础value CE为1 vsCLI默认有效0.72；缺少optimistic/TD等辅助监督 | 已实现的是五项基础损失子集，系数选择和删去辅助监督需分别讨论，不能直接断言训练效果好坏 |
| 抽样与推理语义 | D02：每轨迹状态抽禁手dropout，重复row共享；S17/A03：单朝向固定identity、cache按朝向分开、集成不跳cache | 重复样本相关性及搜索方向随机性不同；Learner D4不能覆盖全部区别 |
| 参数profile差异 | S02/S07/S12–S22：400/70而非2000/350v；reduced min20而非350；cpuct、温度、总噪声浓度、Match价值重加权等不同 | 这些是当前配置选择，不能仅凭“不同”标成公式bug，也不能称主线参数复现 |
| 功能缺口 | S08/S09/S25–S28/N03：PDA、side position、graph、uncertainty、optimistic、noise pruning、Transformer未实现；N01独立plain网络未实现 | 有的来源SP启用，有的来源Match默认启用，应按具体阶段讨论；S23 subtree bias按原清单明确不实现 |
| 数据与发布节奏 | N09/N10/R01/R04/R05/A03/A04：LR/WD刷新、随机数据封顶、近期排序、固定整轮训练/发布/恢复和轮内模型切换不同 | 即使局部公式相同，也不构成完整异步KataGo训练链路等价 |

**Eval应关闭的训练行为**：PCR、Reduce Visits、训练surprise频率重分配、训练输入dropout、learner D4、optimizer/warmup/Lookahead在线更新；根噪声和forced按本表选定Match profile关闭。**仍可保留的搜索行为**：FPU、policy selection pruning、LCB、value weighting、tree reuse、随机朝向NN；来源Match还默认启用variance scaling、uncertainty、optimistic policy和noise pruning，其中uncertainty/optimistic依赖网络支持。SWA是模型发布方式，使用SWA模型评估不等于在eval中更新SWA。

本表覆盖原清单和直接相邻的语义边界，**不是整个KataGo仓库的穷尽证明**。主线配置还包括early/game forks、Go komi/handicap/seki/lead及结束阶段等，五子棋不直接照搬；来源还支持hint/SGF、reanalysis、metadata、Muon与更多网络/调度分支。未列为已实现项的能力不能由名称相近推定存在。首轮完成的是源码审查；文档结构/引用及纯函数对照检查见下方“核查证据”，GPU训练、并行搜索数值和断点续训尚未在本次重新运行。

## 1. 搜索、落子与自对弈采样

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train（SP） | KataGo Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| S01 FPU | 为未访问动作赋初始价值 | [F1][FORMULA-F1]：visited-policy插值、sqrt扣减、loss比例插值 | 非根reduction=0.2、普通根=0；visited-policy开启、power=2；cheap零基础权重根用非根参数 | 非根0.2、根0.1；visited-policy开启、power=2；loss proportion默认0 | [FPU][KG-FPU]、[cheap][KG-CHEAP]、[加载器][KG-SETUP] | **一致（静态）＋环境适配**。SP非根/普通根0.2/0，Eval/Match 0.2/0.1，power=2、loss_prop=0；cheap零基础权重改用非根参数。纯W−L使用radius=1。仅支持visited-policy这一分支。[公式][EZ-FPU-CODE]、[选点][EZ-SIMULATE]、[配置加载][EZ-SETTINGS] |
| S02 Playout cap randomization（PCR） | 混合便宜推进手与完整监督手 | 每手Bernoulli(q)选cheap；基础频率还会经surprise重分配 | q=0.75，full/cheap=2000/350v，cheap基础权重0；hint/fork可改变概率或强制full | Match不走SP的PCR预算分支 | [预算][KG-CHEAP]、[主线配置][KG-SP] | **一致（静态）＋参数差异**。SP full/cheap=400/70v，q=0.75、基础权重0；cheap去根噪声/根温度/forced/根集成且不清树，保留NN温度；可被surprise恢复采样。Eval/Match无PCR。hint/fork例外未实现。[预算][EZ-LIMITS]、[调用][EZ-SP-CALL]、[根处理][EZ-RUN] |
| S03 Forced playout | 优先探索根部尚未达到配额的动作 | W_a<sqrt(k·P_a·T)时强制优先；W含virtual weight | k=2；cheap零基础权重关闭 | 默认k=0，关闭 | [选点][KG-FORCED]、[cheap][KG-CHEAP] | **一致（静态）**。SP k=2，使用search prior和含virtual weight的子边权重；cheap关闭；Eval/Match配置加载不赋forced，保持0。[分数][EZ-SEARCH-MATH]、[加载][EZ-SETTINGS] |
| S04 Policy target pruning | 削减事后看来不必要的探索权重 | [F2][FORMULA-F2]：找稳定边，再用PUCT反解其他边所需权重 | 根落子分布与policy target都执行；没有独立同名配置开关 | 根同样执行，即使forced系数=0；不是eval自动关闭项 | [选择权重][KG-SELECTION]、[PUCT反解][KG-FPU] | **一致（静态）**。稳定边、PUCT反解与ceil均对应来源；SP落子/target及Eval/Match均执行，当前pruning=true。增加可关闭开关属于消融扩展；Go ending/pass处理按环境移除。[实现][EZ-SELECTION] |
| S05 LCB | 偏重价值下界更可靠的动作 | [F3][FORMULA-F3]：加权ESS、方差先验、LCB与选择权重提升 | 开启，z=5、门槛0.15；实际SP落子暂时禁用，生成target时恢复；useNonBuggyLcb=true | 默认开启、z=5、门槛0.15；参与落子，修正索引默认开启 | [LCB][KG-LCB]、[选择][KG-SELECTION]、[SP调用][KG-RUNBOT] | **一致（静态）＋环境适配**。SP实际落子禁LCB，target用LCB；Eval/Match落子用LCB；z=5、门槛0.15，候选index=0有效。加权ESS和radius²修正一致；省去Go ending bonus。[实现][EZ-SELECTION]、[调用分流][EZ-RESULT] |
| S06 Tree reuse／清树 | 推进命中子树，按调用方决定是否清空 | maxVisits含已有访问，maxPlayouts限制新增；原始先验与根探索先验分开 | 普通SP清树；cheap基础权重0时不清树；PDA强制清树；模型变化更新搜索状态 | 不同bot的Match默认不清树；相同botIdx双方强制清树；可配置clearBotBeforeSearch | [SP分支][KG-CHEAP]、[GameRunner][KG-RUNNER]、[预算][KG-BUDGET] | **一致（静态，SP）＋参数差异（Match）**。SP reuse=true、普通搜索clear=true，cheap零基础权重不清树；推进后根探索先验重置。Eval/Match reuse=false，每次新搜索，不采用来源不同bot Match的默认子树保留；模型只在轮间换。[推进][EZ-ADVANCE]、[预算][EZ-LIMITS]、[比赛][EZ-MATCH-CALL] |
| S07 Reduce visits | 一方持续占优时低预算、低权重完成对局 | e=max(min(h),−max(h))；r=((min(e,1)−t)/(1−t))²；cap=round(V+r(Vmin−V))；w=1+r(wmin−1) | t=0.9、lookback=3、Vmin=350、wmin=0.1；与PCR为else-if，同一白方视角历史 | 不使用；Match硬认输是独立设置 | [缩减分支][KG-CHEAP]、[配置][KG-SP] | **一致（静态）＋参数差异**。SP t=0.9、lookback=3、Vmin=20、wmin=0.1；PCR与缩减互斥，历史存game.player()*value，统一黑方正视角，不把双方各自胜率混在一起。缩减full保留探索；Eval/Match不走此分支。[公式][EZ-LIMITS]、[历史][EZ-SP-CALL] |
| S08 Playout doubling advantage（PDA） | 非对称预算和条件输入共同表示算力优势 | d对应预算比2^d；双方cap系数为2·2^d/(1+2^d)、2/(1+2^d)；输入按执色变号 | 普通局概率0.01、让子局0.5、最大预算比8；可补偿komi；清树；side rows无PDA | 搜索支持该条件，普通Match默认d=0 | [PDA][KG-PDA]、[输入][KG-INPUT]、[side][KG-SIDE] | **未实现**。SP无预算优势抽样，输入无PDA条件，亦无清树/komi补偿分支；不能由两个不同visits配置冒充。普通Eval/Match无该功能。[搜索配置][EZ-SETTINGS]、[输入][EZ-GAME] |
| S09 Side position | 搜索实际落子之外的替代局面，增加反驳监督 | 排除实际着后抽替代着；独立搜索提取policy和搜索value；分支不必下到终局 | 概率0.020；主线旧名forkSidePositionProb由loader映射；初始权重1；可再分支 | 不生成训练side rows | [采样／搜索][KG-SIDE]、[旧名映射][KG-PLAYSETTINGS] | **未实现**。SP仅实际轨迹，writer/训练视图没有side rows及搜索value target；开局prefix属于非监督轨迹，不是side position。Eval/Match不生成训练行。[生产][EZ-SP-CALL]、[writer][EZ-WRITER]、[targets][EZ-VIEW] |
| S10 Policy surprise weighting | 多训练网络先验没有预料到的结果 | KL(target∥含噪声搜索先验)；[F4][FORMULA-F4]局内频率重分配 | 比例0.5；cheap/reduced超过1.5倍均值可恢复权重；reanalysis改变cheap例外 | 不重分配训练数据、不改当前落子 | [KL][KG-SURPRISE]、[重分配][KG-WEIGHTS] | **一致（静态，普通path）**。SP alpha=0.5；KL对含噪声search prior，按基础频率重分配，cheap零基础权重可恢复；不再次乘loss权重。无reanalysis扩展；Eval/Match不执行。[KL][EZ-RESULT]、[重分配][EZ-WEIGHTS] |
| S11 Value surprise weighting | 多训练价值预测没有预料到的后续结果 | 终局开始倒序平滑搜索概率，再对原始NN做KL，截[0,1]；[F4][FORMULA-F4] | 比例0.1；now=1/(1+0.016·area)；均值<0.010时同比减弱；useSearchValueSurprise默认false | 不用于eval落子或数据重分配 | [value KL][KG-VSURPRISE]、[重分配][KG-WEIGHTS] | **一致（静态）＋环境适配**。SP beta=0.1，倒序平滑/截断/低surprise减弱一致；搜索及原NN WDL统一黑方视角。只实现默认未来平滑分支，无useSearchValueSurprise开关；draw为五子棋和棋。Eval/Match不执行。[实现][EZ-WEIGHTS] |
| S12 基础weighted PUCT | 平衡价值与先验探索 | Q_a+c(T)f_sigma·P_a·sqrt(T+0.01)/(1+W_a)；W是统计权重 | c0=1.05；Q为Go综合utility | c0加载默认1.0；同样weighted PUCT | [探索][KG-FPU]、[配置][KG-SP]、[默认][KG-SETUP] | **一致（静态，纯W−L公式）＋参数差异**。SP/Eval/Match c0=1.5；weighted PUCT分母用weight，探索总量排除virtual weight。Go score utility未移植，见A01。[公式][EZ-SEARCH-MATH]、[选点][EZ-SIMULATE] |
| S13 Log-scaled cpuct | 搜索越大，探索系数越高 | c(T)=c0+c_log·log((T+b)/b) | c_log=0.28，b=500 | c_log默认0.45，b=500 | [探索系数][KG-FPU]、[加载器][KG-SETUP] | **参数差异**。公式存在，三种profile均c_log=0、b=500，即当前关闭；KataGo主线SP=0.28、Match默认0.45。[公式][EZ-SEARCH-MATH]、[SP][EZ-SP]、[Eval][EZ-EVAL]、[Match][EZ-MATCH] |
| S14 Variance-scaled cpuct | 按utility波动尺度调节探索 | [F5][FORMULA-F5]：f_sigma=1+λ(σhat/σ0−1) | SETUP_FOR_OTHER下λ=0关闭；σ0=0.40、先验权重2；主线cfg未覆盖 | SETUP_FOR_MATCH下λ=0.85开启；σ0=0.40、先验权重2 | [方差估计][KG-FPU]、[入口默认][KG-SETUP] | **一致（静态，公式）＋参数差异**。SP/Eval/Match prior=0.25、weight=1、scale=0；SP关闭状态与来源相同，Match也关闭，来源Match默认0.85。prior/weight在scale=0时不改变最终探索系数。[公式][EZ-SEARCH-MATH]、[加载][EZ-SETTINGS] |
| S15 Value weight exponent | backup时降低较差子树影响 | [F6][FORMULA-F6]：t(3)CDF重加权、保留总权重；η=0关闭；anti-mirror另有例外 | η=0.5；带噪声根且无noise pruning时先chosen prune/subtract | η默认0.25，仍保留机制 | [聚合][KG-AGGREGATE]、[重加权][KG-VWEIGHT] | **一致（静态，纯W−L子集）＋参数差异**。三种profile eta=0.5；SP对应来源，Eval/Match不同于来源默认0.25。t(3)同范围/点数插值，重分配保留总权重，二阶矩/weight_sq同步缩放；无anti-mirror例外。[聚合][EZ-AGGREGATE] |
| S16 Root symmetry ensemble | 多朝向概率平均，降低方向偏差 | 无放回均匀抽K个D4；还原policy后平均policy/value概率；不平均logits | 普通根K=4、cheap零基础权重K=1；根多对称跳过NN cache | 根K=1，但这一次NN请求仍可随机朝向 | [多对称][KG-SYM]、[root调用][KG-NNHELP] | **一致（静态，K>1集成）＋语义差异**。SP普通根K=4，cheap/Eval/Match K=1；无放回D4、policy与WDL概率平均一致。K=1固定identity，来源可随机；K>1未跳cache，见A03。[集成][EZ-EVALUATE-NODE]、[坐标][EZ-STATE]、[cache][EZ-CACHE] |
| S17 叶节点随机D4 | 随机朝向推理并还原输出 | 未指定symmetry且nnRandomize=true时均匀抽0…7；cache命中不重新抽 | nnRandomize=true | match_example同样true；不是eval必须关闭项 | [随机朝向][KG-RANDOMSYM]、[cache][KG-NNCACHE] | **未实现／语义差异**。叶节点、cheap根、Eval/Match的单朝向推理固定identity；没有来源nnRandomize抽样。Learner D4和根K=4不能替代该行为。[K=1路径][EZ-EVALUATE-NODE]、[叶节点调用][EZ-EXPAND] |
| S18 NN policy temperature | 改变全树网络先验尖锐程度 | 合法域softmax(logits/Tnn)；Tnn进入cache hash | 默认Tnn=1，根/非根均生效 | 默认Tnn=1 | [输入参数/hash][KG-INPUT]、[NN后处理][KG-NNCACHE] | **一致（静态）＋参数差异**。SP Tnn=1.1，Eval/Match=1；cheap仍保留1.1。目标cache存raw logits，温度在cache外作用，所以无需把Tnn加入raw-output key；不误标为cache bug。[softmax][EZ-EVALUATE-NODE]、[cache][EZ-CACHE] |
| S19 Root policy temperature | 根部进一步展平先验 | normalize(P^(1/Troot(t)))，先于噪声；[F7][FORMULA-F7]衰减 | 1.5→1.1、half-life参数19；cheap零基础权重改为1 | 早/晚均默认1 | [根温度][KG-NOISE]、[衰减][KG-TEMPERATURE] | **一致（静态）＋参数差异**。SP 1.35→1.15，half-life=12；cheap=1；Eval/Match 1→1、half-life=19。同尺寸归一化衰减公式，顺序在D4概率平均之后、噪声之前。[衰减][EZ-SEARCH-MATH]、[根处理][EZ-RUN] |
| S20 Chosen move temperature | 将搜索权重变成实际动作分布 | normalize(Wselect^(1/Tmove(t)))；小温度/概率保护边界见F7 | 0.75→0.15、half-life参数19；不改变policy target | match_example为0.60→0.20；无配置覆盖的loader默认0.5→0.1 | [落子][KG-CHOSEN]、[温度][KG-TEMPERATURE]、[Match][KG-MATCH] | **一致（静态）＋参数差异**。SP 0.75→0.15、half-life=12；Eval/Match=0（确定性按最大选择权重落子），区别于来源Match示例0.60→0.20。policy target不乘落子温度；onlyBelowProb=1。[温度][EZ-SEARCH-MATH]、[落子][EZ-RESULT] |
| S21 Chosen move prune/subtract | 选择前剔除极低权重、统一扣减 | cutoff=min(prune,maxW/64)，subtract=min(subtract,maxW/64)；低于cutoff置零 | prune=1、subtract=0；LCB后执行，也参与特定根价值聚合 | 默认同样1/0 | [选择末尾][KG-SELECTION]、[聚合][KG-AGGREGATE] | **一致（静态）**。SP/Eval/Match prune=1、subtract=0；选择时LCB之后；噪声根backup前也执行，cheap无噪声不触发该backup分支。当前backup无noise-pruning竞争分支。[选择][EZ-SELECTION]、[backup][EZ-AGGREGATE] |
| S22 Shaped Dirichlet noise | 将部分噪声集中在相对有希望的动作 | [F8][FORMULA-F8]：alpha一半均匀一半按截断log-policy分配，再混根先验 | total concentration=10.83、epsilon=0.25；来源默认shaped，无独立shaped开关；cheap零基础权重关闭 | rootNoise默认false | [alpha与混合][KG-NOISE]、[配置][KG-SP] | **一致（静态）＋参数差异**。SP total=6.75、epsilon=0.25、shaped=true；cheap/Eval/Match关闭。半均匀/半中心化log-policy对应来源；增加uniform开关用于消融。[alpha][EZ-NOISE-CODE]、[混合][EZ-RUN]、[配置][EZ-SP] |
| S23 Subtree value bias | 在线学习相似局部战术的NN价值偏差 | Vcorrected=Vnn+β·Δsum/Wsum；子树权重指数写模式表，释放时衰减 | β=0.30、指数0.8；额外表/锁/释放行为 | 默认β=0.45、指数0.85、free proportion=0.8；并非默认关闭 | [修正/更新][KG-AGGREGATE]、[默认][KG-SETUP] | **明确不实现**。按原清单本阶段不做；当前无模式表、偏差累积和释放衰减。来源SP和Match均非零，故完整搜索不会等价；这项不作为遗漏修复任务。[当前聚合][EZ-AGGREGATE]、[范围][EZ-CONFIG-CODE] |
| S24 Virtual loss | 临时降低并发在途分支吸引力 | v=pending·loss；Q←Q+(Qloss−Q)v/(v+max(0.25,W))；W←W+v；分子只计完成 | loss=1、threads=1；最终监督不计pending | loader默认loss=1；同一机制 | [虚拟权重][KG-FPU]、[默认][KG-SETUP] | **一致（静态）＋环境适配**。SP/Eval/Match virtual_loss=1；pending进入均值与分母，朝纯W−L下界−1插值，探索分子只用已完成权重；正常/异常退出释放pending。baseline search_threads=1，常态无并发虚拟权重。[分数][EZ-SEARCH-MATH]、[并行][EZ-SIMULATE] |
| S25 Graph search／转置 | 相同状态共享节点统计 | 状态/历史hash、重复边界、循环/追赶；child权重按edgeVisits/childVisits换算 | useGraphSearch=true | SETUP_FOR_MATCH默认true | [图搜索][KG-BUDGET]、[child统计][KG-NODE]、[默认][KG-SETUP] | **未实现**。当前Node由单父树拥有，无局面hash转置表/共享节点/图循环处理；Tree reuse和NN cache分别重用子树与NN输出，不构成graph search。SP与Eval/Match均为树。[结构][EZ-SEARCH-HEADER]、[扩展][EZ-EXPAND] |
| S26 Uncertainty-weighted playout | 可信NN评估在聚合中占更高权重 | w=c/(u^p+c/wmax)，u含短期winloss/score error；缺少error heads则w=1 | SETUP_FOR_OTHER默认关闭 | SETUP_FOR_MATCH默认开启；c=0.25、p=1、wmax=8 | [样本权重][KG-UNCERTAINTY]、[默认][KG-SETUP] | **未实现**。自有NN样本固定weight=1，未训练短期error heads；SP默认关闭这一点对应来源，Eval/Match不能启用来源现代网络默认的uncertainty weighting。[初始样本][EZ-EVALUATE-NODE]、[heads][EZ-HEADS] |
| S27 Optimistic policy | 用偏好好结果的辅助策略改变先验 | backend混合logits：l=(1−o)lmain+olopt；训练监督随版本变化 | root/nonroot optimism默认0，但现代版本训练辅助头 | Match默认root=0.2、nonroot=1；旧模型不支持时消去该输入 | [搜索输入][KG-NNHELP]、[backend][KG-OPTIMISM]、[loss][KG-METRICS] | **未实现**。四个policy头是ordinary/opponent/soft/soft-opponent，输出落子只用ordinary，无optimistic训练目标与logit混合。SP搜索optimism=0的效果对应来源，但现代learner及Match默认辅助策略缺失。[heads/loss][EZ-HEADS]、[导出][EZ-EXPORT-CODE] |
| S28 Noise pruning | 在价值聚合中削弱被过度探索的差分支 | 相对前缀utility均值差且W>2·Wprefix·P/Pprefix时，扣除excess·(1−exp(−gap/scale))，受cap限制 | SETUP_FOR_OTHER默认关闭；与S04的选择pruning独立 | SETUP_FOR_MATCH默认开启；scale=0.15、cap=1e50 | [聚合剪枝][KG-NOISEPRUNE]、[默认][KG-SETUP] | **未实现**。聚合仅chosen prune/subtract与value weighting，没有来源按价值差/过度探索扣权分支；SP默认关闭对应来源，Eval/Match缺少默认开启行为。policy_target_pruning=true不补上它。[聚合][EZ-AGGREGATE] |

## 2. 开局、环境与输入

| ID／机制 | 直观理解 | 来源公式／流程 | 来源Train | 来源Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| D01 Balanced opening（KataGomo） | 随机骨架后挑接近平衡的最后一着 | 骨架长度权重{10,30,50,80,60,40,20,10,5,1,0,0}；空点权重∝Σ(1+center)/(distance²+d²)²，d=0.8·Exp(1)；候选权重(1−v²)^b；负根价值与最大候选权重控制拒绝 | Gomo概率0.99、b=4、拒绝率0.995；失败次数>20后降到0.8并继续重试。本表核对NOVC开局分支；来源另支持VC1/VC2的不同骨架长度分布。KataGo无此五子棋几何开局，Go用komi等 | Gomo Match概率1、b=10；随机选botB或botW评估；不等同成对执色协议 | [开局][GM-OPENING]、[调用][GM-PLAY] | **一致（静态，单评估器核心）＋比赛协议差异**。SP概率0.99/b=4，Match 1/b=10，其余0.8/0.995/0.8/>20重试一致，max_tries为降拒绝率门槛。Match固定task.generator的模型完成整个开局，并复用开局交换执色；Gomo每次balance尝试随机选botB/W。无VCN。[开局][EZ-OPENING-CODE]、[比赛调用][EZ-MATCH-CALL] |
| D02 Forbidden-plane dropout（KataGomo） | 训练时随机隐藏禁手提示 | 写每个row时useForbiddenInput~Bernoulli(0.5)；平面与特征可用标志同步改变 | Gomo训练writer每行抽样；仅Renju编码禁手；KataGo无禁手平面 | useForbiddenInput默认true，搜索保留提示 | [writer][GM-WRITE]、[编码][GM-INPUT]、[常量][GM-CONSTANT] | **语义差异（重复行粒度）**。SP每个Renju轨迹状态一次Bernoulli(0.5)，同步隐藏平面与标志，再按row_repeats复制；来源每个输出row重新抽样，所以同局面重复行的dropout相关性不同。搜索/Eval/Match保留提示；其他规则无禁手输入。[抽样][EZ-SP-CALL]、[编码][EZ-GAME]、[重复][EZ-VIEW] |
| D03 Policy init | 纯网络policy随机走开局，增加状态覆盖 | Gomo n=max(0,floor(Exp(1)·mean−2·已有手数))、0.0002均匀分支；mean是抽样公式参数，不保证固定开局手数；Go n=floor(Gamma(k)·area·prop/k)，k=1为指数 | Gomo独立于balanced，SP必须显式配置init开关，mean loader默认12、T默认1；Go主线init=true、prop=0.08、k默认1、T默认1；还含komi/结束逻辑 | Gomo Match init默认false；启用后必须显式给mean（无loader默认），T默认1；Go Match默认init=true、prop=0.04，区别于SP | [Gomo init][GM-POLICYINIT]、[SP加载][GM-PLAYSETTINGS-SP]、[Match加载][GM-PLAYSETTINGS]、[Go init][KG-POLICYINIT]、[Go loader][KG-PLAYSETTINGS] | **一致（静态，Gomo公式）＋参数差异**。SP mean=5/T=1，Match mean=6/T=1.6；减2×已有手数和0.0002均匀分支一致。SP after/failure=true；Match after=true/failure=false。Match policy阶段整段使用开局generator模型，来源按执色使用botB/W；Go面积/komi分支未移植。[init][EZ-POLICY-INIT-CODE]、[调用][EZ-MATCH-CALL] |
| D04 Multi-board size | 一套网络混合尺寸，以mask排除padding | 尺寸按权重抽样；空间层mask；pool和BN按有效面积；value含尺寸条件 | Go尺寸7/9/11/13/15/17/19/8/10/12/14/16/18；权重1/4/3/10/7/9/75/1/2/4/6/8/10；矩形概率0.10 | Match示例19/13/9、权重90/5/5；固定局面按自身尺寸推理 | [抽样][KG-GAMEINIT]、[mask/pool][KG-MODEL]、[配置][KG-SP] | **环境适配＋参数差异**。SP尺寸15/14/13/12/11，权重100/10/5/3/1，canvas=15；Eval/Match默认15。有效点mask、面积pool/最终BN具备对应机制，仅方形，未支持来源矩形抽样。[环境][EZ-ENV]、[输入][EZ-GAME]、[pool/BN][EZ-NORM-CODE] |
| D05 Multi-rule | 条件化网络学习不同真实棋规 | 规则影响转移、终局和全局输入，不能只改标签 | Go ko=SIMPLE/POSITIONAL/SITUATIONAL、scoring=AREA/TERRITORY、tax=NONE/NONE/SEKI/SEKI/ALL、suicide=false/true、button=false/false/true；Gomo basicRule/VCN另按其配置 | 比赛可混合或固定规则；输入条件仍完整 | [Go抽样][KG-GAMEINIT]、[Go输入][KG-INPUT]、[Gomo输入][GM-INPUT] | **环境适配＋参数差异**。支持renju/freestyle/standard，并条件化全局输入；baseline权重1/0/0，实际仅Renju，不是当前同时混训三规则。Eval/Match默认Renju，规则影响终局/长连/禁手失利；Go规则及VCN未移植。[环境][EZ-ENV]、[真实转移][EZ-GAME] |

## 3. 网络、归一化与优化

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train（L） | KataGo Eval／模型发布 | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| N01 Convnet（普通残差） | 两层空间卷积与残差学习局部形状 | x→x+Conv3(ActNorm(Conv3(ActNorm(x))))；部分块带global pooling | 类别由model-kind/block_kind定义，不由selfplay cfg定义 | 同架构；归一化按推理模式 | [ResBlock][KG-RESBLOCK]、[配置][KG-MODELCONFIG] | **部分实现**。内部有普通两卷积残差块，但没有独立plain ResNet架构选择；当前模型始终NBT。不能把NBT内层残差块写成已实现独立Convnet对照。[内层][EZ-NBT-CODE]、[模型入口][EZ-NETWORK-CODE] |
| N02 Convnet+NBT | 在较窄通道内堆残差块，降低计算 | 1×1降通道→K个内部残差块→1×1升通道→外残差；NBT2的K=2 | 主线架构系列含b18c384nbt；核对Eta结构的具体预设为b5c192nbt-fson-mish，不混称b18参数 | 同结构；导出可只保留所需heads | [NBT][KG-NBT]、[b5][KG-B5]、[后缀][KG-SUFFIX] | **一致（静态，指定预设子集）＋环境适配**。C192/B5、mid96、gpool32、head32、value-hidden80，NBT2，outer第2/4块gpool；trunk对应b5c192nbt-fson-mish。输入5平面/4全局、四policy与WDL heads是五子棋子集，不是b18全模型。[NBT][EZ-NBT-CODE]、[全网][EZ-NETWORK-CODE]、[宽度][EZ-WIDTHS] |
| N03 Transformer+NBT | 在瓶颈中用注意力连接远距离落点 | softmax(QKᵀ/√d+mask)V；RMSNorm、2D RoPE、FFN/SwiGLU；NBT2/3按架构 | modelconfigs显式选transformer；不将其等同mainb18时期架构 | 同结构推理，attention排除padding | [NBT Transformer][KG-TRANSFORMER]、[attention][KG-ATTENTION]、[配置][KG-MODELCONFIG] | **未实现**。当前配置没有Transformer架构，未包含attention/RoPE/RMSNorm/FFN；原清单这项保留为能力缺口，不能用global pooling等价替代。[配置能力][EZ-CONFIG-CODE]、[全网][EZ-NETWORK-CODE] |
| N04 初始化、固定尺度、最终masked BN | 固定残差尺度稳定网络，只在末端用BN | fson内部y=((1+γoffset)sx+β)mask；外块入口1/√(i+1)、内部入口1/√(j+1)、NBT出口1/√(K+1)、末端1/√(B+1) | 方差校正截断正态；最终BN有效点总体mean/std，epsilon=1e−4、EMA=0.001；EMA对象为std | running_mean/std推理，导出折入scale/bias；内部fixed scale始终保留 | [NormMask][KG-NORM]、[NBT初始化][KG-NBT]、[全网初始化][KG-INITIALIZE] | **一致（静态，fson二值mask）**。内部可学gamma offset/beta、固定scale、Mish gain及2σ截断方差校正对应来源；最终BN按有效点总体mean/std、EMA=0.001、epsilon=1e−4；Eval用running统计并折叠。非二值mask不是当前协议支持范围。[init/norm][EZ-NORM-CODE]、[NBT][EZ-NBT-CODE]、[推理折叠][EZ-INFERENCE-NET] |
| N05 Global pooling与heads | 注入全盘与尺寸信息 | policy pool=[mean,mean(√A−14)/10,max]；value pool=[mean,mean(√A−14)/10,mean((√A−14)²/100−0.1)] | 训练policy有opponent/soft/optimistic等；value另有score/ownership等Go heads，依版本 | 导出搜索所需policy/value，不用全部训练heads落子 | [pool/heads][KG-MODEL]、[导出][KG-EXPORT] | **一致（静态，pool）＋环境适配／部分实现（heads）**。pool面积公式及gpool bias路径对应来源；Mish下合法有效点max与来源padding −1屏蔽等价。移除pass/score/ownership等Go heads，optimistic与TD未实现；搜索输出ordinary+WDL。[pool][EZ-NORM-CODE]、[heads][EZ-HEADS] |
| N06 Learner D4 | 八对称训练减少方向过拟合 | 每batch均匀抽g，同步变换空间输入/targets，global/value不变 | reader randomize_symmetries=true | validation=false；搜索D4是S16/S17 | [reader][KG-D4]、[train调用][KG-TRAIN] | **一致（静态）**。L每batch均匀抽一个D4，对obs/当前policy/opponent policy同步变换，global和WDL不变；当前开启。Eval/Match无learner增强，推理D4见S16/S17；未设独立validation流程。[变换][EZ-D4-CODE]、[调用][EZ-TRAIN-LOOP] |
| N07 SWA（EMA型） | 平滑参数，形成发布模型 | 首样本复制；之后θs←θs+(θ−θs)/scale；BN buffers复制，非参数平均 | 默认scale=8、period=max(1,samples_per_epoch//2)；只在Lookahead同步后采样，跨epoch保留 | 导出SWA权重；SP与eval均使用发布模型，无在线平均 | [EMA][KG-SWA]、[采样][KG-LOOKAHEAD]、[导出][KG-EXPORT] | **一致（静态）＋调度适配**。L scale=8，period=64000样本（1000×128/2），只在Lookahead同步点采样，BN buffers复制；跨轮保存SWA与计数。SP/Eval/Match均加载发布模型，首次SWA前用raw模型；绝对周期随训练预算缩小。[优化状态][EZ-OPT-CODE]、[发布选择][EZ-OPT-INFERENCE] |
| N08 Lookahead | 定期平均fast与slow权重 | 每k步slow←slow+α(fast−slow)，fast←slow；LR除以α补偿 | 默认k=6、α=0.5、sub_epochs=1；α≥1关闭；每个subepoch开始重置同步计数，epoch末丢弃未同步fast，保留优化器 | 使用发布权重，无在线优化 | [同步][KG-LOOKAHEAD]、[计数重置][KG-LOOKAHEAD-COUNTER]、[LR][KG-LR] | **一致（静态，默认单subepoch同步路径）＋调度适配**。L k=6/alpha=0.5，LR除alpha；轮内连续计数，轮末恢复slow、丢弃未同步fast并清零计数，保留optimizer。按Eta一轮对应来源一个单subepoch的epoch核对；未实现多subepoch中途重置，非k整数倍的分段可改变同步及SWA采样时点。Slow/counter可续训；Eval/Match用已发布权重，不在线平均。[同步][EZ-OPT-STEP]、[轮末][EZ-TRAIN-LOOP] |
| N09 Warmup | 随已消费训练样本数提高LR | 每25万样本分母20/14/10/7/5/3/2/1.4；200万后1 | 默认开启；global_step_samples包含DDP世界大小；同时进入WD公式 | 不运行LR调度 | [warmup][KG-LR] | **一致（公式）＋刷新节奏差异**。L warmup开启，按total_steps×128统计成功更新消费样本，8段分母和200万结束点一致；Eta每步刷新，来源每5/50 batches刷新，阈值跨越时生效手数可不同。Eval无LR调度。[warmup][EZ-WARMUP-CODE]、[调度][EZ-OPT-CONFIGURE] |
| N10 优化器／分组／自适应WD | 按职责更新和衰减，稳定网络范数 | [F9][FORMULA-F9]：SGD/AdamW、head/noreg组、范数ratio自适应衰减，norm_kind决定分支 | SGD默认momentum0.9；head LR factor=0.5；SGD每样本LR=3e−5；fson属于BN型WD分支；当前默认每100批norm快照，关闭only-at-print时取running_metrics的EMA | 优化器仅续训使用 | [WD][KG-WD]、[LR][KG-LR]、[group roles][KG-MODEL]、[norm采样][KG-NORM-TIMING] | **一致（静态，fson组公式）＋刷新节奏差异**。L默认SGD，momentum0.9、head_factor0.5、noreg1、Lookahead0.5；AdamW LR也乘sqrt(batch/256)，与来源一致。基线范数均为初始化L2；当前来源默认每100批norm快照，Eta亦100批；Eta每步更新WD，来源5/50批。来源关闭only-at-print时的EMA扩展未移植。[分组/公式][EZ-OPT-CODE]、[调度][EZ-OPT-CONFIGURE] |
| N11 梯度裁剪与精度 | 限制异常更新；精度与跳过更新影响计数 | cap=c√(global_batch/256)/√max(1e−7,LRscale) | SGD fson/fixup/fixscale c=2500；SGD BN/brenorm/fixbrenorm c=5500；AdamW c=11000；FP16/BF16独立开关 | 主线SP CUDA FP16=true；Match后端auto；训练AMP/推理精度独立 | [裁剪][KG-GRAD]、[AMP][KG-TRAIN]、[SP配置][KG-SP] | **语义差异（当前SGD裁剪）＋参数差异／AMP分支差异**。L clip override=0时默认SGD基数5500，来源fson应2500；batch128/scale1下cap≈3889.09 vs1767.77（2.2倍）。当前L AMP off、SP/Eval/Match float32。开启FP16时Eta重试同batch/同forward且成功才计更新，来源skip后仍推进样本计数。[裁剪][EZ-GRAD-CODE]、[更新][EZ-TRAIN-LOOP]、[配置][EZ-TRAIN] |

## 4. 监督目标与样本权重

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train | KataGo Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| L01 硬policy、opponent、soft policy | 当前/对手策略与展平辅助目标 | soft(p)=normalize(((p+1e−7)·on_board)^0.25)；opponent系数0.15、soft默认8 | 四基础heads之外还有optimistic/Q；ordinary CE系数版本≤11为1、12…17为0.930；opponent不可用权重0 | 使用导出的policy，不在线计算loss | [目标/版本][KG-METRICS]、[opponent CE][KG-POLICYLOSS]、[writer][KG-WRITE] | **部分实现＋系数差异**。L四policy头，opponent0.15、soft8/soft-opponent1.2，soft epsilon/power/domain一致；ordinary系数1，来源b5 version15为0.930。无optimistic/Q监督。opponent来自下一实际手，末手权重0；目标存float32而非来源int16量化权重。[loss][EZ-LOSS-CODE]、[目标][EZ-WRITER] |
| L02 Value target/loss | 终局与搜索辅助监督学习价值 | 基础CE内部1.20再乘value_loss_scale；CLI默认0.6，有效0.72；另有TD/score等loss | 主局真实终局与多时间尺度targets；side可用搜索value | Go第三类no-result，真实draw影响W/L分配；不同于五子棋draw分类 | [CE][KG-VALUELOSS]、[targets][KG-WRITE]、[总loss][KG-METRICS]、[CLI][KG-TRAIN] | **环境适配＋系数差异／未实现辅助目标**。L终局行棋方W/D/L one-hot，CE系数1；来源CLI默认基础CE有效0.72（1.20×0.6）。无多时间尺度TD/search/score目标；第三类draw不能与Go no-result直译等同。Eval终局按真实棋规返回。[target][EZ-VIEW]、[CE][EZ-LOSS-CODE]、[终局][EZ-GAME] |
| L03 搜索合法域／训练softmax域 | 搜索屏蔽非法动作，训练仍抑制棋盘内差着 | 搜索legal mask；训练仅屏蔽padding，soft目标对占用点也有epsilon质量；Go含pass | policy目标归一化，规则/价值含义保留 | 搜索legal域继续生效，五子棋无pass | [mask][KG-METRICS]、[policy head][KG-MODEL]、[NN后处理][KG-NNCACHE] | **一致（静态，域区分）＋环境适配**。搜索legal=棋盘内空点；Renju禁手可落但判负，不因dropout改变规则。L仅屏蔽padding，occupied/禁手仍在softmax；soft epsilon覆盖on-board点。五子棋无pass。[规则][EZ-GAME]、[推理softmax][EZ-EVALUATE-NODE]、[loss][EZ-LOSS-CODE] |
| L04 采样次数、loss归约、计数 | 频率只重分配一次，避免重复加权 | count=floor(w)+Bernoulli(frac(w))；重复整行；CE按样本求和反向，日志再归约 | 普通path先随机取整再write；distill/reanalysis另有数据权重 | visits不同于训练行或梯度更新 | [取整][KG-ROUND]、[writer][KG-WRITE]、[sum loss][KG-METRICS] | **一致（静态，基础path）**。SP w仅用于随机取整频率，训练视图repeat后不再次乘w；L日志mean、backward乘batch恢复sum，与来源subset求和梯度尺度一致。total_samples计成功batch；AMP开启后的计数差异见N11。[取整][EZ-WEIGHTS]、[repeat][EZ-VIEW]、[backward][EZ-TRAIN-LOOP] |

## 5. Replay窗口与shuffle

四个参数控制近期输入窗口与shuffle输出抽样，`4Parameters`是组说明，不是第五个trick。KataGo还支持taper_window_scale/max_rows/add_to_data_rows等，四参数是子集。

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train | Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| R01 MinRows | 窗口起点/下限与随机冷启动计数限制 | [F10][FORMULA-F10]中的m；random/tdata累计最多计m行，N为usable rows | shuffle默认m=250000，脚本不覆盖；不足门槛不shuffle | 不适用 | [窗口][KG-WINDOW]、[random计数][KG-RANDOMROWS] | **一致（四参数公式）＋参数／累计量语义差异**。m=150000；N用catalog所有有效重复训练行总数，无来源random单独封顶。random数据超过m时窗口会不同；不把该条件差异冒充当前已有实验事实。首轮冷启动超额另作为后续replay quota锚点。[窗口][EZ-SHUFFLE-CODE]、[累计][EZ-CATALOG]、[quota][EZ-RUNTIME-PLAN] |
| R02 TaperExponent | 控制窗口渐近增长 | W按累计量的p次幂增长，不是采样概率指数 | shuffle脚本示例p=0.65；CLI默认p=1 | 不适用 | [公式][KG-WINDOW]、[脚本][KG-SHUFFLESH] | **一致（四参数公式）**。p=0.65，匹配来源shuffle脚本示例而非CLI默认1；train/eval无第二套指数。N定义/近期排序差异见R01/R04。[公式][EZ-SHUFFLE-CODE]、[配置][EZ-TRAIN] |
| R03 ExpandPerRow | 控制起点新增一行引起的窗口增量 | 取整前的幂律归一化使起点导数=a，见F10 | 脚本示例a=0.4；CLI默认a=1 | 不适用 | [公式][KG-WINDOW]、[脚本][KG-SHUFFLESH] | **一致（四参数公式）**。a=0.4，匹配来源脚本示例；只支持s=m/d=0/no max_rows的四参数子集，不声称所有CLI扩展已实现。[公式][EZ-SHUFFLE-CODE]、[配置][EZ-TRAIN] |
| R04 KeepTargetRows | 限制每次shuffle输出量，不限制窗口 | q=min(1,K/actual_window_rows)；组内保留round(nq)行；all全保留 | 脚本示例K=20000000；CLI必填；组内先随机打乱再截取 | 不适用 | [组内抽样][KG-SHARDIFY]、[比例][KG-SHUFFLE]、[脚本][KG-SHUFFLESH] | **一致（抽样机制）＋参数／排序适配**。K=10000000，q与组内round抽样对应来源；K不是raw磁盘保留上限。近期完整分片按iteration/created/id排序，来源按文件mtime；组边界不同可能改变round结果。Eval/Match不使用回放池。[抽样][EZ-SHUFFLE-SCATTER]、[排序][EZ-CATALOG]、[窗口组装][EZ-SNAPSHOT] |
| R05 Replay ratio／更新节奏 | 限制每新增数据可消费多少训练样本 | train bucket随shuffle报告的新增usable rows增长，更新扣global batch；有上限/等待 | CLI比例未指定时不限制，由启动命令决定；mainb18不定义；异步SP/shuffle/train | 不适用 | [train bucket][KG-TRAIN]、[启动][KG-TRAINSH] | **调度语义差异**。逐轮SP→shuffle→固定1000×128样本更新→SWA发布，ratio=8，steady-state quota每轮新增16000有效行，完整局会超额；冷启动单独锚定。来源异步train bucket是消费限额，非此固定轮事务；不能由ratio数值一致推导模型更新时点一致。[计划][EZ-RUNTIME-PLAN]、[流水线][EZ-RUNTIME-LOOP]、[reader][EZ-READER] |

## 6. 跨机制边界

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train | KataGo Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| A01 价值视角／终局／utility | 两方都选对自己有利的动作，不混淆任务 | 内部white-positive，选点按pla变号；utility含winloss/draw/no-result/score；终局不调用NN | winLoss=1、staticScore=0.05、dynamicScore=0.30；还有komi/动态score中心 | Match默认staticScore=0.1、dynamicScore=0.3；不是纯W−L | [视角][KG-FPU]、[终局模拟][KG-BUDGET]、[默认][KG-SETUP] | **环境适配**。内部value/WDL以节点行棋方为正，边backup变号，统计historical/surprise时统一黑方视角；奖励0/discount1、终局精确±1/0，无重复终局reward。纯W−L且draw独立分类；Go utility/komi/score不移植，故不宣称完整Go搜索数值相等。[适配器][EZ-STATE]、[终局][EZ-GAME]、[统计][EZ-WEIGHTS] |
| A02 搜索预算／等算力口径 | 分清已有访问、新增模拟、NN请求 | maxVisits含已有+新增；maxPlayouts限制新增；根初始评估计访问；根多对称多次NN但只一次访问 | full cap=2000；并行调度不是逐位固定 | 示例500v；时间限制/共享GPU改变实际算力条件 | [搜索循环][KG-BUDGET]、[多对称][KG-SYM]、[比赛][KG-MATCH] | **一致（访问计数子集）＋参数／算力条件差异**。SP400/70、Eval/Match100v；fresh根1次访问，余下cap−1模拟，reuse达到cap可新增0次。无maxTime/maxPlayouts独立限制；根K4是4次NN但1访问。SP32局线程/单搜索线程、batch32、服务1，区别于来源800/192/8/FP16配置。[预算][EZ-RUN]、[参数][EZ-SP] |
| A03 NN cache／模型／朝向 | 避免重复推理，不混淆不同输入条件 | cache hash含棋局和NNInputParams；单对称朝向不入hash；命中复用原输出；根多对称跳cache | 每模型独立evaluator，nnRandomize=true；主线switchNetsMidGame=true | 同机制；无根噪声也不代表NN确定 | [hash][KG-INPUT]、[cache][KG-NNCACHE]、[集成][KG-SYM] | **语义差异（朝向cache及模型切换）**。cache存raw输出，key为已变换obs+globals，等价按朝向分开缓存；来源单朝向不入hash，根K>1跳cache，Eta集成可命中cache。per-model evaluator隔离存在；SP整轮固定模型，来源switchNetsMidGame=true。温度在raw cache外见S18。[cache][EZ-CACHE]、[D4][EZ-STATE]、[模型复用][EZ-WORKER] |
| A04 Checkpoint／续训／发布 | 区分训练状态和推理权重 | 保存model/optimizer/train_state/metrics/SWA；export可跳optimizer；Lookahead cache/RNG须单独核对 | learner/SP/shuffle独立进程，不定义EtaZero整轮事务恢复 | 加载导出权重；gatekeeper独立于搜索 | [save/restore][KG-SAVE]、[Lookahead][KG-LOOKAHEAD]、[export][KG-EXPORT] | **恢复/发布协议适配，未运行验证**。保存model/optimizer/AMP scaler/RNG/reader/Lookahead/SWA/counters；恢复以已提交整轮为准，未完成轮归档后重做，不承诺恢复中间checkpoint继续同轮。发布SWA（未采样时raw）、验证导出；非KataGo独立异步learner/SP协议；无gatekeeper。[checkpoint][EZ-CHECKPOINT]、[恢复][EZ-RESTORE]、[导出][EZ-EXPORT-CODE] |

## 来源公式与复杂分支

### F1 FPU

令m为已分配子节点的搜索先验质量，a=min(1,m^power)，将utility转换为行棋方视角：

```text
Qbase = a·Qparent + (1−a)·Vnn
FPU0 = Qbase − reduction·sqrt(m)
FPU = FPU0 + loss_prop·(−utility_radius−FPU0)
utility_radius = winLossUtilityFactor + staticScoreUtilityFactor + dynamicScoreUtilityFactor
```

来源关闭visited-policy插值时还有固定fpuParentWeight分支。五子棋纯W−L下radius=1，不能将1写成所有KataGo配置的常量。

### F2 Policy target pruning

```text
stable = argmax_a [W_a·max(0,N_a−1)/max(1,N_a) + 2P_a]
best_score = Q_stable + explore_scaling·P_stable/(1+W_stable)
other_weight = ceil(min(W_a,max(0,explore_scaling·P_a/(best_score−Q_a)−1)))
```

分母非正时不减少，稳定边保留。根选择反解不以forced系数>0为前提；Go还含ending bonus/pass抑制。随后执行LCB及chosen prune/subtract。训练policy不要求等于原始visits归一化。

### F3 LCB

```text
ESS0 = weight_sum²/weight_sq_sum
prior_weight = weight_sum/ESS0³
second_moment = max(E[Q²],E[Q]²+1e−8)
second_moment += utility_radius²·prior_weight/(weight_sum+prior_weight)
ESS = (weight_sum+prior_weight)²/(weight_sq_sum+prior_weight²)
radius = lcb_stdevs·sqrt((second_moment−E[Q]²)/ESS)
LCB = self_utility_with_ending_bonus − radius
```

候选须剪枝后权重>0且达到minVisitPropForLCB乘稳定边原始权重。最高LCB边权重提升到至少 `W_i·((radius_i+excess)/(radius_i+0.2·excess))²`，excess≥0。这是权重修正，不是直接argmax LCB落子。修正版允许最佳边index=0。

### F4 Policy/value surprise与采样次数

基础频率w_t∈[0,1]，S=Σw_t，policy surprise=k_t，value surprise=h_t，均值按基础频率加权。非reanalysis路径仅在S≥1且至少一个surprise系数非零时重分配；否则保留基础频率，仍执行随机取整：

```text
w_final_t = w_t
if S≥1 and (alpha>0 or beta>0):
    policy_prop_t = w_t·k_t + (1−w_t)·max(0,k_t−1.5·weighted_mean(k))
    value_prop_t = w_t·h_t
    beta_effective = beta·min(1,weighted_mean(h)/0.010)
    w_final_t = (1−alpha−beta_effective)·w_t
              + alpha·policy_prop_t·S/max(Σpolicy_prop,1e−10)
              + beta_effective·value_prop_t·S/max(Σvalue_prop,1e−10)
row_count_t = floor(w_final_t)+Bernoulli(frac(w_final_t))
```

value surprise统一白方视角，从终局概率倒序执行 `future←future+now·(search_WDL_t−future)`，再取clip(KL(future∥raw_NN_t),0,1)。来源另有直接搜索value surprise开关；reanalysis可禁止未重分析cheap手的policy excess份额。全零surprise不会伪造为均匀分配。

### F5 Variance-scaled cpuct

```text
sigma_hat² = max(0,((Q²+sigma0²)·prior_weight + max(Q²,E[Q²])·W)
                     /(prior_weight+W−1) − Q²)
f_sigma = 1 + scale·(sigma_hat/sigma0−1)
```

节点权重≤1时取sigma0。估计的是utility样本波动，不是value分类概率的方差。

### F6 Value weight exponent

```text
Qsimple = Σ(W_a·Q_a)/ΣW_a
stdev_a = sqrt(1e−8+1/(1.5·sqrt(W_a)))
factor_a = (F_t(3)((Q_a−Qsimple)/stdev_a)+0.0001)^eta
raw_weight_a = chosen_prune_subtract(W_a)·factor_a
desired_weight_a = raw_weight_a·desired_total/Σraw_weight
```

来源t(3)CDF为插值表。初始NN样本、子树均值、二阶矩和weight_sq一起聚合；weight_sq按缩放平方更新。该机制改变backup，区别于训练频率。η=0仍执行适用的chosen prune/subtract；anti-mirror另有禁用分支。

### F7 根／落子温度

```text
T(t) = Tlate+(Tearly−Tlate)·0.5^(t/halflife·19/sqrt(board_area))
```

来源t可包含initialTurnNumber等历史补偿；根policy与chosen温度共用半衰参数。onlyBelowProb=1表示无高概率保护；小于1时只变换低概率尾部。T≤1e−4且无保护时取首个最大权重；保护开启时不能统一概括为argmax。NN温度与这两种温度相互独立。

half-life字段不是所有棋盘上相同的实际手数：实际半衰手数为`halflife·sqrt(area)/19`。例如EtaZero当前15×15、参数12时约9.47手；来源19×19、参数19时是19手。

### F8 Shaped Dirichlet noise

```text
l_a = log(min(P_a,0.01)+1e−20)
h_a = max(0,l_a−mean_legal(l))
alpha_a = total_concentration·(0.5/num_legal+0.5·h_a/Σh)
noise ~ Dirichlet(alpha)
Psearch = (1−epsilon)·Ptempered+epsilon·noise
```

Σh=0时alpha均匀。实际顺序：NN温度→逐对称概率平均→根温度→噪声。Go合法域包含pass。total concentration是总浓度，不是每动作alpha。

### F9 优化器尺度

对SGD/AdamW及fson，来源核心为：

```text
batch_scale = global_batch/256            # SGD，用于WD等
batch_scale = sqrt(global_batch/256)      # AdamW，用于WD等
LR_SGD = 3e−5·LRscale·warmup·group_factor/Lookahead_alpha
LR_AdamW = 1.33·3e−5·LRscale·warmup·group_factor·sqrt(global_batch/256)/Lookahead_alpha
adaptive = 2^(2·tanh(3·log(norm/norm_baseline+1e−30)))
WD_regular = base_wd·batch_scale·(LRscale·warmup)^0.75·adaptive·group_factor
```

SGD base_wd=0.00125、AdamW=0.009；input/normal_gamma/output/noreg另有系数。AdamW LR确实另乘sqrt(global_batch/256)。当前来源默认KATAGO_MODEL_NORMS_ONLY_AT_PRINT=true，每100批以norm快照覆盖running_metrics相应项；关闭时改用EMA。norm baseline在初始化时建立，恢复/重设分支见train_state。LR/WD在累计样本≤2亿时每5 batches刷新，之后每50 batches刷新；norm的采样/平滑节奏也影响WD。MuOn/NorMuon/Aurora、自动LR、RepVGG等额外分支不代表EtaZero已实现。

### F10 Replay窗口与shuffle

N为usable累计行，m为min_rows，s为taper_window_scale默认m，d为add_to_data_rows默认0：

```text
x = N−m+s+d
W = max(m,int(m+a·(x^p−s^p)/(p·s^(p−1))))
W = min(W,max_rows)                       # 仅指定max_rows时
keep_prob = min(1,K/actual_window_rows)
```

四参数子集s=m、d=0且无max_rows时，化为 `max(m,int(m+a·(N^p−m^p)/(p·m^(p−1))))`。选完整近期分片可使actual window超过W；组内round使输出略偏离K。来源按文件时间选择近期数据，random累计单独封顶；不能只比幂律公式而忽略N定义。

## 核查证据

- **静态来源**：沿KataGo入口→setup/play settings→search/NN→writer→shuffle→learner追分支；沿EtaZero配置加载→实际SP/Match调用→记录/重复行→loss/优化→模型发布与恢复对照。已实现项的“静态一致”限定到本表所写子集与二值mask等输入约束。
- **可复跑入口**：[check_reference_formulas.py](/home/sky/RL/EtaZero-lab/EtaZero_V0/scripts/check_reference_formulas.py:1)仅用标准库，从源码AST提取真实函数及梯度阈值分支；不导入训练模块、PyTorch或CUDA，不更新模型、不写实验产物。结果以JSON输出，含四个被核对文件与核查脚本的SHA256；任一对照不符合预期时非零退出。
- **纯函数对照结果**：replay四参数公式252个案例整数结果完全一致；SGD/AdamW六参数组共31104组案例的LR和WD均一致，浮点容限为相对`1e−12`、绝对`1e−15`。这些是脚本的实际复跑计数，每个参数组计一个案例，同时检查LR与WD；仅证明同输入的局部公式，不消除累计量、norm采样和刷新节奏差异。
- **裁剪差异复核**：12个SGD fson案例均确认Eta/source阈值比例2.20，另12个AdamW案例阈值一致。batch=128、LRscale=1时SGD阈值为3889.08729653 vs1767.76695297；SGD差异单独报告，不计为“一致通过”。
- **文档检查**：57个唯一ID、每行7列、无待填项；全部引用定义和本地文件/行号有效。
- **未验证范围**：本次未启动CUDA、短训练、自对弈/Match整局、并行搜索、AMP overflow或断点续训；没有用静态审查推断训练效果，也没有将既有测试文件视为本次已运行通过。

在仓库根目录执行：

```bash
conda run -n pytorch python EtaZero_V0/scripts/check_reference_formulas.py
```

来源路径默认`~/RL/SkyZero/KataGo`，可用`--katago-root /absolute/path/to/KataGo`指定。该脚本独立于训练入口；更换来源后需重新核对本表的commit和适用分支，不能沿用旧结果。

覆盖范围由脚本中的参数网格定义：

- **Replay**：min_rows为1/16/150000/250000，指数0.5/0.65/1，expand_per_row为0.4/1/2；累计行数取0、m−1、m、m+1、2m、10m、100m。只核对s=m、d=0、无max_rows的四参数子集。
- **LR/WD**：SGD/AdamW、batch=64/128/256/1024、LRscale=0.25/1/4；warmup开/关，0、八个阈值的前一行与阈值本身、400万样本；norm快照缺省或为基线的0.25/4倍；默认分组系数及一组非默认系数，Lookahead alpha=0.5/1。固定LRscale、world_size=1、norm_kind=fixscaleonenorm，关闭Muon及自动/循环LR。
- **梯度阈值**：SGD/AdamW × 上述四个batch × 三个LRscale；只核对自动阈值（Eta override=0、来源无clip multiplier），不涵盖实际梯度、裁剪触发率或AMP overflow。

## 代码索引

[KG-CONFIG]: /home/sky/RL/SkyZero/KataGo/cpp/configs/training/README.md:1
[KG-SP]: /home/sky/RL/SkyZero/KataGo/cpp/configs/training/selfplay8mainb18.cfg:1
[KG-MATCH]: /home/sky/RL/SkyZero/KataGo/cpp/configs/match_example.cfg:1
[KG-SETUP]: /home/sky/RL/SkyZero/KataGo/cpp/program/setup.cpp:549
[KG-PLAYSETTINGS]: /home/sky/RL/SkyZero/KataGo/cpp/program/playsettings.cpp:61
[KG-FPU]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchexplorehelpers.cpp:9
[KG-FORCED]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchexplorehelpers.cpp:166
[KG-CHEAP]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:1097
[KG-RUNBOT]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:1233
[KG-SELECTION]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchresults.cpp:113
[KG-LCB]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchhelpers.cpp:564
[KG-RUNNER]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:2708
[KG-PDA]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:1192
[KG-SIDE]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:1857
[KG-SURPRISE]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchresults.cpp:626
[KG-VSURPRISE]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:1308
[KG-WEIGHTS]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:2091
[KG-AGGREGATE]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchupdatehelpers.cpp:139
[KG-VWEIGHT]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchupdatehelpers.cpp:428
[KG-SYM]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/nneval.cpp:1046
[KG-RANDOMSYM]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/nneval.cpp:934
[KG-NNCACHE]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/nneval.cpp:1103
[KG-NNHELP]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchnnhelpers.cpp:1
[KG-NOISE]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchhelpers.cpp:78
[KG-TEMPERATURE]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchhelpers.cpp:1
[KG-CHOSEN]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchresults.cpp:573
[KG-NODE]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchnode.h:1
[KG-BUDGET]: /home/sky/RL/SkyZero/KataGo/cpp/search/search.cpp:553
[KG-INPUT]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/nninputs.cpp:889
[KG-UNCERTAINTY]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchupdatehelpers.cpp:114
[KG-NOISEPRUNE]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchupdatehelpers.cpp:521
[KG-OPTIMISM]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/eigenbackend.cpp:2543
[KG-GAMEINIT]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:18
[KG-TRAIN]: /home/sky/RL/SkyZero/KataGo/python/train.py:80
[KG-TRAINSH]: /home/sky/RL/SkyZero/KataGo/python/selfplay/train.sh:1
[KG-RESBLOCK]: /home/sky/RL/SkyZero/KataGo/python/katago/train/model_pytorch.py:734
[KG-NBT]: /home/sky/RL/SkyZero/KataGo/python/katago/train/model_pytorch.py:926
[KG-B5]: /home/sky/RL/SkyZero/KataGo/python/katago/train/modelconfigs.py:198
[KG-MODELCONFIG]: /home/sky/RL/SkyZero/KataGo/python/katago/train/modelconfigs.py:1830
[KG-SUFFIX]: /home/sky/RL/SkyZero/KataGo/python/katago/train/modelconfigs.py:2044
[KG-TRANSFORMER]: /home/sky/RL/SkyZero/KataGo/python/katago/train/model_pytorch.py:1942
[KG-ATTENTION]: /home/sky/RL/SkyZero/KataGo/python/katago/train/model_pytorch.py:2100
[KG-NORM]: /home/sky/RL/SkyZero/KataGo/python/katago/train/model_pytorch.py:282
[KG-INITIALIZE]: /home/sky/RL/SkyZero/KataGo/python/katago/train/model_pytorch.py:3655
[KG-MODEL]: /home/sky/RL/SkyZero/KataGo/python/katago/train/model_pytorch.py:509
[KG-EXPORT]: /home/sky/RL/SkyZero/KataGo/python/export_model_pytorch.py:1
[KG-D4]: /home/sky/RL/SkyZero/KataGo/python/katago/train/data_processing_pytorch.py:155
[KG-SWA]: /home/sky/RL/SkyZero/KataGo/python/train.py:924
[KG-LOOKAHEAD]: /home/sky/RL/SkyZero/KataGo/python/train.py:1880
[KG-LOOKAHEAD-COUNTER]: /home/sky/RL/SkyZero/KataGo/python/train.py:1674
[KG-LR]: /home/sky/RL/SkyZero/KataGo/python/train.py:1212
[KG-WD]: /home/sky/RL/SkyZero/KataGo/python/train.py:701
[KG-GRAD]: /home/sky/RL/SkyZero/KataGo/python/train.py:1753
[KG-METRICS]: /home/sky/RL/SkyZero/KataGo/python/katago/train/metrics_pytorch.py:592
[KG-POLICYLOSS]: /home/sky/RL/SkyZero/KataGo/python/katago/train/metrics_pytorch.py:78
[KG-VALUELOSS]: /home/sky/RL/SkyZero/KataGo/python/katago/train/metrics_pytorch.py:121
[KG-WRITE]: /home/sky/RL/SkyZero/KataGo/cpp/dataio/trainingwrite.cpp:472
[KG-ROUND]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:2288
[KG-WINDOW]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:414
[KG-RANDOMROWS]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:1058
[KG-SHARDIFY]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:249
[KG-SHUFFLE]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:1157
[KG-SHUFFLESH]: /home/sky/RL/SkyZero/KataGo/python/selfplay/shuffle.sh:40
[KG-SAVE]: /home/sky/RL/SkyZero/KataGo/python/train.py:645
[KG-POLICYINIT]: /home/sky/RL/SkyZero/KataGo/cpp/program/playutils.cpp:235
[GM-OPENING]: /home/sky/RL/SkyZero/KataGomo/cpp/game/randomopening.cpp:1
[GM-PLAY]: /home/sky/RL/SkyZero/KataGomo/cpp/program/play.cpp:1259
[GM-POLICYINIT]: /home/sky/RL/SkyZero/KataGomo/cpp/program/playutils.cpp:124
[GM-PLAYSETTINGS]: /home/sky/RL/SkyZero/KataGomo/cpp/program/playsettings.cpp:27
[GM-PLAYSETTINGS-SP]: /home/sky/RL/SkyZero/KataGomo/cpp/program/playsettings.cpp:62
[GM-WRITE]: /home/sky/RL/SkyZero/KataGomo/cpp/dataio/trainingwrite.cpp:310
[GM-INPUT]: /home/sky/RL/SkyZero/KataGomo/cpp/neuralnet/nninputs.cpp:589
[GM-CONSTANT]: /home/sky/RL/SkyZero/KataGomo/cpp/core/config.h:10
[EZ-SP]: /home/sky/RL/EtaZero-lab/EtaZero_V0/configs/baseline/selfplay.cfg:1
[EZ-EVAL]: /home/sky/RL/EtaZero-lab/EtaZero_V0/configs/baseline/eval.cfg:1
[EZ-MATCH]: /home/sky/RL/EtaZero-lab/EtaZero_V0/configs/baseline/match.cfg:1
[EZ-TRAIN]: /home/sky/RL/EtaZero-lab/EtaZero_V0/configs/baseline/train.cfg:1
[EZ-ENV]: /home/sky/RL/EtaZero-lab/EtaZero_V0/configs/baseline/env.cfg:1
[EZ-NET]: /home/sky/RL/EtaZero-lab/EtaZero_V0/configs/baseline/net.cfg:1
[EZ-CONFIG-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/config.py:22
[EZ-WIDTHS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/config.py:102
[EZ-SETTINGS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:41
[EZ-SP-CALL]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:136
[EZ-MATCH-CALL]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:324
[EZ-WORKER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:202
[EZ-FPU-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:27
[EZ-SELECTION]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:31
[EZ-NOISE-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:15
[EZ-EVALUATE-NODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:142
[EZ-EXPAND]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:198
[EZ-SIMULATE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:237
[EZ-RUN]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:344
[EZ-RESULT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:396
[EZ-ADVANCE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:409
[EZ-SEARCH-MATH]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search_math.cpp:7
[EZ-AGGREGATE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search_math.cpp:68
[EZ-SEARCH-HEADER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/include/etazero/search.h:1
[EZ-LIMITS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/search_limits.cpp:25
[EZ-WEIGHTS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/record.cpp:15
[EZ-WRITER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/record.cpp:220
[EZ-OPENING-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/opening.cpp:127
[EZ-POLICY-INIT-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/opening.cpp:213
[EZ-GAME]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/include/etazero/game.h:23
[EZ-STATE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/include/etazero/algorithm.h:29
[EZ-CACHE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/inference/batcher.cpp:53
[EZ-NORM-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:14
[EZ-NBT-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:136
[EZ-NETWORK-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:206
[EZ-HEADS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:161
[EZ-LOSS-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:243
[EZ-INFERENCE-NET]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:293
[EZ-D4-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/symmetry.py:4
[EZ-OPT-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:8
[EZ-WARMUP-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:41
[EZ-OPT-CONFIGURE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:117
[EZ-GRAD-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:127
[EZ-OPT-STEP]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:135
[EZ-OPT-INFERENCE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:165
[EZ-TRAIN-LOOP]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/training.py:194
[EZ-CHECKPOINT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/training.py:55
[EZ-VIEW]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/data.py:149
[EZ-CATALOG]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/data.py:225
[EZ-SHUFFLE-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:21
[EZ-SHUFFLE-SCATTER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:85
[EZ-SNAPSHOT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:296
[EZ-RUNTIME-PLAN]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:167
[EZ-RUNTIME-LOOP]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:291
[EZ-RESTORE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:535
[EZ-READER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/reader.py:75
[EZ-EXPORT-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/export.py:33
[KG-NORM-TIMING]: /home/sky/RL/SkyZero/KataGo/python/katago/train/trainloop_helpers.py:285
[FORMULA-F1]: /home/sky/RL/EtaZero-lab/EtaZero.md:129
[FORMULA-F2]: /home/sky/RL/EtaZero-lab/EtaZero.md:142
[FORMULA-F3]: /home/sky/RL/EtaZero-lab/EtaZero.md:152
[FORMULA-F4]: /home/sky/RL/EtaZero-lab/EtaZero.md:166
[FORMULA-F5]: /home/sky/RL/EtaZero-lab/EtaZero.md:184
[FORMULA-F6]: /home/sky/RL/EtaZero-lab/EtaZero.md:194
[FORMULA-F7]: /home/sky/RL/EtaZero-lab/EtaZero.md:206
[FORMULA-F8]: /home/sky/RL/EtaZero-lab/EtaZero.md:216
[FORMULA-F9]: /home/sky/RL/EtaZero-lab/EtaZero.md:228
[FORMULA-F10]: /home/sky/RL/EtaZero-lab/EtaZero.md:243
