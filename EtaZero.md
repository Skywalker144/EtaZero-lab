# EtaZero 算法与训练核查表

本表覆盖57项算法及26项工程条目，逐行记录实现、参数、边界、调用链和实际证据。[第10批核查](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json)对应其保存的源码快照；当前源码变更按受影响条目重新验证，不沿用历史快照的整体验收结论。S23按用户要求暂缓，未实现、未对齐，不计通过；400/70为用户明确保留的预算差异。运行通过不代表训练效果或来源端到端性能一致。

## 基准与读法

本表先从来源源码填写公式、分支与 train/eval 行为，再逐行对照 `EtaZero_V0`。来源配置与 EtaZero 当前配置分别记录；公式相同不代表整个调用链相同。静态核查包含目标工作区的未提交改动。

- **KataGo 源码**：`/home/sky/RL/SkyZero/KataGo`，commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`；读取时工作区干净。
- **统一 selfplay 参数基准**：[selfplay8mainb18.cfg][KG-SP]。[training/README.md][KG-CONFIG] 将它定义为 public distributed 主线切换到 b18c384nbt 后的参考配置；`selfplay1.cfg` 是早期小网络示例，不混入其数值。这是仓库的主线参考配置，不能宣称是线上服务器此刻的实时配置。
- **加载后默认值**：`selfplay` 使用 `SETUP_FOR_OTHER`，`match` 使用 `SETUP_FOR_MATCH`，追到 [setup.cpp][KG-SETUP]，不采用 `SearchParams` 构造函数初始值。在线 `contribute` 使用 `SETUP_FOR_DISTRIBUTED` 与服务器任务配置，是另一条入口。
- **统一 Eval/Match 参数基准**：[match_example.cfg][KG-MATCH] 加 `SETUP_FOR_MATCH` 的默认值。它是比赛示例 profile；analysis/GTP 的额外默认行为不混入。来源比赛温度非零，不能写成“所有eval默认零温度”。
- **Learner / shuffle**：该 commit 的 `train.py`、`model_pytorch.py`、`metrics_pytorch.py`、`shuffle.py` 为机制来源；解析器和派生默认值为参数来源。selfplay配置不定义网络架构、LR或replay窗口；没有线上训练命令，不能推断线上learner参数。shuffle脚本示例值单独注明。
- **KataGomo 专有机制**：`/home/sky/RL/SkyZero/KataGomo`，commit `df152116e3787c75c6a3de099d261ca092b7dfc1`；读取时工作区干净。平衡开局、禁手输入dropout、五子棋policy init使用它的源码；KataGo的Go对应行为另外说明。
- **目标配置**：V0 [baseline selfplay][EZ-SP]、[eval][EZ-EVAL]、[match][EZ-MATCH]、[train][EZ-TRAIN]、[env][EZ-ENV]、[net][EZ-NET]。当前参数以这些配置文件为准；本表的EtaZero列和下方profile表记录当前baseline，来源列与历史证据保留当时数值，不代表其他profile或已有实验的配置。

第10批目标核查快照：EtaZero Git HEAD `185eee220fd786a6e7436ae4e2332b929904bdb0` 加该批记录的工作区；当时引用的28个目标实现/配置文件（不含独立核查脚本）集合SHA256为 `10d6d587b0b3787752c07e7a778bce1959dbb6ac95db585370944f51a2b3d67e`（按仓库相对路径排序，对每行`path + NUL + 文件SHA256十六进制 + LF`求SHA256）。源码改变后应重核相关行，不能仅凭HEAD沿用结论。

Train分为 **SP（自对弈产数据）**、**L（learner更新）**；Eval分为固定局面搜索与Match。公式转换到行棋方视角；KataGo白方正视角、score/no-result与五子棋WDL的差异见A01。

结论标签：**一致（静态）**、**参数差异**、**语义差异**、**环境适配**、**部分实现**、**未实现**、**暂缓**、**不适用**。静态一致表示已核对源码及调用点，不表示完整运行验证或训练效果通过；不同性质的结论可并列。

## 审查结论与使用边界

57行均已对照来源和目标代码填写；独立核查脚本与复跑范围见“核查证据”。优先处理的区别如下，编号可定位详细行及源码：

| 类别 | 已确认的区别 | 对结果的含义 |
|---|---|---|
| Learner基础数值 | N11/L01/L02：fson SGD裁剪2500、ordinary系数0.930、value有效系数0.72；训练AMP heads与skip已验证 | 基础数值及v15辅助监督已验证；三架构与v17可选纯W−L Q已实现，Q整链验收通过 |
| 训练目标 | L01/L02：基础五项系数已落实，v15 optimistic/TD/error已验收，v17可选纯W−L Q已接入 | v15基础与辅助损失已验收；v17纯W−L Q已接入，完整验收通过；不宣称训练效果已验证 |
| 抽样与推理语义 | D02：最终输出row独立dropout并持久化；S17/A03：随机单朝向、canonical cache与集成绕过已实现 | 重复行独立及搜索不受dropout影响已验证；搜索随机朝向与cache已有独立样例验证 |
| 参数profile差异 | S02/S07/S09/S19/S20/R01–R03：当前full/cheap400/70、reduced min50、side0.040、SP half-life15；replay m150000/p0.8/a0.3 | 400/70 是用户明确保留的预算差异；其余当前配置差异也明确登记，公式一致不等于参数profile一致 |
| 已实现与暂缓 | N03：bare Transformer指定预设已实现；PDA、side、reanalysis及hint/game fork已接入；S26—S28三项搜索修正已验收；N01独立plain网络已接入，来源CPU/CUDA前向与梯度通过 | 有的来源SP启用，有的来源Match默认启用，应按具体阶段讨论；S23 subtree bias由用户要求暂缓，未实现、未对齐 |
| 数据与发布节奏 | R01/R04/R05/A03/A04：随机数据封顶、近期排序、固定整轮训练/发布/恢复和轮内模型切换不同 | 即使局部公式相同，也不构成完整异步KataGo训练链路等价 |

**Eval应关闭的训练行为**：PCR、Reduce Visits、训练surprise频率重分配、训练输入dropout、learner D4、optimizer/warmup/Lookahead在线更新；根噪声和forced按本表选定Match profile关闭。**仍可保留的搜索行为**：FPU、policy selection pruning、LCB、value weighting、tree reuse、随机朝向NN；来源Match还默认启用variance scaling、uncertainty、optimistic policy和noise pruning，其中uncertainty/optimistic依赖网络支持。SWA是模型发布方式，使用SWA模型评估不等于在eval中更新SWA。

本表覆盖原清单和直接相邻的语义边界，**不是整个KataGo仓库的穷尽证明**。主线配置还包括early/game forks、Go komi/handicap/seki/lead及结束阶段等，五子棋不直接照搬；来源还支持hint/SGF、reanalysis、metadata、Muon与更多网络/调度分支。未列为已实现项的能力不能由名称相近推定存在。首轮源码审查之后，learner基础数值/AMP及恢复已有真实CUDA证据，见“核查证据”和plan批次1；并行搜索和所选同步整轮恢复已有真实验收；全量复审的范围与证据见第10批逐项记录。

## 1. 搜索、落子与自对弈采样

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train（SP） | KataGo Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| S01 FPU | 为未访问动作赋初始价值 | [F1][FORMULA-F1]：visited-policy插值、sqrt扣减、loss比例插值 | 非根reduction=0.2、普通根=0；visited-policy开启、power=2；cheap零基础权重根用非根参数 | 非根0.2、根0.1；visited-policy开启、power=2；loss proportion默认0 | [FPU][KG-FPU]、[cheap][KG-CHEAP]、[加载器][KG-SETUP] | **一致（独立样例）＋环境适配**。SP非根/普通根0.2/0，Eval/Match 0.2/0.1；visited-policy/power2与固定NN权重分支均实现，权重默认0、power允许0。cheap根改用非根参数；纯W−L radius=1。[公式][EZ-FPU-CODE]、[选点][EZ-SIMULATE]、[加载][EZ-SETTINGS] |
| S02 Playout cap randomization（PCR） | 混合便宜推进手与完整监督手 | 每手Bernoulli(q)选cheap；基础频率还会经surprise重分配 | q=0.75，full/cheap=2000/350v，cheap基础权重0；hint/fork可改变概率或强制full | Match不走SP的PCR预算分支 | [预算][KG-CHEAP]、[主线配置][KG-SP] | **默认与hint/fork优先级来源/真实CUDA通过＋用户预算例外**。SP full/cheap=400/70v，q=0.75、基础权重0；cheap去根噪声/根温度/forced/根集成且不清树，保留NN温度；可被surprise恢复采样。Eval/Match无PCR。精确hint×4强制full/清图，后6手及hintFork cheap概率减半；reanalysis force-full跳过hint/PCR，保留reduce/PDA。3072来源预算组合通过。[预算][EZ-LIMITS]、[调用][EZ-SP-CALL]、[根处理][EZ-RUN] |
| S03 Forced playout | 优先探索根部尚未达到配额的动作 | W_a<sqrt(k·P_a·T)时强制优先；W含virtual weight | k=2；cheap零基础权重关闭 | 默认k=0，关闭 | [选点][KG-FORCED]、[cheap][KG-CHEAP] | **一致（静态）**。SP k=2，使用search prior和含virtual weight的子边权重；cheap关闭；Eval/Match配置加载不赋forced，保持0。[分数][EZ-SEARCH-MATH]、[加载][EZ-SETTINGS] |
| S04 Policy target pruning | 削减事后看来不必要的探索权重 | [F2][FORMULA-F2]：找稳定边，再用PUCT反解其他边所需权重 | 根落子分布与policy target都执行；没有独立同名配置开关 | 根同样执行，即使forced系数=0；不是eval自动关闭项 | [选择权重][KG-SELECTION]、[PUCT反解][KG-FPU] | **一致（静态）**。稳定边、PUCT反解与ceil均对应来源；SP落子/target及Eval/Match均执行，当前pruning=true。增加可关闭开关属于消融扩展；Go ending/pass处理按环境移除。[实现][EZ-SELECTION] |
| S05 LCB | 偏重价值下界更可靠的动作 | [F3][FORMULA-F3]：加权ESS、方差先验、LCB与选择权重提升 | 开启，z=5、门槛0.15；实际SP落子暂时禁用，生成target时恢复；useNonBuggyLcb=true | 默认开启、z=5、门槛0.15；参与落子，修正索引默认开启 | [LCB][KG-LCB]、[选择][KG-SELECTION]、[SP调用][KG-RUNBOT] | **一致（静态）＋环境适配**。SP实际落子禁LCB，target用LCB；Eval/Match落子用LCB；z=5、门槛0.15，候选index=0有效。加权ESS和radius²修正一致；省去Go ending bonus。[实现][EZ-SELECTION]、[调用分流][EZ-RESULT] |
| S06 Tree reuse／清树 | 推进命中子树，按调用方决定是否清空 | maxVisits含已有访问，maxPlayouts限制新增；原始先验与根探索先验分开 | 普通SP清树；cheap基础权重0时不清树；PDA强制清树；模型变化更新搜索状态 | 不同bot的Match默认不清树；相同botIdx双方强制清树；可配置clearBotBeforeSearch | [SP分支][KG-CHEAP]、[GameRunner][KG-RUNNER]、[预算][KG-BUDGET] | **一致（已实现调用分流）＋用户适配**。SP full清树/cheap零基础权重保留；不同模型身份Match两棵树默认reuse，相同身份共享树且每手清树；eval默认不reuse。根从raw先验重新应用探索，set_evaluator清整图，根推进复制共享节点；SP仍按用户范围轮间换模型。[推进][EZ-ADVANCE]、[预算][EZ-LIMITS]、[比赛][EZ-MATCH-CALL] |
| S07 Reduce visits | 一方持续占优时低预算、低权重完成对局 | e=max(min(h),−max(h))；r=((min(e,1)−t)/(1−t))²；cap=round(V+r(Vmin−V))；w=1+r(wmin−1) | t=0.9、lookback=3、Vmin=350、wmin=0.1；与PCR为else-if，同一白方视角历史 | 不使用；Match硬认输是独立设置 | [缩减分支][KG-CHEAP]、[配置][KG-SP] | **公式对照＋当前参数差异**。SP t=0.9、lookback3、Vmin50、wmin0.1；来源Vmin350；PCR与缩减互斥，历史统一黑方视角，缩减full保留探索。Vmin独立于cheap70；smoke/minimal因full cap显式覆盖最低值。Eval/Match不走此分支。[公式][EZ-LIMITS]、[历史][EZ-SP-CALL] |
| S08 Playout doubling advantage（PDA） | 非对称预算和条件输入共同表示算力优势 | d对应预算比2^d；双方cap系数为2·2^d/(1+2^d)、2/(1+2^d)；输入按执色变号 | 普通局概率0.01、让子局0.5、最大预算比8；可补偿komi；清树；side rows无PDA | 搜索支持该条件，普通Match默认d=0 | [PDA][KG-PDA]、[输入][KG-INPUT]、[side][KG-SIDE] | **来源预算/输入对照＋真实CUDA通过**。普通SP概率0.01、均匀优势方/d、最大比8；PCR/reduced后乘双方系数并round，低于5报错，PDA强制清图。全局6维、enabled及signed half-d贯通所有深度/D4/cache/原分片/训练导出；side清零。Eval/Match默认0、显式条件可评估；无Go handicap/komi，int32-max playout视为无额外上限。smoke小预算明确关闭。[预算][EZ-LIMITS]、[输入][EZ-GAME] |
| S09 Side position | 搜索实际落子之外的替代局面，增加反驳监督 | 排除实际着后抽替代着；独立搜索提取policy和搜索value；分支不必下到终局 | 概率0.020；主线旧名forkSidePositionProb由loader映射；初始权重1；可再分支 | 不生成训练side rows | [采样／搜索][KG-SIDE]、[旧名映射][KG-PLAYSETTINGS] | **真实SP/递归/训练消费通过＋当前参数差异**。当前默认概率0.040，来源0.020，排除实际着后70% policy-T1/25% T2/5%均匀候选，终局拒绝；局后独立full搜索，LCB选回复，0.25递归回复+普通NN-T1替代着，side频率1、PDA0。value/TD只用分支搜索WDL，无完整终局gate。原生产样到CUDA更新/导出通过。[生产][EZ-SP-CALL]、[writer][EZ-WRITER]、[targets][EZ-VIEW] |
| S10 Policy surprise weighting | 多训练网络先验没有预料到的结果 | KL(target∥含噪声搜索先验)；[F4][FORMULA-F4]局内频率重分配 | 比例0.5；cheap/reduced超过1.5倍均值可恢复权重；reanalysis改变cheap例外 | 不重分配训练数据、不改当前落子 | [KL][KG-SURPRISE]、[重分配][KG-WEIGHTS] | **默认及reanalysis gate已验收**。SP alpha0.5，KL对含噪声search prior，cheap/reduced excess可恢复；启用reanalysis时未重分析cheap禁用excess，已重分析reduced仍保留。选择先用旧surprise，替换目标后重算并只取整一次；Eval/Match不产行。hint/fork前缀独立排除、hint首值在重分析前后翻转复制next/terminal。[重分配][EZ-WEIGHTS]、[生产][EZ-SP-CALL] |
| S11 Value surprise weighting | 多训练价值预测没有预料到的后续结果 | 终局开始倒序平滑搜索概率，再对原始NN做KL，截[0,1]；[F4][FORMULA-F4] | 比例0.1；now=1/(1+0.016·area)；均值<0.010时同比减弱；useSearchValueSurprise默认false | 不用于eval落子或数据重分配 | [value KL][KG-VSURPRISE]、[重分配][KG-WEIGHTS] | **两分支来源对照/真实CUDA通过＋WDL适配**。SP beta0.1，默认未来平滑/截断/低均值减弱；direct开关以本手search对NN的KL，不使用真实终局。统一黑方视角、重分析后重算；原选择值独立保存，draw为真实和棋。Eval/Match不执行。[实现][EZ-WEIGHTS] |
| S12 基础weighted PUCT | 平衡价值与先验探索 | Q_a+c(T)f_sigma·P_a·sqrt(T+0.01)/(1+W_a)；W是统计权重 | c0=1.05；Q为Go综合utility | c0加载默认1.0；同样weighted PUCT | [探索][KG-FPU]、[配置][KG-SP]、[默认][KG-SETUP] | **一致（纯W−L公式及profile）＋环境适配**。SP c0=1.05、Eval/Match=1.0；weighted分母、0.01偏移、virtual分子排除均保留。score utility不适用。[公式][EZ-SEARCH-MATH]、[选点][EZ-SIMULATE] |
| S13 Log-scaled cpuct | 搜索越大，探索系数越高 | c(T)=c0+c_log·log((T+b)/b) | c_log=0.28，b=500 | c_log默认0.45，b=500 | [探索系数][KG-FPU]、[加载器][KG-SETUP] | **一致（公式及profile）**。SP c_log0.28、Eval/Match0.45，base500；显式0可关闭。[公式][EZ-SEARCH-MATH]、[SP][EZ-SP]、[Eval][EZ-EVAL]、[Match][EZ-MATCH] |
| S14 Variance-scaled cpuct | 按utility波动尺度调节探索 | [F5][FORMULA-F5]：f_sigma=1+λ(σhat/σ0−1) | SETUP_FOR_OTHER下λ=0关闭；σ0=0.40、先验权重2；主线cfg未覆盖 | SETUP_FOR_MATCH下λ=0.85开启；σ0=0.40、先验权重2 | [方差估计][KG-FPU]、[入口默认][KG-SETUP] | **一致（独立样例及profile）＋环境适配**。prior0.40、weight2；SP scale0，Eval/Match0.85；验证weight≤1与一般统计、负方差夹取；纯W−L不移植score。[公式][EZ-SEARCH-MATH]、[加载][EZ-SETTINGS] |
| S15 Value weight exponent | backup时降低较差子树影响 | [F6][FORMULA-F6]：t(3)CDF重加权、保留总权重；η=0关闭；anti-mirror另有例外 | η=0.5；带噪声根且无noise pruning时先chosen prune/subtract | η默认0.25，仍保留机制 | [聚合][KG-AGGREGATE]、[重加权][KG-VWEIGHT] | **一致（纯W−L子集及profile）**。SP eta0.5、Eval/Match0.25；t(3)同插值表，保留总权重及平方和；Go anti-mirror不适用。[聚合][EZ-AGGREGATE] |
| S16 Root symmetry ensemble | 多朝向概率平均，降低方向偏差 | 无放回均匀抽K个D4；还原policy后平均policy/value概率；不平均logits | 普通根K=4、cheap零基础权重K=1；根多对称跳过NN cache | 根K=1，但这一次NN请求仍可随机朝向 | [多对称][KG-SYM]、[root调用][KG-NNHELP] | **一致（独立样例）**。SP正常根K4、cheap/Eval/Match K1；无放回D4，温度后policy及WDL概率平均、坐标还原。K>1读写绕cache，fresh只计1 playout、reused重集成不增加访问。[集成][EZ-EVALUATE-NODE]、[坐标][EZ-STATE]、[cache][EZ-CACHE] |
| S17 叶节点随机D4 | 随机朝向推理并还原输出 | 未指定symmetry且nnRandomize=true时均匀抽0…7；cache命中不重新抽 | nnRandomize=true | match_example同样true；不是eval必须关闭项 | [随机朝向][KG-RANDOMSYM]、[cache][KG-NNCACHE] | **一致（独立样例）＋RNG适配**。baseline全部profile nnRandomize=true；叶/cheap/Match单次朝向由NN服务cache miss后均匀抽，命中不推进朝向RNG；关闭时指定nn_symmetry。服务和搜索RNG独立，非逐位来源复现。[路径][EZ-EVALUATE-NODE]、[服务][EZ-CACHE] |
| S18 NN policy temperature | 改变全树网络先验尖锐程度 | 合法域softmax(logits/Tnn)；Tnn进入cache hash | 默认Tnn=1，根/非根均生效 | 默认Tnn=1 | [输入参数/hash][KG-INPUT]、[NN后处理][KG-NNCACHE] | **一致（profile及cache输入条件）**。所有profile Tnn1，根/叶/cheap均生效；即使保存raw canonical logits，Tnn仍入key，避免温度变化复用旧随机朝向输出。[softmax][EZ-EVALUATE-NODE]、[cache][EZ-CACHE] |
| S19 Root policy temperature | 根部进一步展平先验 | normalize(P^(1/Troot(t)))，先于噪声；[F7][FORMULA-F7]衰减 | 1.5→1.1、half-life参数19；cheap零基础权重改为1 | 早/晚均默认1 | [根温度][KG-NOISE]、[衰减][KG-TEMPERATURE] | **公式一致＋当前参数差异**。SP1.5→1.1、halflife15（来源19）；cheap1；Eval/Match1→1、halflife19。面积缩放，在D4平均后、噪声前；复用根从raw先验重算。[衰减][EZ-SEARCH-MATH]、[根][EZ-RUN] |
| S20 Chosen move temperature | 将搜索权重变成实际动作分布 | normalize(Wselect^(1/Tmove(t)))；小温度/概率保护边界见F7 | 0.75→0.15、half-life参数19；不改变policy target | match_example为0.60→0.20；无配置覆盖的loader默认0.5→0.1 | [落子][KG-CHOSEN]、[温度][KG-TEMPERATURE]、[Match][KG-MATCH] | **公式一致＋当前参数差异**。SP0.75→0.15、halflife15（来源19）；Eval/Match0.60→0.20、halflife19；onlyBelowProb1。动作温度不乘target；零温度显式对照保留。[温度][EZ-SEARCH-MATH]、[落子][EZ-RESULT] |
| S21 Chosen move prune/subtract | 选择前剔除极低权重、统一扣减 | cutoff=min(prune,maxW/64)，subtract=min(subtract,maxW/64)；低于cutoff置零 | prune=1、subtract=0；LCB后执行，也参与特定根价值聚合 | 默认同样1/0 | [选择末尾][KG-SELECTION]、[聚合][KG-AGGREGATE] | **一致（静态）**。SP/Eval/Match prune=1、subtract=0；选择时LCB之后；噪声根backup前也执行，cheap无噪声不触发该backup分支。当前backup无noise-pruning竞争分支。[选择][EZ-SELECTION]、[backup][EZ-AGGREGATE] |
| S22 Shaped Dirichlet noise | 将部分噪声集中在相对有希望的动作 | [F8][FORMULA-F8]：alpha一半均匀一半按截断log-policy分配，再混根先验 | total concentration=10.83、epsilon=0.25；来源默认shaped，无独立shaped开关；cheap零基础权重关闭 | rootNoise默认false | [alpha与混合][KG-NOISE]、[配置][KG-SP] | **一致（公式及profile）**。SP total10.83、epsilon0.25、shaped=true；cheap/Eval/Match关闭。半均匀/半中心化log-policy对应来源；uniform开关为显式消融。[alpha][EZ-NOISE-CODE]、[混合][EZ-RUN]、[配置][EZ-SP] |
| S23 Subtree value bias | 在线学习相似局部战术的NN价值偏差 | Vcorrected=Vnn+β·Δsum/Wsum；子树权重指数写模式表，释放时衰减 | β=0.30、指数0.8；额外表/锁/释放行为 | 默认β=0.45、指数0.85、free proportion=0.8；并非默认关闭 | [修正/更新][KG-AGGREGATE]、[默认][KG-SETUP] | **暂缓，未实现、未对齐**。用户要求暂缓；当前无模式表、偏差累积和释放衰减。来源SP和Match均非零，不计为通过；暂缓不豁免其他机制。[当前聚合][EZ-AGGREGATE]、[范围][EZ-CONFIG-CODE] |
| S24 Virtual loss | 临时降低并发在途分支吸引力 | v=pending·loss；Q←Q+(Qloss−Q)v/(v+max(0.25,W))；W←W+v；分子只计完成 | loss=1、threads=1；最终监督不计pending | loader默认loss=1；同一机制 | [虚拟权重][KG-FPU]、[默认][KG-SETUP] | **一致（静态）＋环境适配**。SP/Eval/Match virtual_loss=1；pending进入均值与分母，朝纯W−L下界−1插值，探索分子只用已完成权重；图共享时pending取子节点全部在途预约，正常/异常退出释放pending。baseline search_threads=1，常态无并发虚拟权重。[分数][EZ-SEARCH-MATH]、[并行][EZ-SIMULATE] |
| S25 Graph search／转置 | 相同状态共享节点统计 | 状态/历史hash、重复边界、循环/追赶；child权重按edgeVisits/childVisits换算 | useGraphSearch=true | SETUP_FOR_MATCH默认true | [图搜索][KG-BUDGET]、[child统计][KG-NODE]、[默认][KG-SETUP] | **来源小图对照通过＋NOVC适配**。SP/Eval/Match graph=true、catch-up leak=0；表共享节点，父边独立访问，下降前CAS追赶，循环终止counted playout，共享子节点virtual loss。LCB的weightSq按edge/child比例、父聚合按比例平方；根复制及静止后mark/sweep避免多父重复删除。关闭图及leak0/中间/1均覆盖。NOVC无Go重复历史，完整局面字节含棋规/画布/执子/终局；调度不同，不称并发序列/性能等价。[结构][EZ-SEARCH-HEADER]、[扩展][EZ-EXPAND]、[推进][EZ-ADVANCE] |
| S26 Uncertainty-weighted playout | 可信NN评估在聚合中占更高权重 | w=c/(u^p+c/wmax)，u含短期winloss/score error；缺少error heads则w=1 | SETUP_FOR_OTHER默认关闭 | SETUP_FOR_MATCH默认开启；c=0.25、p=1、wmax=8 | [样本权重][KG-UNCERTAINTY]、[默认][KG-SETUP] | **来源公式/真实CUDA通过＋纯W−L适配**。SP关闭，Eval/Match开启，c0.25/p1/max8；真实标准差→NN样本权重与平方，D4先平均标准差。支持能力时终局每访问max8，二阶N×64；无辅助能力才weight1。非单位图统计、非默认SP及独立504例通过。[初始样本][EZ-EVALUATE-NODE]、[heads][EZ-HEADS]、[聚合][EZ-AGGREGATE] |
| S27 Optimistic policy | 用偏好好结果的辅助策略改变先验 | backend混合logits：l=(1−o)lmain+olopt；训练监督随版本变化 | root/nonroot optimism默认0，但现代版本训练辅助头 | Match默认root=0.2、nonroot=1；旧模型不支持时消去该输入 | [搜索输入][KG-NNHELP]、[backend][KG-OPTIMISM]、[loss][KG-METRICS] | **来源float logits对照/真实CUDA通过**。ordinary与short optimistic按float main+(opt−main)×o混合，随后温度、合法softmax、D4概率平均。SP0/0，Eval/Match root0.2/leaf1；晋升根条件刷新不增访问；无能力消去条件。cache按exact optimism分键，比来源1/1024离散更细，不称命中率/RNG等价。v17可选纯W−L Q在8c验收。[NN][EZ-EVALUATE-NODE]、[导出][EZ-EXPORT-CODE] |
| S28 Noise pruning | 在价值聚合中削弱被过度探索的差分支 | 相对前缀utility均值差且W>2·Wprefix·P/Pprefix时，扣除excess·(1−exp(−gap/scale))，受cap限制 | SETUP_FOR_OTHER默认关闭；与S04的选择pruning独立 | SETUP_FOR_MATCH默认开启；scale=0.15、cap=1e50 | [聚合剪枝][KG-NOISEPRUNE]、[默认][KG-SETUP] | **来源函数/手算/真实CUDA通过**。SP关闭，Eval/Match开启scale0.15/cap1e50；child激活顺序、已剪前缀、utility gap及excess扣权。value weighting使用剩余总权重；开启时覆盖根chosen prune/subtract，cap0亦然。独立240例、二阶统计及graph组合通过。[聚合][EZ-AGGREGATE] |

## 2. 开局、环境与输入

| ID／机制 | 直观理解 | 来源公式／流程 | 来源Train | 来源Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| D01 Balanced opening（KataGomo） | 随机骨架后挑接近平衡的最后一着 | 骨架长度权重{10,30,50,80,60,40,20,10,5,1,0,0}；空点权重∝Σ(1+center)/(distance²+d²)²，d=0.8·Exp(1)；候选权重(1−v²)^b；负根价值与最大候选权重控制拒绝 | Gomo概率0.99、b=4、拒绝率0.995；失败次数>20后降到0.8并继续重试。本表核对NOVC开局分支；来源另支持VC1/VC2的不同骨架长度分布。KataGo无此五子棋几何开局，Go用komi等 | Gomo Match概率1、b=10；随机选botB或botW评估；不等同成对执色协议 | [开局][GM-OPENING]、[调用][GM-PLAY] | **机制一致（独立/真实CUDA样例）＋比赛协议适配**。SP概率0.99/b4，Match1/b10；每次有效骨架后的balance尝试随机选botB/W，两个根视角及候选共用所选bot。几何/候选权重、拒绝和>20降拒绝率继续重试通过；移除负epsilon及独立RNG差异已登记。generator定义为参考黑方，开局复用于成对换色第二侧，不声称与来源逐局独立初始化协议相同。无VCN。[开局][EZ-OPENING-CODE]、[比赛][EZ-MATCH-CALL]。 |
| D02 Forbidden-plane dropout（KataGomo） | 训练时随机隐藏禁手提示 | 写每个row时useForbiddenInput~Bernoulli(0.5)；平面与特征可用标志同步改变 | Gomo训练writer每行抽样；仅Renju编码禁手；KataGo无禁手平面 | useForbiddenInput默认true，搜索保留提示 | [writer][GM-WRITE]、[编码][GM-INPUT]、[常量][GM-CONSTANT] | **输出行粒度一致（独立/真实CUDA样例）**。writer在repeats确定后逐最终行独立Bernoulli，保存forbidden_input；完整轨迹保留提示，view展开后同步隐藏两个平面及flag。重复行可不同，重新shuffle不再抽样；dropout0/0.5/1不改变开局、搜索、权重及结果。搜索/Eval/Match完整，其他规则flag0。[writer][EZ-WRITER]、[view][EZ-VIEW]。 |
| D03 Policy init | 纯网络policy随机走开局，增加状态覆盖 | Gomo n=max(0,floor(Exp(1)·mean−2·已有手数))、0.0002均匀分支；mean是抽样公式参数，不保证固定开局手数；Go n=floor(Gamma(k)·area·prop/k)，k=1为指数 | Gomo独立于balanced，SP必须显式配置init开关，mean loader默认12、T默认1；Go主线init=true、prop=0.08、k默认1、T默认1；还含komi/结束逻辑 | Gomo Match init默认false；启用后必须显式给mean（无loader默认），T默认1；Go Match默认init=true、prop=0.04，区别于SP | [Gomo init][GM-POLICYINIT]、[SP加载][GM-PLAYSETTINGS-SP]、[Match加载][GM-PLAYSETTINGS]、[Go init][KG-POLICYINIT]、[Go loader][KG-PLAYSETTINGS] | **NOVC机制/加载默认及显式预设已验证**。SP init开关必填、loader mean12/T1，baseline显式采用GM scripts mean6/T1.6；Match默认关闭，开启必须显式mean、T缺省1。每手按参考棋盘执色选botB/W；已有手数补偿与0.0002均匀分支保留。after/failure可配置消融默认均true，非Go面积/komi路径。[init][EZ-POLICY-INIT-CODE]、[调用][EZ-MATCH-CALL]。 |
| D04 Multi-board size | 一套网络混合尺寸，以mask排除padding | 尺寸按权重抽样；空间层mask；pool和BN按有效面积；value含尺寸条件 | Go尺寸7/9/11/13/15/17/19/8/10/12/14/16/18；权重1/4/3/10/7/9/75/1/2/4/6/8/10；矩形概率0.10 | Match示例19/13/9、权重90/5/5；固定局面按自身尺寸推理 | [抽样][KG-GAMEINIT]、[mask/pool][KG-MODEL]、[配置][KG-SP] | **用户环境适配（尺寸/mask已验证）**。保留方形15/14/13/12/11、权重100/10/5/3/1、canvas15；baseline及Eval/Match默认15。真实CUDA各尺寸/规则推理和smoke混合尺寸产样通过；mask/pool/BN回归，矩形配置明确拒绝。三架构均使用有效区域mask，指定预设的原源码前向/梯度及真实训练/导出已核对。[env][EZ-ENV]、[输入][EZ-GAME]、[pool/BN][EZ-NORM-CODE]。 |
| D05 Multi-rule | 条件化网络学习不同真实棋规 | 规则影响转移、终局和全局输入，不能只改标签 | Go ko=SIMPLE/POSITIONAL/SITUATIONAL、scoring=AREA/TERRITORY、tax=NONE/NONE/SEKI/SEKI/ALL、suicide=false/true、button=false/false/true；Gomo basicRule/VCN另按其配置 | 比赛可混合或固定规则；输入条件仍完整 | [Go抽样][KG-GAMEINIT]、[Go输入][KG-INPUT]、[Gomo输入][GM-INPUT] | **用户环境适配＋有限语料独立对照通过**。三规则可混训，baseline权重1/0/0；真实CUDA三规则产样、五种尺寸及长连/恰五/禁手终局通过。136125个KataGomo独立禁手/类别对照含邻域穷举、边缘、同方向双四及假三/递归；未证明所有递归局面等价。无Go规则、VCN或矩形。[env][EZ-ENV]、[转移][EZ-GAME]、[独立检查](/home/sky/RL/EtaZero-lab/EtaZero_V0/tests/reference/check_katagomo_rules.py)。 |

## 3. 网络、归一化与优化

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train（L） | KataGo Eval／模型发布 | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| N01 Convnet（普通残差） | 两层空间卷积与残差学习局部形状 | x→x+Conv3(ActNorm(Conv3(ActNorm(x))))；部分块带global pooling | 类别由model-kind/block_kind定义，不由selfplay cfg定义 | 同架构；归一化按推理模式 | [ResBlock][KG-RESBLOCK]、[配置][KG-MODELCONFIG] | **已实现（指定预设）＋五子棋适配**。配置architecture=plain、C128/B10，mid128，第5/8块gpool32，head32、value-hidden80，来源b10c128-fson-mish/v15。直接使用两层3×3残差块，无NBT外层投影；独立固定来源完整模型映射前向/梯度、初始化调用和参数组在CPU/CUDA通过。真实训练/恢复/导出验收见 [8a证据](EtaZero_V0/data/validation_batch8_plain_20261002/manifest.json)。[内层][EZ-NBT-CODE]、[模型入口][EZ-NETWORK-CODE] |
| N02 Convnet+NBT | 在较窄通道内堆残差块，降低计算 | 1×1降通道→K个内部残差块→1×1升通道→外残差；NBT2的K=2 | 主线架构系列含b18c384nbt；核对Eta结构的具体预设为b5c192nbt-fson-mish，不混称b18参数 | 同结构；导出可只保留所需heads | [NBT][KG-NBT]、[b5][KG-B5]、[后缀][KG-SUFFIX] | **指定预设已验证＋五子棋适配**。C192/B5、mid96、gpool32、head32、value-hidden80，NBT2，outer第2/4块gpool；trunk对应b5c192nbt-fson-mish。输入5平面/6全局、六policy/主WDL/三TD/error是五子棋子集。批次8a以固定来源完整模型核对前向/梯度、所有初始化调用及参数组；独立CUDA对照通过，训练/导出回归见8a证据，不是b18全模型。[NBT][EZ-NBT-CODE]、[全网][EZ-NETWORK-CODE]、[宽度][EZ-WIDTHS] |
| N03 Transformer+NBT | 在瓶颈中用注意力连接远距离落点 | softmax(QKᵀ/√d+mask)V；RMSNorm、2D RoPE、FFN/SwiGLU；NBT2/3按架构 | modelconfigs显式选transformer；不将其等同mainb18时期架构 | 同结构推理，attention排除padding | [NBT Transformer][KG-TRANSFORMER]、[attention][KG-ATTENTION]、[配置][KG-MODELCONFIG] | **指定预设已实现＋五子棋适配**。architecture=transformer严格对应bare b5c192h3nbttfrs/v17：C192/B5、mid96、3×32头、FFN256、fixup/ReLU、head32/value-hidden64；真实RMSNorm/固定2D RoPE/key-only padding/SwiGLU和四内残差，末端无BN。初始化/参数组/独立全模型eager及同条件编译前向梯度均在CPU/真实CUDA核对，三精度训练/恢复及FP32/FP16原生导出通过，见[8b证据](EtaZero_V0/data/validation_batch8_transformer_20261002/manifest.json)；Q缺省关闭，true添加逐动作纯W−L Q，Go score Q去除。使用来源可选NCHW/SDPA，AMP沿融合SwiGLU；与默认NHWC/flex执行性能不等价。[Transformer][EZ-TF-CODE]、[融合核][EZ-SWIGLU-CODE]、[全网][EZ-NETWORK-CODE] |
| N04 初始化、固定尺度、最终masked BN | 固定残差尺度稳定网络，只在末端用BN | fson内部y=((1+γoffset)sx+β)mask；外块入口1/√(i+1)、内部入口1/√(j+1)、NBT出口1/√(K+1)、末端1/√(B+1) | 方差校正截断正态；最终BN有效点总体mean/std，epsilon=1e−4、EMA=0.001；EMA对象为std | running_mean/std推理，导出折入scale/bias；内部fixed scale始终保留 | [NormMask][KG-NORM]、[NBT初始化][KG-NBT]、[全网初始化][KG-INITIALIZE] | **指定分支独立对照＋五子棋适配**。fson二值mask分支的内部可学gamma offset/beta、固定scale、Mish gain及2σ截断方差校正对应来源；最终BN按有效点总体mean/std、EMA=0.001、epsilon=1e−4；Eval用running统计并折叠。Transformer按bare预设使用fixup/ReLU增益、零出口及RMSNorm，无最终BN；170个同种子初始化对照及全部参数调用尺度/角色核对。非二值mask不是当前协议支持范围。[init/norm][EZ-NORM-CODE]、[NBT][EZ-NBT-CODE]、[推理折叠][EZ-INFERENCE-NET] |
| N05 Global pooling与heads | 注入全盘与尺寸信息 | policy pool=[mean,mean(√A−14)/10,max]；value pool=[mean,mean(√A−14)/10,mean((√A−14)²/100−0.1)] | 训练policy有opponent/soft/optimistic等；value另有score/ownership等Go heads，依版本 | 导出搜索所需policy/value，不用全部训练heads落子 | [pool/heads][KG-MODEL]、[导出][KG-EXPORT] | **pool/指定heads已实现＋环境适配**。pool面积公式及gpool bias路径对应来源；Mish下合法有效点max与来源padding −1屏蔽等价。移除pass/score/ownership等Go heads，六policy、TD/error已实现；搜索导出ordinary/short-optimistic/WDL/error stdev，搜索应用在6已验收，plain/Transformer已接入各自预设，v17可选纯W−L Q已接入，验收通过；node targets/量化/loss见[目标][EZ-Q-TARGETS]、[量化][EZ-Q-QUANTIZE]、[Qloss][EZ-Q-LOSS]。[pool][EZ-NORM-CODE]、[heads][EZ-HEADS] |
| N06 Learner D4 | 八对称训练减少方向过拟合 | 每batch均匀抽g，同步变换空间输入/targets，global/value不变 | reader randomize_symmetries=true | 来源validation调用也开启randomize_symmetries=True，sync可跳过validation；搜索D4是S16/S17 | [reader][KG-D4]、[train调用][KG-TRAIN] | **训练/验证D4已验证**。L每batch均匀抽D4，obs/六policy/Q值/Q访问目标同变换，global和值不变。baseline默认启用验证；验证时随机D4独立于训练开关，raw eval/no_grad，AMP/compile真实CUDA通过。[变换][EZ-D4-CODE]、[调用][EZ-TRAIN-LOOP] |
| N07 SWA（EMA型） | 平滑参数，形成发布模型 | 首样本复制；之后θs←θs+(θ−θs)/scale；BN buffers复制，非参数平均 | 默认scale=8、period=max(1,samples_per_epoch//2)；只在Lookahead同步后采样，跨epoch保留 | 导出SWA权重；SP与eval均使用发布模型，无在线平均 | [EMA][KG-SWA]、[采样][KG-LOOKAHEAD]、[导出][KG-EXPORT] | **已验证（基础 EMA/消费时钟）＋调度适配**。SWA period=64000消费样本（1000×128/2），scale8；overflow 也推进累积，只在 Lookahead 同步点采样，BN buffers复制。真实CUDA验证首样本、跨轮保存、SGD/AdamW发布导出与恢复。固定预算/quota及多分段时钟已在9b核查，控制器整轮提交/发布边界已验证，来源调度差异单列。[优化状态][EZ-OPT-CODE]、[发布][EZ-OPT-INFERENCE] |
| N08 Lookahead | 定期平均fast与slow权重 | 每k步slow←slow+α(fast−slow)，fast←slow；LR除以α补偿 | 默认k=6、α=0.5、sub_epochs=1；α≥1关闭；每个subepoch开始重置同步计数，epoch末丢弃未同步fast，保留优化器 | 使用发布权重，无在线优化 | [同步][KG-LOOKAHEAD]、[计数重置][KG-LOOKAHEAD-COUNTER]、[LR][KG-LR] | **已验证（基础同步、skip、状态恢复）＋调度适配**。k6/alpha0.5，LR除alpha；同步按消费batch计数，overflow不暂停时钟；分段入口重置counter但不复制fast，轮末恢复slow。sub_epochs默认1，多分段固定floor预算已接入；手算分段边界及真实CUDA恢复通过。[同步][EZ-OPT-STEP]、[轮末][EZ-TRAIN-LOOP] |
| N09 Warmup | 随已消费训练样本数提高LR | 每25万样本分母20/14/10/7/5/3/2/1.4；200万后1 | 默认开启；global_step_samples包含DDP世界大小；同时进入WD公式 | 不运行LR调度 | [warmup][KG-LR] | **已验证（公式及刷新边界）＋本地轮适配**。warmup由含overflow的消费样本驱动；轮开始刷新，轮内累计样本≤2亿每5批、之后每50批刷新，使用消费后计数供下一batch。八个阈值及2亿边界、真实AMP恢复已验；固定预算/quota及多分段时钟已在9b核查，控制器整轮提交/发布边界已验证，来源调度差异单列。[warmup][EZ-WARMUP-CODE]、[调度][EZ-OPT-CONFIGURE] |
| N10 优化器／分组／自适应WD | 按职责更新和衰减，稳定网络范数 | [F9][FORMULA-F9]：SGD/AdamW、head/noreg组、范数ratio自适应衰减，norm_kind决定分支 | SGD默认momentum0.9；head LR factor=0.5；SGD每样本LR=3e−5；fson属于BN型WD分支；当前默认每100批norm快照，关闭only-at-print时使用打印点衰减的运行均值 | 优化器仅续训使用 | [WD][KG-WD]、[LR][KG-LR]、[group roles][KG-MODEL]、[norm采样][KG-NORM-TIMING] | **已验证（fson/fixup组公式/范数时序）**。SGD/AdamW fson六组31104案例、fixup七组36288案例的LR/WD均与来源一致。范数更新前采样，默认100批snapshot；可选逐batch运行均值在打印点保留历史0.001，含lookahead_print筛选。5760来源时序对照通过；Transformer实际参数逐一核对来源归组，attention WD单列normal_attn并乘0.5，RMSNorm权重归noreg。[分组][EZ-OPT-CODE]、[调度][EZ-OPT-CONFIGURE] |
| N11 梯度裁剪与精度 | 限制异常更新；精度与跳过更新影响计数 | cap=c√(global_batch/256)/√max(1e−7,LRscale) | SGD fson/fixup/fixscale c=2500；SGD BN/brenorm/fixbrenorm c=5500；AdamW c=11000；FP16/BF16独立开关 | 主线SP CUDA FP16=true；Match后端auto；训练AMP/推理精度独立 | [裁剪][KG-GRAD]、[AMP][KG-TRAIN]、[SP配置][KG-SP] | **已验证（fson裁剪/训练AMP/推理精度）＋LibTorch适配**。SGD基数2500、AdamW11000，24组来源阈值一致；平均梯度override保留。训练heads/loss实际FP32，FP16 overflow跳步不重试，消费与成功更新分计。真实CUDA eager/compile、FP16/BF16与恢复通过；SP baseline FP16，Eval/Match auto在CUDA解析为FP16并记录，FP32显式override保留；指定D4/真实多server输出精度已验证。[裁剪][EZ-GRAD-CODE]、[更新][EZ-TRAIN-LOOP]、[配置][EZ-TRAIN] |

## 4. 监督目标与样本权重

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train | KataGo Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| L01 硬policy、opponent、soft policy | 当前/对手策略与展平辅助目标 | soft(p)=normalize(((p+1e−7)·on_board)^0.25)；opponent系数0.15、soft默认8 | 四基础heads之外还有optimistic/Q；ordinary CE系数版本≤11为1、12…17为0.930；opponent不可用权重0 | 使用导出的policy，不在线计算loss | [目标/版本][KG-METRICS]、[opponent CE][KG-POLICYLOSS]、[writer][KG-WRITE] | **指定版本监督已实现＋环境适配**。普通policy CE=0.930（对应v15及v17），opponent0.15、soft8/soft-opponent1.2；独立概率/解析梯度与padding域通过。六policy含optimistic、TD/error和int16量化已在v15及缺省无Q的v17路径实现，独立梯度/真实CUDA通过；v17可选纯W−L Q已接入，验收通过；node targets/量化/loss见[目标][EZ-Q-TARGETS]、[量化][EZ-Q-QUANTIZE]、[Qloss][EZ-Q-LOSS]。[loss][EZ-LOSS-CODE]、[writer][EZ-WRITER] |
| L02 Value target/loss | 终局与搜索辅助监督学习价值 | 基础CE内部1.20再乘value_loss_scale；CLI默认0.6，有效0.72；另有TD/score等loss | 主局真实终局与多时间尺度targets；side可用搜索value | Go第三类no-result，真实draw影响W/L分配；不同于五子棋draw分类 | [CE][KG-VALUELOSS]、[targets][KG-WRITE]、[总loss][KG-METRICS]、[CLI][KG-TRAIN] | **环境适配＋指定价值监督已验证**。真实终局当前方W/D/L，基础CE=0.72（1.20×0.6）；独立one-hot及软WDL梯度通过。TD/error和side数据契约已实现并验证，侧分支产生及重分析已在7接入，hint/fork前缀与首值复制已接入；Go score/no-result不直接映射为五子棋draw。[target][EZ-VIEW]、[CE][EZ-LOSS-CODE]、[终局][EZ-GAME] |
| L03 搜索合法域／训练softmax域 | 搜索屏蔽非法动作，训练仍抑制棋盘内差着 | 搜索legal mask；训练仅屏蔽padding，soft目标对占用点也有epsilon质量；Go含pass | policy目标归一化，规则/价值含义保留 | 搜索legal域继续生效，五子棋无pass | [mask][KG-METRICS]、[policy head][KG-MODEL]、[NN后处理][KG-NNCACHE] | **域区分已验证＋环境适配**。搜索legal为棋盘内空点，Renju禁手可提交并判负，不随dropout改变；终局叶无NN。learner只屏蔽padding，occupied/禁手仍在softmax及soft epsilon域；独立梯度样例覆盖这两个点及padding。无pass；三架构及v17可选Q沿同一on-board训练域。[规则][EZ-GAME]、[NN][EZ-EVALUATE-NODE]、[loss][EZ-LOSS-CODE]。 |
| L04 采样次数、loss归约、计数 | 频率只重分配一次，避免重复加权 | count=floor(w)+Bernoulli(frac(w))；重复整行；CE按样本求和反向，日志再归约 | 普通path先随机取整再write；distill/reanalysis另有数据权重 | visits不同于训练行或梯度更新 | [取整][KG-ROUND]、[writer][KG-WRITE]、[sum loss][KG-METRICS] | **基础频率/归约一致（静态）＋消费/AMP已验证**。采样权重只作用随机取整重复频率；backward使用batch sum，日志mean。total_steps/total_samples计消费（含skip），optimizer_steps计成功更新，reader游标随消费提交；真实CUDA overflow/恢复通过，新增监督及重分析已接入；Q逐行量化已接入，固定训练消费、发布和整轮恢复边界已有真实CUDA验收。[取整][EZ-WEIGHTS]、[repeat][EZ-VIEW]、[backward][EZ-TRAIN-LOOP] |

## 5. Replay窗口与shuffle

四个参数控制近期输入窗口与shuffle输出抽样，`4Parameters`是组说明，不是第五个trick。KataGo还支持taper_window_scale/max_rows/add_to_data_rows等，四参数是子集。

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train | Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| R01 MinRows | 窗口起点/下限与随机冷启动计数限制 | [F10][FORMULA-F10]中的m；random/tdata累计最多计m行，N为usable rows | shuffle默认m=250000，脚本不覆盖；不足门槛不shuffle | 不适用 | [窗口][KG-WINDOW]、[random计数][KG-RANDOMROWS] | **累计/窗口公式已验证＋当前参数差异**。m150000，采用[GM shuffle脚本][GM-SHUFFLESH]的最小行数；KataGo CLI默认250000，N=min(random_rows,m)+postrandom_rows；raw总量与quota独立，近期按实际mtime取完整末片。random/tdata映射为model_id=random:seed前缀；raw range end不封顶。超额random、元数据/mtime交错与2592来源公式通过。[窗口][EZ-SHUFFLE-CODE]、[累计][EZ-CATALOG]、[quota][EZ-RUNTIME-PLAN] |
| R02 TaperExponent | 控制窗口渐近增长 | W按累计量的p次幂增长，不是采样概率指数 | shuffle脚本示例p=0.65；CLI默认p=1 | 不适用 | [公式][KG-WINDOW]、[脚本][KG-SHUFFLESH] | **来源公式已验证＋当前参数差异**。p=.8采用[GM shuffle脚本][GM-SHUFFLESH]，KataGo脚本.65及CLI默认1单列；三窗口扩展与整数取整均实现。2592案例含m/scale/offset/max组合。[公式][EZ-SHUFFLE-CODE]、[配置][EZ-TRAIN] |
| R03 ExpandPerRow | 控制起点新增一行引起的窗口增量 | 取整前的幂律归一化使起点导数=a，见F10 | 脚本示例a=0.4；CLI默认a=1 | 不适用 | [公式][KG-WINDOW]、[脚本][KG-SHUFFLESH] | **来源公式/扩展已验证＋当前参数差异**。a=.3采用[GM shuffle脚本][GM-SHUFFLESH]，KataGo脚本.4及CLI默认1单列；s默认m、offset0、max无上限，可显式设置taper_scale/add_to_data_rows/max_rows。合法性和来源原函数对照通过。[公式][EZ-SHUFFLE-CODE]、[配置][EZ-TRAIN] |
| R04 KeepTargetRows | 限制每次shuffle输出量，不限制窗口 | q=min(1,K/actual_window_rows)；组内保留round(nq)行；all全保留 | 脚本示例K=20000000；CLI必填；组内先随机打乱再截取 | 不适用 | [组内抽样][KG-SHARDIFY]、[比例][KG-SHUFFLE]、[脚本][KG-SHUFFLESH] | **分组/采样/mtime已验证＋随机流适配**。K20M/all；q按MD5过滤前窗口计算，文件先乱序，组累计rows≥阈值才封组，round(nq)均匀子集；waves只采样一次，任务使用独立命名空间，固定输出文件数。分组/规划/字段守恒及重建通过，不宣称OS源RNG序列或资源规模相同。[抽样][EZ-SHUFFLE-SCATTER]、[排序][EZ-CATALOG]、[窗口组装][EZ-SNAPSHOT] |
| R05 Replay ratio／更新节奏 | 限制每新增数据可消费多少训练样本 | train bucket随shuffle报告range end的新增累计行增长；epoch开始预扣samples_per_epoch，实际消费另按global batch计数；有上限/等待 | CLI比例未指定时不限制，由启动命令决定；mainb18不定义；同步脚本也启用bucket/no-repeat，异步各阶段同时运行 | 不适用 | [train bucket][KG-TRAIN]、[启动][KG-TRAINSH] | **调度语义差异**。逐轮SP→shuffle→固定1000×128样本消费→SWA发布，ratio=8，steady-state quota每轮新增16000有效行，完整局会超额，不足累计目标继续补局；冷启动单独锚定，random窗口封顶不改quota。CPU及真实CUDA三轮短产样补局通过。来源同步/异步train bucket均为消费限额，与此固定轮事务不同；不能由ratio数值一致推导模型更新时点一致。工程调用及恢复边界见[E21/E24](#engineering-control)、[E16/E18](#engineering-data)。[计划][EZ-RUNTIME-PLAN]、[流水线][EZ-RUNTIME-LOOP]、[reader][EZ-READER] |

## 6. 跨机制边界

| ID／机制 | 直观理解 | 来源公式／流程 | KataGo Train | KataGo Eval／Match | 源码依据 | EtaZero V0 核查 |
|---|---|---|---|---|---|---|
| A01 价值视角／终局／utility | 两方都选对自己有利的动作，不混淆任务 | 内部white-positive，选点按pla变号；utility含winloss/draw/no-result/score；终局不调用NN | winLoss=1、staticScore=0.05、dynamicScore=0.30；还有komi/动态score中心 | Match默认staticScore=0.1、dynamicScore=0.3；不是纯W−L | [视角][KG-FPU]、[终局模拟][KG-BUDGET]、[默认][KG-SETUP] | **五子棋视角、终局和辅助监督已验证**。节点当前方W/D/L，边backup变号，historical/surprise统一黑方；reward0/discount1、终局精确±1/0且无NN/重复reward。禁手叶为次方+1、父方−1；胜/和/长连及黑白终局CUDA/独立样例通过。纯W−L与独立draw；无Go score/komi，辅助TD/error/optimistic及side/reanalysis的视角和有效性门控已验证。[适配][EZ-STATE]、[终局][EZ-GAME]、[统计][EZ-WEIGHTS]。 |
| A02 搜索预算／等算力口径 | 分清已有访问、新增模拟、NN请求 | maxVisits含已有+新增；maxPlayouts限制新增；根初始评估计访问；根多对称多次NN但只一次访问 | full cap=2000；并行调度不是逐位固定 | 示例500v；时间限制/共享GPU改变实际算力条件 | [搜索循环][KG-BUDGET]、[多对称][KG-SYM]、[比赛][KG-MATCH] | **预算机制已实现＋并行发放/资源差异**。SP400/70是用户明确保留的预算差异，Eval/Match500v。initial_visits/new_playouts/simulations拆分，fresh根占1新playout，ensemble NN计数独立；maxPlayouts/maxTime/显式停止均可用。时间至少2新plays，终局无NN；V0并行严格发放，来源可在途超限，不宣称并行序列/等时间等价。[预算][EZ-RUN]、[参数][EZ-SP] |
| A03 NN cache／模型／朝向 | 避免重复推理，不混淆不同输入条件 | cache hash含棋局和NNInputParams；单对称朝向不入hash；命中复用原输出；根多对称跳cache | 每模型独立evaluator，nnRandomize=true；主线switchNetsMidGame=true | 同机制；无根噪声也不代表NN确定 | [hash][KG-INPUT]、[cache][KG-NNCACHE]、[集成][KG-SYM] | **朝向cache机制已实现＋用户模型适配**。cache存canonical raw输出，key为未变换obs+globals+Tnn、不含sym；单次命中复用首次输出且不消耗NN朝向RNG，根K>1读写绕过。per-model隔离/换模型清树，模型ID或实际路径变化重建cache；同ID换路径的真实CUDA原生协议验证通过；用户保留每轮固定模型，无局内switch。PDA条件的输入/cache隔离和真实CUDA已验。[cache][EZ-CACHE]、[D4][EZ-STATE]、[worker][EZ-WORKER]；[E07](#engineering-concurrency)、[E23](#engineering-control)。 |
| A04 Checkpoint／续训／发布 | 区分训练状态和推理权重 | 保存model/optimizer/train_state/metrics/SWA；export可跳optimizer；Lookahead cache/RNG须单独核对 | learner/SP/shuffle独立进程，不定义EtaZero整轮事务恢复 | 加载导出权重；gatekeeper独立于搜索 | [save/restore][KG-SAVE]、[Lookahead][KG-LOOKAHEAD]、[export][KG-EXPORT] | **恢复/发布协议适配；learner与整轮边界已验证**。来源save保存model/optimizer/metrics/train_state/SWA，未捕获局部Lookahead cache/counter、全局RNG或AMP scaler；Eta额外保存model/optimizer/AMP scaler/RNG/reader/Lookahead/SWA/counters/分段。恢复以已提交整轮为准，未完成轮归档后重做，不承诺控制器恢复中间checkpoint继续同轮；独立learner恢复已逐位验证。发布SWA（未采样时raw）、验证导出，恢复及发布前校验完整manifest/契约/权重/已提交checkpoint身份；非KataGo独立异步learner/SP协议；无gatekeeper。[checkpoint][EZ-CHECKPOINT]、[恢复][EZ-RESTORE]、[导出][EZ-EXPORT-CODE] 同步/异步来源与恢复范围分别见[E23/E24](#engineering-control)。 |

## 来源公式与复杂分支

### F1 FPU

令m为已分配子节点的搜索先验质量，a=min(1,m^power)，将utility转换为行棋方视角：

```text
Qbase = a·Qparent + (1−a)·Vnn
FPU0 = Qbase − reduction·sqrt(m)
FPU = FPU0 + loss_prop·(−utility_radius−FPU0)
utility_radius = winLossUtilityFactor + staticScoreUtilityFactor + dynamicScoreUtilityFactor
```

来源关闭visited-policy插值时，`Qbase=fpuParentWeight·Vnn+(1−fpuParentWeight)·Qparent`；该权重是NN系数，默认0为纯parent。五子棋纯W−L下radius=1，不能将1写成所有KataGo配置的常量。

### F2 Policy target pruning

```text
stable = argmax_a [W_a·max(0,N_a−1)/max(1,N_a) + 2P_a]
best_score = Q_stable + explore_scaling·P_stable/(1+W_stable)
other_weight = ceil(min(W_a,max(0,explore_scaling·P_a/(best_score−Q_a)−1)))
```

分母非正时反解保留原权重，但所有非稳定边仍执行`ceil`；稳定边保留原权重、不取整。这一区分适用于uncertainty或图共享产生的非整数权重，取整发生在LCB资格判断之前。根选择反解不以forced系数>0为前提；Go还含ending bonus/pass抑制。随后执行LCB及chosen prune/subtract。训练policy不要求等于原始visits归一化。

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

来源value surprise统一白方视角，EtaZero统一黑方视角，从终局概率倒序执行 `future←future+now·(search_WDL_t−future)`，再取clip(KL(future∥raw_NN_t),0,1)。来源另有直接搜索value surprise开关；reanalysis可禁止未重分析cheap手的policy excess份额。全零surprise不会伪造为均匀分配。

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
W_a = noise_pruned_weight_a（开启noise pruning）
raw_weight_a = applicable_chosen_prune_subtract(W_a)·factor_a
desired_weight_a = raw_weight_a·desired_total/Σraw_weight
```

来源t(3)CDF为插值表。初始NN样本、子树均值、二阶矩和weight_sq一起聚合；weight_sq按缩放平方更新。该机制改变backup，区别于训练频率。noise开启时覆盖根chosen prune/subtract，均值和desired_total为noise后的剩余权重；η=0仍执行适用分支。初始NN权重由uncertainty决定；anti-mirror另有禁用分支。

### F7 根／落子温度

```text
T(t) = Tlate+(Tearly−Tlate)·0.5^(t/halflife·19/sqrt(board_area))
```

来源t可包含initialTurnNumber等历史补偿；根policy与chosen温度共用半衰参数。onlyBelowProb=1表示无高概率保护；小于1时只变换低概率尾部。T≤1e−4且无保护时取首个最大权重；保护开启时不能统一概括为argmax。NN温度与这两种温度相互独立。

half-life字段不是所有棋盘上相同的实际手数：实际半衰手数为`halflife·sqrt(area)/19`。例如EtaZero当前SP为15×15、参数15时约11.84手；Eval/Match同尺寸、参数19时是15手；来源19×19、参数19时是19手。

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

SGD base_wd=0.00125、AdamW=0.009；input/normal_gamma/output/noreg另有系数。AdamW LR确实另乘sqrt(global_batch/256)。当前来源默认KATAGO_MODEL_NORMS_ONLY_AT_PRINT=true，每100批以norm快照覆盖running_metrics相应项；关闭时逐batch累积运行均值，norm_*_batch仅在打印点将历史和/权重缩为0.001，不使用逐batch的0.995 EMA。norm baseline在初始化时建立，恢复/重设分支见train_state。LR/WD在累计样本≤2亿时每5 batches刷新，之后每50 batches刷新；norm的采样/平滑节奏也影响WD。MuOn/NorMuon/Aurora、自动LR、RepVGG等额外分支不代表EtaZero已实现。

### F10 Replay窗口与shuffle

N为usable累计行，m为min_rows，s为taper_window_scale默认m，d为add_to_data_rows默认0：

```text
x = N−m+s+d
W = max(m,int(m+a·(x^p−s^p)/(p·s^(p−1))))
W = min(W,max_rows)                       # 仅指定max_rows时
keep_prob = min(1,K/actual_window_rows)
```

四参数子集s=m、d=0且无max_rows时，化为 `max(m,int(m+a·(N^p−m^p)/(p·m^(p−1))))`。选完整近期分片可使actual window超过W；组内round使输出略偏离K。来源按文件时间选择近期数据，random累计单独封顶；不能只比幂律公式而忽略N定义。

<a id="engineering-audit"></a>

## 7. 工程源码审查（E01—E26）

### 快照、执行路径与证据边界

本节记录 **2026-10-02** 第10批对E01—E26的最终源码、配置、边界及调用复审。目标HEAD为 `185eee220fd786a6e7436ae4e2332b929904bdb0` 加当前工作区；各条目分别标明真实运行、静态和性能证据范围，不把有限运行样例扩为全部分支验收。

KataGo / KataGomo 的 HEAD 分别与计划固定的 commit 一致，两者工作区均干净；`reference_sources.json` 中 0E 时 KataGo 的39文件及 KataGomo 的7文件 SHA256 匹配；当前来源登记为KataGo51、KataGomo18文件，覆盖禁手、终局、writer/input/loader、采样/图搜索、模型及初始化；已登记training_data_generator、shuffle.sh及synchronous_loop.sh，按固定commit核对文件消费和validation调用。新增读取的同步/异步脚本、summary、training data generator 与 CUDA backend include 属于同一 KataGo commit 的干净源码，不能套用别的版本行为。

工程目标集合为 `git ls-files --cached --others --exclude-standard` 在 `EtaZero_V0/python/etazero`、`cpp/include/etazero`、`cpp/src`、`configs/baseline` 四个路径下返回的tracked及非忽略untracked的全部 **54 个文件**；按仓库相对路径排序，对 `path + NUL + 文件SHA256十六进制 + LF` 求集合 SHA256，0E初始结果为 `3d56e54ed95b23ff8cf65a6a4c077dd8eec2edfd99290219fc96b976edc3778c`，当前第10批完整复审目标结果为 `4f4589c68f3a08731e144f2274c482a2667727271d9f9aae4a5c5b9ff7c67bf4`。0E原始范围为50个tracked文件；当前新增sampling头/实现及Transformer/fused SwiGLU两文件一并纳入。这与算法表的28文件集合是两个范围；后续修改应重新计算受影响快照。

**路径分别判定。** 当前 V0 是单 learner、逐轮串行阶段；来源的[同步路径][E-KG-SYNC]可以作为阶段顺序对照，仍使用 train bucket、no-repeat-files、subepoch 和可选 gating。来源的[常驻异步路径][E-KG-PROFILES]与[shuffle/export loop][E-KG-LOOP]同时推进各阶段、轮询新数据和模型；V0 没有这些能力。单机同步是0E的阶段顺序对照路径；批次0用户现已确定保留本地逐轮规划、不使用训练桶，见第8节。同步脚本的selfplay1参数不覆盖指定的mainb18算法基准。

下表每行均已检查所列来源、目标、baseline 配置和调用链。**0E证据级别为“源码/配置（静态）”；E20/E21/E24相关learner子路径增加了批次1证据，E06/E07及搜索停止/故障子路径增加了批次2证据，E09禁手输出行及开局调用子路径增加批次3独立/真实CUDA证据**：不将已有测试文件、文档中的历史验收或本轮 SHA256 检查算作算法、CUDA、并发、中断或性能验收。结论中的“机制一致”仅限明确写出的子机制；“实现不同/证据不足”保留待验，不能当成整项通过。最后一列给出后续验证及负责批次。

<a id="engineering-control"></a>

### 控制流、模型与恢复

| ID | 来源 → 目标入口 | 实际配置与调用链 | 差异及影响 | 对齐结论 / 证据级别 | 验证范围／证据 |
|---|---|---|---|---|---|
| E01 | [同步脚本][E-KG-SYNC]、[异步说明][E-KG-PROFILES] → [Controller.run][E-EZ-RUNTIME] | iteration 0 仅产数据；其后 selfplay→shuffle→固定 train_steps→export→整轮 state 提交，停止/预算在轮边界检查 | 同步阶段顺序有对应；来源先处理 gating，额度不足或无新文件可退出 learner；V0 每轮请求固定更新。异步阶段重叠与独立重启缺失 | 阶段顺序子集一致；消费/触发差异；异步能力缺失 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E21 | [补桶][E-KG-BUCKET]、[预扣][E-KG-DEBIT]、train.py subepoch/Lookahead/SWA → [iteration_plan][E-EZ-PLAN]、[learner][E-EZ-TRAIN]、[Optimization][E-EZ-OPT] | baseline每轮1000×128消费量，ratio8规划产样；LR/WD在轮开始及5/50批刷新，norm更新前100批snapshot/运行均值；skip推进消费，成功optimizer另计 | 用户保留固定轮规划，无来源bucket预扣/等待；时钟及AMP语义已修正，sub_epochs默认1，非默认floor分段已接入。原始train.py时钟3696步、5760范数时序、warmup/2亿边界及eager/compiled AMP跳步逐位恢复已验 | 固定轮quota/消费及多分段时钟已验证；bucket调度差异保留 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E23 | [轮询/切网][E-KG-POLL]、[manager][E-KG-MANAGER]、[export][E-KG-EXPORT] → [export_model][E-EZ-EXPORT]、[launch][E-EZ-LAUNCH]、[worker][E-EZ-WORKER] | SWA 有采样优先，否则 raw；TorchScript 及 native 稀疏 probes 验证后 rename，整轮提交后更新 current.json；每轮 plan 固定模型，产样完 release | 来源同步也可直接发布而不 gating；异步每 20 秒轮询、acquire/release 多代模型，并可局内切换。V0 没有轮询/局内切换，不存在异步旧模型与在途请求交接；当前只验证所选 sparse probes，不能覆盖所有 Renju 局面 | 完整目录发布机制对应；格式/校验实现不同；模型时点差异与能力缺失（A03/A04） | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E24 | [save][E-KG-SAVE]、[generator][E-KG-GENERATOR]、同步/异步重启 → [commit_checkpoint][E-EZ-SAVE]、[recover_iteration][E-EZ-RESTORE] | V0保存model/optimizer/scaler、Lookahead/SWA、norm snapshot或运行和/权重、消费及成功更新计数、RNG与消费cursor；controller以整轮state为authority | 来源保存范围与Eta整轮协议仍不同；底层learner在AMP skip后保存并恢复全部新增状态，连续/恢复模型、优化器、SWA/slow、scaler、reader和RNG逐位对照通过；controller九处中断及提交前归档重跑/提交后不重跑已验，current恢复保持原utc；入口腐坏拒绝在worker之前 | 完整learner保存范围及选定整轮边界已验证；来源恢复范围/协议差异保留 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |

<a id="engineering-data"></a>

### 样本链路与 shuffle

| ID | 来源 → 目标入口 | 实际配置与调用链 | 差异及影响 | 对齐结论 / 证据级别 | 验证范围／证据 |
|---|---|---|---|---|---|
| E11 | [summary][E-KG-SUMMARY]、[shuffle 累计][E-KG-ROWS] → [Catalog][E-EZ-CATALOG]、Controller.scan | 启动全扫描、后续扫描当前轮，SQLite path/shard/game去重累计实际重复行；已索引文件stat变化则核hash并更新mtime；原始视图核hash及完整轨迹 | 来源坏文件告警排除，本地明确报错；未扫描的旧原始数据不承诺每次counts重新鉴真。窗口random封顶独立于真实产样计数 | 去重/损坏/mtime/原地修改已验；校验策略与source不同 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E12 | [近期窗口][E-KG-ROWS]、shuffle.py 排序/compute_desired_num_rows → [Catalog.entries][E-EZ-CATALOG]、[build_snapshot][E-EZ-SNAPSHOT] | m150000,p.8,a.3,K20M，窗口三参数采用GM shuffle profile；三窗口扩展；actual mtime近期完整片；N=random_capped+postrandom | raw range end=raw_total+int(offset)，不封顶；稳定mtime平局输入顺序不同于source文件系统遍历。CLI、脚本及sync覆盖分别记录 | 2592原函数窗口案例及random超额/mtime交错通过 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E13 | [group_files_by_rows][E-KG-GROUP]、[shardify][E-KG-SHUFFLE] → [_groups][E-EZ-GROUP]、[_scatter][E-EZ-SCATTER] | 输入文件先乱序，累计训练rows到≥group阈值再封组；组内round(nq)均匀子集 | 有效group资源阈值在采样前确定，完整末文件可超额；OS源与可重建本地随机序列不同 | 30来源分组案例、小q round反例及1/3wave守恒通过 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E14 | [shardify][E-KG-SHUFFLE]、shuffle.py merge_bucket → [partition_rows/_scatter][E-EZ-SCATTER]、[_merge_bucket][E-EZ-MERGE] | 共同字段排列、IID桶计数＋均匀排列连续切片，固定F文件均分；取消溢出桶递归 | 独立SeedSequence partition/stage/wave/group/bucket流；实际桶超过预算明确失败，不更改输出计划。不宣称来源种子序列相同 | 桶/字段配对、空文件和实际GPU数据流已验证；理想分布与RNG序列分开陈述 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E15 | [multi-wave][E-KG-WAVE] → [_write_data][E-EZ-WAVE]、_two_phase/_merge_bucket | multi-wave首次scatter应用q一次，wave内q=1，各任务独立命名空间；检查总行数并删临时wave | 保留集合在同种子1/3wave一致，排列不同；空wave无输出，空bucket按固定F输出。资源预算异常明确失败 | wave守恒/重建、真实多服务FP16预取验收 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E16 | [桶/输出计划][E-KG-OUTPUT]、[read_npz][E-KG-READ] → [_merge_bucket][E-EZ-MERGE]、[BatchReader.next][E-EZ-READER] | B=round(approx_rows/bucket_rows)、固定F=bucket_rows/nominal；真实桶均分；每文件完整batch前缀 | 当前nominal65536（bucket同为65536，F=1）不是硬文件上限；小文件/空文件可不供给batch；manifest报告usable和tail。取消跨文件拼batch与尾部回绕 | 324source输出规划、原reader68batch row ID及本地尾部/恢复通过 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E17 | [shuffle Pool/resource/waves/merge][E-KG-SHUFFLE] → [resource_plan][E-EZ-RESOURCE]、[Controller.snapshot][E-EZ-RUNTIME] | 复用spawn池、紧凑输入与逐片merge，采样前资源阈值和磁盘预检；独立scratch清理 | 原始片估算含完整末文件超额，nominal桶保持输出整倍数；实际超预算报错。预算不含解释器/allocator/OS，不是硬RSS或磁盘保证 | 来源采样边界与资源适配明确；实际峰值/吞吐未验 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E18 | [generator][E-KG-GENERATOR]、train.py get_files_for_subepoch、[read_npz][E-KG-READ] → [reader][E-EZ-READER]、[train][E-EZ-TRAIN] | repeat带约1/3间隔reservoir；no-repeat明确耗尽；单轮固定snapshot；depth+1预取＋batch queue | 264来源文件顺序及RNG状态一致；本地另保存文件内消费游标，不以后台预读游标checkpoint。no-repeat不足固定轮预算提前拒绝；无异步目录插入 | 两种模式、尾部/耗尽恢复、预取消费cursor及实际CUDA通过 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |

<a id="engineering-concurrency"></a>

### 并发、存储与数值执行

| ID | 来源 → 目标入口 | 实际配置与调用链 | 差异及影响 | 对齐结论 / 证据级别 | 验证范围／证据 |
|---|---|---|---|---|---|
| E02 | [并发规模][E-KG-CONCURRENCY]、setup.cpp initializeNNEvaluator → [launch][E-EZ-LAUNCH]、[selfplay/evaluator][E-EZ-MAIN] | baseline 每设备项1 worker、32 game threads、每 Search 1 thread、共享1 server，max_batch32；shuffle12进程和 learner CPU2，阶段串行 | game/search/server 是不同层次，潜在并发 eval=game_threads×search_threads；同设备重复项可创建多进程。来源也按这几层分配，规模不同不是 bug；当前配置本身不能证明未过度订阅/利用率相同 | 层次机制一致；机器规模属于资源条件；性能证据不足 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E03 | [gameLoop][E-KG-SP]、manager acquire/release → [selfplay][E-EZ-MAIN]、[worker][E-EZ-WORKER]、[NativeWorker][E-EZ-NATIVE] | worker 进程跨请求存活，game threads 每次请求重新创建，线程内 Search/树内线程跨局复用；每局 seed=mix(worker_seed+id)，feature RNG 独立；产样结束 drain/release | 不应把常驻进程说成 game threads 跨轮常驻。来源长驻 game loop 每局 acquire 最新模型，V0 本轮锁模型，随机服务每请求重建。释放无在途请求的服务后再进入 learner，有避免同阶段权重竞争的路径 | 复用/RNG/释放机制存在；模型时点差异（E23）；三轮同PID/固定输入/阶段前释放已验 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E04 | [playoutDescend][E-KG-SEARCH]、searchnode.cpp、searchupdatehelpers.cpp → [expand/simulation][E-EZ-SEARCH]、Search 节点/线程池 | child tier 原子发布、mutex pool 协调展开、stats_mutex 快照，edge/node pending 加减；catch 释放路径预约，返回检查pending=0；join后推进/清树并复用 workers 回收大树 | 来源竞争展开可重启未计数 playout，V0 EXPANDING 用条件变量等待；终局来源 waitForNextNNEvalIfAny 节流，V0 立即处理；原子分字段与锁快照也不同，可能改变并发访问/统计时点。图共享节点、私有根复制及静止后标记回收已实现（S25） | 图共享及多父生命周期已验收；并发调度/统计时点不同，不宣称来源并行序列或性能等价 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E05 | [serve][E-KG-NN]、[forcePush][E-KG-QUEUE] → [BatchEvaluator][E-EZ-BATCH]、[serve][E-EZ-SERVE] | 名义queue_capacity256，wait_us0立即取最多32；入队forcePush不做容量背压；同步caller拥有观测直到完成，TLS request；closing排空，failure唤醒队内/current batch | 非零wait_us增加组批时延，属于显式调度扩展；source和V0均不因容量阻塞入队。异常传播、backend能力检查保持有效 | forcePush与默认立即取批对应；容量1可形成更大批次的回归见core_test；不宣称所有故障/调度/性能等价 | [实现](EtaZero_V0/cpp/src/inference/batcher.cpp)、[回归](EtaZero_V0/cpp/tests/core_test.cpp) |
| E06 | [nneval server][E-KG-NN]、[CUDA handle][E-KG-CUDA] → [TorchBackend][E-EZ-TORCH] | 每server独立device guard/stream/pinned host缓冲；同一worker共享 LoadedModel 只读权重；输出同步回CPU后复用host；baseline SP FP16，Eval/Match auto CUDA→FP16，FP32显式覆盖 | 所有权层次对应；来源专用CUDA handle/buffers与 V0 LibTorch中间tensor/ScriptModule不同。来源CUDA模型/工作区分配与V0共享Module不能视为同显存/性能。多服务 JIT/FP16/auto 与失败路径按批次2验收；显存/吞吐留E26 | 设备/缓冲生命周期子机制对应；backend 保留LibTorch差异，资源/性能未等价 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E07 | [NN hash/cache][E-KG-CACHE]、nneval evaluateWithSymmetries → [cache][E-EZ-BATCH]、Search.evaluate_node | V0 精确packed平面+global字节比较，直接映射/stripe锁/immutable output；evaluator绑定model/canvas/precision；key为canonical输入+globals+Tnn，朝向排除 | 碰撞只换槽不误返回，旧模型service独立；单次随机推理仅miss抽朝向，hit复用canonical输出；集成读写绕cache。相同并发miss未合并，两者都不可据此认定为graph | 碰撞/模型隔离、朝向/cache/集成子机制一致（独立样例）；PDA符号/global条件隔离与模型ID或实际路径重载已验证；Tnn/optimism精确double key较来源2048/1024量化更细，缓存命中/RNG不同 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E08 | [manager dataWriteLoop][E-KG-MANAGER]、[training writer][E-KG-WRITE] → [RecordWriter][E-EZ-WRITER] | queue32完整局move交接；按最终重复训练行数分片，首文件随机阈值minProp0.15，后续满片为配置shard_rows；无定时flush，finish排空join并发布尾片 | 跨片可携带同一局完整轨迹，metadata row_begin/rows只选择本片最终训练行；TD/opponent完整上下文、逐行Q/dropout不因边界改变；catalog按区间去重。存储组织及本地逐轮writer生命周期仍不同 | 来源行阈值/首文件公式与拆局语义对应；跨片目标和唯一统计回归见test_writer；非来源格式或性能等价 | [实现](EtaZero_V0/cpp/src/selfplay/record.cpp)、[读取](EtaZero_V0/python/etazero/data.py)、[回归](EtaZero_V0/tests/test_writer.py) |
| E09 | [packBits][E-KG-PACK]、trainingwrite fillPolicyTarget/writeGame、[read_npz][E-KG-READ] → [append][E-EZ-PACK]、[training_view][E-EZ-VIEW]、schema.py | packed MSB-first/zero tail；保留T+1轨迹，policy/search数组只存positive repeats位置；view展开multiplicity；policy int16，visits int64，value终局/side WDL及三个TD | 打包位序对应，存储组织可不同；来源最终训练rows含逐行随机增强、int16 policy量化及辅助监督；V0逐输出行dropout及重复行独立已验证（D02），int16顺序及v15辅助targets已验收，side独立搜索标签/gate有效；v17纯W−L Q已贯通，node visits按位置压缩、Q值逐重复输出行随机量化（L01/L02）。rows/plies/sampled_positions是不同统计量 | 打包及逐行禁手增强一致；v15量化和辅助目标经过独立/真实CUDA验证；v17 Q和7的分支产生已验收；固定量消费、每文件尾部、重复/耗尽和整轮恢复已验收 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E10 | [writer rename][E-KG-WRITE]、[shuffle.sh][E-KG-SH]、[export][E-KG-EXPORT] → [atomic_write][E-EZ-STORAGE]、writer.publish、[snapshot][E-EZ-SNAPSHOT]、export_model | raw/shuffle/model私有tmp→完整payload/manifest→rename；Python immutable用link防覆盖，checkpoint与current另有可变pointer；文件和目录fsync；run.lock单controller | 来源也用tmp/rename与完整train.json后发布，不同入口有NFS等待；V0更强本地持久化步骤，不证明跨FS/NFS等价。C++raw先exists后rename防覆盖依赖独占attempt目录，非通用多writer原子排他；双方不能读取私有tmp当已发布产物 | 完整发布机制对应；本地文件系统的所测中断/竞争通过，跨FS/NFS及多writer假设不同 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E19 | [read_npz][E-KG-READ]、trainloop_helpers → [BatchPrefetcher][E-EZ-READER]、[CUDA prefetch][E-EZ-PREFETCH] | CPU有界queue准备batch/cursor；CUDA双pinned slot，复用前event.synchronize，计算wait_event，device tensors record_stream，close同步上传stream | 来源该reader预取完整展开文件，batch .to(device)普通传输；无同构双slot。V0更早预读可以丢弃，consumed_state随消费batch保存；实际CUDA上传/预取消费游标及连续/恢复已对照；不覆盖所有DMA或硬件故障组合 | 实现不同；所测真实CUDA生命周期/恢复通过；数据消费另见E16/E18 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E20 | [compile wrapper][E-KG-COMPILE]、[FP32 heads][E-KG-HEAD]、train.py scaler → [TrainingForward][E-EZ-NET]、[train_iteration][E-EZ-TRAIN]、optimizer_for | baseline compile=true/amp=off；网络+loss单fullgraph/static shape，CUDA AdamW fused；训练主干autocast，heads/loss实际FP32；scaler skip消费下一batch | 来源分别compile网络/metrics；训练FP32 heads、skip后消费/Lookahead/SWA时钟已落实。编译图组织及推理精度仍独立，不推断跨实现逐位或性能等价 | 基础AMP/计数已验证：真实CUDA SGD/AdamW、FP16/BF16、eager/compile、overflow及恢复；另有实际NBT编译短链路证据 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E22 | [shuffle.sh][E-KG-SH]、[epoch validation][E-KG-VAL] → shuffle.py/training.py | baseline默认启用MD5 basename1%holdout；skip对应sync SKIP_VALIDATE，切分前q共用；raw epoch validation随机D4 | 分区跨快照稳定；随机hex原文件名独立于game RNG；validation文件尾丢弃，cap在超过后停；独立D4流不改训练RNG，Go专有metric省略 | 文件隔离/重建及raw eval、AMP、compiled eval GPU通过；不声称棋力评估 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E25 | [搜索停止][E-KG-STOP]、selfplay signal/gameLoop、manager destructor、nneval killServerThreads → main.stop_handler、[Search.run][E-EZ-SEARCHRUN]、NativeWorker.close、Controller.close、reader.close/process.py | 信号阻止新局/取消半局，已完成局writer排空；Search在每次发放新playout前检查外部stop callback；已发放模拟与在途NN完成后收尾；worker close等10秒→SIGINT等10秒→kill；learner batch边界保存后整轮重跑 | 来源search检查shouldStopNow/time，V0在新playout发放边界检查stop/time；同步根NN及在途模拟不会强制取消；强杀会剩私有tmp/丢未持久化完成局。异常传播/join/finally有实现，SIGKILL不执行finally，不能泛称所有后台任务都安全收尾；source线程异常保护与V0错误返回也不同 | SIGINT/TERM正常收尾及训练SIGKILL/阶段异常恢复已验；取消粒度差异；未测故障分支保留静态限定 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |
| E26 | source selfplay runtime/NN counters、[shuffle TimeStuff][E-KG-TIMER]、train.py timer → [Journal/phase][E-EZ-RUNTIME]、[run_history][E-EZ-PLOT]、performance_figure | committed elapsed包含本轮SP/shuffle/train/export/plot，排除初始化/恢复/作废；phase_end按完成尝试计；NN记录requests/batches/max/wait/cache/server rows | 平均batch与queue_wait总量有定义（从入队前到server取出，无后端执行时间），但未记录完整batch分布/峰值RSS/VRAM/编译与暖机拆分；中断无phase_end不进分母。各来源timer覆盖范围不同，不能据现图或源码断言端到端吞吐相同；本批未测硬件/性能 | 统计代码已审；性能证据不足；监控覆盖差异 | 来源/配置/调用已复审；逐项运行与静态范围见[第10批](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json) |

### 工程结论的适用范围

共享 evaluator、多server、后台 writer、两阶段 shuffle、waves、常驻worker/进程池和CPU/CUDA预取均有实际入口。逐项实现、参数、边界、调用和执行证据记录在 [第10批核查](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json)；所测机制成立不代表全部硬件故障或部署方式等价。

窗口使用capped usable rows；来源train bucket使用未封顶range end（另含offset），epoch预扣和batch实际消费分开。本地按用户约定采用固定轮产样quota，无训练桶。组边界采用累计训练rows≥阈值，round(nq)仅降采样一次；多wave、固定文件数和每文件完整batch前缀分别核验。MD5留出在快照间稳定隔离，validation启用时始终随机D4，baseline默认启用验证，skip对应来源同步脚本。

异步阶段重叠、局内换网、DDP、跨机器和专用native backend保留为能力/执行差异，不并入400/70或S23的用户例外。没有相同负载的来源端到端测量，不宣称吞吐、峰值资源或等时间性能一致。

<a id="alignment-targets"></a>

## 8. 实施目标与五子棋映射

本节维护用户确定的**对齐范围与适配约定**；当前行为和验证结论分别见第1—7节及核查证据，配置以事实源为准。第1—6节同ID的来源位置、目标入口和现状差异继续有效；本节补充最终profile、适用分支、环境定义和验收去向。所有KG引用绑定commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`，GM引用绑定commit `df152116e3787c75c6a3de099d261ca092b7dfc1`，不跟随外部仓库后续HEAD。

2026-10-02用户确定：保留EtaZero的固定训练量、产样规划和同步逐轮机制，不引入KataGo训练桶；环境采用现有env的方形尺寸/分布及可配置三规则混训，VCN变体先不用；采用下述三个网络预设。原有400/70和S23暂缓仍有效。这些适配单独验收，不改称来源一致；不因此豁免其他算法、参数或表内分支。

<a id="alignment-profiles"></a>

### 当前baseline profile

SP是主局数据生产；E/M是固定局面eval及Match的搜索profile。Match还经过开局、执色和tree reuse调用，eval对给定局面直接搜索。下表按当前[baseline配置](EtaZero_V0/configs/baseline/)记录实际参数。SP机制参照[mainb18][KG-SP]+SETUP_FOR_OTHER，E/M参照[match_example][KG-MATCH]+SETUP_FOR_MATCH；预算、reduced最低访问数、SP半衰参数、side概率及replay窗口的当前差异单列。来源注释里的示例值不覆盖[加载器][KG-SETUP]。五子棋开局单独取GM，不能用Go面积/komi公式替代。

| 参数组 | SP当前值 | E/M当前值 | 依据／适配 |
|---|---|---|---|
| 搜索上限 | full400、cheap70，概率0.75，cheap基础权重0 | maxVisits500；默认maxPlayouts=2147483647/maxTime=1e20，独立限制可用 | 400/70为用户预算例外；100v评估不作为当前默认，见S02/A02 |
| Reduce visits | 开启；阈值0.9、lookback3、**min50**、weight0.1 | 关闭产样缩减 | min独立于cheap70；当前full400可降至50，来源最低值350；不是从cheap预算自动推导 |
| PUCT / log | c0=1.05、log=0.28、base500 | c0=1.0、log=0.45、base500 | S12/S13 |
| Utility方差缩放 | prior0.40、weight2、scale0 | prior0.40、weight2、scale0.85 | S14；纯W−L utility的波动，保留原参数，不按棋盘面积重缩 |
| FPU | 非根0.2、普通根0、cheap零权重根0.2；visited-policy=true、power2、loss_prop0 | 非根0.2、根0.1；其余相同 | 固定fpuParentWeight分支也实现，关闭visited-policy时默认0，S01 |
| Value weighting / 并发 | exponent0.5、virtual loss1 | exponent0.25、virtual loss1 | S15/S24；Go anti-mirror不适用，图搜索/并发不得省略 |
| NN / 根 / 落子温度 | NN1；根1.5→1.1；落子0.75→0.15 | NN1；根1→1；落子0.60→0.20 | half-life参数SP15（来源19）、E/M19；onlyBelowProb默认1；按实际面积缩放；cheap零权重根温度1，S18—S20 |
| 根噪声 / forced / chosen | shaped，总浓度10.83、混合0.25；forced2；prune1/subtract0 | 无根噪声、forced0；prune1/subtract0 | cheap零权重关闭根噪声/forced；不按15²/19²缩浓度，S03/S21/S22 |
| LCB / target pruning | pruning开启；LCB z5、minProp0.15、修正版；实际落子禁LCB、监督启LCB | pruning及落子LCB开启，同z/minProp | S04/S05；关闭开关是可配置分支，不替代来源默认 |
| NN朝向 / 根集成 | randomize=true；普通根4、cheap根1；叶随机D4 | randomize=true；根1，仍随机朝向 | 单朝向cache复用首次输出；K>1集成跳cache；S16/S17/A03 |
| 搜索修正 | graph=true；uncertainty=false；root/leaf optimism0/0；noise pruning=false | graph=true；uncertainty=true，c0.25/p1/max8；root/leaf optimism0.2/1；noise pruning=true，utility scale0.15/cap1e50 | S25—S28；S23单独暂缓，不把它归为默认关闭 |
| Tree reuse / 模型 | 普通full清树，cheap零权重保留；PDA强制清树；每轮固定发布模型 | Match不同bot默认保留，同botIdx双方清树；固定eval遵循显式reuse设置 | S06/A03；轮间换代对应保留的同步机制，局内轮询切换不进入本次目标 |
| PDA / side / surprise | 普通局概率0.01，最大比8；side0.040（来源0.020），初始频率1；policy/value surprise0.5/0.1，direct-value=false、reanalyze=false | 默认PDA0；不生成side或surprise训练行 | S08—S11；非默认开关仍实现，Go handicap/komi不移植 |
| 禁手提示 | Renju最终输出行dropout概率0.5，重复行独立 | 完整提示 | D02；其他规则无禁手提示 |
| 平衡开局 | 概率0.99，NOVC骨架，dist0.8、balance指数4、reject0.995，失败>20降到0.8继续重试 | Match概率1、指数10；固定eval不初始化开局 | D01；Match每次balance尝试随机选botB/W；保留EtaZero成对换色结果口径 |
| Policy init | 开启，mean6、T1.6；平衡后及未选平衡时均按开关执行 | Match默认关闭；开启时显式mean6/T1.6，按当前执色调用botB/W；eval不初始化 | SP参数单独采用[GM scripts预设][T-GM-PRESET]；GM Match loader默认false。GM另一个15尺寸示例的policyInitProp=10不被当前loader读取，不能当mean10 |
| 网络 / learner精度 | SP推理FP16；learner默认FP32，FP16/BF16可选且heads/loss按来源保持FP32 | CUDA auto精度解析并记录实际精度，支持FP32显式override | N11/E06/E20；LibTorch保留，不承诺native backend等吞吐；独立数值检查使用明确FP32 |
| Learner公式 | SGD momentum0.9，LRscale1/head0.5/noreg1；warmup开启；Lookahead k6/alpha0.5；SWA scale8 | 加载发布权重，无在线learner | N07—N11；范数默认每100batch快照，打印点衰减的运行均值可选；LR/WD每5batch、累计样本>2亿后每50batch刷新；裁剪SGD fson/fixup2500、普通BN5500、AdamW11000 |
| Loss | ordinary0.930，opponent0.15，soft8、soft-opponent1.2；value及三个TD各有效0.72；long/short optimistic系数0.100/0.200 | 导出主/optimistic policy、WDL、误差等搜索输出 | 对应v15/v17六policy默认分支；误差与有效性见下文，不能把0.6当最终CE系数 |
| Replay | min_rows150000、p0.8、a0.3、K20000000；s默认m、offset0、max_rows不设；支持all与三扩展 | 不适用 | R01—R03采用[GM shuffle脚本][GM-SHUFFLESH]的m/p/a；K为当前配置值。KataGo CLI及脚本参数仍在来源列单列 |
| 保留的训练编排 | 每轮1000个消费batch、batch128、ratio8；固定更新量规划新增有效行，整局可超额；SWA period0解析为64000消费样本；轮末发布 | 默认加载已发布模型 | R05/N07/A04：保留逐轮规划及整轮提交恢复，**无train bucket、raw/usable补桶或epoch预扣**。AMP跳过时消费时钟照常推进，成功更新另计数，不重试同batch；来源epoch时钟映射到本地固定轮，sub_epochs默认1/可选floor分段，非照抄来源epoch文件预算 |
| Validation / reader | 默认启用validation（skip_validation=false），按文件MD5留出1%，验证随机D4；可显式跳过；repeat允许，no-repeat可选 | 不适用 | 源码支持sync SKIP_VALIDATE；删除原计划“validation关闭D4”适配。reader每文件丢不足global batch尾部，repeat顺序及no-repeat状态按来源核查；固定训练量不等于可无限静默重用耗尽的no-repeat快照 |
| 本机资源 | 32局/1搜索线程、1NN服务/batch32；shuffle12进程/16384MiB，group/bucket/训练名义分片均65536、waves1；预取深度2、CUDA预取开启；queue/writer字段不豁免调度/样本语义；batch/显存等显式记录 | Match8局/1搜索线程 | 资源规模不是算法数值默认；E05默认立即组批并按来源forcePush无容量背压；E08按最终训练行分片并保留完整目标上下文；E13/E15抽样和seed差异按各项范围说明；不新增DDP、跨机器或native backend |

<a id="alignment-environment"></a>

### 环境与辅助监督契约

**棋盘与规则。** 使用当前env的canvas15，方形15/14/13/12/11，权重100/10/5/3/1；baseline规则权重Renju/Freestyle/Standard=1/0/0，配置可选择任意有正权重的混合，不新增实验矩阵。Eval/Match默认15×15 Renju，命令指定尺寸/规则时使用实际条件。用户选择保持方棋盘，故本次不移植矩形抽样；KataGo矩形概率0.10登记为环境范围差异。VCN暂不纳入，不宣称复现GM全部规则；不加入VCN的pass、firstPassWin和move-limit训练游戏。棋规仍需对照真实禁手/长连/五连边界，已有三个枚举不能证明规则等价。[GM规则][T-GM-RULES]、[15尺寸示例][T-GM-ENV]。

**WDL及视角。** 存储/网络保持当前玩家W/D/L顺序；固定黑方视角轨迹先交换W/L，再在每个训练位置还原行棋方视角，D永不交换。真实终局胜/和/负分别为(1,0,0)/(0,1,0)/(0,0,1)，utility=W−L、draw utility0、半胜信用W+0.5D。Go的W/L/no-result顺序及“draw拆入W/L”不直接搬入；不沿用Go禁用no-result logit的规则判断来清零五子棋draw。禁手失利是实际负局；取消/截断对局不能伪造draw终局。无komi、Go score、ownership、seki、territory、button或pass动作；去除其utility项后FPU/LCB radius=1、score导数为0，variance PUCT仍估计完整W−L统计。

**TD。** 令z_i为固定黑方视角的搜索WDL，z_T为真实终局。对位置t、A=实际棋盘面积，a∈{1/(1+0.176A),1/(1+0.056A),1/(1+0.016A)}，目标为 `a·Σ_{j=t}^{T−1}(1−a)^(j−t)z_j + (1−a)^(T−t)z_T`；a=0得到终局目标。每一项在累加前转为位置t玩家视角；不要逐项用未来落子方视角累加。三个TD输出shape `[N,3,3]`、终局WDL `[N,3]`，概率float32；每个TD的loss为 `1.20×0.6×(CE−target_entropy)`，主value为 `1.20×0.6×CE`。保留来源对有效性权重的区分，不把随机取整频率再次乘入loss。[TD writer][T-KG-TD]、[loss][KG-METRICS]。

**Side和reanalysis。** Side只有一次搜索WDL，来源writer用长度1的值列表、value/TD权重均1，因此主value和三个TD均监督同一搜索概率；没有该分支的真实终局，不借用主局终局。Side的opponent目标无效，默认长期/短期optimistic均因“完整终局数据”条件关闭；short-term error沿来源完整数据gate关闭（error loss使用ownership有效性，而非TD有效性）；不能因TD权重1就给side开启误差监督。主局完整终局标志替代Go ownership有效性标志用于optimistic gate，不伪造ownership。重分析保留原局轨迹/终局，替换所选cheap位置的policy/search/raw-NN等统计；重分析行不使用outcome targets时，opponent目标无效，`full_game_weight=0`关闭error及默认long/short optimistic监督，主value、三个TD及Q仍有效；使用outcome targets时主局完整数据gate保持1。gate按完整轨迹索引映射到每个重复输出行。来源`disable_optimistic_policy=true`分支仍以固定0.5权重训练两个head，不受该gate控制。各项目标沿[重分析][T-KG-REANALYZE]和[side writer][T-KG-SIDEWRITE]落实，不能让“训练数据增强”改变实际对局。

**误差与uncertainty。** 预测的是方差 `e=0.25·softplus(raw/2)^2`（v15/v17），训练目标为 `(stopgrad(TD3预测W−L)−TD3目标W−L)^2+1e−8`，loss为 `2·Huber(e,target,δ=0.4)`，乘来源有效性/global权重；训练softplus反向梯度floor0.05按来源实现。完整主局行且允许outcome targets时误差权重有效，side/不使用outcome targets的重分析行/取消或缺真实终局时误差权重0。推理输出标准差sqrt(e)，D4集成按来源输出量平均，不把方差与标准差混用。五子棋纯W−L下 `u=sqrt(e)`，score项因utility导数0移除，`w=0.25/(u^1+0.25/8)`；缺少error能力的模型才走来源weight1分支，支持该能力时不得用常数替代。[预测缩放][T-KG-ERRORHEAD]、[误差loss][T-KG-ERRORLOSS]、[NN后处理][T-KG-ERRORPOST]、[权重][KG-UNCERTAINTY]。

**Optimistic与Q。** 六policy顺序为普通、opponent、soft、soft-opponent、long-optimistic、short-optimistic。移除Go score项后，long权重为 `(W_target+0.5D_target)^2`，short权重为 `sigmoid(3·((TD3目标W−L−stopgrad(TD3预测W−L))/sqrt(stopgrad(e)+0.0001)−1.5))`，均乘policy有效性和完整主局终局标志；这是明确的无score环境适配，不与Go全loss数值等同。关闭optimistic训练的来源开关仍以0.5权重训练这两个head，不把head删掉。推理照来源将ordinary与**short-optimistic** logits混合，再做温度/softmax，不能拿long head或概率平均代替。v15没有Q；v17 `predict_q_values`缺省false，true分支属于表内Q能力，保留逐动作W−L、visits权重及loss（score Q去除），目标随D4变换。只在支持版本启用，不把v16的强制Q套到v15。[版本及权重][T-KG-OPTLOSS]、[Q loss][T-KG-QLOSS]、[源提取][KG-Q-EXTRACT]、[源node统计][KG-Q-NODE]、[源writer][KG-Q-WRITER]、[目标][EZ-Q-TARGETS]、[量化][EZ-Q-QUANTIZE]、[Q实现][EZ-Q-LOSS]。

**Policy量化。** 来源先按最终选择权重将最大值至少缩放到10，再在最大值>30000时统一乘30000/max，以C++ round写int16；learner读取后除行和归一化，再构造soft目标。不能先存概率再直接乘30000，也不能改成随机量化。不可用opponent采用来源非零占位且有效性权重0，合法监督不能全零。[量化生产][T-KG-POLICYQUANT]。

**PDA。** 普通局以0.01抽样，优势方均匀黑/白，d均匀[0,log2(8))；令r=2^d，优势方预算系数2r/(1+r)，另一方2/(1+r)，先完成PCR/reduced预算再乘系数并round；来源<5的非法预算报错，不静默夹限。输入在现有4全局量后增加enabled和signed half-d两个float32全局量，shape从[N,4]变为[N,6]，当前玩家是优势方时+0.5d、另一方−0.5d，d=0时两者0；写入、reader、export、cache均覆盖，任何模型版本都不能忽略条件。PDA每手清树，side输入及预算恢复d=0。无Go handicap/komi，不移植0.5让子局概率或komi补偿；平衡开局与PDA各自沿流程执行，不能自行新增“PDA后再平衡”研究方案。[抽样][T-KG-PDASAMPLE]、[输入][T-KG-PDAINPUT]、[预算][KG-PDA]。

<a id="alignment-networks"></a>

### 网络预设与非默认分支

| 架构 | 固定来源预设／版本 | 结构与head宽度 | 初始化／归一化与实施边界 |
|---|---|---|---|
| Plain | b10c128-fson-mish，v15 | 10块C128，mid128，gpool32；第5/8块regulargpool；p1/g1/v1=32，v2=80 | fson/Mish，最终masked BN，沿预设初始化；批次8a，不能复用NBT外壳冒充plain |
| NBT | b5c192nbt-fson-mish，v15 | 5块C192/mid96/gpool32，NBT2，第2/4块gpool；p1/g1/v1=32，v2=80 | fson/Mish；完整六policy/WDL/TD/error训练与原生导出 |
| Transformer+NBT | b5c192h3nbttfrs，v17 | 5块C192/mid96，NBT2 transformer，3 heads/3 KV heads，FFN256，RoPE/SwiGLU；p1/g1/v1=32，v2=64 | **bare预设norm_kind=fixup、缺省激活ReLU**，不是-fson-mish；RMSNorm/attention初始化、参数组和外块按该版本，批次8b。不能强加卷积预设v2=80或最终fson BN |

来源[plain][T-KG-PLAIN]、[NBT][KG-B5]、[Transformer preset][T-KG-TFPRESET]与[生成后缀][KG-SUFFIX]已实际加载核对；预设是实现锚点，不要求追加三臂训练。三个版本均用上面的五子棋输入和辅助目标映射；架构不能沿旧head子集宣称整个来源网络一致。仅实现所选预设必要的组件，不把所有modelconfigs名称自动加进范围。

| 分支 | 默认及范围 | 去向／不适用依据 |
|---|---|---|
| FPU visited-policy关闭、onlyBelowProb<1、maxPlayouts/maxTime、graph关闭 | 默认关闭/未设也实现真实路径 | 2/5；A02预算计数和停止边界，不能靠配置拒绝替代已列入能力；fixed eval每次独立局面新建Search，无跨调用tree reuse |
| hint / hintFork / early或game fork | 无外部hint数据时概率0；读取语料校验生效配置中的SHA256；PCR精确hint乘4、后6手cheap概率减半；full/reduced优先级照来源；mainb18 early0.04/late0.01、early位置比例0.025、候选3…12/36，合法空点有放回抽取 | 五子棋开局状态、位置hash、落子与提示输入贯通；候选数量/有放回最优着概率及hint字节变更拒绝由sampling_test核验；Go SGF解析/komi/seki fork不自动移植，禁手终局不能当普通fork起点 |
| reanalysis / direct value surprise | 默认useReanalyze=false、direct=false；开关打开后实现完整选择、force-full、历史截断、target替换、重算和cheap excess gate；所有开启必需参数显式提供，不猜实验比例 | 7c；不删除原轨迹或真实终局，独立验收默认与非默认路径 |
| side递归 / tree rows | 按S09来源实际side生产调用覆盖递归与训练有效性；零概率分支同样可验 | 7b/4；有搜索监督，不能伪造终局；表外独立recordTree产样配置不因同用SidePosition结构自动追加 |
| norm running mean / AdamW / AMP / Q / optimistic-disable | 保留来源开关与对应版本，分别验收；Transformer fixup参数组单独核查 | 1/4/8；fson自适应WD不能套到fixup；AMP失败不重试同batch |
| replay s/max_rows/offset、all、random封顶、repeat/no-repeat | 实现三窗口扩展；random行在usable窗口累计中封顶；与固定产样quota累计分开记，**不增加训练桶** | 9a；每文件尾部/组边界/round/wave种子可影响最终样本 |
| validation启用、无gatekeeper直接发布 | 默认validation启用、gatekeeper不用；按真实文件隔离、随机D4，可显式跳过validation | 9a/9c；来源允许skip和direct export，不自动追加竞技筛选研究方案 |
| Go pass/komi/score/ownership/lead/seki/handicap/ko/anti-mirror | 环境不含其状态/奖励或目标，明确不适用 | D05/A01及上述公式逐项移除；不把缺少draw/TD/error/PDA称为Go专有 |
| VCN／矩形／局内换模型／训练桶 | 用户确定：VCN暂不使用，方棋盘保持；保留逐轮固定模型与训练规划 | 是范围或编排差异，未移植，不能计为来源一致。S23仍是独立的“暂缓，未实现、未对齐” |
| future-position/variance-time/metadata、Go额外heads、其他架构/优化器、DDP/native backend | 没有在57项中具名列为实施能力的不自动追加；future-position/variance-time本可定义五子棋监督，不能笼统标成Go专有 | 保持表内具名能力范围；不宣称整个KataGo训练model.forward或仓库功能完整复现 |

<a id="alignment-items"></a>

### 57项实施与验收登记

每行继承第1—6节同ID的commit绑定来源、代码链接和静态现状；profile值唯一维护于上文，避免57行各抄一套参数。KG-SP/KG-M分别指SP/E/M配置和加载默认，L15/L17指已选模型版本对应的learner分支，GM指上述固定KataGomo来源。下列是最终验收要求；各批验收证据在第3/4节同ID及核查证据中注明；第10批对全部57行另行复审实现、参数、边界、调用和执行证据。

| ID | 来源／版本 | 目标入口 | 环境／范围决策 | 后续验收 | 批次 |
|---|---|---|---|---|---|
| `S01` | KG-SP/KG-M [FPU][KG-FPU] | [选点][EZ-SIMULATE] | 纯W−L radius1；visited与固定parent分支 | 手算质量0/1、parent/NN插值、cheap根与固定权重 | 2 |
| `S02` | KG-SP [limits][KG-CHEAP] | [预算][EZ-LIMITS] | 400/70例外；hint/fork与PCR优先级仍对齐 | 固定RNG覆盖cheap/full、精确hint×4、后6手概率和reanalysis强制full | 2、7c |
| `S03` | KG-SP/KG-M [forced][KG-FORCED] | [分数][EZ-SEARCH-MATH] | SP2、cheap及E/M0；含virtual weight | 配额边界及forced关闭仍执行目标剪枝 | 2 |
| `S04` | KG-SP/KG-M [选择][KG-SELECTION] | [pruning][EZ-SELECTION] | 移除Go pass/ending；默认开启 | 独立小树稳定边、非正分母、ceil及zero-forced | 2 |
| `S05` | KG-SP/KG-M [LCB][KG-LCB] | [选择与调用][EZ-RESULT] | radius1；落子/监督分流 | 手算ESS及提升量、index0、门槛、SP禁用与E/M启用 | 2 |
| `S06` | KG-SP/KG-M [runner][KG-RUNNER] | [推进][EZ-ADVANCE]、[Match][EZ-MATCH-CALL] | 保留逐轮固定模型；PDA/full清树、cheap及异bot复用 | 已有visits/new playouts、同异bot、根先验刷新和模型隔离 | 2、5、7a |
| `S07` | KG-SP [reduced][KG-CHEAP] | [limits][EZ-LIMITS] | 当前min50独立于cheap70（来源350）；统一黑方历史 | 历史阈值/长度/舍入/权重及PCR互斥手算 | 2 |
| `S08` | KG-SP [PDA][KG-PDA] | [limits][EZ-LIMITS]、[输入][EZ-GAME]、[契约][T-EZ-SCHEMA] | 增enabled和signed半d，无Go komi/handicap | r=1/2/8预算、双视角/side0、非法<5及cache分离 | 4、7a |
| `S09` | KG-SP [side][KG-SIDE] | [SP][EZ-SP-CALL]、[writer][EZ-WRITER]、[view][EZ-VIEW] | NOVC无pass；side用搜索WDL，无终局标签 | 排除实际着、替代抽样、递归、side主value/TD同搜索概率及gate | 4、7b |
| `S10` | KG-SP [频率][KG-WEIGHTS] | [权重][EZ-WEIGHTS] | 默认alpha0.5；重分析cheap excess例外纳入 | 全零/基础S<1/cheap恢复/重分析前后独立频率 | 7c |
| `S11` | KG-SP [value KL][KG-VSURPRISE] | [权重][EZ-WEIGHTS] | WDL draw映射；future及direct两分支 | 终局倒序平滑、direct只看本手、clip与低均值衰减 | 4、7c |
| `S12` | KG-SP/KG-M [PUCT][KG-FPU] | [分数][EZ-SEARCH-MATH] | 纯W−L；c0按profile | 手算完成权重、探索0.01、virtual分子排除 | 2 |
| `S13` | KG-SP/KG-M [加载][KG-SETUP] | [分数][EZ-SEARCH-MATH] | SP0.28/E-M0.45，base500 | 多个T的独立log公式、参数实际加载 | 2 |
| `S14` | KG-SP/KG-M [方差][KG-FPU] | [分数][EZ-SEARCH-MATH] | prior0.4/weight2，SP0/E-M0.85 | weight≤1与一般二阶统计独立对照 | 2、6 |
| `S15` | KG-SP/KG-M [backup][KG-VWEIGHT] | [聚合][EZ-AGGREGATE] | SP0.5/E-M0.25，anti-mirror无对应状态 | t3插值、保总权重、二阶矩平方缩放、noise竞争分支 | 2、5、6 |
| `S16` | KG-SP/KG-M [集成][KG-SYM] | [NN][EZ-EVALUATE-NODE] | 方形D4、概率平均、K>1跳cache | 固定各朝向输出独立坐标还原，K1随机与K4无放回 | 2、4 |
| `S17` | KG-SP/KG-M [随机朝向][KG-RANDOMSYM] | [展开][EZ-EXPAND] | 叶/cheap/E-M开启；指定朝向可固定 | 固定RNG及缓存命中后朝向不重抽、policy还原 | 2 |
| `S18` | KG-SP/KG-M [NN温度][KG-NNCACHE] | [NN][EZ-EVALUATE-NODE] | 默认三profile均1；raw cache可保留 | 温度override、合法域、不同cache表示最终概率独立对照 | 2 |
| `S19` | KG-SP/KG-M [根温度][KG-NOISE] | [根处理][EZ-RUN] | SP1.5→1.1、half15（来源19），cheap1 | 实际面积/手数衰减、集成后噪声前、reused root | 2 |
| `S20` | KG-SP/KG-M [落子][KG-CHOSEN] | [结果][EZ-RESULT] | SP0.75→0.15/half15；E/M0.60→0.20/half19；onlyBelowProb非默认分支 | 手算温度边界、概率保护、policy target不受落子温度影响 | 2 |
| `S21` | KG-SP/KG-M [选择][KG-SELECTION] | [选择/backup][EZ-SELECTION] | prune1/subtract0，保留两类调用 | cutoff/subtract边界、LCB后执行、noise pruning竞争 | 2、6 |
| `S22` | KG-SP [Dirichlet][KG-NOISE] | [噪声][EZ-NOISE-CODE] | 总浓度10.83；不按五子棋面积缩放 | alpha总和、均匀退化、合法空点、cheap/E-M关闭 | 2 |
| `S23` | KG-SP/KG-M [subtree bias][KG-AGGREGATE] | [当前聚合][EZ-AGGREGATE] | 用户暂缓，未实现、未对齐，不计通过 | 最终核查缺口及影响仍存在，不以禁用冒充等价 | 暂缓、10 |
| `S24` | KG-SP/KG-M [virtual][KG-FPU] | [并发][EZ-SIMULATE] | 下界−1，loss1；pending不进监督 | 多线程在途/失败释放、均值与探索分母独立统计 | 2、5 |
| `S25` | KG-SP/KG-M [graph][KG-BUDGET] | [节点结构][EZ-SEARCH-HEADER] | NOVC局面/玩家/规则/输入条件及必要历史入key | 同局面多顺序到达的小图、边/节点权重、多父释放/重复路径 | 5 |
| `S26` | KG-M/L15/L17 [权重][KG-UNCERTAINTY] | [NN][EZ-EVALUATE-NODE]、[heads][EZ-HEADS] | 真实W−L误差标准差，Go score导数0 | 手算e→sqrt(e)→w、缺能力回退与支持能力真实输出、图统计 | 4、6a |
| `S27` | KG-M/L15/L17 [版本loss][T-KG-OPTLOSS] | [heads][EZ-HEADS]、[NN][EZ-EVALUATE-NODE] | 去score的long/short权重、推理用short head | 固定logits混合与梯度、完整终局/side gate、D4/cache/版本 | 4、6b |
| `S28` | KG-SP/KG-M [noise pruning][KG-AGGREGATE] | [backup][EZ-AGGREGATE] | 纯W−L utility；scale0.15/cap1e50 | 同visits不同噪声的小树backup及二阶统计，不混同S04 | 6c |
| `D01` | GM [NOVC opening][GM-OPENING] | [开局][EZ-OPENING-CODE]、[Match][EZ-MATCH-CALL] | 方棋盘NOVC；成对换色协议保留，bot选择对齐 | 独立几何权重/拒绝/回退、黑白异模型及成对结果口径 | 3 |
| `D02` | GM [逐行writer][GM-WRITE] | [writer][EZ-WRITER]、[view][EZ-VIEW] | 仅Renju，两个禁手平面及flag同步 | 观察重复输出行独立随机、eval完整，不能只验均值 | 3、4 |
| `D03` | GM [loader][GM-PLAYSETTINGS-SP]、[preset][T-GM-PRESET] | [policy init][EZ-POLICY-INIT-CODE] | SP6/T1.6，Match默认关、显式开启按执色；无Go面积komi | 固定指数抽样/已有手数/0.0002分支、loader实参、异bot | 3 |
| `D04` | GM [尺寸示例][T-GM-ENV]、KG [mask][KG-MODEL] | [env][EZ-ENV]、[pool][EZ-NORM-CODE] | 用户保持现有方形15…11及权重，无矩形 | 各尺寸采样/有效mask/padding隔离/BN/pool/D4、明确拒绝矩形 | 3、8 |
| `D05` | GM [rules][T-GM-RULES]、[输入][GM-INPUT] | [棋规][EZ-GAME]、[env][EZ-ENV] | 三规则可混训，baseline只Renju；VCN暂不纳入 | Renju递归双三/双四/长连/恰五对照、非零混合及规则输入 | 3 |
| `N01` | L15 [plain][T-KG-PLAIN] | [network][EZ-NETWORK-CODE]、[配置][EZ-CONFIG-CODE] | b10c128-fson-mish，15版五子棋heads，已接入独立架构 | 来源权重映射固定前向/梯度、gpool5/8、导出和更新 | 8a |
| `N02` | L15 [NBT preset][KG-B5] | [NBT][EZ-NBT-CODE]、[network][EZ-NETWORK-CODE] | b5c192nbt-fson-mish；v15 heads已接入 | trunk固定输入回归、两级初始化与head宽度，更新/导出 | 4、8 |
| `N03` | L17 [TF preset][T-KG-TFPRESET] | [network][EZ-NETWORK-CODE]、[配置][EZ-CONFIG-CODE] | b5c192h3nbttfrs，fixup/RMSNorm/RoPE/SwiGLU | 固定attention/RoPE/梯度、padding隔离、FP32/AMP/export；不套fson | 8b |
| `N04` | L15/L17 [初始化][KG-INITIALIZE] | [norm][EZ-NORM-CODE]、[NBT][EZ-NBT-CODE] | fson仅两个卷积预设；TF沿fixup组件 | 层尺度/截断方差、masked mean/std及推理折叠独立对照 | 1、8 |
| `N05` | L15/L17 [heads][KG-MODEL] | [heads][EZ-HEADS]、[导出][EZ-EXPORT-CODE] | 6policy/WDL/3TD/error，v17可选Q；Go专有head排除 | 形状/版本/头宽度、pool公式、writer到推理输出往返；无占位 | 4、8 |
| `N06` | L15/L17 [D4 reader][KG-D4] | [增强][EZ-D4-CODE]、[reader][EZ-READER] | 所有新空间target同变换；validation启用则随机D4 | 非对称棋盘标记/逐动作Q与policy坐标、global和值不变 | 4、9a |
| `N07` | L15/L17 [SWA][KG-SWA] | [优化状态][EZ-OPT-CODE]、[发布][EZ-OPT-INFERENCE] | 保留本地轮period64000、scale8；消费时钟与skip | 首样本/EMA/buffers、Lookahead同步边界/轮末发布和恢复 | 1、9 |
| `N08` | L15/L17 [Lookahead][KG-LOOKAHEAD] | [同步][EZ-OPT-STEP]、[轮末][EZ-TRAIN-LOOP] | sub_epochs默认1，可选floor分段，k6/alpha0.5；真实CUDA分段恢复已验 | 手算fast/slow/counter、轮末尾部、skip与SWA及状态加载 | 1、9 |
| `N09` | L15/L17 [LR][KG-LR] | [warmup][EZ-WARMUP-CODE]、[刷新][EZ-OPT-CONFIGURE] | 按消费样本8段，5/50batch刷新 | 八个250k阈值及2亿边界时序，overflow消费不回退 | 1、9 |
| `N10` | L15/L17 [WD][KG-WD] | [分组][EZ-OPT-CODE]、[刷新][EZ-OPT-CONFIGURE] | fson/Transformer fixup分别分组；默认SGD、AdamW/运行范数分支 | 独立每组LR/WD、norm初值及100batch/运行均值生效时点 | 1、8 |
| `N11` | L15/L17 [clip/AMP][KG-GRAD] | [裁剪][EZ-GRAD-CODE]、[train][EZ-TRAIN-LOOP] | fson/fixup2500、BN5500/AdamW11000；FP32 heads，AMP skip | 独立阈值、真实CUDA overflow与计数、精度导出，不CPU替代 | 1、2、8 |
| `L01` | L15/L17 [policy loss][KG-METRICS] | [loss][EZ-LOSS-CODE]、[writer][EZ-WRITER] | 六policy默认、v17 Q可选；无pass；int16量化后归一化 | 可手算CE/soft及量化小权重、optimistic梯度与无效opponent | 1、4 |
| `L02` | L15/L17 [TD/CE][T-KG-TD]、[loss][KG-VALUELOSS] | [targets][EZ-VIEW]、[loss][EZ-LOSS-CODE] | 真draw、三TD、side搜索值、error，value系数0.72 | 手算有限轨迹及视角、CE−entropy、side长度1和无score适配 | 1、4、7 |
| `L03` | KG/GM [mask][KG-METRICS] | [棋规][EZ-GAME]、[NN][EZ-EVALUATE-NODE]、[loss][EZ-LOSS-CODE] | 搜索空点/棋规；训练on-board含occupied/禁手，无pass | 独立合法域/soft epsilon域、dropout不改判负及padding | 3、4 |
| `L04` | KG-SP/L15/L17 [取整][KG-ROUND] | [repeat][EZ-VIEW]、[backward][EZ-TRAIN-LOOP] | 频率随机取整一次，loss样本sum；消费与成功更新分计 | 固定频率取整、重复行有效性不双乘、AMP/重分析统计 | 1、4、7、9 |
| `R01` | KG shuffle [random/window][KG-RANDOMROWS] | [catalog][EZ-CATALOG]、[窗口][EZ-SHUFFLE-CODE] | 当前m150000（GM profile）；usable random封顶与本地产样quota分开 | random超m及m前后窗口/累计，不能把新窗口量拿去补桶 | 9a |
| `R02` | KG shuffle [窗口][KG-WINDOW] | [窗口][EZ-SHUFFLE-CODE] | 当前p0.8（GM profile）；KG脚本0.65、CLI默认1独立记录 | 独立整数公式含s/max/offset，不仅四参数子集 | 9a |
| `R03` | KG shuffle [窗口][KG-WINDOW] | [窗口][EZ-SHUFFLE-CODE] | 当前a0.3（GM profile），s默认m及扩展真实生效 | 起点/增长、scale/offset组合、非法配置拒绝 | 9a |
| `R04` | KG shuffle [groups][KG-SHARDIFY] | [scatter][EZ-SHUFFLE-SCATTER]、[catalog][EZ-CATALOG] | K20M/all；近期mtime、累积组≥阈值、round和无重复seed | 独立row-ID、阈值6+6/round反例、waves每行不丢不重及抽样 | 9a |
| `R05` | KG [同步脚本][E-KG-SYNC]与[train][KG-TRAIN] | [产样规划][EZ-RUNTIME-PLAN]、[控制器][EZ-RUNTIME-LOOP] | 用户保留固定训练量及ratio8规划，无bucket，明确非来源节奏 | 每轮128000消费样本/new rows缺口、完整局超额、冷启动/重启幂等 | 9b、9c |
| `A01` | KG/GM [utility/终局][KG-BUDGET] | [状态][EZ-STATE]、[棋规][EZ-GAME] | 当前方W/D/L，纯W−L、draw0，移除Go score/komi | 精确胜/和/负、backup变号、禁手失利与取消非draw；辅助视角 | 3、4、5、7 |
| `A02` | KG-SP/KG-M [budget][KG-BUDGET] | [run][EZ-RUN]、[参数][EZ-SP] | 400/70例外、E/M500；并发规模单列，time/plays能力补齐 | fresh/reused visits和新plays、终局/根集成NN计数、停止边界 | 2、5 |
| `A03` | KG [hash/cache][KG-NNCACHE] | [cache][EZ-CACHE]、[worker][EZ-WORKER] | 朝向对齐、PDA条件入key；用户保留每轮固定模型，无局内切换 | 碰撞、朝向随机复用/集成绕过、模型/精度隔离、轮间重载 | 2、4、7、9b |
| `A04` | KG [save/export][KG-SAVE] | [checkpoint][EZ-CHECKPOINT]、[恢复][EZ-RESTORE]、[导出][EZ-EXPORT-CODE] | 保留整轮事务、SWA/raw直发布，无gatekeeper；不伪称来源中间恢复 | 连续/中断各边界完整状态、预取消费cursor、发布验证与历史产物不覆盖 | 9c |

## 核查证据

- **静态来源**：沿KataGo入口→setup/play settings→search/NN→writer→shuffle→learner追分支；沿EtaZero配置加载→实际SP/Match调用→记录/重复行→loss/优化→模型发布与恢复对照。已实现项的“静态一致”限定到本表所写子集与二值mask等输入约束。
- **可复跑入口**：[check_reference_formulas.py](/home/sky/RL/EtaZero-lab/EtaZero_V0/tests/reference/check_reference_formulas.py:1)仅用标准库，从源码AST提取真实函数及梯度阈值分支；不导入训练模块、PyTorch或CUDA，不更新模型、不写实验产物。结果以JSON输出，含六个被核对文件与核查脚本的SHA256；任一对照不符合预期时非零退出。
- **纯函数对照结果**：replay四参数公式252个案例整数结果完全一致；SGD/AdamW六参数组共31104组案例的LR和WD均一致，浮点容限为相对`1e−12`、绝对`1e−15`。这些是脚本的实际复跑计数，每个参数组计一个案例，同时检查LR与WD；仅证明同输入的局部公式，不消除累计量、norm采样和刷新节奏差异。
- **裁剪与范数时序**：12个SGD fson和12个AdamW阈值案例均与来源一致；batch128/scale1的SGD cap=1767.76695297。范数snapshot/all-batch、Lookahead筛选开关及2/100打印间隔，5760个累积和/权重/均值对照通过；使用来源metrics_logging及set_snapshot_metrics函数，不自行复制预期公式。
- **文档检查**：57个唯一ID、每行7列、无待填项；全部引用定义和本地文件/行号有效。
- **真实CUDA与恢复**：RTX5090、PyTorch2.12.0+cu132，25项learner/实际NBT短链路及独立FP16推理检查通过，覆盖FP16/BF16、SGD/AdamW、eager/compile、实际head dtype、overflow消费、SWA导出及底层learner续训。另有多推理服务/FP16原生推理、packed/waves/预取回归通过。新目录与命令见plan批次1记录；连续/恢复必要状态逐位相同。
- **批次2搜索/推理证据**：3项C++测试、133项快速测试、11项实际CUDA回归通过；覆盖手算FPU/方差/LCB、canonical cache与miss-only RNG、D4坐标、根集成NN计数、fresh/reused budgets、同bot清树与异bot复用、时间/零playout边界、真实多server失败与信号收尾。独立配置/源码差异/原始运行索引见 [批次2验收manifest](/home/sky/RL/EtaZero-lab/EtaZero_V0/data/validation_batch2_20261002/manifest.json)。
- **批次3环境/输出行证据**：136125个KataGomo独立禁手/类别对照、3项C++、135项快速测试及7项真实CUDA检查通过；含逐输出行dropout、黑白异模型balance/policy角色、五尺寸三规则/padding/终局、禁手叶变号/无NN及on-board训练域。有限规则语料不证明所有递归局面等价，成对换色与来源逐局初始化的区别保留；详见 [批次3验收manifest](/home/sky/RL/EtaZero-lab/EtaZero_V0/data/validation_batch3_20261002/manifest.json)。
- **批次4 v15监督/数据证据**：151项快速检查、4项C++和13项真实CUDA检查通过；含有限轨迹TD、误差及optimistic独立梯度、int16顺序、side独立搜索标签与gate、原生NPZ全链往返、真实侧行训练/导出、AMP编译/恢复和十一loss日志/绘图。该历史v15证据未覆盖可选Q；v17 Q现已在8c完成，见 [v15验收manifest](/home/sky/RL/EtaZero-lab/EtaZero_V0/data/validation_batch4_v15_20261002/manifest.json)。
- **批次5图搜索证据**：152项快速检查、5项C++及5项宿主Address/Undefined/LeakSanitizer、11个不同CUDA case通过。200次单步图与1080项固定来源scalar对照、graph关闭/实际NN cache命中/泄漏、三规则转置、私有根推进/大图回收及共享叶异常覆盖；仅机制验收，无吞吐/棋力结论。见 [图搜索验收manifest](/home/sky/RL/EtaZero-lab/EtaZero_V0/data/validation_batch5_graph_20261002/manifest.json)。
- **批次7采样证据**：157项CPU、7项C++/宿主ASan/UBSan/LeakSanitizer、10个不同CUDA case通过；3072完整hint/PCR/reduce/PDA预算、336真实core频率、204 PDA输入/预算及216 direct/平滑surprise来源对照。PDA/side/reanalysis及hint/early/late/shared-worker-fork贯通真实数据/训练/导出；仅工程机制证据，无棋力或吞吐结论。见 [第七批manifest](/home/sky/RL/EtaZero-lab/EtaZero_V0/data/validation_batch7_sampling_20261002/manifest.json)。
- **三架构及v17 Q证据**：批次4/8完整，CPU180、C++和sanitizer各7/7、29个不同真实CUDA用例通过；来源Q整数函数46140例、真实Metrics解析梯度、三精度eager/同条件编译完整网络对照通过。含node/edge区分、终局/共享图、逐输出行随机量化、side/reanalysis/D4及严格中断恢复；baseline Q关闭。失败、构建、当前配置/源码与原始产物索引见[8c manifest](/home/sky/RL/EtaZero-lab/EtaZero_V0/data/validation_batch8_q_20261002/manifest.json)。
- **最终执行证据**：第10批CPU199通过、C++7/7及专用真实CUDA112/112通过；独立来源对照、三架构完整网络与Transformer Q三精度eager/compiled对照重新执行，原始失败/成功结果及逐项范围见[最终验收](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json)。
- **未验证范围**：正式训练效果、棋力、匹配来源负载的端到端吞吐及全部硬件故障组合未验收。并行调度、缓存键、文件边界/RNG序列、同步整轮编排和后端能力差异仍分别保留，不宣称57+26项全面等价。

在仓库根目录执行：

```bash
conda run -n pytorch python EtaZero_V0/tests/reference/check_reference_formulas.py
```

来源路径默认`~/RL/SkyZero/KataGo`，可用`--katago-root /absolute/path/to/KataGo`指定。该脚本独立于训练入口；更换来源后需重新核对本表的commit和适用分支，不能沿用旧结果。

覆盖范围由脚本中的参数网格定义：

- **Replay**：min_rows为1/16/150000/250000，指数0.5/0.65/1，expand_per_row为0.4/1/2；累计行数取0、m−1、m、m+1、2m、10m、100m。只核对s=m、d=0、无max_rows的四参数子集。
- **LR/WD**：SGD/AdamW、batch=64/128/256/1024、LRscale=0.25/1/4；warmup开/关，0、八个阈值的前一行与阈值本身、400万样本；norm快照缺省或为基线的0.25/4倍；默认分组系数及一组非默认系数，Lookahead alpha=0.5/1。固定LRscale、world_size=1、norm_kind=fixscaleonenorm，关闭Muon及自动/循环LR。
- **梯度阈值**：SGD/AdamW × 上述四个batch × 三个LRscale；只核对自动阈值（Eta override=0、来源无clip multiplier），该纯函数脚本不涵盖实际梯度、裁剪触发率或AMP overflow；真实AMP另由GPU检查覆盖。

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
[EZ-WIDTHS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/config.py:121
[EZ-SETTINGS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:42
[EZ-SP-CALL]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:131
[EZ-MATCH-CALL]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:397
[EZ-WORKER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:245
[EZ-FPU-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:27
[EZ-SELECTION]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:31
[EZ-NOISE-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:15
[EZ-EVALUATE-NODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:206
[EZ-EXPAND]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:288
[EZ-SIMULATE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:327
[EZ-RUN]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:460
[EZ-RESULT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:474
[EZ-ADVANCE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:582
[EZ-SEARCH-MATH]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search_math.cpp:7
[EZ-AGGREGATE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search_math.cpp:100
[EZ-SEARCH-HEADER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/include/etazero/search.h:1
[EZ-LIMITS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/search_limits.cpp:45
[EZ-WEIGHTS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/record.cpp:66
[EZ-WRITER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/record.cpp:380
[EZ-OPENING-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/opening.cpp:131
[EZ-POLICY-INIT-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/opening.cpp:219
[EZ-GAME]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/include/etazero/game.h:39
[EZ-STATE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/include/etazero/algorithm.h:42
[EZ-CACHE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/inference/batcher.cpp:96
[EZ-NORM-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:52
[EZ-NBT-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:152
[EZ-NETWORK-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:211
[EZ-HEADS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:166
[EZ-LOSS-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:370
[EZ-INFERENCE-NET]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:400
[EZ-D4-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/symmetry.py:4
[EZ-OPT-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:8
[EZ-WARMUP-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:46
[EZ-OPT-CONFIGURE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:144
[EZ-GRAD-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:187
[EZ-OPT-STEP]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:195
[EZ-OPT-INFERENCE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:237
[EZ-TRAIN-LOOP]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/training.py:194
[EZ-CHECKPOINT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/training.py:53
[EZ-VIEW]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/data.py:236
[EZ-CATALOG]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/data.py:277
[EZ-SHUFFLE-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:22
[EZ-SHUFFLE-SCATTER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:121
[EZ-SNAPSHOT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:314
[EZ-RUNTIME-PLAN]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:167
[EZ-RUNTIME-LOOP]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:287
[EZ-RESTORE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:549
[EZ-READER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/reader.py:106
[EZ-EXPORT-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/export.py:47
[KG-NORM-TIMING]: /home/sky/RL/SkyZero/KataGo/python/katago/train/trainloop_helpers.py:285
[FORMULA-F1]: /home/sky/RL/EtaZero-lab/EtaZero.md:131
[FORMULA-F2]: /home/sky/RL/EtaZero-lab/EtaZero.md:144
[FORMULA-F3]: /home/sky/RL/EtaZero-lab/EtaZero.md:154
[FORMULA-F4]: /home/sky/RL/EtaZero-lab/EtaZero.md:168
[FORMULA-F5]: /home/sky/RL/EtaZero-lab/EtaZero.md:186
[FORMULA-F6]: /home/sky/RL/EtaZero-lab/EtaZero.md:196
[FORMULA-F7]: /home/sky/RL/EtaZero-lab/EtaZero.md:209
[FORMULA-F8]: /home/sky/RL/EtaZero-lab/EtaZero.md:219
[FORMULA-F9]: /home/sky/RL/EtaZero-lab/EtaZero.md:231
[FORMULA-F10]: /home/sky/RL/EtaZero-lab/EtaZero.md:246

[E-EZ-RUNTIME]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:232
[E-EZ-PLAN]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:167
[E-EZ-LAUNCH]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:471
[E-EZ-RESTORE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/runtime.py:549
[E-EZ-TRAIN]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/training.py:194
[E-EZ-SAVE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/training.py:53
[E-EZ-OPT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/optimization.py:114
[E-EZ-NET]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:384
[E-EZ-NATIVE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/native.py:11
[E-EZ-EXPORT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/export.py:47
[E-EZ-STORAGE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/storage.py:17
[E-EZ-CATALOG]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/data.py:277
[E-EZ-VIEW]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/data.py:236
[E-EZ-GROUP]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:223
[E-EZ-SCATTER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:121
[E-EZ-MERGE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:147
[E-EZ-WAVE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:254
[E-EZ-RESOURCE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:186
[E-EZ-SNAPSHOT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/shuffle.py:314
[E-EZ-READER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/reader.py:18
[E-EZ-PREFETCH]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/reader.py:159
[E-EZ-MAIN]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:131
[E-EZ-WORKER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/commands/main.cpp:245
[E-EZ-BATCH]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/inference/batcher.cpp:59
[E-EZ-SERVE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/inference/batcher.cpp:167
[E-EZ-TORCH]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/inference/torch_backend.cpp:16
[E-EZ-SEARCH]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:288
[E-EZ-SEARCHRUN]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:460
[E-EZ-WRITER]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/record.cpp:181
[E-EZ-PACK]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/record.cpp:258
[E-EZ-PLOT]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/plotting.py:34
[E-KG-PROFILES]: /home/sky/RL/SkyZero/KataGo/SelfplayTraining.md:29
[E-KG-SYNC]: /home/sky/RL/SkyZero/KataGo/python/selfplay/synchronous_loop.sh:91
[E-KG-LOOP]: /home/sky/RL/SkyZero/KataGo/python/selfplay/shuffle_and_export_loop.sh:49
[E-KG-SP]: /home/sky/RL/SkyZero/KataGo/cpp/command/selfplay.cpp:245
[E-KG-POLL]: /home/sky/RL/SkyZero/KataGo/cpp/command/selfplay.cpp:335
[E-KG-MANAGER]: /home/sky/RL/SkyZero/KataGo/cpp/program/selfplaymanager.cpp:348
[E-KG-NN]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/nneval.cpp:791
[E-KG-CACHE]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/nneval.cpp:1125
[E-KG-QUEUE]: /home/sky/RL/SkyZero/KataGo/cpp/core/threadsafequeue.h:117
[E-KG-CUDA]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/cudaandrocmbackend.inc:5085
[E-KG-SEARCH]: /home/sky/RL/SkyZero/KataGo/cpp/search/search.cpp:1253
[E-KG-SHUFFLE]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:199
[E-KG-GROUP]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:384
[E-KG-OUTPUT]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:406
[E-KG-WAVE]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:1241
[E-KG-ROWS]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:1058
[E-KG-SUMMARY]: /home/sky/RL/SkyZero/KataGo/python/summarize_old_selfplay_files.py:46
[E-KG-READ]: /home/sky/RL/SkyZero/KataGo/python/katago/train/data_processing_pytorch.py:28
[E-KG-GENERATOR]: /home/sky/RL/SkyZero/KataGo/python/katago/utils/training_data_generator.py:7
[E-KG-BUCKET]: /home/sky/RL/SkyZero/KataGo/python/train.py:1399
[E-KG-DEBIT]: /home/sky/RL/SkyZero/KataGo/python/train.py:1413
[E-KG-SAVE]: /home/sky/RL/SkyZero/KataGo/python/train.py:645
[E-KG-VAL]: /home/sky/RL/SkyZero/KataGo/python/train.py:1954
[E-KG-HEAD]: /home/sky/RL/SkyZero/KataGo/python/katago/train/model_pytorch.py:4108
[E-KG-COMPILE]: /home/sky/RL/SkyZero/KataGo/python/katago/train/trainloop_helpers.py:150
[E-KG-SH]: /home/sky/RL/SkyZero/KataGo/python/selfplay/shuffle.sh:39
[E-KG-EXPORT]: /home/sky/RL/SkyZero/KataGo/python/selfplay/export_model_for_selfplay.sh:38
[E-KG-PACK]: /home/sky/RL/SkyZero/KataGo/cpp/dataio/trainingwrite.cpp:314
[E-KG-WRITE]: /home/sky/RL/SkyZero/KataGo/cpp/dataio/trainingwrite.cpp:1080
[T-GM-PRESET]: /home/sky/RL/SkyZero/KataGomo/scripts/selfplay.cfg:61
[T-GM-ENV]: /home/sky/RL/SkyZero/KataGomo/python/scripts/selfplay.cfg:1
[T-GM-RULES]: /home/sky/RL/SkyZero/KataGomo/cpp/game/rules.h:10
[T-KG-TD]: /home/sky/RL/SkyZero/KataGo/cpp/dataio/trainingwrite.cpp:411
[T-KG-REANALYZE]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:1370
[T-KG-SIDEWRITE]: /home/sky/RL/SkyZero/KataGo/cpp/dataio/trainingwrite.cpp:1266
[T-KG-ERRORHEAD]: /home/sky/RL/SkyZero/KataGo/python/katago/train/model_pytorch.py:4352
[T-KG-ERRORLOSS]: /home/sky/RL/SkyZero/KataGo/python/katago/train/metrics_pytorch.py:308
[T-KG-ERRORPOST]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/nneval.cpp:1397
[T-KG-OPTLOSS]: /home/sky/RL/SkyZero/KataGo/python/katago/train/metrics_pytorch.py:637
[T-KG-QLOSS]: /home/sky/RL/SkyZero/KataGo/python/katago/train/metrics_pytorch.py:90
[T-KG-PDASAMPLE]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:624
[T-KG-PDAINPUT]: /home/sky/RL/SkyZero/KataGo/cpp/neuralnet/nninputs.cpp:2673
[T-KG-PLAIN]: /home/sky/RL/SkyZero/KataGo/python/katago/train/modelconfigs.py:168
[T-KG-TFPRESET]: /home/sky/RL/SkyZero/KataGo/python/katago/train/modelconfigs.py:1178
[T-EZ-SCHEMA]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/schema.py:1
[T-KG-POLICYQUANT]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:814

[EZ-TF-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/transformer.py:1
[EZ-SWIGLU-CODE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/fused_swiglu.py:24

[EZ-Q-LOSS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/python/etazero/network.py:357

[EZ-Q-TARGETS]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/search/search.cpp:540

[EZ-Q-QUANTIZE]: /home/sky/RL/EtaZero-lab/EtaZero_V0/cpp/src/selfplay/record.cpp:16

[KG-Q-WRITER]: /home/sky/RL/SkyZero/KataGo/cpp/dataio/trainingwrite.cpp:385

[KG-Q-NODE]: /home/sky/RL/SkyZero/KataGo/cpp/search/searchresults.cpp:515

[KG-Q-EXTRACT]: /home/sky/RL/SkyZero/KataGo/cpp/program/play.cpp:862

[E-KG-CONCURRENCY]: /home/sky/RL/SkyZero/KataGo/cpp/command/selfplay.cpp:163

[E-KG-STOP]: /home/sky/RL/SkyZero/KataGo/cpp/search/search.cpp:619

[E-KG-TIMER]: /home/sky/RL/SkyZero/KataGo/python/shuffle.py:699

[GM-SHUFFLESH]: /home/sky/RL/SkyZero/KataGomo/python/shuffle.sh:49
