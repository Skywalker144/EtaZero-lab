# EtaZero V3.2 与 KataGo 对齐审查及实现复核

**当前结论：本轮约定范围内的 U1—U5、U8 已实现并通过下列回归。** 同时补齐了用户追加的矩形围棋与适用 GTP 扩展。U6 的 DDP／新增多卡调度继续暂缓，U7 的 Gumbel tree／fresh-tree 继续保留。实现、数值检查和短训练证据不代表棋力或端到端吞吐已经与 KataGo 等价。

固定来源仍为 KataGo `d91ea855110dae533f0aada947b2b7d78cc8a4e1`，实际路径 `/home/sky/RL/SkyZero-lab/KataGo`；当前清单 **139 个来源文件 SHA256 全部匹配**。原始初审、失败日志和历史结果均保留；结构化复核结果位于 [审查证据](katago-target-review-2026-10-24.json) 的 `implementation_followup`。

## 剩余项的处理结果

| 项目 | 当前实现与核对边界 |
|---|---|
| U1 搜索并行 | 展开竞争释放预约并返回未计数重试；dirty counter 合并回传；MuZero 保留工作线程，选择／统计使用节点锁，推理不持有节点锁。普通 PUCT 完成计数与 Gumbel 严格配额分别保留。 |
| U2 公共协议 | `analysis --katago-json` 使用来源请求、响应、优先级、多分析线程和控制命令。GTP 覆盖矩形尺寸、摆子、SGF、规则、运行时参数、让子、棋钟、搜索／取消、raw NN 对称及适用诊断。HumanSL profile 按请求隔离，输出独立 humanPolicy。 |
| U3 起点与 hint | 使用来源 SGF／PositionSample 解析，覆盖设置子、分支、唯一局面、SGF hash 排除、手数与独立 sampling/training 权重；矩形真实历史进入后续自对弈和数据。 |
| U4 外部／自动模式 | 来源 SGF 遍历、玩家／注释过滤、符号和文件衰减；自动 `_poses.txt` 按尺寸保存，加载时去重、逐位置衰减和按文件边界清理，下次 clear_board 重载。 |
| U5 配置 | 平坦 key/value、引号／注释／原位 include、重复规则、来源别名、缺省与数值边界；GTP 分析／落子参数分开，Go history modes 变更时重放。未知／不适用字段按目标要求明确报错，不沿用来源“告警后继续”的宽松行为。 |
| U8 公共数据 | AlphaZero 默认只写公共训练字段、必要扩展和控制器轻量 bookkeeping；完整研究轨迹显式启用。MuZero 保留连续展开轨迹。公共读取不依赖私有 metadata，raw NPZ 按临时文件关闭后 rename 发布，不逐文件／目录 fsync。 |
| 运行组织 | selfplay、shuffle、learner、candidate、accepted/rejected 职责已逐项映射，见 [运行目录](implementation.md#运行目录)。保留同步阶段和精确续训需要的状态；checkpoint／模型仍按恢复需要持久化。 |

矩形盘的动作、pass、观测 mask、规则历史、SGF、TD／value-surprise 时间尺度、early fork 抽样和 MuZero 吸收态分布均使用实际宽高／面积；NN canvas 仍固定。GTP 报告区分节点 visits/weight 与 edgeVisits，使用根剪枝值，支持对称等价着、FPU minmoves、LZ 数值尺度、PV visits 和空行取消。参数无实际变化时保留搜索状态。

## 保留的合理边界

- 单卡 learner 与同步 selfplay → shuffle → train → publish 不改为 DDP 或新增多卡调度。
- Gumbel 使用 tree，不做 graph 合并；当前 fresh-tree 与跨手复用的后续评估保持分开。
- score utility、score/ownership/seki 监督、完整 passing/dame/anti-mirror、让子补偿仍排除。**未终局 final_score／死活估计依赖被排除的预测头，明确拒绝；终局规则计分、摆子及让子命令可用。**
- Web 多棋类可变会话保持串行；公共 JSON 分析采用来源的并行任务接口。诊断文本格式是本地适配，不能作为搜索性能等价证据。TorchScript 没有的模型版本／variance-time 字段不伪造。

具体命令和配置见 [交互引擎](engine.md)、[搜索控制](search-controls.md)，数据契约见 [数据链路](implementation.md#数据链路)。

## 实际验证

| 检查 | 结果 |
|---|---|
| 原生构建与 CTest | 构建通过，18/18；5 组并行／图搜索关键测试各重复 20 次全部通过。 |
| 完整快速 pytest | **835 passed，425 skipped，0 failed**；跳过项不计为通过。 |
| 真实 RTX 5080 CUDA | 修正并复验后共 24 个不同用例通过，含 JSON/GTP/Web、HumanSL metadata 并发隔离、图竞争／失败清理、AZ 紧凑数据的编译／非编译短训练恢复，以及编译 AMP MuZero 完整短流水线恢复。重复执行不重复计数。 |
| 采样独立来源对照 | 336 组 surprise 权重、3,072 组 hint／PCR／PDA／fork 预算对照通过；矩形面积另有原生数值和重放检查。 |
| Q 独立来源对照 | 46,140 个 float32／随机量化边界样例通过，链接当前生产库。 |
| 协议与数据边界 | SGF setup／分支／排除、矩形双方向 AZ/MZ 分析及自对弈、完整／紧凑记录、MuZero 吸收态、GTP 参数事务、取消／帧分隔、规则重放、benchmark 与缺省棋钟均有执行检查。 |

日志位于 `/tmp/etazero-v32-align-20261024`。GPU 失败与复验均保留；修正涉及导出诊断接口的测试夹具、来源允许的单访问预算，以及 HumanSL weightless playout 与根访问计数的区别。协议回归还发现并修复了 genmove_analyze 结束空行缺失、只读搜索改写终局状态、矩形面积依赖，以及批量 new/play 请求错误取消前序普通建局的竞态；500 条私有状态命令和 20 组 GTP genmove/clear 检查顺序语义。最终复验将 MuZero 随机夹具改为单推理服务，短展开验证连续轨迹，超过完整棋盘最大局长的 40 步展开保证覆盖吸收态；不再假定随机采样必然保留终局附近的起始行。

尚未开展完整棋力训练或端到端等时间吞吐实验；有限测试不能证明所有局面和并发交错等价。初审已通过的其他来源数学／网络对照仍保留原记录，不将其改写为本轮重新运行的结果。

---

## 初审快照（实现前，保留原始结论）

以下问题描述、行号和统计对应实现前工作区；当前状态以上面的实现复核为准。

**结论：尚未全部对齐。** 按当前 `katago-target.md` 的范围，U1—U5、U8 六类差异仍需对齐；U6 的 DDP 和新增多卡调度暂缓，U7 的 Gumbel tree／无图共享已列为约定差异，跨手复用单独评估。运行组织、持久化频率和端到端性能还有待核实。共有数学、来源网络、训练内部时钟和专用 CUDA 推理已有较强的通过证据，但不能据此将公共协议、语料格式和并行行为一并判为通过。

审查日期为 2026 年 10 月 24 日，检查对象是 V3.2 当前工作区，包含审查开始前已有的未提交修改。固定来源为 KataGo `d91ea855110dae533f0aada947b2b7d78cc8a4e1`。约定路径 `/home/sky/RL/SkyZero/KataGo` 不存在，实际使用清单指向的 `/home/sky/RL/SkyZero-lab/KataGo`；其 HEAD 和清单内 **123 个来源文件的 SHA256 全部匹配**。

目标见 [katago-target.md](katago-target.md)，本次结果、失败信息和数值对照记录见 [审查证据](katago-target-review-2026-10-24.json)。本审查没有修改算法、配置或测试，也没有继承旧审查记录的通过结论。

## 差异及适用范围

### U1 搜索竞争与回传的并行行为不同

KataGo 的线程遇到其他线程正在展开的节点时，返回一次未计数的 playout，外层重试；EtaZero AlphaZero 在该节点的条件变量上等待，展开完成后继续。KataGo 用 `dirtyCounter` 合并并发回传，EtaZero 每次回传取得 `update_mutex` 后重新聚合。MuZero 则用一把搜索级 mutex 包围整条路径的选择和回传，并在每次搜索创建工作线程。

这会改变竞争时的工作分配、统计更新时点和锁开销。随机线程交错无需逐位相同，但以上不是单纯的变量名或语言变化；现有单线程数值对照不能证明并行语义与性能等价。MuZero 的 latent 结构也不必然要求将所有路径选择与回传串行化。

本项确认的是实现与执行行为差异，没有证明当前实现一定更慢或存在算法错误。对齐要求以适用的并行语义和性能证据为准，不要求逐行照搬锁的实现。

依据：[EtaZero 等待与回传](../cpp/src/search/search.cpp)，重点为 422、450 行；[MuZero 搜索](../cpp/src/muzero/search.cpp)，重点为 198—247 行；来源 `cpp/search/search.cpp:658,1311`、`cpp/search/searchupdatehelpers.cpp:139`。需要闭合的是竞争、合并更新和线程生命周期，而不是把普通 PUCT 改成 Gumbel 的严格预算。

### U2 JSON 公共协议和分析任务调度不同

来源分析请求使用 `boardXSize`、`boardYSize`、`initialStones`、`analyzeTurns`、`overrideSettings`、`maxVisits` 等字段，并支持分析任务优先级、多分析线程以及来源的控制请求。EtaZero 必须提供 `command`，采用 `size`、`rule`、`visits` 等私有字段和响应包装，由单个 worker 顺序处理队列。

实际向 EtaZero 提交来源形状的请求，会返回 `key 'command' not found`。这不是围棋 score/ownership 排除项造成的，也不是多棋类必须改变的部分。当前 JSON/GTP 会话可运行，不代表来源客户端可以直接使用。GTP 的公共命令覆盖也仍有差距，例如当前 `kata-raw-nn` 的对称参数范围。

依据：[会话解析与单 worker](../cpp/src/commands/session.cpp)，重点为 364、391、398—400 行；来源 `cpp/command/analysis.cpp:112,229,456,604`。本次当前命名预设的真实 GPU JSON/GTP 测试通过；它验证的是 EtaZero 当前接口。

### U3 外部起点和 hint 语料不同

来源读取 SGF、遍历唯一局面，支持摆子、分支和基于 SGF hash 的排除；hint/startpose 使用 `Sgf::PositionSample` 的 JSON 行。EtaZero 起点目录只收集 `.startpos`/`.txt`，逐行要求自定义的 id、棋盘、规则、权重和动作前缀；hint 则要求另一种空白分隔格式。其排除列表匹配自定义 id。

因此，即使概率、turn 衰减和训练权重公式一致，直接提供相同的来源语料也不能产生相同候选集合。需要对齐围棋公共输入语义，再为其他棋类加入必要扩展；仅移植抽样公式不够。

依据：[起点和 hint 读取](../cpp/src/selfplay/sampling.cpp)，重点为 32—66、77—132 行；来源 `cpp/program/play.cpp:244—352`。

### U4 外部 pattern 与自动重复历史链路不完整

EtaZero 外部 pattern 读取 `.patterns` 自定义 JSON 行，按动作序列回放；来源读取 SGF，并遍历其中的局面。当前实现还没有来源 `autoAvoidRepeatDir` 对应的落子样本保存、按棋盘尺寸加载、衰减和超额历史清理链路。

本局重复模式、局部 pattern hash、SVB 的数学通过对照，不等于外部语料和跨会话历史机制已经完整。两者应分别验收。此处也存在 U5 所述的负 utility 被拒绝问题。

依据：[外部 pattern 加载](../cpp/src/search/pattern_bonus.cpp)，重点为 19—45 行；来源 `cpp/search/patternbonustable.cpp:143,217`、`cpp/program/setup.cpp:1079,1189`。

### U5 配置解析和合法边界不同

EtaZero 使用分节 INI、字段归属表和 `ConfigParser`，要求 `@include` 出现在 section 之前，未知键直接报错。来源使用平坦的 key/value 语法，在读取位置处理 include，支持自己的引号、行内注释、重复键选项及未使用键提示。实际读取一个合法来源形状的平坦配置时，EtaZero 报 `MissingSectionHeaderError`。

此外存在机制取值域差异：来源 `avoidRepeatedPatternUtility`、`avoidSgfPatternUtility` 和自动重复模式 utility 允许 `[-3,3]`；EtaZero 将本局 utility 和外部 `penalty` 限定为 `[0,3]`。本次直接验证 `-0.5` 被拒绝。其他边界也需统一核对，例如来源 NN policy temperature 为 `[0.01,5]`，EtaZero 公共校验仅要求正数。

这是合法配置集合发生变化，不能归为“实验选了不同参数”。依据：[配置解析](../python/etazero/config_files.py)、[配置校验](../python/etazero/config.py)，重点为 72、89、132 行；来源 `cpp/core/config_parser.cpp:247`、`cpp/program/setup.cpp:764,768,1104`。

### U6 单卡 learner 保留且 DDP 暂缓

来源训练器具有 NCCL/DDP 初始化、rank/world size、全局 batch 缩放和相关状态处理，单卡时则跳过 DDP 包装。EtaZero 当前 learner 只有单设备路径。DDP 使多张卡共同训练一个模型，与调度多个独立实验或 selfplay worker 分开判断。

按当前目标，保留现有单卡 learner，DDP 及新增多卡／跨机器调度暂缓，不再作为本轮必须补齐的缺项。单卡优化器、数据选择、内部时钟和恢复仍需对齐。依据：[learner](../python/etazero/training.py)，重点为 281—317 行；来源 `python/train.py:264,519,706,922`、`python/katago/train/trainloop_helpers.py:159`。

### U7 Gumbel 使用 tree 且跨手复用单独评估

当前配置和 C++ 构造入口均拒绝 Gumbel 与 `use_graph_search`/`reuse_tree` 组合，每次根搜索调用 `advance(0)` 清空树。按当前目标，Gumbel 使用 tree、不做图共享属于约定差异，不再要求为对齐补做 graph。

tree 并不决定是否跨手复用。当前保留 fresh-tree 实现，跨手复用的必要性及历史节点统计、当前轮候选分配与实际新增 visits 的关系单独评估；不能通过放宽 Gumbel 配额来“对齐”。**MuZero 不复用真实棋盘旧树、不使用真实棋盘 graph，属于其 latent 搜索的约定差异。**

依据：[Gumbel 构造与搜索](../cpp/src/search/gumbel_search.cpp)，重点为 7、85 行；[配置拒绝入口](../python/etazero/config.py)，重点为 451—465 行。

### U8 公共原始数据仍依赖额外的完整私有契约

EtaZero 已写出来源式 `binaryInputNCHWPacked`、`globalInputNC`、policy、global/value/Q targets，并增加必要的 draw 通道；但即便是 AlphaZero，也同时写出完整 `observations`、`globals`、多套 policy/Q/WDL、对局和侧样本轨迹。读取入口要求自定义 metadata，完整校验要求整套 `RAW_DTYPES | SOURCE_DTYPES` 字段。

因此当前不是“来源公共记录加必要棋类和四结果扩展”即可被消费的契约。MuZero 的连续轨迹有明确用途；AlphaZero 的整套重复数组则不能整体借用这一理由。研究工具可以保留自己的数据，但哪些应作为附属产物、哪些确实需要进入共有训练记录，尚未逐字段闭合。额外写入的压缩、存储与 I/O 成本也不能认定与来源等价。

依据：[写入器](../cpp/src/selfplay/record.cpp)，重点为 601—655 行；[原始数据入口](../python/etazero/data.py)，重点为 61—95 行；来源 `cpp/dataio/trainingwrite.cpp:859—892`。本结论针对公共契约和重复数据，不否定已通过的公共 target 数值与量化对照。

## 仍待核实的工程范围

运行目录不能因已有 `.internal/iterations`、`models`、`snapshots`、`checkpoints` 就直接放行，但也不能仅因目录名字不同便判算法错误。需要逐项映射来源的训练目录、候选/accepted/rejected 模型生命周期及外部工具消费方式；同步外层调度所需的状态和研究展示所需的文件应分别说明。

当前 raw shard 写入包含逐文件和目录同步；普通 Python 原子写入则默认不执行 fsync。checkpoint、发布模型、原始数据和可重建缓存的保存强度应按真实恢复需求分别核对。仅凭存在更多同步，不足以断言错误；仅凭能恢复，也不足以证明执行频率与性能对齐。入口为 [storage.py](../python/etazero/storage.py)、[record.cpp](../cpp/src/selfplay/record.cpp)、[runtime.py](../python/etazero/runtime.py)。

尚无同条件下来源与 EtaZero 的完整 selfplay → shuffle → learner → 发布端到端性能对照。本次数值测试不能证明锁竞争、额外记录写入、不同 batch 分布或恢复检查开销不影响等时间比较。也没有据此推出训练效果或棋力收益。

## 十五类共有机制的检查结果

“已核对部分对齐”只覆盖本行所述的机制和测试范围，不是该类别全部配置组合的通过证书。

| 目标类别 | 本次核对结果 | 剩余范围 |
|---|---|---|
| PUCT、展开与回传 | 选点、FPU、价值视角、子树聚合、权重及原生测试通过；共享数学对照没有发现新的确定性偏差。 | U1 并发竞争与合并回传未对齐。 |
| 根探索、目标与落子 | 根温度、策略混合、噪声修剪有来源对照；forced playout、剪枝、LCB 与 Gumbel 根预算有原生回归覆盖。Gumbel 替换根机制属于约定差异。 | 未对所有搜索开关组合做来源端到端分布验证；Gumbel 状态范围见 U7。 |
| 价值与探索修正 | uncertainty、policy optimism、SVB、Go 局部 bias/hash 对照通过；W−L 与修正效用分开。 | U4 外部 pattern/自动历史链路、U5 utility 边界未对齐。 |
| 图、缓存与跨手状态 | 单线程 diamond、catch-up/leak、Go graph hash、重复边界的来源对照通过；NN cache 与搜索统计 cache 分开。Gumbel 使用 tree 为约定差异。 | U7 跨手复用单独评估；真实 GPU 图搜索旧假模型在加载阶段失败，未给出所需的并发覆盖。 |
| 搜索并行与预算 | 普通 PUCT 完成计数及在途任务、最小线程分工、Gumbel 严格配额均有原生测试。 | U1；没有并行行为或吞吐等价证明。 |
| 自对弈采样与权重 | PCR、reduce visits、surprise 权重、随机取整、scale-data-weight、fork 等来源抽取对照通过。 | U3、U4；同一外部语料的候选集合尚不一致。 |
| 监督目标与增强 | 四结果、TD 混合、对手/soft/optimistic policy、Q、variance-time、future-position 及历史 dropout 已核对；MuZero 吸收态与梯度边界按算法适配。 | 全套 GPU 组合未运行；旧三结果 Q 测试须更新，不能把其失败解释为生产损失公式错误。 |
| 网络、metadata 与数值 | 848,640 个来源配置相等；68 个基名逐项前向/梯度检查；真实 metadata、四棋类 CUDA 输出已有验证。 | 一个 depthwise FFN 默认布局断言为来源共有问题；没有覆盖所有预设的 CUDA、compile、AMP 组合。 |
| 优化器与训练时钟 | LR/WD、范数/裁剪、Lookahead、SWA、Muon 等对照通过；reader、内部 subepoch/epoch 跨外层延续，真实 AMP 跳步恢复通过。 | 完整单卡编译组合仍未覆盖；U6 的 DDP 与新增多卡调度按目标暂缓。 |
| 公共数据与 shuffle | 公共字段数值、量化、回放窗口、文件生成器、whole-file subepoch 与两阶段 shuffle 的相关检查通过。 | U8；公共文件契约和额外记录成本未对齐。 |
| 推理与模型交换 | 来源 CUDA/cuDNN 路径、FP32/FP16、真实 batch、四结果与规则 mask、MuZero 三组件/父 latent 复用共 70 项通过。 | 未测全部架构/精度组合与端到端性能，不能由数值相等推断吞吐相等。 |
| 发布与恢复 | 直接接受、gatekeeper 拒绝后的模型选择、learner 状态延续已核对；AZ 跨轮次续训与 MZ 两轮训练/恢复验证通过。 | 运行组织、保存频率及所有中断点的真实 GPU 覆盖仍待核实。 |
| 比赛与 Elo | 独立游戏、执色轮次、认输、W/D/L/NR 计分、gatekeeper 判定与来源 Elo likelihood/prior/协方差/LOS 已核对；相关 CPU 回归通过。 | 本次没有新增完整 arena 棋力实验；等时间研究口径是约定差异，不代表共用统计可豁免。 |
| 交互、棋钟与 HumanSL | 棋钟/ponder、avoid/allow/focus、第二策略及 metadata 路径有实现与已有回归；当前命名网络真实 JSON/GTP 和 HumanSL 模型推理通过。 | U2；旧连续分析/停止夹具未成功进入目标测试，且未完成所有棋钟与 HumanSL 混合选项的来源端到端对照。 |
| 配置、文件与运行组织 | 已检查解析、默认/拒绝路径、语料、raw 数据、checkpoint 和发布入口。 | U3、U4、U5、U8，以及上述运行组织和持久化待核实项。 |

## 实际验证结果

使用 Conda `pytorch`，PyTorch `2.12.0+cu132`，真实 GPU 为 RTX 5080。GPU 检查在可访问宿主设备的环境运行，没有用 CUDA 不可见环境的 skip 冒充通过。

| 检查 | 结果与边界 |
|---|---|
| CMake 构建和 CTest | 构建成功；17/17 原生测试通过。 |
| 全量默认 Python 套件 | 807 通过、424 跳过、1 失败。424 项均为 GPU 开关限制，不能算已验证。 |
| 来源独立对照 | clocks、replay、sampling、sampling weights、search corrections、graph、Go graph、Go bias、SVB、Muon、persistence、forks 通过；另运行公式对照。 |
| Q 量化对照 | 原脚本因缺少当前规则静态库/链接回收选项而链接失败；临时调用补齐链接后，46,140 案例通过。仓库脚本未修改。 |
| 网络配置与前后向 | 全部 848,640 个配置相等。68 个基名中 67 个默认前向/梯度通过，1 个 depthwise FFN 在来源与 EtaZero 均触发同一断言；两边设置 `KATAGO_TRANSFORMER_NHWC=0` 后前向/梯度差均为 0。不是 848,640 个网络都做了前后向。 |
| 专用 CUDA 与规则 | 70/70 通过，涵盖 AZ/MZ、FP32/FP16、真实 batch、Go NR mask 和 HumanSL 来源模型。 |
| learner/gatekeeper 补查 | 4 通过、1 失败；失败为旧 holdout epoch 断言。 |
| 交互/MZ/图搜索/AMP 补查 | 1 通过、3 失败、12 个 fixture error。AMP 分段跳步恢复通过；其余具体原因见下文。 |
| 当前命名网络交互 | 2 个真实 GPU JSON/GTP 测试通过。 |
| 临时补充验证 | holdout 在累计第 6 batch 的内部 epoch 末尾验证一次，16 样本/2 batch；MZ 两轮累计 8 个训练步，三组件权重均更新，完成轮次恢复状态相等。 |

失败需要按根因处理：

- `test_qvalues.py::test_side_q_is_not_gated_by_game_outcome` 仍构造三个结果通道，当前损失访问第四个 NR 通道时越界。它没有检验到声称的 side-Q 行为。
- holdout 用例期待外层 4 步后验证，但 reader 选中的完整文件实际含 6 个 batch。checkpoint 显示 `subepoch_steps=6`、`subepoch_batches=4`、`epoch_end=false`；补跑剩余 2 个 batch 后正确验证。因此应更新测试的内部时钟预期。
- 12 个交互 fixture error 来自给固定命名预设任意设置宽度/深度，或把 AlphaZero 预设名称传给仅接受 `nbt/resnet` 的 MuZero 组件配置。它们在会话运行前就失败。
- MZ pipeline 已完成两轮训练和发布，随后测试访问旧参数名 `representation.stem.weight` 失败。按当前 source-module 参数名补查，representation/dynamics/prediction 均有权重更新，恢复成功。
- 两个真实 GPU graph 用例的假 TorchScript 模型缺少当前必需的 `game_name` 方法，加载阶段即失败，没有到达 graph 数值或故障传播断言。不能计为该路径通过。

原始日志保存在 `/tmp/etazero-v32-target-audit-20261024/`；旁边的 JSON 已保存测试计数、失败详情、来源核验和补充结果。后续应先修复这些测试夹具，再补上当前缺失的来源对照。无需为旧夹具恢复过时的生产接口。

## 完成对齐所需的后续工作

优先处理 U1 的搜索并行行为，以及 U3/U4/U5 涉及实际输入集合和合法取值域的差异；随后闭合 U2/U8 的公共协议与数据契约。U6 的 DDP 和新增多卡调度暂缓；U7 的 graph 不纳入 Gumbel 对齐要求，跨手复用单独评估。运行组织和持久化按实际职责逐项判断。

上述代码差异、验证夹具和性能证据分别闭合后，才适合给出“剩余共有项全部对齐”的结论。当前结论不要求改变既定实验配方、同步外层调度、四结果目标或 MuZero/Gumbel 的必要算法差异。
