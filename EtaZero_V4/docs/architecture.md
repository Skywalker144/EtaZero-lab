# 源码派生架构

## 层次与职责

V4 在 KataGo 上扩展算法和棋类，再由 EtaZero 调度与工具层调用。下表定义目标职责，不表示所有能力已经接入；当前运行范围见下一节。

| 层次 | 职责与边界 |
|---|---|
| KataGo 底座 | 共有 PUCT、网络 backbone、批量推理、采样、训练目标、优化器和数据管线直接维护来源实现；按算法、棋类、四结果与约定删减的实际需要适配，设计文档随实现依据修正。 |
| 算法与棋类扩展 | 接入多棋类、四结果、Gumbel 根/非根搜索，以及 MuZero 的三组件网络、latent 搜索和展开训练；支持 AlphaZero、Gumbel AlphaZero、MuZero、Gumbel MuZero 的有效组合。扩展位于引擎和 learner 的实际调用边界。 |
| EtaZero 调度与工具 | 管理同步逐轮训练、阶段模型切换、产样额度、候选评估与发布、恢复、实验计时，并提供绘图和 Web。算法与性能图只消费原生日志，保留原生输出和统计频率；时间横轴允许使用明确标注的 EtaZero 调度计时。不在展示层重算算法统计或维护另一套训练实现。 |

“外层壳”指调度与用户功能。Gumbel、MuZero、多棋类和四结果需要修改内部职责边界；单卡数据调度、评估任务编排、Elo 调用和绘图则在外围实现，优先使用原生命令、配置、日志与结果。原生改动限于算法／数据适配、约定删除项和确需进程内状态的必要接口。训练图严格读原生日志，缺失指标留空，不为绘图添加引擎计数或从 SGF、训练数据补算；外层计时单独标注。绘图和 Web 应消费本版本运行记录，不依赖旧版本代码来执行训练。

比赛结果汇总、来源 Elo 拟合的调用和绘图位于 `python/etazero/evaluation.py`，直接读取原生逐局结果。外围 `match_cache.py` 管理赛程发布、缓存身份、缺局恢复和增量配额，`match_dispatch.py` 选择驻留窗口。原生 `MatchPairer` 产生配对，`MatchResults` 持有实际 writer 的目录锁、已提交 ID 视图并写出真实逐局结果；`MatchPairBatch` 将每对的局部 ID 映射到同一进程内的执行 ID。模型服务身份仍由原生配置解析与 evaluator 工厂确定。外围不实现配对 RNG、搜索或棋局执行，原生结果接口不决定缓存复用条件或窗口策略。当前实现与恢复语义见 [比赛与 Elo](evaluation.md)。

## 当前执行链

`scripts/run.sh` 设置本版本 Python 路径并进入 `python/etazero/`，`scripts/autoexp.sh` 进入同一外层的实验调度器。配置目录组织、预算、实验臂与 GPU 槽位只在外层处理，见 [配置与启动](configuration.md)。搜索与推理 `.cfg` 原样交给 KataGo；训练与 shuffle 参数保留来源命名。控制器通过参数列表启动同版本原生 selfplay、`python/shuffle.py`、`python/train.py` 以及普通或 MuZero 的来源模型导出入口。子进程逐个完成后才进入下一阶段，learner 的内部 epoch、优化器、LR、Lookahead、SWA、数据生成器及 checkpoint 继续由来源代码管理。

原生程序来自原 CMake 源文件列表。构建适配选择当前 CUDA 架构、定位 Conda/Python 环境中的 CUDA 库以及注册 CTest。搜索统计、数据写入、loss 和推理在来源对应位置扩展四结果通道；共有数学、权重、执行位置与更新频率继续使用来源实现。`native` 命令保留来源参数解析与协议入口。

棋类规则由 `Rules::game` 区分。五子棋与 Hex 通过 `BoardHistory` 分派到 `game/placement.cpp`，共用棋盘存储，但不执行围棋提子、劫或数目裁定。落子维护来源棋盘与历史不变量，搜索继续使用来源的节点、选择、回传和并行实现。非围棋终局不能套用围棋的“终局根仍继续搜索”行为；图与 NN 身份包括棋类，切换棋类时清理局部模式的估值修正。Hex 不使用交换双方目标边的同色对称。

五子棋与 Hex 的输入版本 8 使用 5 个空间特征（有效区域、当前方、对方、按当前方视角分开的黑方禁手提示）和 8 个全局特征（棋规、Renju/Hex 的颜色信息及来源 playout-doubling advantage）。Renju 禁手仍是合法但落子判负的动作；提示不裁减策略域。该输入不编码落子历史，因此 reader 不应用围棋历史 dropout。

模型版本 18 将该输入与来源版本 17 的 backbone、参数角色和适用 heads 连接，移除 pass 分支；Python 策略、Q 和 NPZ writer 的动作数为画布面积。原生共享 NNOutput 仍保留 pass 槽供已有搜索接口使用，但标为非法，没有对应的可训练参数。主价值与 TD 输出保留四通道，NR 在非围棋预测中屏蔽。

Hex 增强使用恒等/180°；混合棋类批次按行选择有效变换，对特征、policy、Q 与未来局面应用同一变换。原生随机推理、根多对称平均和 GTP 的 all 对称枚举同样限制在有效集合中。请求单个无效对称时明确拒绝。

五子棋／Hex 的 SVB 和重复模式保留来源的 5×5／9×9 局部窗口、绝对落点、执色、前一步动作及表的更新与衰减流程。模式身份包含具体棋类，局部窗口只编码棋盘颜色，不带 Go 的 atari／ko 条件；小棋盘按实际边界裁切窗口。Freestyle／Standard 可使用八种几何变换和换色，Renju 仅使用不换色的八种变换。Hex 模式可使用恒等／180°，也可将转置／反转置与换色同时使用，使连接目标边一同变换；这不改变网络增强的同色对称集合。Go 和象棋仍使用各自原有的模式编码。

外部 SGF 模式和自动位置模式按同一棋类身份查询，混合文件不会给其他棋类施加奖励。文件顺序、去重、玩家／手数过滤、衰减与旧文件清理沿用来源流程；非围棋自动位置的 hint 使用真实 `BoardHistory` 合法性，终局、占用点、pass 和矛盾摆子计入无效记录。Renju 禁手按既有“合法落子后判负”规则处理。局部模式、原生比赛和启用这些机制后的短训练检查见 `tests/test_placement_patterns.py`、`tests/test_placement_selfplay.py`、`tests/test_gumbel.py`。

JSON analysis 的落子棋摆子与对称报告保留零气子，不调用 Go 的提子／无气拒绝路径。报告 `thisHash` 包含棋类与执色，`symHash` 只遍历该棋类的有效同色对称；Go 报告哈希仍按来源计算。非围棋终局直接报告规则价值和空候选，若请求 policy，则整个动作域为非法值 `-1`，没有伪造的 NN 原始估值。

当前已验证特征编码、writer 张量、Python 增强与梯度、CUDA FP32/FP16 导出推理和真实模型 PUCT；GameInitializer、对局收尾及外层训练选择已接入，Renju/Hex 的真实两轮训练检查覆盖模型发布与下一轮消费。非围棋外部 SGF/hint 位置已贯通到初始化和自对弈，包含来源 hint fork。模型版本 18 的 Eigen FP32、OpenCL FP32/FP16 与 ONNX Runtime CPU FP32 已有实际数值、规则与对称检查，OpenCL 验证使用 NVIDIA 设备；Metal/CoreML 与 TensorRT 仍拒绝此模型。ONNX 沿用来源图构建器和文件协议，去除非围棋 pass 输出；已验证二进制模型、ONNX／压缩 ONNX 重载与 Eigen 参考对照，其他执行 provider 和 ONNX MuZero 尚未验证／接通。

SGF 的 `GM` 与 `RU` 必须一致：Go 为 1，五子棋族为 4，Hex 为 11；五子棋族还需用规则名区分长连和禁手语义。Hex 没有 `RU` 时可由 `GM[11]` 确定规则。非围棋贴目只能省略或为零，输出不包含 Go 的 `HA`/`KM`，用 `PL` 保留起始行动方。黑方摆子不推断为围棋让子。

`PositionSample` 保存棋类，JSON 仅在非围棋时增加 `game`，保留来源 Go 记录格式。非围棋摆子和 JSON 读入直接重建棋盘链表，不提子或清除零气子；样本去重包含棋类，Renju/Hex 禁止仅交换颜色的变换。SGF 采样、hint 读取和显式起点在各自输入边界检查棋类。Go 仍沿用来源的容错与重新选规则逻辑，落子棋按自身规则重放，不能截断非法记录来伪装有效起点。

落子棋历史初始化识别摆子局面的成五、Hex 连通、满盘和相互矛盾的胜方。Renju 的双三/双四判负需要实际落子历史，不能从任意摆子静态反推；记录中的禁手转移按规则判负。自对弈排除已终局起点，终局没有合法策略时 GTP raw-NN 明确拒绝。hint fork 沿来源流程创建，并保留非围棋起点的原始手数和训练权重。

价值 head 与三组 TD head 输出 W/D/L/NR，搜索始终保存白方视角的 W−L、D、NR，报告时由三者还原 W、L。D 和 NR 不随行动方翻转；writer 只交换 W/L，保留来源的 TD 混合系数、量化与归约。NPZ 中的 D 使用原空闲通道，位置定义在 `cpp/dataio/trainingwrite.h`，不挪动已占用的裁定模式字段。基础结果效用为 W−L，Q 的胜负分量仍监督标量 W−L。

搜索中的 score utility、动态分数中心、根收官分数奖励和 anti-mirror 已移除，包括配置、统计修正、策略干预及自对弈的分数中心预搜索。uncertainty 使用胜负误差，LCB/FPU/虚拟损失使用结果效用范围；保留来源其余公式和执行位置。预测分数的 heads、训练目标、推理缓冲区及报告字段也已删除。围棋规则计分与终局判定继续由 `BoardHistory` 负责。

passing/dame 删减贯通参数、NN 特征与策略后处理、搜索、缓存、GTP 和棋谱生成：不隐藏结束前的历史，不限制 pass 先验，不强制 pass/non-pass 的额外访问，不因对手 pass 而裁减合法候选或预算，不在 GTP 出手前替换着法。根估值缓存和棋谱扩展代价也不再因 pass 被偏好而特殊处理。`friendlyPassOk` 只为这些增强提供提示，因此从规则字段和图/NN 身份移除。真实 pass 转移、终局、计分、encore、pass-alive 裁定及其输入特征保留；GTP 的 `preventCleanupPhase` 仍控制是否对外进入 encore。已终局 Go 位置仍可按来源接口显式分析，搜索中新产生的终局子节点则使用规则结果。

`GameInitializer` 使用 `game` 选择棋类，Go 使用固定 `komiMean` 创建规则，fork 继承原规则贴目；自对弈和比赛不再根据网络估值、让子、policy init、非对称预算或随机噪声改变贴目。让子状态只传递额外黑子数量，贴目由 `BoardHistory::rules` 持有。起始局面工具的自动平衡开关也已移除。

五子棋／Hex 的平衡开局属于棋类算法适配，在 `program/balancedopening.cpp` 共用当前原生模型服务。`GameRunner` 在 policy init 之前调用，外层只配置启用条件；没有复制 GTP 文本输出再重算候选价值。原生棋规与拒绝采样决定接受的前缀，writer 仍从 `startHist` 之后训练。`sampleinitializations` 复用这一流程，普通／MuZero 模型通过现有模型服务接入，展示实际初始化局面并可执行原生诊断搜索。来源、算法、比赛口径与诊断用法见 [平衡开局](openings.md)。

ownership、seki 和逐点计分监督已从主 head、中间 head、loss、writer 和交互展示移除。专用 seki fork、终局额外区域扫描和 seki loss 移动平均随之删除；普通 fork、hint fork 及围棋规则中的双活计分保留。`valueTargetsNCHW` 当前为 8 手与 32 手未来局面两个通道；时距超过终局时取最后棋盘，side/reanalysis 的 outcome mask 和来源 loss 系数不变。全局 C27 仍为终局辅助目标的可用权重，供 optimistic policy、variance-time 和胜负误差预测使用；C34 保留为空闲字段。当前训练格式由实际张量形状检查，不转换旧辅助目标数据。

归属预测的模型层、后端计算和传输缓冲区、`NNOutput` 地图、缓存补取路径与整树平均已删除，普通缓存和对称平均保留。原生模型和 CoreML 转换器拒绝额外的旧模型尾部，ONNX 使用准确的输出集合检查。GTP `final_status_list` 只在实际终局提供规则死活状态，未终局报错；GTP/JSON 的 ownership 参数不再接受。计分 SGF 导入保留终局 pass，要求棋谱实际到达规则终局；未完成记录明确拒绝，不执行归属估计或额外清场。

分数删减贯通 score belief、均值/标准差/lead、TD-score、Q-score 和分数误差；原生 value head 除四结果外只传输 variance-time 与短期胜负误差两个辅助量，空间 future-position 仍为训练专用。NPZ 不再包含 `scoreDistrN`，Q 数组仅包含量化 W−L 与访问次数两个通道；全局目标保留原通道位置，已删除分数的槽位为零。旧模型和目标格式不转换。

optimistic policy 的长期权重保留 `(W + 0.5D + 0.5NR)²`，短期权重保留来源胜负 excess 的 sigmoid、阈值、误差归一化和 detach；删除分数加成。Q 保留胜负交叉熵、平方根访问权重和来源系数。来源 fork 在同一批候选中改用 W−L 排序；hint 预筛、认输和书库扩展代价删除分数条件，保留各自的胜负条件。`genrandomposes.py` 显式使用 `-winloss-rejection-pickiness` 按 W−L 筛选候选，旧目差单位不再适用。GTP `final_score` 只报告实际计分终局，未终局估计已删除。

GTP `kata-raw-nn` 和 JSON analysis 提供独立 D，JSON 根结果还提供 W、L、NR。既有 `winrate` 仍表示 `0.5 + 0.5*(W−L)`，与单独的获胜概率 W 有区别。模型加载器要求四结果输出，不转换旧模型。来源内嵌三结果网络及其旧参考数值不能直接作为 V4 推理验证；当前专项使用本版本导出的四结果模型。

非围棋在规则终局时直接使用胜方或和棋；达到手数上限时不调用 Go 数目计分，不伪造 draw/NR。末状态按同一 PDA 条件评估，形成 TD 的末端 bootstrap；主终局价值及依赖完整结果的 optimistic policy、variance-time、短期误差监督权重为零。来源 policy、Q 和 TD 的适用公式与采样流程保留。

未来局面仍使用来源 8/32 手时距和 loss 系数。Go 保持来源的末棋盘延拓；非围棋真实终局也使用终局棋盘延拓，非围棋截断只保留已经观察到的时距。全局 C75/C76 分别标记不可用的 8/32 手目标，reader 随数据保留，loss 按通道屏蔽。side/reanalysis 的来源 outcome 开关仍先决定该行是否有未来局面监督。

外层配置使用 `game` 和 `algorithm`，支持 AlphaZero、Gumbel AlphaZero、MuZero 和 Gumbel MuZero；Gumbel 非根选择继续由原生配置控制。learner 的输入选择影响新模型的实际输入/策略布局；恢复 checkpoint 不改写其结构，显式选择不一致时拒绝。Go learner 内部 epoch、文件消费、优化器、Lookahead、SWA 等行为不随外层棋类选择改变。

Gumbel 根接入 `cpp/search/searchgumbel.cpp`，复用来源节点；非根可选择来源 PUCT 或 Gumbel 确定性分配。`gumbel.cpp` 负责 completed Q、改进策略、非根分配及根配额调度；训练目标通过独立接口输出完整动作域，报告仍与已分配子节点对齐。默认 SH 出手、可选访问采样和训练策略的区别，以及 fresh tree 和当前组合限制见 [Gumbel 搜索](gumbel.md)。

每轮的 `input_models/` 只引用上轮已发布模型，原生 manager 的目录轮询无法在本轮发现新模型。AlphaZero 首轮为空，使用来源 random evaluator；MuZero 首轮先由同一个 learner 完成 `-initialize-only`，保存零样本 checkpoint 并导出三组件模型。selfplay 完全退出后才 shuffle 和训练；来源 learner 提交 checkpoint 后才导出，模型目录在普通权重或 SWA 的导出完成后原子发布，MuZero 以包含 manifest 和三个组件的整个目录为发布单位。原始数据和旧模型不覆盖。

控制器的 run 目录锁由阶段子进程继承，防止控制器退出后仍在运行的 writer 被并发接管。轮次恢复使用已发布 shard、来源 epoch checkpoint 和原子目录发布；已提交更新不重复，未提交工作重做。具体边界见 [调度与恢复](controller.md)。`source.tar.gz` 在创建 run 时记录代码，包括尚未提交的新文件；不将源码哈希作为读取和恢复的运行门槛。

## 代码归属

| 职责 | 权威实现 | 接入原则 |
|---|---|---|
| 棋规与历史 | `cpp/game/` | 保留 Go Board/BoardHistory；五子棋与 Hex 在同一历史接口中使用独立的合法性与终局规则。象棋完整规则状态已连接来源 BoardHistory、PUCT/Gumbel Search、GameRunner、SGF/hint 和训练数据 writer，原生 selfplay 与四种算法的逐轮调度已接通；比赛与 gatekeeper 已接入，局部模式与 SVB 已接入，普通 PUCT／Gumbel Search 支持无 Go metadata 的次级 HumanSL，原生普通／MuZero JSON analysis 和 GTP 已接入；普通／MuZero 次级网络已接入搜索及 JSON／GTP 入口，见 [象棋](xiangqi.md)。 |
| 选点、回传、图与结果报告 | `cpp/search/` | 保留来源公共调用链；分别提供算法定义的训练目标、实际落子、根统计和 PV；报告不反向定义训练策略。 |
| NN 特征、模型加载、组批与 CUDA | `cpp/neuralnet/` | 在模型真实输入和动作布局边界扩展；保留来源批处理、缓存和生命周期。 |
| 自对弈采样与记录 | `cpp/program/`、`cpp/dataio/` | 保留采样、加权、量化的生成/消费关系；新增棋类改变实际必需的字段。 |
| backbone、heads、loss、优化器 | `python/katago/train/`、`python/muon/` | 直接维护来源实现；四结果与棋类 heads 的改动同时贯通 writer、训练和导出。 |
| shuffle 与 learner 时钟 | `python/shuffle.py`、`python/train.py`、`python/katago/utils/` | 外层暂停不把算法时钟重新定义成 EtaZero 轮次。 |
| EtaZero 阶段编排 | `python/etazero/` | 处理产样规划、阶段边界、模型发布和运行证据；不复制来源内部优化步骤。 |

目录名称和 `katago` Python 包名保留来源对应关系；不维护一套只供展示的来源副本和另一套实际运行的引擎。V3.4 作为独立参考版本，不通过 import、链接或运行路径隐式依赖其实现。

独立 `benchmark` 和 GTP `kata-benchmark` 复用来源搜索计时循环，模型生命周期通过现有 `SelfplayModel` 同时接入普通／MuZero 服务。Go 内置局面与经验调参保留；其他棋类使用合法轨迹或指定 SGF，新算法与非围棋按实测吞吐比较线程。`benchmarknn` 的普通前向与 MuZero initial／recurrent 共用来源计时循环，各自直接调用原有后端，输入准备和加载排除在计时外；详见 [搜索与推理 benchmark](benchmark.md)。其他命令的适配仍分别核对。

原生 `evalsgf` 通过同一 `SelfplayModel` 和 `AsyncBot` 接入普通／MuZero 主次模型，复用棋谱重放、搜索与报告流程。额外着法、限制、分支和策略展示按实际动作域处理；输入导出经共有 `NNInputs::fillRow` 分派，不再假定围棋特征或正方形画布。具体参数与验证范围见 [原生棋谱诊断](evalsgf.md)。

## 后续接入边界

Web 会话使用原生 GTP 保存棋局，`kata-get-state` 导出规则局面和报告视角，`kata-get-analysis` 及分析命令的 `json true` 复用来源 `Search::getAnalysisJson`。`python/etazero/gtp.py` 负责常驻进程、响应分帧与取消；`web_session.py` 保存来源棋盘快照并执行会话动作，`web_app.py` 串行组织人机回合、持续分析、目录刷新和模型切换。它们不实现棋规或搜索，不把缺少的来源字段补造为算法统计。发布模型目录已接入控制器；共享 `web/http.py` 独立于版本和引擎格式，直接传输控制器状态。`web/native_server.py` 和独立原生字段页面已接通启动路由；SGF 通过原生 `loadsgf` 与只读历史快照接入，平衡开局与按需网络诊断已通过原生命令接入，象棋逐手动画仍待接入，详见 [Web 与原生会话](web.md)。

以下区分当前基础与仍需接入的部分。

1. **多棋类与动作。** 棋类边界包含合法动作、状态转移、终局、特征、有效对称和图身份。当前五子棋与 Hex 的规则、网络输入/动作、writer、SGF/hint 起点、selfplay 与逐轮训练已贯通。Go 内部继续使用 `Loc`，公共策略域不能继续假定 `canvas²+pass`；象棋要贯通动作索引、writer、网络、缓存和协议。象棋规则与原生 BoardHistory 完整历史、动作／特征、Python 联合策略 head、来源 loss、原生导出、CUDA 联合策略读出、evaluator、PUCT/Gumbel 搜索、GameRunner、SGF/hint、原生 writer、来源 shuffle/reader 与 learner 的当前边界见 [象棋](xiangqi.md)；命令级 selfplay 与四种算法的逐轮训练已贯通，普通／MuZero 混合比赛、来源 Elo 与 gatekeeper 已接入；局部模式与 SVB 已接入，普通 PUCT／Gumbel Search 支持无 Go metadata 的次级 HumanSL，原生普通／MuZero JSON analysis 和 GTP 已接入；普通／MuZero 次级网络已接入搜索及 JSON／GTP 入口。
2. **Gumbel。** 根配额、sequential halving、completed Q、改进策略及可选 PUCT/Gumbel 非根已贯通；来源 weightless root focus、普通／MuZero 次级 HumanSL、非根 visit-cap 快照及根对称剪枝已接入，保留严格根配额。HumanSL 根 weightful 探索和根 visit-cap 快照与配额冲突，明确拒绝。跨手及同根续搜按约定使用 fresh tree；直接继承旧访问与固定 Mctx 的阶段、预算及 Q 计数不兼容，具体原因与影响见 [Gumbel 搜索](gumbel.md#树复用边界)。Gumbel 树复用不再作为未完成项，普通 PUCT 复用保持来源行为。
3. **MuZero。** 三组件使用来源 `Model` 与适用 heads/loss；latent 搜索在来源 `Search` 内保留选点和回传，只替换真实状态转移及其不适用的机制。模型、轨迹、learner 与同步逐轮发布已有闭环，具体边界见 [MuZero](muzero.md)。象棋已接入三组件网络、联合动作编码、导出、原生 CUDA／Eigen／OpenCL 推理、局面适配及 PUCT/Gumbel latent 搜索，轨迹 writer/shuffle/reader、正式 learner、GameRunner/selfplay 与逐轮调度已接通，普通／SWA 发布及 gatekeeper 恢复共用现有流程。原生 JSON analysis 与 GTP 已接入全部棋类；MuZero 次级 HumanSL 已接入搜索及 JSON／GTP；其余原生工具仍需按职责适配。
4. **完整调度。** 累计时间预算已接入完整轮次边界，扣除实际 learner 编译耗时，计时缺失时明确失效并按已观测墙钟预算；来源 gatekeeper 通过完整模型副本比赛接入，拒绝候选只保留原 accepted，不回滚 learner。replay-ratio 按完整来源 epoch 分配额度、实际消费扣款，阶段恢复按实际提交点继续。绘图已消费本版本的提交与来源统计，独立比赛已复用来源配对器和 Elo，支持混合模型与逐局恢复，见 [比赛与 Elo](evaluation.md)；历史模型发现和独立 arena 使用提交记录选择实测训练时间点；多实验臂 GPU 槽位调度与来源 learner 共享初始化已接入，训练后自动 Elo、固定参考与原生逐对结果复用已接入；独立 match／arena 与 autoelo 跨配对评估已支持模型驻留窗口，Web 的原生会话、页面与 SGF 已接入，平衡开局与按需网络诊断已通过原生命令接入，象棋逐手动画仍待接入；保留来源 learner 时钟，不用每轮重新初始化代替恢复。

验收沿完整调用链进行：来源对照、手算边界、多线程/真实 GPU、采样到更新到加载的闭环，以及独立性能测量分别提供证据。直接继承来源源码可以减少重写偏差；新增适配仍需验证。
