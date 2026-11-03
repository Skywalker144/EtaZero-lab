# 中国象棋的接入边界

象棋的棋规状态、原生 `BoardHistory`、固定动作域、特征编码、Python 网络／loss、原生模型导出与 CUDA evaluator，以及来源 PUCT／Gumbel 搜索已实现。原生 `selfplay` 已通过 `GameRunner`、SGF/hint 起点、writer、来源 shuffle/reader 和 learner 接入 EtaZero 逐轮调度，支持 AlphaZero、Gumbel AlphaZero、MuZero 和 Gumbel MuZero。原生 `match` 和 `gatekeeper` 共用完整对局状态，支持独立比赛、来源 Elo 与训练晋级。普通 PUCT／Gumbel `Search` 支持版本 19、无 Go metadata 的次级 HumanSL；原生 JSON `analysis` 支持普通／MuZero 网络的 PUCT／Gumbel，以及普通 PUCT／Gumbel 次级 HumanSL；原生 GTP 支持相同的网络／搜索组合；**普通／MuZero 次级网络已接入搜索及 JSON／GTP**，MuZero 主模型搭配普通次级网络时限于真实根。无需模型的 `samplesgfs` 和 `viewstartposes` 可读写象棋记录。

## 棋规状态

[xiangqi.cpp](../cpp/game/xiangqi.cpp) 使用固定 Pikafish 的 `Position`、`MoveList<ALL>` 和 `rule_judge`。来源 commit、已有适配和许可证位于 [reference_sources.json](../reference_sources.json)、[Pikafish 许可证](../licenses/Pikafish.txt)。只编译棋规相关组件，不引入 Pikafish 的 alpha-beta 搜索或评估网络。C++20 要求限定在该规则库，KataGo 的其他模块保留原编译标准。

`Xiangqi::State` 拥有完整 `StateInfo` 历史，复制时重新绑定前驱指针。合法走法与将军、照面、蹩马腿、塞象眼、炮架、宫界、河界，以及重复、长将、长捉和 60 回合裁决直接使用来源实现。没有合法走法判负，包括未被将军的困毙；普通重复、无进展和来源材料裁决产生 D，不产生 NR。

`BoardHistory` 从完整 `State` 构建，复制时拥有独立的当前规则状态，并共享不可变的起始快照。`copyToInitial` 与动作重放保留起点之前的重复、将捉及特征历史；不会从 FEN 重建来代替完整历史。原生手数、最近棋盘、动作记录、下一方与终局结果随真实状态更新，非围棋结果不设置计分或 NR 标记。只给出颜色棋盘的构造入口不能恢复棋子种类，会明确拒绝象棋。

现有 `Board` 保持颜色占位视图，按完整棋子状态重建，不调用围棋提子；其 `pos_hash` 仍仅表示颜色占位，不能作为象棋局面身份。象棋的 `GraphHash` 使用完整规则历史，不受 Go 重复边界参数重置；重算直接读取该历史哈希。`koHashHistory` 只记录本 `BoardHistory` 起点以来的位置出现次数，使用规则库的 64 位位置键；实际重复裁决读取 `State` 的完整历史。棋子类型不同或重复前史不同，即使颜色视图相同，也具有不同的图与 NN 身份。

原生 `Player` 类型中，`P_BLACK` 在此表示先手红方，`P_WHITE` 表示后手黑方；棋子正负号分别表示红、黑。七种棋子的编号沿用 Pikafish。FEN 使用来源格式，载入时不能恢复此前的循环历史；FEN 重载与完整状态复制是不同操作。初始规范化 FEN 与每个后续动作共同形成 128 位历史哈希；它保留全历史顺序，不套用 Go 图搜索的有界历史重置。该哈希仍有哈希碰撞的理论可能，不是精确序列存储的替代品；实际裁决始终读取完整规则历史。

## 动作与输入

[Python 布局](../python/katago/game/xiangqi.py) 定义固定 2086 个联合走法：所有横直线走法、马步，以及双方合法士位和象位之间的走法。当前位置的阻挡、自将和终局等仍由规则状态筛选。生成的 [C++ 布局](../cpp/game/xiangqilayout.h) 使用相同顺序，不在运行时依赖其他 EtaZero 版本。修改布局后运行该 Python 文件更新表。

动作坐标以红方本位的 9×10 棋盘表示，紧凑文本例如 `b0c2`。转换到黑方视角时绕物理棋盘旋转 180°，可再作左右镜像；变换同时作用于起点与终点，是可逆的动作排列。变换使用实际九路十行，不把 10×10 画布的 padding 当成棋盘。

原生 `Move` 的 `Loc` 存储通过 `actionToLoc`／`locToAction` 保存联合动作，保留来源的 NULL 和 pass 哨兵。这些值不能索引 `Board` 的逐点数组，也不能交给围棋坐标转换。来源搜索使用对应的动作映射，SGF 与 hint 使用下述完整状态表示；仅包含颜色棋盘的象棋摆子不能恢复棋子种类，会明确拒绝。

`fillFeatures` 输出 float32 的 113 个空间通道和 8 个全局量，支持 NCHW/NHWC。空间通道是有效区域 mask，以及最近八帧、每帧己方和对方各七类棋子；缺失帧与 padding 为零。所有历史帧使用同一个指定视角和镜像。全局量依次为红方视角、无进展计数／120、当前局面此前重复次数／2、PDA 是否启用、带符号 PDA／2、可用帧数／8、当前方被将军和总半回合数／200。PDA 符号由调用者按指定价值视角提供。`maxHistory` 限制额外的过去帧，当前帧始终保留；缺省最多七帧历史。该限制会改变空间历史与可用帧数，不改变用于真实裁决的完整规则历史、重复计数或无进展计数。

## 网络与监督

Python 模型版本 19 使用来源 `Model` 的 backbone、归一化、初始化、全局 pooling 和 heads。`PolicyHead` 的最终卷积为每个策略／Q head 输出 50 个位移通道，再按起点与位移 gather 为 2086 个联合 logits；不将独立起点和终点分布相乘。六项策略和可选纯 W−L Q、主 head 与 intermediate head 共用这一动作顺序，没有 pass 分支。

`Metrics` 保留来源的目标公式、系数和归约，只将策略长度与静态动作域适配为联合走法。软策略的静态域不等于实际合法走法 mask，与来源按静态棋盘域平滑目标的职责一致。价值及三个 TD 仍输出 W/D/L/NR，NR 被屏蔽且不接收梯度；未来局面使用已有的逐时距可用性 mask。Go 元数据和 Go 专用特征选项不能用于该输入布局。

目前已验证规则裁决、完整历史复制、2086 动作与镜像、padding、NCHW/NHWC、联合策略读出，以及来源 loss 的梯度路径。CUDA FP16 eager 与 Inductor 检查覆盖实际参数更新、默认 GradScaler 的溢出跳过／回退和权重保存加载。

## 数据与独立 learner

原生 `TrainingDataWriter` 的输入版本 9 直接重放完整 `BoardHistory`，使用相同的主线／side／reanalysis 采样、权重与随机取整。NPZ 保留来源数组：113 通道空间输入按位打包，8 个全局输入；策略和 Q 均为 `[N,2,2086]`，Q 继续使用来源的 W−L 量化和访问数。两路策略分别记录当前手与下一手，二者都转换到**当前行输入**的玩家朝向，下一手不能独立切换朝向。原生搜索中的联合动作仍使用固定红方坐标。

W/D/L/NR、三组 TD、variance-time、surprise 和 outcome 权重沿用已有公式；TD 的面积使用实际 9×10。未来局面仍为 8/32 手的己方／对方颜色占位，和输入使用相同朝向，padding 为零。真实终局按来源方式延拓终局棋盘；截断屏蔽不可用时距与终局监督，side 和未采用实际后续的 reanalysis 不伪造未来局面或下一手策略。

`shuffle.py` 原样处理完整行，不另设象棋打乱流程。来源 reader 对输入、两路策略、Q 与未来局面执行同一次左右镜像，变换只覆盖物理 9×10，不翻转画布 padding。历史遮蔽沿用来源逐层 0.98 的保留概率，适配为七帧过去的完整棋子特征；当前帧始终保留，可用帧数同步减少，重复与无进展等真实规则量不被遮蔽。采样继续使用来源 Torch RNG，由 learner 的现有 checkpoint 恢复。

`train.py -input-features xiangqi` 为新模型选择版本 19；恢复时从 checkpoint 推断，显式指定其他输入族会拒绝。普通网络结构仍由 `model-kind` 选择，来源 epoch、文件消费、optimizer、Lookahead、SWA、AMP 与恢复逻辑保持原有职责。外层 `game=xiangqi` 自动选择相同输入族，消费原生 selfplay 的合法 NPZ。Go metadata 仍明确拒绝；MuZero 的网络配置与轨迹训练见下文。

`tests/test_xiangqi_data.py` 使用原生规则产生完整记录，对照双方朝向、历史帧、联合目标、TD 几何权重、side/reanalysis、终局与截断、10/12 画布及有效镜像。实际 shuffle 前后逐行内容一致；真实 CUDA FP16 检查覆盖独立 learner 更新、重启更新、普通/SWA 导出及导出后的原生 Gumbel 搜索。`tests/test_xiangqi_play.py` 进一步消费实际 `GameRunner` 的搜索目标和采样记录，完成原生自对弈产样到 shuffle、FP16 learner 更新、导出和再次搜索的检查。

## 原生对局与采样

`GameInitializer` 使用原生 `bSizesXY = 9-10` 与 `bSizeRelProbs`，从象棋标准初始状态创建完整历史；其他尺寸和 Go 让子明确拒绝。`GameRunner` 继续使用原有主线、policy init、cheap search、重分析、surprise 加权、side/tree-side、early/late fork 和目标随机取整流程。起点随机推进的长度与 TD 面积均使用实际 90 格，联合动作数量不代替棋盘面积。

policy init 和随机策略采样遍历完整 2086 动作，随机合法走法从规则状态取得。fork 的候选抽样仍有放回，温度、分支概率与候选价值排序沿用同版本共有实现。策略目标、Q 与原始策略熵覆盖整个联合动作域。日志使用联合着法文本和完整 FEN，避免把动作编号当成棋盘点。

fork 重放从 `BoardHistory::copyToInitial()` 恢复完整起始快照，保留快照之前的重复、长将与长捉历史。hint 匹配使用完整规则历史身份，hint fork 保留训练权重和实际起点；从 SGF／`PositionSample` 文件导入的起点同样恢复完整状态。原生对局的实际终局由棋规裁决；手数截断沿用非围棋末状态 bootstrap 与目标掩码。

原生夹具覆盖 PUCT、Gumbel 根配合 PUCT/Gumbel 非根、policy init、side/tree-side、全量 cheap-search 重分析、两种 outcome 选项、early/late fork、保留长将前史的 hint fork，以及 60 回合终局和手数截断。真实 CUDA FP16 检查逐行比对完整历史特征与联合策略、复核原始策略熵，检查实际对局的 SGF 往返，并将实际产样交给来源 shuffle 和 learner。

## selfplay 命令与逐轮调度

原生 `selfplay` 使用 `game=xiangqi`、`bSizesXY=9-10`，不能同时配置 `bSizes` 或 `allowRectangleProb`。写入画布 `dataBoardLen` 至少为 10，输入版本缺省为 9，显式选择其他输入族会拒绝。普通模型目录为空时使用来源 random evaluator；真实模型必须为版本 19。`useMuZero=true` 且模型目录为空时使用 MuZero 随机 evaluator，保留完整联合动作域和 latent 搜索语义，不执行真实 NN 推理；已有三组件 bundle 时使用真实模型，其画布须等于 `dataBoardLen`。来源模型发现、组批、manager、写入线程和停止流程保持同一调用链。普通模型默认推理缓冲区按实际 9×10 棋盘分配，MuZero 使用组件画布，写入画布均为方形；缓冲区选项沿用原生配置。

外层配置仍使用 [配置目录](configuration.md) 的 `run.cfg`、`selfplay.cfg`、`train.cfg` 和 `shuffle.cfg`。`[run]` 选择 `game=xiangqi` 和四种算法之一，`train.cfg` 的 `pos-len` 至少为 10；控制器以该画布设置原生 writer。普通网络使用 `model-kind`，MuZero 使用 `muzero-config` 和展开步数，零样本初始化与后续训练都由正式 learner 执行。现有 `run.sh`、`autoexp.sh`、共享初始化、普通／SWA 发布和轮次恢复使用同一流程。普通网络支持原生 `subtreeValueBiasFactor`、`avoidRepeatedPatternUtility` 和外部模式表；MuZero 的这些真实棋盘选项需关闭，同时关闭真实棋盘 graph，Gumbel 组合限制见 [Gumbel](gumbel.md)。

`tests/test_xiangqi_controller.py` 验证正式命令的随机冷启动、两种画布、三种搜索组合、SGF/hint 导入后的实际终局、脚本入口与共享初始化。真实 CUDA FP16 两轮检查覆盖 AlphaZero 普通权重与 Gumbel AlphaZero SWA 权重的发布、重启和下一轮真实模型消费，以及恢复到已完成轮数时不重复训练。这些小规模检查验证执行链，不构成棋力或吞吐结论。

## 比赛与晋级

原生 `match` 和 `gatekeeper` 以同一个 `GameRunner` 创建双方搜索，保留各自的完整象棋历史，使用来源 `MatchPairer`、模型服务与线程生命周期。版本 19 只在匹配的 `game=xiangqi` 下加载；比赛的逐 bot 搜索选项和模型去重继续使用来源索引。配置使用 `bSizesXY=9-10`，其他搜索组合限制与上文相同。带 `hintLoc` 的起点沿用来源 hint 搜索的清树前提，需要 `clearBotBeforeSearch=true`；普通 SGF 或无 hint 的起点继续使用普通树复用行为。

比赛可混用普通模型和 MuZero bundle，逐 bot 的 `useMuZero`、根／非根搜索选项由原生配置解析。同一路径的 bundle 只加载一次，可供多个搜索共享。MuZero 每次真实落子后清理 latent 树；gatekeeper 使用完整三组件目录评估与移动候选，外层发布模型保持不可变。

比赛记录使用上述完整状态 SGF。原生 B／W 名称和结果继续分别对应红／黑，D 表示真实规则和棋。持久化比赛与 gatekeeper 不将手数截断裁定为 D／NR：没有规则结果会报错，候选模型不会因此晋级。配对、gatekeeper 的得分与提前结束公式均沿用共有实现。逐局恢复、完成后的结果复用及来源 Elo 口径见 [比赛与 Elo](evaluation.md)。

外层 `gatekeeper_config` 可指向独立的象棋原生配置。拒绝候选时保留 learner checkpoint，后续产样继续使用已接受模型；已提交的晋级决定在恢复时不会重新对局。`tests/test_xiangqi_match.py` 覆盖完整长将前史、真实和棋、源配对、逐局恢复、截断拒绝和外部模式表的原生配置索引；CUDA FP16 检查覆盖实际版本 19 比赛、模型去重、Elo、接受／拒绝、晋级提交后的恢复及下一轮模型消费。

## 棋谱与 hint 文件

SGF 使用 [FF[4] 的象棋标识 `GM[7]`](https://www.red-bean.com/sgf/properties.html#GM) 和 `SZ[9:10]`。V4 的棋类扩展约定：`B` 对应内部 `P_BLACK`（红方），`W` 对应 `P_WHITE`（黑方），联合着法直接写作 `B[b0c2]`；坐标与原生动作表一致。它不是对所有外部象棋 SGF 方言的兼容承诺。`RU[xiangqi]` 可明确规则；`GM[7]` 未指定 RU 时使用本版本固定棋规，GM/RU 冲突会拒绝。

根节点的 `XQFEN` 保存规则历史最初的 FEN，`XQHIST` 的各个值保存起始快照之前的完整联合着法，`XQTURN` 保存 `BoardHistory` 的起始手数。后续 B/W 节点对应实际 `moveHistory`，来源的对局结果、搜索注释和起点索引继续保留。省略 XQFEN 表示标准初始局面，省略 XQHIST 表示没有前史；只保存当前 FEN 不能替代带前史的记录。完整状态序列从既有 `StateInfo` 链在文件边界提取，搜索状态不另存一份重复动作日志。

来源的树遍历可读取分支及带 XQFEN 的中间局面重设。颜色摆子 AB/AW/AE 和没有完整状态支撑的换方 PL 会拒绝；联合动作不访问 Go 的劫禁逐点数组。象棋非法动作即使在 tolerant 模式也会报错，不通过丢弃历史来修复。`CompactSgf` 沿用普通连续棋谱的限制，不展开中间局面重设。

hint JSON 使用 `game=xiangqi`、`xiangqiFen` 与 `xiangqiHistory` 表示近期走法之前的完整状态，颜色占位 `board` 不作为第二份输入源。来源 `moveLocs`／`movePlas` 仍保存至多五手近期走法，`hintLoc` 使用同一联合坐标或 `null`，权重、训练权重、手数与 metadata 继续使用原字段。读取后重建完整规则状态及颜色视图；`previousPosition` 只回退近期走法，不丢失更早的裁决历史。象棋不执行仅翻转颜色的增强。

`GameInitializer` 的原生 `startPosesFromSgfDir` 和 `hintPosesDir` 已使用这些记录，保留来源选择概率、去重、终局过滤和权重。`samplesgfs` 的普通采样与普通／MuZero value-fluctuation 加权均保留完整前史；`viewstartposes`、`trystartposes` 和 `dataminesgfs` 也已接入象棋模型与搜索，用法见 [原生棋谱诊断](evalsgf.md)。`tests/test_xiangqi_sgf.py` 与原生夹具验证捕获后的历史恢复、长将前史、SGF/hint 往返、分支、局面重设、非法记录、去重和文件起点加载；其他来源工具仍按各自入口限制，不能由共享了解析器推断为全部完成适配。

## 原生导出与推理

原生 [`evalsgf`](evalsgf.md) 可诊断保留完整前史的象棋 SGF，支持普通／MuZero PUCT／Gumbel 及次级网络。额外着法、hint、avoid 和分支均使用联合动作，策略与 Dirichlet 形状逐动作输出；输入导出保留 113 通道历史与物理棋盘 padding。

`export_model_pytorch.py` 使用原生二进制层 writer，导出普通、乐观以及可选 Q head 的完整 50 通道位移块。描述器将逻辑策略通道数（2 或 3）与最终卷积通道数（100 或 150）分开检查，不写入 pass 分支。原生层优化、scale8 及输出缩放仍由来源描述器处理。版本 19 的 MuZero 三组件导出和加载复用相同格式，保留原生 trunk 与联合策略 head，见下述边界。

CUDA 继续使用来源 trunk、heads、组批缓冲区和计算流。输入已经按指定玩家视角编码；后端仅接受恒等或物理九路棋盘上的左右镜像，padding 不参与镜像。策略读出按起点／位移 gather 为 2086 项，逆映射镜像，并沿用来源普通／乐观 logits 混合公式。当前 gather 在主机端读取原生卷积输出，尚无端到端性能结论。

`NNOutput` 保留原有棋盘点数组规模，只为联合动作输出另行分配存储；复制、赋值、平均和噪声副本覆盖实际存储长度，合法性不一致时沿用来源选择第一个结果的行为。其他棋类不会因为象棋动作域扩大到 2086 项而分配同样大小的策略数组。

113 通道输入的部分 cuDNN 跨布局卷积不受支持，因此象棋在第一层前复用来源 NCHW/NHWC 转换 kernel；已有 CUDA 模型仍执行原来的卷积路径。测试覆盖两种输入布局与两种内部布局、FP32/FP16、10×10／12×12 画布、镜像、完整联合 logits 与重复使用缓冲区时的批大小变化。默认 FP32 允许来源 cuDNN 的 TF32 数值路径，同时使用测试进程的 `NVIDIA_TF32_OVERRIDE=0` 作严格数值对照；不改变正式运行默认值。

Eigen CPU 后端使用来源 FP32 NHWC 网络层；策略缓冲区按实际 100／150 个卷积通道分配，读出复用同一联合动作词表与物理镜像规则。`tests/test_eigen_backend.py` 使用与 CUDA 相同的 Python 数值参考，检查全部动作、乐观策略、四结果与辅助输出、padding、批大小变化、普通／Transformer evaluator 的缓存与并行请求，以及 PUCT／Gumbel 和次级网络搜索。Eigen 也支持 MuZero initial／recurrent，复用同一联合动作与 latent 语义；数值、并发、局面及 PUCT／Gumbel 检查见 `tests/test_muzero_eigen.py`。

OpenCL 后端保留来源 NCHW 网络层与 kernel，策略缓冲区按实际 100／150 个通道分配；FP16 读回先按来源布局移除空间 padding，再按相同的物理镜像与联合动作表映射全部 2086 项。`tests/test_opencl_backend.py` 使用现有 PyTorch 数值参考，检查普通／乐观策略、四结果、辅助输出、不同画布与批大小、普通／Transformer、缓存与并行组批，以及 PUCT／Gumbel 和次级网络搜索。实际 NVIDIA OpenCL 检查覆盖 FP32、FP16 存储和来源 WMMA 路径，不构成其他设备或性能结论。MuZero initial／recurrent 也已接入，设备 latent、两张动作平面和联合策略沿用同一象棋语义；数值、并发、搜索与两轮训练检查见 `tests/test_muzero_opencl.py`。

原生后端返回 logits，后续由来源 evaluator 完成归一化与视角处理。`getSupportedRules` 不把版本 19 当作 Go／落子棋，避免进入错误的特征与策略路径。ONNX Runtime 的普通网络已接入来源图构建器：不生成 pass 输出，空间策略为 100／150 通道，读出时映射到 2086 个联合动作。CPU FP32 已验证物理棋盘镜像、padding、变动／固定 batch、scale8、Transformer 两种内部布局、缓存、并行搜索、HumanSL、ONNX 重载与两轮普通／SWA 训练发布；MuZero 尚未接入 ONNX。ROCm 的共享读出代码已适配，但未在相应设备验证；Metal 和 TensorRT 明确拒绝象棋模型。

## MuZero 网络与 latent 搜索

象棋已接入三组件网络、逐步 source loss、原生二进制导出／描述器及原生 CUDA initial/recurrent。representation 使用 113 个历史空间特征与 8 个全局特征；dynamics 使用根的固定坐标，把一个联合动作编码为起点、终点两个平面；prediction 使用相同的版本 19 联合策略／Q 和四结果 heads。三组件复用来源网络层、参数角色、归一化与优化路径，不使用真实后继棋盘裁定 latent 动作是否合法。

象棋 `PositionEvaluator` 与来源 `Search` 的 PUCT/Gumbel latent 路径已接入。根缓存保留完整规则历史身份，根策略仅包含真实合法动作；递归使用完整 2086 联合动作，不访问真实后继棋盘。根执色与物理左右镜像随设备状态保留，递归动作和 policy 始终使用根坐标，价值视角随 latent 执色切换。来源根噪声、对称平均、hint 与 PUCT/Gumbel 的 weightless focus、uncertainty、回传和多线程发布沿用已有实现；真实落子清理旧 latent 树。网络语义与接口见 [MuZero](muzero.md)。

完整轨迹 writer、来源 shuffle、展开 reader 与正式 learner 已接入。原始目标保留逐行行动方视角，展开时将动作和空间目标转到固定根坐标；真实终局、截断 bootstrap、无后续 side 和未采为根的中间位置保持各自语义。learner 复用已有初始化、训练时钟、optimizer、Lookahead、SWA、AMP、编译与恢复流程，普通／SWA 三组件导出可由原生 CUDA 重新加载。

`GameRunner`、selfplay 命令与逐轮 MuZero 调度共用已接入的完整链路。policy init、cheap 后重分析、side/fork 与树内 side 均以真实象棋状态记录目标；树内 side 只记录该真实位置的合法动作。外层初始化、逐轮 shuffle/learner、普通／SWA 整包发布及下一轮加载使用同一控制器；比赛和 gatekeeper 也通过原生模型服务接入。

`tests/test_muzero_model.py` 检查双平面编码、重复联合动作、展开 loss／梯度、NR 屏蔽和 AMP 更新后保存／恢复。`tests/test_muzero_export.py` 检查三个版本 19 组件及错配边界；`tests/test_muzero_cuda.py` 对照 PyTorch 的完整策略、价值与 latent，覆盖 FP32/FP16、10/12 画布、NCHW/NHWC、Transformer warmup、不同批次保留的状态和组件生命周期。实际 GameRunner/selfplay、两轮 CUDA FP16 训练与独立导出对照分别见 `tests/test_muzero_play.py` 和 `tests/test_muzero_controller.py`；混合模型比赛、完整长将前史与三轮晋级恢复见 `tests/test_xiangqi_match.py`。这些工程检查不构成棋力或吞吐结论。

`tests/test_muzero_position.py` 比较红黑根、两种镜像、连续递归的完整策略与四结果，检查根缓存、平均策略保留单一 latent、完整长将前史和真实终局拒绝。`tests/test_muzero_search.py` 运行单／四线程 PUCT、Gumbel 根及 Gumbel 非根，逐节点核对递归输出和来源 edge catch-up 回传，覆盖重复动作长链、真实胜负／和棋、根噪声与新一手重新编码。FP32/FP16 使用真实 CUDA；这些检查不构成棋力结论。

`tests/test_muzero_data.py` 逐字比较普通写入与开启轨迹后的原始行，核对源采样随机流、完整轨迹、独立 TD 几何级数参考、shuffle 和三组件更新。`tests/test_xiangqi_muzero_data.py` 覆盖固定根坐标、长将胜负／重复和棋／截断、吸收态棋盘、左右镜像、联合动作域和 reader RNG 恢复。`tests/test_muzero_learner.py` 使用原生轨迹运行正式 learner 的初始化、训练与续训，覆盖 CPU 的 SGD/BN/BatchRenorm，以及真实 CUDA 的编译、FP16、AdamW、普通／SWA 导出和原生 initial/recurrent 数值对照。

## 局面 evaluator

`NNEvaluator::evaluate(const Xiangqi::State&, …)` 接收完整规则状态，使用原有请求队列、跨线程组批、缓存表、计数与停止流程。特征按当前方编码；后端先逆映射左右镜像，evaluator 再把后手的策略旋转回固定红方坐标。结果策略按真实合法走法筛选并沿用来源温度与 softmax，非法项为 −1。价值继续使用来源四结果后处理，保存 `P_WHITE`（象棋黑方）视角，并屏蔽 NR；终局没有合法策略，直接请求 evaluator 会明确报错，搜索应使用棋规结果。

原生 `BoardHistory` evaluator 入口分派到同一个状态 evaluator，共享缓存与组批；`NNInputs::fillRow` 的输入版本 9 直接读取完整象棋状态，历史帧不受颜色视图的最近棋盘数量限制。模型版本 19 只匹配象棋，版本 18 只匹配四种落子棋。已认输或规则终局的历史不进入策略推理，当前方不一致也会报错。

NN 缓存以完整象棋历史及实际可见帧数为基础，再使用来源 PDA、策略温度与乐观度的哈希混合。隐藏特征历史不会把重复裁决不同的状态合并。对称仍按来源约定不进入缓存身份；需要指定对称的新结果时绕过缓存。多对称平均复用同一采样与平均实现，象棋的有效集合为恒等和左右镜像。随机 evaluator 也提供完整联合动作域，供来源冷启动与工程检查使用。

Transformer 预热使用真实象棋初始特征。普通网络与 Transformer 的真实 CUDA 检查覆盖 FP32/FP16、双方价值与动作视角、不同历史上限、合法性、镜像平均、缓存命中／清除、两个服务线程接收 16 个并发客户端，以及服务线程重建后的缓存与推理。原生 `benchmarknn` 使用固定 9×10 初始位置，支持普通前向与 MuZero initial／recurrent，普通网络可使用矩形缓冲区，MuZero 仍使用 bundle 的方形画布；计时范围见 [纯推理 benchmark](benchmark.md#纯推理-benchmarknn)。这些小规模检查不构成硬件性能或棋力结论。

## 原生搜索

`Search` 使用同一套 PUCT/FPU、选点、展开、四结果回传、LCB、uncertainty、noise pruning、根噪声及温度公式，只在联合动作的索引与容量边界适配。Gumbel 使用现有根配额、sequential halving、completed Q 和可选 Gumbel 非根。策略缓存、临时缓冲区、报告与训练策略提取覆盖 2086 个动作，不把起点或终点当成独立落子。局面推进通过 `BoardHistory`，实际终局直接回传规则结果，不向 evaluator 请求终局策略。

普通棋盘点策略保留原有栈数组规模；较大动作域才分配额外缓冲区。子节点继续以 8、64 和最终容量分段分配，最终容量由动作域决定；复制、扩容与收缩采用原有生命周期。节点增加容量字段，象棋缓冲区与规则状态复制也带来实际开销；这些检查不构成吞吐或性能等价结论。

PUCT 可使用完整历史图身份、来源 eval cache 与跨手树复用。Gumbel 仍使用 fresh tree，遵守同版本其他棋类已有的组合限制。象棋另维护镜像后的完整历史哈希；只有初始状态与整个动作历史左右镜像一致时，根才合并镜像动作。避开动作、代表动作及分析 PV 同时映射联合动作，镜像不同的历史不会因当前颜色棋盘相同而被合并。

搜索内部分析 JSON 的 `policy` 按固定 2086 动作表排列，着法与 PV 使用 `b0c2` 形式。`thisHash` 与 `symHash` 分别使用完整历史身份及其左右镜像规范值。调用原生 `getPolicy` 时，缓冲区须按 `getPolicyBufferSize()` 分配；避开动作数组按 `getMoveLocBufferSize()` 分配。原生普通／MuZero JSON analysis 和 GTP 已连接联合动作与 PV 报告。

### 局部模式与 SVB

普通网络的 PUCT／Gumbel 使用来源 SVB 和重复模式表，保留节点创建时绑定、偏差与权重更新、锁及搜索前刷新模式表的执行位置。跨手保留子树时，对淘汰节点按 `subtreeValueBiasFreeProp` 衰减贡献；整树 `clearSearch` 则沿用来源行为，不扣除表内累计值；没有节点引用的条目随后由原有清理流程删除。SVB 仍使用 5×5 局部窗口，重复模式仍使用 9×9；象棋分别编码动作起点和终点周围的完整棋子类型，两个窗口使用独立哈希。绝对联合动作、执色和 SVB 的前一步动作也参与身份。Go 的局部模式、atari／ko 条件及随机哈希序列保持原有实现，象棋使用独立哈希，不把颜色棋盘或联合动作编号传给 Go 点数组。

模式匹配始终使用落子前的棋盘。象棋可使用物理左右镜像；换色必须同时旋转 180°，不能照搬 Go 的任意旋转或单独换色。SVB 不做对称合并，继续按相同模式累计神经网络估值与搜索结果的偏差。它是局部估值修正，身份不等同于 NN 缓存或完整规则图身份。

当前对局重复模式会回放 `Xiangqi::State` 中保存的全部可用动作，包括导入局面和 fork 的前史。SVB 从同一完整状态常数时间读取前一步动作。外部 SGF 沿用来源的逐文件去重、玩家名称／手数／`%SKIP%` 过滤和衰减；自动位置记录保留 hint 合法性、手数过滤、衰减与旧文件清理。奖励仍按白方效用记账。MuZero latent 搜索按约定不使用 SVB 或真实棋盘模式表。

`tests/test_xiangqi_patterns.py` 检查所有 2086 动作的窗口对称、两端棋子身份、局部范围、重复去重、完整前史、模式文件过滤／衰减／清理、实际叶子奖励以及单／多线程 SVB 更新与释放。原生搜索和两轮 CUDA FP16 训练检查也启用这些机制；检查证明接入行为，不构成棋力或性能结论。

原生检查覆盖全动作容量、镜像身份、真实规则终局、单／多线程、hint/focus、树复用与 dense Gumbel 目标；真实 CUDA 检查覆盖普通网络／Transformer、FP32/FP16 和连续三手搜索。象棋的 SVB、重复局部模式奖励和外部模式表已接入；普通 PUCT／Gumbel 的次级 HumanSL 已接入；Gumbel 根 weightful 探索与严格配额冲突；普通／MuZero 主次网络组合及根节点限制见下文。MuZero latent 搜索及训练链见上文；普通／MuZero JSON analysis 和 GTP 已接入。


## 次级 HumanSL 搜索

普通 PUCT／Gumbel `Search` 可传入第二个版本 19 的 `NNEvaluator`，两个模型使用相同画布和象棋联合动作域。沿用来源的根／己方／对方探索比例、weightful／weightless 探索、HumanSL PUCT 系数以及最终选点的策略混合与 PIKL 修正；不新增象棋专用探索公式。Gumbel 的根 weightless／非根探索和最终策略混合规则见 [Gumbel 搜索](gumbel.md#次级-humansl)，根 weightful 探索因严格配额冲突而拒绝。第二网络不要求 metadata，不能用 Go 的 HumanSL 模型或段位 profile 代替象棋模型。`humanSLProfile` 在无 metadata 网络上沿用来源的未使用提示。

weightless 探索仍会扩展节点，但不增加祖先边访问、统计权重或根访问数；预算由实际 playout 计数约束。树复用后，根多对称重算可能先替换 NN 输出，聚合统计要等有权重的访问才重新计算。这些行为保持来源语义，不能把根访问数直接当作全部推理或探索量。

原生检查覆盖根策略、根及树内探索、单／多线程、两种合法对称平均和连续落子；CUDA 检查使用不同权重的两个版本 19 网络，覆盖普通／Transformer、FP32／FP16、联合策略报告及最终选点分布。这验证双网络机制，不代表已有训练好的人类棋风模型。原生 JSON analysis 和 GTP 可通过 `-human-model` 加载次级网络；次级 MuZero 使用三组件目录，普通主模型按真实节点评估次级网络，MuZero 主模型沿独立 latent 递归；MuZero 主模型搭配普通次级网络时只支持根节点用途，见 [MuZero](muzero.md#次级-humansl)。


## 原生 JSON analysis

`katago analysis -config ... -model ...` 使用版本 19 的普通网络，配置须声明 `game=xiangqi`，NN 画布须包含物理 9×10 棋盘，例如 `maxBoardSizeForNNBuffer=10`。其余并行、缓存、搜索和日志字段继续由原生 config 解析。普通 PUCT／Gumbel 可同时指定 `-human-model`；Gumbel 使用原生搜索算法字段选择，仍须满足 [Gumbel 组合限制](gumbel.md)。MuZero 使用 `useMuZero=true` 并将 `-model` 指向完整 bundle，配置与 latent PV 语义见 [MuZero analysis](muzero.md#原生-json-analysis)。

请求保留来源 JSON 结构：

```json
{"id":"position","rules":"xiangqi","boardXSize":9,"boardYSize":10,"moves":[["B","b0c2"],["W","h9g7"]],"analyzeTurns":[0,2],"includePolicy":true,"includePVVisits":true}
```

`B` 表示红方、`W` 表示黑方。走法是固定绝对坐标的四字符联合动作，如 `b0c2`；不能传 Go 点坐标或 pass。默认从标准初始局面开始；可用字符串字段 `xiangqiFen` 提供其他起点，其中 FEN 的 `w` 是红方、`b` 是黑方。非空 `initialStones` 无法表达棋子类型，会被拒绝；显式 `initialPlayer` 须与 FEN 一致。

`moves` 保存从该起点开始的完整合法顺序，不能用当前 FEN 加省略的历史替代。对每个 `analyzeTurns`，请求队列持有含完整长将／追逐／重复历史的 `BoardHistory` 副本；turn 编号仍是请求中 `moves` 的索引。错方走子、非法走法、终局后续走、棋类／模型不匹配和错误尺寸会返回带请求 id 与字段名的错误，后续请求可以继续处理。

`allowMoves`、`avoidMoves`、`focusMoves` 和动态 `set_focus` 均解析同一联合动作，限制数组覆盖全部 2086 个动作。来源的每方／深度限制、优先级队列、逐步报告、取消、缓存清理与关机等待保留原流程。普通／MuZero 的 Gumbel focus 保留来源不计根访问的额外搜索语义，静态请求和动态更新均支持；次级 HumanSL 的根 weightful 探索与 Gumbel 配额冲突，会在进入搜索前明确拒绝。

结果沿用来源 `rootInfo`、`moveInfos`、PV、访问／权重及四结果字段。policy／humanPolicy 是按 [动作表](../python/katago/game/xiangqi.py) 排列的 2086 项，不是棋盘点数组；非法动作标为 `-1`。`thisHash` 包含完整象棋历史，`symHash` 比较原历史与左右镜像历史。规则终局返回实际价值、空走法列表和全非法 policy；没有网络估计时不提供 `raw*` 字段。原生搜索日志以 FEN 表达象棋局面，选中分支使用联合动作。

`tests/test_analysis.py` 覆盖多 turn、相同棋盘不同历史、长将终局、合法性拒绝、联合动作限制、动态 focus／取消及继续处理、终局和次级策略输出；CUDA 检查覆盖普通网络／Transformer、FP32／FP16、PUCT／Gumbel 及双网络。Go／落子棋请求和零气子摆子另有回归检查。MuZero 与 Gumbel MuZero 的真实 CUDA 协议检查见 `tests/test_muzero_analysis.py`。

## 原生 GTP

使用 `katago gtp -config /path/to/gtp.cfg -model /path/to/model`，原生配置设置 `game=xiangqi`、`rules=xiangqi`。普通模型使用版本 19 文件；MuZero 设置 `useMuZero=true` 并传入三组件 bundle 目录。搜索组合限制与 JSON analysis 相同，普通／MuZero PUCT／Gumbel 可用 `-human-model` 加载次级版本 19 文件或三组件目录，组合边界见上文。字段仍由原生 `ConfigParser` 读取。

实际棋盘固定为 9×10，未指定默认尺寸时自动使用此尺寸。普通网络默认使用 10×10 画布，MuZero 使用 bundle 画布；有 padding 时 `requireMaxBoardSize` 必须为 false。`boardsize` 的其他尺寸会报错并保留局面。`clear_board` 恢复标准象棋开局。

`B` 表示红方、`W` 表示黑方。`play B b0c2`、genmove、Kata／LZ 分析、allow／avoid／focus、PV 和 `debug_moves` 使用固定朝向的联合动作；没有 pass。搜索和 raw-NN 请求必须与完整规则状态的下一方一致，不能通过指定另一方重置规则历史。已经终局时可查询 `final_score`，不再启动搜索或 ponder。

`set_position fen <FEN>` 建立一个新的规则起点，保留 FEN 的行动方与计数，但无法恢复此前的将捉和重复历史；颜色／点坐标摆子不适用于象棋。`loadsgf`／`printsgf` 保留 `XQFEN`、`XQHIST` 和 `XQTURN`。悔棋及需要重新重放的参数／规则更新使用完整起始快照，不从当前 FEN 重建历史。引擎不能在象棋与其他棋类间切换。

`kata-raw-nn`／`kata-raw-human-nn` 的 `policy` 后逐行输出 `<联合动作> <概率>`，顺序为固定 2086 动作表，非法动作是 `NAN`，没有 `policyPass`。有效对称为 0 和 2。棋钟、ponder、可取消搜索、统计、缓存及 auto-avoid 模式记录使用原生调用链；模式记录保存完整象棋状态和联合 hint。Go 让子和死活列表不适用于象棋，会明确报错。`kata-benchmark` 与独立 `benchmark` 使用象棋合法轨迹或指定 SGF，保留完整前史；普通／MuZero 服务、吞吐统计和与 Elo 的区别见 [搜索 benchmark](benchmark.md)。

`tests/test_xiangqi_gtp.py` 覆盖载入含前史的长将局面、悔棋与重放、FEN／SGF、原生计时及异步控制。真实 CUDA FP32／FP16 检查覆盖普通／Transformer、PUCT／Gumbel、MuZero 和普通 PUCT 次级 HumanSL，并将 raw 策略／价值与同局面的 JSON analysis 对照；MuZero 联合 PV 和 rawstats 继续使用 latent 后继。

