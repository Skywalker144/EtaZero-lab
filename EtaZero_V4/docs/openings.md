# 多棋类平衡开局

五子棋与 Hex 的平衡开局在原生 [balancedopening.cpp](../cpp/program/balancedopening.cpp) 实现，由 `GameRunner` 在来源 policy init 之前调用。selfplay、match、gatekeeper 共用这一入口，普通网络和 MuZero 共用 `Search::evaluatePosition`；MuZero 对候选真实棋盘执行 representation/initial，不用 latent 递归代替棋规转移。外围负责运行配置、任务编排和展示。

来源是 [reference_sources.json](../reference_sources.json) 固定的两份 KataGomo 实现。范围只包括无 VC 五子棋平衡开局和 Hex 首手拒绝采样，不包含 VC 棋规、Hex 特殊模板或随机填盘。Go 与象棋不支持此开关。Web/GTP 的交互入口共用此原生模块，见 [交互开局](web.md#平衡开局)。

## 配置与执行边界

原生 selfplay、match 或 gatekeeper 配置中设置 `balanceOpeningProb`，表示每局启用平衡开局的概率；默认 0，需显式启用。`balanceOpeningDropPow` 控制候选权重或接受率的指数。Hex 另外保留来源的 `makeOpeningFairRate` 和严格为正的 `minAcceptRate`。其余参数缺省值按来源的自对弈／比赛用途选择，唯一事实源是 [Settings::load](../cpp/program/balancedopening.cpp)。外围不翻译字段或重算接受率。

`GameRunner` 只初始化黑先的空棋盘；显式 SGF、hint、fork 起点及已经落子的棋盘不会重做平衡开局。AlphaZero 无模型冷启动跳过，后续有模型时启用；MuZero 初始发布模型可直接使用。每次候选推理前后检查停止请求，未接受或被中断的尝试不写回棋盘；`GameRunner` 丢弃中断对局。

平衡完成后仍按原配置执行 KataGo policy init，保留来源的 gamma／指数长度、策略温度和随机选点流程。两段前缀都进入原生棋谱和 `startHist`，训练从前缀之后开始；普通 writer 与 MuZero 完整轨迹均不为开局前缀伪造搜索目标。原生训练目标 C53 记录总前缀长度，C51 是当前实际手数；原生日志 `Balanced opening:` 记录平衡阶段的手数。

比赛每局独立生成开局，继续使用来源 `MatchPairer` 和 `GameRunner`，不额外强制黑白配对共享同一开局。恢复时在途游戏重新生成；改变开局配置须使用独立结果目录。Elo 继续使用来源拟合实现。

## 五子棋

随机骨架长度分布、首手中心截断正态分布、后续邻近与中心加权、距离参数的指数采样均来自 KataGomo 无 VC 路径。骨架触发规则终局时整次重试；Renju 禁手按现有原生棋规判负，因此也排除在接受的开局之外。

每个有效骨架随机选黑方或白方模型作为本次参考，整个候选扫描不更换模型。先评估实际先手和假设换手两种根价值，使用该玩家视角的 W−L 按来源公式提前拒绝。当换手价值为正且棋盘非空时，仅扫描距离已有棋子不超过 3 格的候选。每个合法且非终局候选按后继玩家视角价值 `v` 赋权 `(1−v²)^dropPow`，先按最大权重拒绝整次尝试，再按权重抽取一手。超过 20 次失败后按来源降低拒绝概率，并继续尝试至成功或取消。

来源累积抽样中的负 epsilon 已去除，避免抽到零权重的占用点或终局点；这是沿用 V3.4 的数值边界修正。全零候选继续重试，非有限或非法价值明确报错，不用均匀策略替代。

## Hex

在 `balanceOpeningProb` 和 `makeOpeningFairRate` 两次概率门控之后，均匀提议黑方首手，由白方模型评估落子后的白方**获胜概率 W**。接受率为 `max((1−(2W−1)²)^dropPow, minAcceptRate)`，接受后轮到白方。这里沿用来源 W 的定义，不把 W−L 或包含和棋的期望得分代入公式。

## 查看初始化后的局面

原生 `sampleinitializations` 通过 GameRunner 执行起点采样、让子、平衡开局和 policy init，再打印训练记录段开始前的局面。支持六种棋类与四种算法；哪些初始化步骤实际启用，由传入的原生 selfplay 配置决定。它将 `maxMovesPerGame` 设为 0，因此不继续自对弈，也不写训练数据。

```bash
bash scripts/run.sh native sampleinitializations \
  -config /absolute/path/to/selfplay.cfg -model /absolute/path/to/model \
  -num 5 -evaluate
```

MuZero 使用 `useMuZero=true`，`-model` 指向三组件 bundle；模型棋类须匹配，画布须容纳配置中的全部棋盘尺寸。象棋打印实际规则状态的 FEN，搜索仍使用完整历史，不能把打印出的 FEN 当作可恢复循环裁定历史的完整记录。

省略 `-evaluate` 只展示局面；启用时沿用来源诊断设置，将搜索 `maxVisits` 设为 20、`numThreads` 设为 1，其他搜索参数及预算限制继续生效。结果保留 `Winloss`，并输出白方视角的 `White W/D/L/NR`；象棋对应后手黑方。若非围棋 policy init 已到规则终局，则标注 `Terminal position (rule result)` 并打印实际裁定，不进行终局网络搜索。这些结果用于检查初始化与搜索行为，不是开局平衡质量或棋力的统计结论。

## 导出初始化局面

原生 `genposesfromselfplayinit` 使用同一 GameRunner 初始化流程，将每局的 `startHist` 交给来源 `PositionSample` 和异步 `PosWriter`，按行输出 JSON。支持六棋类与普通／MuZero、PUCT／Gumbel；MuZero 的 bundle、棋类和画布要求与上面的初始化诊断相同。

```bash
bash scripts/run.sh native genposesfromselfplayinit \
  -config /absolute/path/to/selfplay.cfg -model /absolute/path/to/model \
  -outdir /absolute/path/to/new-positions -poses 100 -threads 2 \
  -max-poses-per-out-file 50
```

保留来源工具的默认搜索参数及 `maxVisits=5`；配置选择 PUCT／Gumbel 及 Gumbel 参数，GameRunner 的 cheap／reduce visits 等设置仍生效。它仍按配置的 `maxMovesPerGame` 运行对局，然后只导出初始化局面；只需导出起点时可显式设置 `maxMovesPerGame=0`。初始化阶段到达的终局也会写出，后续 selfplay 的起点加载器按原有规则过滤，不能把输出行数当作可训练起点数。

输出为 `N.outposes.txt`。可直接用原生 `viewstartposes -start-poses-file FILE -check-legality` 检查；作为 `hintPosesDir` 输入时，将所需文件另存为加载器识别的 `.startposes.txt` 或 `.hintposes.txt` 后缀。局面沿用来源最近五手的快照结构；象棋额外保留快照之前的完整规则历史。随机增强只选择同色有效对称：Go／五子棋八种，Hex 恒等或 180°，象棋恒等或左右镜像。象棋的起始 FEN、完整前史和最近着法一同镜像，保留长将／长捉和无进展裁定；落子棋不删除无围棋气的棋子。

SIGINT／SIGTERM 丢弃尚未完成的当前对局，保留并写完已入队的完整局面，再退出模型服务。它只导出起点，不代替原生训练数据 writer。

## 随机 policy 前缀诊断

`evalrandominits` 支持六棋类、普通／MuZero 网络与 PUCT／Gumbel。每次从标准起点开始，沿用来源均匀抽取 0–199 手的初始化长度，以温度 1 调用 `getGameInitializationMove`，再运行单线程、`maxVisits=40` 的诊断搜索。它用于单独检查随机策略前缀；完整 GameRunner 初始化由上面的 `sampleinitializations` 检查。

```bash
bash scripts/run.sh native evalrandominits \
  -config /absolute/path/to/gtp.cfg -model /absolute/path/to/model \
  -num 10 -seed initialization-check
```

省略 `-num` 或设置为 0 时保留来源无限循环行为；正数指定完整输出的局面数。`-seed` 只设置初始化 RNG，不设置搜索或网络推理的随机种子，也不承诺不同模型产生相同的后续前缀。

`game` 选择棋类，省略时可从 `rules` 的棋类推断。围棋保留来源工具的日本规则，配置中的其他 Go `rules` 不替换这一诊断规则。未显式指定尺寸时 Go／落子棋使用 19 路，象棋使用标准 9×10 起点；可通过原生 `defaultBoardSize` 或成对的 `defaultBoardXSize`／`defaultBoardYSize` 指定尺寸。落子棋必须为正方形，象棋必须为 9×10。实际规则和尺寸写入原生日志；MuZero 的棋类与画布要求同上。

stdout 输出带字段名的 CSV，保留前两列的请求初始化手数和白方 W−L，再输出 W/D/L/NR、`valueSource` 和实际初始化手数。`valueSource=search` 是搜索值；`rule` 是非围棋在初始化中已到终局的实际裁定，此时不再请求网络策略。Go 保留来源允许终局后显式分析的行为，因此仍输出搜索值。提前终局时实际手数可小于请求手数；NR 在非围棋为 0。

需要纯 CSV 时设置 `logToStdout=false`，可用 `logToStderr=true` 保留诊断日志。SIGINT／SIGTERM 会停止初始化或搜索，丢弃尚未完成的当前记录，等待模型服务退出；已输出的完整记录保留。

## 验证

原生数学与状态边界检查在 [testbalancedopening.cpp](../cpp/tests/testbalancedopening.cpp)：Hex 可手算的条件抽样概率、参考模型和价值视角、五子棋候选及可重放历史、零权重点排除、禁用时不消费随机数、取消和无效价值不修改原局面。真实 CUDA 链路检查在 [test_balanced_opening.py](../tests/test_balanced_opening.py)，覆盖四种算法的原生产样、前缀／轨迹边界、来源 shuffle 与 learner，以及普通／MuZero 混合比赛和来源 Elo。工程检查不构成平衡质量或棋力效果结论。

[test_web_opening.py](../tests/test_web_opening.py) 覆盖 GTP 建议不修改已有棋盘／搜索树、四算法的原生着法执行与逐手回退、生成中和排队时的取消、候选进程退出、失败保留旧会话，以及真实浏览器操作。交互生成的双方参考都是当前主模型；它不改用次级 HumanSL 模型，也不额外运行训练专用 policy init。

[test_initializations.py](../tests/test_initializations.py) 覆盖原生初始化诊断的六棋类／四算法模型加载与四结果、导入象棋历史、policy init 终局及平衡前缀；普通网络的 Eigen 检查复用同一数值参考。

[test_random_initializations.py](../tests/test_random_initializations.py) 使用固定的零手／110 手前缀请求验证真实搜索和 policy 转移，覆盖来源 Go 缺省行为、非围棋终局、有限／无限运行和普通／MuZero 服务的中断退出。

[test_generated_initializations.py](../tests/test_generated_initializations.py) 检查六棋类／四算法的真实模型导出与重新加载、有效对称和无气子保留、矩形棋盘着法变换、象棋完整历史再次进入 selfplay writer，以及中断后的文件与模型服务清理。象棋状态镜像的循环及终局裁定另由原生规则测试核对。
