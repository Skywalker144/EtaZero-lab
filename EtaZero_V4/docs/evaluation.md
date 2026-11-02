# 独立比赛与 Elo

`match` 调用本版本原生 `cpp/command/match.cpp`，使用来源 `MatchPairer` 与 `GameRunner`。普通网络和 MuZero 三组件模型可以在同一比赛中对弈，每个 bot 独立选择 PUCT 或 Gumbel。外层只管理运行目录、续跑和结果呈现；Elo 直接调用 `python/katago/utils/elo.py`，不维护第二套拟合公式。

## 入口与配置

```bash
bash scripts/run.sh match --config /absolute/path/to/match.cfg \
  --output-dir /absolute/path/to/new-evaluation --anchor baseline

# 同一命令续跑；全部完成时直接使用已保存结果，不重新加载模型。
bash scripts/run.sh match --config /absolute/path/to/match.cfg \
  --output-dir /absolute/path/to/new-evaluation --anchor baseline

# 只重新拟合、绘图，不需要原配置、模型、原生程序或 GPU。
bash scripts/run.sh match --output-dir /absolute/path/to/new-evaluation \
  --fit-only --anchor baseline --first-player-advantage
```

配置由原生 `ConfigParser` 解析，包括来源的 include、逐 bot 搜索参数、`includeBots`、`secondaryBots`、`extraPairs` 与 `extraPairsAreOneSidedBW`。比赛条件使用独立于训练和 Web 的公共与棋类配置：

| 配置 | 职责 |
|---|---|
| [common.cfg](../configs/match/common.cfg) | 公共搜索预算、温度、认输、执行并行与推理参数；采用普通 PUCT。 |
| [go.cfg](../configs/match/go.cfg) | 围棋棋规、尺寸分布、固定贴目与初始化。 |
| [gomoku.cfg](../configs/match/gomoku.cfg) | Freestyle 尺寸分布、平衡开局与初始化；任务可覆盖为 Standard 或 Renju。 |
| [hex.cfg](../configs/match/hex.cfg) | Hex 方盘分布、首手平衡采样的两层概率、接受率指数与下限。 |
| [xiangqi.cfg](../configs/match/xiangqi.cfg) | 中国象棋标准棋盘与初始化。 |

四个棋类文件 include `common.cfg`，显式声明平衡开局、policy init 和棋类温度半衰期。公共配置不放参赛模型、配对和局数；直接 match 在任务配置中填写这些字段，arena／autoelo 由生成的 `participants.cfg` 提供。搜索预算和棋盘分布可以在任务中显式覆盖。原有实验比赛配置仍独立使用，不自动切换到这些公共文件。

例如在本版本新建任务配置 `configs/my_match/match.cfg`，include 所选棋类文件，并定义两个不同算法族的 bot。原生 include 须使用相对路径，相对包含它的文件解析；首个命令行 `-config` 可以使用绝对路径。

```ini
@include ../match/hex.cfg

numBots = 2
numGamesTotal = 40
botName0 = baseline
nnModelFile0 = /absolute/path/to/ordinary/model.bin.gz
useMuZero0 = false
rootSearchAlgorithm0 = puct

botName1 = candidate
nnModelFile1 = /absolute/path/to/muzero-model-directory
useMuZero1 = true
rootSearchAlgorithm1 = gumbel
nonrootSearchAlgorithm1 = gumbel
useGraphSearch1 = false
subtreeValueBiasFactor1 = 0
avoidRepeatedPatternUtility1 = 0
rootSymmetryPruning1 = false
useLcbForSelection1 = false
chosenMoveTemperatureEarly1 = 0
chosenMoveTemperature1 = 0
chosenMoveSubtract1 = 0
chosenMovePrune1 = 0
gumbelMaxConsideredActions1 = 16
gumbelNoiseScale1 = 0
```

普通模型路径指向原生模型文件，MuZero 路径指向含 `model.json` 与三组件的目录。`useMuZeroN` 优先于公共 `useMuZero`，缺省为 false；其余算法组合约束见 [Gumbel](gumbel.md) 和 [MuZero](muzero.md)。MuZero 须关闭该 bot 的 graph、SVB 和局部重复模式，Gumbel 须关闭 graph 并明确其选点参数。公共配置保留既有普通 PUCT 的 graph／SVB 设置；arena／autoelo 只从模型记录推导 `useMuZeroN`，不会自动改变搜索算法或消除冲突。若整场采用 MuZero 兼容的公共搜索，可在任务配置中统一设置 `useGraphSearch=false`、`subtreeValueBiasFactor=0`；普通模型也将使用这组条件。同一配置使用同一棋类，模型的输入布局、MuZero 棋类及画布必须与比赛匹配。

arena／autoelo 的 `--config` 可直接指定对应的棋类文件；需要修改公共搜索条件时，指定 include 该文件的任务配置。自动实验的 `[evaluation] config` 同样填写所选配置路径。文件路径相对 `exp.cfg` 解析，不会根据训练棋类自动选择比赛文件。

运行前可用原生解析器检查 include 和覆盖后的参数，不加载模型或启动比赛：

```bash
bash scripts/run.sh native match -config configs/my_match/match.cfg -print-config
```

输出 JSON 中的 `resolved_config` 是展开后的原生参数文本；它不补齐全部代码默认值，因此关键比赛条件仍在 cfg 中显式维护。实际运行的合并配置与参赛信息保存在 `results/plan.json`。`cpp/configs/match_example.cfg` 继续作为来源完整模板，其他高级参数可参考该文件与原生实现。

普通网络的 `avoidSgf…` 模式表保留来源逐 bot／逐配置组索引。Go、各落子棋和象棋使用各自模式身份，外部 SGF 可共用文件目录；窗口和有效对称见 [架构](architecture.md)。MuZero 仍排除真实棋盘模式与 SVB。

象棋使用 `game=xiangqi`、`bSizesXY=9-10` 与版本 19 模型，支持普通／MuZero 混合模型的 PUCT／Gumbel 比赛及原生 gatekeeper。原生 B／W 分别对应红／黑，完整规则前史通过 SGF/hint 起点保留。局部模式等当前组合限制见 [象棋比赛与晋级](xiangqi.md#比赛与晋级)。

搜索参数的索引是 bot 编号；来源的 `nnRandSeedN`、`gpuToUseModelNThreadM` 等推理选项索引是**按首次使用顺序去重后的模型编号**。多个 bot 使用同一个模型路径时共享模型服务，但可采用不同搜索参数。普通模型文件与 MuZero 目录不能用同一路径声明成不同算法族。参与 Elo 的 bot 名称必须唯一，且不能使用来源保留名称 `P1Advantage`。

外层将配置所在目录作为原生工作目录，因此模型、SGF 起点等相对资源路径以该目录为准；include 仍由来源解析器处理。原生直调 `native match` 则使用调用者的工作目录。

## 对局与恢复

首次运行保存来源源码快照与调用记录。原生配对器生成执色配对，外围 `python/etazero/match_cache.py` 按本次配额保存完整赛程；比赛只在计划成功发布后开始。默认配对保留来源每轮打乱顺序的双向对局；游戏数未覆盖完整配对轮次时，不能声称所有组合已均衡执色。指定单向 extra pairs 时也不额外补反向对局。

每局完成后，原子发布 `results/game-N.json`，包含执色、结果、实际棋谱、游戏 hash、手数、双方用时及结束原因 `end_reason`（`rule_result`、`resignation`、`no_result`、`move_limit`）和 `hit_turn_limit`。

围棋达到 `maxMovesPerGame` 仍未终局时，使用训练端同一个 `BoardHistory::endAndScoreGameNow` 对最终棋盘计分，保留本局实际的数子／数目、税、贴目及其他规则；不改为另一套统一棋规，不恢复收官辅助。结果按计分记作 B/W/D，截断本身不记作 NR。完整截断棋谱保存在逐局 JSON 和汇总 SGF，SGF 的 `RE` 保存裁定结果，根注释包含 `endReason=move_limit,adjudication=training_score`。非围棋未到规则终局的手数截断仍会报错。认输产生的终局仍是有效比赛结果；信号中断的在途对局不裁定、不提交。只有计划中的所有对局完成后才拟合 Elo，避免把中断时先结束的对局子集当成完整样本。

续跑保留赛程和已发布游戏，只重做未完成游戏；在途棋盘、开局与搜索 RNG 不恢复，重做游戏可能采用新的随机开局和搜索随机数。固定配置与工作目录是结果复用的条件；模型和起点等输入路径的内容须保持不变，不逐文件散列检查。当前不支持在原目录增加 `numGamesTotal` 或更换配置；新比赛使用新目录。

五子棋／Hex 可通过原生 `balanceOpeningProb` 启用 [平衡开局](openings.md)。每局使用来源 GameRunner 独立初始化；普通模型与 MuZero 混合比赛也调用同一原生实现。开局配置属于比赛条件，改变后不能复用原目录结果。候选筛选不在 Elo 拟合或外围任务调度中执行。

外围根据原生解析后的配置及工作目录检查缓存身份，读取逐局文件以确定缺局，不依赖可重建的 `summary.json` 或汇总 SGF。外层目录锁由原生子进程继承；原生结果写入器另持有结果目录独占锁，保证外围控制器退出后，仍在运行的原生 writer 不会被并发接管。原生内存只保存赛程与完成标记，不保留所有已完成棋谱；赛程的内存与磁盘开销仍随总局数增长。原子发布不承诺整场比赛跨文件断电事务。

围棋的 `go_move_limit_scoring` 裁定语义同时记录在原生配置身份和保存的赛程中。autoelo 为不同裁定语义使用独立缓存与会话；直接 match／arena 拒绝在旧裁定目录中续跑，需另选输出目录。已有原件保持不变，仍可用 `--fit-only` 重建已完成比赛的旧口径结果。

`-results-dir` 与 `-pair-jobs` 是外围控制器使用的原生结果接口。原生端通过 `ETAZERO_MATCH_STATE` 行请求缓存状态或提交来源配对提案，由外围读取／发布计划并回复；`ETAZERO_MATCH_READY` 表示准备结束。它们需要控制器处理 stdin/stdout 协议，不能作为脱离外围的自动续跑命令。普通原生 match 的来源按需配对与 SGF 输出入口仍可独立使用。

`summary.json` 由 EtaZero 外层在原生比赛进程正常退出后读取逐局记录生成，按执色分别汇总 B/W/D/NR；原生 `match` 本身不生成该文件。汇总以及 match／arena／autoelo 的 `elo.json` 均包含 `end_reasons`、`games_truncated` 和 `truncation_fraction`。比例分母是该结果实际纳入的已完成局数，不包含未完成局或超出请求前缀的缓存对局；有结束原因缺失的旧记录时标为 `unknown`，比例为 `null`，不伪装成零截断。汇总失败会警告，但不影响有效对局和完整比赛的 Elo 拟合。`--fit-only` 直接读取原始结果，不依赖或改写汇总。

目录中的主要产物为：

| 位置 | 内容 |
|---|---|
| `source.tar.gz`、`source.json` | 创建比赛时的代码及来源信息 |
| `invocation.json`、`invocations/` | 配置位置、每次调用参数、工作目录和独立日志 |
| `results/plan.json` | 外围保存的原生配置、模型与 bot 信息、实际执色赛程 |
| `results/game-N.json` | 不覆盖的逐局结果与棋谱 |
| `results/summary.json` | 外层从原生逐局结果重建的 B/W/D/NR 汇总 |
| `sgfs/` | 来源按工作线程输出的汇总棋谱 |
| `elo.json`、`elo.csv`、`elo.png`、`elo.svg` | 可重建的拟合结果与图 |

## 评分口径与范围

原始对局分别保存 B 胜、W 胜、D 与 NR；仅在来源 Elo 拟合中将 D 和 NR 各折算半分。采用来源胜率似然、先验、中心约束、协方差和 LOS。`--prior-games` 默认 2，控制每个参与者接近 Elo 0 的来源先验强度；可选 `--first-player-advantage` 使用来源先手优势项及其先验。实际对弈图必须连通，不以先验代替不同组之间的比赛。

`--anchor` 指定相对 Elo 的零点，缺省取参与比赛的首个 bot。图和 CSV 的 `stderr` 是通过协方差计算的“相对 anchor 的差值标准误”，anchor 自身为零，不表示其绝对棋力已知。图中误差棒约为一倍标准误；`elo.json` 另存来源原始坐标下的协方差、LOS 和可选先手优势估计。来源在极少样本下不能定义的 LOS 会显式警告并保存 null。小样本工程检查不构成棋力结论。

原生 `native match` 未指定 `-results-dir` 时仍保留来源按需配对路径，不执行上述持久化与完整比赛拟合。多个游戏并发会竞争计算资源，来源匹配器的定时搜索条件不能自动解释为单局独占资源的棋力比较。

## 历史模型与训练时间

`arena` 从单个运行目录或包含多个运行子目录的数据目录发现模型，建立比较图，并通过同一个原生 match 进程执行全部比赛。配置仍使用独立的来源 match profile；训练时的搜索参数不会自动覆盖评估参数。

```bash
bash scripts/run.sh arena --data /absolute/path/to/runs \
  --config /absolute/path/to/match.cfg --output-dir /absolute/path/to/arena --dry-run
bash scripts/run.sh arena --data /absolute/path/to/runs \
  --config /absolute/path/to/match.cfg --output-dir /absolute/path/to/arena
bash scripts/run.sh arena --output-dir /absolute/path/to/arena --fit-only
```

发现过程以一次读取的 `state.json` 完成轮数为边界，逐轮读取不可变 `commit.json`。只选择 accepted 模型，并按模型路径首次接受的轮次去重；拒绝的候选、重复沿用旧模型的轮次和未提交的导出都不成为新棋手。无需读取训练观测汇总或重新打开训练数据。计时必须有效、有限、单调，并使用 `committed_wall_excluding_compile`；缺失或失效时明确拒绝等时间比较，不用文件修改时间补齐。

模型 ID 为 `运行目录名:八位轮次编号`。默认取首个、按 `--stride` 选择的轮次和最后一个模型；`--sample-seconds` 则在已完成时间覆盖的正整数倍网格上选最近模型，距离相同时选较早轮次，同模型去重。最后模型距离最后选中模型超过半个间隔才追加；未覆盖首个网格时保留最后模型。默认仍保留首个模型，`--exclude-first` 可关闭此项。显式 `--anchor` 即使不在采样网格上也保留。图的横轴是首次接受模型时的实际提交时间，不把时间改写成目标网格，也不把后续重复使用旧模型的时间当成新测量。

`--neighbors` 控制每个已选模型与本臂前几个模型对弈。缺省跨臂配对连接共同时间范围内的邻近模型；`--cross-seconds` 按双方已覆盖的固定时间网格选择跨臂对手，`--final-cross` 额外连接各臂最后选中的模型。生成的实际对弈图必须连通。各臂必须属于同一棋类，可以混合普通模型与 MuZero。这属于相近累计训练时间的采样比较，轮次粒度不保证双方时间完全相同，也不保证固定 visits 的思考时间相同。

`--games-per-pair` 是每个无序模型对的正整数局数。生成的 `participants.cfg` 通过来源多配置文件覆盖机制指定模型、棋类、配对和局数；原 profile 的搜索、开局、认输与推理参数继续由来源解析器管理，相对资源仍基于原 profile 所在目录。推理索引规则与独立 match 相同。arena 不从训练算法名称推断本次使用 PUCT 还是 Gumbel，由评估 profile 明确选择。

原生 `numGamesPerPair` 模式需要 `-results-dir`，并要求配对列表包含每对的两个方向各一次、`numGamesTotal` 等于每对局数乘模型对数。完整轮次沿用来源配对器；奇数配额的最后一轮也由来源随机打乱，每对只取首次出现的方向。每对最终执色相差至多一局。原生未设置该字段时保持原来的总局数语义。

arena 的 `plan.json` 保存参与者、时间和图；`match/` 保留前述完整比赛记录与分类评分图；顶层 `elo.json`、`elo.png` 和 `elo.svg` 按训练臂呈现实测模型点及相对 anchor 的标准误。连线仅连接已测点，不测量中间时刻、不向训练结束后外推。同一目录续跑要求发现的计划不变，比赛恢复沿用逐局提交边界。训练继续产生新模型或采样条件改变时使用新的 arena 目录；`--fit-only` 只使用已有计划和逐局记录，即使训练目录、模型或原配置已归档也可重建结果。

`--dry-run` 只读取提交证据并输出模型／赛程，不创建输出目录、不启动原生程序；它不替代原生配置与模型加载检查。首次正式执行保存的源码快照位于 `match/`，外层锁由比赛子进程继承。

### 模型驻留窗口

`match` 和 `arena` 接受外层参数 `--max-loaded-models N`，N 至少为 2；省略时保留来源一次加载全部参与模型的行为。容量和窗口选择属于外围调度，不是原生搜索配置字段；每次调用的参数保存在 `invocations/`，arena 的 `--dry-run` 也显示该执行选项。

窗口按去重后的模型服务计数，同一原生模型路径的多个 bot 共享一个服务，MuZero 的三个组件整体算一个服务。上限是模型数量，不能解释为显存字节上限。加载仍使用来源的 evaluator 工厂与模型索引，`nnRandSeedN`、设备选择和 per-bot 搜索参数不因窗口内排序重新编号。

原生 `MatchPairer` 先产生完整赛程，原生端报告尚未完成的游戏 ID 及双方实际使用的模型服务编号。`python/etazero/match_dispatch.py` 在外围选择窗口，优先处理共享模型的对局并复用上一窗口的服务；游戏 ID、双方、执色与配额保持不变。原生端只接受恰好覆盖全部待执行 ID 的批次，拒绝重复、遗漏和无效 ID，然后在同一个进程中执行。窗口内使用来源 `numGameThreads` 对局池跨模型对补局，没有逐对完成屏障。全部对局线程退出后才释放不再需要的模型，因此在途搜索与 MuZero latent 不会引用已经卸载的服务。共有模型继续保留权重、批处理服务和缓存。

外围通过原生 `-external-dispatch` 接口交换一次窗口计划：stdout 的 `ETAZERO_MATCH_DISPATCH` 行报告待执行 ID 与服务编号，stdin 接收包含 `windows` 的 JSON 对象，每个窗口是游戏 ID 数组。该接口需要 `-results-dir` 或 `-pair-jobs`；模型权重尚未加载时完成窗口交换，完整缓存命中时直接退出。原始请求、返回批次及容量保存在 `invocations/dispatch/`，不是新的配对或缓存身份。普通原生 match 不需要该协议；`-pair-jobs` 未启用窗口接口时仍由外围管理赛程，全部待执行对局在一个全驻留批次中完成。

中断后只调度缺失的游戏，已提交原件保持不变；可以用不同容量恢复同一赛程。窗口设置和重载会改变资源竞争、缓存命中与随机推理的分配，不保证与全驻留或另一容量逐局结果相同；采用时间搜索预算时尤其不能把它视为单局独占算力。比较时保留每次调用参数及日志。完全完成的比赛仍可在无 GPU 时直接重用。

原生日志的 `Model window` 行记录实际驻留、加载、沿用模型数、局数和准备耗时；每次释放服务时记录其真实 `NN rows`、`NN batches` 与平均 batch size，共有服务的统计跨窗口累计至实际释放。窗口准备顺序加载模型，尚未加入可配置的并行模型加载器。

`autoelo` 在外围编排多对缓存，并使用一个原生 match 进程执行，默认驻留上限为 2，也接受 `--max-loaded-models N`。每对仍使用各自的原生配置、配对器、计划和逐局记录。共享 profile 由原生解析器解析；只有参与者和局数可在任务间不同，其他差异会被拒绝。外围任务清单不引入另一套搜索或棋局执行器。

原生 per-bot 搜索字段和推理字段始终按每对内的原始索引解析。跨配对只有同路径、同模型族且推理选项相同的服务才共享；不同的显式种子、设备、精度或缓冲区设置会保留独立服务。同一路径因此可能占用多个驻留名额。未指定种子时，共享服务在实际加载时生成并记录一个原生种子；共享和重载会改变随机序列分配，不承诺与逐对独立进程逐局一致。

实际批量命令、任务清单和原生日志保存在会话的 `invocations/`，每对缓存的调用记录指向同一命令与日志。驻留容量只记录在调用证据和 `--dry-run` 的 `execution` 中，不参与配对缓存或会话计划身份。Web 棋局与分析使用独立原生 GTP 会话，见 [Web](web.md)；评估结果不依赖该页面。

## 增量 autoelo 与训练后评估

`autoelo` 使用与 arena 相同的历史模型选择、时间点、配对图和来源 Elo。首次运行固定 anchor 的模型 ID 与路径，后续选点仍保留该模型；更换参考模型需要独立输出根目录。

```bash
bash scripts/run.sh autoelo --data /absolute/path/to/runs \
  --config /absolute/path/to/match.cfg --output-dir /absolute/path/to/elo \
  --games-per-pair 10
bash scripts/run.sh autoelo --output-dir /absolute/path/to/elo/latest --fit-only
```

每对模型的 `pair_cache/` 保存外围计划与原生逐局 JSON、日志和棋谱。缓存依据原生解析器解析后的配置（含 include）、模型路径及资源工作目录区分，仅局数不参与身份。输入路径内容须保持不可变；参数或资源目录变化形成独立缓存，不混合不同评估条件。只更换同目录下内容相同的配置文件名可以复用。

增加每对局数时，外围保留原赛程前缀和全部已提交对局，只追加不足部分；旧赛程为奇数局时先补反向执色，再向原生 `MatchPairer` 索取剩余配对，继续来源随机双向轮次。扩展前的计划单独保存在 `results/plans/`。减少局数时只读取所请求的完整前缀，不删除缓存中的额外对局。前缀已经完成时无需加载模型或 GPU；中断后只补所请求前缀中的缺局。普通 match 仍使用前述固定赛程语义。

每次选点与配额计划对应 `sessions/` 中的独立记录，保留来源快照、当时的比赛计划及逐局原件的硬链接。后续缓存扩展不改写旧会话。只有全部配对满足请求局数后才拟合，`latest` 指向最近成功完成的会话。`--fit-only` 指向会话目录或 `latest`，无需原配置、训练目录或 GPU；它使用已保存原生对局，允许重新选择拟合先验。来源拟合和误差口径不变，图的横轴明确为 EtaZero 调度累计时间。

实验 `exp.cfg` 可用 `[evaluation]` 配置训练后的自动评估，例如：

```ini
[evaluation]
config = match.cfg
games_per_pair = 10
final_cross = true
max_loaded_models = 2
```

所有当前实验臂达到预算后，`autoexp.sh` 在第一个 GPU 槽位执行 autoelo，结果位于实验输出的 `elo/`。评估配置在启动训练前检查外层字段；实际原生参数和模型由比赛入口检查。评估失败或中断单独记录，不回滚训练提交；再次执行相同实验只补未完成训练和评估。增量复用减少重复对局，不构成独立训练重复，也不改变来源统计口径。
