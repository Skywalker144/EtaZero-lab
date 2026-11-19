# 原生棋谱诊断

`evalsgf` 在 `cpp/command/evalsgf.cpp` 中保留 KataGo 的棋谱重放、逐局面搜索、树／PV、计时和 JSON 报告流程。支持六种棋类的普通网络与 MuZero，以及 PUCT／Gumbel；模型服务和搜索直接复用训练、GTP 与 analysis 使用的实现。

## 入口与模型

```bash
bash scripts/run.sh native evalsgf \
  -config /absolute/path/to/gtp.cfg -model /absolute/path/to/model \
  /absolute/path/to/game.sgf -m 12 -visits 128 \
  -print-json -print-policy -print-root-nn-values
```

`-m N` 分析主线已经走完 N 手的位置，`-move-num-end M` 按顺序分析到 M 手，包含两端。配置字段仍由原生 `ConfigParser` 解析；`-visits` 和 `-threads` 覆盖本次调用的搜索预算与线程数。省略配置中的 `game` 时，由 SGF 规则确定棋类；显式配置必须与棋谱一致。

普通模型使用原生模型文件。MuZero 设置 `useMuZero=true`，`-model` 指向完整三组件目录；真实根使用 initial，后续树节点使用 recurrent。搜索约束与 [MuZero](muzero.md)、[Gumbel](gumbel.md) 相同，不能启用不适用的真实棋盘机制。

`-human-model` 可指向普通模型文件或 MuZero 目录，复用原生次级网络策略、探索和选择。主次模型需要兼容的棋类和 NN 画布。MuZero 主模型搭配普通次级网络时，次级网络仅用于根节点；涉及 latent 树内探索时需要次级 MuZero。需要 metadata 的模型使用 `humanSLProfile`。

## 局面与动作

`-extra-moves` 在目标局面依次执行额外合法着法；`-avoid-moves` 限制根候选，`-hint-loc` 指定原生根 hint，`-print-branch` 选择要打印的树分支。落子棋与 Go 使用原生棋盘坐标；象棋使用 `b0c2` 形式的联合动作，并保留 SGF 的完整 FEN／前史。象棋的坐标、规则历史和终局语义见 [中国象棋](xiangqi.md)。

普通搜索的分支必须能按真实棋规重放。MuZero 的分支允许进入真实棋盘无法继续的位置，命令明确标注后续为 latent 节点；不会用真实棋规过滤 latent 后继。`-extra-moves` 始终是在真实棋局中落子。

Go 的 `-override-komi` 与 `-override-rules` 继续作用于棋谱重放；非围棋贴目只能为零。`-print-score-now` 是 Go 的规则计分操作，不使用网络目差或 ownership。已排除的预测分数、ownership 和收官奖励选项不再提供。

## 输出与输入导出

`-print-root-nn-values` 输出 W／D／L／NR，采用白方视角；象棋白方对应后手黑棋。`-print-json` 直接调用来源搜索报告，包含 draw 与 no-result。`-print-policy`、`-print-log-policy` 和 `-print-dirichlet-shape` 使用当前棋类动作域：象棋逐动作列出，落子棋不打印 pass。来源 JSON 点策略仍保留最后一个 pass 槽，非围棋该槽为 −1。Dirichlet 展示使用原生形状公式，覆盖实际合法动作，Go 包含 pass。

`-raw-nn` 直接评估真实局面，不运行搜索；存在次级模型时也打印其次级结果。普通网络使用原生 evaluator，MuZero 使用 initial。非围棋终局没有 NN 合法策略，此请求明确拒绝；正常搜索报告仍可显示规则终局结果。

`-dump-npz-input-to /absolute/path/to/inputs.npz` 在搜索后导出未做随机对称变换的真实观测，并对同一观测执行一次原生推理。空间特征由实际模型输入版本和完整棋类历史生成，字段为：

| 字段 | 布局 |
|---|---|
| `binaryInputNCHW` | FP32 `[1, C, NN高度, NN宽度]`；包括画布 mask 和 padding |
| `globalInputNC` | FP32 `[1, G]` |
| `metadataInputNC` | 仅需要 metadata 的模型输出，FP32 `[1, 192]` |

MuZero 导出 representation 的观测输入。诊断导出沿用独立的原生输入参数：固定对称为 0，使用 `rootPolicyOptimism`；不包含搜索路径中的 latent、噪声或每节点状态。`-raw-nn` 会直接返回，与输入导出选项并用时不会执行导出；按范围分析时导出路径会依次写入，最终保留最后一个位置，需要逐位置文件时应分别调用。

`tests/test_evalsgf.py` 覆盖棋类推断、动作选项、象棋前史与终局、矩形 Go、主次模型组合和 metadata，并将六种棋类／四种算法的真实 CUDA FP32／FP16 输出与 Python 模型对照。它验证诊断与推理语义，不提供棋力或性能结论。

## 按棋谱汇总搜索熵

原生 `searchentropyanalysis` 在一组位置上运行搜索，汇总搜索策略熵和 policy surprise。它复用 `Search::getPolicySurpriseAndEntropy`，支持六种棋类、普通／MuZero 网络与 PUCT／Gumbel；模型参数与以上入口相同，但不提供次级模型选项。

```bash
bash scripts/run.sh native searchentropyanalysis \
  -config /absolute/path/to/gtp.cfg -model /absolute/path/to/model \
  -sgf /absolute/path/to/first.sgf -sgf /absolute/path/to/second.sgf
```

`-sgf` 可重复，所有文件须属于同一棋类。省略配置中的 `game` 时从第一份棋谱推断，其余棋谱必须匹配；各棋谱使用自身的规则、布局和历史。棋谱通过原生重放器读取，无法重放的着法会报错，不跳过后拼接成另一局。象棋保留 `XQFEN`／`XQHIST` 的完整裁定历史；MuZero 画布须容纳所有棋谱尺寸。

来源内置围棋数据集仍用 `-boardsizedataset 9`、`13`、`19`、`10x14` 或 `rectangle` 选择，与 `-sgf` 互斥。内置集保留来源的默认规则重放、跳过非法着法及固定洗牌口径，不按每条内嵌 SGF 的 `RU` 重新选择规则；与显式导入同一段棋谱的结果不能直接混为一种测量。

每份主线有 N 手时枚举第 0 到 N−1 手之前的位置，不包含最后一手之后的位置；跨文件重复位置不去重。所有位置沿用来源固定种子洗牌，已终局的位置不搜索。因此 `Total positions to search` 是枚举数，`Searched numPositions` 是实际得到统计的数量，二者可以不同。

熵与 surprise 使用自然对数。PUCT 使用来源选点权重，Gumbel 使用覆盖未展开合法动作的改进策略；surprise 是该策略相对原生根策略的 KL，不是四结果 value surprise。搜索参数中的根噪声、温度和策略处理继续生效。输出每十个有效位置更新一次，结束时再输出全体均值与总体标准差；没有有效位置时只打印数量，不伪造均值。该独立诊断不修改训练采样或训练日志。

`tests/test_search_entropy.py` 用均匀策略和常值网络手算 PUCT／Gumbel 的熵、KL 与总体方差，覆盖六棋类／四算法、所有内置数据集、棋类与画布边界、不可重放记录和象棋前史终局。Eigen 普通网络检查复用相同参考。

## 查看与搜索起点文件

`viewstartposes` 与 `trystartposes` 读取来源的逐行 `PositionSample` JSON，支持六种棋类、普通／MuZero 模型与 PUCT／Gumbel。输入可以来自 SGF 采样、hint 文件或 [初始化导出](openings.md#导出初始化局面)，文件参数均可重复。

```bash
# 无模型：查看局面，或添加 -check-legality 只报告不合法的重放／hint。
bash scripts/run.sh native viewstartposes -start-poses-file /absolute/path/to/positions.txt

# 带模型：按配置搜索，并将文件中的 hint 交给原生根 hint 机制。
bash scripts/run.sh native viewstartposes \
  -start-poses-file /absolute/path/to/positions.txt \
  -config /absolute/path/to/gtp.cfg -model /absolute/path/to/model -idx 0

# 来源诊断搜索；hint 未被选中时，在真实棋局落下该手后重新搜索。
bash scripts/run.sh native trystartposes \
  -startposes /absolute/path/to/positions.txt \
  -config /absolute/path/to/gtp.cfg -model /absolute/path/to/model
```

无模型查看可混合棋类。带模型时，参与搜索的局面必须属于同一棋类，显式 `game` 必须匹配；省略时从局面推断。`viewstartposes` 的 `-idx` 使用合并文件后的索引，两条命令的 `-min-weight` 均按来源行为过滤；未选中的行不参与模型棋类和 MuZero 画布检查。模型族须支持该棋类，MuZero 画布须容纳所有待搜索局面。象棋从完整状态和前史重放联合动作，不通过最终 FEN 重置循环裁定。

`viewstartposes` 的 Go 规则来自配置，保留来源不加载 `komi` 覆盖的行为；非围棋使用对应棋规。`trystartposes` 的 Go 规则仍逐局面调用来源 `genRandomRules`，配置中的 Go 规则不会替换这一步。它沿用来源关闭出手温度、根噪声、根策略温度及 forced playout 系数，并将根 FPU reduction 设为非根的一半；不是训练搜索参数的原样回放。已排除的 `-auto-komi` 不提供。

`trystartposes -random-sample` 保留来源有放回的累计权重抽样，抽取次数等于已加载局面数；先抽样，再判断 `-min-weight`，所以实际搜索数可能更少。非法 hint 被报告并跳过，合法且未被选中的 hint 在真实状态执行一次后重搜。MuZero 也遵守这一真实落子边界，随后才进入 latent 搜索。文件中的 `trainingWeight` 只展示，不重写诊断搜索权重。

搜索输出保留来源树和 PV，另列白方视角 `White W/D/L/NR`；象棋白方对应后手黑棋。非围棋终局报告实际规则结果，不调用终局网络策略。`viewstartposes -check-legality` 会报告并跳过非法重放或 hint；若同时提供配置，合法局面仍会运行来源搜索。两条命令都不修改输入文件或训练数据，也不提供次级模型参数。

[test_startpose_search.py](../tests/test_startpose_search.py) 用常值网络核对四结果，覆盖六棋类／四算法、象棋前史与 hint 真实转移、metadata、筛选与加权抽样、模型棋类／画布边界和来源错误处理。普通模型的 Eigen 检查复用相同参考。

## SGF hint 的原生策略概率

`checksgfhintpolicy` 统计 SGF 注释中 `%HINT%` 所标记着法的原生网络概率。它不运行搜索：普通模型调用 evaluator，MuZero 调用真实局面的 initial；Gumbel、PUCT 和搜索访问预算不参与这一统计，也不使用 Gumbel 改进策略或 MuZero recurrent 策略。

```bash
bash scripts/run.sh native checksgfhintpolicy \
  -config /absolute/path/to/gtp.cfg -model /absolute/path/to/model \
  -sgfdir /absolute/path/to/sgfs
```

`-sgfdir` 可重复。六种棋类均使用来源 SGF 遍历、含注释和父局面的去重身份，以及着法之前的 `PositionSample` 重放。棋类从第一条可评估 hint 推断，显式 `game` 必须匹配；一次调用的有效 hint 须属于同一棋类。模型延迟到第一条有效 hint 加载，MuZero 使用完整 bundle 并检查棋类与画布。象棋保留初始 FEN、完整前史和联合动作索引。需要 metadata 的主模型使用原生 `humanSLProfile`，省略时明确报错。

Go 沿用来源的中国／日本两套规则，使用各自默认贴目，不按 SGF 的 `RU`／`KM` 或配置中的 Go 规则替换；其他棋类使用对应规则。每条规则下执行八次 `skipCache=true` 的原始推理。这是八次重复评估，**不是显式遍历八种对称**：`nnRandomize`、`nnForcedSymmetry` 仍由 evaluator 处理，Hex／象棋只允许各自有效的同色对称。历史特征模式取自普通模型或 MuZero representation 的声明，不借用无关搜索设置。

输出 `Total number of hint positions` 是来源遍历找到、且能回退一手的 hint 数；各规则重放失败时仍按来源记录警告并跳过该次规则评估。`Average log policy across all hints` 保留来源 `log(p + 1e-30)` 的逐次等权平均，使用自然对数；它不是先平均概率再取对数。没有评估权重时保留来源 `nan`，不能作为有效测量。去重只在单个文件内生效；`%HINTLIGHT%`、无前一手的根注释，以及来源 `allowGameOver=false` 排除的终局位置不参与评估，包括非围棋恰好获胜／和棋的最后一手。

[test_hint_policy.py](../tests/test_hint_policy.py) 用均匀策略的合法动作数手算平均值，覆盖六棋类、普通／MuZero、来源去重与筛选、八次原生推理计数，以及 Go 两套规则和 metadata 的 Python 数值参考。象棋非均匀策略与原生 GTP 的完整前史、恒等／镜像输出对照；普通模型另有 Eigen 检查。

## SGF 波动加权采样

`samplesgfs` 保留来源不加载模型的普通采样，也支持六种棋类的 value-fluctuation 加权路径。后者使用普通 evaluator 或 MuZero 的真实局面 initial，不运行搜索或 recurrent；PUCT／Gumbel 设置不参与计算。输出仍由来源 `PosWriter` 写入 `*.startposes.txt`。

```bash
bash scripts/run.sh native samplesgfs \
  -sgfdir /absolute/path/to/sgfs -outdir /absolute/path/to/new-startposes \
  -sample-prob 1 -turn-weight-lambda 0 \
  -value-fluctuation-model /absolute/path/to/model \
  -config /absolute/path/to/inference.cfg -num-threads 2
```

`-config` 和 `-override-config` 可选，使用原生推理字段；MuZero 配置 `useMuZero=true` 并传入完整 bundle，需要 metadata 的主模型配置 `humanSLProfile`。第一份通过文件筛选的 SGF 决定模型棋类，显式 `game` 必须匹配；模型延迟到此时加载，并由工作线程共享。没有有效输入时不加载模型。普通模型保留来源 evaluator 默认值，MuZero 检查棋类与表示画布。模型或配置加载失败不会被 `-tolerate-illegal-moves` 当成坏棋谱吞掉；该开关仍用于逐份 SGF 的解析与重放错误。

波动路径使用 SGF 的实际规则，Go 未指定规则时沿用来源 simple-territory 默认；不会自动平衡贴目。每个非终局真实局面执行一次 `skipCache=true` 推理；非围棋真实终局直接使用规则的 W−L，Go 保留来源的网络评估流程。输入历史模式来自模型声明。象棋的主线与分支样本都保留完整前史、原始手数和联合动作，输出可由原生 `hintPosesDir` 消费。

加权公式保持来源实现：以白方 W−L 的相邻差平方为波动，先向前面的手数分配指数平滑权重，再可选地向后平滑；按 count、二元交叉熵 surprise 和 `1−(W−L)²` uncertainty 的配置组合归一化，最后施加单位置上限。这里的 surprise 仍是来源基于 W−L 的二元量，不替换成四分类 KL。中盘起点的手数索引及前置零区间参与来源统计；零波动不产生均匀替代样本。

主线保留来源直接采样的行为，分支按相同手数的主线权重处理，并排除主线已出现的局面。通用的采样概率、手数衰减、pass 因子、训练权重、阈值与注释筛选继续由原有 handler 处理；波动主线传入空注释，`%SAMPLE%` 等强制注释权重因此只作用于后续分支遍历。`-for-testing` 保留固定模型种子、恒等推理对称和排序；并发文件的处理顺序仍受线程调度影响。

[test_samplesgfs.py](../tests/test_samplesgfs.py) 用可手算的四结果常数网络验证双向平滑、归一化、Go NR 掩码、上限与注释权重，检查非围棋真实终局、象棋完整历史进入原生 writer、并行／串行非均匀网络结果、metadata 和输入错误；普通模型另有 Eigen 验证。

## SGF 挖掘与训练起点

`dataminesgfs` 的 `-game-mode`、`-tree-mode` 和 `-surprise-mode` 互斥，均支持六种棋类、普通／MuZero 模型和 PUCT／Gumbel。它仍是来源原生命令：普通模型与 MuZero bundle 共用搜索入口、并发队列和 `PosWriter`，没有外围挖掘或评分实现。

```bash
bash scripts/run.sh native dataminesgfs \
  -config /absolute/path/to/selfplay.cfg -model /absolute/path/to/model \
  -sgfdir /absolute/path/to/sgfs -outdir /absolute/path/to/new-hints \
  -threads 2 -tree-mode
```

配置使用原生 `GameInitializer` 的 `game`、棋盘尺寸与规则字段。挖掘按配置生成规则，保留来源不直接采用 SGF 规则的行为；棋类不符的文件记录原因并过滤。MuZero bundle 必须匹配棋类，画布须容纳配置尺寸。需要 metadata 的主模型使用 `humanSLProfile`。树模式对候选着法执行四次不读缓存的原始推理，再取 `1.1 × 几何平均概率`；整局模式每个局面执行一次不读缓存的策略推理，并沿用来源快搜值的前后平滑。MuZero 的原始策略来自真实局面的 initial，搜索内部才使用 recurrent。

来源的 surprise 筛选、快慢搜索预算、utility 阈值、hint 权重、随机取整、训练权重、去重、分片与文件轮换保持原有公式。整局模式仍只接受 19×19 Go；其他棋类按配置尺寸筛选。昂贵搜索之前仍要求棋盘上至少有六枚棋子。`-manual-hint-only`、`%HINT%`／`%HINTLIGHT%`／`%NOHINT%` 与 `%START%` 的注释筛选用于树／surprise 遍历，整局模式沿用自己的筛选流程。没有达到 surprise 条件时可以正常结束而不写任何样本。

输出 `*.hintposes.txt` 可直接由原生 `hintPosesDir` 消费，也可先用 `viewstartposes -check-legality` 检查。象棋样本保存初始 FEN、完整规则前史、截取点之后的动作和原始手数，策略与 hint 使用联合动作；不会把早期历史截成只剩 FEN。Renju、Hex 和象棋不执行启发式换色。surprise 模式保留来源回退一手、两手样本的 1/2、1/4 权重。停止或解析／搜索失败时先退出工作线程、排空 writer，再释放共享模型；未完成分析不补成有效样本。

[test_datamine.py](../tests/test_datamine.py) 覆盖真实 CUDA 三种模式、手算策略权重、surprise 回退权重、象棋样本进入原生自对弈 writer、metadata、筛选和有界队列取消；普通网络另有 Eigen 验证。
