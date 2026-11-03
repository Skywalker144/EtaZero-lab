# 原生开局库

开局库继续使用 KataGo 的 `Book`、对称归并、minimax、访问统计和扩展成本计算，代码位于 `cpp/book/`；原生命令位于 `cpp/command/genbook.cpp`。库中的价值是白方视角 W−L，分数预测相关字段与成本项按 V4 的约定排除。

## 状态与对称

`Book` 的建节点、查找、重放和存取支持围棋、Freestyle／Standard／Renju 五子棋、Hex、中国象棋。

- 围棋保留来源的重复边界哈希与对称。矩形棋盘只使用不交换长宽的四种对称。
- 五子棋保留八种对称；Hex 只允许恒等与 180°，避免把不同连接方向的局面合并。落子棋保留零气子，单调落子后的身份使用原生棋类状态哈希。
- 象棋使用联合动作及左右镜像；初始状态保存原始 FEN 和完整走子历史。重放、镜像、保存和加载保留长将、长捉、重复及无进展裁定所需的历史，不用末局面 FEN 代替。
- 非围棋另存初始手数；它与规则历史分别保存，支持保留导入起点的训练手数。

这些适配处理实际棋类状态；共有的价值回传与扩展成本公式不因棋类切换而替换。

## 命令边界

| 入口 | 当前范围 |
|---|---|
| `checkbook` | 六棋类；逐节点重放并检查原生 book 哈希 |
| `comparebooks` | 六棋类；来源 W−L 差异统计，象棋路径使用联合动作字符串 |
| `findbookbottlenecks` | 六棋类；保留来源最小改变量计算，输出使用棋类有效对称和动作 |
| `genbook` | 六棋类、普通／MuZero、PUCT／Gumbel；原生扩展、续建、SGF bonus 和 trace |
| `booktoposes` | 六棋类；普通／MuZero 原始推理、来源加权采样、完整象棋起点与 hint 导出 |
| `writebook`／HTML 导出 | 六棋类；原生统计表、有效对称链接、Hex 连接边、象棋棋子与联合动作 |

原生命令直接运行，例如：

```bash
bash scripts/run.sh native checkbook -book-file /absolute/path/to/book.katabook
bash scripts/run.sh native comparebooks -book1 /absolute/path/to/first.katabook -book2 /absolute/path/to/second.katabook
```

`checkbook` 沿用来源的日志报告方式，完整性失败会记录在输出中；不能只凭退出码认定通过。

## 建库、续建与导入

`genbook` 使用来源按成本挑选节点、搜索未入库着法、记录达到访问阈值的子树、补搜叶节点和重算成本的流程。`numGameThreads` 控制并行建库搜索，`numSearchThreads` 控制每次搜索；书中已有的动作通过原生 avoid 接口排除。原生 `BookParams`、随机化参数、保存周期与 trace 阈值继续生效。

```bash
bash scripts/run.sh native genbook \
  -config /absolute/path/to/book.cfg -model /absolute/path/to/model \
  -book-file /absolute/path/to/book.katabook -log-file /absolute/path/to/book.log \
  -num-iters 10 -save-every 2
```

配置保留来源字段；围棋示例位于 `cpp/configs/book/`。`game`／`rules` 选择棋类，`boardSizeX`／`boardSizeY` 指定实际棋盘；落子棋须为正方形，象棋为 9×10。模型与棋类必须匹配。已有 book 自动续建，`-num-iters` 表示本次增加的迭代数；初始局面、棋规、重复边界和历史模式沿用已有 book。`-allow-changing-book-params` 只允许调整来源建库成本参数，不替换这些状态约定。

普通网络使用文件，`useMuZero=true`（或索引 0 的同名字段）使用三组件目录；MuZero 的 NN 缓冲配置须匹配模型 canvas，canvas 须容纳实际棋盘。搜索遵循 [MuZero 的参数边界](muzero.md)，根使用真实规则，内部使用 latent。从搜索树提取开局库时，只重放真实合法的后继；实际终局按棋规写入 W−L，并停止该分支，不请求终局的非围棋策略。这不改变 latent 搜索内部的动作域、回传与统计。

`rootSearchAlgorithm=gumbel` 接入原生 [Gumbel 搜索](gumbel.md)。每次搜索使用新树，普通 PUCT 继续保留来源复用行为。默认 Gumbel 根将 SH 最终赢家作为必录着法；其他节点和达到访问阈值的分支仍按原生记录流程处理。改进训练策略不作为书中 raw policy。

书中策略成本使用原始 NN policy 的有效对称平均：围棋／五子棋八种，Hex／象棋两种；每次推理跳过缓存读取。可选 `-human-model` 沿用来源职责，只提供建库 policy，搜索仍使用主模型。次级文件或 MuZero bundle 均可使用；需要 metadata 的模型必须配置 `humanSLProfile`。主、次模型可以使用不同的 policy canvas，动作按各自输出布局索引。

`-bonus-file` 保留 SGF 中 `BONUS`、`EXPAND`、`VISITS`、`BRANCH` 的来源解释；非围棋 setup 保留零气子，象棋使用完整 `XQFEN`／`XQHIST` 及 `XQTURN`。续建时检查完整初始象棋历史，不能用相同末棋盘替代。`-trace-sgf-file` 沿用来源遍历与终局过滤；`-trace-book-file` 在 `-num-iters 0` 时按访问阈值选择叶节点并重放到达它们的路径，不保证复制图中所有可替代的父边。节点的规范朝向可能旋转非空起点；重放前通过棋类有效对称将整条路径对齐目标库的初始状态，象棋匹配包含完整历史。导入可能补入更高 raw policy 的着法，随后按来源流程补搜和重算。

## 加权导出训练起点

`booktoposes` 使用来源的深度与访问数筛选、最低权重过滤、hint 选择和 `PosWriter`。普通网络使用原生模型文件，`useMuZero=true` 时使用三组件 bundle 目录和原生 initial inference。该命令不运行 PUCT 或 Gumbel 搜索；搜索参数不改变这里的原始推理口径。

```bash
mkdir -p /absolute/path/to/poses
bash scripts/run.sh native booktoposes \
  -config /absolute/path/to/evaluator.cfg -model /absolute/path/to/model \
  -book-file /absolute/path/to/book.katabook -out-dir /absolute/path/to/poses \
  -include-depth 12 -constant-weight 1 -enable-hints
```

配置中的 `game` 可以省略，由 book 提供；显式配置必须一致。MuZero 的棋类和固定 canvas 必须适合该 book。需要 metadata 的模型使用 `humanSLProfile`（或索引 0 的同名字段），缺失时明确报错。推理按 book 保存的 `BoardHistoryModes` 重放，沿用来源开局库的历史语义。

权重保留来源公式：常数项、指数衰减的深度项、最佳书中着法的 policy surprise，以及从 W−L 转换得到的二元价值交叉熵。价值统计仍分别记录不可约熵和 KL 差额；不把它改成四分类 loss。围棋保留八次显式、跳过缓存读取的推理；五子棋使用八种对称，Hex 和象棋分别使用两种有效对称，平均的是 log policy 和交叉熵。没有书中后继着法时，策略项为零；策略和价值权重均关闭时不执行前向推理。

起点保留来源的最多五步显式走子；象棋同时保存快照之前的完整规则历史。非围棋保留初始手数偏移，规则终局不导出为可继续自对弈的起点。输出可直接由原生 selfplay 的 SGF/hint 起点流程消费。权重与各项总量均来自该原生命令日志。

## HTML 浏览

`genbook -html-dir` 在建库结束时导出页面；`writebook` 可对已有书单独导出，沿用来源配置、成本重算、bonus、`-html-min-visits` 和 `-html-dev-mode`。

```bash
bash scripts/run.sh native writebook \
  -config /absolute/path/to/book.cfg -book-file /absolute/path/to/book.katabook \
  -html-dir /absolute/path/to/html
```

直接打开导出目录的 `root/root.html`，或将整个目录作为静态文件服务。各页继续使用 KataGo 的走法排序、raw policy、minimax、访问数、调整后访问数、价值不确定性及开发模式成本表；浏览器不执行搜索或棋规重放。`BWin%` 是黑方结果价值 `P(win) + 0.5 × (P(draw) + P(no-result))`，不是排除和局后的纯胜率；象棋对应红方，列名为 `RWin%`。实际终局另显示胜方、和棋或 no-result。

围棋／五子棋使用交叉点棋盘。Hex 使用与原生六邻接一致的格点几何，黑方连接上下边、白方连接左右边。象棋显示原生状态中的实际棋子；选择有书中着法的棋子后点击目标格，也可直接点击统计表中的联合动作。悬停显示对应路径，坐标使用 `a0` 至 `i9`，红方位于下方；编号徽标不遮住被吃棋子的字形。

页面的 `symmetry` 参数与后继、规范父节点链接使用棋类有效对称；非法显示对称回到恒等朝向。象棋同时变换联合动作的两个端点，Hex 不使用会交换双方连接方向的对称。页面数据由完整原生历史重放后导出，不通过颜色棋盘或末局面 FEN 重建象棋规则状态。

## 验证

`cpp/tests/testbook.cpp` 保留来源围棋开局库回归，并检查六棋类的合法动作、对称归并、文件往返、成本重算和逐节点重放。专项案例覆盖 Hex 不等价反射、零气子、象棋完整历史及继续对局后的重复、长将、长捉、120 半回合裁定。`tests/test_book.py` 调用实际原生命令检查存取结果、差异路径和瓶颈分析；这些检查不需要神经网络或 GPU。

`tests/test_genbook.py` 检查六棋类建库、续建、trace 与 bonus，使用真实 CUDA 普通／MuZero 模型核对原始 policy 概率、深层 latent 提取和真实终局裁定，并覆盖次级模型及 metadata。Eigen 专项使用同一概率参考检查普通网络的 Gumbel 建库。建库输出均通过原生逐节点重放检查；这些检查不代表开局库质量或棋力结论。

`tests/test_booktoposes.py` 检查来源权重公式、筛选、有效对称、metadata、终局过滤和象棋完整历史，并使用真实 CUDA 普通／MuZero 模型及原生自对弈 writer。Eigen 使用同一权重参考检查普通网络。原始推理与搜索是分开的验证范围。

`tests/test_book_html.py` 使用实际原生导出和 Chromium，检查六棋类的统计、后继与父节点跳转、对称等价着法、访问阈值、移动端布局，以及象棋镜像吃子、重复／长将／长捉终局和围棋 pass。Hex 棋盘几何另按六邻接距离核对。HTML 检查不需要训练或神经网络推理。
