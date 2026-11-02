# Web 工作台与原生会话

V4 的会话由常驻 `katago gtp` 进程持有，棋规、合法着法、落子、悔棋、搜索和模型生命周期均使用本版本原生实现。外层传输位于 `python/etazero/gtp.py`，模型发现位于 `python/etazero/web_catalog.py`，会话动作和异步控制位于 `python/etazero/web_session.py`、`python/etazero/web_app.py`。仓库 `web/native_server.py` 连接控制器和 `web/static/native.html`／`native.js` 页面，共用版本无关的 `web/http.py` 传输。`web/webui.sh` 按所选版本路由到对应的服务和页面。

不使用 V3 的 TorchScript manifest 或私有 JSON 会话协议。普通网络加载原生 `.bin`／`.bin.gz`，MuZero 加载三组件 bundle；配置仍由原生 `ConfigParser` 读取，算法、设备、精度、HumanSL 和搜索组合约束由原生消费者检查。传输层以配置文件所在目录为工作目录，只覆盖 `logToStdout=false` 保持 GTP 输出可解析，不在 Python 改写搜索参数或实现棋规。

## 启动与原生配置

从仓库根目录执行：

```bash
# 默认使用最高版本，扫描该版本的 data/；进入 pytorch 环境并增量构建。
bash web/webui.sh

# 显式选择 V4，限定模型目录并指定独立推理配置。
bash web/webui.sh --version EtaZero_V4 --models-dir /absolute/path/to/runs \
  --config /absolute/path/to/gtp.cfg

# 直接加载原生模型；MuZero 的 --model 指向三组件 bundle 目录。
bash web/webui.sh --version EtaZero_V4 --model /absolute/path/to/model.bin.gz \
  --config-dir /absolute/path/to/inference-config --no-open
```

`--config-dir` 使用该目录的 `gtp.cfg`，与 `--config` 二选一。`--model` 可选择已发现的发布模型，也可指定目录之外的模型；后一种情况必须明确配置，页面不补造其训练代数、算法或累计时间。`--human-model` 原样交给原生 GTP，支持普通次级模型或 MuZero bundle。`--binary` 可显式使用本版本的已构建程序并跳过自动构建；`--help` 不触发构建。默认地址为 `http://127.0.0.1:8766`，关闭服务会关闭当前原生会话。

四种算法 baseline 和各棋类入口均提供独立 `gtp.cfg`。公共参数见 [baseline_az/gtp.cfg](../configs/baseline_az/gtp.cfg)，其他算法继承公共配置并覆盖约束，棋类入口再包含 `games/` 下的原生 GTP 片段。使用原生 `key = value` 与 `@include`，不读取带节的训练配置、V3 的 `analysis.cfg` 或 `ANALYSIS_*` 字段，也不自动读取训练配置的 `.local`。自定义运行目录仍需有对应的 `gtp.cfg`，或用 `--config`／`--config-dir` 明确指定；缺少配置时页面显示未配置，创建按钮不可用。

公共推理参数参考固定 KataGo `d91ea855` 的 GTP 示例和源码默认，显式启用 optimistic policy、uncertainty、noise pruning 和 PUCT 波动调整；根对称采样保留项目约定并由棋类限制有效数量。推理预算独立于 selfplay，根随机噪声与 forced playout 关闭。`analysisWideRootNoise` 是仅供分析的候选拓宽参数，与对弈根噪声分开；Gumbel 片段将两者都关闭，保留 sequential halving，关闭 LCB 选点和访问修剪。MuZero 另外关闭真实棋盘 graph、SVB 和重复棋形奖励。

Gomoku 片段参考固定 KataGomo `df152116` 的 [比赛 GTP 配置](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/scripts/gomocup/default_gtp.cfg)，采用其 PUCT log 系数、NN 策略温度、SVB、搜索线程及交互行为设置；普通网络使用其缓存容量，MuZero 仍禁用缓存。棋规使用 V4 已选定的 Freestyle，保留独立 D/NR 语义及 V4 的 optimism／uncertainty，不复制 Renju/VC 专用字段或 NR 偏好。各棋类温度半衰期沿用项目约定；默认 Gumbel SH 选点不使用落子温度。这些是来源适配后的起始配置，未据此宣称棋力或设备吞吐最优。

GTP 是交互推理入口，正式 `match/autoelo` 仍使用独立比赛配置。公共配置保留来源的领先时缩减搜索预算行为；需要固定计算量的比较应在比赛配置中明确关闭这类倍率。日志输出到 stderr，避免在配置目录内生成日志文件；Web 会话由传输层接收 stderr。

- Go 使用对应的 `rules` 和 `defaultBoardSize`，贴目仍由原生配置或创建局面的覆盖指定。
- 五子棋／Hex 的 `game`、`rules` 使用对应棋类名，`defaultBoardSize` 设置实际方形棋盘；象棋使用 `game=xiangqi`、`rules=xiangqi`，物理尺寸由原生默认的 9×10 决定。
- `rootSearchAlgorithm` 决定 PUCT／Gumbel，`nonrootSearchAlgorithm` 独立指定非根机制；Gumbel 要求 `useGraphSearch=false`，其他组合约束见 [Gumbel](gumbel.md)。不会从训练算法标签推断本次评估搜索。
- MuZero 使用 `useMuZero=true`、`useGraphSearch=false`、`subtreeValueBiasFactor=0`、`avoidRepeatedPatternUtility=0`，其余约束见 [MuZero GTP](muzero.md#原生-gtp)。模型格式、画布和棋类仍由引擎检查。GTP 公共配置允许小于 NN 缓冲区的棋盘；显式开启 `requireMaxBoardSize` 时棋盘必须与缓冲区一致，棋盘也不能超过缓冲区。尺寸不匹配会在设置棋盘时拒绝。

## 页面与数据来源

页面支持六种棋类的棋盘、人机／手动模式、AI 单步、可取消的持续分析、悔棋、历史回看与分支、模型刷新和切换、原生参数修改，以及浅色／深色和窄屏布局。围棋棋盘保留实际提子；Hex 标示黑方上下、白方左右的目标边；象棋使用原生棋子与合法起终点，红方在下。左栏按模型、棋盘与棋规、对局模式的顺序设置，选定模型后自动创建棋局，后续更改自动生效。滑动条从运行保存的 `native_selfplay` 快照读取训练时有非零采样权重的方形尺寸；直接模型或缺少该快照时可手动输入，留空使用原生配置。训练过的尺寸不等同于模型全部可推理的尺寸，实际画布和棋规仍由引擎检查。象棋固定为 9×10；五子棋可选 Freestyle、Standard、Renju。

根 W/D/L/NR、得分率、访问数、候选及 PV 均来自原生分析 JSON，页面标明来源报告的 B/W 视角；W−L 由相同根 W 和 L 相减。候选表展示原生 `visits`、`edgeVisits`、先验、LCB、效用和 `playSelectionValue`；它们保留各自尺度，选点值不冒充归一化概率或 Gumbel 改进策略。星号只标记原生搜索返回的建议着法。棋盘候选按原生 `order` 取 Top-K，最低访问过滤以根访问数为分母；对称副本保留原生报告含义，不累加成额外搜索量。

策略面板可在来源 `policy` 与次级模型的 `humanPolicy` 之间切换，直接显示原生数组，包括非法动作的负值标记和非围棋被屏蔽的 pass 槽；象棋从唯一的 2086 动作表按起点筛选。当前报告没有 `humanPolicy` 时明确提示缺失，不用主模型策略代替。切换仅改变展示，不触发额外推理、搜索或参数修改，原始 JSON 面板保留整个报告。没有来源字段的搜索时间或 NN 请求数不会补造。象棋逐手移动动画仍待接入。

“原生网络诊断”独立于搜索报告，通过按钮调用现有 `kata-raw-nn <symmetry> <optimism>` 或 `kata-raw-human-nn <symmetry>`。它支持六棋类的普通／MuZero 主、次级模型，保留原生文本，并由外围 `python/etazero/web_raw_nn.py` 解码展示。诊断始终对当前真实局面执行一次指定对称的原生推理，MuZero 使用 initial；不进行搜索或落子。切换选项和刷新页面不触发推理。缺少次级模型、非法对称或参数由原生入口明确拒绝，不能回退为主模型。终局不请求网络诊断。

诊断概率是原生合法性处理后的结果，不是 logits。`optimism=0` 与 `1` 分别请求普通与乐观策略，中间值保留原生混合方式；实际可用输出取决于模型，带 metadata 的 HumanSL 沿用其原生处理。次级命令不接受 optimism，使用引擎自身的参数语义。来源导出只包含用于推理的策略通道，页面不冒充已取得全部训练策略头，也不通过取对数反推原始 logits。概率保留原生六位小数及 `NAN` 屏蔽标记，外围不重新归一化；热图颜色按当前最大概率缩放。W/D/L/NR 明确使用原生白方视角（象棋为黑方），不随搜索报告视角切换。其他标量保留在原始文本中。

诊断快照附带其局面、会话手数和实际命令。落子后保留原快照并标明局面差异；可以定位回看，需分支到当前局面后才能对历史位置重新推理。悔棋或原生参数修改清除诊断，创建新会话也不沿用旧模型结果。持续分析中请求诊断时，控制器先停止并读完旧搜索响应，执行有限的原生命令，再恢复持续分析。诊断不清除已有搜索树；原生 raw-NN 的跳过缓存读取和写入缓存行为照常保留，不能把它理解为完全不影响推理缓存的纯内存查询。

人机模式可设置人类先后手、后台 pondering，以及每手新增 visits、总 visits 或思考时间。新增 visits 使用原生 `maxPlayouts`，总 visits 使用 `maxVisits` 并包含复用的根访问，时间使用 `maxTime`；未选用的预算解除限制，来源搜索倍率及并行搜索语义仍然生效。Gumbel 的顺序淘汰配额由访问上限计算；只限制时间时可能在完成淘汰前中止，不等同于已知访问预算的完整顺序淘汰。后台思考在等待人类落子时启动，使用配置中的 pondering 上限。Gumbel 每次搜索使用新树，MuZero 不跨真实落子复用 latent 树，因此 pondering 不保证为下一手保留树访问，详见 [Gumbel](gumbel.md) 和 [MuZero](muzero.md)。

键盘支持 `A` 开关分析、`S` 单步、`U` 悔棋、`Esc` 停止搜索、左右键回看、`Home` 初始局面和 `End` 当前局面。刷新页面保留服务端棋局，多个标签页共享一个会话；主题与手数显示保存在浏览器。

## 发布模型目录

`discover_catalog` 递归发现同时包含 `config.json` 与 `state.json` 的 V4 运行目录。每次读取一次 `completed_rounds`，只列出这一边界内 `rounds/…/commit.json` 记录的发布模型，不扫描未提交的候选或 `.tmp`。原始配置提供棋类、算法、画布和普通／SWA 权重来源；模型代数来自首次记录它的提交轮次，同一个被复用的 accepted 模型不重复列成新一代。

已提交但被 gatekeeper 拒绝的候选仍可检查，目录数据记录其接受状态；默认当前模型来自提交一致的 `accepted_model` 指针。尚未发布模型的运行保留空目录记录。损坏或缺少文件的运行进入显式 `errors` 列表，不妨碍其他运行被发现；发现过程不修改原始运行或加载 GPU 模型，实际二进制结构和搜索组合由原生命令加载时检查。

独立推理配置采用原生 `gtp.cfg`，可传入全局配置路径覆盖；否则先按本版本 `data/` 相对路径映射到 `configs/…/gtp.cfg`，再查找已保存 `selfplay_config` 所在目录的 `gtp.cfg`。找不到时保留为未配置，不静默选择围棋或其他算法的配置，不把训练配置重写成推理配置。配置内容与 include 不在发现阶段解析。

训练时间只显示有效提交中已有的累计计时，明确标注 EtaZero 外层来源；缺测为 null，不补零。该目录服务于交互模型选择；正式等时间 Elo 仍使用 `arena`／`autoelo` 的时间证据校验与配对流程。`tests/test_web_catalog.py` 检查四种算法、提交边界、被拒候选、复用模型、缺测计时与损坏运行的可见错误。

## 会话控制

`web_app.App` 消费上述目录，每个工作台持有一个当前 `web_session.Session`，由单个工作线程顺序执行动作。创建棋局使用目录中的原生模型路径和 `gtp.cfg`；尺寸、规则、贴目、象棋 FEN 及运行参数分别交给原生 `boardsize`、`kata-set-rules`、`komi`、`set_position` 和 `kata-set-params`。参数名、校验与实际取值由原生入口管理，外层返回 `kata-get-params` 和 `kata-get-models` 的原始结果，不按设备猜测精度或将配置重写成旧版字段。

`new`、`apply`、`play`、`undo`、`branch`、`configure`、`step`、`retry`、`analyze`、`diagnose`、`stop`、`refresh` 覆盖创建、手动落子、悔棋、历史分支、修改参数、AI 单步、重试、持续分析、网络诊断、停止和目录刷新。`diagnose` 接受 `source=main|human`、整数 `symmetry` 及主模型的 `optimism`，省略时为主模型、对称 0、optimism 0；快照保存在 `raw_nn`，没有成功诊断时为 null。人机模式在创建棋局、页面应用设置、用户落子或显式重试后，若轮到 AI 则自动执行 AI 回合；底层 `configure` 和悔棋不自动出手，历史分支转为手动模式。人机悔棋回到最后一次人类落子之前，手动模式撤回一手。浏览器请求携带状态版本及棋局 ID，旧棋局的动作会被拒绝；分析快照的更新不使同一局面的合法操作失效。

有限搜索使用来源可取消的 `kata-search_analyze_cancellable`，完成后才用原生 `play` 落下建议着法。停止与最终落子的提交有明确先后：先接受的停止请求不会落下部分搜索的建议。持续分析期间可提交落子、悔棋、参数修改等操作，控制器先停止并读完旧响应，执行操作，再在当前局面恢复分析。旧任务的报告不能覆盖新棋局；关闭工作台会取消搜索、等待响应结束并回收进程，长轮询也会被唤醒。

每次成功落子保存原生棋盘快照供历史回看，围棋提子与象棋棋子类型不靠前端重放推测。快照不重复保存此前全部着法，空间开销随手数线性增长；实际悔棋仍执行原生 `undo`，不从 FEN 重建完整规则历史。分析保留原始 JSON，并附其所属局面、会话手数与运行／完成／停止状态，落子后的新棋盘不会被误认为旧报告的搜索根。GTP 的 `resign` 单独记录认输方、结果和响应来源，不改写原生 `isGameFinished`；撤回可清除这一记录，未搜索的终局数据不会被补造。

模型切换先用原生 `printsgf` 导出棋局，在新进程中载入后核对局面和逐手历史，成功后替换当前会话并关闭旧进程；认输和开局来源随会话保留。改变尺寸或棋规则创建新局。设置失败保留旧棋局；准备阶段会暂时同时驻留新旧模型。页面合并连续修改，等待正在加载的操作结束后应用最新设置。目录刷新不重载活动模型，即使其训练目录已移动，现有会话仍可继续。原生断开会保留最后已知局面并显示错误及断开状态，需重新创建会话后继续操作。

当前控制器使用原生坐标和参数，不接受旧私有协议字段。平衡开局使用独立的 `opening` 参数对象，未知字段由原生命令拒绝，不被静默忽略或替换为空盘。

HTTP 传输只组织 `/api/state`、`/api/catalog` 与上述动作，不导入版本特有的引擎、模型或配置模块；V3 与 V4 分别注入自己的控制器。V4 的 HTTP 状态直接保留原生局面、配置和分析字段；不适配为旧私有引擎格式。状态长轮询只读取控制器缓存，不通过 GTP 查询打断活动搜索。V4 页面直接消费这一原生格式，旧版页面继续使用其原有控制器。

## 平衡开局

在左栏点击“生成平衡开局”，会用当前选择创建一局并生成开局。只支持黑先空盘的 Freestyle／Standard／Renju 和 Hex，要求真实主模型；AlphaZero、Gumbel AlphaZero、MuZero、Gumbel MuZero 调用同一套 [原生算法](openings.md)。页面使用 `gtp.cfg` 或原生缺省的开局参数；HTTP API 可通过 JSON 覆盖 `balanceOpeningDropPow` 和 Hex 专用的 `minAcceptRate`。生成不执行训练中的 policy init，不使用次级 HumanSL 作为参考模型。

HTTP `new` 的 `opening` 是参数对象，`{}` 表示使用缺省设置，省略表示普通创建。它与 SGF、SGF 手数和 FEN 覆盖互斥；尺寸、棋规、贴目及搜索参数仍可设置，实际棋类与合法性由原生检查。原生 GTP 扩展为：

```text
kata-suggest-opening
kata-suggest-opening {"balanceOpeningDropPow": 6, "minAcceptRate": 0.005}
```

命令先停止活动搜索，再在独立棋盘／历史上调用 `BalancedOpening::initialize`。正常返回 JSON，包含 `status=complete`、`game`、按原生 GTP 坐标表示的 `moves` 和实际 `parameters`。它不替换当前棋盘或已有搜索树；客户端可选择是否逐手执行 `play`。不支持的棋类、非空起点、无模型或非法选项返回 GTP 错误，进程仍可使用。

命令运行期间，任何后续输入行（包括裸换行）或输入 EOF 都请求取消并等待工作线程退出；取消响应为 `status=cancelled` 和空着法列表。应等当前响应结束后再查询棋盘，不能将额外查询当作生成期间的轮询。外围对该命令允许无限等待，不为来源拒绝采样增设时间／重试上限；停止仍通过来源 GTP 的换行机制执行。

Web 在新候选会话中请求建议，并用原生 `play`、`kata-get-state`、`kata-get-history` 执行与读取完整前缀。成功后才替换原会话；失败、停止或关闭工作台会回收候选，原棋局保持原样。生成期间页面显示“停止生成”，首次创建尚无棋局时也可取消。完成后的所有开局着法都可回看、撤回和分支；`opening` 记录创建时的原生建议，撤回后仍作为起点来源保留，不冒充当前棋盘的着法列表。

这条路径只增加需要当前模型和原生棋规的 GTP 入口；Python 和页面不实现候选采样、价值换算或棋规。`tests/test_web_opening.py` 使用真实普通／MuZero 模型验证建议、原生历史、取消与进程回收，并通过浏览器检查创建及历史分支。

## SGF 导入与回看

创建棋局时可传入 `sgf` 文本，`sgf_turn` 可选，表示载入多少手之后的局面（0 表示 SGF 初始局面，省略载入全部）。外层只写入临时文件并换算为原生 `loadsgf` 的下一手编号；主变化选择、规则、贴目、setup stones、合法性和完整前史均由来源读取器负责。超过棋谱手数时沿用原生行为载入到末尾。SGF 与尺寸、棋规、贴目、FEN 的创建覆盖互斥，避免导入时无声覆盖用户设置；这些字段应留空，缺省规则仍来自原生配置。

原生 `kata-get-history` 从当前 `BoardHistory.copyToInitial()` 起，使用与 `loadsgf` 相同的 tolerant replay 返回紧凑局面快照。它不改变引擎局面或搜索树，不执行推理，象棋起点保留已有的重复、将军和追捉历史。快照不重复着法列表或合法动作域。该命令读取当前搜索历史；GTP 中人为改变行棋方造成的历史重置仍按来源执行，不能将其等同于任意 GTP 指令序列的完整操作日志。Web 在 SGF 导入后使用它初始化回看，后续逐手沿用正常会话快照。

SGF 的所有已载入着法都可以回看、悔棋和继续分支，初始 setup／象棋前史属于起点。连续同色落子的 Go 棋谱在原生悔棋重放时可能重置规则历史和手数，外围保留这一行为，并用实际悔棋后的状态更新当前快照。创建使用新的候选会话，解析失败、棋类不匹配或模型不支持时保留旧棋局。服务接收 UTF-8 文本，上传文件名不进入 GTP 指令；V4 的整个 HTTP 动作请求上限为 4 MiB，超限明确报错，旧版请求上限不变。

## 局面与分析查询

`kata-get-state` 是只读的结构化局面查询，不执行神经网络推理，返回：

- `game`、原生 `rules`、实际 `boardXSize`／`boardYSize`、`currentPlayer` 与 `turnNumber`；`analysisPerspective` 是原生配置在该局面解析出的报告视角（B/W），当前方模式随局面行棋方变化。
- `board`：物理棋盘按原生坐标逐行排列，0 为空，1 为先手方，−1 为后手方。落子棋 y=0 对应最高行；象棋 y=0 对应红方底线。
- `legalMoves`：当前行棋方的完整规则合法动作，使用原生 GTP 坐标；真实终局为空。Renju 仍按规则允许落下禁手并判负，不将其提前伪装为非法着法。
- `moves`：本 GTP 会话可撤回的着法及执子方。FEN 或 SGF 起点不凭空添加为用户落子；完整规则历史仍保存在引擎内部，不能仅凭当前棋盘或 FEN 重建。
- `isGameFinished`、`isNoResult`、`isResignation`、`result` 和 `encorePhase`。未终局的 `result` 为 null，不用网络估值判断终局。
- 象棋另有 `fen` 与 `pieces`，后者按物理 9×10 棋盘排列，正负区分红／黑，绝对值沿用原生 Pikafish 棋子编码。

Web 在读完棋盘后通过 `kata-start-pondering` 重启后台思考；该命令要求已启用 pondering，且不会执行落子。

GTP 的 `genmove`／`kata-search` 返回 `resign` 是引擎的认输建议，来源不会把这条响应写入 `BoardHistory`；外层需要按 GTP 对局语义处理，不能误称它为规则自动终局。已有 `play`、`undo`、`set_position`、`loadsgf`、棋钟及参数命令保持来源行为。

`kata-get-analysis` 读取已有搜索树，直接使用 `Search::getAnalysisJson`，输出与原生 JSON analysis 相同的 `rootInfo`、`moveInfos`、`policy`、可用的 `humanPolicy`，并包含 `turnNumber`。它不重新搜索；当前没有搜索结果时返回 GTP 错误。局面、历史和分析查询沿 GTP 规则先停止活动搜索，不能在持续分析期间将查询当成无副作用的轮询。

所有 KataGo 风格的分析命令可添加 `json true`：

```text
kata-analyze B 25 json true pvVisits true noResultValue true
kata-search_analyze_cancellable B 25 json true pvVisits true noResultValue true
```

间隔仍以来源的百分之一秒为单位。JSON 模式只是替换分析快照的序列化方式，不改变搜索、预算、选点和停止路径；GTP 响应头、结束空行，以及有限搜索的 `play <move>`／`play cancelled` 响应均保留。分析 JSON 始终包含来源根统计和策略，`pvVisits`／`pvEdgeVisits` 任一开启时包括两种 PV 访问数；`noResultValue` 控制候选 NR 字段。JSON 使用来源自己的候选列表，不接受文本模式的 `minmoves`／`maxmoves` 截断选项；LZ 格式不接受 `json`。

来源点棋盘 JSON 的 `policy` 长度为实际面积加 1，末项是 pass 槽位；五子棋／Hex 的该项为非法标记，不能作为可落子的策略头。象棋策略长度为固定 2086 联合动作。`winrate` 是和棋与 NR 各计半分的得分率，W/D/L/NR 另有原生根字段；`reportAnalysisWinratesAs` 决定报告视角，不能默认所有值都属于当前行棋方。MuZero 的 PV 是 latent 动作序列，不能当成真实棋盘上的合法续着重放。

## 传输、取消与验证

`GtpProcess` 使用原生请求编号与 GTP 空行边界，每次只允许一个外层命令消费输出。持续快照经回调交付，只保留最后一份，不无限累计历史报告；持续分析停止并读完响应后，再调用原生 `kata-get-analysis` 获取最终树，避免把停止前的周期报告误标为最终访问数。当前实现的回调不可重入执行另一个命令。另一个线程或报告回调可调用 `stop()`，它发送来源支持的空行，打断持续分析或 cancellable 搜索并完整读完原响应。停止本身没有额外响应，不会污染下一条命令。

原生拒绝命令会抛出 `GtpError`，进程可继续使用。响应超时、协议损坏或回调失败后关闭该进程，避免将未读完的旧响应交给新命令；超时按等待下一行的时间计算，持续收到报告的分析不会因总运行时间而超时。空闲或已经退出的进程不会收到取消输入。

`tests/test_web_native.py` 检查六种棋类的原生局面、搜索与查询一致性、真实终局、围棋提子、象棋完整历史、悔棋、持续／有限搜索取消、错误后继续使用和进程清理。真实 CUDA 对照覆盖普通／MuZero、PUCT／Gumbel、FP32／FP16，将 GTP 中的网络证据与原生 JSON analysis 的同局面输出比较。

`tests/test_web_session.py` 通过真实 GTP 进程检查会话动作、人机回合、历史分支、报告视角、版本冲突、配置拒绝、取消与重试、分析中修改局面、进程断开恢复及关闭清理。CUDA 检查覆盖四种算法在围棋、Renju、Hex、象棋上的会话对局与回退，以及原生认输和普通模型到 MuZero 的切换。`tests/test_web_http.py` 用本机临时 HTTP 服务检查六种棋类的动作、长轮询、取消、冲突响应及不支持功能的明确拒绝；浏览器交互另由 `tests/test_web_browser.py` 检查，覆盖六种棋类的 PUCT／Gumbel 页面操作、原始报告一致性、历史、参数、刷新、主题与窄屏，并在真实 CUDA FP16 上覆盖 Go／Renju／Hex／象棋的四种算法。`tests/test_web_sgf.py` 逐前缀对照原生 SGF 读取结果，覆盖六种棋类、提子、象棋前史、失败后继续及分支，以及四算法的 CUDA 搜索与回退。

`tests/test_web_raw_nn.py` 对照六棋类原生文本、真实合法动作及现有搜索树，检查 PUCT／Gumbel 诊断、持续分析恢复、历史定位、终局和缺失次级模型。真实 CUDA FP16 检查使用不同的普通／MuZero 主、次级模型（Go 次级含 metadata），分别核对两端 optimism 与 HumanSL 原始响应；浏览器检查热图／象棋动作列表、屏蔽标记、原始文本和窄屏布局。
