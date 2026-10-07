# 固定评估与等时间 Elo

`analysis` 做局面搜索和常驻交互，`match` 做一对模型的可恢复比赛，`arena` 选择历史模型、安排比赛并输出 Elo。所有公开动作使用实际棋盘零起始行优先编号，native 内部才转换为网络画布编号。分析和比赛必须用 `--model` 或 `--run-dir` 明确指定模型来源，不从配置目录猜测训练目录。模型由 manifest 校验身份与内容，同场模型画布须相同，网络宽度和深度可以不同。训练直接发布导出模型，不使用 gatekeeper；训练 validation loss 独立于分析和比赛。

## 独立配置与搜索预算

[engine.cfg](../configs/baseline/engine.cfg) 的 `[engine]` 定义公共搜索、推理、棋盘和种子默认值。[analysis.cfg](../configs/baseline/analysis.cfg) 的 `[analysis]` 只覆盖分析用途的差异；[match.cfg](../configs/baseline/match.cfg) 的 `[match]` 增加局数、对局并发及必要的模式覆盖，`[opening]` 管理比赛开局。两种模式共用字段定义、校验和 C++ 搜索参数转换，解析后只保存本次模式的完整生效配置。配置改变不影响训练配置身份。

在文件首部用 `@include engine.cfg` 或 `@include ../baseline/analysis.cfg` 显式引用公共文件和父模式；相对路径以引用文件为基准，含空格的路径加引号。引用按顺序读取，再应用当前文件。`[analysis]` / `[match]` 覆盖公共 `[engine]` 值。随后依次应用所选文件目录的 `engine.cfg.local`、所选模式文件的 `.local`、公共 `ENGINE_<字段>` 环境覆盖、模式 `ANALYSIS_<字段>` / `MATCH_<字段>` 覆盖；开局使用 `MATCH_OPENING_<字段>`。父文件的 `.local` 不自动传播。缺失文件、引用循环、未知或重复字段明确报错。训练 `run.cfg` 的继承链和训练环境变量不参与加载。

命令可用 `--config /path/to/analysis.cfg` 或 `--config /path/to/match.cfg` 直接加载独立文件，也可用 `--config-dir` 选择对应模式文件所在目录；两者互斥。字段事实源是 [engine_config.py](../python/etazero/engine_config.py)。

根访问上限由所选评估／比赛配置指定；根初始访问和完成的边访问合计为 `root_visits`。`initial_visits` 为搜索前保留量，`new_playouts` 是本手新增访问（含首次根初始化），`simulations` 为本手新增根边模拟。不同 bot 的 Match 默认复用推进命中的子树，相同模型身份双方强制每手清树；同身份必须对应同模型路径。固定局面 analysis 默认不复用，新的 500v 根通常为一个初始化和 499 个边模拟。`max_playouts` 独立约束新增访问，`max_time` 从开始搜索计时，时间停止保留至少两个新 playout，显式停止可阻止全部搜索。在途路径完成后返回；详细边界和严格并行发放与来源的差异见 [搜索预算](algorithms.md#puct-与搜索预算)。

NN 单朝向默认随机 D4，cache 命中复用首次 canonical 输出，根多对称绕 cache。落子温度使用来源 Match profile 的半衰期调度；policy target 不乘落子温度。analysis/match 的 `inference_precision=auto` 在本实现指定 CUDA 时使用 FP16、指定 CPU 时使用 FP32，结果记录实际精度，显式 FP32 保留。网络缓存、终局节点和根集成使 NN 请求数不等于 visits。

评估和比赛均无训练根噪声；落子温度和搜索线程数由各自配置指定。零温度时，并列最大行为权重按来源选择首个已分配子边。FPU、子树价值加权、policy target pruning、LCB、根多对称、根／全树 policy 温度与落子温度半衰期由独立 profile 显式配置；评估及比赛落子使用剪枝和 LCB 后的权重，原始 visits 单独输出。WDL 搜索 Q 为 W−L。比赛共享组批 evaluator、多局线程的结构参考 KataGo match，使用 EtaZero LibTorch 后端，不宣称复制 KataGo 的全部比赛功能或数值行为。

## 比赛与续测

```bash
# 在 EtaZero_V1.4/ 下，run-dir 用于寻找当前模型和保存默认输出。
bash scripts/run.sh analysis --config configs/baseline/analysis.cfg --run-dir data/my_run --size 15 --rule renju --moves 112,113
bash scripts/run.sh match --config-dir configs/baseline --model /path/to/a/model.pt --model-b /path/to/b/model.pt --games 40 --output data/my_match
```

单局面结果保存到运行目录 `analysis/<标识>/`，包含生效配置、模型和二进制身份、请求以及原始结果；`--output` 可另指定结果 JSON 路径。搜索 WDL 是局面预测，比赛胜率来自真实赛果。搜索建议点沿用配置落子温度，可能不是选择权重最大的点。

`analysis --stream` 启动与 Web 共用的常驻原生引擎，权重和推理服务跨请求复用，独立局面搜索每次清树。协议为 stdin 每行一条命令、stdout 每行一个 JSON 响应；它是同步会话协议，未实现 KataGo 的异步 JSON 任务队列。支持 `new <size> <rule>`、`play <action>`、`analyze <visits>`、`genmove <visits>`、`state`、`undo <count>`、`quit`；`analyze` 返回分析而不落子，`genmove` 返回相同分析并执行建议动作。Web 还可附带比赛开局配置以生成平衡开局。

```bash
bash scripts/run.sh analysis --config configs/baseline/analysis.cfg --model /path/to/model.pt --stream
# 输入示例：new 15 renju，然后 analyze 100、play 112、genmove 100、quit。
```

每对模型局数为四的正倍数。以A或B作为参考黑方的开局各占一半（任务 `generator` 表示参考黑方模型），另一个模型为参考白方。每次balance尝试随机选参考botB/botW评估两个根视角与全部候选；可选policy init每手按当前棋盘执色选参考模型。参数遵循固定KataGomo：Match平衡指数10，policy init默认关闭，开启时显式mean、温度缺省1。开局记录保存参考黑方及实际balance/policy模型索引（0黑/1白）。同一个已生成开局交换A/B执黑／执白下两局，棋盘颜色和Renju规则不变。这是EtaZero的成对换色协议，第二侧不重新生成开局；与KataGomo逐局独立初始化区分。只接受成功且非终局的开局。无认输、无提前截断，按真实棋规结束。每局保存完整落子、胜者、时间与逐手根访问数。

开局和单局完成后分别原子落盘，重启只安排缺失的一侧或新开局。中断的半局重跑，同开局已完成的一侧保持不变。相同输出目录可以用 `--games` 增加成对开局，不能减少目标；模型、搜索条件、种子、二进制或配对身份变化时使用新输出目录。训练采用整轮提交，比赛采用逐开局／逐局提交，两者边界独立。

## 历史模型与 Elo

```bash
bash scripts/run.sh arena --data data/my_experiment --output data/my_elo --dry-run
bash scripts/run.sh arena --data data/my_experiment --output data/my_elo
bash scripts/run.sh arena --data data/my_experiment --output data/my_elo --fit-only
```

`--data` 可以是包含各臂的父目录，也可以是单个运行目录。只发现 state 已提交的模型；包含首个可用、按 `--stride` 间隔选择的模型以及最后模型。组内近邻与跨组邻近训练时间配对组成连通比较图。默认参数见 [arena.py](../python/etazero/arena.py)，`--dry-run` 不创建输出或启动比赛。固定的赛程重启续测；训练继续产生新模型后重新发现的赛程与旧赛程不同，应使用新的评估输出目录。

横轴为模型所属轮次的 `elapsed_seconds`，即已提交完整轮次扣除 `torch.compile` 编译后的累计训练墙钟。暂停和作废轮次不进入横轴；恢复后的重跑成功轮只计一次，编译前完成的自对弈仍计时。计时字段与历史结果口径限制见 [运行观测与证据](implementation.md#运行观测与证据)。图上连线用于展示实测模型点，不代表测量了中间时刻棋力，不向训练停止后外推。原始活动时间仍在训练事件日志中，不能与这个横轴混用。

评分与区间沿用 MuZero_V2 的 [elo.py](../python/etazero/elo.py)：联合 Bradley–Terry 得分模型，和棋记半分，同时拟合共同先手优势；anchor 只规定相对 Elo 零点，不代表外部等级分。弱高斯正则防止全胜／全负导致无穷评分，强度随输出记录。95% 区间按配对分层、以同开局两局为单位 bootstrap；全胜配对会标记先验敏感性。区间条件于已选择的模型、开局与协议，不表示独立训练种子的不确定性。所有开局的两侧和整个赛程完整后才拟合。

输出 `elo.json`、`elo.csv`、`elo.png`、`elo.svg`，并保存模型／配置／二进制身份、赛程、逐开局和逐局原始记录。`--fit-only` 从已有赛果重建评分，不启动 GPU 比赛。

## 自动实验 Elo

[scripts/autoelo.sh](../scripts/autoelo.sh) 使用 [autoelo.py](../python/etazero/autoelo.py) 规划固定轮次采样与增量赛程，复用 arena 的 C++ 对战、逐局恢复、联合拟合和绘图。`autoexp` 在所有调度臂成功达到预算后默认运行；`exp.cfg` 的 `experiment.autoelo = false` 或 `AUTOELO=false` 可关闭。训练中断或失败时不启动 Elo。评估失败明确报错并单独保存状态，已完成训练不重跑；原命令重启会跳过完成训练并恢复评估。

```bash
# 在本版本目录执行：仅发现实际已训练的数据臂。
CONFIG_DIR="configs/<实验伞目录>" bash scripts/autoelo.sh --data "data/<实验伞目录>" --dry-run
CONFIG_DIR="configs/<实验伞目录>" bash scripts/autoelo.sh --data "data/<实验伞目录>"

# 不传 --data 时使用伞目录所有配置臂及其真实 run_dir，缺测臂明确报错。
CONFIG_DIR="configs/<实验伞目录>" bash scripts/autoelo.sh

# --output 指已有具体评估目录；不启动比赛或检查 native binary。
bash scripts/autoelo.sh --fit-only --output "data/<实验伞目录>/elo/<评估标识>"
```

设置在伞目录 `elo.cfg` 的 `[elo]`；默认值以 [baseline/elo.cfg](../configs/baseline/elo.cfg) 为准。优先级为对应命令行参数 > `ELO_<字段大写>` 环境覆盖 > 伞目录 `elo.cfg.local` > 伞目录 `elo.cfg` > baseline 默认值。未知字段、非法百分比或局数、断开的比较图明确拒绝。

| 字段 | 含义 |
|---|---|
| `stride` | 每隔固定 iteration 选择一个模型，另保留当前最后模型 |
| `neighbors` | 在已选模型序列上向前连接的近邻级数 |
| `cross_seconds` | 跨臂固定累计训练时间间隔，单位秒；在双方固定轮次模型覆盖的时间内取最近模型 |
| `final_cross` | 各臂最终模型是否两两比赛 |
| `games_per_pair` | 每对总局数，必须为四的正倍数 |
| `bootstrap_samples` | 成对开局 bootstrap 重采样次数 |
| `anchor` | Elo 零点模型 ID；空值复用结果根的 `anchor.json`，首次选择首臂最早的采样模型 |
| `pair_workers` | 同时运行的 C++ 模型对进程数 |

采样选择 iteration 为 `stride` 整倍数的已提交模型，最后模型强制保留且去重；横轴仍使用模型所属轮次实际累计净墙钟，不按迭代数推算。延长训练后，旧临时末点退出当前采样，其原始比赛保留。固定 anchor 即使不在新采样网格上也保留，模型内容改变或参考臂缺失明确拒绝；显式 anchor 可改变本次图的零点，不重写默认固定参考。

组内按已选模型连接 `neighbors` 级近邻。跨臂按 `cross_seconds` 的整数倍选点，仅使用固定轮次模型，并要求每对臂的固定模型时间覆盖目标；最近距离相同时选择更早模型。每对臂独立确定覆盖范围，新增臂不改变其他臂的时间配对。临时末点不参与固定时间配对，`final_cross` 单独安排当前最终模型两两比赛。配对重复时只安排一次；比较图必须连通。全部赛果联合重新拟合，不先分别估计再平移，不约束曲线单调；增加数据后旧模型的 Elo 与区间可能变化。

所有模型共用伞目录 `match.cfg`，显式引用公共 engine 和比赛 profile，再应用伞目录 `match.cfg.local` 和 `MATCH_` 环境覆盖，不继承各臂的训练或比赛条件。局数由 `elo.games_per_pair` 控制。MuZero 要求关闭图搜索与子树复用、根对称数量为一，不兼容条件在启动前拒绝。混合 AZ/MZ 时须在共享比赛配置中满足这些约束。固定 visits 不等于相同思考时间，评估耗时不进入训练横轴。

独立 `--data` 发现有已提交 state 的实际数据臂；autoexp 使用本次完整调度臂列表和真实输出路径，支持自定义 run_dir。autoexp 的 CUDA 比赛使用第一个训练 GPU 槽位，子进程通过 `CUDA_VISIBLE_DEVICES` 映射为 `cuda:0`；独立 autoelo 遵循比赛 device 与当前可见设备。

默认结果根为数据伞目录的 `elo/`；配置臂输出分散在不同父目录时使用 controller 默认目录下的 `elo/`，也可用 `--output` 指定根目录。模型、赛程、采样、比赛条件、二进制或并行执行条件改变时创建独立评估子目录，旧原始结果不覆盖。局数改变也创建新评估。`pair_cache/` 按模型对身份保存原始开局和逐局记录，各结果子目录保存独立快照；新增臂、延长训练、改变采样或拟合设置时，匹配的比赛直接复用。提高 `games_per_pair` 只补后续开局；降低局数时只读取所需前缀。

缓存身份包含有序双方模型 ID 与内容校验值、实际比赛种子、比赛与开局配置（不含目标局数）、native binary、比赛配置转换源码以及并行执行条件。模型路径、训练时间、完整赛程、anchor、采样和 Elo 拟合源码不决定比赛身份。改变 visits、精度、开局、搜索／组批参数或对战二进制等条件时重新比赛。历史评估目录中身份一致的对局自动导入，独立重跑同一模型对时每个开局只采用一个试验来源，不叠加重复赛果或混合两次试验的换色对局。中断时已完成开局和逐局记录也进入缓存，恢复只补缺失的一侧。

每次调用显示 `reused_games`，性能记录保存复用局数和新增局数。全部完成才拟合并原子更新 `elo/latest`，当前图入口为 `elo/latest/elo.png`。

每套结果保存 `plan.json`、`manifest.json`、`resolved.cfg`、`status.json`，以及 `pairs/` 的开局、逐局赛果与 native 日志；失败不发布新的 latest。`invocations/` 保存每次调用的性能记录，恢复调用另存。

## C++ 并行与性能观测

Python 只规划、调度 native 进程和落盘；棋规、开局、搜索、对局和 GPU 组批推理均在 C++。先用多局并发产生独立请求，不把最大 batch 当作实际请求量。每对比赛的 `game_threads` 超过局数不会增加并发；可提高 `pair_workers` 同时运行多对，但每对加载双方模型服务，增加显存、CPU 和 CUDA 上下文成本。不同模型对进程目前不共享推理队列或权重。

| 配置位置 | 参数 | 作用 |
|---|---|---|
| `elo.cfg` | `pair_workers` | 整个赛程同时运行的模型对数量 |
| `match.cfg` | `game_threads` | 每对 C++ 并发局数，开局生成也使用此上限 |
| `engine.cfg`，可由 `match.cfg` 覆盖 | `search_threads` | 每局树内搜索线程，可能改变并行搜索轨迹 |
| `engine.cfg`，可由 `match.cfg` 覆盖 | `max_batch` | 每个模型服务的一批请求上限 |
| `engine.cfg`，可由 `match.cfg` 覆盖 | `server_threads` | 每个模型的推理服务线程；过多服务可能分散 batch，MuZero latent 固定路由到创建它的服务 |
| `engine.cfg`，可由 `match.cfg` 覆盖 | `batch_wait_us` | 填充 batch 的最大等待窗口 |
| `engine.cfg`，可由 `match.cfg` 覆盖 | `cpu_threads` | LibTorch CPU 算子线程数，区别于对局和搜索线程 |
| `engine.cfg`，可由 `match.cfg` 覆盖 | `queue_capacity` | 推理待处理队列配置，不是 batch 大小 |

先保留单搜索线程和单推理服务，调节模型对及对局并发，再按实际 batch、吞吐决定填批等待和搜索并行。不要仅凭加线程声称性能提升。比较时记录模型、棋盘、规则、visits、精度、开局和执行条件；树内线程、随机 D4 请求调度与缓存可能影响赛果。

C++ 每次输出 `match_stats`，记录包含模型加载、开局和对局的耗时、完成局数，以及双方请求数、batch 数、最大实际 batch、队列等待和缓存统计，保存在 `pairs/<配对>/performance/`。autoelo 每次调用的 `invocations/` 记录新增局数、总耗时、吞吐及 native 统计；总耗时包含比赛校验与恢复扫描、拟合和绘图，不等于纯 GPU 推理吞吐。
