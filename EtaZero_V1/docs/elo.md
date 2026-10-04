# 固定评估与等时间 Elo

`evaluate` 做单局面搜索，`match` 做一对模型的可恢复比赛，`arena` 选择历史模型、安排比赛并输出 Elo。所有公开动作使用实际棋盘零起始行优先编号，native 内部才转换为网络画布编号。评估必须用 `--model` 或 `--run-dir` 明确指定模型来源，不从评估配置目录猜测训练目录。模型由 manifest 校验身份与内容，同场模型画布须相同，网络宽度和深度可以不同。

## 独立配置与搜索预算

[evaluation 配置](../configs/baseline/eval.cfg) 和 [比赛配置](../configs/baseline/match.cfg) 只沿 `run.cfg` 的 extends 继承链读取自身文件与叶子目录的 `.local`。训练文件、训练环境变量和另一个评估文件不会覆盖它们，修改评估条件不改变训练配置身份。可使用 `EVAL_VISITS=100`、`MATCH_VISITS=100` 等前缀环境覆盖；开局字段使用 `MATCH_OPENING_` 前缀。字段校验集中在 [eval_config.py](../python/etazero/eval_config.py)。

baseline 搜索预算为 500v，smoke_test 显式覆盖为 100v；根初始访问和完成的边访问合计为 `root_visits`。`initial_visits` 为搜索前保留量，`new_playouts` 是本手新增访问（含首次根初始化），`simulations` 为本手新增根边模拟。不同 bot 的 Match 默认复用推进命中的子树，相同模型身份双方强制每手清树；同身份必须对应同模型路径。固定局面 eval 默认不复用，新的 500v 根通常为一个初始化和 499 个边模拟。`max_playouts` 独立约束新增访问，`max_time` 从开始搜索计时，时间停止保留至少两个新 playout，显式停止可阻止全部搜索。在途路径完成后返回；详细边界和严格并行发放与来源的差异见 [搜索预算](algorithms.md#puct-与搜索预算)。

NN 单朝向默认随机 D4，cache 命中复用首次 canonical 输出，根多对称绕 cache。落子温度使用来源 Match profile 的半衰期调度；policy target 不乘落子温度。eval/match 的 `inference_precision=auto` 在本实现指定 CUDA 时使用 FP16、指定 CPU 时使用 FP32，结果记录实际精度，显式 FP32 保留。网络缓存、终局节点和根集成使 NN 请求数不等于 visits。

评估和比赛均无训练根噪声，默认零落子温度、单搜索线程；并列最大行为权重按来源选择首个已分配子边。FPU、子树价值加权、policy target pruning、LCB、根多对称、根／全树 policy 温度与落子温度半衰期由独立 profile 显式配置；评估及比赛落子使用剪枝和 LCB 后的权重，原始 visits 单独输出。WDL 搜索 Q 为 W−L。比赛共享组批 evaluator、多局线程的结构参考 KataGo match，使用 EtaZero LibTorch 后端，不宣称复制 KataGo 的全部比赛功能或数值行为。

## 比赛与续测

```bash
# 在 EtaZero_V1/ 下，run-dir 用于寻找当前模型和保存默认输出。
bash scripts/run.sh evaluate --config-dir configs/baseline --run-dir data/my_run --size 15 --rule renju --moves 112,113
bash scripts/run.sh match --config-dir configs/baseline --model /path/to/a/model.pt --model-b /path/to/b/model.pt --games 40 --output data/my_match
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

[scripts/autoelo.sh](../scripts/autoelo.sh) 使用 [autoelo.py](../python/etazero/autoelo.py) 规划时间采样与赛程，复用 arena 的 C++ 对战、逐局恢复、联合拟合和绘图。`autoexp` 在所有调度臂成功达到预算后默认运行；`exp.cfg` 的 `experiment.autoelo = false` 或 `AUTOELO=false` 可关闭。训练中断或失败时不启动 Elo。评估失败明确报错并单独保存状态，已完成训练不重跑；原命令重启会跳过完成训练并恢复评估。

```bash
# 在本版本目录执行：仅发现实际已训练的数据臂。
CONFIG_DIR=configs/az_mz bash scripts/autoelo.sh --data data/az_mz --dry-run
CONFIG_DIR=configs/az_mz bash scripts/autoelo.sh --data data/az_mz

# 不传 --data 时使用伞目录所有配置臂及其真实 run_dir，缺测臂明确报错。
CONFIG_DIR=configs/az_mz bash scripts/autoelo.sh

# --output 指已有具体评估目录；不启动比赛或检查 native binary。
bash scripts/autoelo.sh --fit-only --output data/az_mz/elo/<评估标识>
```

设置在伞目录 `elo.cfg` 的 `[elo]`；默认值以 [baseline/elo.cfg](../configs/baseline/elo.cfg) 为准，实验例子为 [az_mz/elo.cfg](../configs/az_mz/elo.cfg)。优先级为对应命令行参数 > `ELO_<字段大写>` 环境覆盖 > 伞目录 `elo.cfg.local` > 伞目录 `elo.cfg` > baseline 默认值。未知字段、非法百分比或局数、断开的比较图明确拒绝。

| 字段 | 含义 |
|---|---|
| `points` | 每臂目标时间点数，目标为自身累计时间的 `1/points` 到 `100%` |
| `neighbors` | 在已选模型序列上向前连接的近邻级数 |
| `cross_time_fractions` | 相对于所有参与臂共同时间上限的跨臂对战位置，各臂取最近的已选模型；空值关闭 |
| `final_cross` | 各臂最终模型是否两两比赛 |
| `games_per_pair` | 每对总局数，必须为四的正倍数 |
| `bootstrap_samples` | 成对开局 bootstrap 重采样次数 |
| `anchor` | Elo 零点模型 ID；空值选择首臂最接近自身中期的已选模型 |
| `pair_workers` | 同时运行的 C++ 模型对进程数 |

时间采样使用已提交轮次的实际累计净墙钟，不按迭代数推算；最近模型重复则去重，最后模型强制保留。实际点数不足时显示真实数量，不伪造模型。跨臂和最终配对重复时只安排一次；全部赛果联合拟合，不先分别估计再平移，不约束曲线单调。

所有模型共用伞目录 `match.cfg`，直接覆盖 baseline 比赛 profile，再应用伞目录 `match.cfg.local` 和 `MATCH_` 环境覆盖，不继承各臂的训练或比赛条件。局数由 `elo.games_per_pair` 控制。MuZero 要求关闭图搜索与子树复用、根对称数量为一，不兼容条件在启动前拒绝。[az_mz/match.cfg](../configs/az_mz/match.cfg) 给出混合 AZ/MZ 的统一协议。固定 visits 不等于相同思考时间，评估耗时不进入训练横轴。

独立 `--data` 发现有已提交 state 的实际数据臂；autoexp 使用本次完整调度臂列表和真实输出路径，支持自定义 run_dir。autoexp 的 CUDA 比赛使用第一个训练 GPU 槽位，子进程通过 `CUDA_VISIBLE_DEVICES` 映射为 `cuda:0`；独立 autoelo 遵循比赛 device 与当前可见设备。

默认结果根为数据伞目录的 `elo/`；配置臂输出分散在不同父目录时使用 controller 默认目录下的 `elo/`，也可用 `--output` 指定根目录。模型、赛程、采样、比赛条件、二进制或并行执行条件改变时创建独立评估子目录，旧原始结果不覆盖。局数改变也创建新评估。同一计划重跑只补缺失比赛，全部完成才拟合并原子更新 `elo/latest`，当前图入口为 `elo/latest/elo.png`。

每套结果保存 `plan.json`、`manifest.json`、`resolved.cfg`、`status.json`，以及 `pairs/` 的开局、逐局赛果与 native 日志；失败不发布新的 latest。`invocations/` 保存每次调用的性能记录，恢复调用另存。

## C++ 并行与性能观测

Python 只规划、调度 native 进程和落盘；棋规、开局、搜索、对局和 GPU 组批推理均在 C++。先用多局并发产生独立请求，不把最大 batch 当作实际请求量。每对比赛的 `game_threads` 超过局数不会增加并发；可提高 `pair_workers` 同时运行多对，但每对加载双方模型服务，增加显存、CPU 和 CUDA 上下文成本。不同模型对进程目前不共享推理队列或权重。

| 配置位置 | 参数 | 作用 |
|---|---|---|
| `elo.cfg` | `pair_workers` | 整个赛程同时运行的模型对数量 |
| `match.cfg` | `game_threads` | 每对 C++ 并发局数，开局生成也使用此上限 |
| `match.cfg` | `search_threads` | 每局树内搜索线程，可能改变并行搜索轨迹 |
| `match.cfg` | `max_batch` | 每个模型服务的一批请求上限 |
| `match.cfg` | `server_threads` | 每个模型的推理服务线程；过多服务可能分散 batch，MuZero latent 固定路由到创建它的服务 |
| `match.cfg` | `batch_wait_us` | 填充 batch 的最大等待窗口 |
| `match.cfg` | `cpu_threads` | LibTorch CPU 算子线程数，区别于对局和搜索线程 |
| `match.cfg` | `queue_capacity` | 推理待处理队列配置，不是 batch 大小 |

先保留单搜索线程和单推理服务，调节模型对及对局并发，再按实际 batch、吞吐决定填批等待和搜索并行。不要仅凭加线程声称性能提升。比较时记录模型、棋盘、规则、visits、精度、开局和执行条件；树内线程、随机 D4 请求调度与缓存可能影响赛果。

C++ 每次输出 `match_stats`，记录包含模型加载、开局和对局的耗时、完成局数，以及双方请求数、batch 数、最大实际 batch、队列等待和缓存统计，保存在 `pairs/<配对>/performance/`。autoelo 每次调用的 `invocations/` 记录新增局数、总耗时、吞吐及 native 统计；总耗时包含比赛校验与恢复扫描、拟合和绘图，不等于纯 GPU 推理吞吐。
