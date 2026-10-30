# 同步调度与恢复

入口是 [run.py](../python/etazero/run.py)，配置目录与脚本用法见 [配置与启动](configuration.md)。`run --iterations N` 的 N 是累计完成轮数；`--max-hours H` 是累计时间预算，按下文口径计时并在完整轮次边界检查。两者至少指定一个，同时指定时任一达到即停止开始新轮次。在同一个运行目录使用相同配置再次运行，会继续未完成轮次；可以增加轮数或时间上限继续训练。

阶段恢复要求运行目录已经提交初始 `state.json`。创建运行记录时、尚未写出初始状态就中断的目录，以及任意其他非空目录，当前不会被自动接管；此时尚未开始自对弈或训练。

每轮固定一个输入模型，依次执行原生 selfplay、来源 shuffle、来源 learner、模型导出和可选 gatekeeper。AlphaZero 首轮使用来源 random evaluator；MuZero 首轮先由正式 learner 保存零样本初始 checkpoint，再导出三组件模型。后续轮次只消费已接受的模型；普通权重／SWA 选择不改变 learner 恢复的训练权重。

## 初始权重与独立训练状态

新运行可使用 `run --weights /absolute/path/to/checkpoint.ckpt`。文件必须来自匹配网络配置的来源 learner；只加载普通模型权重和模型 buffers，不加载其中的 optimizer、SWA、训练计数、运行指标、reader 或 RNG 状态。权重保留为运行内的 `initial/weights.ckpt`，原始路径记录在 `source.json`。来源 learner 的 `-initial-weights` 入口在构造独立 optimizer、SWA 和初始化范数基准之前加载权重。初始化范数基准据此使用本臂实际采用的权重。

显式提供初始权重时，AZ 与 MuZero 都由正式 learner 保存本臂零样本完整 checkpoint，再导出 `models/initial`，首轮 selfplay 使用该模型。初始化与初始导出的耗时计入该臂首轮。省略权重时，AZ 保留来源 random evaluator 冷启动，MuZero 保留必需的三组件初始化。

续训不再传 `--weights`，始终恢复本臂的完整 learner checkpoint；原始外部权重文件可以不再存在。初始模型与训练 checkpoint 分开保留，后续训练不会覆盖初始化证据。来源 `-initial-checkpoint` 仍保留它自身载入训练状态的语义，不用于实验臂的仅权重共享。

## 多实验臂调度

`scripts/autoexp.sh`（等同于 `run.sh experiment`）启动独立逐轮控制器，每个 GPU 槽位同时运行一个实验臂；单个 learner 仍为单设备。实验目录的 `exp.cfg` 使用 `[experiment]` 配置预算与槽位，默认自动发现直接子目录中的 `run.cfg`；可选 `[arms]` 显式指定参与臂。格式与路径规则见 [配置目录](configuration.md#实验目录)。

```bash
bash scripts/autoexp.sh --config-dir /absolute/path/to/experiment \
  --work-dir /absolute/path/to/experiment-output --dry-run
bash scripts/autoexp.sh --config-dir /absolute/path/to/experiment \
  --work-dir /absolute/path/to/experiment-output
```

`iterations` 与 `max_hours` 至少指定一项，都是每个臂的累计预算，语义与 `run` 相同。`gpus` 可使用 CUDA 索引或 GPU/MIG UUID 字符串；省略时串行运行并继承当前设备可见性。显式槽位通过 `CUDA_VISIBLE_DEVICES` 隔离设备，臂内来源配置须使用隔离后的本地设备索引（单个可见 GPU 为 0），不能继续填宿主机全局索引。同一物理设备不要同时用索引与 UUID 重复列入。

各臂保存于 `work-dir/runs/臂名`，`configs/` 保存已解析的外层配置，`logs/` 追加各臂控制器输出，原生日志继续由各臂的来源程序管理。`plan.json` 和 `status.json` 用于查看调度；恢复与完成判定读取各臂实际 `state.json`，不信任上一次调度状态。已有臂的配置须保持一致；增加预算或增加新臂会继续相应工作，达到目标的臂不重复训练。仍有活动轮次的臂先按控制器恢复边界处理。

`shared_init` 默认为 true：相同棋类、画布与网络配置的臂共享一次来源 learner 的零样本初始化；PUCT/Gumbel、优化器、训练预算和 MuZero 展开长度不拆分相同网络。初始化直接运行本版本 `train.py -initialize-only`，无 forward、数据消费或 optimizer 更新；来源随机种子保留在初始化原生日志中。公共初始化在 CPU 完成，保存于 `initializations/`，不计入任何一臂的训练时间；各臂实际构造独立 learner 与导出模型的耗时仍计入本臂。共享的是权重，不共享随机流或回放池。关闭该项时，各臂沿用独立 `run` 的默认冷启动。

一臂失败会停止启动排队臂并中断正在运行的其他臂。SIGINT/SIGTERM 同样停止队列并通知整个臂进程组；最多等待 10 秒后强制结束仍存活的进程，恢复范围依各来源阶段的实际提交点确定，不承诺信号期间完成整轮。实验目录锁由控制器及其来源子进程继承，调度器异常退出而子进程仍在写入时，不允许新的调度器接管。

可选 `[evaluation]` 指定独立原生 match 配置与 autoelo 选点／拟合设置，全部臂完成后在第一个 GPU 槽位执行自动 Elo。评估状态另存 `elo_status.json`；评估失败后重试不会重复已完成训练，原生已提交对局可以复用。省略该节时只训练。autoelo 当前逐对加载两个模型；独立 match／arena 已有驻留窗口，autoelo 跨配对共享驻留尚未接入，见 [历史模型评估](evaluation.md)。

## Replay-ratio 额度

省略 `replay_ratio` 时，每轮使用 `games_per_round` 和 `epochs_per_round`。配置正的 `replay_ratio` 后，`games_per_round` 用作冷启动的校准批次；先完成这些完整游戏，测量每局有效根行数，再按缺口补完整游戏，直到达到 `bootstrap_min_rows`，省略该值则采用 `samples_per_epoch`。校准与补产都在首轮内，达到门槛才 shuffle 和训练，不以外层短轮重置来源内部时钟。首次训练完成后，把实际累计行数记录为 `origin_rows`，冷启动数据不发放正常阶段的额度。冷启动完全没有训练行时会明确失败，不能用空数据初始化训练。

正常轮次采用最近两轮的总有效行数除以总游戏数，按局数加权估计产样率。有效行数包含来源 writer 的实际重复行与 side 根；MuZero 的未来轨迹位置不重复计入根样本额度。行数读取直接使用来源 shuffle 的 NPZ header 函数。成功的原生回执包含零行游戏；中断后无法确认的游戏计数限制见下文。

令 `S=samples_per_epoch`、`E=epochs_per_round`、`R=replay_ratio`，本轮计划为：

```text
计划新增行数 = S * E / R
计划游戏数 = max(1, ceil(计划新增行数 / 近期每局行数))
可用样本额度 = 上轮剩余额度 + 本轮实际新增根行数 * R
本轮 epoch 数 = min(E, max(0, floor(可用样本额度 / S)))
提交后剩余额度 = 可用样本额度 - checkpoint 中实际新增的消费样本数
```

正常轮完成计划游戏数后不继续追补行数缺口，实际产样变化由训练额度吸收。两轮均无有效行、无法估计产样率时，会警告并使用 `games_per_round` 再做一批校准；不会虚构正的产样率或额度。

**训练额度以完整来源 epoch 为粒度。** V4 不沿用 V3.4 的 batch 边界截断，也不以快照的一遍 batch 数封顶：来源 reader 仍按其文件顺序、随机整文件选择和重复文件设置工作。`S` 是目标量，实际消费可能超出它；超额记为负额度，之后的轮次先偿还。额度不足一个 epoch 时，仍完成 selfplay 和 shuffle，但不调用 learner、不推进训练时钟、不重新导出模型，继续使用上一轮模型。新游戏全部为零训练行时也可以完成这样的轮次，已有回放数据保留。

产样计划在 selfplay 前保存；训练前保存本轮 epoch 数和可用额度；新剩余额度只在轮次最终提交时生效。恢复不会按新增 shard 再次发放同一笔额度，也不会重复已提交 epoch。记录的 `actual_replay_ratio` 是累计实际消费样本数除以累计有效根行数，包含冷启动；它与正常阶段的额度账本口径不同，不能解释为每一轮都严格等于配置的 R。来源自带的 train-bucket 限制不能同时通过 `train_args` 设置第二套额度所有权。

## 候选验收

`gatekeeper_config` 指定独立的原生 gatekeeper 配置，相对声明它的运行配置文件解析；省略时直接接受候选。该配置拥有 `numGamesPerGating`、比赛搜索预算、搜索算法、执色配对、开局、认输和推理设置，参考来源 [gatekeeper1.cfg](../cpp/configs/training/gatekeeper1.cfg)。控制器只指定模型目录、棋类和 MuZero 模型族，不用 selfplay 的搜索配置覆盖比赛配置。`gatekeeper_required_win_prop` 控制来源接受阈值，缺省 0.5。

首个训练模型直接接受，建立训练基准；MuZero 的零样本初始 bundle 不作为这一轮的门控对手。之后候选与当前 accepted 模型比赛。来源 `MatchPairer` 的执色轮次与随机配对、`GameRunner`、点数阈值、提前胜负判定和候选胜平规则保留。D 与 NR 分别报告，两者各计半分。Go 保留来源规则计分收尾；非围棋超过手数上限且尚未规则终局时明确失败，不把该局变成和棋、NR 或用围棋计分裁定。

候选被拒绝时，learner checkpoint、训练计数、SWA 和 replay 额度继续推进，下一轮 selfplay 继续使用旧 accepted 模型。无训练更新的轮次不重复导出或比赛。`models/` 保存所有不可变的导出候选，是否用于 selfplay 由 `state.json` 的 `accepted_model` 决定；不能按最新文件时间猜测接受结果。

每轮 `gatekeeper/` 下建立候选与基准模型的硬链接副本，原生命令按来源流程把候选副本移动到 `accepted/` 或 `rejected/`，不会移动或覆盖 `models/` 中的原件。`matches/尝试编号/` 保留该次原生配置、日志与 SGF。原生 `-result-file` 模式处理一个候选后退出，比赛决策原子记录候选／基准、已统计局数、计划局数、点数和 W/D/L/NR；已有决策文件不能覆盖。原生按文件时间直接拒绝旧模型时仅记录 `reason=older_model`，不伪造赛果；外层比较的是本轮固定候选，因此使用来源 `-no-autoreject-old-models`。

决策文件在模型移动前提交。恢复若读到匹配本轮候选与基准的已提交决策，就补完目录移动并提交轮次，不重赛、不重训；SIGTERM 或进程中断发生在决策前时不更新 accepted。无决策文件的比赛会在新的尝试目录重跑，先前部分 SGF 只保留作历史记录，不拼接为已完成赛果。尚未保存的棋局和门控比赛 RNG 不恢复。

## 提交边界

控制器在 `state.json` 记录轮次、阶段尝试次数、已完成阶段和训练起始计数。各阶段以实际生产者的提交点判断恢复进度，不按日志内容猜测成功。

| 阶段 | 已提交状态 | 再次运行 |
|---|---|---|
| MuZero 或显式权重初始化 | 零样本完整 checkpoint 与 `initial/checkpoint.ckpt` | 复用原始初始化，不重新初始化已经开始的训练。 |
| 自对弈 | 原生 writer 原子发布的 NPZ；正常退出后、全部 writer 排空后的完成回执 | 成功回执确认本次额度；冷启动仍需满足行数门槛。没有成功回执则统计已发布数据中的唯一 game hash，只补足原计划的剩余游戏数。 |
| shuffle | 完整候选目录原子改名为 `shuffleddata/轮次` | 直接复用已发布快照；未发布目录移到该轮 `discarded/`，从原始数据重新 shuffle。 |
| learner | 来源每个 epoch 结束时保存的完整 checkpoint | 以 `export_cycle_counter` 对照本轮开始计数，只执行尚未提交的 epoch，不重复已提交更新。 |
| 导出 | 包含全部必需模型文件的候选目录原子发布到 `models/` | 复用已发布目录；部分导出移到 `discarded/`，从已提交 checkpoint 重新导出。 |
| gatekeeper | 原生完整决策原子写入 `gatekeeper/result.json` | 复用决策并补完副本移动；没有决策则在独立尝试目录重赛，保留旧 accepted。 |
| 轮次 | 模型路径、消费计数、额度、完成轮数与计时先写入 `rounds/轮次/commit.json`，再更新 `state.json` | 若轮次已提交但根状态指针尚未更新，直接采用该提交；不重跑阶段、不重复累计时间。之后才写观测用 `result.json`，报告失败不会撤销有效提交。 |

AlphaZero 与 MuZero 共用此流程。MuZero manifest 和三个组件以整个目录为发布单位；初始化 checkpoint 单独保留，不会被训练 checkpoint 的后续原子替换改变。原始自对弈 shard、历史模型与先前尝试日志均保留。

恢复不会保存或重建自对弈进程的内存状态：在途对局、尚未写出的行、fork 队列、搜索树及其 RNG 会丢失，重新启动后产生新对局。已发布行来自已经结束的对局，目标不会被改成虚假的部分对局目标；MuZero shard 保留所引用位置的完整轨迹。同一游戏跨多个 shard 时只计一次。被中断尝试中没有留下任何训练行的游戏无法从 NPZ 计数，可能额外补产；正常完成的原生回执则包含零训练行游戏的额度完成情况。故恢复保证已发布数据可继续使用，不承诺精确重演中断前的自对弈过程或恢复全部 buffered rows。

learner 保留来源整文件选取、batch 尾部处理、epoch/subepoch、Lookahead 与 SWA 节奏，不增加逐 batch checkpoint。未完成 epoch 中尚未保存的更新会重做；恢复点是最后一个完整来源 checkpoint。完整 checkpoint 包含模型、optimizer、SWA、训练与文件消费状态，以及单设备的训练／验证 reader RNG、Python/NumPy/Torch RNG、实际 learner CUDA 设备 RNG 和 AMP scaler。

为保存普通 AlphaZero reader 的增强随机流，单设备 learner 也持有训练／验证两个 generator 并传入 reader；增强的抽样分布和执行位置保留，不再每次打开 reader 时从系统熵重新播种。MuZero 同样使用独立的训练／验证流。导出专用的 `skip_optimizer` checkpoint 没有完整 optimizer/runtime，不能替代同目录续训 checkpoint。不承诺跨设备、库版本或非确定性 CUDA 路径逐位一致。

## 并发与计时

同一运行目录使用非阻塞排他锁。阶段子进程继承锁的文件描述符：控制器异常退出而子进程仍在运行时，第二个控制器会拒绝接管。等现有写入进程退出后再次执行相同命令即可恢复，不能通过删除锁文件抢占运行目录。

计时以控制器实际观测的逐轮单调墙钟为基础，包含 selfplay、shuffle、learner、导出、gatekeeper 及这些阶段间的编排开销。MuZero 必需的零样本初始化和初始导出计入首轮；模型加载、数据等待、来源程序内部等待也计入。运行目录与源码快照的创建、两次调用之间的暂停、轮次计时结算之后的提交文件写入和观测报告不计入。阶段耗时供诊断，不能相加替代完整轮次墙钟。

learner 通过 [timed_learner.py](../python/etazero/timed_learner.py) 在原进程执行来源入口，记录 PyTorch 的非重叠外层编译墙钟差值：包含 forward 与 lazy backward 编译、后续重编译和验证图；Muon 自身的编译同样统计，即使网络使用 `-no-compile`。不累加重叠的 backend/Triton 子阶段，不额外预热、更新参数或同步设备。每次 learner／初始化尝试的回执保存在该轮 `timing/`，控制器从轮次墙钟中扣除编译时间，得到有效累计 `elapsed_seconds`。缓存加载仍按 PyTorch 编译计数器报告的时间计入扣除量，不将其解释成纯编译器 CPU 工作量。

轮次提交后默认更新训练、loss 和性能图；绘图与可选手动重建见 [训练观测与绘图](observations.md)，不计入预算，失败不撤销提交。

时间预算在开始新轮次前检查。最后一轮可以超出预算，恢复未完成轮次时先完成该轮，不截断来源 epoch 或采样阶段来凑时间。`commit.json` 同时记录本轮时间、累计时间和模型路径，可用于后续绘图和模型比较；不能依据模型文件的修改时间推断训练时间。

每次阶段尝试写独立日志，已经观测到的失败尝试耗时也累计。若旧控制器在阶段或阶段之间异常退出、未能记录完整时间，或 learner 编译回执缺失／无效，则将 `timing_valid` 标为 false 并警告；未知编译量保存为 null。失效会传播到累计时间，后续成功轮次不能使此前缺失的时间重新有效。此时预算统一使用累计已观测墙钟（不再扣除任何轮次的编译时间），`timing_basis=observed_wall_incomplete`，不得用于精确等时间比较；缺失时段不会补零伪装有效。正常捕获并完整记录的失败后重试不包含停机间隔。

当前不提供跨文件系统断电事务；原子发布解决进程中断边界，不能代替持久化备份。
