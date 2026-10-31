# 同步调度与恢复

入口是 [run.py](../python/etazero/run.py)，配置目录与脚本用法见 [配置与启动](configuration.md)。`run --iterations N` 的 N 是最后一个 iteration 的编号；iteration 0 只做 bootstrap，iteration 1 开始训练；`--max-hours H` 是累计时间预算，按下文口径计时并在完整轮次边界检查。配置和命令行都未指定这两项时默认无限运行，直到手动中断；同时指定时任一达到即停止开始新轮次。在同一个运行目录使用相同配置再次运行，会继续未完成轮次；可以增加轮数或时间上限继续训练。

阶段恢复要求运行目录已经提交初始 `state.json`。创建运行记录时、尚未写出初始状态就中断的目录，以及任意其他非空目录，当前不会被自动接管；此时尚未开始自对弈或训练。

每轮固定一个输入模型，依次执行原生 selfplay、来源 shuffle、来源 learner、模型导出和可选 gatekeeper。AlphaZero 首轮使用来源 random evaluator；MuZero 首轮先由正式 learner 保存零样本初始 checkpoint，再导出三组件模型。后续轮次只消费已接受的模型；普通权重／SWA 选择不改变 learner 恢复的训练权重。

同步控制器在持有运行目录锁、selfplay 正常退出且 writer 排空后，以 `-input-files-stable` 调用 shuffle，省去来源工具为并发写入保留的 3 秒文件系统等待。独立 shuffle 默认仍等待；只有输入已停止写入且已发布文件不再修改时才可指定此选项。窗口选择、随机打散、压缩和训练快照发布流程不变。

网络预设表在首次按名称选择网络时构造，保留来源的全部名称与派生配置。从 checkpoint 恢复或导出时使用其中保存的完整配置，无需为导入网络与特征工具而生成全部预设。

普通与 MuZero 导出在持锁的控制器进程内调用来源入口，跨轮复用 Python/PyTorch 模块；每次仍从已提交 checkpoint 加载独立 CPU 模型。调用结束或失败时恢复调用者的 CPU RNG、日志处理器、输出流、命令行参数与 NumPy 打印设置。导出仍独立计时、写入本次阶段日志，并在完整文件生成后原子发布目录；独立 CLI 导出入口继续可用。

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

多实验臂调度仍要求 `iterations` 与 `max_hours` 至少指定一项，都是每个臂的累计预算，计数与停止条件和 `run` 相同。`gpus` 可使用 CUDA 索引或 GPU/MIG UUID 字符串；省略时串行运行并继承当前设备可见性。显式槽位通过 `CUDA_VISIBLE_DEVICES` 隔离设备，臂内来源配置须使用隔离后的本地设备索引（单个可见 GPU 为 0），不能继续填宿主机全局索引。同一物理设备不要同时用索引与 UUID 重复列入。

各臂保存于 `work-dir/runs/臂名`，`configs/` 保存已解析的外层配置，`logs/` 追加各臂控制器输出，原生日志继续由各臂的来源程序管理。`plan.json` 和 `status.json` 用于查看调度；恢复与完成判定读取各臂实际 `state.json`，不信任上一次调度状态。已有臂的配置须保持一致；增加预算或增加新臂会继续相应工作，达到目标的臂不重复训练。仍有活动轮次的臂先按控制器恢复边界处理。

`shared_init` 默认为 true：相同棋类、画布与网络配置的臂共享一次来源 learner 的零样本初始化；PUCT/Gumbel、优化器、训练预算和 MuZero 展开长度不拆分相同网络。初始化直接运行本版本 `train.py -initialize-only`，无 forward、数据消费或 optimizer 更新；来源随机种子保留在初始化原生日志中。公共初始化在 CPU 完成，保存于 `initializations/`，不计入任何一臂的训练时间；各臂实际构造独立 learner 与导出模型的耗时仍计入本臂。共享的是权重，不共享随机流或回放池。关闭该项时，各臂沿用独立 `run` 的默认冷启动。

一臂失败会停止启动排队臂并中断正在运行的其他臂。SIGINT/SIGTERM 同样停止队列，各臂控制器通知自己的阶段进程组并等待收尾、checkpoint 和计时回执，不以固定短超时强杀正在保存的进程。恢复范围依各来源阶段的实际提交点确定，不承诺信号期间完成整轮。实验目录锁由控制器及其来源子进程继承，调度器或控制器被强杀而阶段子进程仍在写入时，不允许新的调度器接管；须等现有写入进程退出。

可选 `[evaluation]` 指定独立原生 match 配置与 autoelo 选点／拟合设置，全部臂完成后在第一个 GPU 槽位执行自动 Elo。评估状态另存 `elo_status.json`；评估失败后重试不会重复已完成训练，原生已提交对局可以复用。省略该节时只训练。autoelo 将多对比赛交给同一个原生进程，外围按 `max_loaded_models` 编排驻留窗口，跨配对复用已加载的模型服务；逐对缓存、缺局恢复和独立 match／arena 的驻留语义见 [历史模型评估](evaluation.md)。

## 验证数据

来源 learner 已保留完整的验证流程，读取当前 shuffle 快照的 `val/`。来源 [shuffle.sh](../python/selfplay/shuffle.sh) 使用 `shuffle.py` 的 `-only-include-md5-path-prop-lbound`／`-only-include-md5-path-prop-ubound`，按原始文件路径哈希把数据划分为 train 与 val；来源 [synchronous_loop.sh](../python/selfplay/synchronous_loop.sh) 则在小规模同步运行中显式设置 `SKIP_VALIDATE=1`。

V4 逐轮控制器当前仅生成 `train/` 和 `train.json`，没有接入可选的验证集准备步骤。因此通过 `run.sh` 正常启动时 learner 跳过验证，验证 loss 留空；不是验证 head、loss 或 reader 被删除。独立使用原生工具可以准备验证数据，MuZero 的验证快照还需自己的 `val.json` 展开元数据。若接入自动验证，划分与 shuffle 编排应留在外围，继续调用原生 shuffler 和 learner；不能把训练样本复制成验证样本。

## 自适应逐轮调度

外层沿用 V3.4 的采样规划和训练额度：原生 selfplay → 来源 shuffle → 按实际额度训练 → 导出候选 → 可选 gatekeeper。`run.cfg` 的 `bootstrap_games`、`train_steps`、`replay_ratio` 分别控制探测局数、每轮基准训练步数上限和新增样本复用倍率；`shuffle.cfg` 的 `min-rows` 是首轮训练的数据门槛。

- iteration 0：运行 `bootstrap_games` 局，记录实际 rows/game，不 shuffle、不训练。
- iteration 1：根据 bootstrap 的平均产量估算补足 `min-rows` 所需局数，乘 `bootstrap_game_multiplier` 后向上取整；不足时继续补局。此前已经达到门槛则不再补局。该轮训练不向后续发放 bootstrap 样本额度。
- iteration ≥ 2：合并前两轮的实际 rows 与 games 计算平均产量，按 `train_steps × batch_size / replay_ratio` 规划新增 rows，再反推局数。运行完整局数后，以实际产量计算训练额度；不在本轮追补计划 rows。

第一轮实际训练步数为 `min(train_steps, 快照完整 batch 数)`。后续可用额度为 `上轮剩余额度 + 本轮新增 rows × replay_ratio`，实际步数再受 `floor(可用额度 / batch_size)` 限制，扣除实际训练消费量后把剩余额度带到下一轮。快照 batch 数按各 NPZ 单独取整后求和；MuZero 按展开后的根 batch 计数，未来展开步不重复计入训练时钟。额度结余不改变下一轮的基准产样目标。

内部来源 epoch 的名义样本量为 `train_steps × batch_size`，保留整文件选择和 subepoch 行为。外层预算可以在文件或 epoch 中途用完：learner 的 `-stop-at-samples` 接受精确累计样本目标，保存当前文件列表与已消费 batch、epoch/subepoch 游标、训练模型、optimizer、Lookahead fast/slow 与同步计数、SWA 和随机状态后退出。下一轮先读完原来选中的 subepoch，在下一个选择边界采用最新 shuffle。旧快照在这些游标仍可引用时保留。

SWA 周期、平滑尺度与 Lookahead 参数仍由 `train.cfg` 指定；外层换轮不重置它们，不在发布时强制追加 SWA 采样。只有内部完整 epoch 才执行来源 Lookahead 收尾与验证。每个实际训练步数大于零的 iteration 保存一个候选，无需等待内部 epoch 完成；来源独立命令仍支持原生导出周期，逐轮控制器则拥有自己的发布边界。零步轮次保留现有模型。

选择 SWA 发布时，若 `n_averaged` 仍为零则警告并跳过候选，继续后续采样和训练，不回退到普通权重。已有有效 SWA 但本轮未更新时，发布当前已有的平均权重。候选保存在 `torchmodels_toexport/round-N-s样本数-d累计数据行数/`，用于续训的是 `train/checkpoint.ckpt` 中的训练权重与完整状态。

`result.json` 记录采样计划、实际 rows/games、训练步数上限、快照 batch 数、可用及剩余训练额度、实际步数和内部完成 epoch 数。`state.json.completed_rounds` 保存最大已完成 iteration 编号，新运行尚未完成 bootstrap 时为 -1。

## 候选验收

`gatekeeper_config` 指定独立的原生 gatekeeper 配置，相对声明它的运行配置文件解析；省略时直接接受候选。该配置拥有 `numGamesPerGating`、比赛搜索预算、搜索算法、执色配对、开局、认输和推理设置，参考来源 [gatekeeper1.cfg](../cpp/configs/training/gatekeeper1.cfg)。控制器只指定模型目录、棋类和 MuZero 模型族，不用 selfplay 的搜索配置覆盖比赛配置。`gatekeeper_required_win_prop` 控制来源接受阈值，缺省 0.5。

首个训练模型直接接受，建立训练基准；MuZero 的零样本初始 bundle 不作为这一轮的门控对手。之后候选与当前 accepted 模型比赛。来源 `MatchPairer` 的执色轮次与随机配对、`GameRunner`、点数阈值、提前胜负判定和候选胜平规则保留。D 与 NR 分别报告，两者各计半分。Go 保留来源规则计分收尾；非围棋超过手数上限且尚未规则终局时明确失败，不把该局变成和棋、NR 或用围棋计分裁定。

候选被拒绝时，learner checkpoint、训练计数、SWA 和训练额度继续推进，下一轮 selfplay 继续使用旧 accepted 模型。没有就绪候选的轮次不导出或比赛。`models/` 保存所有不可变的导出候选，是否用于 selfplay 由 `state.json` 的 `accepted_model` 决定；不能按最新文件时间猜测接受结果。

每轮 `gatekeeper/候选名/` 下建立候选与基准模型的硬链接副本，原生命令按来源流程把候选副本移动到 `accepted/` 或 `rejected/`，不会移动或覆盖 `models/` 中的原件。`matches/尝试编号/` 保留该次原生配置、日志与 SGF。原生 `-result-file` 模式处理一个候选后退出，比赛决策原子记录候选／基准、已统计局数、计划局数、点数和 W/D/L/NR；已有决策文件不能覆盖。原生按文件时间直接拒绝旧模型时仅记录 `reason=older_model`，不伪造赛果；外层比较的是本轮固定候选，因此使用来源 `-no-autoreject-old-models`。

决策文件在模型移动前提交。恢复若读到匹配本轮候选与基准的已提交决策，就补完目录移动并提交轮次，不重赛、不重训；SIGTERM 或进程中断发生在决策前时不更新 accepted。无决策文件的比赛会在新的尝试目录重跑，先前部分 SGF 只保留作历史记录，不拼接为已完成赛果。尚未保存的棋局和门控比赛 RNG 不恢复。

## 提交边界

控制器在 `state.json` 记录轮次、阶段尝试次数、已完成阶段和训练起始计数。各阶段以实际生产者的提交点判断恢复进度，不按日志内容猜测成功。

| 阶段 | 已提交状态 | 再次运行 |
|---|---|---|
| MuZero 或显式权重初始化 | 零样本完整 checkpoint 与 `initial/checkpoint.ckpt` | 复用原始初始化，不重新初始化已经开始的训练。 |
| 自对弈 | 原生 writer 原子发布的 NPZ；正常退出后、全部 writer 排空后的完成回执 | 成功回执确认本次额度；随后由 shuffle 判断是否达到行数门槛。没有成功回执则统计已发布数据中的唯一 game hash，只补足原计划的剩余游戏数。 |
| shuffle | 完整候选目录原子改名为 `shuffleddata/轮次` | 直接复用已发布快照；未发布目录移到该轮 `discarded/`，从原始数据重新 shuffle。 |
| learner | 内部 epoch、外层预算或正常停止边界的完整 checkpoint | 恢复 fast/slow、optimizer、SWA 和文件内游标，继续到本轮已记录的绝对样本目标；不重复消费已保存的额度。 |
| 导出 | 包含全部必需模型文件的候选目录原子发布到 `models/` | 复用已发布目录；部分导出移到 `discarded/`，从已提交 checkpoint 重新导出。 |
| gatekeeper | 原生完整决策原子写入 `gatekeeper/候选名/result.json` | 复用决策并补完副本移动；没有决策则在独立尝试目录重赛，保留旧 accepted。 |
| 轮次 | 模型路径、消费计数、完成轮数与计时先写入 `rounds/轮次/commit.json`，再更新 `state.json` | 若轮次已提交但根状态指针尚未更新，直接采用该提交；不重跑阶段、不重复累计时间。之后才写观测用 `result.json`，报告失败不会撤销有效提交。 |

AlphaZero 与 MuZero 共用此流程。MuZero manifest 和三个组件以整个目录为发布单位；初始化 checkpoint 单独保留，不会被训练 checkpoint 的后续原子替换改变。原始自对弈 shard、历史模型与先前尝试日志均保留。

恢复不会保存或重建自对弈进程的内存状态：在途对局、尚未写出的行、fork 队列、搜索树及其 RNG 会丢失，重新启动后产生新对局。已发布行来自已经结束的对局，目标不会被改成虚假的部分对局目标；MuZero shard 保留所引用位置的完整轨迹。同一游戏跨多个 shard 时只计一次。被中断尝试中没有留下任何训练行的游戏无法从 NPZ 计数，可能额外补产；正常完成的原生回执则包含零训练行游戏的额度完成情况。故恢复保证已发布数据可继续使用，不承诺精确重演中断前的自对弈过程或恢复全部 buffered rows。

learner 在内部完整 epoch、外层预算暂停和正常停止边界保存，不逐 batch 写 checkpoint。单设备 learner 收到 SIGINT/SIGTERM 后完成当前 batch 的更新、AMP 跳步、Lookahead/SWA 和游标推进再保存；若已到 epoch 末尾则完成该 epoch 的收尾。异常或强杀退出后，最后一次保存以来的内存更新需要重做。完整 checkpoint 保存模型、optimizer、Lookahead slow 权重与同步计数、SWA 及累计余量、LR/brenorm 更新时钟、文件选择与已消费游标，以及训练／验证 reader RNG、Python/NumPy/Torch RNG、实际 learner CUDA 设备 RNG 和 AMP scaler。读取游标恢复时跳过已消费 batch，不重复消耗增强随机数。已发布 shuffle 快照继续保留。

checkpoint 同时保存原生训练／验证指标日志的字节位置。恢复时将该位置之后的旧记录区间记入 `train/metrics_recovery.jsonl`，绘图排除这些已回退更新的观测，原始日志保留不变。已有 checkpoint 缺少日志位置时，按其已保存样本数筛除此前日志中的超界更新，同样本数的验证记录保守留空。指标与模型恢复分开处理，不从日志恢复 optimizer，也不因丢弃作废指标而扣除已花掉的时间。

为保存普通 AlphaZero reader 的增强随机流，单设备 learner 也持有训练／验证两个 generator 并传入 reader；增强的抽样分布和执行位置保留，不再每次打开 reader 时从系统熵重新播种。MuZero 同样使用独立的训练／验证流。导出专用的 `skip_optimizer` checkpoint 没有完整 optimizer/runtime，不能替代同目录续训 checkpoint。不承诺跨设备、库版本或非确定性 CUDA 路径逐位一致。

## 并发与计时

同一运行目录使用非阻塞排他锁。阶段子进程继承锁的文件描述符：控制器异常退出而子进程仍在运行时，第二个控制器会拒绝接管。等现有写入进程退出后再次执行相同命令即可恢复，不能通过删除锁文件抢占运行目录。

Ctrl+C／SIGTERM 请求停止后，控制器向自己持有的阶段进程组发送停止信号并等待退出。selfplay 排空已完成对局的 writer 队列并写出最终统计；learner 在上述安全边界保存；正在进程内执行的导出先完成调用。控制器结算本次墙钟、读取编译回执后才释放锁，不再开始下一阶段。启动脚本在当前 shell 激活环境后直接执行控制器，使终端中断能等待完整收尾。

计时以控制器实际观测的逐轮单调墙钟为基础，包含 selfplay、shuffle、learner、导出、gatekeeper 及这些阶段间的编排开销。MuZero 必需的零样本初始化和初始导出计入首轮；模型加载、数据等待、来源程序内部等待也计入。运行目录与源码快照的创建、两次调用之间的暂停、轮次计时结算之后的提交文件写入和观测报告不计入。阶段耗时供诊断，不能相加替代完整轮次墙钟。

learner 通过 [timed_learner.py](../python/etazero/timed_learner.py) 在原进程执行来源入口，记录 PyTorch 的非重叠外层编译墙钟差值：包含 forward 与 lazy backward 编译、后续重编译和验证图；Muon 自身的编译同样统计，即使网络使用 `-no-compile`。不累加重叠的 backend/Triton 子阶段，不额外预热、更新参数或同步设备。每次 learner／初始化尝试的回执保存在该轮 `timing/`，控制器从轮次墙钟中扣除编译时间，得到有效累计 `elapsed_seconds`。缓存加载仍按 PyTorch 编译计数器报告的时间计入扣除量，不将其解释成纯编译器 CPU 工作量。

轮次提交后默认更新训练、loss 和性能图；绘图与可选手动重建见 [训练观测与绘图](observations.md)，不计入预算，失败不撤销提交。

时间预算在开始新轮次前检查。最后一轮可以超出预算，恢复未完成轮次时先完成该轮，不截断原生 epoch 或采样阶段来凑时间。`commit.json` 同时记录本轮时间、累计时间和模型路径，可用于后续绘图和模型比较；不能依据模型文件的修改时间推断训练时间。

每次阶段尝试写独立日志，已经观测到的失败尝试耗时也累计。若旧控制器在阶段或阶段之间异常退出、未能记录完整时间，或 learner 编译回执缺失／无效，则将 `timing_valid` 标为 false 并警告；未知编译量保存为 null。失效会传播到累计时间，后续成功轮次不能使此前缺失的时间重新有效。此时预算统一使用累计已观测墙钟（不再扣除任何轮次的编译时间），`timing_basis=observed_wall_incomplete`，不得用于精确等时间比较；缺失时段不会补零伪装有效。正常捕获并完整记录的失败后重试不包含停机间隔。

当前不提供跨文件系统断电事务；原子发布解决进程中断边界，不能代替持久化备份。
