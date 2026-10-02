# EtaZero V0

EtaZero 面向 Freestyle、Standard、Renju 五子棋。当前实现 WDL AlphaZero + PUCT 图搜索（局面共享、边访问追赶、FPU、Shaped Dirichlet Noise、Playout Cap Randomization、Reduce Visits、Forced Playout / Policy Target Pruning、训练与评估 LCB、Policy / Value Surprise Weighting）的完整逐轮训练链路：多局自对弈、共享多服务 GPU 批量推理与 NN 缓存、观测 bit packing 的完整对局与采样搜索数组 NPZ、压缩视图缓存与 multi-wave 窗口 shuffle、固定训练量与可选本地分段、checkpoint、模型校验发布与整轮恢复（启动前核验已提交权重与checkpoint）。冷启动使用 random evaluator；网络阶段支持 KataGomo 平衡开局与独立 policy init。自对弈支持非对称 PDA 预算与条件输入、独立 side 分支及递归；可显式启用 cheap-position reanalysis 和 direct value surprise。并行与主要数据管线参考 KataGo，算法语义见 [算法说明](docs/algorithms.md)。

推理可明确选择 FP32 / FP16，导出时在推理设备上预计算归一化逆标准差并保留原运算顺序；搜索采用分级 child 统计存储、复用线程和并行图节点回收。训练支持 batch 级 D4 增广、SGD / AdamW、参数分组与自适应衰减、样本计数 warmup、Lookahead、SWA、FP32 / AMP、网络与损失联合编译、CUDA fused AdamW、后台 batch 准备和独立 stream 上传。推理后端当前是 LibTorch；KataGo 专用原生算子后端尚未移植，不宣称性能完全对齐。

MuZero 和 Gumbel 属于后续阶段，选择它们会在启动 worker 前报错。当前提供单卡 learner、常驻 selfplay worker 和 shuffle 进程池、增量数据索引与可重建的派生数据回收；轮次仍依次执行各阶段。多卡 DDP、跨机器调度和异步 learner 不属于当前能力。

## 构建与小规模验证

以下命令在本版本目录执行，使用已有 Conda `pytorch` 环境。构建依赖 C++17、CMake、zlib、OpenSSL Crypto 和该环境的 LibTorch；构建与训练脚本不读取其他研究仓库的代码。`scripts/` 放构建、运行、调度和性能测量工具，独立来源对照检查放在 `tests/reference/`，用途与命令见[验收与限制](docs/implementation.md#验收与限制)。

```bash
bash scripts/build.sh
bash scripts/run.sh check-config --config-dir configs/smoke_test
CONFIG_DIR=configs/smoke_test bash scripts/run.sh --run-dir data/my_check --iterations 2 --plot
```

训练默认使用 `configs/baseline`，也可通过 `CONFIG_DIR` 指定配置；显式 `--config-dir` 优先于环境变量。不带子命令时直接训练，仍支持 `run`、`check-config`、`evaluate`、`match`、`arena` 和 `plot` 子命令。

```bash
CONFIG_DIR=configs/baseline bash scripts/run.sh
```

配置中的相对 `run.run_dir` 以本版本目录为基准，baseline、minimal_test、smoke_test 分别保存到 `data/baseline/`、`data/minimal_test/`、`data/smoke_test/`，包含自对弈数据、checkpoint、模型、日志与运行证据。`--run-dir` 可指定其他目录，相对路径以调用时的工作目录为基准。

iteration 0 用 random evaluator 完成 `selfplay.bootstrap_games` 局，只测量并保存实际采样行数／局。iteration 1 继续使用随机评估，补足 `replay.min_rows` 后执行首轮训练与模型发布，并以此时实际累计有效行数固定 replay 记账起点。iteration 2 起，一次 `iteration` 是完整的自对弈 → shuffle → 训练 → 模型校验与发布，自对弈使用本次迭代开始时的已发布模型，按固定训练量和 replay ratio 规划产样；shuffle 从历史数据窗口生成训练快照；learner 消费配置的 `training.train_steps` 个 batch；新模型校验发布后，下一次迭代使用它生成数据。各阶段顺序执行，阶段内部并行。

`[replay]` 使用 KataGo 的 power-law 窗口、random累计封顶、实际文件mtime近期排序和组内随机采样；支持 `taper_scale`、`add_to_data_rows`、`max_rows` 三项来源扩展。`keep_target_rows = all` 保留完整窗口。reader 每文件只消费完整batch并丢弃尾部；默认允许带间隔的文件重排重复消费，`training.no_repeat_files = true` 耗尽明确停止。默认启用验证（`skip_validation = false`），按原始文件basename的MD5留出1%，每轮训练结束用raw模型、随机D4验证，train/val跨快照不串用。`training.replay_ratio` 单独控制新增产样预算，配置和资源语义见 [窗口与 shuffle](docs/implementation.md#窗口与-shuffle)。

bootstrap 编号为 0，训练迭代从 1 编号，初始化 checkpoint 使用 0；`step` 是本次迭代内消费的 batch 数，`total_steps` 是整个运行累计消费 batch 数；`total_samples` 包含 AMP 跳步消费，`optimizer_steps` 单独记录成功优化器更新数。`.internal/state.json` 的 `iteration` 表示下一次待处理的迭代，中断时仍指向尚未完成的当前迭代，恢复从上一完整轮的 checkpoint 重跑中断轮；该轮原始产物归档到 `.internal/discarded/`，不进入回放或指标。baseline 的 `run.max_iteration = 0`、`run.max_seconds = 0`，默认持续运行，直到手动停止或发生错误。

`baseline` 是完整配置的起点。`minimal_test` 继承 baseline，网络参数由 [net.cfg](configs/minimal_test/net.cfg) 指定，使用 11×11 棋盘、`replay.min_rows = 100000` 和 `training.train_steps = 500`，并使用独立输出目录；评估和比赛棋盘同步为 11×11。搜索预算及 reduced 最低访问数由 [selfplay.cfg](configs/minimal_test/selfplay.cfg) 明确覆盖；其他设置继承 baseline，包括编译、batch、规则权重和并行度。`smoke_test` 是自动化快速验收配置，使用 5/6 混合尺寸、三种规则及很小的数据和训练预算。baseline 与 minimal_test 开启 `training.compile`，首次使用某个 shape / 精度时会有编译耗时；smoke_test 关闭它。构建默认 CUDA 架构为 RTX 5090 的 12.0，其他目标可通过 `TORCH_CUDA_ARCH_LIST` 指定。设备由配置明确指定，不自动回退到 CPU；在隐藏 GPU 的托管沙箱中，CUDA 命令须在宿主设备可访问的执行环境运行。

## 实验目录与开局

每个实验目录的公开入口为 `config/`、`logs/`、`selfplay/`、`snapshots/`、`checkpoints/` 和 `models/`，每轮提交后自动更新根目录 `training.png`、`loss.png` 和 `logs/performance.png`，支持手动重建。缓存、索引、轮次恢复状态和源码证据集中在 `.internal/`；完整布局见 [运行目录](docs/implementation.md#运行目录)。

`selfplay.cfg` 的 `[opening]`、`[policy_init]` 分别配置平衡开局和 policy init，基础配置沿用 KataGomo 的平衡开局参数。随机冷启动跳过二者，首次训练发布后启用。开局动作保留用于恢复完整对局，但不进入训练监督；局长包含开局，产样预算与 shuffle 统计 cap 与 surprise 随机取整后的实际训练行。机制、来源及适配差异见 [平衡开局](docs/algorithms.md#平衡开局与-policy-init)。

## 网络、优化器与输入

`network.architecture` 可选择 `nbt`（默认）、`plain` 或 `transformer`。NBT 对照 KataGo `b5c192nbt-fson-mish`，可按 `network.channels` 和 `network.blocks` 缩放，内部瓶颈及 Head 宽度自动推导；主干通道须为不小于24的偶数，Head宽度向上对齐到8的倍数。plain 对照 `b10c128-fson-mish`，须配置128通道、10块，第5/8块含gpool，Head宽度32、Value隐藏层80。两个卷积预设采用Mish和固定尺度归一化，仅主干末端使用masked BatchNorm。Transformer严格对应bare `b5c192h3nbttfrs`：192通道、5块、mid96、3头、FFN256、fixup/ReLU和v17默认无Q监督；使用真实RMSNorm、2D RoPE、attention及SwiGLU，Value隐藏层64。三架构共享训练和导出入口，按预设使用各自参数组与初始化；`network.predict_q_values` 默认 false，仅Transformer v17可设true，添加第七个逐动作纯W−L Q训练输出；卷积v15拒绝启用。配置入口为 [net.cfg](configs/baseline/net.cfg)，结构与适配边界见 [输入与网络](docs/algorithms.md#观测动作与网络)。改变结构使用新的空运行目录，历史checkpoint不迁移；已发布的TorchScript模型可独立加载。

训练按固定 KataGo 来源的 fson 优化分支组织，默认 SGD＋momentum 0.9，可显式选择 AdamW。`[optimizer]` 管理 LR scale、分组学习率与衰减、warmup、Lookahead 和 SWA；字段以 [train.cfg](configs/baseline/train.cfg) 和 [optimization.py](python/etazero/optimization.py) 为准。训练输出包含六项 policy、WDL、三个时间尺度 TD WDL 和短期价值误差；v17可选Q按子节点访问数的平方根加权，不受side完整主局标志门控；基础策略系数、TD 的 CE 减目标熵、误差梯度及 optimistic 权重采用固定来源的 v15 定义与五子棋无 score 映射。推理导出主/短期 optimistic logits、WDL 和价值误差标准差，搜索按 root/leaf optimism 混合主/短期 optimistic logits，并可用误差标准差加权 NN 样本；noise pruning 修正价值聚合。反向传播使用 batch 总和，日志保持样本均值。普通 policy CE 系数为 0.930，基础 value CE 为 0.72（来源内部 1.20 × 默认 scale 0.6）。训练 AMP 的主干使用所选精度，policy/value heads 与 loss 保持 FP32；FP16 overflow 跳过本次 optimizer 更新，消费、Lookahead 和 SWA 时钟继续推进。默认开启训练 D4，每个消费 batch 均匀选择一个对称，同步变换空间特征、mask、policy、Q值与Q访问数目标。机制与适配边界见 [学习与数据使用](docs/algorithms.md#学习与数据使用)。

模型发布优先采用有采样的 SWA 权重，否则使用当前训练权重；checkpoint 同时保存训练模型、优化器、Lookahead 和 SWA。整轮成功提交与发布后，删除已提交轮的中间 checkpoint `.pt`，按 `training.checkpoint_keep` 保留最近若干轮末的完整训练状态（初始化也计入）；所有 checkpoint `.json`、历史推理模型、原始对局与日志继续保留。训练入口仍从上一个完整轮恢复。优化器配置或训练条件变化使用新的运行目录，旧优化器状态不作迁移；原始数据与模型输入契约保持当前定义。

输入对齐 SkyZero_V7.19 的 **5 个空间平面＋6 个全局特征**：有效棋盘、己方、对方、黑视角黑方禁手、白视角黑方禁手；全局量为 Standard、Renju、Renju 执色及禁手特征开关。随后为 PDA 启用标志及按执色变号的半倍 doubling 值；未加入 draw utility。`environment.forbidden_feature_dropout_prob` 对每个最终输出的 Renju 训练行独立抽样，同步隐藏两个禁手平面和开关；重复行可不同，抽样结果随原始分片保存，搜索、平衡开局与评估始终使用完整特征。详细定义见 [输入与网络](docs/algorithms.md#观测动作与网络)。

`environment` 中尺寸与规则分别按权重抽样；零权重表示不采样，每组至少有一个正权重。所有尺寸共用网络画布，通过 mask 排除 padding。输入契约变更后的模型和数据须使用新的运行目录，不混用其他契约的产物。

SP 的 uncertainty/noise pruning 默认关闭、root/leaf optimism 为零；Eval/Match 默认启用两项修正并采用 root/leaf optimism 0.2/1。公式、缺少辅助能力时的分支和统计口径见 [搜索修正](docs/algorithms.md#误差加权optimistic-policy-与-noise-pruning)。

SP、Eval/Match 默认启用图搜索，相同局面共享节点统计，父边保留独立访问；配置可关闭图共享或设置追赶泄漏率。NOVC key包含棋盘、执子、棋规、画布及终局结果；根推进复制共享子节点，静止后标记可达节点并回收其余节点。模型变化或不兼容局面清空图，NN cache独立于图节点表。详见 [图搜索](docs/algorithms.md#图搜索与局面共享)。

WDL 输出顺序为当前玩家视角 W/D/L，搜索 Q 为 W−L；节点采用 KataGo 的子树价值加权与加权 ESS，PUCT 保留 0.01 探索偏移及可配置的 log／标准差项，virtual loss 以 pending × 配置值形成虚拟权重。自对弈落子不使用 LCB，训练 policy target 使用 LCB，评估与比赛落子使用 LCB。根支持 1–8 个 D4 对称的概率平均；全树 NN policy 温度、根 policy 温度与落子动作温度分别配置，后两者按棋盘面积缩放的半衰期衰减。当前 baseline 自对弈的根温度为1.5→1.1、落子温度为0.75→0.15、半衰参数为15（来源主线为19），根使用4个对称；eval/match 单次推理默认随机 D4，policy 温度 1，落子温度按 Match profile 衰减。单朝向 cache 与朝向无关，根集成绕过 cache；训练 AMP 与推理精度独立，SP 默认 FP16，eval/match 支持并记录 auto 解析后的实际精度。默认完整搜索清树，零权重 cheap 搜索保留推进后的子树，关闭根噪声、根温度、多对称与 forced playout，保留全树温度。参数见 [selfplay.cfg](configs/baseline/selfplay.cfg)，公式与采样口径见 [搜索和学习](docs/algorithms.md#puct-与搜索预算)。模型输出和原始数据契约已包含 WDL 与采样次数，实验需使用新的空运行目录。

PDA 的普通局抽样及 side 分支概率见 [selfplay.cfg](configs/baseline/selfplay.cfg)。Side 始终清除 PDA，以自己的搜索 WDL 训练，不借用主局终局；reanalysis 在局后替换所选 cheap 位置的搜索目标，保留真实对局。开启 reanalysis 须显式提供其采样/权重/目标字段；默认关闭。机制与 gate 见 [采样分支](docs/algorithms.md#pda侧分支与重分析)。自对弈已接入 hint / hintFork 及 early/game fork。外部 hint 默认概率为0；hint/fork前缀独立记账并不进入监督，fork池跨常驻worker轮次保留，但不写入checkpoint，进程重启清空，与KataGo的内存池生命周期对应。配置与输入格式见 [采样分支](docs/algorithms.md#pda侧分支与重分析)。

## 训练图与自动实验

`training.png` 使用深色三行两列布局，顶部显示黑白胜／和棋和包含开局的局长，中部保留固定分组的策略 loss 与价值 loss，底部显示裁剪前梯度范数和 NN 缓存命中率。策略组包含普通／对手、soft 和 optimistic policy，价值组包含主 value、三个 TD、短期价值误差及启用时的 Q；两组图例在面板内分两列显示。缓存命中率为该轮网络推理的 cache hits／submitted requests，随机冷启动不伪造零值。总 loss 和有效行数／局保留在日志中，不单独占用概览面板。累计有效行数横轴采用 `1.2e5` 形式的紧凑科学计数。loss 与梯度是该轮已提交更新的原始算术均值，无平滑；bootstrap 编号为 0，不含训练指标。恢复后绘图按 checkpoint 提交链去重，只展示整轮已提交更新。

`loss.png` 以四列面板逐项展示总 loss 与每个实际启用的 loss 分量，蓝色实线为该轮训练均值，红色虚线为轮末 raw 模型的验证均值；每项独立纵轴，正值使用对数刻度。无验证或验证无完整 batch 时保留缺测，不填零；恢复重试前的验证记录不进入图表。训练是轮内均值，验证是轮末 eval/no_grad，二者时点和数据不同，曲线差距不直接等同于过拟合。阶段耗时、吞吐、组批、排队和缓存命中另存 `logs/performance.png`。

```bash
bash scripts/run.sh plot --run-dir data/my_check
# 两个相同的 smoke_test 工程验收臂，逐个占用 GPU 0；不是正式研究配置。
CONFIG_DIR=configs/autoexp_example bash scripts/autoexp.sh --dry-run
CONFIG_DIR=configs/autoexp_example bash scripts/autoexp.sh
```

`autoexp.sh` 从实验伞目录的 `exp.cfg` 读取统一预算和 GPU 槽位，发现包含 `run.cfg` 的直接子目录作为实验臂。每臂配置继承、产物和恢复仍遵循普通训练规则。已达预算的臂跳过，其他臂排队恢复；失败停止排队并通知其他 controller 安全保存退出。`shared_init` 按网络结构与种子共享初始权重，各臂独立生成 bootstrap 数据。调度用法、环境覆盖与恢复边界见 [自动实验](docs/implementation.md#自动实验)。

## 续训与评估

```bash
# --iterations 为该运行希望完成的累计轮数，已完成轮次不会重复训练。
CONFIG_DIR=configs/smoke_test bash scripts/run.sh --run-dir data/my_check --iterations 3

# 默认评估已发布模型，动作使用实际棋盘上的零起始行优先编号。
bash scripts/run.sh evaluate --config-dir configs/smoke_test --run-dir data/my_check --size 5 --rule renju --moves 0,6

# model-b 指向另一份已校验发布的 model.pt，比赛生成成对平衡开局并交换模型执色。
bash scripts/run.sh match --config-dir configs/smoke_test --run-dir data/my_check --model-b /absolute/path/to/models/model_id/model.pt --size 5 --rule freestyle --games 4 --output data/my_check_match

bash scripts/run.sh plot --run-dir data/my_check
```

输出目录存在 `.internal/run.json` 时自动恢复完整训练状态，无需 `--resume`；显式 `--resume` 要求已有运行。新运行必须使用空目录，非空但缺少运行记录的目录会报错。配置的 `run.max_iteration` 和 `--iterations` 都是累计迭代次数上限，`0` 表示不限制迭代次数；达到非零上限后重启不会追加训练，追加训练使用 `--iterations` 提高累计目标。`--iterations` 只覆盖此次调用的目标，不修改保存的生效配置。

`--weights /absolute/path/to/checkpoint.pt` 为新运行导入兼容网络的模型状态，重新初始化优化器、计数和随机流，不能用于已有运行。恢复要求生效配置一致，改变实验条件时按用户的新运行或续训方案处理。

## Web 开发工作台

在仓库根目录运行 `bash web/webui.sh`，打开 <http://127.0.0.1:8766>。默认选择 `minimal_test` 当前已发布模型，支持模型刷新、人机对弈、手动局面研究、仅分析与 AI 单步、历史回看与回退继续、完整候选分布和策略热力图。采用紧凑三栏布局，可查看模型元信息、生效搜索配置和原始分析数据。使用常驻 LibTorch 模型和本版本原生棋规、PUCT；用法、加载范围和验证入口见 [Web 开发工作台](../web/README.md)。

## 等时间 Elo

训练时间与 `autoexp` 预算采用已提交完整轮次的累计墙钟，包含该轮自对弈、shuffle、训练、导出和绘图，排除暂停、启动恢复和作废轮次；预算在整轮边界检查，可能超出一轮。原始日志的 `active_seconds` 仅作运行审计，不用作 Elo 横轴。

`eval.cfg` 与 `match.cfg` 独立于训练配置，修改它们不影响续训。baseline 默认 **500 根访问（500v）**、无根噪声，使用 Match profile 的温度、FPU、方差探索、目标剪枝与 LCB。smoke_test 显式使用 100v。不同 bot 的 Match 默认复用命中子树，相同模型身份双方每手清树；固定局面 eval 默认不复用。访问上限计入已有树统计，max_playouts 限制新增 playout，max_time 限制本手时间；计数与停止边界见 [搜索预算](docs/algorithms.md#puct-与搜索预算)。比赛使用 KataGomo 的随机 balance bot及按执色 policy init，保留 EtaZero 成对换色、逐局续测与联合 Elo 体系，详见 [评估与 Elo](docs/elo.md)。

```bash
# data/my_experiment 包含各实验臂；也可直接传单个运行目录，比较历史模型。
bash scripts/run.sh arena --data data/my_experiment --output data/my_experiment_elo --dry-run
bash scripts/run.sh arena --data data/my_experiment --output data/my_experiment_elo
# 原命令重启即可续测；只重建评分和图：
bash scripts/run.sh arena --data data/my_experiment --output data/my_experiment_elo --fit-only
```

旧的阶段恢复目录不自动转成整轮提交格式；新实验使用新目录，原有产物保持不变。

## 文档路由

| 任务 | 阅读入口 |
|---|---|
| 查看对局线程、搜索线程、共享组批及 C++ / Python 分工 | [执行架构](docs/implementation.md#执行架构) |
| 测量并调整 selfplay 并行局数、推理 batch 和服务数 | [并行参数短测](docs/implementation.md#selfplay-并行参数短测) |
| 理解原始对局、两阶段 shuffle 和训练预取 | [数据链路](docs/implementation.md#数据链路) |
| 查看 replay ratio、累计缺口及 selfplay 局数 | [产样规划](docs/implementation.md#固定训练量与自对弈产量) |
| 调整平衡开局及独立 policy init | [开局机制](docs/algorithms.md#平衡开局与-policy-init) |
| 查看实验产物和内部状态位置 | [运行目录](docs/implementation.md#运行目录) |
| 调整配置、继承与本机覆盖 | [配置组织](docs/implementation.md#配置组织) |
| 固定评估预算、成对比赛、续测与 Elo 图 | [评估与 Elo](docs/elo.md) |
| 网页对弈、手动局面研究与搜索诊断 | [Web 开发工作台](../web/README.md) |
| 查看 checkpoint、模型发布、故障恢复与运行证据 | [发布与恢复](docs/implementation.md#模型发布与恢复) |
| 理解棋规、网络、价值视角、PUCT 和训练目标 | [AlphaZero](docs/algorithms.md#第一轮-alphazero) |
| 查看未来算法的组合约束和接入边界 | [算法组合](docs/algorithms.md#算法与搜索组合)、[后续接入](docs/algorithms.md#后续算法接入) |
| 查看对齐profile、网络版本、五子棋监督映射和57项验收范围 | [对齐目标](../EtaZero.md#alignment-targets)、[实施计划](../plan.md) |
| 核对工程执行路径、样本消费与并发边界的静态差异 | [E01—E26 工程审查](../EtaZero.md#engineering-audit)、[分批实施与验收](../plan.md) |
| 核对代码来源和许可证 | [参考来源](THIRD_PARTY.md)、[来源校验值](reference_sources.json) |
| 训练图、实验伞目录与 GPU 排队 | [绘图与自动实验](docs/implementation.md#训练图与性能图)、[调度](docs/implementation.md#自动实验) |
| 运行检查和了解验证限制 | [验收与限制](docs/implementation.md#验收与限制) |

本 README 是唯一总入口。工程与数据行为维护在 `docs/implementation.md`，算法语义维护在 `docs/algorithms.md`；字段以 [config.py](python/etazero/config.py) 的归属与校验、[baseline](configs/baseline/) 的完整参数为事实源。
