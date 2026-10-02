# EtaZero V0

EtaZero 面向 Freestyle、Standard、Renju 五子棋。当前实现 WDL AlphaZero + PUCT（FPU、Shaped Dirichlet Noise、Playout Cap Randomization、Reduce Visits、Forced Playout / Policy Target Pruning、训练与评估 LCB、Policy / Value Surprise Weighting）的完整逐轮训练链路：多局自对弈、共享多服务 GPU 批量推理与 NN 缓存、观测 bit packing 的完整对局与采样搜索数组 NPZ、压缩视图缓存与 multi-wave 窗口 shuffle、固定训练量、checkpoint、模型校验发布与整轮恢复。冷启动使用 random evaluator；网络阶段支持 KataGomo 平衡开局与独立 policy init。并行与主要数据管线参考 KataGo，算法语义见 [算法说明](docs/algorithms.md)。

推理可明确选择 FP32 / FP16，导出时在推理设备上预计算归一化逆标准差并保留原运算顺序；搜索采用分级 child 统计存储、复用线程和并行大树回收。训练支持 batch 级 D4 增广、SGD / AdamW、参数分组与自适应衰减、样本计数 warmup、Lookahead、SWA、FP32 / AMP、网络与损失联合编译、CUDA fused AdamW、后台 batch 准备和独立 stream 上传。推理后端当前是 LibTorch；KataGo 专用原生算子后端尚未移植，不宣称性能完全对齐。

MuZero 和 Gumbel 属于后续阶段，选择它们会在启动 worker 前报错。当前提供单卡 learner、常驻 selfplay worker 和 shuffle 进程池、增量数据索引与可重建的派生数据回收；轮次仍依次执行各阶段。多卡 DDP、跨机器调度和异步 learner 不属于当前能力。

## 构建与小规模验证

以下命令在本版本目录执行，使用已有 Conda `pytorch` 环境。构建依赖 C++17、CMake、zlib 和该环境的 LibTorch；脚本不读取其他研究仓库的代码。

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

iteration 0 用 random evaluator 完成 `selfplay.bootstrap_games` 局，只测量并保存实际采样行数／局。iteration 1 继续使用随机评估，补足 `replay.min_rows` 后执行首轮训练与模型发布，并以此时实际累计有效行数固定 replay 记账起点。iteration 2 起，一次 `iteration` 是完整的自对弈 → shuffle → 训练 → 模型校验与发布，自对弈使用本次迭代开始时的已发布模型，按固定训练量和 replay ratio 规划产样；shuffle 从历史数据窗口生成训练快照；learner 完成配置的 `training.train_steps` 次 batch 更新；新模型校验发布后，下一次迭代使用它生成数据。各阶段顺序执行，阶段内部并行。

`[replay]` 使用 KataGo 的窗口与采样模式，仅设置 `min_rows`、`taper_exponent`、`expand_per_row` 和 `keep_target_rows`；窗口按公式增长，每次 shuffle 从窗口中随机采样接近 `keep_target_rows` 行，`all` 表示全部保留。`training.replay_ratio` 控制新增产样预算，`[shuffle]` 配置并行与存储资源，具体语义见 [窗口与 shuffle](docs/implementation.md#窗口与-shuffle)。

bootstrap 编号为 0，训练迭代从 1 编号，初始化 checkpoint 使用 0；`step` 是本次迭代内成功完成的优化器更新次数，`total_steps` 是整个运行累计更新次数。`.internal/state.json` 的 `iteration` 表示下一次待处理的迭代，中断时仍指向尚未完成的当前迭代，恢复从上一完整轮的 checkpoint 重跑中断轮；该轮原始产物归档到 `.internal/discarded/`，不进入回放或指标。baseline 的 `run.max_iteration = 0`、`run.max_seconds = 0`，默认持续运行，直到手动停止或发生错误。

`baseline` 是完整配置的起点。`minimal_test` 继承 baseline，网络参数由 [net.cfg](configs/minimal_test/net.cfg) 指定，使用 11×11 棋盘、`replay.min_rows = 100000` 和 `training.train_steps = 500`，并使用独立输出目录；评估和比赛棋盘同步为 11×11。其余设置直接继承 baseline，包括编译、batch、搜索预算、规则权重和并行度。`smoke_test` 是自动化快速验收配置，使用 5/6 混合尺寸、三种规则及很小的数据和训练预算。baseline 与 minimal_test 开启 `training.compile`，首次使用某个 shape / 精度时会有编译耗时；smoke_test 关闭它。构建默认 CUDA 架构为 RTX 5090 的 12.0，其他目标可通过 `TORCH_CUDA_ARCH_LIST` 指定。设备由配置明确指定，不自动回退到 CPU；在隐藏 GPU 的托管沙箱中，CUDA 命令须在宿主设备可访问的执行环境运行。

## 实验目录与开局

每个实验目录的公开入口为 `config/`、`logs/`、`selfplay/`、`snapshots/`、`checkpoints/` 和 `models/`，每轮提交后自动更新根目录 `training.png` 和 `logs/performance.png`，支持手动重建。缓存、索引、轮次恢复状态和源码证据集中在 `.internal/`；完整布局见 [运行目录](docs/implementation.md#运行目录)。

`selfplay.cfg` 的 `[opening]`、`[policy_init]` 分别配置平衡开局和 policy init，基础配置沿用 KataGomo 的平衡开局参数。随机冷启动跳过二者，首次训练发布后启用。开局动作保留用于恢复完整对局，但不进入训练监督；局长包含开局，产样预算与 shuffle 统计 cap 与 surprise 随机取整后的实际训练行。机制、来源及适配差异见 [平衡开局](docs/algorithms.md#平衡开局与-policy-init)。

## 网络、优化器与输入

默认网络使用 KataGo 风格的 NBT2 卷积主干、全局 PolicyHead 和尺寸条件化 ValueHead，采用 Mish 与固定尺度归一化，只有主干末端使用 masked BatchNorm。baseline 与 minimal_test 的架构由各自 `net.cfg` 指定；缩放网络只需调整 `network.blocks` 和 `network.channels`，内部瓶颈及各 Head 宽度自动推导，无需独立设置 `value_hidden`。通道须为不小于 24 的偶数，Head 宽度向上对齐到 8 的倍数；完整结构、推导和 KataGo 适配边界见 [输入与网络](docs/algorithms.md#观测动作与网络)。改变结构使用新的空运行目录，历史 checkpoint 不迁移；已有 TorchScript 推理模型可独立加载。

训练按 SkyZero_V8.1 的 native BN 优化策略组织，默认 SGD＋momentum 0.9，可显式选择 AdamW。`[optimizer]` 管理 LR scale、分组学习率与衰减、warmup、Lookahead 和 SWA；字段以 [train.cfg](configs/baseline/train.cfg) 和 [optimization.py](python/etazero/optimization.py) 为准。损失为 policy、opponent policy、soft policy、soft opponent policy 与 WDL 五项加权交叉熵；四项策略目标与系数对齐 KataGo，推理只使用主 policy。反向传播使用 batch 总和，日志保持样本均值。默认开启训练 D4，每个消费 batch 均匀选择一个对称，同步变换空间特征、mask 和 policy 目标。机制与适配边界见 [学习与数据使用](docs/algorithms.md#学习与数据使用)。

模型发布优先采用有采样的 SWA 权重，否则使用当前训练权重；checkpoint 同时保存训练模型、优化器、Lookahead 和 SWA。整轮成功提交与发布后，删除已提交轮的中间 checkpoint `.pt`，按 `training.checkpoint_keep` 保留最近若干轮末的完整训练状态（初始化也计入）；所有 checkpoint `.json`、历史推理模型、原始对局与日志继续保留。训练入口仍从上一个完整轮恢复。优化器配置或训练条件变化使用新的运行目录，旧优化器状态不作迁移；原始数据与模型输入契约保持当前定义。

输入对齐 SkyZero_V7.19 的 **5 个空间平面＋4 个适用的全局特征**：有效棋盘、己方、对方、黑视角黑方禁手、白视角黑方禁手；全局量为 Standard、Renju、Renju 执色及禁手特征开关。未接入 draw utility/PDA。`environment.forbidden_feature_dropout_prob` 只对 Renju 训练行同步隐藏两个禁手平面和开关，搜索、平衡开局与评估始终使用完整特征。详细定义见 [输入与网络](docs/algorithms.md#观测动作与网络)。

`environment` 中尺寸与规则分别按权重抽样；零权重表示不采样，每组至少有一个正权重。所有尺寸共用网络画布，通过 mask 排除 padding。输入契约变更后的模型和数据须使用新的运行目录，不混用其他契约的产物。

WDL 输出顺序为当前玩家视角 W/D/L，搜索 Q 为 W−L；节点采用 KataGo 的子树价值加权与加权 ESS，PUCT 保留 0.01 探索偏移及可配置的 log／标准差项，virtual loss 按样本权重处理。自对弈落子不使用 LCB，训练 policy target 使用 LCB，评估与比赛落子使用 LCB。根支持 1–8 个 D4 对称的概率平均；全树 NN policy 温度、根 policy 温度与落子动作温度分别配置，后两者按棋盘面积缩放的半衰期衰减。baseline 自对弈采用 SkyZero_V8.1 的温度设置和 4 个根对称，eval/match 默认单对称、policy 温度 1、落子温度 0。默认完整搜索清树，零权重 cheap 搜索保留推进后的子树，关闭根噪声、根温度、多对称与 forced playout，保留全树温度。参数见 [selfplay.cfg](configs/baseline/selfplay.cfg)，公式与采样口径见 [搜索和学习](docs/algorithms.md#puct-与搜索预算)。模型输出和原始数据契约已包含 WDL 与采样次数，实验需使用新的空运行目录。

## 训练图与自动实验

`training.png` 沿用 MuZero 的深色六面板布局，显示黑白胜／和棋、包含开局的局长、总 loss、四项 policy／value 分量、排除开局的有效行数／局和裁剪前梯度范数。loss 与梯度是该轮已提交更新的原始算术均值，无平滑；bootstrap 编号为 0，不含训练指标。恢复后绘图按 checkpoint 提交链去重，只展示整轮已提交更新。阶段耗时、吞吐、组批、排队和缓存命中另存 `logs/performance.png`。

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

## 网页人机对弈

在仓库根目录运行 `bash web/webui.sh`，打开 <http://127.0.0.1:8766>。默认选择 `minimal_test` 当前已发布模型，可切换历史权重、棋规、执子和搜索预算，支持悔棋、对局记录及网络／搜索策略热力图。使用常驻 LibTorch 模型和本版本原生棋规、PUCT；用法、加载范围和验证入口见 [Web 对弈室](../web/README.md)。

## 等时间 Elo

训练时间与 `autoexp` 预算采用已提交完整轮次的累计墙钟，包含该轮自对弈、shuffle、训练、导出和绘图，排除暂停、启动恢复和作废轮次；预算在整轮边界检查，可能超出一轮。原始日志的 `active_seconds` 仅作运行审计，不用作 Elo 横轴。

`eval.cfg` 与 `match.cfg` 独立于训练配置，修改它们不影响续训。默认均为 **100 根访问（100v）**、无根噪声、零落子温度、不复用树，使用各自 profile 的 FPU、目标剪枝与 LCB；新根的一次初始评估加 99 次边模拟共 100v。比赛沿用 MuZero_V2 的双方生成平衡开局、交换执色、逐局续测与联合 Elo 体系，详见 [评估与 Elo](docs/elo.md)。

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
| 理解原始对局、两阶段 shuffle 和训练预取 | [数据链路](docs/implementation.md#数据链路) |
| 查看 replay ratio、累计缺口及 selfplay 局数 | [产样规划](docs/implementation.md#固定训练量与自对弈产量) |
| 调整平衡开局及独立 policy init | [开局机制](docs/algorithms.md#平衡开局与-policy-init) |
| 查看实验产物和内部状态位置 | [运行目录](docs/implementation.md#运行目录) |
| 调整配置、继承与本机覆盖 | [配置组织](docs/implementation.md#配置组织) |
| 固定评估预算、成对比赛、续测与 Elo 图 | [评估与 Elo](docs/elo.md) |
| 加载 minimal_test 模型进行网页人机对弈 | [Web 对弈室](../web/README.md) |
| 查看 checkpoint、模型发布、故障恢复与运行证据 | [发布与恢复](docs/implementation.md#模型发布与恢复) |
| 理解棋规、网络、价值视角、PUCT 和训练目标 | [AlphaZero](docs/algorithms.md#第一轮-alphazero) |
| 查看未来算法的组合约束和接入边界 | [算法组合](docs/algorithms.md#算法与搜索组合)、[后续接入](docs/algorithms.md#后续算法接入) |
| 核对代码来源和许可证 | [参考来源](THIRD_PARTY.md)、[来源校验值](reference_sources.json) |
| 训练图、实验伞目录与 GPU 排队 | [绘图与自动实验](docs/implementation.md#训练图与性能图)、[调度](docs/implementation.md#自动实验) |
| 运行检查和了解验证限制 | [验收与限制](docs/implementation.md#验收与限制) |

本 README 是唯一总入口。工程与数据行为维护在 `docs/implementation.md`，算法语义维护在 `docs/algorithms.md`；字段以 [config.py](python/etazero/config.py) 的归属与校验、[baseline](configs/baseline/) 的完整参数为事实源。
