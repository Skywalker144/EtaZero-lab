# MuZero 网络与展开训练

MuZero 的三组件网络、逐步 source loss、组件导出、原生描述器、CUDA／Eigen／OpenCL initial/recurrent、跨线程组批、局面适配及来源 `Search` 的 latent 路径已实现，入口位于 `python/katago/train/muzero_model.py`、`muzero_training.py`、`cpp/neuralnet/muzerobatcher.h`、`muzeroposition.h` 和 `cpp/search/search.h`。完整轨迹 writer、来源 shuffle、展开 reader、正式 learner 与原生 `GameRunner` 已接通，见下文。selfplay 命令与 EtaZero 外层控制器已接通三组件初始化、逐轮训练、普通/SWA 发布和阶段恢复；replay-ratio 与可选 gatekeeper 共用外层控制器，门控比赛加载完整三组件服务，拒绝候选不回滚 learner。

## 网络与状态

representation、dynamics、prediction 使用三个独立的来源 `Model`，分别拥有参数和归一化统计。配置直接使用 KataGo 的网络配置字典，可分别选择深度、宽度、块类型及归一化。Go、Freestyle/Standard/Renju、Hex 和象棋已贯通三组件网络、展开 loss、组件导出、原生 CUDA initial/recurrent、局面适配、PUCT/Gumbel latent 搜索、轨迹 writer/shuffle/reader/learner、GameRunner/selfplay 与逐轮调度。`MuZeroConfig` 保存三个配置及 latent 通道数，运行时不依赖旧版 EtaZero。

来源 `Model` 增加仅输出 trunk、自定义空间通道、无 global 输入以及 MuZero 零主干 prediction 的入口。默认 AlphaZero 构造方式不变。两个 latent body 不建立未使用的预测 heads；prediction 保留来源主头及全部适用辅助输出；有主干时可按来源配置启用中间头。卷积/NBT/attention、register、共享 attention 数据、归一化、初始化和 trunk 更新在原来的调用位置执行，没有另写一个简化的 block 循环。各来源优化路径的设备适用条件仍然有效。

representation 接收该棋类的真实观测、global 输入和可选 metadata。latent 为 FP32 `[B,C+1,H,W]`：前 C 通道是 learned state，末通道为不可变的有效棋盘 mask。每个样本在有效格点和所有特征通道上做 min/max；跨度小于 `1e-5` 时，分母为跨度加 `1e-5`，否则使用原跨度。棋盘外归零，常量 latent 归零。来源与许可证见 [reference_sources.json](../reference_sources.json) 的 MiniZero、EtaZero_MuZero 项。

dynamics 只接收 latent 和 int64 `[B]` 动作。落子棋动作编码为棋盘 one-hot；Go 的 pass 编码为全零动作平面。象棋使用固定 2086 项联合动作表，编码为起点和终点两个 one-hot 平面，dynamics 的空间输入宽度为 C+3（mask、C 个 latent 通道、两个动作通道）。动作、latent 始终使用根的固定朝向，允许重复动作；不得在这里使用真实后继棋盘、占用状态、超级劫或终局规则。输入数据与推理调用方负责保证 mask 非空、动作处于该棋类的有效静态域。

prediction 只读取 latent。目录配置的 `prediction_blocks` 默认 0，对应原生组件字典的空 `block_kind`：hidden 特征直接进入 heads，mask 用于掩码和池化，不额外经过输入卷积、归一化或激活。设为正整数时使用所选 KataGo 预设的输入卷积、指定数量的主干块和末端归一化，再进入 heads；配置说明见 [网络配置](configuration.md)。输出沿用来源 policy、W/D/L/NR、三组 TD、Q、future-position、variance-time 和短期误差 heads。非围棋训练经 `postprocess_output` 按模型版本 18/19 的路径屏蔽 NR，原始 head logits 尚未应用该屏蔽。象棋沿用版本 19 的联合策略 head，policy/Q 按起点与位移 gather 为 2086 项，没有 pass 分支或分别归一化的起终点分布。价值预测对应当前 latent 步的行动方，空间输出与动作始终使用固定根坐标；后续搜索在边上交换 W/L，训练数据按每步真实执色提供目标。棋类没有中间奖励，本接口不添加 reward 或 terminal predictor。

## 展开与 loss

`MuZeroModel.unroll` 接收一份根观测、int64 `[B,K]` 动作和可选 bool `[B,K+1]` 有效步掩码，产出每步的原始行索引、latent 与 source heads。网络不读取未来观测。有效步从某处结束后不再恢复；side 独立位置只在根有效，退出的行在下一次 dynamics **之前**移除，不参与虚构转移或 recurrent BatchNorm。

`unroll_loss` 对每步目标直接调用来源 `Metrics`，保留适用 loss 的权重、detach、量化后的归一化和 batch-sum 语义。未来步的目标不由该函数推测，由下述轨迹 reader 提供。有效步掩码表示是否存在转移，不能用某步 policy/value 权重为零代替它；reader 提供 CPU 掩码时，展开的行索引也留在 CPU，避免每步通过 GPU 索引读取 CPU 掩码或执行 CUDA nonzero。

反向系数为根 1、未来各步 `1/K`，日志保存未施加该反向系数的 forward loss。第一次 representation→dynamics 不缩放 hidden 梯度，第二次及以后按 `hidden_gradient_scale` 缩放。该函数不推进 epoch、优化器、Lookahead 或 SWA 时钟；正式入口由来源 `python/train.py` 管理这些机制。

参数分组来自各组件的 `Model.add_reg_dict`。representation/dynamics 的末端归一化属于内部参数，将来源末端的 output/output_noreg 角色分别映射为 normal_gamma/noreg；latent 投影属于 normal。prediction 的 heads 保留来源输出角色；有主干时，其末端归一化也保留来源输出角色。没有 global 输入的组件不建立虚假的 global 参数。构造、调用 `initialize()`、再加载 state dict 的顺序与来源加载流程一致；初始化还设置不在 state dict 中的固定尺度。

## 完整轨迹与来源数据管线

`TrainingDataWriter` 的 `writeMuZeroTrajectories` 选项在普通起点 NPZ 内附带完整主线轨迹。起点仍由来源权重、随机取整和重复行逻辑决定，`trainingWeight` 仍在原位置写入。附加轨迹覆盖从 `startHist` 开始的每个实际步骤及最后观测状态，包括未被采为起点的 cheap/低权重位置；不把 policy-init 或 SGF 开局前缀变成训练起点。每个 shard 包含其起点所引用的完整对局，因此文件边界不会截断展开。

普通起点与完整轨迹调用同一个 `TrainingWriteBuffers::addRow`，保留来源 TD 混合、variance-time、future-position、Q 量化及 reanalysis 的 outcome 开关。附加目标使用独立随机流，不消耗原有起点采样、Q 量化、历史 dropout 或文件名随机数。原始文件中保存根的真实特征；轨迹只增加动作、未取整采样权重和各步目标，不保存未来网络输入。字段布局由 [muzerowrite.cpp](../cpp/dataio/muzerowrite.cpp) 和 [muzero_data.py](../python/katago/train/muzero_data.py) 定义。

象棋原始轨迹继续使用来源 writer 的逐行行动方坐标，真实动作与该行 policy/Q 使用同一联合动作表。shuffle 构造展开视图时，将各未来步的动作、两路 policy、Q 和未来局面位置转到采样根坐标；奇数步在物理 9×10 范围内旋转 180°，padding 不参与。价值与 Q 数值仍属于各步当前行动方，不因坐标变化再次交换 W/L 或取负。超过真实终局的吸收态固定使用末局面的根坐标，只随执色翻转棋子颜色和价值标签；不逐步旋转末局面，也不从真实终局获取后继合法动作。

来源 `shuffle.py -muzero-unroll-steps K` 在加载 raw shard 时展开 K 步目标，再与根数组一起使用来源的窗口、keep 比例、同一排列、分桶、合并及多 wave 流程。改变 K 不需要重新采样原始对局。选定文件窗口后、起点 keep 采样前，按 game hash 去重计算真实主线位置的平均未取整权重；均值和 K 写入来源 snapshot JSON。根已由 multiplicity 加权，不能再次乘主线采样权重；真实未来步乘自己的未取整权重除以该窗口均值，再保留来源 `trainingWeight`。终局和截断末状态不计入均值，side 也不计入主线均值。

side position 只在根有效，不沿原对局伪造分支后续。主线的真实终局使用终局四结果延伸吸收态，按每步执色交换 W/L，D/NR 不变；策略为静态动作域均匀分布，Q/visits 为零，未来棋盘沿用终局棋盘并翻转视角。吸收态动作由 reader 在每次消费时从根的静态域采样，包含 Go pass，不使用真实终局合法性裁剪。手数截断只展开到最后观测状态：该末状态保留 TD 末端目标，屏蔽不存在的 policy、终局及空间辅助监督，后续步骤不进入 dynamics 或 BatchNorm。Go 真实步骤的来源手数上限裁定标签不因此改变，但人工裁定不会被当作真实终局延伸。

`read_npz_muzero_data` 复用来源有序文件预取，按每文件整 batch 消费并丢弃不足一 batch 的尾部，根特征在预取线程解包。future 目标保持紧凑存储，到 batch 消费时转为训练张量。根、所有动作和 future 目标使用同一次来源对称采样，Hex 只使用恒等/180°，象棋只使用恒等/物理左右镜像；Go 与象棋的历史 dropout 只应用于根，象棋复用普通 reader 的七帧历史遮蔽及可见帧数更新。象棋吸收态动作从完整 2086 联合动作域采样。RNG 由 learner 持有并保存到 checkpoint，预取线程不推进它。普通 AlphaZero reader 明确拒绝 MuZero 文件。

保存完整对局会增加 raw shard 大小，K 步目标会增加 shuffle 与训练数据占用；来源的资源估算仍只覆盖普通根数组，不能据此估算 MuZero 的总内存、磁盘或吞吐。正式 learner 已消费这些数据接口；原生 `GameRunner` 已能生成真实 MuZero 轨迹，外层控制器通过同一 writer/shuffle/learner 路径执行逐轮训练。

## 原生对局与样本提取

`MatchPairer::BotSpec` 可持有普通 evaluator 或 MuZero `PositionEvaluator`，由同一个 `GameRunner` 创建来源 `Search` 并执行对局。规则初始化解析 representation 声明的 history modes；policy-init、让子放置、fork 选点、raw NN statistics、非围棋截断末状态和多步 side fork 的真实观测评估通过 `Search::evaluatePosition` 调用 initial。该调用使用每个 Search 独立的 buffer/RNG，不改动当前搜索树，也不额外消耗来源对局采样随机流。普通 AlphaZero 仍直接调用原 evaluator。

PCR/cheap、reduce visits、surprise、主线 reanalysis、side 搜索、fork/hint 与权重随机取整在来源位置执行；每次真实落子与 side/reanalysis 起点使用 fresh latent tree。主线重分析用实际棋盘重新编码，保留来源 cheap 位置的选择、搜索预算和 outcome-target 开关。单场对局固定模型，MuZero 不接受局内普通 evaluator 切换回调；原生 manager 在局间获取最新 bundle，外层只在 selfplay 完全结束后发布下一轮模型。

树内 side 位置沿用来源的访问阈值、最大提取深度、best-path 条件和排除分支。只在搜索完成后的采样阶段重放真实合法路径，遇到真实终局停止提取。latent 节点的 policy 与 Q 仅保留该真实位置的合法根动作，policy 在量化前重设有效最大值的来源编码尺度；surprise 和 entropy 使用同一动作域下重新归一化的目标与 prior。没有合法策略目标质量的候选不产样，仍可继续检查合法后继。W/D/L/NR 和合法动作 Q 仍来自该 latent 节点的搜索统计，raw NN statistics 则重新编码真实观测。以上是把 latent 节点作为真实 side 根监督时的必要适配，不改变树内动作域、选择、访问统计或回传，也不额外搜索或虚构 side 后续轨迹。

原生 `selfplay` 配置 `useMuZero=true` 后，以模型目录中的 `model.json` 发现完整 bundle，三个组件文件不单独参与最新模型选择。空目录使用 MuZero 随机 evaluator 冷启动：CPU 生成标准高斯策略 logits、标准差 0.2 的四结果 logits，经现有后处理归一化；不分配 latent tensor，根方向及静态动作域仍保留，递归不推进真实棋盘。随机路径关闭 policy optimism，剩余时间估计为零，短期价值误差沿用来源零 raw 输出的变换。发布三组件模型后使用真实 NN，真实模型不支持 `debugSkipNeuralNet`。模型棋类必须与对局一致，模型画布必须等于 `dataBoardLen` 并包含所有配置棋盘。生产者在模型轮询目录外完成导出，再原子发布整个不可变目录；不直接向正在轮询的目录导出。

设备与线程选择、批大小、精度、输入布局、缓存、对称和 NN 尺寸选项共用 `Setup` 的来源配置解析与优先级。MuZero 的 NN buffer 尺寸必须等于 bundle 画布；固定的实际棋盘小于画布时默认允许 padding，显式 `requireMaxBoardSize` 仍生效。输入和计算布局的可用组合遵循来源 CUDA/cuDNN 约束，Transformer 仍要求 NHWC 计算。

`SelfplayModel` 持有一个普通 NNEvaluator，或整个 MuZero batcher 与 position evaluator；后者先释放根缓存中的 latent，再销毁服务。来源 `SelfplayManager` 的 acquire/release、旧模型清理、数据队列、写盘线程与排空顺序保留。MuZero 要求 `switchNetsMidGame=false`，对局持有的旧服务不会被新 bundle 提前销毁。外层 `run` 在轮次边界使用完整 bundle；中断轮按实际提交点继续，边界见 [调度与恢复](controller.md)。

## EtaZero 逐轮控制器

标准配置目录在 `selfplay.cfg` 的 `[agent]` 选择 `algorithm=muzero` 及根／非根搜索，在 `env.cfg` 配置棋类与画布，在 `net.cfg` 的 `[muzero]` 配置三组件、latent 宽度和 `unroll_steps`。转换层生成控制器的 `muzero`／`gumbel_muzero` 算法选择、`MuZeroConfig` 字典与 shuffle 的 K；MuZero 不使用普通 `model_kind`。文件职责见 [配置目录](configuration.md)。graph、SVB 等 latent 路径不支持的组合仍按当前适配边界处理。

独立运行未提供初始权重时，首轮使用空模型目录的 random evaluator，learner 在攒足数据后的首次训练中初始化网络。显式提供初始权重，或通过实验调度器共享初始化权重时，先调用正式 `train.py -initialize-only`。该模式要求新的单设备训练目录，复用来源模型初始化、参数组、optimizer、Lookahead 初值、SWA 初值与完整保存函数，在读取训练数据和执行 forward 之前保存零样本 checkpoint；epoch、样本和 SWA 累积计数均未推进。后续 learner 从这个完整 checkpoint 继续，保留初始化范数基准与 RNG。初始化模式拒绝覆盖已有 checkpoint，不用于恢复训练。外层保留不可覆盖的 `initial/checkpoint.ckpt`，并从其普通权重导出 `models/initial/` 给首轮使用。

每轮只向独立的 `input_models/` 链接本轮完整输入模型；随机冷启动时该目录为空。原生轮询不会读到本轮结束后才发布的模型。selfplay 的完整轨迹进入来源 shuffle，由 `muzero_unroll_steps` 生成展开数据；learner 使用跨轮保留的训练目录，同步控制器按有符号样本额度选定完整文件，每个有训练更新的轮次消费完所选文件并收尾一个 epoch。文件／batch 游标用于轮内中断恢复，下一正常轮次从最新快照重新选择文件，不跨轮携带未完成 epoch。完整额度规则见 [调度与恢复](controller.md#自适应逐轮调度)。按本轮最终 checkpoint 导出新目录，所有组件完成后才原子发布。

`publish_weights=ordinary` 或 `swa` 明确选择下一轮 selfplay 的权重；SWA 的累积、参数和计数遵循来源 learner。learner 本身继续使用完整 checkpoint 的训练权重和 optimizer。继续运行不会重复初始化或重复已完成轮；未完成轮复用已提交的阶段产物，learner 恢复完整 checkpoint 中的训练权重、fast/slow 状态、随机流和数据游标。轮次状态提交后再写观测报告，报告写入失败会警告，已提交模型与完成轮数继续有效。

## 正式 learner

`python/train.py -muzero-config /path/to/network.json` 创建三组件模型；JSON 是 `MuZeroConfig` 的字段，包括棋类、latent 通道数和三个来源网络配置。继续同一训练目录时直接从 checkpoint 推断网络，不必重复提供该参数。数据目录仍使用来源的 `train/`、`val/` 和相邻 snapshot JSON；训练与验证分别读取自己的 K 和权重均值。普通 AlphaZero 与 MuZero snapshot 不能互换。

`muzero_learner.py` 只连接模型、展开 loss 和批次接口。参数角色、SGD/AdamW/Muon、学习率、warmup、参数衰减、Lookahead、SWA 和文件消费继续在来源训练循环中执行。每个根 batch 只推进一次优化器与根样本计数，K 个未来步不增加训练时钟。Lookahead 在 epoch 末恢复 slow 权重并清零同步计数，最后不足 k 步的 fast 增量不进入发布权重；SWA 仍只在同步点按根样本周期累积。同步路径没有独立 subepoch 调度，正常训练轮在完整 epoch 后退出；轮内中断可保存文件／batch 游标以及 Lookahead fast/slow 状态和计数。SWA 参数、采样计数与累积余量随完整 checkpoint 恢复，零训练轮不推进 epoch 或 SWA 时钟。独立 learner 仍支持多 subepoch 和来源文件选择。

三组件可分别配置归一化，因此自适应 weight decay 使用各组件的初始范数与观测范数，梯度裁剪按各组件的来源阈值独立执行；不把 prediction 的 norm 设置或范数比例用于 representation/dynamics。全局 gnorm 上界仅用于报告，没有第二次整体裁剪。来源共有公式与调用频率保留；拆分范数和裁剪域是三网络扩展。既有 `wdnormal` 指标对应 prediction，其他组件的真实衰减保存在 optimizer 参数组中。组件梯度范数及 excess 统计沿用来源 `detensorify_metrics` 的有效性判断，在一次设备传输后处理标量；失效批次不计入对应统计的平均，不补零。

编译作用于三个组件的 forward 和来源 loss，展开的 CPU 有效行控制留在编译图外。AMP 包含网络 forward，来源后处理与 loss 保持 FP32。attention penalty 在每步 forward 后取得，未来步采用与 loss 相同的反向系数；未执行的 dynamics 不产生 penalty。日志的普通准确率、权重和样本数保持根语义，`mu_root_*` 与 `mu_recurrent_*` 区分根和展开监督，`loss` 为未施加反向缩放的总 loss。

完整 checkpoint 保存三组件权重、optimizer、SWA、来源训练状态与文件消费状态，以及 MuZero reader 的训练/验证 RNG、Python/NumPy/Torch RNG、实际 learner CUDA 设备的 RNG 和 FP16 GradScaler。`-muzero-hidden-gradient-scale` 省略时使用已保存值，新模型默认 0.5。恢复仍以来源 checkpoint 保存点为边界，未保存的批次不能恢复；不承诺跨硬件逐位一致；外层恢复范围见 [调度与恢复](controller.md)。来源供导出使用的 `skip_optimizer` checkpoint 不包含 optimizer/runtime，不能当作同目录完整恢复状态。

## 组件导出

`python/export_muzero.py::export_components` 接收已选定权重的 `MuZeroModel`，在新的目录中写出 representation、dynamics、prediction 三个 `.bin` 和最后写入的 `model.json`。命令入口通过 `-checkpoint`、`-pos-len`、`-export-dir` 和 `-model-name` 导出正式 learner checkpoint，`-use-swa` 明确选择 SWA，未保存 SWA 时会报错。导出目录必须不存在，失败的部分目录保留，不可发布；自对弈发布由外层控制器负责。导出使用 CPU/eval 副本，来源 FFN 剪枝和 attention logit 上界检查作用于各组件，不修改训练模型。

三组件与普通 AlphaZero 共用 `export_model_pytorch.py` 的二进制层 writer。来源模型头的预留 G/H 槽分别记录 latent 宽度及组件职责；普通模型仍为零。representation/dynamics 只写来源 trunk 和 1×1 latent 投影，prediction 写适用的 policy/value heads。零主干 prediction 的 trunk 只记录宽度和零块数，不写虚假的输入卷积或末端归一化；描述器仅对 MuZero prediction 接受这一布局。CUDA／Eigen／OpenCL 在此路径直接取出输入中的 latent 特征，不执行卷积；CUDA 保留 FP32／FP16 及 NCHW／NHWC 布局转换。没有 global 输入的组件不写虚假矩阵。原生组件导出当前要求模型版本至少为 15；落子棋使用 18，象棋三个组件均使用 19，分别要求 113/8、C+3/0、C+1/0 个空间／全局输入通道。棋类、组件版本与 latent 宽度必须一致。

`cpp/neuralnet/desc.cpp` 复用来源层描述器解析组件，检查职责、输入和投影尺寸，支持来源权重变换、参数统计与释放。零主干 prediction 不使用 trunk scale8，避免改变直接输入 hidden 的数值。若来源 scale8 优化适用，latent 投影补偿 trunk 的 1/8 尺度，保留 min/max 小跨度保护所依赖的绝对单位。普通棋盘 evaluator 的加载入口拒绝组件文件；CUDA／Eigen／OpenCL 组件执行由下述专用入口接入。

## 原生 CUDA 执行

`cpp/neuralnet/muzero.h` 的 `MuZero::ModelBundle` 一次读取三组件模型，检查职责、latent 宽度、棋类与画布，再由多个 worker 共享只读描述器与权重。`MuZero::Evaluator` 在一个 GPU worker 上执行可变大小的 initial/recurrent 批次；来源 `ComputeContext` 决定精度、布局、attention、融合与权重共享选项，三个来源 `Model` 共用一个 CUDA handle、执行流和 scratch allocator。任意组件含 Transformer 时都落实来源的 NHWC 要求。Eigen／OpenCL 的执行边界见下文；其余后端当前明确拒绝 MuZero 执行。

initial 接收 FP32 NCHW 观测、NC global 和可选 metadata；在输入边界按已解析的输入布局保留 NCHW 或转为 NHWC，来源 backend 再处理计算布局。representation 执行后，只有投影、masked min/max 和 mask 拼接属于新增边界。recurrent 只上传设备状态指针和动作，GPU 上打包 dynamics 输入并执行 prediction，不下载 latent。learner 与 GPU 使用同样的常量/小跨度归一化；非有限的有效 latent 值不会在 min/max 中被忽略。

象棋画布至少为 10，输入 mask 必须恰好覆盖左上角的 9×10 实际棋盘。recurrent 在主机端将联合动作转换为两个格点索引，由同一 packing kernel 写入起终点平面；不增加真实后继状态或 latent 回传。prediction 的完整位移卷积输出按同一个联合动作表读出，保留来源 scale8 恢复和 FP32/FP16、NCHW/NHWC 数值路径。原始接口不做玩家视角转换，由下述 `PositionEvaluator` 处理真实根与 latent 的坐标关系。

`State` 是不可变的设备状态句柄，保留 FP32 NCHW latent 与根的静态 mask，可跨批次、跨展开深度保留，同一父状态可重复展开。每批输出共享设备存储，最后一个引用释放时回收；状态可以晚于 evaluator 销毁，也可以在其他线程释放。状态绑定产生它的 evaluator/model，禁止拿旧模型状态交给另一个 evaluator。evaluator 本身按来源 ComputeHandle 约定在同一 GPU worker 创建、调用和销毁，调用不能重叠。

动作只检查静态棋类动作域与有效画布，不检查真实后继占用、超级劫或终局；Go pass 是画布面积对应的动作。输出是当前 latent 行动方的 policy、W/D/L/NR、variance-time 与短期胜负误差原始 logits，已还原来源 scale8；policy 空间掩码与 Python head 一致。softmax、根合法性、NR 屏蔽与价值视角切换由下述局面适配负责，再交给来源搜索。`readStateForTesting` 是显式诊断下载入口，正常 initial/recurrent 和搜索不调用它。

## 原生 Eigen 执行

`cpp/neuralnet/muzeroeigenbackend.inc` 在原生 Eigen 后端内复用 `Trunk`、卷积、`Model` 的 prediction heads、scratch allocator 和 FP32 NHWC 数值路径。representation／dynamics 只在来源 trunk 后接导出的 1×1 latent 投影；没有 global 输入时跳过不存在的全局线性层，与 CUDA 边界一致。普通网络的层与执行流程保持原有实现。CUDA、Eigen 和 OpenCL 共用 `muzerobundle.inc` 的 bundle 加载和组件检查。

公共 API 仍接收 NCHW 观测；Eigen 适配层转换为来源 NHWC 输入，执行相同的 masked min/max、小跨度分母与固定根动作编码。状态是不可变的 CPU FP32 NCHW 存储，末通道保存静态 mask；跨批保留、重复展开、worker 归属和晚于 evaluator 释放的语义与 CUDA 一致。prediction 读出全部普通／乐观／可选 Q 通道、四结果和辅助 logits，象棋使用全部 2086 个联合动作。Eigen 不支持 FP16 或 NCHW 计算配置，显式请求会沿来源边界报错；没有设备 graph 编译，warmup 与普通 Eigen 一样不执行额外前向。

CPU 与 CUDA 共用下述 Batcher、PositionEvaluator、Search、GameRunner 和逐轮控制器。构建与检查命令见 [README](../README.md#构建与入口)。`tests/test_muzero_eigen.py` 使用独立 PyTorch CPU 参考验证六棋类 initial／多步 recurrent、常量／小跨度状态、metadata、NBT／Transformer、镜像与历史、NR 掩码、缓存与静态动作域；并覆盖两个 CPU worker、24 个客户端、归属拒绝、停机排空和异常唤醒。搜索检查包含单／四线程 PUCT、Gumbel 根及 Gumbel 非根。Go MuZero／象棋 Gumbel MuZero 的两轮 CPU 训练检查覆盖初始化、产样、展开训练、普通／SWA 发布及完整轮次继续运行，不构成棋力或性能结论。

`testgpuerror` 可用 Eigen 导出 initial 及各 recurrent 深度的原生参考，再由 CUDA 读取，检查串行／并发 FP32 与当前精度。参考按确定性静态动作序列重建各后端自己的 latent；文件命名、轨迹匹配与使用边界见 [推理精度与批处理误差](benchmark.md#推理精度与批处理误差)。

## 原生 OpenCL 执行

`cpp/neuralnet/muzeroopenclbackend.inc` 复用来源 `Trunk`、卷积、prediction `Model`、scratch allocator、tuner 与精度选择。输入计算布局为 NCHW，保留 OpenCL 空间对齐填充；FP32、FP16 存储及来源设备支持的 tensor-core 路径使用同一适配入口。representation／dynamics 的导出投影仍由原生卷积执行，来源 scale8 在投影边界补偿，归一化保持 Python 的 masked min/max 与小跨度分母。

状态保存在设备 `cl_mem` 中，使用紧凑 FP32 NCHW 和末通道静态 mask。OpenCL 1.2 不能传递跨 buffer 的设备指针数组，recurrent 在设备上收集父状态，再由 kernel 写入 mask、latent 和一／两张动作平面；不会把 latent 下载到 CPU 推理。不同批次的父状态、重复展开、worker 归属与晚于 evaluator 释放的生命周期保留。prediction 使用原生 heads，读出全部策略通道、象棋 2086 动作、四结果与辅助 logits，来源 scale8 只还原一次。原生 `PROFILE_KERNELS` 收集与释放复用于每个 initial／recurrent 请求，新增打包／归一化 kernel 进入同一 profiling 输出。

来源遮罩核作一项有实际故障依据的修正：Winograd 未写入的对齐填充可能含 NaN，原有乘零无法清除它。遮罩为零的位置直接写零，保证后续来源池化所要求的填充区为零；有效棋盘上的计算公式保持原样。该修正同样适用于普通 OpenCL 网络。

`tests/test_muzero_opencl.py` 在三种精度模式下检查六棋类 initial／多步 recurrent、常量／小跨度状态、metadata、NBT／Transformer、跨批状态与生命周期；同一 Batcher、PositionEvaluator 和 Search 检查覆盖并发、停机排空、错误唤醒、历史／对称／缓存、单／四线程 PUCT 和 Gumbel。原生 `testgpuerror` 另用 Eigen 比较每个 recurrent 深度。Go MuZero 与象棋 Gumbel MuZero 的两轮检查使用 OpenCL 自对弈和单卡 CUDA learner，核对普通／SWA 发布及下一轮恢复。当前设备验证使用 NVIDIA；数值和流程检查不构成其他设备、吞吐或棋力结论。

## 跨线程组批

`MuZero::Batcher` 接收单行张量请求，调用线程等待结果。initial 使用共享队列；recurrent 按设备状态的所属 evaluator 分流，即使两个 worker 位于同一 GPU，也不混用状态。每个 worker 在 initial 与自身 recurrent 队列中选择最早请求的种类，立即取走当前可用的至多一个批次，不等待凑满、不增加定时窗口或队列容量限制。initial 与 recurrent 的计算图不同，因此不能混在一个 forward 中；这是相对来源单一观测队列的必要适配。

请求缓冲在调用线程复用，完成后释放输入引用与状态。组批上限可在运行时调小，仅作用于后续批次。行数与批次数在硬件推理成功完成时累计，warmup 不计入；其他 worker 同时失败时，已经计算完成但随后返回全局错误的批次仍可计入，统计不表示成功调用数。

CUDA 含 Transformer 时，各 worker 在接受请求前对 `1..maxBatch` 的 initial/recurrent 做来源 graph warmup。输入使用全有效 mask 与零特征/global/metadata，只用于编译图形状，不采样真实局面或消耗采样随机数；warmup 的 lenient graph 标志沿用来源后端，并在完成或失败时恢复。

`finish()` 由一个所有者调用，停止接收新请求，排空已接受请求并等待 worker 退出；调用方可与它并发提交，但必须在销毁服务前结束所有客户端线程。服务不原地重启或换模。worker 异常会使服务永久失败，唤醒待处理请求；其他正在执行的 worker 先完成或同步自身批次，再释放调用方。后续调用持续报告首次错误，没有重试或推理回退。来源 worker 的未处理异常会终止进程，此处显式传播给等待者，以完成 MuZero 状态与输入的生命周期管理。

该服务只处理原始张量和设备状态；根特征、对称、棋盘哈希缓存、合法性与 logits 后处理由独立的局面适配负责，再由来源 `Search` 消费。原生 `GameRunner` 与 selfplay 命令通过上述完整模型句柄消费该服务。

## 局面、朝向与根缓存

`MuZero::PositionEvaluator` 在真实根接收 `Board`、`BoardHistory`、执色、metadata 和来源 `MiscNNInputParams`。它检查模型棋类与画布，调用来源版本化特征函数和 metadata encoder，并按真实棋规生成根合法动作。每个调用线程复用一个 `PositionBuffer`；公共 `NNInputs::fillRow` 只分派到原有特征函数，普通 `NNEvaluator` 使用同一入口。

Go 与落子棋根输入按整个 NN 画布做对称，因此小棋盘和矩形棋盘的 mask 也一起移动。未指定朝向时采用配置默认值或均匀随机选择，Hex 仅允许恒等与 180°。象棋特征先按根行动方编码，只允许物理九路棋盘的左右镜像，padding 保持在原位置。随机朝向只在缓存未命中后采样。selfplay 使用配置 `nnRandSeed` 与调用方随机流派生每个 position buffer 的独立随机流；没有指定 seed 的直接 API 沿用调用方 `Rand`。来源普通 evaluator 在 worker 内采样，两者保持允许集合及均匀分布，不要求线程随机流逐位相同。根多对称平均仍使用来源搜索调用方的采样流。递归状态始终保留根朝向，动作从绝对棋盘坐标映射到 latent 朝向，policy 再逆变换回来；不逐步旋转 latent。象棋另外保留根执色，后续每步均使用这个执色对应的旋转，而不是当前 latent 执色。

`Evaluation` 保留设备状态、执色、根朝向/尺寸及 NR 是否允许，不保存真实后继棋盘或历史。递归使用静态棋盘点及 Go pass；象棋使用完整 2086 个联合动作。已占用点、重复动作、连续 pass 与象棋中按真实后继棋规非法的动作仍是 learned transition，不据此裁定真实终局。根合法性和真实终局仍由实际规则状态决定；象棋的完整长将／重复历史保留在根缓存身份中。递归输入参数只读取策略温度与乐观值，PDA、历史和 metadata 等观测信息已编码在根状态中。

输出沿用 `NNOutput` 的白方视角与非法策略值 `-1`。普通 evaluator 和 MuZero 共用 `NNPostprocess` 的策略 softmax、温度、四结果 softmax、variance-time 和短期误差转换；乐观策略先按来源公式混合 logits。representation 需要 metadata 时关闭 policy optimism。NR 按根棋类及 Go 的 ko/scoring 条件屏蔽，保留来源判断；W/L 随当前 latent 执色转换，D/NR 不翻转。原始 MuZero 后端输出已经还原 scale8，因此这里的输出尺度设为 1，不重复乘 8。

根缓存直接使用来源 `NNCacheTable` 与 `NNInputs::getHash`，包含来源温度、PDA、乐观值、历史可见性及 metadata 身份；不另造完整输入哈希或 recurrent 棋盘缓存。每项同时持有预测、设备 latent 与朝向，命中时沿用缓存朝向，不能把它解释为本次请求的另一朝向。`skipCache` 与来源一致，只跳过读取，仍写入新结果；清理或替换项时在锁外释放状态。缓存容量也决定被保留的 latent 显存，不能仅按普通 NNOutput 的主机内存估算。

根多对称按来源方式无放回采样、跳过缓存读取并平均后处理结果。第一个均匀抽取的对称提供后续递归使用的状态；其余预测参与根平均，不平均非线性的 latent，也不把根多对称扩展为每层 ensemble。平均结果使用独立对象，不修改已经写入缓存的单对称输出。

## 来源搜索中的 latent 路径

`Search` 可接收 `MuZero::PositionEvaluator&`，沿用来源节点、并发展开、PUCT/FPU、虚拟损失、根策略处理、访问预算、选点和统计回传。`useGumbel` 与 `useGumbelNonroot` 仍使用同一 Gumbel 调度器和选择函数，不维护另一套 MuZero PUCT 或 Gumbel 数学。evaluator 和其 batcher 必须晚于 search 及客户端线程销毁。

来源的边访问补记也保留：即使关闭 graph，多线程仍可能在另一个线程已更新子节点、尚未更新父边时补记访问，因此访问数不必等于实际展开节点数。关闭权重修正时，每条边按其访问次数加权子节点均值，当前节点自身 NN 的权重为 1；单线程树才退化为各展开节点的等权平均。

根 initial 读取真实局面；非根 recurrent 读取父节点的设备状态与所选动作。线程只推进 latent 深度和执色，保留真实根棋盘用于坐标尺寸，不执行后继 `BoardHistory::makeBoardMoveAssumeLegal`。深度相关 avoid 规则使用 latent 深度；非根合法性来自静态策略域。真实根终局仍使用棋规结果，Go 显式分析已结束根的来源行为保留；非根连续 pass、重复落点和超出真实对局长度的路径不触发真实终局。

`MuZero::Evaluation` 作为 `NNOutput` 的派生对象保存预测与设备状态，使用来源节点的同一次原子发布。并发重复评估只有获胜结果被采用；不会分别发布 logits 和 latent。根噪声/温度复制输出时保留派生状态，来源延迟回收机制继续保护搜索线程读取的旧对象。

每次开始搜索使用 fresh tree，实际 `makeMove` 先清掉旧 latent 树，再按真实棋规推进环境。graph、SVB、重复模式 bonus 明确拒绝，不能把根棋盘冒充 latent 后继供这些机制使用。这些静态冲突及普通次级网络的树内探索冲突，在创建搜索对象或更新参数时同步报错；非法更新不修改现有参数或清树，不等到异步搜索线程启动后才拒绝。根预测缓存继续独立工作。若来源的罕见缓存/合法性冲突触发活动树的根重算，MuZero 明确报错，要求清理根缓存并重新开始搜索；不在已有后代仍被线程使用时更换根 latent。普通棋盘搜索仍沿用原有局部重算路径。

策略目标、选中动作、访问统计、PV 和 JSON 报告沿用来源搜索接口。latent PV 是模型预测动作序列，可能包含真实棋规不允许的重复落点；JSON 与 GTP 不用真实 Go encore 阶段重放并截断它。原生 selfplay、JSON analysis 和全部棋类的 GTP 已连接这一语义；不能把 latent PV 当作可直接重放的真实棋谱。

`visitCapContempt` 使用来源节点统计和子节点权重快照，不需要真实后继棋盘；PUCT 保留根／非根路径，Gumbel 支持非根快照。受限方及 ponder 沿用来源语义，根快照与 SH 配额的冲突处理见 [Gumbel 搜索](gumbel.md#visit-cap-对手建模)。

### 次级 HumanSL

MuZero 主模型的 `Search`／`AsyncBot` 支持两种次级网络。普通 `NNEvaluator` 只读取真实根局面，可用于根探索和最终选点混合；配置己方或对方的树内探索时明确拒绝，因为普通网络没有 latent 后继可读取。次级 `MuZero::PositionEvaluator` 则支持根与树内探索：每个节点沿用来源 `humanOutput` 保存该模型自己的预测、latent 和朝向，递归从它自己的父状态推进相同的绝对动作。两个模型须使用相同棋类及画布，latent 通道数和权重可以不同，不共享状态，也不要求根朝向相同。

普通主模型也可搭配 MuZero 次级网络。普通搜索每个节点都有真实局面，因此次级网络对每个节点使用 representation／prediction；不在真实搜索树中额外维护一条 learned transition。来源的 graph、树复用和 SVB 继续作用于普通主模型的真实搜索。

两种路径均使用来源 HumanSL PUCT 系数、探索概率、weightful／weightless 回传边界、PIKL 与最终选点混合。主网络继续提供节点价值；次级策略不替换独立计算的 Gumbel 训练目标。根多对称平均分别由各模型执行，后续递归保留各自采样的朝向。次级 representation 需要 metadata 时，未设置 profile 就不启用次级输出，沿用来源行为；主 representation 必需的 metadata 仍须提供。Gumbel 根 weightful 探索与严格配额的冲突仍明确拒绝，其他规则见 [Gumbel 搜索](gumbel.md#次级-humansl)。

原生 JSON analysis 和 GTP 通过 `-human-model` 加载次级网络：普通模型使用文件路径，MuZero 使用含 `model.json` 的三组件目录，沿用原生 `humanSL…` 配置字段。`useMuZero` 只选择主模型类型。双方使用相同画布；命令在配置、规则和请求边界检查不支持的组合，避免异步 worker 因模型不匹配而退出。模型查询、统计、缓存清理和关闭流程共用模型服务；原生次级 evaluator 的推理配置索引约定不变。

## 原生 JSON analysis

`katago analysis -config /path/to/analysis.cfg -model /path/to/bundle` 直接加载包含 `model.json` 和三个组件的目录。配置使用 `useMuZero=true`，`game` 必须与 bundle 一致；`useGraphSearch=false`、`subtreeValueBiasFactor=0`、`avoidRepeatedPatternUtility=0`，不加载真实棋盘外部模式表。次级 HumanSL 使用上述 `-human-model` 入口。Gumbel 使用同一套 `rootSearchAlgorithm`／`nonrootSearchAlgorithm` 字段。其余搜索、并行、缓存、设备和日志设置由来源解析器处理；指定 NN buffer 尺寸时须与 bundle 画布一致。原生 `requireMaxBoardSize=true` 会要求请求尺寸恰好等于画布。

请求与普通网络使用相同的 JSON 协议；象棋字段见 [象棋 analysis](xiangqi.md#原生-json-analysis)。真实根保留完整历史和棋规合法性，非根只推进 latent 状态。`allowMoves`／`avoidMoves` 的深度对应 latent 搜索深度。来源请求队列、并行 worker、逐步报告、动态 focus、取消和关机流程直接复用；PUCT 与 Gumbel 均支持来源 weightless root focus，Gumbel 的正常访问仍遵守严格配额。棋类、尺寸、必需 metadata 和不支持的搜索组合在请求入队前报错，错误不会占住异步 worker。

主 representation 含 metadata encoder 时，通过原生 `humanSLProfile` 或请求的 `overrideSettings` 提供 profile，缺失时拒绝分析。`query_models` 保留来源字段：内部名称和版本来自 prediction，`usesHumanSLProfile` 来自 representation，`usingFP16` 仍报告配置模式。`clear_cache` 清理根预测缓存。原生 `NN rows`／`NN batches`／`NN avg batch size` 由共享推理服务提供，其中 rows 包含 initial 与 recurrent 推理，不另造绘图计数。关机先释放搜索树和缓存中的 latent，再关闭推理服务，最后执行后端清理。

## 原生 GTP

Go、Freestyle／Standard／Renju、Hex 和象棋使用 `katago gtp -config /path/to/gtp.cfg -model /path/to/bundle`。在原生 GTP 配置中设置 `useMuZero=true`，`rules` 与 bundle 棋类一致；若同时指定 `game`，它也须一致。latent 搜索的 graph、SVB、重复模式与次级 HumanSL 限制与 analysis 相同。其余参数沿用原生字段；省略 `nnMaxBatchSize` 时，使用来源 GTP 按搜索并发推导 batch 上限的逻辑。

主模型或次级模型为 MuZero 时，画布由 bundle 固定；在画布内改变实际棋盘只重建 bot，不重新加载模型。Go 可使用矩形棋盘，落子棋仍要求方形，象棋固定为 9×10；`requireMaxBoardSize` 可要求实际棋盘与画布完全一致。尺寸错误在资源切换前拒绝，保留原局面。仅使用普通网络时沿用来源按尺寸加载 evaluator 的路径。`kata-raw-human-nn` 读取次级网络的真实根预测，metadata／有效对称／终局检查与模型相符。

`play`、`genmove`、`undo`、`loadsgf`、`printsgf` 和规则切换处理真实历史；搜索使用 fresh latent tree。`kata-raw-nn` 评估真实局面，`debug_moves ... rawstats` 的 MuZero 分支使用 initial／recurrent，后续不按真实棋规推进。Kata／LZ 分析的 PV、PV visits 和 edge visits 保留完整 latent 路径。来源棋钟、ponder、同步／可取消搜索和 stop 流程保持原调用关系；非法 Gumbel／MuZero 参数组合在启动搜索前拒绝，失败的运行时参数更新不会改变有效配置。

主 representation 使用 metadata 时，可以通过 `kata-set-param humanSLProfile ...` 设置 profile；未设置时拒绝需要主网络的命令。开启 ponder 前也必须有 profile。原生模型查询、缓存清理和逐次 genmove 的统计清零均连接共享服务；零推理批次的平均 batch size 与来源一样为 NaN。Go 的固定／自由让子及 `kata-benchmark` 保留来源位置与统计流程；内置 benchmark 按棋类选用 Go 来源棋谱或其他棋类的合法轨迹；独立 benchmark 也可指定 SGF，使用同一 MuZero 服务和来源计时流程，见 [搜索 benchmark](benchmark.md)。非围棋不使用围棋让子启发式。象棋的完整规则状态、联合动作、FEN／SGF 及终局边界见 [象棋 GTP](xiangqi.md#原生-gtp)。

## 验证与待接入边界

`tests/test_muzero_model.py` 检查来源 trunk 的输出与梯度、mask 与小跨度归一化、重复/pass 动作、side 行在 BN 前移除、独立参数分组，以及展开的 hidden/逐步 loss 梯度系数。真实 CUDA 检查通过 `ETAZERO_GPU_TESTS=1` 启用，覆盖 FP16 更新、GradScaler 溢出跳步，以及组件 checkpoint 的模型/优化器/scaler 恢复和下一次有效更新。组件检查使用构造的合法目标，不能证明轨迹生成正确或 MuZero 已具备棋力。

`tests/test_muzero_export.py` 覆盖卷积/NBT/Transformer 和 Go/落子棋的组件导出、原生解析、投影权重顺序与尺度、异常格式拒绝，以及训练权重不被导出修改。

`tests/test_muzero_cuda.py` 在真实设备上比较 initial 和多次 recurrent 与 PyTorch，覆盖 FP32/FP16、NCHW/NHWC、混合组件结构、metadata、常量/小跨度状态、重排与跨批保留。来源默认 TF32 路径和禁用 TF32 的严格 FP32 对照分别验证，测试精度设置不修改生产后端。

`tests/test_muzero_batcher.py` 使用同一实际 GPU 上的两个 worker 和 24 个客户端，对照 PyTorch 的根与四次递归输出，覆盖 FP32/FP16、metadata、NBT、Transformer warmup、动态批上限、状态归属与释放、停机排空、启动失败及运行异常唤醒。这验证了同设备多 worker，没有验证跨物理 GPU 或端到端搜索吞吐。

`tests/test_muzero_position.py` 在真实 CUDA FP32/FP16 上用独立 PyTorch 对称与 softmax 对照根和多步递归，覆盖小棋盘、矩形 Go 棋盘、全部有效对称、双方价值视角、NR 条件、metadata、温度/乐观策略、根占用与递归静态域。原生检查同时覆盖根缓存身份、命中朝向、skipCache 写入、多对称平均、随机/默认朝向和缓存状态释放。

`tests/test_muzero_search.py` 使用真实 CUDA FP32/FP16，在 Go/Renju/Hex 上检查 PUCT、Gumbel 根及可选 Gumbel 非根，覆盖 1/4 搜索线程、根预算、预测与父 latent 的对应、关闭权重修正时的访问加权回传、重复/pass 链、PV/JSON、真实落子清树和真实根终局。单线程链达到 31 层；多线程检查访问守恒及实际展开深度，并用固定补记交错和手算加权值检查回传。另检查来源根噪声、温度、多对称、hint/focus、uncertainty 与 noise pruning 的组合，以及禁止的真实棋盘机制和根重算冲突。

`tests/test_muzero_data.py` 通过原生 writer 比较普通起点与开启轨迹记录后的逐字段一致性，覆盖未入选主线、跨 shard 的完整轨迹、side/reanalysis、终局/截断、四结果视角及独立几何级数 TD 参考。测试还覆盖来源 shuffle 的单/多 wave、keep 与 Q 选择、reader 的文件顺序和尾部处理、RNG 恢复、固定朝向增强，以及原生数据经过 shuffle/reader 后的三组件展开更新；CUDA 模式使用真实 FP16 AMP。测试使用可检查的规则对局与构造的搜索统计，不能证明正式 MuZero selfplay 或棋力。

其余原生工具和外围功能的接入范围见 [架构](architecture.md)。外层已按阶段实际提交点恢复中断轮次，恢复范围和在途工作限制见 [调度与恢复](controller.md)。

`tests/test_muzero_humansl.py` 在真实 CUDA FP32／FP16 上检查 Go、Renju、Hex、象棋的双网络搜索，覆盖不同 latent 宽度和权重、不同根朝向、metadata、根多对称、单／多线程、连续落子与异步调用。latent 搜索逐节点从次级模型自己的父状态重新推理作对照；普通搜索则重放真实棋规，对照每个节点的 initial 预测。另核对探索计数、最终策略混合、Gumbel 训练目标独立性，以及 MuZero 主模型搭配普通次级网络的根节点限制。验证的是执行与数值语义，不代表已经得到人类棋风或棋力效果。

`tests/test_humansl_protocols.py` 在六种棋类上覆盖 JSON／GTP 的普通和 MuZero 主次模型组合，并以次级模型单独加载的输出对照 `humanPolicy` 与 `kata-raw-human-nn`；另检查 profile 激活、非法配置后的继续请求、尺寸切换、模型查询与释放，以及双网络下的动态 focus、取消、ponder 和棋钟。

`tests/test_muzero_analysis.py` 使用真实 CUDA FP32／FP16，覆盖六种棋类、PUCT／Gumbel、Transformer、完整象棋历史和终局、真实根与重复动作 latent PV、metadata、模型查询、并发请求、动态 focus、排队／运行中取消、缓存清理和快速关机。错误请求后继续提交合法请求，检查 worker 保持可用；普通网络协议与共享批量服务另有回归验证。

`tests/test_gtp.py` 检查普通网络的配置、摆子、悔棋、错误后继续处理和来源 benchmark。`tests/test_muzero_gtp.py` 在实际 CUDA FP32／FP16 上覆盖 Go 和四种落子棋、PUCT／Gumbel、Transformer、metadata、实际棋盘尺寸切换、SGF、悔棋、重复落点／pass 的 latent PV、raw-NN 与 JSON 的同局面数值对照、统计清零、Go 让子／benchmark、棋类对应的认输条件、棋钟、ponder 及可取消搜索。

纯推理 `benchmarknn` 直接使用同一 MuZero Evaluator，按来源循环分别测量 initial 与 recurrent，保留设备 latent；输入准备、预热和模型销毁不计时。API 内的 mask、传输、分配和输出整理计入耗时，不能与普通网络前向视为相同计算单元，详见 [纯推理 benchmark](benchmark.md#纯推理-benchmarknn)。`tests/test_benchmarknn.py` 覆盖六种棋类、FP32／FP16、metadata、Transformer、多个工作线程及独立 operation。

`tests/test_muzero_learner.py` 检查正式 learner 的根样本时钟、独立组件范数、side attention 边界、训练与验证 RNG，以及真实训练、保存、重启更新和普通/SWA 导出。GPU 专项覆盖 FP16、编译和 Muon，并将 checkpoint 导出的三组件交给原生 CUDA initial/recurrent 与所选权重比较。这些是实现检查，不是棋力或完整自对弈闭环证据。

`tests/test_muzero_play.py` 使用 Go/Renju/Hex/象棋的真实 CUDA 对局，覆盖来源 PUCT、Gumbel 根与两种非根选择、policy-init/Go 让子、cheap 后重分析、side/fork、树内 side 的真实合法动作与零/单动作熵边界，以及原生轨迹经来源 shuffle 后的正式 FP16 learner 更新。另有原生 selfplay 命令检查覆盖 bundle 发现、配置优先级、实际产样和画布不匹配拒绝；外层检查见下述控制器验证。

`tests/test_muzero_random.py` 在隐藏 GPU 且没有模型文件的条件下，检查六棋类的随机根／递归预测、缓存、静态动作域及 PUCT/Gumbel 轨迹产样。

`tests/test_muzero_controller.py` 运行 Go/象棋 MuZero、Renju/Hex/象棋 Gumbel MuZero 的实际两轮 CUDA FP16 闭环，覆盖 PUCT/Gumbel 非根、编译训练、随机冷启动、显式权重初始化、不可变输入 bundle、普通/SWA 发布、展开快照与完整轮次继续运行；发布的三组件逐字比较于独立选择 checkpoint 权重后的导出。工程检查不表示棋力或吞吐已经得到验证。

象棋 GTP 的完整规则历史、联合策略和 latent 分支由 `tests/test_xiangqi_gtp.py` 检查，包含与 JSON analysis 的同局面数值对照及原生异步控制。
