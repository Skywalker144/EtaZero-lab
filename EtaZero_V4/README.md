# EtaZero V4

V4 以固定 KataGo `d91ea855110dae533f0aada947b2b7d78cc8a4e1` 的完整源码为底座，保留来源模块和调用关系，在实际职责边界接入 EtaZero。来源信息见 [reference_sources.json](reference_sources.json)，来源说明与许可证见 [KATAGO_README.md](KATAGO_README.md) 和 [LICENSE](LICENSE)。

目标定位是 **基于 KataGo 合理接入多棋类、Gumbel 与 MuZero 的“KataZero”，配合 EtaZero 外层调度与工具**。引擎层支持 AlphaZero、Gumbel AlphaZero、MuZero、Gumbel MuZero 及多棋类；外层采用 EtaZero 的同步逐轮调度，并提供实验管理、评估、绘图和 Web 等功能。共有搜索、网络、采样和 learner 机制保留 KataGo 实现，必要扩展在来源模块内接入，不另造一套引擎。[目标与适配边界](docs/katago-target.md) 是可修正的设计记录，具体接入以用户目标、算法语义和来源实现为依据。以下说明当前已实现范围。

当前已贯通的外层闭环是 **采用 W/D/L/NR 的 AlphaZero、Gumbel AlphaZero、MuZero、Gumbel MuZero 训练链及 EtaZero 逐轮控制器，支持围棋、Freestyle/Standard/Renju 五子棋、Hex 和中国象棋**：原生 selfplay → 来源 shuffle → 来源 learner → 来源模型导出 → 直接接受或来源 gatekeeper。每轮固定一个输入模型，阶段顺序执行，训练目录跨轮保留，下一轮使用已接受的模型。原生 GTP、JSON analysis、比赛和测试命令由来源引擎直接提供。

控制器通过 `publish_weights` 选择 checkpoint 的普通权重（`ordinary`，默认）或来源 SWA 权重（`swa`）供下一轮 selfplay 使用。learner 始终续用完整 checkpoint 中的训练权重、optimizer 和来源内部时钟。

底层 `Rules`／`BoardHistory` 支持 Freestyle、Standard、Renju 和 Hex，复用来源 PUCT、图搜索和树复用。Renju 禁手是落子判负；Hex 使用六邻接和保持双方目标边的对称。规则、历史、图身份及搜索的边界检查见 `cpp/tests/testplacement.cpp`，搜索检查使用来源随机 evaluator。

四结果贯通终局标签、搜索加权统计与缓存、surprise、主价值及三组 TD 目标、loss、导出和原生推理。D 与 NR 分别监督；D 的效用固定为零，NR 通过 `noResultUtilityForWhite` 加权，并可在 selfplay 中按 `noResultStdev` 逐局随机化。搜索白方效用为 `winLossUtilityFactor × (W−L) + noResultUtilityForWhite × NR`。网络输出顺序为 W/D/L/NR，旧三结果模型不能加载。CUDA FP32/FP16 和真实两轮训练闭环有专项检查；普通网络的 Eigen CPU FP32、OpenCL FP32/FP16、ONNX Runtime CPU FP32 已通过六种棋类的数值与搜索检查，OpenCL 验证使用 NVIDIA 设备。其余来源后端尚未在对应设备验证。

搜索不包含 score utility、根收官分数奖励或 anti-mirror；配置这些旧参数会明确报错。PUCT/FPU、虚拟损失、LCB 与 uncertainty 的效用只使用结果价值，保留来源的共有公式；不再用预测目差或分数误差修正搜索效用及权重。

passing/dame 增强不再修改策略、搜索预算、合法着法候选或 GTP 出手：没有 pass 先验上限、强制额外访问、填单官、清除死子后改手或结束前的历史隐藏。`friendlyPassOk` 不属于规则裁定，已从规则描述及缓存身份中移除。围棋 pass、超级劫、计分和日本规则 encore 仍按规则执行。

自对弈和比赛使用固定规则贴目：`komiMean` 是明确的整数或半整数，新局面与 SGF/hint 采样起点使用该值，fork 保留原局面的贴目。随机贴目、自动平衡、让子补偿、policy init 后补偿和比赛动态贴目均已移除；相关旧配置及工具开关会报错。让子落子、policy init、fork、非对称搜索预算和围棋规则计分仍保留。

ownership、seki 和逐点计分的训练 head、目标与 loss 已移除，专用 seki fork 和归属预测的推理、缓存补算、搜索汇总及协议展示也已删除。空间目标只保留来源的 8/32 手未来局面监督；Go 的时距、截断、视角、权重与归约不变，非围棋截断按下述可用时距掩码处理；含旧辅助通道的数据不能用于训练。

score 分布、均值、方差、lead、TD-score、Q-score 和分数误差的 heads、目标、loss、推理及协议字段已移除。optimistic policy 和 Q 监督保留来源的胜负分量与系数；棋谱生成、hint、fork 和认输不再依赖预测目差。实际终局计分仍由棋规完成。

GTP `final_score` 只返回已完成且非认输对局的规则结果，未终局不会估分。`final_status_list` 要求实际终局；JSON/GTP 的 ownership 请求会报错。计分 SGF 导入要求棋谱实际到达规则终局，不再通过网络归属预测和额外清场补全记录。

五子棋与 Hex 使用模型版本 18：5 个空间特征、8 个全局特征，策略只包含棋盘点，不包含 pass head；主价值和 TD 的 NR 被屏蔽。特征、原生 writer、Python 数据增强与 loss、模型导出和 CUDA 推理已经接通并有专项检查。Hex 的同色对称限制为恒等与 180°，包括训练、随机推理、根多对称平均和 GTP 分析。模型版本 18 的 Eigen FP32、OpenCL FP32/FP16 与 ONNX Runtime CPU FP32 路径已有真实数值、规则与对称检查；Metal/CoreML、TensorRT 尚未接入，会明确拒绝。

原生 selfplay 按 `game` 创建规则，五子棋/Hex 不使用围棋让子或数目收尾。非围棋真实终局产生规则结果；手数截断时以末状态 NN 价值衔接 TD，关闭主终局标签和结果辅助监督，并分别屏蔽不可用的 8/32 手未来局面目标。SGF 输出使用实际棋类和胜负结果。

五子棋／Hex 的 [平衡开局](docs/openings.md) 已作为棋类适配接入原生 GameRunner，供 selfplay、match、gatekeeper 共用，普通网络与 MuZero 使用同一推理入口。五子棋采用固定 KataGomo 的随机骨架、候选接受、默认启用概率及带既有手数补偿的 policy init，随机 evaluator 冷启动同样执行；Hex 仍显式启用。开局前缀不生成搜索训练目标。Web/GTP 通过可取消的原生开局建议命令接入，生成失败或取消时保留原棋局。原生 `sampleinitializations` 可检查六种棋类、四种算法初始化后的局面与四结果搜索值，`genposesfromselfplayinit` 使用来源 writer 导出包含棋类有效对称及象棋完整历史的初始化局面；`evalrandominits` 单独检查均匀长度的随机 policy 前缀及后续搜索，用法见该文档。

五子棋/Hex 的 SGF 和 hint 起点已接入原生 selfplay。SGF 使用 `GM` 与 `RU` 标识棋类，五子棋须明确 `freestyle`、`standard` 或 `renju`；hint JSON 使用 `game`，省略表示来源 Go 格式。导入保留零气子，Renju/Hex 不自动换色，棋类不匹配会报错或按来源文件加载流程记录并跳过。终局起点不参与自对弈起点采样；来源的 hint fork 保留局面、原始手数和训练权重。`samplesgfs` 的普通采样及 value-fluctuation 加权路径均支持六棋类，后者使用普通／MuZero 原始推理并保留来源权重公式，见 [SGF 波动加权采样](docs/evalsgf.md#sgf-波动加权采样)；`viewstartposes` 与 `trystartposes` 已接入六棋类、普通／MuZero 和 PUCT／Gumbel 搜索，保留各自的筛选、加权抽样与 hint 处理，见 [起点文件诊断](docs/evalsgf.md#查看与搜索起点文件)。其余来源高级工具仍须按实际使用路径核对，不因共享了解析器而视为全部完成多棋类适配。

Gumbel 根在来源搜索树上使用独立配额、sequential halving 和 completed Q，训练策略覆盖未展开合法动作。默认 SH 出手与训练策略分别计算；非根可使用来源 PUCT 或 Gumbel 确定性分配。来源根对称剪枝、代表动作上的 focus 和对称分析报告已接入；训练 writer 仍按来源要求关闭剪枝。按约定使用 fresh tree、关闭 graph，不要求同根续搜或跨手树复用；计数冲突的原因、影响范围及其他组合限制见 [Gumbel 搜索](docs/gumbel.md)。Go/Renju/Hex 的原生自对弈和 Renju 的真实两轮 CUDA FP16 闭环有专项检查。

MuZero 的三组件网络、展开 loss、组件导出、原生描述器、CUDA／Eigen／OpenCL initial/recurrent 执行、跨线程组批、局面适配及原生 PUCT/Gumbel latent 树已接通，直接复用来源 `Model`、`Metrics`、二进制层 writer 和 `Search`，见 [MuZero 网络与展开训练](docs/muzero.md)。根按真实棋规过滤动作，递归只使用静态动作域；预测与 latent 一起发布，保留固定朝向，共用四结果后处理、选点和回传。完整轨迹 writer、来源 shuffle 与展开 reader 也已接入，起点采样仍使用来源逻辑，未来目标保留未入选的主线位置，side 不伪造后续，真实终局与截断分别处理。正式 learner 已连接展开 reader/loss，并保存三组件、optimizer、SWA 和 RNG/scaler 状态；支持 checkpoint 的普通/SWA 三组件导出。原生 `GameRunner` 已接入真实自对弈、重分析、side/fork 与轨迹写入；selfplay 命令已接入三组件 bundle 的发现、加载和生命周期管理；外层控制器已接通零样本初始化、逐轮 selfplay/shuffle/learner、普通/SWA 整包发布和阶段恢复。原生 JSON analysis 与 GTP 已接入全部棋类，GTP 共用三组件模型、raw-NN、棋钟、ponder 和可取消搜索；象棋协议保留完整规则历史并使用联合动作。

外层 `game` 选择棋类，`algorithm` 接受 `alphazero`、`gumbel_alphazero`、`muzero` 和 `gumbel_muzero`，learner 自动选择匹配的输入/动作布局。Renju、Hex 已有真实两轮 CUDA 训练检查，Hex 还覆盖来源 `torch.compile` 与 FP16 AMP。MuZero/Gumbel MuZero 的 Go、Renju、Hex、象棋两轮 CUDA 检查覆盖初始模型、普通/SWA 发布和下一轮消费，其中 Hex 覆盖编译训练。**中国象棋的四种算法已贯通原生 selfplay、SGF/hint 起点、writer、shuffle、learner、普通／SWA 发布与逐轮调度，并通过真实 CUDA FP16 两轮检查；原生普通／MuZero 混合比赛、来源 Elo 与 gatekeeper 已接入；局部模式与 SVB 已接入；普通 PUCT／Gumbel Search 支持无 Go metadata 的次级 HumanSL，原生普通／MuZero JSON analysis 和 GTP 已接入；普通／MuZero 次级网络已接入搜索及 JSON／GTP 入口，当前边界见 [象棋](docs/xiangqi.md)。** Go 与落子棋模型的输入边界会检查棋类，不能混用。外层采用 V3.4 的自适应逐轮调度：iteration 0 探测产量，iteration 1 补足训练门槛，后续按近期 rows/game 规划局数、按实际新增样本额度放缩训练步数。内部 epoch、Lookahead 与 SWA 跨 iteration 延续，每轮训练后导出候选并执行可选门控。阶段恢复与计时见 [调度与恢复](docs/controller.md)。

## 构建与入口

使用现有 Conda `pytorch` 环境、CUDA 13 与该环境中的 cuDNN/cuBLAS。默认构建架构为 SM120，可通过 `CUDA_ARCHITECTURES` 指定来源支持的目标；额外参数直接传给 CMake。

```bash
bash scripts/build.sh
bash scripts/run.sh --version
bash scripts/run.sh native version
bash scripts/run.sh check-config --config configs/checks/source_go
```

原生程序位于 `build/katago`，保留来源命令和版本标识。`native` 后的参数原样转发，不重写 GTP/JSON 输出。原生用法见 [Compiling.md](Compiling.md)、[SelfplayTraining.md](SelfplayTraining.md) 及来源文档；其中示例的源码根目录应理解为本目录。

CPU 推理可单独构建来源 Eigen 后端，支持全部六种棋类的普通／MuZero 网络与 PUCT／Gumbel 搜索。它沿用来源 FP32、NHWC 输入要求；MuZero API 的 NCHW 观测在适配边界转为 NHWC。安装 Eigen3，或给 CMake 添加 `-DEIGEN3_INCLUDE_DIRS=/absolute/path/to/eigen` 后执行：

```bash
cmake -S cpp -B build-eigen -DUSE_BACKEND=EIGEN -DBUILD_DISTRIBUTED=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-eigen -j 2
ETAZERO_EIGEN_TESTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n pytorch python -m pytest -q tests/test_eigen_backend.py tests/test_muzero_eigen.py
```

此构建的原生命令入口为 `build-eigen/katago`；外层命令使用 `--binary` 指定该文件。Eigen 检查复用 CUDA 的 Python 数值参考，覆盖 Go 四结果与规则掩码、四种落子棋的策略，以及象棋联合动作、镜像、缓存、并行组批和普通／Transformer 搜索。MuZero 检查另覆盖多步 latent、常量／小跨度归一化、metadata、状态归属、停机排空和 PUCT／Gumbel，并以 CPU learner 完成 Go MuZero／象棋 Gumbel MuZero 的两轮训练、普通／SWA 发布与恢复。

OpenCL 后端支持全部六种棋类的普通网络与 PUCT／Gumbel 搜索，沿用来源 NCHW 输入、计算层、kernel 和自动调优。构建需要 OpenCL 开发头文件、ICD loader 和设备驱动；安装说明见 [Compiling.md](Compiling.md)。非标准安装路径可用 `OpenCL_INCLUDE_DIR` 和 `OpenCL_LIBRARY` 指定。

```bash
cmake -S cpp -B build-opencl -DUSE_BACKEND=OPENCL -DBUILD_DISTRIBUTED=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-opencl -j 2
ETAZERO_OPENCL_TESTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n pytorch python -m pytest -q tests/test_opencl_backend.py tests/test_muzero_opencl.py
```

外层用 `--binary build-opencl/katago` 选择该程序，推理仍使用原生 `openclUseFP16`、`openclTunerFile` 等配置。检查覆盖 FP32、FP16 存储及来源 WMMA kernel；后者需要相应设备并设置 `ETAZERO_OPENCL_TENSOR_TESTS=1`。另设 `ETAZERO_EIGEN_TESTS=1` 可运行 Eigen 参考对照。测试中的固定调优参数只服务数值检查，不是生产配置或性能结论；目前实际验证的是 NVIDIA OpenCL，不能据此推断其他厂商设备行为。MuZero initial／recurrent 也复用原生 OpenCL 层，支持 FP32、FP16 存储及来源 tensor-core 路径；设备状态、组批、搜索和两轮训练检查见 [MuZero](docs/muzero.md#原生-opencl-执行)。

ONNX Runtime 后端支持六棋类的普通网络和 PUCT／Gumbel 搜索，沿用来源图构建器、运行时、缓存及配置。构建需要 ONNX Runtime C++ 包和 Protobuf 开发包／`protoc`，详见 [Compiling.md](Compiling.md)。

```bash
cmake -S cpp -B build-onnx -DUSE_BACKEND=ONNX -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime -DBUILD_DISTRIBUTED=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-onnx -j 2
ETAZERO_ONNX_TESTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n pytorch python -m pytest -q tests/test_onnx_backend.py
```

外层使用 `--binary build-onnx/katago`，原生配置明确选择 `onnxProvider=cpu`、`inputsUseNHWC=false`、`onnxUseFP16=false`。当前实际验证为 ONNX Runtime 1.23.2 CPU FP32：包括六棋类、象棋联合动作与物理棋盘镜像、Transformer 两种内部布局、scale8、固定／动态 batch、HumanSL 和 ONNX 文件重载。另设 `ETAZERO_EIGEN_TESTS=1` 可运行独立参考对照，`ETAZERO_GPU_TESTS=1` 可验证 ONNX 自对弈接 CUDA FP16 learner 的两轮训练、普通／SWA 发布与恢复。其他执行 provider 尚未验证，ONNX 的 MuZero 和 TensorRT 的非围棋网络尚未接通；文件协议见 [ONNX 模型文件](docs/ONNX_Model_Files.md)。

## 小规模闭环检查

```bash
bash scripts/run.sh run --config configs/checks/source_go \
  --run-dir /absolute/path/to/new-run --iterations 2
```

`--iterations` 是累计完成轮数；也可使用 `--max-hours` 指定累计时间预算。配置和命令行都未指定这两项时默认无限运行，直到手动中断；两项并用时任一达到即停止开始新轮次。时间预算在完整轮次边界检查，允许最后一轮超出预算；统计扣除实际 learner 编译时间，包含 MuZero 必需的初始化。已完成轮次可以继续增加上限，达到目标后再次运行不重复训练。计时失效和恢复口径见 [调度与恢复](docs/controller.md)。AlphaZero 首轮使用来源 random evaluator；MuZero 首轮由正式 learner 保存零样本的完整初始 checkpoint，再导出三组件模型。之后使用已接受的原生模型。工程检查也可使用 `configs/checks/source_placement`，其 `game` 可选择四种已接入的落子棋；象棋配置入口为 `configs/checks/source_xiangqi`。非围棋原生配置省略 `komiMean` 或设为 0，禁止 Go 让子。以上配置只用于工程检查，不是推荐训练配方。

`run.cfg` 使用 `bootstrap_games`、`train_steps` 和 `replay_ratio` 规划采样与实际训练预算，首轮数据门槛来自 shuffle 的 `min-rows`。`train.cfg` 保留优化器、SWA 和 Lookahead 参数；内部 epoch 的名义大小由基准步数与 batch size 决定，外层预算暂停时保留未完成的训练状态。具体调度、SWA 与恢复口径见 [调度与恢复](docs/controller.md)。

配置目录按职责拆分：`env.cfg` 管棋类和画布，`net.cfg` 管普通／MuZero 网络，`selfplay.cfg` 按组管理算法、原生搜索参数、采样与执行，`train.cfg`、`shuffle.cfg` 使用原生命令行参数名，`run.cfg` 管外围调度。`configs/baseline/` 保存完整公共参数，`games/` 和 `algorithms/` 分别保存环境和算法差异，各运行目录显式 include。外围转换层生成原生 `selfplay.resolved.cfg`、learner 参数和 MuZero 三组件配置；引擎与 learner 核心保持原有接口。`run.sh`、`autoexp.sh`、`autoelo.sh` 提供单臂、多臂和评估入口，支持 `CONFIG_DIR`、`.local` 与 `--dry-run`。用法、字段归属及与原生同步循环的差异见 [配置与启动](docs/configuration.md)。

原生 learner 的验证流程保留；当前逐轮控制器只生成训练快照，正常运行不产生验证 loss。来源验证数据划分方式和当前接入边界见 [验证数据](docs/controller.md#验证数据)。

外层支持**按实际提交点恢复中断轮次**：保留已发布自对弈 shard，补足剩余游戏；复用已发布 shuffle／模型目录；learner 恢复完整 checkpoint 中的 optimizer、Lookahead fast/slow 与同步计数、SWA、文件内游标及 reader/learner RNG、scaler。Ctrl+C／SIGTERM 等待自对弈排空或 learner 在安全 batch 边界保存，并结算本次耗时。强杀后最后一次保存之后的内存工作需要重做，原始日志保留但作废更新不进入有效曲线，自对弈在途状态不恢复；恢复边界、锁与计时限制见 [调度与恢复](docs/controller.md)。

训练提交后默认生成 `training.png`、`loss.png` 与 `logs/performance.png`；`auto_plot=false` 可关闭自动渲染。可使用 `bash scripts/run.sh plot --run-dir /absolute/path/to/run` 独立重建。算法与性能图直接读取原生日志，保留其统计口径和输出频率；时间横轴使用明确标注的 EtaZero 调度计时，缺测保留为空；详细定义与 V3.4 的口径区别见 [训练观测与绘图](docs/observations.md)。原生 `selfplaysurprisedump` 支持六种棋类与四种算法的逐手四结果、surprise 和重分析统计，独立诊断用法见该文档。

多实验臂可使用 `bash scripts/autoexp.sh --config-dir /absolute/path/to/experiment --work-dir /absolute/path/to/output`，按 GPU 槽位排队，共享初始权重但保持各臂独立的 learner 状态；配置和恢复方式见 [多实验臂调度](docs/controller.md#多实验臂调度)。新建单臂也可用 `run --weights` 载入初始权重。

独立比赛可使用 `bash scripts/run.sh match --config /absolute/path/to/match.cfg --output-dir /absolute/path/to/evaluation`；支持普通网络与 MuZero 混合对弈。`arena` 可读取训练提交记录，选择历史模型并绘制训练时间曲线。比赛恢复、配置索引、采样与 Elo 口径见 [独立比赛与 Elo](docs/evaluation.md)。

原生 `benchmark`／GTP `kata-benchmark` 已接入全部棋类与普通／MuZero 搜索，保留来源抽样、计时及调参流程。`benchmarknn` 使用来源前向计时循环，支持多棋类尺寸及分别测量 MuZero initial／recurrent。多棋类和新算法使用实际吞吐进行线程比较；Go PUCT 的来源经验 Elo 估计与对局 Elo 必须区分，见 [搜索与推理 benchmark](docs/benchmark.md)。

原生 `testgpuerror` 已接入六棋类、四结果和 MuZero initial／多层 recurrent，保留来源精度与并发比较指标及阈值；普通模型支持 Eigen 参考文件与 CUDA 对照，范围见 [推理误差检查](docs/benchmark.md#推理精度与批处理误差)。MuZero 支持 initial 与每个 recurrent 深度的 Eigen 参考读写和 CUDA／OpenCL FP32／FP16 对照，参考轨迹不使用真实后继棋盘。`testbackendreference` 支持六棋类、四结果及 MuZero initial 的原始输出采集与显式参考检查；来源 Go 检查路径保留，V4 训练族须自行提供对应参考数据和校准阈值，见 [训练族参考检查](docs/benchmark.md#训练族参考检查)。

原生 `evalsgf` 支持全部棋类、四种算法与普通／MuZero 次级网络，复用来源棋谱重放、搜索、树／JSON 报告与计时。动作选项按棋类解析，输入导出使用实际模型的特征版本和 NCHW 布局，支持 Go metadata；`searchentropyanalysis` 可从多份同棋类 SGF 汇总六棋类／四算法的搜索熵与 policy surprise，保留来源内置围棋数据集及统计流程。`checksgfhintpolicy` 已接入六棋类的普通／MuZero 原生策略评估，保留来源 hint 遍历与重复推理口径；`dataminesgfs` 的整局、树与 surprise 模式共用原生搜索和 writer，支持四种算法及象棋完整历史。用法与边界见 [原生棋谱诊断](docs/evalsgf.md)。

原生 `Book` 的状态、动作、对称、存取和重放支持六棋类；`checkbook`、`comparebooks` 与 `findbookbottlenecks` 可直接使用。`genbook` 支持六棋类、四种算法的搜索建库、续建、bonus 和 trace，保留来源成本与扩展流程；MuZero 子树在写入实际局面时检查真实合法性与终局。`booktoposes` 支持六棋类的普通／MuZero 原始推理，保留来源权重与筛选，并导出含完整象棋历史的原生训练起点。原生 `writebook`／HTML 已接入六棋类，保留来源统计与节点链接，展示 Hex 连接边及象棋棋子、联合动作和规则终局；范围见 [原生开局库](docs/book.md)。

## 文件与验证

运行目录保存 `config.json`、`source.json` 和包括未提交适配代码的 `source.tar.gz`；`rounds/` 保存逐轮输入模型目录、阶段日志和结果，`selfplay/` 保存原始数据，`shuffleddata/` 保存训练快照，`train/` 保存来源 checkpoint，`models/` 保存不可覆盖的发布模型。MuZero 另有 `initial/checkpoint.ckpt` 保存原始零样本状态，`muzero_network.json` 是交给 learner 的网络配置副本。阶段失败不推进完成轮数或 accepted 指针；已经导出的候选与已提交 learner checkpoint 保留。配置 `gatekeeper_config` 时使用独立原生比赛配置，拒绝候选不回滚训练；规则和恢复范围见 [调度与恢复](docs/controller.md)。

```bash
conda run --no-capture-output -n pytorch ctest --test-dir build --output-on-failure
conda run --no-capture-output -n pytorch python -m pytest -q tests python/tests
ETAZERO_GPU_TESTS=1 conda run --no-capture-output -n pytorch python -m pytest -q tests/test_four_results.py tests/test_controller.py -k 'compiled_training or exported_cuda or training_only_heads or pass_prior or source_model'
```

来源 GPU 专项按其自身条件运行；CPU 测试通过不代表 CUDA 或训练效果已经验证。Web 工作台从仓库根目录执行 `bash web/webui.sh --version EtaZero_V4`，使用原生 `gtp.cfg`；默认启动选择最高版本，参数见 [Web 与原生会话](docs/web.md)。

V4 的 Web 已接通结构化棋局查询、来源分析 JSON、可取消的常驻 GTP 传输、发布模型目录、异步会话控制及共享 HTTP 传输。创建、落子、悔棋、历史分支、分析中修改局面、参数设置与模型切换均由原生会话执行。页面与启动路由已接通，支持六种棋类、四种算法及原生 SGF 导入；平衡开局和按需网络诊断使用原生命令，象棋逐手动画仍待接入，接口语义与验证范围见 [Web 与原生会话](docs/web.md)。

| 任务 | 入口 |
|---|---|
| 查看源码职责、当前边界与接入顺序 | [架构](docs/architecture.md) |
| 查看中国象棋规则、动作与网络接入边界 | [中国象棋](docs/xiangqi.md) |
| 修改 MuZero 三组件网络与展开 loss | [MuZero 网络与展开训练](docs/muzero.md) |
| 修改 Gumbel 根/非根选择、改进策略与报告 | [Gumbel 搜索](docs/gumbel.md) |
| 配置五子棋／Hex 的原生平衡开局、检查初始化局面 | [平衡开局与初始化诊断](docs/openings.md) |
| 查看最终能力与允许偏差 | [KataGo 对齐目标](docs/katago-target.md) |
| 修改配置目录、启动单臂与多臂 | [配置与启动](docs/configuration.md) |
| 查看阶段恢复、并发锁与计时边界 | [调度与恢复](docs/controller.md) |
| 查看训练图、loss 图及性能统计 | [训练观测与绘图](docs/observations.md) |
| 运行独立比赛、恢复对局与拟合 Elo | [独立比赛与 Elo](docs/evaluation.md) |
| 测量搜索／推理速度、扫描线程与 batch、检查推理误差 | [原生 benchmark](docs/benchmark.md) |
| 分析 SGF／起点文件、打印搜索树、导出模型输入与汇总搜索熵 | [原生棋谱诊断](docs/evalsgf.md) |
| 生成、续建、检查与浏览开局库、分析瓶颈和导出训练起点 | [原生开局库](docs/book.md) |
| 接入 Web 原生棋局、分析和取消 | [Web 与原生会话](docs/web.md) |
| 修改阶段编排与发布 | [run.py](python/etazero/run.py) |
| 修改共有搜索、推理、自对弈和训练 | `cpp/search/`、`cpp/neuralnet/`、`cpp/program/`、`python/katago/train/`、`python/train.py` |
