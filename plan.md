# EtaZero V0：KataGo / KataGomo 对齐实施记录

本文保留 2026-10-02 已完成的 V0 对齐计划、阶段决策和验收历史，不作为 V1 的待办、当前配置要求或测试参数约束。历史证据保持原有范围；当前能力按目标版本的文档与实现核对。

## 目标与验收状态

本文件的实施批次记录保留各批当时的配置和证据；当前baseline参数以 [EtaZero.md当前profile](EtaZero.md#alignment-profiles) 和 [baseline配置](EtaZero_V0/configs/baseline/) 为准。历史m250000/p0.65/a0.4、side0.020、SP half-life19、reduced min350和默认跳过validation均不作为当前默认配置。

以 `EtaZero.md` 的全部 57 项为算法验收清单，另增加本文件 E01—E26 的工程审查清单，分批修正已有实现、补齐缺失机制，再重新逐项审查。对齐需要覆盖公式、默认参数、分支、调用链、数据语义、训练与评估行为；不能仅凭函数存在或公式相同判定完成。

目标版本为开发中的 `EtaZero_V0`。本计划依据 2026-10-02 的代码审查，起点 HEAD 为 `185eee220fd786a6e7436ae4e2332b929904bdb0`；实施前目标文件集合的核查值为 `e83d53b8012b67dc67dfa8901f28eaa9174f258c442bedbe047b901045c7b4b1`。文档中的旧 HEAD 不作为新实现的完成依据。

2026-10-02 已完成批次0E、0—10，包括9a/9b/9c及最终57项算法、26项工程复审。来源、实际配置、边界、SP/learner/eval/Match调用与执行证据分别记录；静态结论、实际运行与性能结论保持区分。S23按用户要求暂缓，未实现、未对齐，不计通过。最终验收见[第10批证据](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json)。

## 已确定的范围与例外

1. **S02：完整 / cheap 搜索预算保持 400/70。** 不改为来源的 2000/350。这只豁免这两个预算值；PCR 概率、cheap 权重、清树、温度、集成及关联分支仍需对齐。S07 的 reduced 最低预算是独立参数，不随此例外豁免。
2. **S23：Subtree value bias 暂缓实现。** 不列入当前实施批次；最终复审明确写“用户要求暂缓，未实现、未对齐”，不计为通过。仅此机制暂缓，不推及 graph search 或其他搜索修正。
3. 其他参数差异、缺失分支和调度差异均进入计划，不因“当前没启用”而自动豁免。可配置开关关闭时也需具有正确实现，开启未实现功能应明确报错。
4. 算法对齐范围是 `EtaZero.md` 列出的机制及其中影响结论的分支；工程范围由本次新增的 E01—E26 清单明确，不扩展成整个 KataGo 仓库的所有功能。表中提及的 hint/fork、reanalysis、FPU 另一分支、直接 value surprise 等需逐项登记适用范围，不能静默遗漏；表外功能不自动追加。
5. 围棋专有的 pass、komi、score、ownership、no-result 等不能直接套入五子棋。每项必须有明确的五子棋映射或“不适用”的源码依据；五子棋已有对应机制时核查 KataGomo。涉及 VCN、矩形棋盘及辅助价值目标的范围在批次 0 中逐项落实，不以“环境适配”笼统免验。
6. **批次0用户确定的编排与环境适配**：保留EtaZero固定训练量与产样规划，不引入KataGo训练桶；保留同步逐轮固定模型、整轮提交恢复。环境沿当前方形15/14/13/12/11及权重100/10/5/3/1，三规则可配置混训，baseline规则权重保持1/0/0；矩形不纳入，VCN暂不使用。网络预设采用b10c128-fson-mish（v15）、b5c192nbt-fson-mish（v15）、b5c192h3nbttfrs（v17）。这些是明确适配，不标成来源一致，其他参数/分支不因此豁免。
7. 不机械复制来源的机器规模与并发配置，也不把资源配置差异误标为算法 bug。搜索预算例外、网络预设、棋盘/规则分布、推理精度、batch/epoch 和并行度分别登记；除已明确的例外外，不自行把当前数值认定为保留项。
8. 不覆盖旧数据、checkpoint、模型和实验结果。输入、目标或训练状态契约发生变化时，用新的运行目录验证；开发中的 V0 不增加旧协议兼容层。

## 来源和参数基准

| 部分 | 固定来源 |
|---|---|
| KataGo | `/home/sky/RL/SkyZero/KataGo`，`d91ea855110dae533f0aada947b2b7d78cc8a4e1` |
| 自对弈搜索 | `cpp/configs/training/selfplay8mainb18.cfg`，加 `SETUP_FOR_OTHER` 加载后默认值 |
| Eval / Match | `cpp/configs/match_example.cfg`，加 `SETUP_FOR_MATCH` 加载后默认值；固定局面 eval 与比赛开局分别核查 |
| Learner | 该 commit 的 `python/train.py`、`python/katago/train/model_pytorch.py`、`metrics_pytorch.py` 等；按对应模型版本分支核查 |
| Replay / shuffle | `python/shuffle.py` 的实现；`python/selfplay/shuffle.sh` 的示例参数与 CLI 默认值分别记录 |
| 工程编排 | `SelfplayTraining.md`、`python/selfplay/synchronous_loop.sh` 与异步 loop scripts 分别核查；同步/异步不混作唯一来源路径 |
| KataGomo | `/home/sky/RL/SkyZero/KataGomo`，`df152116e3787c75c6a3de099d261ca092b7dfc1`；平衡开局、policy init、禁手输入和五子棋规则 |
| 当前网络对应预设 | NBT 对照 `b5c192nbt-fson-mish`；搜索配置的 b18 名称不意味着必须把网络扩大到 b18 |

不把仓库参考配置表述为线上实时配置。来源版本变更需重新核查受影响项，不混用不同 commit 的公式与默认值。

## 工程审查：先核查，再安排修改

工程审查先于实现批次开展。现有实现已经包含共享 evaluator、多推理服务、后台 writer、两阶段 shuffle、multi-wave、常驻进程池和 CPU/CUDA 预取；不能因为算法清单尚未齐全，就推定这些工程实现也需要重做。上一轮未对本节全部边界形成审查结论。

**先确定来源执行路径。** KataGo 的 `SelfplayTraining.md` 同时介绍单机同步逐轮和常驻异步训练；`synchronous_loop.sh` 顺序执行各阶段，但仍使用 train bucket、no-repeat-files 等控制训练消费，并可跳过 validation。当前 EtaZero 的逐轮组织可先与同步路径对照，再单列与主线异步路径的能力差异。此前计划把“改为异步”直接列为必做工程改造，现修正为：审查数据消费、额度与模型时点的实际差异，按确定的目标 profile 安排修改。同步脚本的旧 selfplay1 数值不替换本计划选定的算法参数基准。

**每项同时记录两个维度。**

- 对齐结论：机制一致；实现不同且已证明语义等价；存在会改变抽样/数值/消费/模型时点的差异；能力缺失；来源路径不适用；证据不足。实现不同不自动算通过，也不自动算 bug。
- 证据级别：源码和生效配置已核查；独立小样例已验证；真实 CUDA/并发/中断条件已验证；端到端性能已测量。代码相似、线程数相同和使用相同压缩格式都不能证明吞吐相同。

每项记录 `来源位置 → 目标入口 → 实际配置/调用链 → 差异及影响 → 验证证据 → 结论 → 后续批次`。来源与目标的文件名和代码结构可以不同，采样分布、训练样本消费、模型版本和资源生命周期必须逐项核查。

### 工程审查清单

各行已完成首轮静态审查；逐项来源/调用链、差异、结论与待验条件见 `EtaZero.md` 的 E01—E26 工程章节。下表入口列用于路由，不是通过证据。来源文件相对 KataGo 根目录，目标文件相对 `EtaZero_V0`。

| ID | 审查对象与关键问题 | 目标入口 | KataGo 来源入口 |
|---|---|---|---|
| E01 | 阶段顺序/重叠、触发与退出条件；同步和异步两条路径分别判定 | `python/etazero/runtime.py` | `SelfplayTraining.md`、`python/selfplay/*loop.sh` |
| E02 | CPU/GPU 任务分配、局数线程/树内线程/服务数的层次，是否过度订阅 | `python/etazero/native.py`、`cpp/src/commands/main.cpp` | `cpp/command/selfplay.cpp`、`cpp/program/setup.cpp` |
| E03 | 对局线程和常驻 worker 的复用、独立 RNG、模型释放与重新加载 | `cpp/src/commands/main.cpp`、`python/etazero/native.py` | `cpp/command/selfplay.cpp`、`cpp/program/selfplaymanager.cpp` |
| E04 | 树内并行、原子统计发布、mutex pool、pending/virtual loss、线程/树回收 | `cpp/src/search/search.cpp` | `cpp/search/search*.cpp`、`searchnode.cpp`、`mutexpool.*` |
| E05 | NN 队列、立即组批/等待组批、容量与背压、停止和失败唤醒 | `cpp/src/inference/batcher.cpp` | `cpp/neuralnet/nneval.cpp`、`cpp/core/threadsafequeue.*` |
| E06 | 多推理服务的设备、stream、输入/输出缓冲和只读权重生命周期 | `cpp/src/inference/torch_backend.cpp` | `cpp/neuralnet/nneval.cpp` 及所选 backend |
| E07 | Cache key、碰撞、并发命中、模型隔离、朝向与根集成绕过 | `cpp/src/inference/batcher.cpp`、`cpp/src/search/search.cpp` | `cpp/neuralnet/nneval.cpp`、`cpp/search/searchnnhelpers.cpp`；关联 A03/S16/S17 |
| E08 | Writer 有界队列、完整局交接、flush 阈值/定时、错误传播和退出排空 | `cpp/src/selfplay/record.cpp` | `cpp/program/selfplaymanager.cpp`、`cpp/dataio/trainingwrite.cpp` |
| E09 | Bit packing/NPZ/压缩、行数定义、目标数组同步、重复行与 dtype | `cpp/src/selfplay/record.cpp`、`python/etazero/data.py`、`schema.py` | `cpp/dataio/trainingwrite.cpp`、`python/katago/train/data_processing_pytorch.py` |
| E10 | 临时文件到可消费文件的发布，rename/持久化屏障、读写竞争 | `python/etazero/storage.py`、writer/shuffle/export | `trainingwrite.cpp`、`python/shuffle.py`、export scripts |
| E11 | 增量 catalog/summary、文件不可变假设、去重、损坏与累计 usable rows | `python/etazero/data.py` | `python/summarize_old_selfplay_files.py`、`python/shuffle.py` |
| E12 | 近期窗口、random 数据封顶、文件排序与超额完整分片 | `data.py`、`shuffle.py` | `python/shuffle.py`；关联 R01—R04 |
| E13 | 输入组边界、每组降采样/round、RNG 派生及采样相关性 | `shuffle.py` 的 `_groups/_scatter` | `python/shuffle.py` 的 `group_files_by_rows/shardify` |
| E14 | 两阶段随机分桶与桶内 shuffle 的联合分布；所有数组共用排列 | `shuffle.py` 的 `partition_rows/_two_phase/_merge_bucket` | `python/shuffle.py` 的 `shardify/merge_bucket` |
| E15 | Multi-wave 分配、只降采样一次、行集合守恒、输出混合与清理 | `shuffle.py` 的 `_write_data` | `python/shuffle.py` 的 multi-wave 路径 |
| E16 | 输出 shard 大小/均分策略、batch 尾部丢弃及实际样本使用率 | `shuffle.py`、`reader.py` | `python/shuffle.py`、`data_processing_pytorch.py`；关联 R05 |
| E17 | Shuffle 多进程任务划分、池复用、数组内存/临时磁盘预算和溢出桶 | `shuffle.py`、`runtime.py` | `python/shuffle.py` 的 pool/resource/wave/merge 路径 |
| E18 | Reader 文件顺序、预取深度、重复消费、游标与已预取/已消费的区别 | `python/etazero/reader.py` | `data_processing_pytorch.py`、`python/train.py` |
| E19 | CPU batch 准备、pinned memory、CUDA 上传 stream/event 与缓冲覆盖 | `reader.py` 的两类 batch prefetcher | `data_processing_pytorch.py`、`trainloop_helpers.py`、`python/train.py`；无直接同构实现时核查语义 |
| E20 | Compile/fused/AMP 的启用条件、精度、图边界和 eager 数值关系 | `training.py`、`optimization.py`、`network.py` | `python/train.py`、`trainloop_helpers.py`；关联 N09—N11 |
| E21 | Epoch/subepoch、train bucket、无数据等待、样本/更新/调度计数 | `runtime.py`、`training.py`、`optimization.py` | `python/train.py` 与所选 loop script；关联 R05/N07—N11 |
| E22 | Validation 分片选择、复用泄漏、D4 关闭与 profile 的 skip 选项 | `shuffle.py`、`training.py` | `python/selfplay/shuffle.sh`、`synchronous_loop.sh`、`python/train.py` |
| E23 | Export、校验、发布、模型轮询/局内切换、旧模型/在途请求生命周期 | `export.py`、`runtime.py`、`native.py` | export scripts、`cpp/command/selfplay.cpp`、`selfplaymanager.cpp` |
| E24 | Checkpoint 与恢复边界；优化器/RNG/reader/SWA/额度等可恢复范围 | `training.py`、`runtime.py` | `python/train.py` 与 loop scripts；关联 A04 |
| E25 | SIGINT/SIGTERM、异常、部分写入、取消半局、死锁及所有后台任务收尾 | worker/writer/batcher、`reader.py`、`process.py` | `cpp/command/selfplay.cpp`、`selfplaymanager.cpp`、`nneval.cpp`、训练/数据脚本 |
| E26 | 阶段/端到端墙钟、编译暖机、队列等待、batch 分布、吞吐与峰值资源 | 日志、`runtime.py`、`plotting.py` | 对应来源计时/统计入口；性能需相同负载实测 |

### 审查顺序与证据复用

1. 先核查 E01/E21/E23/E24 的控制流和状态交接，再审查 E11—E18 的样本链路，最后核查 E02—E10/E19/E20/E25 的并发与资源生命周期。
2. 读取 `EtaZero_V0/reference_sources.json`、相关代码及已有验收记录，核对来源/目标源码、配置、硬件与实际覆盖路径。旧测试结论仅在证据仍适用时复用；文档“已验证”不是本轮复跑结果。
3. 第一阶段只做代码层面审查，填写逐项证据和结论，不同时进行优化或启动吞吐测试。Shuffle 重点核查样本集合、采样概率、排列/分组、waves、文件尾部与消费次数；并发重点核查所有权、背压、发布可见性、失败/退出路径。
4. 源码不足以决定的分布/并发疑点，列出所需独立样例、真实 CUDA 或中断检查。若进入验证阶段，对 shuffle 可使用唯一 row ID 检查多重集合、字段配对和实际消费；检查 RNG 序列与分布等价时分别给出结论。
5. 语义已对齐的实现保留并登记；确认会改变算法/数据语义的差异进入相应修正批次。DDP、跨机器、专用 native backend 等能力差异单列，不能把来源支持的所有部署方式自动变成本项目必做项。
6. E26 先检查统计代码和口径；性能结论另用明确负载、设备、精度、访问预算、数据布局/压缩和暖机条件的端到端测量。不会由源码审查推断“性能基本相同”。

工程审查结果补入 `EtaZero.md` 的独立工程章节，以 E 编号引用；本文件负责修正工作的批次和验收，不另建重复的工程报告。算法行和工程行交叉引用，避免 R05/A03/A04 等重复维护不同结论。

## 实施方式与完成标准

每次实施一个批次，较大的批次按下述子批次拆开。进入批次前读对应来源实现、配置加载、上下游和测试，写清本批具体差异；完成后更新本文件状态以及 `EtaZero.md` 受影响行，更新版本 README 和职责文档中的当前行为。

每批完成必须同时满足：

- 来源公式、必要分支和实际调用链均已落实，SP、learner、eval、Match 分别可检查。
- 配置字段能进入实际执行路径，尚未实现的依赖不使用占位或静默回退。
- 有与改动风险相称的验证：优先手算样例、独立来源结果及边界性质；训练链路改动再用小规模真实运行验证。
- 验证记录注明源码、配置、命令、结果及未验证范围。不得把静态一致、运行成功和学习效果混为一谈。
- 本批涉及的新输入、目标、计数和状态可保存、加载；最终整链路恢复在批次 9 按选定执行 profile 验收。

不为文字改动机械补测试，也不重复实现同一公式作为预期值。不因补实现启动正式训练或扩展研究实验矩阵。

## 批次总览

| 批次 | 交付内容 | 主要依赖 | 状态 |
|---|---|---|---|
| 0E | E01—E26 工程源码审查；保留已有实现，登记真实差异 | 现有代码/来源/优化验收记录 | 已完成（2026-10-02，仅静态；CUDA/并发/中断/性能待后续验证） |
| 0 | 逐项确定参数、五子棋映射和分支范围 | 现有审查 | 已完成（2026-10-02，范围及用户选择） |
| 1 | 已有 learner 的数值、刷新时点和 AMP 语义修正 | 0 | 已完成（2026-10-02，公式/边界及真实CUDA；最终编排在9） |
| 2 | 已有搜索参数、随机 D4、cache、reuse 和预算边界 | 0 | 已完成（2026-10-02，独立树/坐标/cache/预算及真实CUDA/并发/中断） |
| 3 | KataGomo 开局、禁手 dropout、棋盘与规则 | 0、2 的推理接口 | 已完成（2026-10-02，逐输出行/异bot/有限棋规语料及真实CUDA） |
| 4 | 辅助 heads、训练目标、loss 与推理输出 | 1、3 的数据契约 | 已完成：v15/v17可选纯W−L Q及数据/训练/导出链验收通过 |
| 5 | Graph / transposition search | 2 | 已完成：来源小图、C++/sanitizer及真实CUDA通过 |
| 6 | Uncertainty、optimistic policy、noise pruning | 2、4、5 | 已完成：来源公式、二阶统计及真实CUDA通过 |
| 7 | PDA、side position 和相邻采样分支 | 3、4、6 | 已完成：PDA/side/reanalysis/hint/fork及频率来源/CUDA通过 |
| 8 | 独立 plain Convnet 与 Transformer+NBT | 1、4 | 已完成：三架构、v17 Q开关及对应loss/精度/恢复/导出验收通过 |
| 9 | 按工程审查确定的 profile 修正 Replay、训练编排、发布与恢复 | 0E、1—8 | 已完成（9a/9b/9c） |
| 10 | 对照 `EtaZero.md` 的算法/工程清单全量复审与必要的整链路验收 | 0E、1—9 | 已完成（57算法/26工程逐项复审，独立来源及真实CUDA验收） |

以上依赖允许复用已验收工作；实际执行仍按批次推进，不能用后续计划替代本批交付。批次 5、8 可以各自独立验收。

### 批次 0E：工程源码审查交付

- [x] 2026-10-02：从控制流/状态交接、样本链路、并发/资源三个路径完成全部 E01—E26 静态审查。结果唯一维护在 [EtaZero.md 工程章节](EtaZero.md#engineering-audit)，包括逐项来源→目标→生效配置/调用→影响→静态结论→后续验证/批次。
- [x] 核对来源 commit/干净工作区及 `reference_sources.json` 已登记的 39+7 文件 hash；固定目标 HEAD `185eee220fd786a6e7436ae4e2332b929904bdb0` 和 50 文件工程集合 hash `3d56e54ed95b23ff8cf65a6a4c077dd8eec2edfd99290219fc96b976edc3778c`，不沿用旧 HEAD 作为新验收证据。
- [x] 分别登记同步与常驻异步路径；同步阶段顺序可以对照，但最终 profile 仍由批次0落实。新增差异并入已有后续批次，不追加 DDP/跨机器/native backend 实施范围。
- [x] 确认 E13 分组阈值改变round采样、E15跨wave merge seed重复、E16/E18文件尾部与repeat消费差异；E20 AMP heads/skip、E21额度/时钟、E23/E24模型/恢复不同均有实际调用依据。
- [x] 更正 R05 对来源桶的描述：本commit按 train.json.range[1] 补桶、epoch开始预扣；更正 N06/E22 的validation来源描述：本commit开启随机D4。0E时将usable/逐batch扣款及validation关闭D4目标留待批次0决定；现已确定不引入桶，validation启用时沿来源随机D4。
- [x] 文档/引用/编号/来源及目标hash检查完成，结果见本文件“验证入口与记录”。只做源码与文档检查；不运行优化/吞吐、训练或CUDA检查。已有测试入口登记但未复跑，无新训练产物。

交付入口：`EtaZero.md` 工程章节及 R05/A03/A04 交叉引用；版本 README 的文档路由与 `docs/implementation.md` 当前能力边界。验收结果：26项无漏项、每项均有静态结论及后续去向；运行等价、性能和全面对齐未验收。

### 批次 0：把对齐目标落实到每个条目

- [x] 为 57 项登记：来源 commit/位置、生效配置及模型版本、目标入口、现状差异、验收方式、环境映射和明确例外；与 0E 的工程结论交叉核对。
- [x] 确定各 profile 的最终参数：除 400/70 外，逐项处理 cpuct、variance/value weighting、各温度、Dirichlet、reduced min、Match visits、replay 等差异，不能默认维持现状。
- [x] 网络结构按各自对应预设核查；确定 plain / Transformer 的来源预设及其 heads/loss 版本，避免把 version 15 的分支套到所有架构。
- [x] 对 WDL draw、TD、short-term error、PDA 输入和预算建立明确的五子棋定义；记录去除 Go score 等输出后哪些公式随之改变。
- [x] 明确 D04/D05 的棋盘尺寸、矩形支持、规则采样及 KataGomo VCN 的适用范围；本批所需研究选择已取得用户回复，不擅自扩展实验矩阵。
- [x] 把表内非默认分支与真正不适用分支区分；S23 保持暂缓。表内非默认分支已有去向；表外功能不自动追加。

验收：57 项均有明确去向，只有用户明确的例外可豁免；不能留下“后面再说”的未登记范围。0E调度取舍已落实：用户保留本地同步逐轮产样规划，无训练桶，因此raw/usable补桶和epoch/逐batch扣款均不进入目标。Validation默认skip，启用分支沿来源随机D4，取消原计划关闭D4适配。

交付：[`EtaZero.md`实施目标](EtaZero.md#alignment-targets)集中维护profile、网络版本、环境公式、非默认分支及57项来源/目标/验收登记。只完成范围与参数定义，未修改算法或生效配置；后续批次按该目标实施。用户确认日期2026-10-02；检验记录见文末。

### 批次 1：修正现有 learner

涉及 N07—N11、L01/L02/L04，首先修当前已启用分支的错误，再补列入范围的可选分支。

- [x] fson + SGD 自动裁剪基数改为来源的 2500，保留 AdamW 的独立分支及显式 override 语义。
- [x] 对应 version 15 的 ordinary policy CE 系数落实为 0.930；基础 value CE 按来源内部 1.20 与默认 scale 0.6 的乘积核查，不混淆内部系数和配置系数。
- [x] 对齐 LR/WD 的 5/50 batch 刷新、warmup 阈值、100 batch 范数采样与生效时点，以及列入范围的运行范数分支（来源norm_*_batch在打印点保留历史0.001，不是逐batch EMA）。
- [x] 拆清 consumed samples、optimizer 成功更新、Lookahead 计数和 SWA 时钟；AMP overflow 按来源 skip 语义推进相关计数，消除同 batch 重试造成的差异。
- [x] 对齐 autocast 边界，特别是 policy/value heads 在来源中的 FP32 路径；在 loss 中转 float 不能代替 head 运算的 FP32。
- [x] 核查 Lookahead 的 subepoch 重置、epoch 末 slow 恢复、SWA 同步点与采样周期；最终本地轮/分段调度在批次9接入，不引入来源训练桶。

验收：独立 LR/WD/裁剪公式对照，跨刷新与 warmup 边界的状态序列，overflow 后计数与更新序列；必要时真实 CUDA AMP 小规模运行及训练状态往返加载。

交付：`optimization.py`、`training.py`、`network.py`、配置与来源核查脚本；来源`train.py:1638–1910`、`model_pytorch.py:4108`、`metrics_pytorch.py:127/643`和`metrics_logging.py:10/29`。本地默认每轮单分段；分段计数重置方法已实现并手算验证，非默认分段编排留批次9。推理精度配置、辅助heads及其他架构仍留各自批次，不计本批通过。

### 批次 2：修正现有搜索与推理

涉及 S01—S07、S12—S22、S24、A02/A03；已一致项作为回归约束。

- [x] 对齐各 profile 的 cpuct/log/variance 参数、Match value exponent、NN/root/move 温度和总噪声浓度；保留 S02 的 400/70。
- [x] 补齐叶节点、cheap 根及 Match 的随机 D4；还原 policy 坐标，区分指定朝向与随机朝向。
- [x] 单朝向 cache 按来源棋局/输入条件复用，不能仅因朝向不同产生独立 cache；根多对称请求绕过 cache。raw-logit 存储可以保留，但其最终可观察行为须核查。
- [x] 对齐 Match 不同 bot 的 reuse 与相同 bot 的清树分支，根先验重置和模型状态刷新；不能仅把一个 reuse 配置改为 true 就结束。
- [x] 覆盖 fresh/reused root、已有 visits 与新增 playouts、终局、多对称 NN 请求计数；补齐表内 maxTime/maxPlayouts 限制及停止边界。
- [x] 核查 FPU 表内另一分支和 S07 reduced 参数；PCR 的 hint/fork 例外转入批次 7，不因此把 S02 判为全部完成。
- [x] 分别核查训练和推理精度配置；算法验证使用固定输入，真实并发下另验证 pending/virtual loss 的释放及异常清理。

验收：可手算搜索树、指定网络输出的对照局面；D4 输出坐标、单朝向 cache 命中、根集成绕过、reuse 预算以及 SP/Match 分流均符合来源。S03/S04/S05/S21/S24 不得回归。

### 批次 3：KataGomo 环境与训练行

涉及 D01—D05、L03、A01。

- [x] 禁手 dropout 从“每轨迹状态一次”改为“每个最终输出训练行一次”，重复行独立抽样；两禁手平面和全局标志同步，搜索使用完整提示。
- [x] 平衡开局的拒绝/回退、权重和随机选择对应来源；Match 的平衡搜索及 policy init 按来源正确选择 botB/botW，保留成对换色评估的明确口径。
- [x] 对齐 policy init 开关、mean、温度及已有手数补偿，分别检查 SP/Match 的加载默认与显式参数。
- [x] 根据批次 0 的映射补齐棋盘/规则支持和采样；本次保留方棋盘；不新增矩形或VCN配置，明确拒绝超出已实现环境的请求。
- [x] 对 Renju 禁手、长连、恰好五连、双四/双三及递归边界建立 KataGomo 的独立对照；修正实际差异，不能由少量普通局面断言规则完全等价。
- [x] 保持搜索合法域与训练 on-board softmax 域的区别，终局价值视角和备份变号正确。

验收：按输出行观察 dropout 随机粒度；黑白异模型开局对照；规则边界与混合尺寸的 mask/终局核查。未覆盖的规则分支明确保留待验。

### 批次 4：辅助网络输出、监督与数据契约

涉及 N05、L01/L02/L04，以及 S09/S26/S27 的前置依赖。

- [x] 按来源模型版本补 optimistic policy、TD、多时间尺度价值及 short-term error 输出，落实形状、价值视角、目标构造、有效性权重和 loss 系数。
- [x] 普通主局与未下到终局的 side row 区分；为后者提供来源定义的搜索价值监督及 target 权重，不伪造真实终局标签。
- [x] Policy target 对齐来源的缩放、int16 量化与 learner 归一化顺序；检查稀疏、小权重与零权重边界。
- [x] 若相应模型版本含 Q/其他列入清单的 heads，落实对应分支；不向 version 15 强加只在其他版本存在的目标。
- [x] 贯通 writer → 原始存储 → view → shuffle → reader → loss → export → C++ evaluator，列出训练专用输出与搜索所需输出。
- [x] 新目标同步参与 D4；保持样本频率与 loss 权重只在来源规定的位置作用，禁止重复乘权。

v15及v17缺省六policy/WDL/TD/error、side数据契约与可选v17第七个纯W−L Q已接入并验收。Q按当前玩家视角child NODE visits提取，含终局；逐输出行独立随机量化、sqrt(visits)加权及side/reanalysis/D4/reader/learner/export往返通过；Go score Q不适用，v15拒绝Q。批次6/7的搜索应用及侧分支保持验收结果。

验收：以可手算轨迹独立构造终局/TD/error/optimistic 目标和梯度；跨存储链路往返一致；真实小规模训练、导出、加载和搜索消费输出。Go 专有监督的排除依据写入审查表。

### 批次 5：Graph / transposition search

涉及 S25，并回归 S03—S06、S12/S15/S24、A02/A03。

- [x] 按来源实现局面共享、节点/边统计、访问与权重更新、重复路径处理和生命周期。
- [x] 保持图搜索与 tree reuse、virtual loss、pending visits、cache、根聚合和预算的正确关系；不能用 NN cache 冒充 graph search。
- [x] 核查规则/历史条件对局面 key 的影响，以及多父节点、重复到达、终局和并发释放路径。

验收：构造同一局面由不同落子顺序到达的五子棋图，与来源的小图统计独立对照；覆盖多线程、根推进和销毁。Graph 关闭分支也需回归。

### 批次 6：补搜索修正机制

按 6a uncertainty、6b optimistic policy、6c noise pruning 分别验收，涉及 S26—S28。

- [x] Uncertainty 使用批次 4 的真实 short-term error 输出，对齐权重公式、上下界、统计与 SP/Match 默认值，不用固定常数替代预测。
- [x] Optimistic policy 使用来源的 ordinary/optimistic logits 混合与 root/leaf 参数，处理温度、D4、cache 和输出版本；不能用概率平均替代 logits 混合。
- [x] Noise pruning 按来源修正 value backup，贯通根噪声、聚合和权重；与 S04 的 policy target pruning 分开核查。

验收：构造同样 visits、不同 error/logits/noise 的对照树，检查输出和二阶统计；SP 关闭、Match 开启时实际路径均符合来源，验证互相组合及 graph 路径。

### 批次 7：PDA、side position 与采样分支

按 7a PDA、7b side position、7c 表内相邻分支分别验收，涉及 S02/S06/S08—S11。

- [x] PDA：采样预算比、双方 budget 系数、执色符号输入、清树和条件推理；side rows 清除 PDA。Go komi 补偿按批次 0 的五子棋映射处理。
- [x] Side position：排除实际着的候选采样、独立搜索、policy/search-value target、递归分支、权重及写入；不把开局前缀当 side position。
- [x] 对齐 hint/fork 引发的 full/cheap 例外及表内直接 value surprise 分支；按批次 0 的适用清单落实 reanalysis 与 cheap policy excess 处理。
- [x] 普通、cheap、reduced、PDA、side、reanalysis 的频率和有效性权重分别核查，保持 policy/value surprise 的正确价值视角与随机取整顺序。

2026-10-02完成：PDA/side/reanalysis、hint/hintFork/early/game fork及PCR例外均接入；预算/输入、完整频率重分配、前缀/gate和真实CUDA闭环验收见下方7a/7b与7c证据。

验收：预算比与输入符号的手算例，side 无终局记录的真实训练消费，以及新增/重复行计数；对全零 surprise、低均值、cheap 恢复采样和侧分支边界作来源对照。S23 不纳入。

### 批次 8：补独立网络架构

按 8a plain Convnet、8b Transformer+NBT 分别验收，涉及 N01—N05。

- [x] 增加来源对应的 plain ResNet 选择；结构、gpool block、初始化、归一化和 heads 符合其预设。
- [x] 增加 Transformer+NBT：attention、padding mask、RMSNorm、2D RoPE、SwiGLU/FFN 与来源 block 结构，不能以 global pooling 替代 attention。
- [x] 对每种架构落实配置合法性、参数组、loss 版本、推理导出及精度边界；保持原 NBT 预设的正确结构。

验收：来源权重映射或固定张量的独立前向/梯度对照，混合尺寸及 padding 隔离；每种架构完成小规模更新、保存、加载和导出。架构交付不等于训练效果验证。

### 批次 9：Replay 和训练整链路

涉及 R01—R05、N06—N11、A03/A04，并消费 0E 的工程审查结果。同步逐轮和常驻异步分别核查，先确定所对齐的执行 profile；已有工程实现通过审查时保留，仅修改确认的语义差异。当前逐轮协议与主线异步仍有差异，不能因来源也支持同步就宣称异步能力已实现。

**9a：Replay 与 batch 消费**

- [x] 对齐 random 数据在累计 usable rows 中的封顶、窗口参数及可选扩展、近期文件排序、group 边界、保留概率与 round；分别记录 CLI 默认和脚本示例。
- [x] 对齐每文件完整 batch 的消费及尾部丢弃语义；消除跨文件凑 batch、尾部绕回旧快照等来源没有的行为。
- [x] 核查 shuffle 输出 shard 大小与样本使用率；按选定 profile 落实 validation 切分/跳过及 epoch 验证，启用validation时按来源开启随机D4；批次0取消原计划关闭D4的适配。同步来源的 SKIP_VALIDATE 分支单独记录。

**9b：采样、shuffle、learner 编排与模型切换**

- [x] 保留用户的固定训练量与产样规划：ratio8、每轮1000×128消费样本，核查新增有效行缺口、整局超额、冷启动锚点和无数据情况。消费样本、成功更新及产样quota分别记账；不引入train bucket、raw/usable补桶、epoch预扣或逐batch扣桶。来源bucket机制继续作为明确调度差异记录，不计为来源一致。
- [x] 保留用户确定的同步逐轮阶段边界；已有worker/进程池和预取保留，修改确认的样本语义及计数差异。常驻异步能力不进入本次实施，不宣称已实现。
- [x] 落实本地轮/分段、consumed samples、成功更新与Lookahead/SWA/LR/WD独立时钟；1000消费batch映射为本地固定训练量，SWA默认64000样本，不称来源默认epoch。AMP跳过不重试同batch。
- [x] 保留每轮固定模型、轮末发布和下轮换代；不增加局内轮询切换，mainb18的switchNetsMidGame差异登记到A03。在途NN请求、每模型cache和输入输出契约不能串用。

**9c：保存、发布与恢复**

- [x] 保存 model、optimizer、scaler、Lookahead、SWA、各类计数、RNG、reader及数据消费/产样规划状态（不引入bucket）；按目标 profile 核查恢复边界。区分来源实际保存范围、EtaZero 额外恢复能力及重跑语义，不能假定 KataGo 自动恢复全部在途状态。
- [x] 发布 SWA/raw 模型并验证产物完整性；中断和多进程竞争不能读取半写模型或重复消费数据。
- [x] 验证所选 profile 的阶段重启及写入/更新/发布边界中断；常驻异步不进入本次目标。保留旧产物，明确哪些在途工作重新执行。
- [x] A04 中 gatekeeper 的适用性按批次 0 登记；来源支持直接发布流程，不能仅凭仓库存在 gatekeeper 推定所有运行必须使用它。

验收：所选 profile 的真实小规模整链路运行；检查固定训练量与新增产样quota缺口、样本消费符合 no-repeat/repeat 规则、模型切换安全；连续运行与中断恢复的必要状态/后续更新对照。保留的同步适配不能称为常驻异步对齐。真实 CUDA、AMP 和并发路径按启用配置验证。

### 批次 10：重新对照 EtaZero.md 全量审查

- [x] 固定完成后的目标 HEAD、工作区差异、配置和目标文件集合 hash；核对来源 commit，重新检查所有引用位置。
- [x] 从来源重新检查全部 57 项，包含此前判为一致的行，不能仅复制本计划的勾选结果。
- [x] 同时重新检查 E01—E26 工程条目，明确来源执行 profile、已对齐实现与保留的能力差异；源码结论与性能实测结论分别记录。
- [x] 每行分别判定：实现/公式、参数、边界分支、SP/learner/eval/Match 调用和运行证据；没有执行证据时保留“静态”限定。
- [x] 400/70 写为“用户明确保留的预算差异”；S23 写为“用户要求暂缓，未实现、未对齐”。批次0确认的编排/方棋盘/VCN/模型时点适配也单独列明；其他剩余差异不能并入用户例外。
- [x] 运行相关快速回归、独立来源对照和必要的小规模整链路检查；训练效果及正式实验不属于本次代码对齐验收。
- [x] 更新 `EtaZero.md`、版本 README 和职责文档，使其描述最终实际行为；本文件保留批次进度与验收依据。

验收：57 项算法清单与 26 项工程清单无漏项；所有实施范围内非豁免项达到各自明确的对齐标准。尚有未完成/未验证项时继续列明，不能宣称“全面对齐”。

## 57 项追踪表

此表记录实现现状和负责批次；批次0的57项最终目标/验收登记见[实施目标](EtaZero.md#alignment-items)。实施后逐项更新“状态 / 验收依据”，每项最终均进入批次 10；同一项涉及多个批次时，前置批次完成不代表整项完成。

| ID | 机制 | 负责批次 | 最终复审 / 验收依据 |
|---|---|---|---|
| S01 | FPU | 0、2 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S02 | PCR | 2、7 | 400/70为用户明确保留的预算差异；第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S03 | Forced playout | 2、5、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S04 | Policy target pruning | 2、5、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S05 | LCB | 2、5、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S06 | Tree reuse / 清树 | 2、7、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S07 | Reduce visits | 2、7 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S08 | PDA | 0、7 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S09 | Side position | 4、7 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S10 | Policy surprise | 0、7 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S11 | Value surprise | 0、7 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S12 | Weighted PUCT | 2、5、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S13 | Log-scaled cpuct | 2 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S14 | Variance-scaled cpuct | 2、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S15 | Value weighting | 2、5、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S16 | Root symmetry ensemble | 2 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S17 | 随机 D4 推理 | 2 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S18 | NN policy temperature | 2 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S19 | Root policy temperature | 2 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S20 | Chosen move temperature | 2 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S21 | Chosen prune / subtract | 2、5、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S22 | Shaped Dirichlet noise | 2 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S23 | Subtree value bias | 无；10 中复审 | 用户要求暂缓，未实现、未对齐；不计通过 |
| S24 | Virtual loss | 2、5、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S25 | Graph / transpositions | 5 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S26 | Uncertainty weighting | 4、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S27 | Optimistic policy | 4、6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| S28 | Noise pruning | 6 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_search_review.json)与EtaZero.md对应行 |
| D01 | 平衡开局 | 3 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| D02 | 禁手 dropout | 3 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| D03 | Policy init | 3 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| D04 | 混合棋盘尺寸 | 0、3、8 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| D05 | 混合规则 | 0、3 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N01 | Plain Convnet | 8 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N02 | Convnet + NBT | 4、8 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N03 | Transformer + NBT | 8 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N04 | 初始化 / fson / masked BN | 1、8 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N05 | Global pooling / heads | 4、8 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N06 | Learner D4 | 4、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N07 | SWA | 1、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N08 | Lookahead | 1、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N09 | Warmup | 1、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N10 | Optimizer / WD | 1、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| N11 | 裁剪 / 精度 | 1、2、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| L01 | Policy losses / targets | 1、4、8 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| L02 | Value losses / targets | 0、1、4、7 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| L03 | 搜索合法域 / 训练域 | 3、4、8 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| L04 | 重复次数 / loss 归约 | 1、4、7、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| R01 | MinRows / usable rows | 9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| R02 | TaperExponent | 9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| R03 | ExpandPerRow | 0、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| R04 | KeepTargetRows / shuffle | 9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| R05 | Replay ratio / 更新节奏 | 0E、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| A01 | 价值视角 / 终局 / utility | 0、3、4、7 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| A02 | 搜索预算 / 算力口径 | 0、2、5、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| A03 | Cache / 模型 / 朝向 | 2、4、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |
| A04 | Checkpoint / 恢复 / 发布 | 0E、0、1、4、8、9 | 第10批复审完成；实现/参数/边界/调用及执行、适配和静态限定见[逐项记录](EtaZero_V0/data/validation_batch10_final_20261002/algorithm_remaining_review.json)与EtaZero.md对应行 |

## 验证入口与记录

现有入口包括 `EtaZero_V0/tests/reference/check_reference_formulas.py`、`EtaZero_V0/tests/`、`EtaZero_V0/scripts/build.sh`、`EtaZero_V0/scripts/run.sh check-config` 和 `configs/smoke_test`。实际命令按受影响路径选择，不把未运行的检查标记为通过。

初始静态核查不能替代修改后的验收。当前已执行独立来源、CPU/C++以及真实CUDA、AMP、并发和恢复检查，具体范围见各批原始证据与第10批清单。

Python 默认使用 Conda `pytorch`。真实 CUDA 验证前遵循仓库约定：普通沙箱看不到 CUDA 时先做最小宿主级检查；宿主检查成功后在可访问设备的权限下运行，不改为 CPU 冒充通过。每批只做所需小规模验证，原始结果保存到独立运行目录。

每批完成时在对应任务旁补充：完成日期、实际修改入口、来源位置、验证命令与产物位置、结果和剩余限制。批次0E及0—10已完成；早期分批记录描述当时验证范围，最终现状以第10批逐项复审为准。


### 0E 的本轮检查记录（2026-10-02）

- `bash EtaZero_V0/scripts/run.sh check-config --config-dir EtaZero_V0/configs/baseline`：通过，只解析配置。输出保存在 `/tmp/etazero_0e_baseline_config.json`；确认400/70、1000×128、ratio=8、32局线程/1搜索线程/1推理服务、waves=1。eval/match通过各自`load_evaluation_config`解析，确认100v、不reuse、单根对称；没有启动模型或设备。
- `conda run -n pytorch python /tmp/etazero_0e_verify.py`：临时源码/文档检查通过，57个算法ID及26个工程ID无重复无遗漏，表列数正确，201个引用定义、文件与行号及本地链接有效；KataGo/KataGomo固定commit及干净工作区、46个已登记来源hash、50文件目标工程集合hash均匹配。临时脚本和输出位于`/tmp`，不是训练产物或仓库测试入口；持续验收入口仍以版本`scripts/`和`tests/`为准。
- `git diff --check`：通过；仓库改动仅`EtaZero.md`、`plan.md`、`EtaZero_V0/README.md`、`EtaZero_V0/docs/implementation.md`。`plan.md`原为未跟踪文件，继续保持未跟踪；未提交或推送。
- 限制：本批没有执行算法测试、独立抽样分布样例、CUDA/AMP/并发/中断或吞吐实测；这些验证已逐项路由到后续批次，不能用文档/配置通过替代运行验收。

### 0 的本轮检查记录（2026-10-02）

- 用户回复已落实到目标：保留自己的产样/固定训练量机制，不用KataGo桶；环境沿现有env，三规则可混训、VCN先不用；采用三个指定来源预设。方棋盘、同步固定模型/整轮提交及无bucket按该机制登记，未扩展研究实验。
- 来源modelconfigs实际加载核对三个预设：plain/NBT为v15/fson/Mish，Transformer为v17/fixup、C192/mid96、3 heads、FFN256、v2=64，缺省无Q；没有把卷积head宽度或损失版本强加到Transformer。
- 57项均登记commit绑定来源、目标入口、最终profile/环境映射、非默认分支、后续验收和负责批次；0E工程差异逐项交叉引用。WDL draw、TD有限轨迹、side长度1搜索值、误差方差/标准差、无score optimistic/uncertainty、PDA输入/预算均有明确公式。S23暂缓不计通过。
- `PYTHONDONTWRITEBYTECODE=1 conda run -n pytorch python /tmp/etazero_batch0_verify.py`：通过；57个目标登记ID与原清单逐项对应、每行6列、批次0六项定义任务均登记完成，来源预设版本/归一化/head宽度及目标anchor有效。用精确分数独立核对有限轨迹TD/WDL视角、PDA预算比例与总系数、uncertainty权重和loss系数手算样例；这些仅检查写出的数学契约，不是已实现算法运行测试。
- `PYTHONDONTWRITEBYTECODE=1 conda run -n pytorch python /tmp/etazero_0e_verify.py`：通过；原57项/26项编号和表列、218个引用定义/本地文件行号、固定来源commit/干净工作区和46项来源hash均有效；50文件工程目标集合hash仍为`3d56e54ed95b23ff8cf65a6a4c077dd8eec2edfd99290219fc96b976edc3778c`，代码与baseline未变。此前解析的当前baseline/eval/match仍作为现状，不把目标500v等写成已经生效。
- `git diff --check`及计划尾部空白检查：通过。只改`EtaZero.md`、`plan.md`、版本README和`docs/implementation.md`；`plan.md`继续未跟踪，没有提交/推送。
- 未启动训练，未执行CUDA/AMP/并发/中断或性能实测。临时检查脚本在`/tmp`，非训练产物或新增仓库测试；不把后续验收表当作本批已通过测试。

### 1 的本轮检查记录（2026-10-02）

- 目标为HEAD `185eee220fd786a6e7436ae4e2332b929904bdb0` 加本批未提交改动；固定来源commit未变。新增来源hash登记 `metrics_logging.py`，KataGo/KataGomo共47文件校验；当前26文件算法集合hash见 `EtaZero.md`，工程及修改文件hash、实际验证配置见独立验收manifest。
- 修正fson SGD cap2500、ordinary CE0.930和value有效CE0.72；训练heads/loss实际FP32，推理precision保持独立。LR/WD使用轮开始与5/50消费batch时点；norm在更新前采样并在WD刷新前进入统计，snapshot和all-batch运行均值/Lookahead筛选均落实。
- 删除FP16同batch重试，overflow消费后推进Lookahead/SWA。`step`/`total_steps`为消费batch，`total_samples`含skip，`optimizer_steps`为成功更新；新增状态、norm累积及reader消费cursor保存加载。日志保存skip与原始overflow范数，绘图loss覆盖所有消费batch，梯度均值只取成功更新并记录skip/有效batch数。
- `conda run -n pytorch python EtaZero_V0/tests/reference/check_reference_formulas.py`：replay252、LR/WD31104、SGD/AdamW cap各12、norm时序5760案例通过，相对容限1e-12/绝对1e-15。范数预期直接执行固定来源函数，不将来源误记为逐batch0.995 EMA。
- `conda run -n pytorch python -m pytest EtaZero_V0/tests -q`：132通过、53跳过；跳过是默认关闭的GPU项，真实CUDA另执行，不能计为通过。快速检查含八个warmup阈值、2亿/5/50/100边界、norm更新前快照、分段reset、skip后slow/SWA、状态恢复和日志聚合。
- 宿主CUDA检查通过：RTX5090、PyTorch2.12.0+cu132。使用宿主权限执行 `ETAZERO_GPU_TESTS=1 conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests/test_gpu.py -q -k 'real_amp_updates or training_autocast_heads_are_fp32 or amp_skip_resume_preserves or learner_resume_matches_uninterrupted or optimizer_checkpoint_d4_and_swa_export' --basetemp /tmp/etazero_batch1_cuda_20261002`：17通过，覆盖FP16/BF16、SGD/AdamW、eager/compile、所有head卷积/linear实际dtype、真实GradScaler overflow、成功更新与消费分计、SWA导出、底层learner续训。人工注入一次inf梯度及真实高scale overflow分别覆盖skip；连续/恢复model、optimizer、slow/SWA、scaler、reader及Torch/CUDA RNG逐位一致。
- 同权限执行 `ETAZERO_GPU_TESTS=1 conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests/test_gpu.py -q -k 'b5c192nbt_compiled_11x11_pipeline or compiled_training_resume or gpu_pipeline_and_idempotent_completed_resume' --basetemp /tmp/etazero_batch1_pipeline_20261002`：7通过。真实b5c192 NBT在11×11、FP32/FP16/BF16编译下完成两轮SP→shuffle→learner→导出校验/发布；固定样本compiled恢复与完成后幂等重启通过。仅使用8行batch/4消费batch短预算，未启动正式训练。
- 同权限执行 `ETAZERO_GPU_TESTS=1 conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests/test_gpu.py -q -k 'multiple_servers_fp16_packed_waves_and_prefetch' --basetemp /tmp/etazero_batch1_fp16_inference_20261002`：1通过；TorchScript/原生多服务独立FP16推理仍实际输出FP16，并覆盖packed数据、waves及预取。真实CUDA总计25项通过。
- 新数据、checkpoint及原始日志保存在上述独立 `/tmp` 验收目录；与实际配置/源码hash关联的验收manifest保存到 `EtaZero_V0/data/validation_batch1_20261002/`，不覆盖已有实验产物。当前正常训练入口仍按整轮重跑；本批底层learner续训检查不替代9c的controller中断全边界验收。
- 限制：后续搜索参数/随机D4/cache/reuse、辅助监督/量化、架构、replay和非默认分段编排未实现或未复审；未做吞吐对比、学习效果或正式实验。所有57+26项仍进入批次10，S23继续暂缓。
- `bash EtaZero_V0/scripts/run.sh check-config --config-dir EtaZero_V0/configs/baseline`、文档/来源/hash检查及 `git diff --check`：通过。57算法ID/26工程ID、218引用定义和本地文件/行号有效，47来源hash与固定commit/干净工作区匹配；50文件工程集合hash为 `32473a476ee6c34eae1126fffb8f48ff865d1220c28d04734a2b508e2b05aa9b`，26文件算法集合hash为 `695c7124bd456517390fa4e44b41358c97a22fb6cc32cf13f2a21b42b88bd1f3`。完整配置与未提交diff保存到独立验收目录；未提交/推送。

### 2 的本轮检查记录（2026-10-02）

- 固定KataGo/KataGomo commit不变，44+7来源文件SHA256及干净工作区通过。修改入口为baseline/smoke/minimal配置、config/eval_config、Search/AlphaZeroState/BatchEvaluator/TorchBackend及SP/Match/evaluate/serve调用点；补齐固定FPU权重、随机D4与miss-only独立NN RNG、canonical输入+globals+Tnn cache、根集成读写绕过、same/different bot分流及模型清树。
- 保留full/cheap400/70例外；baseline reduced min350独立，cpuct/log/stdev/value exponent及三种温度/噪声按目标profile落实；E/M baseline500v、SP推理FP16、E/M auto CUDA→FP16并记录。小资源profile明确覆盖预算/精度/最低访问数。
- 新增initial_visits与new_playouts，原simulations仍为新增根边模拟；fresh根初始化算1新playout，reused根重集成只耗NN请求；maxTime至少2新playouts、maxPlayouts可为0，显式停止可阻止全部搜索。五子棋终局无NN、不扩展Go forceNonTerminal；所有返回等待在途完成，异常释放pending。原有严格并行发放继续保留，与来源可能在途超限的差异明确登记，未宣称并行序列/等时间等价。
- `conda run --no-capture-output -n pytorch bash -c 'cmake --build EtaZero_V0/build -j2 && python EtaZero_V0/scripts/write_build_manifest.py EtaZero_V0/build && ctest --test-dir EtaZero_V0/build --output-on-failure'`：3/3通过；另有独立无Torch core构建与3/3验证。固定网络样例覆盖FPU两分支/零power、方差prior/低权重/二阶夹取、D4全画布与globals、cache首次canonical输出/温度条件/miss-only RNG/集成绕过、root priors重算、模型清树、预算/停止/终局计数及原有S03/S04/S05/S21/S24回归。
- `PYTHONPATH=EtaZero_V0/python conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests -q --junitxml=/tmp/etazero_batch2_cpu_final_results.xml`：133通过、57默认关闭GPU项跳过；跳过不计通过。
- 主机CUDA检查成功：RTX5090、PyTorch2.12.0+cu132。主机权限执行 `PYTHONPATH=EtaZero_V0/python ETAZERO_GPU_TESTS=1 conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests/test_gpu.py -q -k 'native_parity_all_sizes_rules_and_match or root_d4_probabilities or single_d4_root_only or match_same_bot_clear or real_cuda_parallel_leaf_failure or multiple_servers_fp16 or reduce_visits_pcr_native or native_selfplay_signal or eval_100_visits_and_arena_resume' --basetemp=/tmp/etazero_batch2_cuda_verified_20261002 --junitxml=/tmp/etazero_batch2_cuda_verified_results.xml`：11通过、46未选择，85.81秒。覆盖FP32/auto实际FP16的指定D4、时间/零playout公开接口、8根集成、同bot每手清树/异bot真实reuse、2server/4search线程Match、8search线程CUDA叶故障传播、FP16多服务/cache/waves/prefetch、PCR/Reduce及短训练恢复、SP信号收尾、比赛中断续测与Elo产物。
- 第一次GPU检查中新增测试包装加载顺序错误及旧raw诊断预填cache断言失败，已修正；失败XML与原始目录保留，最终检查11项全通过，未隐藏失败或覆盖旧运行。最终配置、源码快照、原始事件/比赛/故障日志在独立 `/tmp/etazero_batch2_cuda_verified_20261002/`，索引、结果XML、diff和来源/目标hash在 `EtaZero_V0/data/validation_batch2_20261002/`。
- 基础来源公式（replay252/LR-WD31104/cap24/norm5760）、baseline配置解析、文档/编号/引用/来源/hash和 `git diff --check` 通过；当前集合hash见EtaZero.md与验收manifest。未提交/推送；未做吞吐对比或学习效果实验。hint/fork、PDA、graph/uncertainty/optimistic/noise pruning及辅助heads仍按后续批次；S23保持暂缓，不计通过，全部57+26项最终复审留批次10。

### 3 的本轮检查记录（2026-10-02）

- 修改入口为OpeningConfig/initialize_opening、SP/Match调用、RecordWriter/schema/training_view、config/eval_config及baseline；固定KataGo/KataGomo commit不变，来源登记44+18文件。writer在repeats确定后按每最终输出行抽样，保存forbidden_input，完整轨迹及所有搜索/开局保留提示；view只应用已保存决定，不再随机化。输入/数据契约改变，验收使用新目录，未覆盖原始产物，未加兼容层。
- balance每次有效骨架后的尝试随机选参考botB/W，同一尝试根两视角/全部候选同bot；policy init每手按棋盘当前执色调用对应bot。Match的generator定义为参考黑方模型，参考执色一半A黑/一半B黑，同一开局交换A/B实际执色下两局，第二侧不重新生成；记录参考黑方及实际模型索引，明确与来源逐局独立初始化的协议差异。
- SP开关必填，loader mean12/T1，baseline按已确定GM scripts显式mean6/T1.6；Match开关默认false，开启时mean必填、T缺省1，after/failure显式消融默认true。保留已有手数补偿、0.0002均匀分支及终局policy前缀零监督；无Go面积/komi分支。
- 保留方形15/14/13/12/11与100/10/5/3/1，baseline规则1/0/0；混合三规则可配置。native整数解析完整消费，15x14不再被stoi静默解释成15；VCN/超画布明确报错。Renju禁手依旧可提交并判负，恰五优先，终局无NN、次方价值精确且backup变号；训练仅屏蔽padding，occupied/禁手仍在on-board softmax。
- `conda run --no-capture-output -n pytorch python EtaZero_V0/tests/reference/check_katagomo_rules.py`：136125个禁手/类别独立对照通过（none125563/overline4474/double_four3834/double_three2254），seed309，尺寸11—15。直接编译固定来源CForbiddenPointFinder，覆盖3^8邻域穷举、十格轴向穷举加交叉形状、随机密集/边缘、同方向双四、恰五/长连交叉及过长延伸假三；现有递归RenjuAnalyzer无需修改。有限语料不是全部递归棋盘的等价证明；矩形/VCN/pass未纳入。
- `conda run --no-capture-output -n pytorch bash -c 'cmake --build EtaZero_V0/build -j2 && python EtaZero_V0/scripts/write_build_manifest.py EtaZero_V0/build && ctest --test-dir EtaZero_V0/build --output-on-failure'`：3/3通过。新增五尺寸/三规则长连/恰五终局、禁手叶无NN/变号、双bot balance与逐手policy、已有手数补偿、T2手算赔率统计与0.0002均匀分支；原有权重/几何/拒绝/回退/取消及搜索回归通过。
- `PYTHONPATH=EtaZero_V0/python conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests -q --junitxml=/tmp/etazero_batch3_cpu_verified_results.xml`：135通过、59默认关闭GPU项跳过；跳过不计通过。含重复行不同augmentation、提示同步/padding与occupied/禁手梯度域、加载默认/显式参数及不支持环境的拒绝。
- 宿主最小CUDA检查成功：RTX5090、PyTorch2.12.0+cu132。主机权限执行 `PYTHONPATH=EtaZero_V0/python ETAZERO_GPU_TESTS=1 conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests/test_gpu.py -q -k 'mixed_rule_openings_and_training_feature_dropout or match_opening_uses_both_models_by_source_roles or target_five_sizes_three_rules_cuda_and_unsupported_requests or native_parity_all_sizes_rules_and_match or eval_100_visits_and_arena_resume or multiple_servers_fp16_packed_waves_and_prefetch or native_selfplay_signal' --basetemp=/tmp/etazero_batch3_cuda_verified_20261002 --junitxml=/tmp/etazero_batch3_cuda_verified_results.xml`：7通过、52未选择。覆盖五目标尺寸×三规则实际NBT/原生CUDA输出对照、dropout0/.5/1完整产样与重复行不同flag、黑白异模型实际TorchScript开局选择、FP16多服务/packed/waves/预取、两轮短SP→shuffle→learner→导出、SP信号收尾、成对比赛中断/扩局续测与Elo。
- 为稳定观察重复行，dropout检查显式使用已有有效surprise配置（full搜索、不reduce、policy混合1/value0），属于小规模机制验收，未改变baseline或新增研究实验。0/.5/1产样的完整轨迹、搜索target、频率/胜负逐项相同，只有持久化forbidden_input及metadata不同；真实重复Renju行有不同flag，全部搜索/开局完整提示。
- 首轮新增fixture的字符串替换误匹配已修正；具名同方向双四fixture也已修正后通过独立源对照。首轮GPU失败来自验收期间补C++测试使构建身份过期，以及原smoke权重/局数未生成重复Renju行；重建并使用上述明确次数条件后最终7项通过。全部失败XML/原始目录保留。原始目录在 `/tmp/etazero_batch3_cuda_20261002/`、`/tmp/etazero_batch3_dropout_cuda_20261002/`、`/tmp/etazero_batch3_cuda_verified_20261002/`；验收manifest、源码diff/配置/hash/规则结果/原始文件索引在 `EtaZero_V0/data/validation_batch3_20261002/`。
- baseline配置解析、基础来源公式（replay252/LR-WD31104/cap24/norm5760）、57/26编号/表列/引用/固定来源/hash及git diff --check通过。未提交或推送，未做吞吐/学习效果或正式训练；辅助heads、graph/uncertainty/PDA、架构/replay/完整controller恢复留后续4—10，S23继续按用户要求暂缓。


### 4 的 v15 链路检查记录（2026-10-02）

- 本批保持进行中：完成六policy、主WDL、三个TD和short-term value error；v17可选Q尚未实现，未勾选对应任务。后续5的图搜索可使用已验收v15输出；6的uncertainty/optimistic消费、7的side生成/重分析及8的架构与Q均未计通过。S23继续按用户要求暂缓。
- 来源固定commit未变。核查model_pytorch.py:135/2700/2843/4352、metrics_pytorch.py:129/308/697/740/872、trainingwrite.cpp:411/548/578以及play.cpp:815的真实分支。五子棋保持W/D/L、纯W−L，移除Go score、ownership/no-result专用监督；完整主局flag控制error/默认optimistic，source disable开关以0.5继续训练两个policy头。
- 修改network/TrainingForward、schema/data/reader、SearchResult/Step/RecordWriter、export/TorchBackend/BatchEvaluator、训练及绘图。TD按实际面积和固定黑方完整轨迹累加，三个CE−entropy各0.72；error使用来源squared-softplus forward和floor0.05 surrogate backward，预测量stop-gradient。所有十一loss分量记录有效系数，batch求和反向及重复行频率不重复乘权。
- policy保留归一化前最终选择权重，最大值至少10、超过30000缩小后C++ round写int16，learner归一化后soften。新契约同时覆盖side独立观测/搜索WDL、完整数据/opponent gate及逐输出行dropout；side value和三TD同自身搜索值，绝不借主局终局。实际SP侧分支尚未产生。
- `conda run --no-capture-output -n pytorch bash -c 'cmake --build EtaZero_V0/build -j2 && python EtaZero_V0/scripts/write_build_manifest.py EtaZero_V0/build && ctest --test-dir EtaZero_V0/build --output-on-failure'`：4/4通过；含手算量化边界及原生side writer。原生empty/full-side NPZ→validate→view→catalog→shuffle→reader分别通过。
- `PYTHONPATH=EtaZero_V0/python conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests -q --junitxml=/tmp/etazero_batch4_cpu_completed_results.xml`：151通过、61默认关闭GPU项跳过。覆盖独立有限轨迹TD、actual area/white视角/draw、error梯度floor、TD熵/梯度、预测detach、complete gate与disable开关、所有D4以及side存储往返；skip不计通过。
- 宿主最小CUDA检查成功：RTX5090、PyTorch2.12.0+cu132。主机权限运行 `ETAZERO_GPU_TESTS=1 ... pytest EtaZero_V0/tests/test_gpu.py -q -k 'auxiliary_outputs_train_export_native_cuda or training_autocast_heads_are_fp32 or compiled_training_resume or native_parity_all_sizes_rules_and_match or multiple_servers_fp16_packed_waves_and_prefetch or match_opening_uses_both_models_by_source_roles' --basetemp=/tmp/etazero_batch4_cuda_verified_20261002 --junitxml=/tmp/etazero_batch4_cuda_verified_results.xml`：11通过、49未选。六policy/TD/error真实更新、FP16/BF16 eager/compile FP32 heads、off/FP16/BF16 compiled精确恢复、两轮SP→shuffle→learner→导出、mixed size/rule原生aux输出、FP16多server/cache/waves/prefetch及双bot开局回归通过。
- 同权限运行 `... pytest EtaZero_V0/tests/test_gpu.py -q -k 'native_side_supervision_cuda_training_and_export or gpu_pipeline_and_idempotent_completed_resume' --basetemp=/tmp/etazero_batch4_side_cuda_verified_20261002 --junitxml=/tmp/etazero_batch4_side_cuda_verified_results.xml`：2通过、59未选。原生side shard的11行（含2侧行）实际CUDA消费22样本/2更新并保存、导出；完整轮恢复幂等及十一项loss总和/绘图通过。真实CUDA合计13不同项通过，仅验收小预算，无正式训练/效果/吞吐结论。
- 首轮CUDA发现空side数组使zlib crc32(null,0)重置已累加CRC；已修复零长度数组并加入原生empty-side回归。此前构建插入位置及测试shape/float容限/旧五loss断言也已修正；失败XML与独立运行目录保留，最终全部相关检查通过。没有兼容旧契约，没有覆盖旧数据/checkpoint或先前验收目录。
- baseline配置、来源公式、57/26编号/引用/固定来源hash和git diff --check通过。代码/实际配置、完整diff、新增源码、结果XML与运行文件索引保存到 `EtaZero_V0/data/validation_batch4_v15_20261002/`；batch4整批尚未完成，未提交/推送，目标保持全部计划实施。

### 5 的图搜索检查记录（2026-10-02）

- 修改 `SearchState/AlphaZeroState`、Search节点表/遍历/统计/根推进/回收、config/shared evaluation fields、native settings与评估诊断，SP/Eval/Match baseline默认graph=true、catch-up leak=0。来源固定commit不变；新增读取graphhash.cpp/.h，KataGo46+KataGomo18文件hash核对。关闭graph、leak0/0.4/1均有真实分支，没有用NN cache替代图统计。
- 完整局面字节包含尺寸/画布/执子/手数/棋规/终局与结果/棋子；哈希只选表分段。NOVC每手增加棋子，无Go提子/ko/pass/判重历史，来源graphSearchRepBound不适用。共享节点使用独立父边计数、共享virtual loss；下降前CAS追赶每次加一，不新增child visit；重复路径增加该边后结束counted playout。节点visits独立于子边导出的权重。
- 对照来源区分LCB的weightSq线性edge/child缩放与父聚合的平方缩放。根推进复制共享子节点作为独立根；静止后从根标记共享后继，每个未标记节点只删除一次，超过4096节点由已有线程并行删除。root条件不一致/reset/换模型清图；不支持key的通用adapter明确拒绝。保留已登记的严格并行预算/展开等待/锁快照差异，不宣称来源调度序列或等时间等价。
- `conda run --no-capture-output -n pytorch python EtaZero_V0/tests/reference/check_katago_graph.py --output /tmp/etazero_batch5_graph_reference.json`：通过。独立编译固定来源childWeight/childWeightSq、maybeCatchUpEdgeVisits和recomputeNodeStats聚合循环；200次diamond单步、1080个scalar比较、48次leak0追赶，节点/边访问、终局、各矩及泄漏1分支相符，另有线性/平方缩放手算样例。不是整个Go或并行执行等价证明。
- `cmake --build EtaZero_V0/build -j2 && ctest --test-dir EtaZero_V0/build --output-on-failure`（pytorch环境）：5/5通过，日志 `/tmp/etazero_batch5_graph_cpp_v2.log`。graph测试覆盖真实五子棋不同顺序到达、三规则、输入条件、终局共享、多个父、graph关闭、1/8线程、leak0/0.4/1、循环、根推进、reset/模型清图、并发故障pending清零和6000模拟大图回收。第一次cycle fixture误要求leak0必走cycle分支，已按来源先追赶的顺序纠正；失败日志保留。
- 独立无Torch构建 `/tmp/etazero_batch5_asan_build` 使用 `-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g`。普通沙箱在5项退出时均报LeakSanitizer不支持ptrace，未计为通过；宿主权限执行同一 `ctest` 后 **5/5通过**，含leak检查，日志 `/tmp/etazero_batch5_asan_host.log`。原始失败日志保留，没有关闭leak检查冒充通过。
- `PYTHONPATH=EtaZero_V0/python conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests -q --junitxml=/tmp/etazero_batch5_cpu_verified_results.xml`：152通过、63默认关闭GPU项跳过，skip不计通过。新增配置继承/env覆盖、leak边界及NaN/Inf拒绝；原有数据/训练/恢复等快速回归通过。
- 最小宿主CUDA检查成功：RTX5090、PyTorch2.12.0+cu132；CUDA命令直接使用宿主权限。初组 `... pytest tests/test_gpu.py -k 'graph_real_cuda_shared_nodes_leak_and_cache or gpu_pipeline_and_idempotent_completed_resume or match_same_bot_clear_and_different_bot_reuse or real_cuda_parallel_leaf_failure_releases_waiters or multiple_servers_fp16_packed_waves_and_prefetch or native_parity_all_sizes_rules_and_match'`：6通过，43.76秒；第二组 `-k 'graph_shared_cuda_failure_releases_all_parents or graph_real_cuda_shared_nodes_leak_and_cache or root_d4_probabilities_and_policy_temperatures or single_d4_root_only_and_inference_precision or native_selfplay_signal_flushes_complete_games'`：6通过，28.10秒，包含1项重复；**11个不同CUDA case**通过。真实SP→shuffle→learner→导出/幂等恢复、混合尺寸三规则、FP16多服务/4或8搜索线程、根D4/时间/零预算、同模型清图/不同模型复用、4手转置处CUDA异常及停止收尾均覆盖。
- 另外重跑具名graph CUDA检查，补充graph关闭且NN cache实际命中的分支：访问分布与关闭cache相同，节点仍独立；与graph共享明确区分。日志/XML和cases.json另存 `/tmp/etazero_batch5_cache_cuda_final_20261002`。500v合成局面中关闭cache时tree为500节点/501NN请求，graph为132节点/133NN请求、368追赶；这是机制验证，不是吞吐或棋力实验。
- baseline配置、基础来源公式、57算法/26工程编号、引用位置与本地链接、固定来源/hash及git diff --check通过。实际配置、源码快照/未提交diff、失败和成功日志/XML、原始CUDA文件索引保存在 `EtaZero_V0/data/validation_batch5_graph_20261002/`，不覆盖前批与旧训练产物。未提交/推送，未启动正式训练；批次4的v17/Q仍待4/8、6—10仍待实施，S23按用户要求暂缓且不计通过。

### 6 的搜索修正检查记录（2026-10-02）

- 三项任务完成：SearchMath/evaluate_node/recompute/terminal/advance、BatchEvaluator、Native backend/settings、Python profile和参数校验贯通 uncertainty、short optimistic logits 与 noise pruning。SP off/0/0/off；Eval/Match on/root0.2/leaf1/on。真实能力声明和输出一致，缺能力才weight1/optimism0；原契约不符模型拒绝。仍有批次4可选v17/Q及7—10，S23继续暂缓。
- `check_katago_search_corrections.py` 执行固定来源未改函数/混合语句：504 uncertainty＋240 noise＋48 float logits＝792例通过；Go score导数0。`check_katago_graph.py` 回归200 diamond步骤/1080 scalar/48追赶通过。NN cache继续精确输入，并以exact optimism分键，比来源1/1024离散更细；不称命中率、RNG、调度或性能等价。
- `cmake --build EtaZero_V0/build -j2`、`write_build_manifest.py` 与 `ctest --test-dir EtaZero_V0/build --output-on-failure`：6/6通过。新增手算非单位NN/终局权重、D4标准差平均后加权、float混合后温度、根晋升零新增预算刷新、无能力以及noise cap/根chosen-prune竞争。独立ASan/UBSan/LeakSanitizer构建，宿主6/6通过。
- `PYTHONPATH=EtaZero_V0/python conda run --no-capture-output -n pytorch python -m pytest EtaZero_V0/tests -q --junitxml=/tmp/etazero_batch6_cpu_verified_results.xml`：153通过、65默认GPU跳过，不计skip通过。baseline解析和来源公式回归通过，保留400/70、固定1000×128/ratio8及逐轮发布，不改变用户例外。
- 宿主CUDA权限运行，RTX5090/PyTorch2.12.0+cu132；`test_gpu.py` initial/verified/selfplay三个选择记录合计13个不同用例通过，含真实error0/.5/2和root optimism0/.2/1、真实终局二阶、500v/8线程/2server/8根D4图组合、E/M默认、SP三项显式开启、FP16 packed、多尺寸规则、真实两轮SP→shuffle→learner→export、根D4/cache、并发故障及SIGINT。
- 首轮auto FP16单D4一项失败保留：测试此前以eager为预期，实测eager与native差1.39e−4，同一发布TorchScript在Python与native差4.82e−13，四次稳定。对照改为实际发布模型，原容差未改，FP32/auto重跑通过；失败日志/模型/probe保留，不把失败计通过。
- 验收原始日志、XML、来源/配置/编译与源码hash、失败probe及CUDA产物索引集中于 [batch6证据](EtaZero_V0/data/validation_batch6_search_corrections_20261002/manifest.json)。不覆盖历史结果；无正式训练、学习效果或吞吐结论，所有改动尚未提交。

### 7a/7b 的 PDA / side / reanalysis 检查记录（2026-10-02，阶段快照）

- 本批保持进行中：PDA/side两项已实现验收；同时接入reanalysis和direct value surprise。hint/hintFork、early/game fork、相邻PCR例外及完整频率复审仍待完成，后两项保持未勾选。批次4可选v17/Q、8—10及S23用户暂缓状态不变。
- 主要入口为Game/schema六全局条件、search_limits、sampling、main SP、record/data及native配置。PDA普通局0.01/max8、PCR/reduced后乘倍数、min5、每手清图；图key与NNcache含条件。smoke因为tiny caps明确prob0。side0.020排除实际着、70/25/5混合候选、full搜索/LCB回复、0.25递归、PDA0；真实终局候选拒绝，side自己的WDL进入11项监督gate。
- reanalysis默认false；开启必须给出全部参数。cheap-only binomial数量＋surprise幂无放回选择，全零均匀，零比例不消耗选择RNG；使用该手之前的原搜索历史force-full及PDA，保留真实轨迹/终局，替换训练搜索target并重算surprise。原visits/选择值/目标开关单独保存；use-outcome=false仅关闭opponent，主局WDL/TD/error/optimistic仍按来源完整轨迹。未重分析cheap关闭excess，reduced已重分析仍可恢复。
- `check_katago_sampling.py` 编译固定来源原预算/输入及value-surprise函数：204 PDA有限visits/playouts/signed input＋216 direct/未来平滑WDL例通过。Go handicap/komi不适用；int32-max作为无额外playout上限不乘大数；来源RNG及并发序列不宣称相同。三搜索修正792例回归通过。
- C++及ASan/UBSan/LeakSanitizer各7/7通过。包含预算双方比例/取整、PDA与PCR/reduced组合、显式有限playout、min5非法预算、图条件、搜索深度执色翻号、side零条件/回复递归、cheap excess门控、原轨迹保留、原选择diagnostics、全零/关闭及零比例RNG。CPU最终155通过/67默认GPU跳过；新契约side writer/view/catalog/shuffle/reader及reanalysis outcome gate往返通过。
- 宿主RTX5090/PyTorch2.12.0+cu132，7个不同真实CUDA用例通过：direct开/关两组PDA/side/reanalysis实际SP→shuffle→2更新/16样本→save/export/native；每个主局/重分析NN WDL对真实条件、neutral及正负符号根对照、PDA全局投影两列实际更新；根D4/单D4 FP32与auto、图共享/cache、真实两轮闭环/完成恢复通过。失败fixture及配置入口调用记录保留；没有放宽原数值容差。
- 初始side对照错误地要求搜索WDL等于零条件NN概率，忽略真实终局backup，改为独立one-visit根对照及有效搜索统计；第一次独立evaluate调用遗漏network配置字段，改为实际native所需合并配置，并使用公开wdl字段。初期CPU写入测试与native重编译并行，旧二进制缺新raw字段；随后在构建完成后155项回归通过。所有失败XML/log/原shard保留；宿主ctest首次PATH缺失未执行测试，Conda环境下sanitizer正常通过。
- 配置、原始日志/XML、源码/构建hash、独立来源结果、失败及真实CUDA产物索引见 [7的分阶段证据](EtaZero_V0/data/validation_batch7_pda_side_reanalysis_20261002/manifest.json)。保留400/70、1000×128/ratio8及逐轮固定模型，未覆盖旧结果，未提交/推送；未做正式训练、棋力或吞吐实验。

### 7c 的 hint / game fork 与完整采样验收（2026-10-02）

- 第七批四项任务全部完成；整体计划继续实施，批次4可选v17/Q与8—10仍未完成，S23按用户要求暂缓且不计对齐。
- 接入hint可重放NOVC文本输入、文件hash/生效配置、独立起点kind/前缀计数及canvas坐标。普通hint概率0（无外部数据），baseline early0.04/late0.01、early位置比例0.025、候选3…12/36。hint/fork跳过平衡与policy init；fork不抽PDA。共享池随机取出并移除，跨同一worker轮次及模型释放/更换保留，进程重启重建；初始动作完整保留但不监督。
- 精确hint×4 full/finite-playouts、跳过PCR/reduce，再PDA；hint和hintFork后6手cheap减半。根先温度/噪声，再float 2% prior移给hint，按0.8其他最重边门槛强制探索；hint切换/移除清图。first hinted search WDL按正确视角复制next或terminal，并在重分析前后刷新；force-full跳过hint/PCR，reduce/PDA保留。
- early优先/late后备，指数或完整历史均匀取局面；无放回抽实际空点，用对手NN的纯W−L排序候选。终局起点/选中终局fork拒绝；未选hint产生非终局hintFork。NOVC采用可重放文本、纯WDL和实际可用候选数，不移植Go SGF/score/komi/seki或额外corpus训练权重；每局有效性保持默认1。EtaZero int32 cap拒绝hint溢出，无额外playout ceiling不乘大数；不承诺来源RNG/调度或重启逐位一致。
- 固定KataGo完整getSearchLimitsThisMove函数体3072组合通过：hint位置匹配/不匹配、6手边界、force-full、PCR/reduce优先级、双方PDA及finite-playouts。生产core库直接链接（未复制本地实现）的336重分配组合对照来源完整频率block和value KL：全零/低surprise、sum<1、cheap excess、未重分析cheap排除、已重分析reduced恢复及双mix通过；来源float与内部double序列化后的最大差为5.960464477539063e−8，容差rel3e−7/abs6e−8。预算组合除weight的abs6e−8外保持rel1e−12/abs1e−14。204 PDA输入/预算、216 WDL surprise、792修正及200diamond/1080图scalar回归通过。
- 最终C++ 7/7，宿主ASan/UBSan/LeakSanitizer 7/7；CPU 157通过/70默认GPU跳过。hint raw prefix/kind/action/PDA约束、配置范围/外部文件身份及完整读写/shuffle/reader通过；prefix无频率、主局surprise只取整一次、side频率1/自身WDL和有效性0，重分析outcome关闭仅影响opponent，均保持原gate。
- 最终宿主RTX5090/PyTorch2.12.0+cu132共10个不同CUDA case全部通过（40.10s）：PDA/side/reanalysis direct开关2组、early/late/hint真实SP→shuffle→2更新/16样本→checkpoint/export 3组；worker跨轮/释放/换模型fork保留；图/cache1组、root D4/分层温度1组、单D4 FP32/auto2组、两轮闭环/完成恢复1组。仅工程验证，没有正式训练、棋力或吞吐结论。
- 初始测试fixture错误均保留：CPU compact_search已重建dropout数组，测试再次截断导致shape错；GPU读取glob顺序当游戏顺序、将local hint index当canvas index。修正独立检查后157CPU及10CUDA通过，没有放宽原数值容差。source adapter初期重复函数签名/缺Loc与getOpp、KL占位符误命中和链接未指定实际zlib均保留原失败日志；完成shim/实际CMake zlib链接后来源3072+336对照通过。
- 最终配置、文档/来源集合hash、真实二进制/库hash、未提交diff与新增源码、失败和成功日志/XML、原始CUDA产物校验索引见 [第七批完整证据](EtaZero_V0/data/validation_batch7_sampling_20261002/manifest.json)。此前7a/7b证据保持原字节；400/70、1000×128/replay ratio8和逐轮固定模型保持用户预算，不提交/推送、不覆盖旧训练产物。

### 8a 的独立 plain 网络与 NBT 回归验收（2026-10-02）

- 8a完成，第八批保持进行中：Transformer+NBT及可选v17/Q尚未实施；每架构整体验收任务在8b完成前不勾选。9—10仍待实施，S23按用户要求暂缓且不计对齐。
- 配置增加必填 `network.architecture=nbt/plain`；plain严格选择b10c128-fson-mish/v15，C128/B10、mid128、gpool32在第5/8块、head32/v2=80。直接两层3×3残差，没有NBT外层瓶颈；按来源fson初始化/固定缩放/最终masked BN。共享完整v15监督、六参数组和TorchScript/native导出。三个现有profile仍显式选择NBT；不新增实验臂，不兼容旧配置或历史checkpoint。
- 新增 `tests/reference/check_katago_network.py`：直接导入固定commit未修改的KataGo真实完整模型，映射所有Eta参数与梯度，零填额外Go输入，移除pass/score/ownership等输出，W/L/no-result行选择映射到W/D/L。CPU和宿主CUDA分别覆盖plain128×10及NBT192×5，11/15混合棋盘、两次训练BN和一次eval、所有参数组和实际初始化调用；65个同seed截断初始化及fan_tensor例逐位相等。CPU最大前向差2.384185791015625e−7、最大梯度差3.0517578125e−5；CUDA最大前向差2.384185791015625e−7、最大梯度差1.52587890625e−5，均满足预设相对/绝对容差。生产代码不依赖来源目录；复核已登记的trainloop_helpers依赖hash，来源47KG+18GM均匹配且工作区干净。
- CPU最终162通过、75默认GPU跳过（不计通过）。真实宿主RTX5090/PyTorch2.12.0+cu132：初组9通过/1失败，修正验收条件后第二组6通过，共13不同case通过。包含plain真实11/15配置下两轮SP→shuffle→8更新/64样本、FP32/FP16/BF16联合编译、gpool/stem实际改变、SWA发布、完整轮幂等恢复及中断后的模型/reader/scaler逐位对照；两架构非平凡BN/TF32条件下FP32与FP16原生导出；NBT192×5三精度闭环及NBT编译恢复回归。
- 首次来源梯度对照错误包含二值几何mask的连续梯度（来源另有-5000 logit padding），只比较实际连续特征/全局量/参数后通过；mask不是训练变量，所有padding特征梯度为零。首次plain FP32恢复仅一个权重差3.637978807091713e−12，CUDA卷积反向在普通条件下非确定；改用与已有eager恢复相同的确定性CUDA条件，三精度均保持rtol=atol=0通过。未改生产执行设置，没有放宽容差；原失败日志/模型/原始目录全部保留。
- 配置、来源/目标hash、CPU/XML、成功和失败CUDA原始产物索引、未提交diff及新增来源对照脚本见 [8a证据](EtaZero_V0/data/validation_batch8_plain_20261002/manifest.json)。400/70、1000×128/ratio8与同步逐轮发布保持用户约定；不覆盖旧产物，不提交/推送，无正式训练、棋力或吞吐结论。

### 8b 的 bare Transformer 预设与三架构回归（2026-10-02）

- Transformer结构任务完成，第八批仍保留一项完整分支任务：可选v17逐动作Q尚未实现，与批次4对应任务保持未勾选；9—10仍待实施，S23按用户要求暂缓且不计对齐。baseline仍选择NBT，400/70、1000×128/replay ratio8、15/14/13/12/11分布及逐轮固定模型保持用户约定。
- `architecture=transformer`严格选择bare b5c192h3nbttfrs/v17：C192/B5、mid96、3×32 attention、FFN256、theta100固定2D RoPE、RMSNorm eps1e−6、SwiGLU、四内残差；fixup/ReLU、外块零出口、无最终BN，head32/value-hidden64。缺省无Q时沿来源六policy/WDL/TD/error及相应loss，不冒用fson/Mish预设。attention七参数组和normal_attn_wd_factor接入SGD/AdamW。
- 原始来源SwiGLU Triton前向、重算反向及AMP舍入完整保留，仅改custom-op namespace并注明MIT来源；来源文件登记48 KataGo＋18 KataGomo共66个。独立对照使用来源完整Model的公开NCHW/SDPA路径，仅将输入投影适配为5空间/6全局并映射非Go输出；不改写其attention/RMSNorm/RoPE/FFN函数。170个同种子初始化、所有134映射片段的初始化尺度/参数角色、11/15混合尺寸两次train/一次eval前向与全部特征/参数梯度通过。
- CPU全套167通过、81默认GPU跳过（不计通过）。宿主RTX5090/PyTorch2.12.0+cu132真实CUDA：三架构16个不同case全部通过，包括Transformer SGD三精度及AdamW FP16的两轮SP→shuffle→8消费batch/64样本、SWA/native发布、幂等恢复及确定性条件下模型/reader/scaler逐位中断对照；三架构非平凡归一化或非零attention/RMSNorm权重下TF32的FP32/FP16原生导出。FP16两配置均消费8批，其中2次overflow正常跳过、6次成功更新，不将跳步计为更新。
- 初次BF16闭环第四批的编译梯度异常及非零attention下优化后JIT导出误差已修复，失败日志/XML与原始目录保留。Transformer full-graph编译关闭自动channels-last改写以保留选定NCHW布局；eager和同布局编译来源对照分别验证。RoPE乘加、SiLU/gating保持中间舍入，通过原导出容差。不是CPU/精度回退、调LR/初始化或放宽容差。
- 三精度eager及同条件编译的六项来源对照全部通过。额外的compiled-Eta对eager-source FP16检查曾失败，保留原结果；核对原来源自身的编译舍入后，同条件编译的FP16/BF16前向最大差为0，全部梯度沿原先声明的容差通过。两个实现自身编译/eager前向差完全一致，FP16约7.86e−4/9.20e−4，BF16约6.50e−3/5.88e−3。FP32同条件编译沿原FP32容差通过；不宣称跨执行方式逐位一致。SGD/AdamW fson31104组、fixup36288组来源LR/WD公式通过，24个裁剪及5760范数时序对照仍通过。
- 执行边界：使用来源可选NCHW/SDPA及显式mask，源默认NHWC/flex/skip-mask性能不同；未做吞吐、棋力或正式训练。可选Q仍不支持，不宣称整个第八批或全部计划已完成；三架构已有路径回归并不能代替9的全状态消费/恢复审查。最终配置、源码/未提交diff、失败与成功记录、运行产物SHA索引及旧证据校验见 [8b验收manifest](EtaZero_V0/data/validation_batch8_transformer_20261002/manifest.json)。未提交/推送，不覆盖8a/7/6证据与旧训练产物。

### 8c 的可选 v17 纯 W−L Q 与完整架构验收（2026-10-02）

- 批次4剩余Q任务及批次8完整架构任务完成，9—10仍待实施，S23按用户要求暂缓，未实现、未对齐且不计通过。baseline仍选择NBT且Q关闭；400/70、1000×128/replay ratio8、尺寸/规则分布和逐轮固定模型保持用户约定。
- `network.predict_q_values`必填bool，仅Transformer v17可启用；v15拒绝Q，v17关闭时六policy不变，开启时添加第七个纯W−L pre-tanh输出。Go score Q不适用。Q为训练专用，不交给native搜索替代子树统计。主局、side及reanalysis从全部已分配child的正visits/weight提取node统计，按edge转换为当前玩家视角；终局无NN READY也有效。共享子节点及一步胜/禁手败的视角均独立检查。
- writer使用float32 Q×32000并限于±32000，node visits限于0…32000；Q值在每个最终重复输出行独立随机量化，访问数按采样位置压缩。两类Q目标贯通主局/side存储、D4、shuffle、reader、预取、训练、保存/恢复及导出。独立每局种子流不影响对局/禁手增强；删除Go score及Rand后不宣称随机序列相同。Qloss采用1.5×sqrt(visits)权重的BCE(2×pre-tanh,(1+Q)/2)/(1+sum sqrt(visits))，无数据行梯度0、side主局gate0仍监督，不重复乘采样频率。
- CPU全套180通过，89个默认GPU跳过不计通过；C++及宿主ASan/UBSan/LeakSanitizer各7/7通过。来源未修改整数量化函数与实际core库46140例逐项相等；来源Metrics的Qloss/全部梯度与独立解析计算通过。64个重复主/侧输出含不同量化值；float32乘法顺序按来源修正后保持零容差梯度对照。
- 原始完整来源Q网络在CPU及宿主RTX5090/PyTorch2.12.0+cu132的三精度eager/同条件full-graph六组前向与全参数/输入梯度对照均通过；170个初始化样例、134个参数映射片段，Q模型参数数1332941。保持原8b数值容差与NCHW/AMP边界，未降精度或回退CPU。
- 29个不同实际CUDA用例全部通过。首组23通过/6测试NameError；修正plain/NBT中误放的Q断言后复验10通过，合并为29不同case。包括三架构、Transformer Q开关、SGD三精度/AdamW FP16、真实11/15两轮SP→shuffle→8消费batch/64样本、Q头实际更新及正Qloss日志、SWA/native发布、完整轮幂等恢复和确定性条件下中断模型/reader/scaler逐位恢复；四种架构/Q组合的非平凡归一化/attention权重下FP32/FP16原生导出；PDA/side/reanalysis四组及hint/fork三组回归。AMP overflow消费与成功更新分别记录。
- 初始plot缩进、原Q乘法顺序、终局READY过滤、共享图测试范围、native fixture字段拼写及GPU测试NameError均修正并保留原失败记录。额外检查错误假定刚清树的node/edge visits相等；四线程catch-up与在途回传实际产生8个不同计数项，取消该错误等式，保留真实child node目标，不修改生产统计或放宽容差。来源6/7/8a/8b证据原字节校验通过，旧native二进制/库/构建清单在新证据目录保留。
- 配置、来源与当前目标hash、未提交diff/新增源码、失败和成功日志/XML、构建/二进制hash及全部原始CUDA产物SHA索引见 [8c验收manifest](EtaZero_V0/data/validation_batch8_q_20261002/manifest.json)。原始产物不覆盖，不提交/推送，无正式训练、棋力或吞吐结论。全计划仍进行中，9的全状态/消费与10最终复审不能由架构验收替代。

### 批次9a Replay、文件消费与validation验收

- random行只在窗口usable累计中封顶m，真实产样和quota保持独立；基准m250000、p.65、a.4、K20M/all，三扩展真实生效。近期按实际mtime、完整末片；源range end是raw total+int(offset)。来源CLI、shuffle.sh和同步脚本覆盖分别登记，未引入训练桶。
- 组累计训练rows到≥阈值才封组，保留round(nq)均匀子集；源计划固定B/F、桶内floor等分，可以有空文件和小文件。取消overflow递归，实际桶超数组预算明确失败。shuffle的partition/stage/wave/group/bucket命名空间独立，1/3wave只采样一次，保留字段配对与可重建hash。
- reader每文件完整batch前缀，尾部丢弃；没有跨文件/跨pass补batch。默认repeat采用source reservoir间隔；no-repeat耗尽StopIteration，耗尽游标恢复仍耗尽；不足固定轮预算在learner更新前拒绝。当前仅固定单轮快照，来源异步目录插入与轮询不计为已实现。
- 默认skip_validation对应sync SKIP_VALIDATE。启用MD5原始basename1%留出，两个分区共用切分前q；跨快照稳定隔离，验证payload参与原子发布/回收/恢复。writer生成独立OS随机64位hex basename，不改变game RNG。训练轮末raw eval/no_grad验证，随机D4独立于训练开关，支持AMP、compiled eval、文件顺序选择和完整batch后超过cap再停。验证不更新learner/BN/训练计数；无完整验证batch明确记录0样本。
- 固定来源原函数2592窗口、30group、324输出计划、10000MD5案例通过；原TrainingDataGenerator264文件顺序及RNG状态一致，原Go NPZ reader68个batch逐行一致。CPU187通过/94专用CUDA跳过；新分区随机流唯一性与holdout重建7项通过；C++7/7；12项不同真实CUDA用例通过，覆盖NBT FP32/FP16/BF16与v17可选Q compiled BF16验证、no-repeat不足、实际多轮SP/shuffle/learner/发布、FP16多服务/wave/prefetch及AMP跳步和连续/恢复对照。没有吞吐或棋力结论。
- 首轮旧断言/fixture依赖跨文件凑batch、输出名按行offset导致空文件重名、测试漏save_json、微型K12/waves3无法供给batch8，均定位并修正；全部失败记录保留。最后跨train/val随机流重复已用partition命名空间修正并重验。当前新checkpoint/目录只采用新配置和reader模式，不为历史配置增加兼容层。
- 配置/源码/来源hash、完整diff、新增文件、成功和失败XML/日志、构建与真实CUDA产物SHA索引见 [9a证据](EtaZero_V0/data/validation_batch9_replay_20261002/manifest.json)。旧6/7/8证据不改字节；400/70、1000×128/ratio8及每轮固定模型保持用户约定。9b/9c与10仍待实施，S23保持用户暂缓、不计通过；未提交/推送、未正式训练。

### 批次9b 固定产样quota、训练分段与模型切换验收

- 9b四项完成；9c/10仍待实施，S23按用户要求暂缓、未实现/未对齐且不计通过。基准仍400/70、1000×128消费batch、ratio8、SWA period0解析64000样本。无train bucket、raw/usable补桶、epoch预扣、异步或局内轮询切换。
- 修复产样只补min_rows而未补正常轮累计目标的问题：每次实际完整局结束后继续核对target_rows，短产样持续补局，正常超额结转；冷启动实际raw锚点单独提交。无均值/零有效产样明确失败。CPU反例与真实CUDA三轮故意缩短首请求验证，不把估计局数当实际行数。
- 9a遗漏的真实身份已纠正：原生冷启动metadata是random:seed，原helper仅匹配random导致真实路径未封顶；现在按random:前缀判断，仅窗口usable封顶m，raw quota不封顶。真实SP/shuffle检查及独立窗口/主局+side计数通过。旧9a归档保持原字节，不宣称它验证过这个真实前缀。
- sub_epochs默认1，非默认固定budget按floor等分且各段非空；段入口只重置Lookahead计数，保留fast/slow与SWA，LR/WD和norm打印沿整轮时钟，轮末validation/slow拷贝只一次。保存分段位置、消费、成功更新、scaler、RNG、reader、norm及所有优化时钟。固定来源train.py未修改AST块3696步scalar对照通过，含skip、2亿边界、分段reset与SWA。来源概率文件预算不同，本地轮末counter立即归零、来源到下一段归零，适配明确登记。
- 每轮input_model固定，全部SP完成后release握手才进入shuffle/learner，记录PID/模型/释放顺序。真实三轮同PID和固定模型成立。同模型ID但不同实际路径也重建evaluator/cache；原生CUDA协议A→B→release→B验证输出，防止旧权重或cache串用。
- CPU192通过/99专用CUDA默认跳过不计通过，C++7/7；真实宿主RTX5090/PyTorch2.12.0+cu132共20个不同CUDA case通过，含四组eager/compiled三分段AMP实际skip及段内/边界中断后全部状态逐位一致、多worker FP16/wave/prefetch、validation四架构/Q分支及固定quota/cache重载。独立replay252、fson31104/fixup36288、cap24/norm5760公式对照通过。未重跑sanitizer，不将历史结果算本批通过；无棋力/吞吐/正式训练结论。
- 初始scalar oracle漏logging、CPU假定每次补局数固定、旧CUDA断言漏side rows，以及测试将metadata字节数组误作dict，均修正并保留失败日志/XML/原始目录。没有放宽数值或逐位恢复容差。配置/源码/来源hash、diff、新增源码、构建二进制/库、全部检查及原始运行SHA索引见 [9b证据](EtaZero_V0/data/validation_batch9_schedule_20261002/manifest.json)。旧6/7/8/9a证据原字节校验；未提交/推送，不覆盖旧产物。

### 批次9c 保存、发布及整轮事务恢复验收

- 9c四项及整个第九批完成；10全量57+26复审仍待实施，S23按用户要求暂缓、未实现/未对齐且不计通过。每轮同步固定模型/预算、无训练bucket、无异步/局内换网、400/70与1000×128/ratio8保持用户约定。
- checkpoint实际保存model/optimizer/scaler、Lookahead fast/slow/counter、SWA权重/buffers/累积、norm快照或和/权重、round/subepoch位置、消费/成功更新、四类RNG、文件内已消费cursor、配置/源码/父链/update身份。独立来源未修改save函数四分支实际执行与本地真实CPU checkpoint字段往返通过；来源保存model/optimizer/metrics/running_metrics/train_state/val/config/SWA，没有捕获全局RNG、scaler或局部Lookahead cache/counter。来源train_state持有SWA累积和文件使用状态，不能冒称来源也具备Eta全部恢复能力。
- 恢复/发布前检查已提交checkpoint SHA、模型manifest及路径/contract/canvas/权重SHA/checkpoint身份；真实三种损坏产物均在worker启动前拒绝。SWA优先、未采样raw保持，native/JIT验证后完整目录rename，再在整轮state提交之后更新current指针。正确current.json恢复保持原发布时间与字节，缺失/陈旧/损坏指针以state重建。恢复归档遗留私有export stage，不把暂存当可消费模型。
- 真实CUDA九处注入中断通过：SP完成、shuffle完成、checkpoint payload写完尚未有sidecar、learner指针提交、export暂存fsync、export目录rename、轮末metrics写完、state提交后、publication指针后。提交前从已提交轮末重做整轮并归档完整原始产物；提交后不重复消费/产样/quota，不重复有效指标。验证下一轮总8消费batch/64样本、冷启动anchor不重置、actual quota、catalog仅统计活跃raw、history唯一0/1/2及完成后幂等重启。独立learner段内精确恢复沿9b验证，不混称控制器续训中间checkpoint。
- CPU198通过/112禁用CUDA默认跳过不计通过，含多进程控制器锁、不可覆盖payload竞争、并发reader只见完整pointer/payload、归档rename中断可重试与发布时间保持。真实宿主RTX5090/PyTorch2.12.0+cu132共18不同CUDA case通过，含以上九边界/三腐坏，以及原shuffle/export/after_publish故障、训练SIGKILL、native SIGINT/持久worker退出。C++构建和7/7结果复用9b未变的native源码/二进制证据，不称本批新跑sanitizer。没有正式训练、棋力或吞吐结论。
- 初期source oracle错用本地字段名、边界fixture quota硬编码4而smoke实际128、假定已提交中间payload不会按配置回收均已纠正。边界测试还发现并修复重复恢复重写current指针而丢utc的问题；失败日志/XML及每次原始目录均保留，数值/逐位容差未放宽。配置/源码/来源hash、diff、新增源码、二进制/库、检查与原始运行SHA索引见 [9c证据](EtaZero_V0/data/validation_batch9_recovery_20261002/manifest.json)。旧6/7/8/9a/9b归档原字节校验；未提交/推送，不覆盖旧训练产物。


### 10 的最终检查记录（2026-10-02）

- 全部57算法及26工程条目重新核查实现、参数、边界、调用与证据。400/70为“用户明确保留的预算差异”；S23为“用户要求暂缓，未实现、未对齐”，不计通过；同步固定轮/无训练bucket、方棋盘、NOVC及网络预设另行登记。
- 修复BatchEvaluator空backend在检查前解引用，以及完整零训练行对局被产样控制器误判为失败；训练配额继续按实际完整对局补足。真实CUDA测试同步更新联合forward监测、随机文件名、量化策略权重及每文件完整batch的验收条件。
- CPU 199通过/112专用GPU跳过；C++7/7；专用真实CUDA112/112通过。首轮CUDA93通过/19失败日志及全部原始产物保留；原函数时钟首次命令遗漏必需参数的失败也保留，修正命令后3696案例通过。
- 独立来源图搜索、搜索修正、采样频率/分支、fork、Replay、Q量化、learner时钟/持久化、基础公式、KataGomo有限棋规语料重新执行。三架构来源完整网络CPU/CUDA对照，以及Transformer Q三精度eager/compiled六组CUDA对照通过；容差及source execution gaps按各结果原样记录。
- 来源固定commit/69文件hash、目标HEAD/工作区diff/配置/集合hash、引用位置、旧6—9证据字节与原始运行SHA索引见[完整证据](EtaZero_V0/data/validation_batch10_final_20261002/manifest.json)。文档及git diff --check通过。未正式训练、未验证棋力或来源端到端吞吐，不宣称全部并发序列/故障组合/部署能力等价；未提交/推送。
