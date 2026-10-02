# 运行框架与 AlphaZero 实现

当前实现第一轮 AlphaZero 训练框架，算法语义见 [algorithms.md](algorithms.md)。MuZero / Gumbel 尚未实现，配置检查会拒绝这些选择。

## 参考来源与边界

主要并行架构与训练数据管线参考 KataGo：对局线程共享 evaluator、多推理服务、NN 缓存、原子搜索统计与 mutex pool、有界后台写盘、二进制观测打包、NPZ 文件边界、power-law 回放窗口、随机分桶再桶内洗牌与 multi-wave，以及训练文件预取。源码提交、入口和校验值见 [reference_sources.json](../reference_sources.json)，许可证与改编范围见 [THIRD_PARTY.md](../THIRD_PARTY.md)。平衡开局直接参考 KataGomo；冷启动记账、Renju 与 TorchScript / LibTorch 边界另参考现有 MuZero_V2。

当前采用逐轮编排：固定训练量，按 replay ratio 规划 selfplay。KataGo 的同步脚本同样顺序执行阶段，但 learner 使用额度桶、no-repeat-files 和 epoch/subepoch；当前固定轮协议具有独立的repeat/no-repeat文件消费控制，不引入来源的训练额度桶，也没有常驻异步阶段推进或局内换网。已实现工程机制、确认差异及验证范围见 [E01—E26 工程审查](../../EtaZero.md#engineering-audit) 与 [实施计划](../../plan.md)。Go 专有监督的映射和辅助 heads 见算法文档，不能由执行架构推定支持范围。共享架构不表示已有相同的吞吐、后端优化或训练效果。

## 执行架构

```mermaid
flowchart LR
    C[保存轮次计划] --> G[C++ 多局并行搜索]
    G --> Q[共享推理队列]
    Q --> I[LibTorch GPU 批量推理]
    I --> G
    G --> W[有界队列与后台 writer]
    W --> R[完整对局 NPZ]
    R --> H[窗口选择与两阶段 shuffle]
    H --> D[不可变训练快照]
    D --> T[固定训练步数]
    T --> K[完整 checkpoint]
    K --> E[导出与 Python/C++ 校验]
    E --> M[原子发布模型]
    M --> C
```

### 对局、搜索与推理

[原生命令](../cpp/src/commands/main.cpp)为每个 worker 建立一个模型服务和多个对局线程。每局独立持有状态、搜索图和随机流，对局共享 evaluator。[Search](../cpp/src/search/search.cpp) 为每局保留搜索线程，逐手派发任务；单线程使用同一实现。

节点展开由共享 mutex / condition-variable pool 协调，以 release/acquire 发布合法动作先验与 child 统计。合法动作全部保留，child 统计按 8、64、全部合法动作三档按需增长；选择扫描已激活 child，并通过排序先验求出未访问动作的最大分数和全部并列候选，不裁剪动作域。节点累计访问与在途数直接参与 PUCT，避免每次扫描全部合法动作求和。每个搜索线程复用状态、路径和候选暂存，AlphaZero 状态按原位复制重置。边分别保存完成访问和在途预约，图节点统计在多父之间共享；virtual loss使用共享子节点在途数，仅参与调度。回传或来源定义的访问追赶完成后增加正式访问。失败释放全部预约，搜索返回前等待该手全部任务结束，访问／playout 严格发放预算；时间或显式停止时允许少于额度返回，已在途路径完成后释放所有 pending。

对局线程跨对局复用 Search 和搜索线程，重置图与随机流。节点表拥有全部节点，根推进时复制统计及边并保留共享后继。停止所有搜索线程后标记根可达节点，未标记节点各自移交一个删除链表；大图使用已有搜索线程并行回收。标记阶段使用可达集合与队列，析构只删除本节点，不递归访问共享边。并发选择可能读取不同时间的原子统计并改变搜索顺序；单线程 PUCT 与动作并列时的抽样定义不变。

[BatchEvaluator](../cpp/src/inference/batcher.cpp) 的请求具有唯一身份、模型身份和固定输入形状。入队采用 KataGo `forcePush` 语义，不因超过名义 `queue_capacity` 阻塞；调用者入队后等待自己的推理结果。每个 evaluator 固定绑定模型、画布和推理精度，多个服务共同消费队列。服务数由 `inference.server_threads` 指定，设备由独立 worker 的 `devices.selfplay` 项确定。默认立即取队列最多 N 项；非零 `batch_wait_us` 可显式增加等待期限，属于性能和调度条件。关闭后拒绝新请求并排空已入队请求；失败唤醒当前批和队列内调用者，后续调用直接报错。

[TorchBackend](../cpp/src/inference/torch_backend.cpp) 在各自服务线程绑定设备并建立独立 CUDA stream 与 pinned host 输入缓冲区，模型加载一次并共享只读权重，批量前向后一次回传 policy logits / WDL 概率；检查模型契约、画布、输出形状和数值。LibTorch 来自训练使用的同一 PyTorch 环境。多个服务共享模型权重，各自的输入缓冲与工作区仍需要额外显存；同卡并发不保证吞吐更高。

同步 evaluate 调用期间，调用者持有原始观测与 D4 变换后的暂存直到结果返回；线程复用请求、等待条件与 cache key 缓冲，服务线程复用 batch 暂存。CPU 上变换空间输入，再填入 pinned 缓冲；输出还原 canonical 坐标后缓存。结果一次回传 FP32 后转换到公共搜索数值类型。

SP 的 `inference.inference_precision` 明确选择 FP32 或 FP16 autocast，baseline 使用 FP16。eval/match 支持 `auto`，当前 LibTorch 实现按明确指定的 CUDA 设备解析为 FP16、CPU 设备解析为 FP32；原生结果记录实际精度，显式 FP32 override 保留。FP16 仅适用于 CUDA，归一化仍显式累积 FP32，输出 head 必须实际使用 FP16；训练 AMP 由另一项配置独立决定。导出文件保存 FP32 权重，每轮导出不执行推理数值对照；数值正确性由独立测试验证。该后端仍使用 TorchScript / LibTorch，没有移植 KataGo 的专用 CUDA/cuDNN 算子后端。

单朝向默认随机 D4，cache 命中保留首次输出且不推进朝向 RNG；指定朝向覆盖未命中请求的默认随机行为；已有 cache 仍复用首次输出，绕 cache 才强制实际朝向。根多对称请求绕过缓存读写；raw 诊断指定 identity 并绕缓存。服务的朝向 RNG 独立于搜索与落子 RNG，共享服务请求次序仍依赖线程调度。

NN 缓存采用有界直接映射表与分段锁，保存不可变原始主/短期 optimistic logits、WDL 概率及误差标准差。键比较变换前五个二进制空间平面的全部位、六个全局浮点特征的原始字节、NN policy 温度及有效 optimism，朝向不入键，包含棋盘尺寸、Renju 执色、棋规和禁手特征开关；哈希只用于选槽，碰撞不会返回其他输入的结果。模型与精度由 evaluator 生命周期隔离。`inference.cache_entries = 0` 显式关闭缓存，空间输入必须为二进制，全局特征允许 Renju 黑方的 -1。缓存不合并同时在途的相同请求，也不缓存根噪声或搜索策略。optimism 使用精确 double 字节分键，无辅助能力时归零；根/叶条件刷新及标准差到统计权重的转换见 [搜索修正](algorithms.md#误差加权optimistic-policy-与-noise-pruning)。Backend 声明的辅助能力须与实际输出一致，多服务必须使用同一能力契约。原生 evaluate 同时输出 network_sample_weight、network_value_stdev、search_weight 和 search_weight_sq，用于区分访问数和加权统计。

公共搜索遍历通过 [SearchState](../cpp/include/etazero/algorithm.h) 获得状态复制、动作域、转移、叶评估、终局值、奖励、折扣与视角转换。AlphaZeroState 提供真实棋盘适配；公共遍历不自行调用五子棋规则。支持图共享的状态还须提供包含 continuation 与 NN 输入条件的完整 `graph_key`；不支持时明确拒绝。未来 latent 状态可从同一边界接入，当前没有 MuZero 占位类。

### 轮次与设备

[Python 控制器](../python/etazero/runtime.py)依次执行 selfplay → shuffle → train → export。阶段内部保留对局并行、树内并行、后台 writer、shuffle worker 和文件预取，Shell 只负责薄启动。

bootstrap 为 iteration 0，只执行 selfplay，不 shuffle、不训练、不导出初始网络。iteration 1 补足有效训练行并完成首轮训练发布；后续四个阶段合为一次 `iteration`，完成模型发布与状态提交后才增加迭代编号。`.internal/state.json` 的 `iteration` 从 0 开始，表示当前未完成或下一次待处理的迭代；初始化 checkpoint 的迭代编号为 0。`.internal/iterations/<编号>/` 保存计划、阶段状态和 learner 进度，原始分片保存到 `selfplay/iteration_<编号>/`。checkpoint 的 `step` 是迭代内消费的 batch 数，`total_steps` 跨迭代累计消费；`total_samples` 包含 AMP 跳步消费，`optimizer_steps` 单独累计成功更新。`run.max_iteration` 是整个运行的累计完成目标，0 表示无限迭代；`--iterations` 可临时覆盖目标，时间限制与停止信号仍有效。

`devices.selfplay` 每个列表项启动一个独立 worker，记录设备、worker 身份和种子；可以指定多个 GPU，或同一 GPU 上的多个进程。learner 使用 `devices.train`，当前为单卡训练。没有 DDP 或跨机器调度。

worker 随机流从运行种子、iteration、持久化 attempt 序号和 worker 身份派生，产物 UUID 只负责文件身份。同一配置的独立运行具有一致的 worker 种子，重试使用新序号。实际每局种子随轨迹保存，树内并行调度仍可能影响访问顺序。

每轮使用计划中固定的输入模型，新模型发布后，下一轮才换代。[常驻 worker](../python/etazero/native.py) 按请求处理 bootstrap 与补局。随机阶段使用 CPU RandomBackend 经过共享队列与正常搜索，无模型前向或 CUDA 推理分配；每次 attempt 的随机输出由其种子和完整观测决定，不依赖服务线程调度。网络阶段在同一生产阶段复用模型服务和 CUDA 上下文；每次请求仍使用独立 attempt 目录和记录。在进入 shuffle / train 前排空推理、释放 selfplay 模型与 CUDA allocator 缓存，worker 进程保持存活，下一轮明确加载新模型。正常停止阻止新局并取消半局，在途搜索收尾后 writer 排空已完成对局；半局不会写成和棋。控制器退出关闭 worker 输入并等待其退出。

常驻进程仍保留 CUDA 上下文的固定显存开销；训练显存预算须计入该开销。模型权重、推理输入和 allocator 缓存随 release 请求释放。

### Selfplay 并行参数短测

[benchmark_selfplay.py](../scripts/benchmark_selfplay.py) 使用指定配置和固定网络权重逐项测量完整 selfplay 请求，包括平衡开局、主局、side 搜索与后台 writer 排空。每组参数使用独立常驻 worker，先预热，再以相同局数和种子序列测量；NN cache 与 fork pool 按生产 worker 的生命周期保留。模型加载和首次 JIT 执行包含在预热中，不计入测量。产物写入新的独立目录，不进入训练回放。

```bash
conda run --no-capture-output -n pytorch python scripts/benchmark_selfplay.py \
  --config-dir configs/minimal_test --model /absolute/path/to/model.pt \
  --output data/selfplay_benchmark_new --games 1024 --warmup-games 128 --repeats 2 \
  --candidates 32:32:1:1:0 128:128:1:1:0
```

候选项依次为 `game_threads:max_batch:server_threads:search_threads:batch_wait_us`。模型须符合当前输入契约和配置画布；`--evaluator random` 可省略模型，用于单独检查冷启动，此时每个请求重建随机服务，与生产行为一致。GPU 不可见的托管沙箱须在可访问宿主 CUDA 的执行环境运行。比较时保持网络、访问预算、棋规和采样参数一致，优先看完整请求的训练行／秒与局／秒，并检查局长及采样量波动。`simulations` 只统计主局逐手搜索，不含开局和 side 搜索；其吞吐用于辅助区分工作量变化。所有棋局完成后才结束计时，因此包含批次末尾并行度下降的耗时。

输出保留生效配置、源码快照、二进制和模型校验值、硬件与依赖、原始事件及 NPZ，并记录实际平均 batch、cache 命中率、排队时间与每秒采样的 GPU／内存指标。资源采样自身有少量开销；这是 selfplay 短测，不是完整训练轮性能或棋力结论。改变并行度会改变请求、随机 D4 和 fork pool 的调度顺序；提高树内线程数还会改变搜索访问顺序，须作为实验条件变化记录。

## 配置组织

每套配置位于 `configs/<name>/`，分为 `run.cfg`、`env.cfg`、`net.cfg`、`selfplay.cfg`、`train.cfg`、`eval.cfg`、`match.cfg`。`env.cfg` 的 `[environment]` 管理棋盘、规则和训练行的禁手特征 dropout。`selfplay.cfg` 首部 `[search]` 集中完整／cheap 根访问预算、cheap 概率与权重；搜索技巧按独立 section 配置，底部 `[parallelism]`、`[inference]`、`[writer]` 管理执行资源。训练字段以 [config.py](../python/etazero/config.py) 为事实源；评估／比赛仅读取自身文件，字段以 [eval_config.py](../python/etazero/eval_config.py) 为事实源，不参与训练配置身份。

派生配置在 `run.cfg` 的 `[run]` 中使用 `extends = baseline`，父目录名优先在所选配置的同级解析，找不到时在本版本 `configs/` 下解析，因此实验伞目录中的臂也可直接 `extends = baseline` 或 `minimal_test`。解析次序是父配置、当前配置、当前目录的 `*.cfg.local`；父目录本机覆盖不向子配置传播。继承循环、父目录缺失、未知/重复字段、错误文件归属、非法枚举与范围、非法组合或未实现能力，均在启动 worker 前失败。

Python 保存完整生效配置及其 SHA-256 身份，生成 C++ 消费的 `config/effective.cfg`，C++ 不维护第二套默认值。观测、动作、模型输出与原始分片类型由 [schema.py](../python/etazero/schema.py) 定义，构建时生成 C++ 头文件，模型和数据记录契约内容校验身份。

恢复要求生效配置一致，并核对 native 配置未被另外修改。当前不做任意配置热更新，不实现历史格式兼容层；正式实验条件由用户确定。

## 运行目录

| 路径 | 内容 |
|---|---|
| `config/effective.json`、`config/effective.cfg` | 完整生效配置及 native 输入 |
| `logs/` | 原始事件日志、worker stdout/stderr |
| `selfplay/iteration_<编号>/` | 按 attempt 与 worker 分开的完整对局 NPZ |
| `snapshots/` | 不可变 shuffle manifest 和可重建训练 payload |
| `checkpoints/` | 最近若干轮末的完整训练状态 `.pt`，以及全部历史 checkpoint 的 `.json` 提交记录 |
| `models/`、`models/current.json` | 已校验推理模型和发布指针；首轮训练前无已发布网络 |
| `training.png` | 每轮提交、正常退出和恢复后自动重建的训练图；也可手动重绘 |
| `loss.png` | 总 loss 与每项实际启用的损失分量，逐面板对照训练均值与轮末验证均值 |
| `logs/performance.png` | 阶段耗时、吞吐与推理统计 |
| `logs/iterations/` | 与整轮提交对应的指标、累计训练时间和模型身份 |
| `.internal/discarded/` | 不参与训练或图表的中断轮原始产物 |
| `.internal/` | run/state/session、锁、iterations、catalog、training_views 和 source 源码证据 |

原始对局、历史推理模型和提交记录保持不可变；训练状态 `.pt` 按 checkpoint 保留策略回收，派生缓存集中存放不影响恢复与快照重建。新配置与轨迹契约不提供旧格式兼容，已有历史目录不会被自动迁移或覆盖。

## 数据链路

### 原始对局分片

[RecordWriter](../cpp/src/selfplay/record.cpp) 经有界队列接收完整对局，按随机取整后的最终训练行数写压缩 NPZ。一局按主局重复行、side重复行的顺序写出，可以跨文件拆分。首文件阈值采用来源公式 `maxRows-int(maxRows*(1-minProp)*U)`，`U` 均匀取自 `[0,1)`；`writer.first_file_min_random_proportion` 对应 mainb18 的 `firstFileRandMinProp`。后续满片恰好包含 `writer.shard_rows` 行，不按定时器提前发布；退出排空队列并发布未满尾片。数组在 flush 后复用，轨迹存储可随稀疏采样增长；不使用 pickle 对象数组。

空间观测为 `uint8[N,5,ceil(canvas²/8)]`，独立的 `globals` 为 `float32[N,6]`，各平面按行优先顺序使用高位在前的 bit packing，最后一个字节的多余位必须为零，与 KataGo 的 packBits / NumPy unpackbits 位序一致。压缩 ZIP 时直接流式读取数组，不构造另一份完整 NPY 字节串。

每局保存 T+1 个观测和行棋方、T 个动作与奖励、新增模拟数、行为温度、真实终局与原因，以及尺寸、棋规、种子和唯一对局身份。T 包括所有开局动作；`train_mask` 的前缀为零，前缀 simulations/temperature 为零。另保存 opening/balanced/policy moves、attempts、status、后继视角 start value 与失败原因。每步保留轻量的 network/search WDL、cheap 标记、policy/value surprise、期望 target weight 和实际 `row_repeats`，用于审计完整对局及权重计算。轨迹观测保留完整禁手提示；writer按每个最终重复行独立抽样，另存 `forbidden_input:uint8[rows]`，训练视图展开后同步应用到两禁手平面及全局开关；重新shuffle不改变这些决定。

大数组 `policies`、`opponent_policies` 和 `visits` 与完整轨迹分别索引：仅保存 `sample_indices = flatnonzero(row_repeats > 0)` 指定的位置，shape 为 `[S,canvas²]`，S 是采中的不同位置数。每个位置只存一次，训练视图按采样次数展开。过滤发生在 surprise weighting 和随机取整之后，能保留重新获得权重的 cheap 搜索；零权重 full、cheap 以及开局均不保存独立训练行；零采样搜索仍可作为前一手的 opponent 目标随前一手保存。policy 使用最终选择权重的 int16 确定性量化，visits 保持 int64；归一化顺序见 [训练目标](algorithms.md#学习与数据使用)。另存 `[S]` 的 `opponent_policy_weights`，终局手及显式不使用 outcome targets 的 reanalysis 行为零；后继目标在 C++ 完整对局尚在内存时提取，shuffle 不依赖后继手也被采中。对局与采样数组在同一 NPZ 内原子发布，避免两个文件交接不完整；未采中位置的 visits 不再支持事后恢复。

分片保留所涉及棋局的完整轨迹和搜索数组，跨片棋局会在相邻分片重复携带上下文。元数据 `row_begin` 与 `rows` 指定本片在这些棋局的完整最终训练行序列中实际承载的区间 `[row_begin,row_begin+rows)`；只有该区间进入训练。先用完整轨迹计算 TD 和 opponent 目标，再取区间，禁止将分片末尾当作终局或重新抽取重复频率、dropout、Q量化。`plies` 和 `games` 描述本片携带的轨迹，不能跨片直接相加当作唯一局数；catalog 的唯一统计见下文。独立 policy init 若已结束整局，仍保留零训练行的轨迹证据。分片元数据关联 run、iteration、worker、attempt、配置、源码与输入模型。

先写同文件系统的临时文件，flush、fsync 后重命名，再 fsync 目录。每次 worker 启动使用新的 attempt 目录，拒绝覆盖已有输出；读取只消费已发布的 `.npz`。写入错误唤醒生产者并传播失败。正常信号收尾会发布已完成对局的未满尾片；SIGKILL 或断电可能丢失队列和缓冲中尚未发布的训练行及轨迹。

主局另存 `reanalyzed`、`reanalysis_used_outcome`、`reanalysis_original_visits`、`reanalysis_policy_surprise` 和 `reanalysis_value_surprise`。重分析只替换所选 cheap 手的训练搜索数组及轻量 NN/search WDL，不修改真实轨迹；原始预算/选择信息单独记录，surprise 在替换后重算。opponent 可用性同时检查末手和 outcome-target 开关。PDA 主局 flag 和 signed-half-d 在相邻玩家间翻号且全局保持同一优势；完整轨迹保留它，禁手 dropout 只修改原禁手字段。历史不符新契约的数据/模型直接拒绝。hint/fork在`initial_position_kind`单独标记ordinary/hint/earlyFork/gameFork/hintFork（0…4），`hint_actions`为canvas编号或-1。总prefix为balanced+policy+initial；initial prefix与开局搜索互斥，无搜索/频率监督，fork必须PDA0。

[读取与检查](../python/etazero/data.py)核对类型、形状、offset、交替视角、逐手棋盘变化、mask、采样索引、已保存访问数与策略分布、奖励和训练行区间。AlphaZero WDL 目标由终局结果和该状态行棋方派生为 W/D/L one-hot。训练视图展开各棋局的主局及side重复行，再按本片区间选择。SQLite catalog 可从原始分片重建；按 run/attempt/worker/game 身份核对轨迹指纹与总训练行数，拒绝重复或重叠区间，逐区间累加实际行数，局数只计一次。胜和负、完整局长和开局统计归属于包含该局第0个训练行的分片，零行棋局单独保留，跨片上下文不会重复计数。启动时扫描历史产物，运行中仅扫描当前 iteration；开局前缀不进入训练行数。近期窗口按实际文件mtime从新到旧选完整片，不用未选中的上下文行抵扣窗口或产样quota。

原始搜索Q目标随所有架构保存，启用Q的learner才计入loss。`q_visits`为int16[S,A]，S对应`sample_indices`的正采样位置；`q_values`为int16[R,A]，R为主局最终重复输出行数，对应`forbidden_input`。访问数按位置压缩，Q随机量化按输出行独立保存，重复行可以不同。side对应`side_q_visits` [D,A]及`side_q_values` [R_side,A]，零重复side仍保留访问数而没有Q输出行。训练视图统一为float32[N,A]的Q值（除32000）与访问数；D4、shuffle、reader与预取保持两者和观测配对。源契约由[schema.py](../python/etazero/schema.py)唯一维护，旧契约明确拒绝。

独立 `side_*` 数组保存关联主局索引、观测、玩家、policy、visits、搜索 WDL、频率和逐输出行提示决定。side 搜索未走到终局，完整主局标志和 opponent 权重为零，主 value/三个 TD 均取自身搜索 WDL。统计 rows 包含普通及 side 的重复行，plies/games 仍仅指实际主局。side 由真实 SP 替代着和回复/替代着递归产生；PDA 两个全局量始终为零，writer 会拒绝非零条件。开局前缀不充当 side。

### 窗口与 shuffle

[shuffler](../python/etazero/shuffle.py) 使用 KataGo 的 power-law 窗口。窗口累计 `N = min(random_rows,m) + postrandom_rows`，random身份来自原始数据的 `model_id = random:seed`（判断前缀`random:`）；该封顶只用于窗口，不改变真实累计产样和quota。按文件实际mtime从近期向前选择完整分片至窗口阈值，最后一片允许超额。相同mtime按catalog稳定输入顺序打破平局，不声称复现来源文件系统遍历顺序。

令 `s = taper_scale`（0时取m）、`d = add_to_data_rows`、`x = N−m+s+d`，窗口为 `max(m, int(m + a·(x**p−s**p)/(p·s**(p−1))))`，设置 `max_rows` 时再截到上限。参数和非法范围由 [train.cfg](../configs/baseline/train.cfg) 与 [config.py](../python/etazero/config.py) 唯一维护。快照另存来源范围 `[raw_total+int(d)−actual_window_rows, raw_total+int(d)]`；其end是未封顶的真实累计，不能替代N或本地产样quota。

`keep_target_rows` 接受正整数或 `all`。在MD5切分之前计算 `q = min(1,K/actual_window_rows)`；窗口文件先乱序，再累计训练行到 `group_rows` 阈值才封组，末文件可以跨阈值。每组保留 `round(group_rows_actual·q)` 行的均匀子集，取整采用Python round，包括半整数的偶数取整；因此总输出可能偏离K。`q=1`保留全部，`q=0`的空结果明确拒绝，不能用旧快照顶替。

两阶段shuffle对保留行独立均匀分桶，再按固定计划的桶数和每桶文件数输出：`B=max(1,round(approx_rows_per_wave/bucket_rows))`，`F=bucket_rows/training_shard_rows`，要求整除；真实桶内按 `floor(j·N/F)` 等分。training_shard_rows是名义大小，真实文件可更大、更小或空，不根据实际桶大小重新决定F。所有字段共同排列。source的均匀排列＋multinomial桶计数与该随机分配分布相同；这里使用可重建的独立SeedSequence流，不宣称来源os.urandom序列相同。初次采样、wave内scatter和merge分别使用含partition/stage/wave/group/bucket的命名空间，跨桶任务不复用流。

来源shuffle CLI默认 `min_rows=250000,p=1,a=1`，K必填，group80000、输出70000、桶默认等于输出；`selfplay/shuffle.sh` 显式使用 `p=.65,a=.4,K=20000000`，普通分支MD5留出1%，SKIP_VALIDATE分支全作train。`synchronous_loop.sh` 还覆盖m100000、s50000、K600000并设置SKIP_VALIDATE=1。EtaZero的窗口、保留目标与本机shuffle资源参数以 [train.cfg](../configs/baseline/train.cfg) 为准；当前baseline窗口采用 [KataGomo shuffle profile](/home/sky/RL/SkyZero/KataGomo/python/shuffle.sh:49) 的 `min_rows=150000,p=0.8,a=0.3`，保留目标K为20000000，默认启用验证（`skip_validation=false`）。shuffle为12进程、16384MiB数组内存预算，group/bucket/名义训练分片均65536行、waves=1。同步示例中的train bucket/no-repeat/训练数量不是本地固定轮预算的来源默认。
首次消费原始分片时完成 SHA-256 与全轨迹验证，然后在 `.internal/training_views/` 保存 DEFLATE level 1 压缩的紧凑训练视图和来源证书。后续 shuffle 校验缓存哈希；原始文件大小或 mtime 变化时重新核对来源哈希，避免窗口内反复解压和逐步轨迹验证。原始分片遵循不可变约定，缓存不代替原始轨迹；损坏缓存明确报错。已离开当前窗口的视图在本轮提交后回收。

`shuffle.waves > 1` 先在输入组中完成一次采样，将保留的每行独立均匀分配到一个 wave，再逐 wave 执行两阶段 shuffle；第二阶段不重复采样，完成后立即删除该 wave 的临时文件。额外 I/O 换取更低的同时存活分桶文件数量，不改变已选样本集合。

`[shuffle]` 管理 worker 数、组/桶/训练分片行数、wave 数、内存、临时目录与快照保留数。`shuffle.memory_mb` 为全部 worker 的数组内存规划预算。按实际观测、policy、visits 布局估算，必要时在采样前确定有效组阈值及桶名义行数；桶保持名义输出行数的整数倍，组估算允许完整末文件超额。不能容纳一个完整原始分片或名义训练分片时，在开始写入前报错。实际随机桶超过数组预算时明确失败，要求扩大内存或减小桶名义大小，不递归再散桶改变输出计划。该估算不包含 Python 进程、分配器与 OS 开销，不是进程 RSS 的硬上限。快照 manifest 和 `shuffle_resources` 事件记录有效行数限制、采样比例、输出行数估算、数组内存、临时磁盘与文件数量估算。磁盘容量在开始前核对，随机波动、压缩开销和外部磁盘使用仍可能使实际值不同。

`shuffle.temp_dir` 可指定独立临时目录，空值使用 snapshots 所在目录。每次尝试使用唯一私有工作目录，`shuffle.compress_temp` 默认启用 DEFLATE level 1；关闭时使用未压缩 NPZ，适合空间充足且压缩 CPU 成本受限的场景。该开关覆盖 wave、scatter 的中间文件，不影响持久化视图缓存的压缩。中间文件不哈希、不 fsync；最终训练文件仍压缩、校验并持久化后发布。压缩不改变数组、采样随机流或最终训练文件校验值；磁盘容量检查继续采用未压缩容量的保守估算，不假定固定压缩率。正常完成或异常退出清理本次临时目录，原始分片保持不变；SIGKILL/断电可能留下未发布目录，需要删除对应的 `.shuffle_*` / `.tmp_*` 目录后重建。

全部文件与 manifest 在临时目录准备，记录来源及 SHA-256、随机种子、窗口行数、输出行数与校验值，完成后原子发布整个快照。learner 持有本轮快照，不混读不同代次。

`shuffle.snapshot_keep` 限制保留完整 payload 的已完成快照数量，其余只回收可重建的 `data/`，保留 manifest、配方、原始对局与 checkpoint。读取被回收快照时，在文件锁内用原来源、随机种子和资源计划重建相同采样，要求文件名、行数与 SHA-256 全部符合原 manifest，再原子发布；因此 checkpoint 的数据游标仍可恢复。原始对局本身不按此配置删除，磁盘占用仍随累计产样增长。

[BatchReader](../python/etazero/reader.py) 预取当前文件及depth个后续文件，保持紧凑布局，仅解包当前batch。每文件只读取 `floor(rows/batch_size)·batch_size` 的前缀，尾部丢弃；低于batch的文件不供给样本，也不与其他文件凑batch。manifest记录每pass完整batch数、可消费行数和尾部丢弃行数。

默认repeat：首pass随机文件排列；后续pass照来源reservoir算法推迟刚用过的文件，使再次出现至少隔开约1/3文件集。no-repeat：每个文件在一个快照中只读一次，耗尽抛出StopIteration，恢复耗尽游标仍耗尽。本地固定轮不等待异步数据；启用no-repeat而完整batch不足剩余固定训练预算时，learner在任何更新前报错。切换快照只发生于下一同步轮；来源的目录轮询、新文件队列插入和20目录历史不适用于持有不可变单轮快照的本地协议，不能称已实现常驻异步generator。

后台batch队列和CUDA上传stream有界。每batch携带消费游标，包括文件order、已结束文件、文件/行位置、pass、RNG、repeat模式、split与batch大小；checkpoint取已消费游标，包含AMP skip。源generator在pop文件时记用过，本地额外保留文件内游标以恢复后续完整batch。未消费的预取可丢弃重读。两个pinned槽通过event防止DMA覆盖，record_stream保证GPU生命周期。

baseline默认启用验证；`training.skip_validation = true` 可选择同步来源SKIP_VALIDATE分支。启用时MD5原始文件basename的前13个十六进制字符除2**52，`[0,.99)`作为train，`[.99,1)`作为validation；目录和模型代次不影响所属分区。writer以独立OS随机流生成64位十六进制basename，不消耗game RNG。两分区共用原窗口与切分前q，分别按来源分桶规划，manifest持久记录源文件分区与全部输出hash。验证payload位于同一个原子data目录的validation子目录，回收及重建同时覆盖两个分区。

每轮训练结束、最终checkpoint之前，validation用当前raw模型eval/no_grad读取每文件完整batch，默认按文件名排序；`randomize_validation_files`可随机文件顺序。验证始终随机D4，即使训练D4关闭；采用独立随机流，保持训练RNG不受验证影响。AMP精度及FP32 heads沿训练设置，启用compile时单独生成eval图。`max_validation_samples=0`不设上限，正值在完整batch使总数超过上限后停止，沿来源允许一batch超额。验证不推进optimizer、Lookahead、SWA、训练样本或成功更新计数，不修改BN统计；无完整validation batch时明确记录samples=0，不捏造loss。事件记录样本均值loss、batch数量和D4计数。此验证不是棋力评估。
## 固定训练量与自对弈产量

`training.replay_ratio` 是训练样本消费次数与新增 selfplay 实际采样行数的目标比例，重复 shuffle、降采样、窗口淘汰和重复消费不改变分母。

冷启动分为两个独立阶段：iteration 0 完成 `bootstrap_games` 局随机评估搜索，只测量有效行数／局，不训练；中断未提交的bootstrap同样归档完整已发布对局后重跑整轮，不用半局补监督。iteration 1 根据实测均值估算 `min_rows` 缺口，实际不足就继续补局，直至达到门槛。两阶段均跳过网络平衡开局与 policy init，仍执行正常搜索和真实终局监督。

首次有效训练、模型导出与发布完成后，在同一次持久化提交中保存实际累计行数 `replay_origin_rows`，将 `target_rows` 重置为该值。此后的恢复沿用已提交起点，不再次重置。首轮及 bootstrap 多产的数据保留在回放池中，但不抵扣 iteration 2 以后的新增预算。

```text
U_i = train_steps_i * batch_size_i
D_1 = iteration 1 提交时的实际累计有效行数
D_i = D_(i-1) + U_i / replay_ratio_i  (i >= 2)
C = 累计实际采样行数（主局与side各自row_repeats之和）
deficit = max(0, D_i - C)
rows_per_game = 近期完成对局有效行数之和 / 对局数
games_i = max(1, ceil(deficit / rows_per_game))
```

累计目标保留非整数，只在换算局数时取整。计划在 selfplay 前持久化，中断重跑从上一轮已提交状态重新建立同一目标 D；作废轮的完整对局也不抵扣新一次尝试的产样预算。正常整局产量误差进入下一轮，不改变训练量；每个正常轮次至少生成一局，沿用 MuZero V2 的调度边界。

例如每轮 1000 步、batch 128、ratio 8，iteration 2 新增目标为 16000 行，均值 80 行／局时计划 200 局，与冷启动累计多产量无关。以后正常轮次多产 4000 行时，下一轮约 150 局。局数是估计；每次产样后检查真实累计C，不足D就用近期实测行数／局继续补完整局，直到C≥D。PCR、降权或随机取整可以让完整对局产生零训练行，此时继续补局；worker既没有新增完整对局也没有新增训练行，或没有可用均值时明确失败，不用旧快照掩盖缺口。训练仍消费1000个batch，AMP overflow不会重试同一batch。日志中的累计实际 ratio 使用已提交训练消费量除以累计有效行数，包含冷启动；该观测量与排除冷启动的产样预算分开解释。

`training.sub_epochs`将固定消费budget划为非空本地分段，缺省1；第j段结束于`floor(j*train_steps/sub_epochs)`。分段入口只重置Lookahead周期与分段计数，保留fast/slow、SWA及全局消费状态。LR/WD和范数打印仍用整轮batch时钟，validation及finish_round仅在整轮末执行。此预算映射不同于来源按概率选文件的subepoch预算。

每轮计划锁定input_model_id，所有实际主局和side沿用该输入。全部产样请求完成后才对各worker做release握手，再进入shuffle/learner；日志记录PID、输入模型与释放时点。worker跨轮存活，模型ID或模型路径变化均重建evaluator/cache，避免同ID不同权重串用；不实施局内轮询换模型。

## 模型发布与恢复

[learner](../python/etazero/training.py) 实现固定步数、batch 级 D4、SGD / AdamW、分组 LR/WD、warmup、范数自适应衰减、Lookahead 与 SWA；配置入口在 `train.cfg`，数学与适配边界见 [学习与数据使用](algorithms.md#学习与数据使用)。非有限 loss 或无法恢复的非有限梯度明确失败；FP16 缩放溢出按下述机制跳步，消费与成功更新分别计数。

`training.compile` 将网络与完整损失联合交给 torch.compile / Inductor，使用 fullgraph 和静态 shape 捕获：learner 的 batch 大小与画布固定，各网络结构、尺寸与精度使用各自的图。超过重编译限制时明确失败，避免长运行静默退回 eager；首次 shape / 精度编译有额外耗时。选择 AdamW 时 CUDA 使用 fused 实现，CPU 使用普通实现；SGD 使用 momentum 0.9。编译和融合可能改变浮点归约顺序，属于显式执行条件，不能据此声称与旧 eager 路径逐位一致。SGD momentum 或 AdamW 的一阶／二阶矩与 step 随 checkpoint 保存并恢复。

FP16 缩放溢出由 GradScaler 跳过当前 optimizer 更新并降低 scale，然后消费下一 batch，不保留图、不重试。当前 batch 的前向统计、随机流、reader 游标、样本与 Lookahead/SWA 时钟正常推进，成功更新数单独累计；每次溢出记录 `amp_overflow`，batch 日志记录 `amp_skipped`。训练 autocast 只影响主干，policy/value heads 和 loss 显式关闭 autocast 并使用 FP32。BF16 / FP32 的非有限梯度及非有限 loss 直接失败。

`training.checkpoint_every` 按本轮消费 batch 数定期保存 checkpoint；轮末及正常停止也会保存，间隔单位是消费 batch，而非成功更新、对局或 iteration。

checkpoint 保存训练模型、优化器、范数基准、snapshot 或运行均值的累积和/权重、Lookahead slow 权重与计数、SWA 权重/buffers/采样累积、AMP scaler、本轮及分段消费 batch、分段位置、累计消费样本与成功更新计数、Python / NumPy / Torch / CUDA RNG、数据读取状态、iteration 和本轮步数、配置契约、源码身份、父 checkpoint 与已提交更新身份。warmup 由已恢复的消费样本计数推导；optimizer 组保存当前 LR/WD，轮内恢复不额外刷新，保留 5/50 batch 时点。使用唯一文件名，不覆盖历史产物，持久化后才原子更新 learner 指针。

整轮成功提交并发布后，以磁盘上的 `.internal/state.json` 为权威沿 checkpoint `.json` 父链清理：已提交轮的中间 `.pt` 删除，只保留 `training.checkpoint_keep` 个最近轮末 `.pt`（正整数，初始化 checkpoint 也计入）。当前续训所需状态始终保留；尚未提交轮的文件和 `.internal/discarded/` 不由此机制清理。所有 `.json`、日志、历史推理模型与原始对局保留，历史指标重绘和旧模型评估不依赖已删除的 `.pt`；已回收历史点不再支持完整训练状态加载。清理在整轮提交与发布后执行，作为轮间维护，不计入该轮已提交训练时间。

清理先验证完整提交链和待保留文件，再删除 payload；删除过程中中断可重复执行。重启在归档未提交轮后补做清理，即使累计迭代上限已达到也会执行；清理事件记录在 `logs/events.jsonl`。训练结束仍保留最近的完整状态，`checkpoint_every` 的轮内保存频率不受保留数量影响。

[exporter](../python/etazero/export.py) 优先使用已采样的 SWA 权重，无采样时使用训练权重，并在 manifest 中记录选择与采样次数。在实际推理设备上预计算 evaluation normalization 的 FP32 逆标准差，保留原归一化运算顺序、卷积权重、精度设置与 mask 位置，再导出独立 TorchScript 模型。归一化仅对新建中间量分开执行原地乘、加，避免 JIT 融合改变舍入。每轮导出只生成模型、记录 manifest 和 SHA-256 并原子发布，不执行 eager / TorchScript / C++ 数值对照或 FP16 对 FP32 的容差检查；这些检查由独立推理测试承担。checkpoint 身份与配置、模型 manifest、契约、路径和文件校验继续保留，实际推理时检查输出维度和非有限值。不修改训练网络和 checkpoint，整轮提交后更新 `models/current.json`，身份来自 checkpoint，不依赖 mtime，没有比赛胜率门控。

`.internal/state.json` 是整轮提交的唯一权威：本轮完成自对弈、shuffle、训练、模型导出和绘图，保存 `logs/iterations/<iteration>.json` 后才原子推进 state。`models/current.json` 在提交后更新，启动时先核对已提交checkpoint SHA及模型manifest/契约/路径/权重SHA和checkpoint身份，再从state重建发布指针；校验失败在启动worker之前拒绝。iteration 0 的 bootstrap 也按完整轮提交。

无论中断发生在 selfplay、shuffle、train、export 还是最终提交前，重启都从上一完整轮的 checkpoint 重跑整轮，不采用本轮 `learner.json`。中断轮的对局、checkpoint、快照、模型、轮次状态和候选指标及强制终止遗留的私有export staging目录移入 `.internal/discarded/<id>/`，保留原始字节；catalog 作为派生索引重建。半局始终不构造监督。已经原子提交但发布指针尚未更新的轮次仍有效，不重复训练。底层 learner 的 checkpoint/游标恢复能力用于内部验证，用户训练入口采用整轮恢复。同目录控制器由OS flock独占；checkpoint不可覆盖的原子link、模型完整目录rename、state/current原子replace分别构成发布边界。未发布暂存不被consumer当作模型。

固定KataGo来源的save函数保存model、optimizer、metrics/running_metrics、train_state、validation metrics、config及可选SWA；train_state含SWA采样累积和文件使用状态。它没有捕获Python/NumPy/Torch/CUDA RNG、AMP scaler或函数局部Lookahead fast/slow cache/counter。Eta额外保存这些恢复状态和文件内已消费游标；不能把本地精确learner续训或整轮回滚协议说成来源默认能力。独立入口[check_katago_persistence.py](../tests/reference/check_katago_persistence.py)执行原save函数四种分支并实测本地checkpoint字段。

来源export_model_for_selfplay.sh的USEGATING=0直接发布到models，非零时交给gatekeeper；本地按用户profile直接发布，无额外棋力门控。同步来源示例仍运行gatekeeper，但该示例不能证明所有来源运行都必须gating。


运行锁禁止同目录多个控制器，在取得锁后以 `.internal/run.json` 判断是否自动续训；非空但缺少运行记录的目录不作为新运行使用。续训沿用累计轮数目标，已达到目标时不重复训练，追加训练通过 `--iterations` 提高目标。启动与默认输出路径见 [README](../README.md#续训与评估)。缺失文件、重复数据、契约不符、配置变化或模型/checkpoint 校验失败均报错，不退化为仅加载权重。新运行 `--weights` 只导入模型状态，并保存导入来源校验值，自动续训同样拒绝该参数。

### 运行观测与证据

每次执行保存生效配置、完整源码快照、Git 身份与工作树状态、依赖与硬件、设备、种子、worker 命令、stdout/stderr，以及模型和数据来源。源码快照排除版本根目录的 `data/`、构建和运行产物，避免把其他运行的数据纳入源码身份。构建清单记录源码与二进制校验值，运行前拒绝过期构建。

`logs/events.jsonl` 记录计划、缺口、实际产量、推理请求/批数/最大批大小/队列等待、缓存命中与各服务实际处理行数、shuffle 资源估算、损失与梯度、checkpoint、发布和阶段耗时。绘图按 checkpoint 提交身份筛选 loss，回滚更新不冒充有效进度，保留原始值，不隐式平滑。

日志由有界后台队列批量写入，最长每秒执行一次 fsync；checkpoint 的指标与提交事件先通过持久化屏障，随后更新 learner 指针，轮次状态提交也经过屏障。强制退出可能丢失尚未提交的日志尾部，已提交 checkpoint 的更新指标保持可追溯。

训练预算和 Elo 使用 state 与逐轮指标中的 `elapsed_seconds`：仅累加成功提交轮从计划准备到绘图完成的墙钟，包含 bootstrap、自对弈、shuffle、训练和导出，排除初始化、启动恢复、暂停、排队和作废尝试。`max_seconds` 在轮次边界检查，可能超出一轮；不将最后模型假定为恰好位于预算点。日志 `active_seconds` 与 heartbeat 保留实际活动耗时作为审计信息，包含作废工作，不作训练比较横轴。阶段时间不累加线程耗时冒充墙钟。

## 验收与限制

`tests/test_*.py` 是 pytest 自动发现的 Python 测试，`cpp/tests/` 是 CTest 使用的原生测试；`tests/reference/check_*.py` 是手动运行的来源对照检查，不随普通 pytest 自动执行。来源检查读取本地 KataGo / KataGomo 源码，多数会核对 `reference_sources.json` 中固定的 commit 和 SHA256；链接生产库或运行 `build/` 下测试程序的检查须先执行 `scripts/build.sh`。构建、训练、调度与性能测量入口保留在 `scripts/`。

| 来源对照检查 | 范围 |
|---|---|
| [Q量化](../tests/reference/check_katago_q.py) | 固定来源的随机量化函数与生产库逐项比较 |
| [网络](../tests/reference/check_katago_network.py) | plain、NBT、Transformer 的参数映射、初始化、输出与梯度 |
| [图搜索](../tests/reference/check_katago_graph.py)、[搜索修正](../tests/reference/check_katago_search_corrections.py) | 子节点权重、边访问追赶、聚合及 uncertainty / noise / optimistic 混合 |
| [采样](../tests/reference/check_katago_sampling.py)、[权重](../tests/reference/check_katago_sampling_weights.py)、[分支预算](../tests/reference/check_katago_forks.py) | PDA、value surprise、权重重分配及 hint / PCR / reduced 预算分支 |
| [回放](../tests/reference/check_katago_replay.py) | 窗口、分组、文件顺序与整 batch 读取 |
| [训练时钟](../tests/reference/check_katago_clocks.py)、[保存状态](../tests/reference/check_katago_persistence.py) | LR / Lookahead / SWA 时机及 checkpoint 字段范围 |
| [禁手](../tests/reference/check_katagomo_rules.py)、[标量公式](../tests/reference/check_reference_formulas.py) | KataGomo Renju 判断及窗口、优化器、梯度阈值和范数统计公式 |

以下来源检查示例在版本目录执行；`--output` 使用新的结果路径，检查失败会以非零状态退出。其他参数见各检查入口的 `--help`；保存状态检查直接运行并将 JSON 打印到 stdout。

```bash
conda run -n pytorch python tests/reference/check_reference_formulas.py
conda run -n pytorch python tests/reference/check_katago_q.py --output /tmp/etazero_q_reference.json
```

```bash
# 在版本目录执行
conda run -n pytorch ctest --test-dir build --output-on-failure
conda run -n pytorch python -m pytest -q tests

# 在宿主 CUDA 可访问的执行环境运行
ETAZERO_GPU_TESTS=1 conda run -n pytorch python -m pytest -q tests
```

[开局检查](../cpp/tests/opening_test.cpp)覆盖随机评估复现、双方视角、开局轨迹、重试与拒绝率切换、取消、失败和独立 policy init。[C++ 检查](../cpp/tests/core_test.cpp)覆盖三种棋规、长连、满盘、递归活三参考样例、价值视角、终局停止推理、精确模拟预算、穷举 PUCT 对照、分级 child 增长与大树回收、子树复用、状态适配、失败回滚与等待者唤醒。[搜索技巧检查](../cpp/tests/search_features_test.cpp)覆盖 WDL 与和棋回传、FPU 手算值、半均匀噪声、强制探索与剪枝、train/eval LCB 区别、cheap 树复用、surprise 重分配和随机取整。[Python 检查](../tests/test_python.py)覆盖配置、独立数学样例、轨迹与目标、窗口、bit 顺序和非整字节尾部、随机分桶分布、单 wave / multi-wave 守恒、视图缓存校验、快照精确重建、增量索引、日志持久化屏障、失败清理、资源规划、重复消费和游标恢复。[真实 GPU 检查](../tests/test_gpu.py)覆盖 iteration 0～3、bootstrap 中断整轮重跑、首轮记账重置后的恢复、开局前缀过滤、组批、两轮发布、全部测试尺寸/规则的数值、先后手比赛、AMP、eager 与 compiled 固定样本续训一致性、强制终止训练、整轮故障恢复、快照回收后的旧数据恢复、多 worker 和 selfplay 信号收尾、常驻协议的 Unicode / 换行路径与退出，以及多推理服务、FP16 推理、归一化预计算精度、紧凑 NPZ 与 multi-wave 的联通。

验证环境为 RTX 5090、PyTorch 2.12.0+cu132 与小规模配置。多 worker 的验收使用同一 GPU 上两个进程，多物理 GPU 尚无对应硬件验证。固定样本 learner 恢复的一致性结论限于相同环境，并行 selfplay 不承诺逐位复现。不保存半局和全部在途树。

专用原生推理算子后端、DDP rank 数据分片及各 rank 状态恢复、跨阶段异步流水线仍未实现。当前使用单卡逐轮语义，CUDA 上传 stream、派生视图缓存和可恢复快照回收为 EtaZero 的实现适配，并非直接复制本地 KataGo reader。

可运行、恢复正确与棋力提升是不同层次的证据，工程验收不能替代正式训练、算法收益比较或目标负载吞吐测量。


## 训练图与性能图

[plotting.py](../python/etazero/plotting.py) 根据 `logs/events.jsonl` 和 checkpoint 的 sidecar 提交链重建图。`training.png` 使用深色三行两列布局：顶部为胜负和局长，中部固定分为策略 loss 与价值 loss，底部为梯度范数和 NN 缓存命中率。策略组包含六项普通／对手、soft 和 optimistic policy；价值组包含主 value、三个 TD value、短期价值误差及启用时的 Q。两组图例在面板内分两列显示，分量全为正时使用对数纵轴；总 loss 和有效行数／局保留在日志中，不单独占用概览面板。总局长包含开局动作，有效行数排除开局并按采样次数计数；自对弈和缓存命中率横轴为迭代完成时的累计有效行数，包含 iteration 0 的 bootstrap，刻度采用 `1.2e5` 形式的紧凑科学计数。缓存命中率为该轮各 worker 的 cache hits 总数／submitted requests 总数，只显示有网络请求的已完成轮次，随机冷启动不填零。训练横轴从 iteration 1 起，loss 按该轮已提交消费 batch 取算术均值，包含 AMP 跳步，无平滑；梯度均值只使用成功更新的有限范数，原始 overflow 范数仍在日志中，聚合同时记录跳步与有效梯度 batch 数。梯度范数为整个网络裁剪前平均 loss 的 L2 范数；每条 loss 曲线为已乘训练系数的 batch 均值，Q关闭时十一项之和等于总loss，Q开启时另有`q_winloss_loss`曲线并计入总loss；关闭时不伪造零值曲线。优化器解耦衰减不计入 loss。

`loss.png` 使用四列面板，逐项展示总 loss 与实际启用的所有分量，蓝色实线为训练、红色虚线为验证。每个面板单独决定纵轴范围，正值使用对数刻度。验证取已完成轮次最后一次尝试的 `validation` 事件；新的 `plan` 清除该轮先前尝试的验证记录，当前未提交轮次不显示。无验证事件或无完整验证 batch 时留空，不填零且不跨缺测轮连接曲线。训练为轮内已提交消费 batch 的均值，验证为轮末 raw 模型 eval/no_grad 的样本均值，二者时点、模型状态与数据不同，差距不能直接全部归因于过拟合。验证不会替代已发布 SWA 模型的棋力评估。

已完成轮次的自对弈统计取最后一次尝试的对应日志。训练指标只采用 state checkpoint 提交链中的 update ID，并按 ID 去重；当前未完成轮次、未保存更新和废弃分支均不显示。阶段耗时和推理计数只取成功尝试，原始失败日志仍保留供审计。日志中间损坏会报错，仅末尾未完成 JSON 可忽略。

`logs/performance.png` 单独显示各阶段已完成尝试的累计 wall seconds、有效行／自对弈秒、已提交训练样本／训练秒，以及 NN 平均 batch、每请求排队微秒和缓存命中率。训练耗时包含对象创建、数据等待、计算和 checkpoint 等开销。没有完成事件的中断阶段缺少耗时，不进入分母；当前未完成训练轮次也不显示吞吐，因此这些图不能直接作为完整跨中断端到端性能比较。推理计数按 iteration、attempt、worker 去重。bootstrap 使用 random evaluator，未发生网络组批时相关面板无值。

每轮在 state 提交前先 flush 日志并准备三张 PNG，将绘图耗时计入本轮；controller 正常返回和重启时可从已提交 state 重建；绘图不修改 checkpoint、数据或随机流。`bash scripts/run.sh plot --run-dir <实验目录>` 可手动重建，`--plot` 保留为返回后的显式重绘选项。绘图异常会明确报错，已提交训练状态保留，重启可重建。

## 自动实验

[scripts/autoexp.sh](../scripts/autoexp.sh) 使用 Conda `pytorch` 运行 [experiment.py](../python/etazero/experiment.py)，不执行 `exp.cfg` 中的 shell 代码。示例为 [autoexp_example](../configs/autoexp_example/exp.cfg)，只提供两个相同的小规模工程验收臂，不定义研究对照或正式训练预算。

实验伞目录包含 `exp.cfg` 和实验臂的直接子目录。每臂须有 `run.cfg`，可继承本版本 baseline 或其他配置，完整字段仍以 `config.py` 为准。`exp.cfg` 的 `[experiment]` 要求以下字段：

| 字段 | 语义 |
|---|---|
| `max_iteration` | 每臂累计完成训练轮数上限，0 不限 |
| `max_seconds` | 每臂累计已提交完整轮次墙钟上限，0 不限；不含排队、暂停、启动恢复和作废轮次 |
| `arm_gpus` | GPU 编号或 GPU/MIG UUID 的逗号列表，每槽同时一个臂；空值串行使用各臂原有 devices |
| `shared_init` | true/false；按网络结构和 seed 分组共享初始模型权重 |

两个预算至少一个为正，先达到任一预算即完成。预算通过普通训练入口的 `--iterations`、`--max-seconds` 覆盖此次调用，不改变生效配置，并记录在 `session_start` 的实际停止上限中；默认不自动构建，缺失或过期的 native binary 提示先执行 `scripts/build.sh`。时间上限在完整轮次边界检查，可能超过一轮的预算余量。

`MAX_ITERS`、`MAX_TIME_SECONDS`、`ARM_GPUS`、`SHARED_INIT` 可覆盖伞配置，`SHARED_INIT` 使用 true/false。`CONFIG_DIR` 或 `--config-dir` 选择伞目录，`DRY_RUN=1` 或 `--dry-run` 仅打印解析计划，不创建目录、初始化权重或启动任何任务。`--binary` 指定本版本已验证构建；`--work-dir` 指定调度产物目录，默认 `data/experiments/<伞目录名>_<路径摘要>/`。

指定 GPU 槽位时，子进程的 `CUDA_VISIBLE_DEVICES` 设为该槽，CUDA devices 映射为 `cuda:0`，CPU devices 保留。每臂至多一个 selfplay device，多设备配置会明确拒绝；GPU 槽位不得重复。每臂 `run.run_dir` 为独立目录，重复、嵌套或包含调度目录的路径会在启动前拒绝。GPU 槽位仅控制本次 scheduler，不管理外部任务或保留宿主 GPU。

共享初始化文件保存于调度目录 `.internal/initializations/`，按网络结构、seed 与契约分组并校验 SHA-256。它只携带模型参数及 buffers，各臂重新初始化 optimizer、计数和 RNG；random bootstrap 仍各自生成，不共享对局、回放或首轮训练后的模型。结构或种子不同的臂得到不同初始化；相同种子的随机对局可能相同，但产物与生命周期独立。`shared_init = false` 时各臂由自己的初始化路径启动。

调度目录公开 `configs/<臂名>/` 的解析配置与 `logs/<臂名>.runner.log`，内部身份、计划、状态与锁位于 `.internal/`。恢复时重新检查 run 生效配置和初始权重校验值，根据持久化 state 及累计已提交轮次时间 跳过已达预算的臂，否则恢复。臂配置、成员、输出目录和 shared_init 必须与调度身份一致；可以提高统一预算或调整 GPU 槽位，改变实验条件使用新配置和新产物目录。并发启动同一伞目录会被锁拒绝。

SIGINT/SIGTERM 停止排队，转发到运行中的 Python controller，由 controller 关闭 native worker，尚未整轮提交的产物保留待下次启动归档。调度器返回 130；中断臂下次从上一个完整轮重跑。某臂失败或意外在预算完成前退出时，调度器停止其他臂并报错，不将失败记作完成。状态文件仅是可查看的调度记录，完成判断以实际运行证据为准。

当前能力与对齐范围分别维护：上述行为以现有代码/配置为准；[57项对齐目标](../../EtaZero.md#alignment-targets)列出已确定的来源profile、网络版本、五子棋监督和验收范围。用户保留同步逐轮固定训练量与产样规划，不引入KataGo训练桶；方棋盘/规则沿现有env，VCN暂不纳入。v15辅助输出与三项搜索修正已接入；PDA/side、reanalysis及hint/early/game fork已接入。网络可选择NBT、独立plain（b10c128-fson-mish）或Transformer（bare b5c192h3nbttfrs/v17/fixup）；配置验证、架构参数组、训练和导出通过同一入口，baseline仍使用NBT。Transformer选择NCHW/SDPA及来源融合SwiGLU，编译时保留NCHW布局，具体数值与性能边界见[输入与网络](algorithms.md#观测动作与网络)。v17可选纯W−L Q仅在Transformer通过`network.predict_q_values`启用，baseline关闭；来源node visits、逐行随机量化及loss见[学习与数据使用](algorithms.md#学习与数据使用)。
