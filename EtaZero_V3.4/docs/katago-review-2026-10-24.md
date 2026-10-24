# EtaZero V3.3 KataGo 对齐审查

> 本文保留审查时的发现与证据；下列四项已修复，目录也已按职责整理。当前模块与共享边界见 [代码职责与依赖](implementation.md#代码职责与依赖)。正文中的原目录描述和行号属于审查时快照。

V3.3 已经复用了大量 KataGo 源码，也具备较完整的搜索、数据、训练和恢复实现；本轮检查发现四项明确的共有机制差异，尚不能认定满足 [katago-target.md](katago-target.md) 的整体对齐目标。优先处理普通优化路径的数据时钟和搜索统计缓存身份，再补齐配置与公共分析字段。

组织方式建议逐步转向保留 KataGo 职责划分的维护分支：直接复用可共享的实现，把棋类、四结果、Gumbel、MuZero 和外层训练编排的变化放在明确位置。当前证据支持这项整理，但不足以支持用整套 KataGo 一次性替换 V3.3。

## 审查范围

- 审查对象是 2026 年 10 月 24 日的 V3.3 工作树，包括开始审查前已经存在的未提交修改。仓库 HEAD 为 `42cee2294258e36a9f4648442b52a5ea69829d23`；它单独不能代表本次审查的全部代码。
- 按要求未读取历史版本目录。读取范围为仓库级约定、V3.3 及固定 KataGo 来源；未使用旧 EtaZero、旧 SkyZero 或旧 MuZero 实现推断当前行为。
- `~/RL/SkyZero/KataGo` 在本机不存在；使用 V3.3 `reference_sources.json` 指定的 `/home/sky/RL/SkyZero-lab/KataGo`。该来源 HEAD 为 `d91ea855110dae533f0aada947b2b7d78cc8a4e1`，工作树干净，清单内 139 个 KataGo 文件的 SHA256 全部匹配。
- 这是审查结果与结构建议。本次没有修改生产代码、实验配置、对齐目标或既有训练产物。

以下 P1 表示应优先解决的实验语义问题，P2 表示有具体触发条件的正确性、接口或对齐问题。未对齐不一定意味着程序崩溃，也可能意味着它执行了目标未授权的另一套行为。

## 已确认的差异

### P1 普通优化模式同时改变数据读取与内部训练时钟

**位置：** [training.py](../python/etazero/train/learner.py) 第 311–319、334–336、419–443 行；[plain_optimization.py](../python/etazero/train/plain_optimization.py) 第 29–48 行；[optimization.py](../python/etazero/train/optimization.py) 的 `optimization_for`。

**触发条件：** AlphaZero 设置 `[plain_optimizer]`，或 MuZero 设置 `muzero_training.katago_optimizer=false`。

learner 用 `hasattr(optimization, 'epoch_steps')` 决定数据管线。默认 `Optimization` 走 `SourceBatchReader`，跨外层轮次保存文件队列、当前整文件子 epoch 和消费位置；`PlainOptimization` 不含该属性，改走 `BatchReader`，新轮次丢弃上一轮 reader 位置、使用 `seed + iteration` 重新排文件，并将本轮实际预算等分为 subepoch。验证也由来源 epoch 结束改为每个外层轮次结束。普通路径的 Batch renorm 更新计数同样以外层轮次重置。

固定 KataGo 的 `python/train.py:get_files_for_subepoch`（第 1458 行起）通过 `TrainingDataGenerator` 选择整文件；epoch／subepoch 控制数据选择、验证及相应更新时机。这与选择 SGD、AdamW 或其他优化器不是同一个职责。目标明确要求内部数据选择与时钟继续对齐来源，普通优化器消融也没有获得更换数据管线的豁免。

实际构造两种合法控制器得到：

```text
默认路径  Optimization       source_clock=True
普通路径  PlainOptimization  source_clock=False
```

**影响：** 比较普通优化器与来源优化器时，文件暴露顺序、重复间隔、轮间未消费数据处理及验证／归一化更新时机也会改变，实验差异不能只归因于优化器。即使两条路径分别能够正确恢复，这一混杂仍存在。

**建议：** 让来源 reader 和内部 epoch 状态独立于优化器配方；普通优化模式只改变明确约定的 LR／WD、反传缩放、裁剪、Lookahead／SWA。验证同一数据和消费预算在不同外层分段下保持数据队列与内部时钟一致，同时保留普通优化器本身的数学差异。

### P2 搜索统计缓存未区分动态搜索条件

**位置：** [search.cpp](../cpp/search/search.cpp) 第 139–174、817–838 行；[alignment_search_test.cpp](../cpp/tests/alignment_search_test.cpp) 的缓存跨清树测试。

**触发条件：** 同时启用 `use_graph_search` 与 `use_eval_cache`，同一 Search 实例经历 full／cheap／reduce-visits 等搜索预算或根设置变化。这是搜索统计缓存，不是神经网络输出缓存。

EtaZero 的缓存键只有 `graph_identity + capped_player`。`SearchRun.max_visits`、`max_playouts`、`remove_root_noise` 可以在调用之间改变，但不进入键；清树也保留该缓存。

固定 KataGo 在 `cpp/search/search.cpp:814` 计算 `searchParams.getHash()`，在 `search.h:715` 将其合入缓存键。`searchparams.cpp:540` 的哈希覆盖全部 SearchParams，包括预算与根设置；`program/play.cpp:1233` 起的 `runBotWithLimits` 会实际修改 PCR 对应的预算、噪声、温度、根 FPU 等参数。因此，来源会区分这些条件下的缓存项。

现有 C++ 测试已经直接证明当前行为：先完成一次较大搜索，再 `clear_before_search=true; max_playouts=1`，仍要求读到先前的 root 缓存。该测试通过，说明它验证了当前实现，但没有验证这里的来源身份语义。根不直接混入缓存均值，并不消除问题：缓存还参与 first-explore FPU，后续子节点也会读取相同表。

**影响：** 不同搜索条件下积累的搜索结果会跨条件进入 FPU 和非根聚合；该行为不属于多棋类或四结果所需的变化。默认未启用该功能的运行不受此项影响。

**建议：** 按实际生效的搜索参数构造缓存身份，包含临时预算与根设置；保留相同条件复用、不同条件隔离和切回原条件后复用的语义。无需把整个运行配置或无关编译信息加入键。

### P2 来源配置的公共字段与缺省值尚未完整对齐

**位置：** [config_files.py](../python/etazero/config_files.py) 第 8–28、145–165 行；[engine_config.py](../python/etazero/engine_config.py) 的 `ENGINE_DEFAULTS`。

两份最小合法来源配置通过 `load_engine_config` 实测均被拒绝：

```text
numSearchThreads=1
maxVisits=12
nnCacheSizePowerOfTwo=20
→ Unknown key or ambiguous namespace: nnCacheSizePowerOfTwo

numSearchThreadsPerAnalysisThread=1
maxVisits=12
→ Unknown key or ambiguous namespace: numSearchThreadsPerAnalysisThread
```

来源 `cpp/command/analysis.cpp:107` 明确支持 `numSearchThreadsPerAnalysisThread` 别名；`cpp/program/setup.cpp:288` 支持 `nnCacheSizePowerOfTwo`，包括 `-1` 的禁用语义。当前只提供 `nnCacheSize → cache_entries`，既缺少来源拼写，也缺少指数到容量的转换。

未设置缓存参数时，EtaZero analysis 实测采用 65,536 项；来源 analysis 的默认值是 `2^23` 项。来源 GTP／match 又分别使用自己的默认值，不能用统一常量代表所有模式。显式实验 profile 可以选择自己的容量，这里指出的是公共解析器的缺省行为。

**影响：** 来源公共配置不能直接表达这些已有机制；省略字段时还会改变内存占用、缓存命中与执行行为。当前错误是明确拒绝，未发现这两个字段被静默忽略。

**建议：** 补齐公共字段映射、值转换与分模式缺省；用固定来源的真实最小配置作对照。保留 EtaZero 为多棋类和独立训练／分析 profile 所需的命名空间，但不能据此整体豁免来源字段。

### P2 HumanSL 分析缺少逐手 humanPrior

**位置：** [source_analysis.cpp](../cpp/command/source_analysis.cpp) 第 78–98 行。

固定 KataGo `cpp/search/searchresults.cpp:2071` 在存在 secondary human output 时，为每个 `moveInfos` 项输出 `humanPrior`；该字段不依赖 `includePolicy=true`。EtaZero 仅在 `includePolicy=true` 时输出整盘 `humanPolicy`，没有写入逐手 `humanPrior`。

使用真实 CPU 模型、启用第二策略模型、发出 12 visits 的公共 KataGo JSON 请求，得到：

```text
humanPolicy 存在
moveInfos 数量 7
含 humanPrior 的 moveInfos 数量 0
```

**影响：** 依赖逐手 human prior 的来源协议消费者无法直接使用该字段；整盘数组不是同一公共接口。它不依赖被排除的 score／ownership 头，也不是 MuZero 或四结果所需的差异。

**建议：** 在存在 human output 时补齐该字段，并覆盖 `includePolicy=false` 与根对称复制后的动作坐标。

## 共有机制覆盖情况

下表对应目标文档的全部 15 个共有机制范围。“已对齐”只限定于本行明确列出的行为与证据，不表示该子系统所有组合都已穷尽验证。

| 范围 | 本轮结论与证据 | 尚不能宣称的范围 |
|---|---|---|
| PUCT 展开与回传 | 公共 FPU、探索缩放、价值权重、终局视角及图边／节点统计有源码对照与 C++ 测试；来源公式检查通过。 | 两套搜索的全部并发轨迹、所有开关交叉组合仍待核实。 |
| 根探索与落子 | 噪声分布、温度、forced playout、目标剪枝、LCB、量化及行为／训练目标分离有具体实现和测试。Gumbel 根替换属于约定差异。 | 未使用同一外部 NN oracle 对两套完整引擎逐步比较全部根分支。 |
| 价值与探索修正 | uncertainty、optimism、noise pruning、SVB、pattern、futile、capped-side 的数学与适用分支得到来源检查支持。WDL 与修正效用分离。 | MuZero 的选择和调度仍有独立实现，不能仅凭复用数学函数认定全部调用频率相同。 |
| 图与缓存及跨手状态 | Go GraphHash、历史近似后的合法性复核、SVB 关联和 NN cache 机制已有对照；**搜索统计缓存有上述 P2**。 | 缓存键缺项未解决前，该范围不能整体标为已对齐。 |
| 搜索并行与预算 | 普通 PUCT 按完成计数停止并排空在途工作；Gumbel 预约／提交严格配额。C++ 并发与停止测试通过。 | 锁竞争、清理成本和端到端吞吐没有与来源做同负载测量。 |
| 自对弈采样与权重 | PCR／reduce-visits 分支、surprise、fork／side／reanalysis、hint、随机取整等已有来源检查；相关检查通过。 | 完整随机过程的长程分布、所有外部起点组合未作统计等价实验。 |
| 监督目标与增强 | 四结果、TD、policy／opponent／soft／optimistic、误差、Q、future-position、variance-time 的数据和损失路径已检查；目标、mask、增强和量化测试通过。 | 未对每种棋规及所有辅助头组合进行完整来源 learner 梯度对照；四结果扩展本就需要适配后的参考。 |
| 网络与数值 | 68 个来源基础预设通过 CPU 前向／反向对照；metadata 与四棋类 CUDA 推理通过选定测试。 | 未穷举派生配置；未验证所有预设的 CUDA、AMP、融合算子和编译组合。 |
| 优化器与时钟 | 默认来源路径的分组、LR／WD、Muon、内部 epoch 和恢复已有对照；**普通优化路径有上述 P1**。 | 公式检查不能替代 reader 到 optimizer 的整体调用链验证；所有 AMP 跳步组合本轮未重跑。 |
| 公共数据与 shuffle | 来源字段适配、整数 policy、重复行、留出、回放窗口、两阶段 shuffle、整文件 source reader 与消费游标有测试；来源 replay 检查通过。 | 普通优化器绕过 source reader；整个目录／元信息结构也不能直接声明为来源格式等价。 |
| 推理与模型交换 | 固定来源 CUDA/cuDNN 实现已实际接入，模型描述增加四结果及 MuZero 扩展；FP32／FP16 多 batch 对照通过。 | 不等于支持来源全部模型版本与全部后端。实际吞吐与来源等价仍待测。 |
| 发布与恢复 | accepted 模型与 learner checkpoint 分开；拒绝候选不回滚 learner。阶段恢复、原子发布及选定 GPU 中断续训检查提供支持。 | 不据此承诺任意崩溃时点或断电下的跨文件事务。 |
| 比赛与 Elo | 独立开局、执色轮换、胜负／和棋计分、gatekeeper 提前判定有测试。vendor `elo.py` 与固定来源字节相同。 | 没有运行正式棋力比较；研究工具的等时间取样仍是独立口径。 |
| 交互棋钟与 HumanSL | 矩形 Go、SGF、PositionSample、JSON 优先级／多 worker／停止、棋钟及第二模型路径已有测试；**humanPrior 有上述 P2**。 | 全部 GTP 扩展的逐命令、逐错误响应来源对照尚未穷尽，不能称为完整协议对齐。 |
| 配置文件与运行组织 | 引用、重复键、部分来源别名、模式缺省有实现和测试；**公共配置有上述 P2**。 | 自有阶段目录和 checkpoint 包装需要逐项说明用途，不能整体归入外层同步的豁免。 |

以下边界没有被误列为缺陷：多棋类与四结果；约定排除的 score／ownership／seki 和完整围棋增强；Gumbel tree 与严格根配额；MuZero latent 转移、不读取真实后继、不使用真实棋盘 graph／pattern／SVB、不跨手复用；同步外层调度与单卡 learner。未来棋盘辅助目标不在被排除的空间监督之列，当前实现保留了它。

MuZero 还硬性要求 `root_symmetries=1`。它明确拒绝配置，没有伪装支持；但目标表没有直接列出这项限制。多朝向输出如何对应 latent 及后续 recurrent 状态需要单独确定，现阶段标为待核实，不擅自认定必要豁免或要求简单平均 latent。

## 代码组织与维护成本

### 来源代码与项目实现的边界

`cpp/external` 本身是正常布局，问题在于其中的 `katago` 已同时承担外部来源和本项目核心实现两个角色。

排除其嵌套第三方依赖后，对 `cpp/external/katago` 的文件逐一比较，92 个文件中 86 个与固定来源字节相同，4 个已经修改，2 个是本地新增。修改集中在 `neuralnet/desc.cpp`、`desc.h`、`cudaandrocmbackend.inc`、`nninterface.h`；新增的是 `muzerohelpers.cu/.h`。这些扩展本身可以合理，但其维护责任已经属于 EtaZero。

Python 也有同类混合：`_vendor/katago/model_pytorch.py` 有实质适配；`modelconfigs.py` 将派生预设生成移到了项目模块；`training_data_generator.py` 注入可恢复的 RNG；融合算子修改注册命名空间。相对地，`elo.py` 保持来源原样。不能把整个 `_vendor/katago` 视为无需理解和维护的黑盒。

建议将原样第三方库继续放在 `external`／`_vendor`，将持续维护的来源派生代码放入对应生产模块，保留来源提交、许可证和可追溯对应关系。无需在活动源码旁再放一份永不调用的整套 KataGo；固定外部 checkout 已可用作对照。

### 职责划分与重复实现

`cpp/src/muzero` 中包含模型后端、batcher、搜索和 Gumbel 桥接，采用算法目录并非错误；但当前 AZ 按职责分布，MZ 按算法聚合，形成两种并列的导航方式。寻找同一项预算、HumanSL 或分析逻辑时，需要跨目录再找一套实现。

`cpp/search/search.cpp` 为 1,044 行，MuZero 搜索为 321 行；后者有 30 行超过 200 字符、最长 505 字符。MuZero 行数少不意味着流程简单，它仍独立维护 worker、selection、backup、根设置、分析与时间预算的组织。公共数学已有共享，这是好的基础；需要进一步明确哪些控制流程必须一致，哪些必须保留不同。

构建依赖也反映了职责混合：`etazero_go_rules` 同时包含通用 core、SGF、metadata、NN inputs 与 local pattern；CUDA engine 依赖该目标；`etazero_core` 又包含搜索、自对弈、writer 和组批；`etazero_muzero` 依赖整个 core。仅移动文件不会解除这些关系。

此外，`main.cpp` 同时承担配置到搜索参数的转换、模型服务装配和命令编排，`session.cpp` 混合可变棋局、交互和 GTP 行为。相比改目录名，拆开这些实际职责更能降低审查成本。

### 建议的目录方向

下面是候选组织，不是本轮已实施的迁移：

```text
cpp/
  core/                     通用小工具与配置基础
  game/                     各棋类规则和状态适配
  search/                   params、selection、backup、results、time
    alphazero/              真实状态、图、跨手复用
    muzero/                 latent 节点和 recurrent 展开
    gumbel/                 completed Q 与严格配额调度
  neuralnet/
    alphazero/              普通推理请求
    muzero/                 initial、recurrent、设备 latent
    cuda/                   维护中的来源 CUDA 派生实现
    torch/                  参考后端
  dataio/                   SGF、起点、训练记录与 writer
  program/                  selfplay、match 和服务生命周期
  command/                  GTP、analysis 与薄 CLI 入口
  external/                 原样第三方依赖
  tests/

python/etazero/
  models/                   来源模型与 MuZero 三组件
  data/                     字段、目标、shuffle、reader
  train/                    损失、优化器、内部时钟、checkpoint
  run/                      同步外层调度与发布
  evaluation/               比赛、gatekeeper、Elo
  _vendor/                  原样来源模块
```

优先保留 KataGo 的职责和函数对应关系，不强制逐文件同名。公共 budget／selection／backup／结果处理只有在语义相同时才提取共享；MuZero 的 latent 所有权、真实根合法性、吸收态与 Gumbel 配额保持明确边界。无需先设计通用策略插件框架，也不应把热路径每一步都改成动态分发。

## 是否直接以 KataGo 为基础重建

| 方案 | 当前收益 | 主要成本 | 判断 |
|---|---|---|---|
| 保持现有布局并持续逐点对齐 | 迁移量最小，现有功能保持可用。 | 来源调用链被拆散，公共控制存在多份，后续每次修正仍需重新找对应位置。 | 可以短期修问题，不适合作为长期唯一策略。 |
| 整套复制 KataGo 后重新接入 EtaZero 功能 | AZ Go 的原始职责和调用关系更完整，便于直接 diff。 | 多棋类、四结果、Gumbel、MuZero、数据与恢复必须重新穿过深度 Go 化的边界，容易丢失已验证能力。 | 当前不建议一次性替换。 |
| 按子系统收敛成保留来源结构的维护分支 | 保留已验证扩展，同时降低共有机制的重复维护与审查成本。 | 需要按职责迁移并逐段建立来源与端到端验证。 | 建议路线。 |

把 KataGo 搬过来确实能减少“重新实现共有机制”的成本，但不会自动消除适配成本。来源 search 与 `Board/BoardHistory`、NN 输出、score utility 和训练记录紧密相连；EtaZero 的扩展会穿过这些边界。MuZero 也不能用更换 `Board` 类型的方式接入来源搜索。

因此建议采用 KataGo 作为共有机制的结构基础，逐步扩大直接复用；已有 Go 规则、模型与 CUDA 后端实际上已经走在这条路上。重点是减少同一机制的第二份实现，并保留来源调用时机，而不是再复制更多未接入的文件。

如果未来研究范围明确缩为主要研究 AZ Go，整套 fork 的吸引力会更大；在当前多棋类、四结果、Gumbel、MuZero 都要保留的范围内，逐子系统收敛更合适。本次未自行改变研究范围或创建新版本。

建议实施顺序是：先修正上述四项具体差异；再拆分并规范关键搜索与 learner 流程，清楚标记维护中的来源代码；随后迁移目录和构建目标。把纯移动、行为修复与性能调整分别验证，避免在一次大改中失去差异来源。完成这些后，再对相同网络、棋局、预算、精度和线程配置运行来源对照与性能测量。

## 验证结果

环境为 conda `pytorch`，Python 3.10.20、PyTorch 2.12.0+cu132、CUDA 13.2、cuDNN 9.20，GPU 为 RTX 5080。构建与以下检查均针对本次 V3.3 工作树。

| 检查 | 实际结果 | 证据范围 |
|---|---|---|
| `bash scripts/build.sh` | 成功 | 重新编译原生核心、命令与 CUDA 后端。 |
| CTest | 18／18 通过 | 规则、搜索、采样、并发、Gumbel、MuZero 等原生测试。 |
| `python -m pytest -q tests` | 916 通过，427 跳过，215 warnings | 常规套件；跳过项不能计为通过。 |
| 选定真实 GPU 测试 | 64 通过，29 warnings | 四棋类 AZ／MZ、FP32／FP16、动态 batch、来源预设、HumanSL；两条小规模训练／恢复链路。 |
| 13 项固定 KataGo 来源检查 | 全部返回成功 | graph、search corrections、Go graph、SVB、Go bias、sampling、sampling weights、forks、Q、clocks、replay、persistence、reference formulas。 |
| 来源网络预设前向与反向 | 68 个通过 | CPU，5×5 小输入；最大前向绝对差 `4.76837158203125e-7`，最大梯度绝对差 `9.5367431640625e-7`。未运行全部派生配置枚举。 |
| 来源 Muon 与 LR／WD | 4 次更新、624 项 LR／WD 对照通过 | 小规模优化器数值与状态对照。 |
| 定向审查探针 | 重现上述配置拒绝、默认缓存容量、优化器分流和缺失 humanPrior | 当前生产入口的实际行为，探针没有修改生产代码。 |

真实 GPU 命令为：

```bash
ETAZERO_GPU_TESTS=1 conda run --no-capture-output -n pytorch python -m pytest -q \
  tests/test_cuda_backend.py \
  tests/test_muzero_cuda_backend.py \
  tests/test_gpu.py::test_gpu_pipeline_and_idempotent_completed_resume \
  'tests/test_muzero_pipeline.py::test_cuda_complete_pipeline_and_resume[nbt-True-False]'
```

来源网络检查使用 `tests/reference/check_katago_architectures.py --backward`；Muon 检查使用 `tests/reference/check_katago_muon.py`。来源检查的具体入口和参数以各脚本为准，KataGo 路径均指向本报告固定来源。

本次完整日志、JSON 结果及定向探针位于 `/tmp/etazero-v33-audit-20261024/`，属于临时验证产物；本报告已保留关键结果与触发条件。测试中的 warnings 主要包括 TorchScript tracing 与弃用提示，不能据此证明所有动态形状均安全。

这些结果支持当前工程可运行、被覆盖的公式和若干端到端链路正确，不证明全功能严格等价、所有中断恢复组合正确、等时间性能相同或棋力收益。尤其是来源检查常通过抽取函数建立 oracle，不能替代整套搜索／数据／训练调用链的对照。本轮未运行正式训练或棋力实验，也未读取历史目录来补齐 MuZero／其他来源的验证。
