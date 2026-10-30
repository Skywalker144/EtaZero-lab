# EtaZero V4 的目标与偏差边界

基准为 [reference_sources.json](../reference_sources.json) 中固定的 KataGo `d91ea855110dae533f0aada947b2b7d78cc8a4e1`。本文件记录当前设计、适配边界与核对依据，可依据用户目标、来源实现和验证结果修正，不独立于项目定位定义需求。与用户当前要求不一致时，以用户要求为准；当前已实现范围见 [README](../README.md)，不能将复制了源码解释为全部适配已经完成。

V4 的定位是基于 KataGo 合理接入多棋类、Gumbel 和 MuZero 的“KataZero”。算法范围包括 AlphaZero、Gumbel AlphaZero、MuZero、Gumbel MuZero；EtaZero 提供外层同步逐轮调度以及实验管理、评估、绘图和 Web 等功能。外层负责何时运行与如何呈现，KataGo 及其必要扩展负责实际搜索、推理、产样与学习。外层适配不能成为重写共有算法或改变 learner 内部时钟的理由。

接入以必要的侵入范围为限：Gumbel、MuZero、多棋类、四结果及约定删除项在原生职责内修改；单卡数据调度、评估任务编排、Elo 调用、绘图、实验与配置目录管理在外围实现。外围功能优先使用原生命令、配置、日志与结果；确需原生进程内状态时只增加必要的接口，不另造搜索、训练或评分实现。

具体接入按算法语义与实际调用链判断。适用的共有机制保留原生实现；不适用于新算法或棋类的机制须解释原因，不能机械套用，也不能仅因本文件列出某项就改动来源代码。来源已有后端和工具尽量保留；新增算法、棋类向哪些后端和工具扩展，依据实际需求与适用性确定，不自动要求穷尽所有平台和功能组合。已声明支持的路径必须完整接通并验证，尚未支持的组合明确拒绝，不以静默回退代替支持。

训练图的算法与性能指标严格读取原生日志；缺失指标留空，不从 SGF、checkpoint 或训练数据补算，也不为绘图向引擎增加计数或汇总日志。等时间横轴可用明确标注来源的 EtaZero 外层计时。

## 允许的差异

- 保留围棋，增加 Freestyle/Standard/Renju 五子棋、Hex、中国象棋，按棋类适配状态、动作、历史、对称和必要的平衡开局。
- 使用 W/D/L/NR。普通和棋与 no-result 分开监督，非围棋屏蔽 NR；基础效用 W−L，D 和 NR 的贡献均为 0。
- 排除 score utility、score/ownership/seki 监督、随机贴目、让子补偿及完整 passing/dame/anti-mirror 增强。围棋规则必需的贴目、计分、pass、超级劫、合法性和终局保留。
- 增加 Gumbel 根搜索、completed Q、sequential halving、改进策略目标及可选 Gumbel 非根。使用 tree、关闭 graph 属于允许差异；严格根配额与冲突的 PUCT 根机制需要明确处理。Gumbel 使用 fresh tree，不要求同根续搜或跨手树复用；避免将旧访问直接混入本次 SH 阶段、预算与 completed Q。普通 PUCT 的来源树复用继续保留。
- 增加 MuZero 三组件网络、latent 搜索和展开训练。真实规则负责环境与根合法性；latent 节点不使用真实后继棋盘、真实棋盘 graph、重复模式或 SVB，不跨真实落子复用旧树。
- 网络输入/动作/四结果 heads 作必要适配；来源 backbone、参数角色、归一化、初始化及适用数值路径保留。
- 外层采用 EtaZero 同步逐轮调度，阶段间切换模型；保持单卡 learner，暂不接入 DDP 或新增分布式调度。来源内部 epoch/subepoch、文件选择、优化器、Lookahead、SWA 等时钟独立于外层轮次。
- 实验可以选择不同参数与开关组合；研究工具、Web 呈现和等时间实验口径保留各自用途。

## 默认应对齐的部分

PUCT/FPU/展开与回传、根噪声与温度、forced playout/target pruning/LCB、uncertainty/optimistic policy/noise pruning/SVB、图与缓存、跨手状态、并行预算与停止；PCR/cheap/reduce visits、surprise、policy init、fork/side/reanalysis、hint、训练权重与随机取整；适用的 policy/value/TD/Q/variance-time/future-position 目标、增强、loss 权重和归约；网络、metadata、优化器、训练时钟；公共数据、shuffle、预取和文件消费；推理、导出、模型生命周期；发布、恢复、比赛、Elo、交互协议、棋钟与 HumanSL；以及配置和文件组织的实际语义。

多棋类、Gumbel 和 MuZero 只豁免实际需要改变的步骤，不构成整体改写共有机制的理由。自有接口、格式或目录也不自动获得豁免。新增阶段可以明确保留尚未完成的范围，入口必须拒绝未接入配置，不用其他算法或棋类代替。

审查记录区分“已对齐”“约定差异”“未对齐”“待核实”，关联来源行为、当前执行路径、影响和验证依据。随机数与线程调度无需逐位相同，但分布、计数、更新频率和并行语义仍需核对。GPU 正确性、性能和棋力分别验证。
