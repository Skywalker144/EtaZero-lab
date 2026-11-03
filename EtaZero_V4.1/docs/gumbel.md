# Gumbel 搜索

当前支持 **Gumbel 根 + 可选 PUCT/Gumbel 非根**，适用于已接入的 Go、五子棋、Hex 和象棋普通／MuZero 训练链。数学来源与固定版本见 [reference_sources.json](../reference_sources.json) 中的 Mctx 项。

## 搜索与目标

根搜索每次从新树开始。第一次 NN 评估计入来源的根访问与 playout 计数，因此 `min(maxVisits,maxPlayouts) - 1` 是可分配给根动作的预算；搜索倍率和 pondering 使用对应的有效上限。根动作按 Mctx 的 considered-visit 序列进行 sequential halving，评分为 `log(prior) + Gumbel + completed-Q`。

completed Q 在已访问动作上使用来源子节点效用统计；未访问动作使用原始 NN 价值与已访问动作的先验加权 Q 混合。混合权重使用已完成的访问次数，再按配置进行节点内 min/max 缩放和 `(cVisit + maxVisits) * cScale` 缩放。价值转换到父节点行动方视角，待完成访问不进入 Q 混合。运算在合法/允许动作域上进行，棋盘外点和非法 pass 槽等不参与 min/max；来源数值对照也使用相同动作域。

默认落子为已完成访问次数最多的候选中评分最高者。训练策略独立计算为 `softmax(log(prior) + completed-Q)`，不含根 Gumbel 噪声，也不等于访问分布。策略覆盖未展开的合法动作，沿来源 writer 量化；Q 监督仍只包含实际搜索过的动作。主局面、side/reanalysis 与 surprise 使用同一个改进策略入口。

`gumbelSampleVisits` 可显式选择按完成访问次数及来源落子温度采样；零动作访问时使用先验。它仅改变实际出手，训练策略仍为改进策略。未启用 HumanSL 最终策略混合时，默认 SH 出手不应用落子温度；启用后的选点规则见下文。

非根可继续使用来源 PUCT，或选择 Gumbel 的确定性分配：在允许动作中最大化 `pi'(a) - (N(a)+pending(a))/(1+sum(N+pending))`。`pi'` 使用同一 completed-Q 改进策略；不采样新的 Gumbel 噪声。单线程时对应 Mctx 的非根规则，在途访问只影响分配欠额，不进入 completed Q 或训练目标。按深度生效的 avoid 动作仍受约束；没有允许动作时沿来源路径回传当前 NN 价值，不制造非法展开。

## 并行和来源机制

根配额由独立的预约、完成和取消计数管理，避免多个搜索线程重复领取同一配额。跨 considered-visit 层级前等待当前在途访问完成；失败的树展开释放预约后重试。停止信号和棋钟可提前结束搜索，已发出的访问完成后收束；此时结果使用实际完成次数，不声称完成了原计划的所有淘汰阶段。

根选择绕过 PUCT forced playout、根 FPU、LCB 出手奖励和策略目标修剪。根噪声使用 Gumbel；`rootNoiseEnabled=false` 时关闭该噪声，不叠加 Dirichlet 或根策略温度。hint 的先验调整及允许动作掩码仍生效。节点、批量 evaluator、虚拟损失、结果统计、uncertainty、noise pruning 和 SVB 继续沿用来源实现；选择 PUCT 非根时，其原有选择路径保持不变。因此这里是 KataGo 统计上的 Gumbel 扩展，不是整个 Mctx 搜索器的逐位复制。

未启用 HumanSL 最终策略混合时，分析报告的根选择权重表示完成的边访问次数，LCB 只供诊断；SH 模式把当前胜出动作列为首项，并使根 PV 首手与其一致。启用混合时报告混合后的选择权重，边访问字段仍是实际计数。根结果直接报告节点汇总值，不再用 PUCT 根修剪重算。PUCT 非根报告沿用来源选择值；Gumbel 非根报告已展开动作的改进策略，并在启用时应用来源 HumanSL 混合，不加 LCB 选点奖励。稠密训练策略通过独立接口产生，避免破坏来源报告数组与已分配子节点的对应关系。

来源 root focus 已接入普通与 MuZero 的 Gumbel 搜索：仍按原生概率和相对权重抽取合法、未被 avoid 的目标，执行不计入根边、根访问或 playout 预算的额外搜索。其子树会获得搜索结果，但本次不直接向根回传；后续正常访问可以使用更新后的子树估值。Gumbel 的该次预约会取消并重试，完成配额只统计正常访问，淘汰阶段仍等待所有在途预约结束。展开冲突重试保持同一次 focus 选择，避免额外访问误占严格配额。动态设置、清空、旧 focus 对象延迟回收及概率上限沿用来源机制。JSON `focusMoves`／`set_focus` 与 GTP `focus` 均可使用；额外搜索会影响耗时及后续 Q，不承诺与关闭 focus 时出手相同。

当前要求 `useGraphSearch=false`。根对称剪枝和根多对称 NN 平均可以使用。

## 根对称剪枝

`rootSymmetryPruning` 沿用来源 `beginSearch` 的对称识别与代表选择，再将来源 `isAllowedRootMove` 限制传给 Gumbel 候选域。sequential halving、completed Q、改进策略和最终选点均使用同一代表动作域；保留代表本身的 NN 先验，并在该域归一化，不把被剪掉副本的先验累加给代表。这与来源 PUCT 的候选限制一致，不改变 SH 配额公式。

Go 的劫／超级劫、encore 状态和矩形棋盘限制沿用来源判断；Hex 只使用恒等和 180°，象棋只在完整规则历史保持镜像时使用左右镜像。来源 `setRootSymmetryPruningOnly` 和 avoid 仍决定可用代表。focus 指向被剪掉的副本时，沿来源映射到实际搜索的代表；HumanSL 探索与最终混合也受同一限制。

JSON／GTP 分析仍由来源报告代码展开对称动作，并用 `isSymmetryOf` 标注副本；副本展示代表的访问与变换后的 PV，不能把这些重复行相加当作额外搜索量。实际根配额只统计代表边。开启剪枝时，稠密策略接口覆盖全部允许的代表，包括未展开代表；来源训练 writer 明确要求 `rootSymmetryPruning=false`，正常自对弈训练继续在完整合法动作域产生目标，不人为复制或均分对称目标。

## 树复用边界

Gumbel 按约定使用 fresh tree，每次 `beginSearch` 都清树，`makeMove` 也先清掉旧树；同一根停止后重启、ponder 切换与跨手搜索均不继承旧树访问和来源 `effectiveSearchTimeCarriedOver`。普通 AlphaZero 的来源树复用保持原状；MuZero 不跨真实落子复用 latent 树属于已约定差异。evaluator 的独立预测缓存不因 Gumbel 根配额重建而改写成另一套缓存。

固定 Mctx 的 `policies.gumbel_muzero_policy` 从本次 root 输入和新 Gumbel 噪声发起搜索；`action_selection.gumbel_muzero_root_action_selection` 用根边访问之和作为 considered-visit 表的阶段索引。当前 C++ 调度器也从零预约并要求完成计数与实际根边访问一致。KataGo 的原生跨手提升会保留子树访问及加权价值；直接保留这些访问、只重置 SH 调度器，会使阶段索引、剩余预算和最终胜出条件不一致，旧访问还会进入 completed Q 的混合与缩放。只清根边计数也不是等价修复，因为子树仍包含旧统计。

因此保留 fresh tree，不为继承树访问重新定义 SH 阶段、预算和 completed Q。Gumbel 的同根续搜及跨手复用不属于当前接入要求；这是 [目标约定](katago-target.md) 中明确允许的算法差异。普通 PUCT 继续使用来源树复用，不能将这一差异扩展到其他共有机制。

来源常规 selfplay 本就要求清树，使新根噪声生效；但零训练权重的 cheap-search 路径可允许复用，Gumbel 仍清树。比赛、GTP、analysis 及该 cheap-search 分支因此与来源复用行为有差异，不能据常规训练清树推断所有路径已经对齐，也不能据固定 visits 推断相同耗时或棋力。

## Visit-cap 对手建模

普通和 MuZero 的 Gumbel 搜索支持来源 `visitCapContempt` 的非根路径。受限方的节点达到阈值时，沿用来源逻辑保存子节点权重分布；随后使用来源的分布欠额公式匹配快照，优先于 PUCT、Gumbel 非根和 HumanSL 探索。虚拟损失、并行发布和当前子节点价值的回传保留来源实现。快照固定访问分配，不冻结子节点价值；noise pruning、uncertainty 和 value weighting 的相关影响仍按来源配置生效。

`visitCapContemptPla` 省略时限制当前搜索服务方的对手；常规搜索的服务方为根行动方，ponder 则延续上次服务方，尚无记录时假定为根的对手。也可用原生字段显式指定 B／W。因此同一配置在不同执色或 ponder 下可能限制到根节点。

**受限方为根行动方时，快照分配与严格 SH 配额冲突，此组合明确拒绝。** JSON analysis 在整组指定 turn 入队前检查，不留下半组请求；GTP 的同步、可取消和持续分析请求在输出成功头和启动搜索前检查。后台 ponder 在原生控制锁内、worker 空闲时检查，冲突时记录 `Ponder search not started`，不改变生效配置，之后可继续合法搜索。普通 PUCT 的根 visit cap 不受此限制。

## 次级 HumanSL

普通网络的 Gumbel 搜索支持来源次级 `NNEvaluator`，JSON analysis 和 GTP 使用原生 `-human-model`。Go metadata/profile、无 metadata 的其他棋类网络、HumanSL PUCT 系数及最终策略混合沿用来源接口和公式。MuZero 的 `Search`／`AsyncBot` 也支持次级网络：普通网络只用于真实根，独立 MuZero 网络可以沿 latent 路径提供树内策略；具体边界见 [MuZero 次级 HumanSL](muzero.md#次级-humansl)。JSON／GTP 使用同一 `-human-model` 入口加载次级文件或 bundle 目录；普通主模型也可以对真实节点使用次级 MuZero 的 initial 预测。

根允许 `humanSLRootExploreProbWeightless`：按 HumanSL PUCT 选择动作，更新子树而不回传根或增加根边计数；对应 SH 预约取消。它与来源一致，消耗 playout 预算，因此预算用完时 SH 可以提前结束，报告实际完成的配额。root focus 仍是额外且不消耗 playout 预算的搜索，两者同时命中时保持 focus 的指定动作。`humanSLRootExploreProbWeightful>0` 会改写严格配额分配，入口明确拒绝这一组合。

非根支持己方／对方的 weightful 和 weightless 探索比例。命中 HumanSL 探索时使用来源 HumanSL PUCT，未命中时使用配置的 PUCT 或 Gumbel 非根规则；weightless 回传边界沿用来源行为。

`humanSLChosenMoveProp>0` 时，默认 SH 胜出动作作为单点基础策略；`gumbelSampleVisits=true` 则使用完成访问分布，零动作访问时使用完整合法先验。随后应用来源 HumanSL 策略混合、PIKL、pass 保留选项和落子温度。实际选点允许来源支持的未展开合法动作；分析报告仍与已分配子节点对应，不伪造其访问次数。最终混合不改变独立计算的 Gumbel 改进训练目标或 surprise。

## 配置与验证入口

外层 `algorithm=gumbel_alphazero` 选择根算法；直接运行原生命令时使用 `rootSearchAlgorithm=gumbel`。`nonrootSearchAlgorithm` 可设为 `puct`（缺省）或 `gumbel`；Gumbel 非根要求 Gumbel 根，入口明确拒绝混成一个未定义的 AlphaZero 配置。其余参数由 [setup.cpp](../cpp/program/setup.cpp) 解析，数学与预约实现在 [gumbel.cpp](../cpp/search/gumbel.cpp)，来源树接入在 [searchgumbel.cpp](../cpp/search/searchgumbel.cpp)。外层不复制搜索逻辑，也不另设 learner。

原生测试覆盖手算 mixed-Q、预约取消与精确配额、真实搜索树的单/多线程路径、停止与异常收束，以及策略写入。`tests/reference/check_mctx_gumbel.py --source /path/to/pinned/mctx` 执行固定来源的函数体，对比根计数、胜出动作、策略以及逐次非根选点。该对照使用确定先验/Q 和零噪声；不证明随机数逐位一致，也不覆盖所有浮点极值。C++ 使用 double 和正数先验下界，极端 logits 与 Mctx 的 dtype/截断可有差异。

`tests/test_gumbel.py` 检查三个棋类、两种非根选择的原生自对弈及稠密目标；设置 `ETAZERO_GPU_TESTS=1` 可运行同时启用 Gumbel 根与非根的 Renju 两轮 CUDA FP16 产样、更新、导出、发布与再次加载检查。短闭环验证执行与数据语义，不是棋力或性能结论。

`tests/test_gumbel_focus.py` 检查六种棋类的原生 JSON／GTP focus、避开动作、精确根配额与额外子树访问；CUDA 检查覆盖 Go、Renju、Hex、象棋的普通 FP32 和 MuZero FP16。原生 Gumbel 测试另覆盖 1/4 线程、PUCT/Gumbel 非根、无效目标、取消 focus、停止与异常后的预约清理。动态更新与取消共用 `tests/test_analysis.py` 和 `tests/test_muzero_analysis.py` 的协议检查。

`tests/test_gumbel_humansl.py` 检查六种棋类、两种非根选择、HumanSL 探索组合、预算及实际边访问；CUDA FP32／FP16 检查使用两个不同模型，覆盖 Go metadata、Renju、Hex、象棋与 GTP 实际出手。原生测试另以手算 PIKL 和混合分布核对公式，并检查多线程配额、训练目标独立性及连续落子。这些检查验证双网络机制，不代表人类棋风或棋力效果。

`tests/test_gumbel_visitcap.py` 检查六种棋类、PUCT／Gumbel 非根、显式受限方、多 turn 请求、focus、根冲突及 GTP 错误后的继续搜索。原生普通网络测试覆盖单／多线程、HumanSL 和跨手／ponder；MuZero 的 CUDA FP32／FP16 原生测试检查 latent 节点的实际快照分布与价值回传。快照验证采用来源测试的访问分布不变量及并行误差范围，另核对 SH 计数与实际根边访问一致。

`tests/test_gumbel_symmetry.py` 检查六种棋类的 JSON／GTP 剪枝与动态开关、focus、avoid 和对称报告；MuZero CUDA FP32／FP16 覆盖 Go、Renju、Hex、象棋。原生搜索测试另检查 1／4 线程、PUCT／Gumbel 非根、矩形 Go、零动作访问下的代表先验目标、受限对称群、HumanSL weightless 探索及落子后的重新搜索，核对实际代表边与 SH 计数。
