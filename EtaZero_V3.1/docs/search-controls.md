# 通用搜索控制

实现参照 `KataGo d91ea855110dae533f0aada947b2b7d78cc8a4e1`。自对弈字段位于 `selfplay.cfg` 的相应组；交互和比赛字段平铺在 engine／analysis／match profile。解析缺省和合法范围以 [config.py](../python/etazero/config.py)、[engine_config.py](../python/etazero/engine_config.py) 为准，运行预算仍由所选 profile 决定。

## 根探索与历史

`[search_extras]` 提供根对称剪枝、wide root noise、futile visits 和 visit-cap contempt。根对称剪枝只保留棋规、历史及动作限制下等价的一着：围棋沿用来源的劫／超级劫处理，Gomoku 使用 D4，Hex 使用同色保向的恒等／180°旋转，象棋使用左右镜像并检查历史。它和根 NN 多对称平均是不同机制。

Wide root noise 调整探索先验，并以一半概率加入半正态 utility 奖励。Futile visits 根据剩余次数及棋钟估计，排除无法获得足够访问的根候选。Visit-cap contempt 达到阈值后冻结指定执棋方的子边权重比例，以配额差继续探索；它不直接截断整棵搜索。未指定执棋方时作用于搜索服务方的对手，pondering 保留服务方身份。

`[search_history]` 的 `ignore_pre_root_history` 只让 NN 看见搜索根之后的历史；`ignore_all_history` 隐去全部历史特征。棋规历史、合法性、终局和图身份仍完整保留。MuZero 在真实根编码时应用遮蔽，后续按 latent 动作展开。

PUCT 同分时保留先分配的 child；新着只由最高先验的未展开动作竞争，同先验按动作顺序，分数严格更高才替换已有 child。Gumbel 保留自身根候选和 sequential halving，不套用 PUCT 根的噪声、强制访问或冻结配额；两种非根算法均支持动作限制及 focus，并在配置 HumanSL 或 visit-cap 时应用相应可选控制。

## 图评估缓存与并行

`[graph_search] use_eval_cache` 在真实棋盘图搜索中保存已搜索节点的 WDL、访问权重和首次探索信息。缓存条目在一次搜索完成后发布，在后续搜索中按来源的访问比率衰减；它与按完整输入缓存 NN 输出的缓存分开。新局和更换模型清空搜索评估缓存。MuZero 保留不使用真实棋盘图共享的边界。

`node_table_shards_power_of_two` 控制图节点表分片，`nn_mutex_pool_power_of_two` 独立控制推理缓存锁池，后者为 −1 时禁用缓存。AlphaZero 和 MuZero 的 initial cache 均接通该控制。`[parallelism] min_playouts_per_thread` 依据本次剩余新访问数减少活动线程；已有严格预约预算仍适用。

## 重复模式的 utility 惩罚

`[patterns] avoid_repeated_pattern_utility` 从当前真实对局重放搜索服务方的着法，构建局部模式惩罚。模式在落子前取键，每局按对称及颜色变换去重；其修正进入搜索 utility，不直接写入纯 WDL 目标。树推进后根自身的模式奖励清除，保留子树按当前模式表重新聚合。

围棋使用来源的 9×9 LocalPatternHasher、绝对落点、棋盘尺寸和完整棋串的打吃标记，排除 pass 和形成简单劫的提子。Gomoku 用落点附近 9×9 棋子，Hex 的换色同步转置目标方向，象棋使用起终点附近的棋子类型并在换色时旋转棋盘。后两者使用各棋类有效的变换。MuZero latent 节点没有真实棋盘，明确拒绝重复模式配置，和真实棋盘 SVB 的适用边界相同。

`pattern_corpora` 是 JSON 数组，每项指定 `paths`、`penalty`，可选 `decay`、`min_turn`、`max_files`、`players`。各组独立按修改时间从新到旧加载；目录递归读取 `.patterns` 文件。相对路径按 profile 目录解析。每行是一条完整或部分真实动作序列，分支可分别写成多行；同一文件的模式只计一次。格式示例：

```json
{"size":9,"rule":"go","black":"player-a","white":"player-b","komi":7.5,"moves":[0,1,9,10],"comments":["","","%SKIP%",""]}
```

动作使用实际棋盘编号；Go pass 为 `size²`，象棋使用固定动作表并可给 `fen`。`comments` 可省略，包含 `%SKIP%` 的着法不加入惩罚；`players` 根据 black／white 名字筛选。`go_rules` 和 `komi` 可随记录指定。损坏记录会报告并跳过，缺失文件报错。外部记录仅用于加载，不删除或改写原始文件。

## 第二策略网络

`[human_sl] human_model` 指定独立导出模型；布局须与主模型匹配。该网络只提供探索／最终混合策略，价值回传继续来自主模型。根、根执棋方和对手各有 weightless／weightful 探索概率；weightless 在该边及以上不计访问和价值，其下的搜索仍更新，并仍消耗本次 playout 预算。独立的 HumanSL cpuct 常数和 permanent 项控制这种探索。

最终落子可将第二策略按已搜索 utility 作 PIKL 指数偏移，再按 `human_sl_chosen_move_proportion` 混入选择权重；未展开动作使用已有 child utility 的简单均值。`human_sl_chosen_move_ignore_pass` 保留原来的 pass 质量。零 PIKL lambda 使用温度趋于零的极限，避免除零。混合在落子温度及整数量化之前执行，也反映在分析和棋钟判断中。

AlphaZero 可以在每个真实节点调用第二模型。MuZero 的非根第二策略必须来自另一个 MuZero 模型，两个 latent 沿相同动作独立演进；AlphaZero 第二模型可用于 MuZero 的真实根及最终策略混合，配置其非根探索会明确报错。已有零 metadata 的适配保持不变；使用该能力需要实际可用的第二策略模型，不自动生成或训练人类模型。

按深度限制、focus、棋钟和 pondering 的命令格式见 [原生交互引擎](engine.md)。核心实现入口为 [AlphaZero 搜索](../cpp/src/search/search.cpp)、[MuZero 搜索](../cpp/src/muzero/search.cpp)、[时间预算](../cpp/src/search/time_budget.cpp) 和 [模式表](../cpp/src/search/pattern_bonus.cpp)。
