# 原生交互引擎

原生引擎提供 KataGo 公共 JSON 分析、GTP 和 Web 多棋类会话。公共分析服务按优先级分配给 `numAnalysisThreads` 个常驻工作线程，共享批量推理服务；GTP 和 Web 的可变棋局由单工作线程串行管理，输入线程可取消搜索。停止时排空在途推理再改变棋局，模型只加载一次。

## 命令行

在版本目录执行，`--model` 指向带 manifest 的 CUDA `.bin` 或参考 `.pt` 模型，也可用 `--run-dir` 选择当前发布模型：

```bash
bash scripts/run.sh gtp --config-dir configs/az_g --run-dir data/my_run
bash scripts/run.sh analysis --katago-json --config-dir configs/az_g --run-dir data/my_run
bash scripts/run.sh analysis --json-stream --config-dir configs/az_g --run-dir data/my_run
```

原生入口为 `build/etazero gtp` 或 `build/etazero analysis --stream true --protocol katago|json|text`，另需解析后的 `--config`、`--model`、`--model-id`、`--device` 和 `--seed`。Python 入口负责配置解析、模型校验和画布设置。`analysis.cfg` 管理交互配置，不读取训练配置。

## KataGo 公共 JSON 分析

`--katago-json` 接受来源格式的 JSON lines，没有 ready 消息和私有 `command/event/ok` 包装。每行包含 `id`、`boardXSize`、`boardYSize`、`rules` 和 `moves`；棋盘可为矩形，宽高均不超过网络画布。支持摆子、初始行棋方、贴目、多个分析时点、优先级、参数覆盖、动作限制、focus、持续报告、PV visits 和 policy。

```json
{"id":"position","boardXSize":9,"boardYSize":7,"rules":"chinese","komi":6.5,"initialStones":[["B","A1"]],"initialPlayer":"W","moves":[["W","D4"]],"analyzeTurns":[0,1],"maxVisits":800,"includePolicy":true,"includePVVisits":true}
{"id":"models","action":"query_models"}
{"id":"cancel","action":"terminate","terminateId":"position"}
```

任务按 priority 从高到低执行，同优先级按提交顺序；EOF 排空有限任务。`query_version`、`query_models`、`clear_cache`、`terminate`、`terminate_all`、`set_focus` 对应来源控制接口；取消可限定 turns。每个分析时点有独立最终响应，进行中的报告带 `isDuringSearch: true`。`overrideSettings.humanSLProfile` 仅影响该请求，metadata 和 NN cache 按真实输入隔离。

响应含 `rootInfo`、`moveInfos`，可附 policy、humanPolicy 和 PV visits；存在 HumanSL 输出时，每个 `moveInfos` 项都含对应动作的 `humanPrior`，与 `includePolicy` 无关，并按对称等价着的实际坐标取值；使用来源根剪枝聚合值、边访问／节点访问区分及对称等价着报告。四结果扩展 `drawRate`，不把普通和棋并入 NR。`rawVarTimeLeft` 仅在后端暴露 variance-time 输出时提供，参考 TorchScript 后端不伪造该值。score/ownership/seki 相关请求明确拒绝，响应省略这些预测量。

## GTP

完整命令列表由 `list_commands` 给出。保留可选数字编号、`=` / `?` 响应和空行结束符。坐标跳过 I 列，A1 在左下角；围棋支持 `pass`，象棋使用 `a0b0` 起终点坐标。围棋的非交替着法按来源清除不适用的劫／轮次历史；其他棋类要求当前行棋方。

| 用途 | 命令 |
|---|---|
| 棋局与尺寸 | `boardsize N`、`boardsize X:Y`、`rectangular_boardsize X Y`、`clear_board`、`play`、`undo`、`set_position`、`showboard` |
| 规则与贴目 | `komi`、`get_komi`、`kata-get-rules`、`kata-set-rule`、`kata-set-rules`、`kgs-rules` |
| 参数与模型 | `kata-list-params`、`kata-get-param[s]`、`kata-set-param[s]`、`kata-get-models`、`clear_cache` |
| SGF 与让子 | `loadsgf`、`printsgf`、`fixed_handicap`、`place_free_handicap`、`set_free_handicap` |
| 搜索与分析 | `genmove`、`genmove_debug`、`kata-search`、`kata-search_debug`、`kata-search_cancellable`、`kata-analyze`、`lz-analyze`、`kata-genmove_analyze`、`lz-genmove_analyze`、`kata-search_analyze`、`kata-search_analyze_cancellable` |
| 规则计分 | `final_score`、`final_status_list` |
| 网络与诊断 | `kata-raw-nn`、`kata-raw-human-nn`、`debug_moves`、`kata-benchmark`、`cputime`、`gomill-cpu_time`、`kata-debug-print-tc` |

围棋 `showboard` 返回来源 ASCII 棋盘；其他棋类返回状态 JSON。`etazero-rule` 是多棋类扩展。`loadsgf 文件 [手数]` 保留摆子、规则、矩形尺寸及合法历史；`printsgf` 输出 SGF，指定路径时写文件。固定让子使用来源落点规则，自由让子使用网络策略。规则和运行时参数修改先验证，再整体生效；来源限制初始化后不可更改的缓存／分片参数仍拒绝修改。`alwaysComputePassAliveUnderSuicideRules`、`excludeTerritoryAdjacentToAtari` 接受 true/false/auto，auto 读取模型偏好；修改后从原始摆子及着法重新裁定历史。

`kata-analyze [颜色] [间隔]` 持续搜索至下一条命令、空行、stop 或 EOF；间隔为百分之一秒，可为小数。支持 `minmoves`、`maxmoves`、`pvVisits`、`pvEdgeVisits`、`rootInfo`、`noResultValue`、`allow/avoid` 和加权 `focus`。EtaZero 的 `visits` 扩展可令分析按有限预算结束。`kata-genmove_analyze` 结束后输出 `play 坐标` 并落子，`kata-search*` 只返回着法。cancellable 命令可由后续输入取消；取消不落子。

`analysisWideRootNoise` 和 `analysisIgnorePreRootHistory` 只用于持续分析，genmove 及 genmove_analyze 使用基础搜索参数。Kata 格式报告浮点 winrate，LZ 格式按 10000 缩放为整数。PUCT 可报告 LCB，即使它未参与落子选择；Gumbel 不使用 LCB。根报告使用剪枝聚合值，并区分子节点 visits/weight 与父边 edgeVisits。未访问候选以来源 FPU 提供 minmoves 报告，但私有 JSON 仍将未采样价值保留为 null。

`kata-raw-nn [颜色] all|对称编号 [optimism]` 和 human 版本返回来源样式的文本概率及策略，使用该棋类有效对称，不包含被排除的分数／归属。`debug_moves` 查看真实已搜索分支，诊断统计采用 JSON；`kata-benchmark` 使用来源内置 SGF 局面，报告本地搜索计数和墙钟，不将诊断文本格式或测试值当作 KataGo 吞吐等价证据。

`final_score/final_status_list` 对已完成棋局按真实规则计分和识别 pass-alive 区域。未终局的分数／死活估计需要已排除的 score/ownership 预测，因此明确报错。可配置 W−L 阈值、连续回合数和最低手数进行认输；依赖预测目差的让子保护与动态 PDA 补偿不在目标范围。完整 passing/dame/anti-mirror 仍按目标排除。非法或不适用的命令、选项明确返回错误。

## JSON lines 与批量局面

启动返回 `event: ready` 和模型画布。每行一个请求对象，必须包含字符串或整数 `id` 及 `command`；尚未完成的请求编号不能重复。所有输出也是单行 JSON，带同一编号。`event: analysis, final: false` 为过程快照，`event: result, final: true` 为最终响应，失败响应含 `ok: false` 与 `error`。一个请求的最终响应发出后方可复用编号。

```json
{"id":"board","command":"new","size":15,"rule":"renju"}
{"id":"move","command":"play","action":112}
{"id":"search","command":"analyze","visits":10000,"report_interval_ms":250}
{"id":"cancel","command":"stop","target_id":"search"}
```

Gomoku／Hex／Go 公共动作编号为实际棋盘的零起始行优先索引，不使用网络画布步长；Go pass 为 `width*height`。象棋编号为 [固定动作表](../python/etazero/games.py) 的索引。`new` 接受 `size`、`rule` 和可选 `moves`，Go 还可传 `width/height/komi/go_rules`，象棋可传 `fen`；平衡开局用 `opening: balanced`、字符串 uint64 `seed` 与 `timeout_ms`，复用比赛开局配置，不能与 moves 同时使用。`undo` 的 `count` 默认为 1，`state` 返回状态，`raw_nn` 返回网络诊断，`genmove` 搜索后落子。

`analyze` 不落子。`analyze` / `genmove` 接受 visits、report_interval_ms 和 fresh；默认从新根开始，`fresh: false` 在算法允许且配置开启时保留树。`continuous: true` 仅适用于 analyze，覆盖访问、playout 和时间上限，直到停止。

停止可指定 `target_id`，或省略以取消当前及已排队的工作；被取消的请求仍返回最终响应。进行中的搜索排空后返回部分结果，尚未开始的请求返回 cancelled。取消 genmove 不落子。状态修改命令会先取消正在执行的工作，再顺序更新棋盘。批量分析可在各请求自带完整局面，按输入顺序排队；读到 EOF 后排空有限任务，取消持续任务：

```json
{"id":"a","command":"analyze","size":15,"rule":"renju","moves":[112,113],"visits":800,"report_interval_ms":0}
{"id":"b","command":"analyze","size":11,"rule":"hex","moves":[60,61],"visits":800,"report_interval_ms":250}
```

不支持的字段和非法局面明确报错。批量队列为单模型顺序搜索；不是多局面并行调度。

## 快照口径

所有候选统计都从根行棋方看：W/D/L 与 W−L 是未加入 SubtreeValueBias 的概率统计，得分率为 `W + D/2`；效用、LCB 与置信半径保留搜索实际使用的效用尺度。PUCT 的 LCB 诊断与是否参与选点分开。未访问候选的价值与 LCB 为 null，不用网络根值填充。PUCT 的原始选点权重来自实际选点函数，归一化后对应选择分布；Gumbel 显示改进策略和最终行为分布，过程中的最终行为分布为 null。

过程快照来自同一次搜索，不重启、不额外采样落子，也不生成训练目标。PUCT 读取已发布子节点及加锁统计，多线程快照的不同边可能来自稍有错开的时刻，访问合计按本次包含的已完成边计算；最终结果在所有工作退出后读取。MuZero 分别读取节点边和统计快照，不用搜索级锁串行化整条路径。过程报告会增加交互搜索开销，不应用其耗时与无报告运行做等时间比较。

主变化最多 32 手，按最大访问延伸；图搜索防止循环，MuZero 只发布可在真实棋盘重放的合法前缀，不把重复 latent 动作当成合法棋步。Web／私有 JSON 的网络四头诊断只在正常结束后额外执行一次，不计入搜索耗时和 NN 请求数；被取消的搜索省略它。

实现入口：[会话与协议](../cpp/command/session.cpp)、[快照提取](../cpp/search/analysis.cpp)、[Web 引擎](../../web/engine.py)。

## 动作限制与 focus

JSON 的 `analyze`／`genmove` 接受 `avoid_moves` 或 `allow_moves`，同一执棋方不能混用。每项包含执棋方、实际棋盘动作列表和 `until_depth`；根深度为 0，在深度小于该值时限制生效。allow 每个执棋方只允许一项。`avoid_rescale_root` 控制是否将保留的根先验重新归一化。

```json
{"id":"restricted","command":"analyze","visits":100,"allow_moves":[{"player":"b","moves":[0,1],"until_depth":1}]}
{"id":"focused","command":"analyze","visits":100,"focus_moves":[0,1],"focus_weights":[3,1],"focus_probability":0.5}
```

Focus 按权重选择指定根动作，更新其子树，不计入根边访问／价值，也不消费普通搜索的 playout 配额；概率按来源限制到 0.99，保证仍有正常探索。对称别名映射到保留的代表动作。运行中可以发送 `set_focus` 并给出当前搜索的 `target_id`，原子替换这三个 focus 字段；空 moves 清除 focus。真实落子或新局会清除临时控制。

GTP 的 `kata-analyze`／`kata-genmove_analyze` 可追加 `avoid B A1,B2 2` 或 `allow W C3 1`，最后一个数是深度。`maxmoves` 仍仅限制报告条目。

## 棋钟与 pondering

支持 GTP `time_settings`、`time_left`、`kgs-time_settings`、`kata-time_settings`、`kata-list_time_settings`。标准 `time_settings main period stones` 表示加拿大读秒；KGS 支持 none、absolute、byoyomi、canadian。`kata-time_settings` 另支持 `fischer main increment` 与 `fischer-capped main increment max_time max_time_per_move`。`time_left player seconds stones` 更新对应方剩余时间；日本读秒中 stones 表示剩余周期数，加拿大读秒中表示本周期剩余手数。

JSON 提供同一状态接口：

```json
{"id":"clock","command":"time_settings","clock_type":"fischer","clock_arguments":[300,2]}
{"id":"remaining","command":"time_left","player":"b","time":240,"stones":0}
{"id":"ponder","command":"ponder"}
```

没有显式 maxVisits/maxPlayouts/maxTime 时，GTP 入口按来源设置 1 秒主时、5 次 5 秒读秒的初始棋钟；显式预算保持其原含义。`genmove` 使用当前方棋钟，普通分析不受对局棋钟约束。时间预算采用来源 TimeControls，包含 lag buffer、阶段系数、明显招法的策略熵／surprise 调整、读秒取整及 futility 提前结束；搜索中的动态预算最多每 0.1 秒刷新。复用子树按保留访问比例和 `tree_reuse_carry_over_time_factor` 折算此前搜索时间。时钟值由调用者更新，返回结果不会自行扣减远端时钟。

Ponder 使用独立 `max_visits_pondering`／`max_playouts_pondering`／`max_time_pondering`。配置 `pondering_enabled` 后，genmove 落子完成可自动在后台思考，下一条请求会取消正在运行的 ponder；明确的 `ponder` 命令返回搜索结果。停止和 EOF 会排空在途工作。MuZero 不跨真实步保留 latent，因此不会获得子树时间折算。

更多开关及第二策略模型配置见 [通用搜索控制](search-controls.md)。胜负报告不包含模式惩罚；utility 报告反映实际搜索修正。

公共配置接受 `numSearchThreadsPerAnalysisThread` 作为 `numSearchThreads` 的别名，两者不能同时出现（含 include 文件）。`nnCacheSizePowerOfTwo` 按来源指数转换为容量，`-1` 禁用缓存；未设置容量和锁池时按 analysis、GTP、match 模式取各自缺省。字段转换和缺省值见 [配置解析](../python/etazero/config_files.py) 与 [引擎配置](../python/etazero/engine_config.py)。
