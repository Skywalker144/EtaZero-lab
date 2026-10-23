# 原生交互引擎

GTP、JSON 分析和 Web 使用同一个原生会话，管理已发布推理模型、真实棋盘、历史、开局和搜索。输入线程接收命令，一个工作线程顺序执行棋局操作与搜索；停止请求直接设置取消标记，等待正在执行的推理和搜索工作退出后再改变棋局。推理模型只加载一次。

交互入口使用 `analysis.cfg`，训练、比赛和 arena 继续直接调用原有搜索接口。网络结构、训练配置身份、样本和 checkpoint 格式不受协议影响；诊断默认关闭，自对弈不收集候选分析或发送过程报告。已有运行仍按原有配置及状态恢复。

## 命令行

在版本目录执行，`--model` 指向带 manifest 的已发布 TorchScript，或用 `--run-dir` 选择当前发布模型：

```bash
bash scripts/run.sh gtp --config-dir configs/az_g --run-dir data/my_run
bash scripts/run.sh analysis --json-stream --config-dir configs/az_g --run-dir data/my_run
```

原生入口为 `build/etazero gtp` 与 `build/etazero analysis --stream true --protocol json`，另需提供解析后的 `--config`、`--model`、`--model-id`、`--device` 和 `--seed`。Python 入口负责配置解析、模型校验和画布设置。

`analysis --stream` 保留简短文本命令入口；单局面 `analysis --moves ...`、`match` 和 `arena` 的使用方式见 [README](../README.md)。

## GTP

支持命令以 `list_commands` 为准。基础命令包括 `protocol_version`、`name`、`version`、`known_command`、`boardsize`、`clear_board`、`play`、`undo`、`genmove` 和 `quit`；保留可选数字编号、`=` / `?` 响应以及空行结束符。坐标跳过 I 列，A1 在左下角，范围为实际棋盘。执子必须符合当前行棋方；Go 支持 `pass` 与 `komi`；象棋使用 `a0b0` 形式的起终点坐标（红方底线为 0）。不提供 resign、分数估计或 ownership 输出。

`etazero-rule freestyle|standard|renju|hex|go|xiangqi` 切换棋规并清空棋局；`showboard` 返回当前状态 JSON。

```text
1 boardsize 15
2 etazero-rule renju
3 play b H8
4 kata-analyze w 10 maxmoves 10
5 stop
6 kata-genmove_analyze w 10
7 undo
8 kata-raw-nn 0
9 quit
```

- `kata-analyze [颜色] [间隔]` 运行一次持续搜索，直到 `stop`、后续 GTP 命令或输入结束。间隔单位是百分之一秒；也接受 `interval N`、`maxmoves N` 和 `visits N`，设置 visits 时按有限预算结束。间隔范围 0–6000，0 不发过程报告；maxmoves 只限制输出数量。
- `kata-genmove_analyze 颜色 [间隔]` 搜索时发送相同 `info`，正常完成输出 `play 坐标` 并落子。`genmove` 返回坐标。显式取消保留棋盘，不执行部分搜索选出的落子。
- info 含访问、根方得分率、和棋率、纯网络先验、搜索效用、采样权重、排序和主变化。`weight` 是子树有效采样权重，`selectionWeight` 是 PUCT 原始选点权重，两者不同。启用 PUCT LCB 时输出 `utilityLcb` 和半径；兼容字段 `lcb=(utilityLcb+1)/2` 是线性坐标转换，不能当成概率。Gumbel 不输出 LCB，`policy` 为改进策略。
- `kata-raw-nn [0]` 返回固定朝向四个网络策略头与原始价值 logits / WDL 的 JSON；仅支持朝向 0。它是 EtaZero 的策略诊断扩展，不提供 KataGo 的 Go 分数、ownership 或任意对称参数。

未支持的命令或分析参数明确返回错误。协议提供这些棋类所需的 GTP 子集，不声明覆盖 KataGo 的全部 Go 扩展。PUCT 按配置复用树；MuZero、Gumbel 保留禁止树复用和图搜索等原有约束。

## JSON lines 与批量局面

启动返回 `event: ready` 和模型画布。每行一个请求对象，必须包含字符串或整数 `id` 及 `command`；尚未完成的请求编号不能重复。所有输出也是单行 JSON，带同一编号。`event: analysis, final: false` 为过程快照，`event: result, final: true` 为最终响应，失败响应含 `ok: false` 与 `error`。一个请求的最终响应发出后方可复用编号。

```json
{"id":"board","command":"new","size":15,"rule":"renju"}
{"id":"move","command":"play","action":112}
{"id":"search","command":"analyze","visits":10000,"report_interval_ms":250}
{"id":"cancel","command":"stop","target_id":"search"}
```

Gomoku／Hex／Go 公共动作编号为实际棋盘的零起始行优先索引，不使用网络画布步长；Go pass 为 `size²`。象棋编号为 [固定动作表](../python/etazero/games.py) 的索引。`new` 接受 `size`、`rule` 和可选 `moves`，Go 还可传 `komi/go_rules`，象棋可传 `fen`；平衡开局用 `opening: balanced`、字符串 uint64 `seed` 与 `timeout_ms`，复用比赛开局配置，不能与 moves 同时使用。`undo` 的 `count` 默认为 1，`state` 返回状态，`raw_nn` 返回网络诊断，`genmove` 搜索后落子。

`analyze` 不落子。`analyze` / `genmove` 接受 visits、report_interval_ms 和 fresh；默认从新根开始，`fresh: false` 在算法允许且配置开启时保留树。`continuous: true` 仅适用于 analyze，覆盖访问、playout 和时间上限，直到停止。

停止可指定 `target_id`，或省略以取消当前及已排队的工作；被取消的请求仍返回最终响应。进行中的搜索排空后返回部分结果，尚未开始的请求返回 cancelled。取消 genmove 不落子。状态修改命令会先取消正在执行的工作，再顺序更新棋盘。批量分析可在各请求自带完整局面，按输入顺序排队；读到 EOF 后排空有限任务，取消持续任务：

```json
{"id":"a","command":"analyze","size":15,"rule":"renju","moves":[112,113],"visits":800,"report_interval_ms":0}
{"id":"b","command":"analyze","size":11,"rule":"hex","moves":[60,61],"visits":800,"report_interval_ms":250}
```

不支持的字段和非法局面明确报错。批量队列为单模型顺序搜索；不是多局面并行调度。

## 快照口径

所有候选统计都从根行棋方看：W/D/L 与 W−L 是未加入 SubtreeValueBias 的概率统计，得分率为 `W + D/2`；效用、LCB 与置信半径保留搜索实际使用的效用尺度。LCB 只在启用时输出。未访问候选的价值与 LCB 为 null，不用网络根值填充。PUCT 的原始选点权重来自实际选点函数，归一化后对应选择分布；Gumbel 显示改进策略和最终行为分布，过程中的最终行为分布为 null。

过程快照来自同一次搜索，不重启、不额外采样落子，也不生成训练目标。PUCT 读取已发布子节点及加锁统计，多线程快照的不同边可能来自稍有错开的时刻，访问合计按本次包含的已完成边计算；最终结果在所有工作退出后读取。MuZero 在其搜索互斥区内读取。过程报告会增加交互搜索开销，不应用其耗时与无报告运行做等时间比较。

主变化最多 32 手，按最大访问延伸；图搜索防止循环，MuZero 只发布可在真实棋盘重放的合法前缀，不把重复 latent 动作当成合法棋步。网络四头诊断只在正常结束后额外执行一次，不计入搜索耗时和 NN 请求数；被取消的搜索省略它。

实现入口：[会话与协议](../cpp/src/commands/session.cpp)、[快照提取](../cpp/src/search/analysis.cpp)、[Web 引擎](../../web/engine.py)。

## 动作限制与 focus

JSON 的 `analyze`／`genmove` 接受 `avoid_moves` 或 `allow_moves`，两者互斥。每项包含执棋方、实际棋盘动作列表和 `until_depth`；根深度为 0，在深度小于该值时限制生效。allow 每个执棋方只允许一项。`avoid_rescale_root` 控制是否将保留的根先验重新归一化。

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

`genmove` 使用当前方棋钟，普通分析不受对局棋钟约束。时间预算采用来源 TimeControls，包含 lag buffer、阶段系数、明显招法的策略熵／surprise 调整、读秒取整及 futility 提前结束；搜索中的动态预算最多每 0.1 秒刷新。复用子树按保留访问比例和 `tree_reuse_carry_over_time_factor` 折算此前搜索时间。时钟值由调用者更新，返回结果不会自行扣减远端时钟。

Ponder 使用独立 `max_visits_pondering`／`max_playouts_pondering`／`max_time_pondering`。配置 `pondering_enabled` 后，genmove 落子完成可自动在后台思考，下一条请求会取消正在运行的 ponder；明确的 `ponder` 命令返回搜索结果。停止和 EOF 会排空在途工作。MuZero 不跨真实步保留 latent，因此不会获得子树时间折算。

更多开关及第二策略模型配置见 [通用搜索控制](search-controls.md)。胜负报告不包含模式惩罚；utility 报告反映实际搜索修正。
