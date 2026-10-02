# 固定评估与等时间 Elo

`evaluate` 做单局面搜索，`match` 做一对模型的可恢复比赛，`arena` 选择历史模型、安排比赛并输出 Elo。所有公开动作使用实际棋盘零起始行优先编号，native 内部才转换为网络画布编号。评估必须用 `--model` 或 `--run-dir` 明确指定模型来源，不从评估配置目录猜测训练目录。模型由 manifest 校验身份与内容，同场模型画布须相同，网络宽度和深度可以不同。

## 独立配置与 100v

[evaluation 配置](../configs/baseline/eval.cfg) 和 [比赛配置](../configs/baseline/match.cfg) 只沿 `run.cfg` 的 extends 继承链读取自身文件与叶子目录的 `.local`。训练文件、训练环境变量和另一个评估文件不会覆盖它们，修改评估条件不改变训练配置身份。可使用 `EVAL_VISITS=100`、`MATCH_VISITS=100` 等前缀环境覆盖；开局字段使用 `MATCH_OPENING_` 前缀。字段校验集中在 [eval_config.py](../python/etazero/eval_config.py)。

默认搜索预算为 100v：根的一次初始评估和后续完成的边访问合计 100。默认不复用树，因此每手新增 99 次模拟；单局面输出 `root_visits=100`、`simulations=99` 和总和为 99 的动作访问数组。若明确启用树复用，已有访问计入上限，不再额外新增 100 次模拟。此预算参考 KataGo `maxVisits` 包含复用访问的语义，训练 full cap 为 `search.full_search_visits`，同样计入复用量；也不同于 MuZero_V2 将根评估外的边模拟数命名为 visits 的计数方式。网络缓存命中、终局节点和开局评估使 NN 请求数不等于 visits。

评估和比赛均无训练根噪声，默认零落子温度、单搜索线程；并列最大行为权重按来源选择首个已分配子边。FPU、子树价值加权、policy target pruning、LCB、根多对称、根／全树 policy 温度与落子温度半衰期由独立 profile 显式配置；评估及比赛落子使用剪枝和 LCB 后的权重，原始 visits 单独输出。WDL 搜索 Q 为 W−L。比赛共享组批 evaluator、多局线程的结构参考 KataGo match，使用 EtaZero LibTorch 后端，不宣称复制 KataGo 的全部比赛功能或数值行为。

## 比赛与续测

```bash
# 在 EtaZero_V0/ 下，run-dir 用于寻找当前模型和保存默认输出。
bash scripts/run.sh evaluate --config-dir configs/baseline --run-dir data/my_run --size 15 --rule renju --moves 112,113
bash scripts/run.sh match --config-dir configs/baseline --model /path/to/a/model.pt --model-b /path/to/b/model.pt --games 40 --output data/my_match
```

每对模型局数为四的正倍数。双方各生成一半平衡开局，同一个开局交换模型执黑／执白下两局，棋盘颜色和 Renju 规则不变。平衡开局及 policy init 的默认参数取 MuZero_V2 的比赛配置，独立于训练开局参数；只接受成功且非终局的开局。无认输、无提前截断，按真实棋规结束。每局保存完整落子、胜者、时间与逐手根访问数。

开局和单局完成后分别原子落盘，重启只安排缺失的一侧或新开局。中断的半局重跑，同开局已完成的一侧保持不变。相同输出目录可以用 `--games` 增加成对开局，不能减少目标；模型、搜索条件、种子、二进制或配对身份变化时使用新输出目录。训练采用整轮提交，比赛采用逐开局／逐局提交，两者边界独立。

## 历史模型与 Elo

```bash
bash scripts/run.sh arena --data data/my_experiment --output data/my_elo --dry-run
bash scripts/run.sh arena --data data/my_experiment --output data/my_elo
bash scripts/run.sh arena --data data/my_experiment --output data/my_elo --fit-only
```

`--data` 可以是包含各臂的父目录，也可以是单个运行目录。只发现 state 已提交的模型；包含首个可用、按 `--stride` 间隔选择的模型以及最后模型。组内近邻与跨组邻近训练时间配对组成连通比较图。默认参数见 [arena.py](../python/etazero/arena.py)，`--dry-run` 不创建输出或启动比赛。固定的赛程重启续测；训练继续产生新模型后重新发现的赛程与旧赛程不同，应使用新的评估输出目录。

横轴为模型所属轮次的 `elapsed_seconds`，即已提交完整轮次累计训练墙钟。暂停和作废轮次不进入横轴；恢复后的重跑成功轮只计一次。图上连线用于展示实测模型点，不代表测量了中间时刻棋力，不向训练停止后外推。原始活动时间仍在训练事件日志中，不能与这个横轴混用。

评分与区间沿用 MuZero_V2 的 [elo.py](../python/etazero/elo.py)：联合 Bradley–Terry 得分模型，和棋记半分，同时拟合共同先手优势；anchor 只规定相对 Elo 零点，不代表外部等级分。弱高斯正则防止全胜／全负导致无穷评分，强度随输出记录。95% 区间按配对分层、以同开局两局为单位 bootstrap；全胜配对会标记先验敏感性。区间条件于已选择的模型、开局与协议，不表示独立训练种子的不确定性。所有开局的两侧和整个赛程完整后才拟合。

输出 `elo.json`、`elo.csv`、`elo.png`、`elo.svg`，并保存模型／配置／二进制身份、赛程、逐开局和逐局原始记录。`--fit-only` 从已有赛果重建评分，不启动 GPU 比赛。
