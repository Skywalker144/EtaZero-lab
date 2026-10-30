# 原生日志与外层绘图

训练提交后默认更新 `training.png`、`loss.png` 与 `logs/performance.png`。外层配置 `auto_plot=false` 可关闭自动绘图；绘图发生在计时结算和模型提交之后，失败会警告，不撤销训练进度。手动重建不需要 GPU 或加载网络：

```bash
bash scripts/run.sh plot --run-dir /absolute/path/to/run
bash scripts/run.sh plot --run-dir /absolute/path/to/run --output-dir /absolute/path/to/figures
```

**训练图的算法与性能指标严格从原生日志读取。** 原生 learner 的文本日志、metrics JSON、selfplay 和 gatekeeper 日志保留原样。外层只解析、汇总和绘制，不读取 SGF、checkpoint、NPZ 或调度结果文件为图补充统计，不为绘图向引擎新增计数或汇总日志，不修改来源日志的打印频率。时间横轴允许使用 EtaZero 调度计时，图上明确标为 `EtaZero controller ... time`；不能把它称为 KataGo 原生计时。

实现入口为 [plotting.py](../python/etazero/plotting.py) 与 [observations.py](../python/etazero/observations.py)。绘图仅原子替换派生图片，不改写原始日志。旧运行的外层 `metrics.json` 不作为图表数据源。

## 数据来源

| 展示内容 | 数据来源 |
|---|---|
| 训练 loss、梯度和 MuZero 展开分量 | 来源 learner 的 `train/metrics_train.json`，每行一条原生 JSON 记录 |
| 验证 loss | 来源 learner 的 `train/metrics_val.json` |
| 自对弈游戏数、实际搜索手数、推理与缓存计数 | 原生 selfplay 的 `selfplay/轮次/log*.log` 中的 `Final ...` 记录 |
| 写出根行数、原生 selfplay 运行时间 | 同一原生日志中的 `Final data rows`、`Total selfplay runtime (seconds)` |
| gatekeeper 候选得分 | `rounds/轮次/gatekeeper/matches/尝试/log*.log` 中的 `Candidate won/lost match, score ...` |
| 独立 Elo | 原生 match 的逐局比赛记录，调用来源 Elo 拟合，见 [比赛与 Elo](evaluation.md) |
| 完成轮次边界、累计训练时间 | EtaZero `state.json` 与 `rounds/轮次/commit.json`，仅用于展示边界和明确标注的外层时间轴 |

来源 learner 通常每 100 个 batch 输出训练 metrics；短 epoch 没有触发输出时，图上没有该轮 loss 点。不能从 checkpoint 补一个轮末值，也不能把之前的日志值搬到当前轮末。验证日志同样只使用实际输出的记录，没有验证数据时不补值。原始文本日志和 JSON metrics 均继续由来源程序管理。

## 统计与横轴

loss 和梯度点直接使用原生日志中的数值，保留来源 `metrics_logging.py` 的移动权重、日志窗口、归约和 Lookahead 观测语义，不二次平滑。loss 的 `_sum` 后缀如何变为日志字段、`_batch` 的窗口如何衰减，均由原生 logger 决定。展示层不重新计算 sums/weights，也不通过读取模型结构改写记录。

训练点横轴使用每条记录的 `nsamp`，验证点使用 `nsamp_train`。不会把稀疏记录对齐到轮末样本数。只展示已完成轮次覆盖的样本范围；原始日志可能包含失败后重做的更新，同一计数可以有重复观测。计数重复或回退时断开曲线，保留记录，不声称它们都是 checkpoint 的统计。日志不足、非有限值或损坏记录会警告或留空，不补零。某轮没有新记录时不沿用上一轮的值。

来源 loss 分量保留各自的系数位置，分量之和不保证等于总 loss。梯度范数是来源裁剪前的 batch-sum loss 梯度，不能解释为平均 loss 的梯度。MuZero 的普通分量是根统计，`mu_recurrent_*` 和 `mu_step_k_loss` 使用原生适配后 learner 实际输出的定义；没有这些字段时不生成替代曲线。验证值来自来源实际执行的验证过程，不代表 SWA 模型的棋力。

自对弈汇总读取原生日志中队列排空后的 `Final games finished`、`Final moves played` 等记录。一个指标没有覆盖全部尝试时，该指标留空，不能把其余尝试的子集冒充整轮。`Final moves played` 只计实际由搜索走出的手数，排除预置开局和导入起点；图中均值以此除以完成游戏数，不称为包含开局的完整局长。原有日志没有完整的 W／L／D／NR／截断汇总，胜负分布图留空并注明缺失，不读取 SGF 补齐。

有效根行数读取 native writer 的 `Final data rows`，含实际重复采样与 side 根；MuZero 的展开位置不重复计为根样本。日志缺失时，后续累计根行数也未知，不能从数据文件补齐该图。gatekeeper 只显示唯一一条最终比赛总结的原生得分；日志缺失或有多个不明确的结果时留空，不读取控制器续跑所需的 `result.json` 补值。

普通网络的局面缓存命中率为原生日志 `Final NN cache hits / (Final NN cache hits + Final NN rows)`。MuZero 的 `Final NN rows` 同时包含 initial 和 recurrent，原有日志没有独立 initial 计数，因此其局面缓存命中率留空。推理平均 batch 为原生网络行数除以 batch 数，MuZero 包含两类请求；来源随机 evaluator 不增加 NN 行数和 batch 计数，不作为网络推理测量。外层只汇总这些原生计数。

顶部时间刻度使用实际轮末的 EtaZero 累计耗时，不假设恒定吞吐，不把样本中间点插值成精确训练时间。计时失效的轮次不产生时间刻度，定义见 [调度与恢复](controller.md#并发与计时)。独立 arena 的时间横轴使用同一来源并明确标注。

## 性能图

`logs/performance.png` 展示原生 selfplay 运行时间、写出行数／原生运行时间、平均推理 batch、局面缓存命中率、learner 的 `time_since_last_print` 与 gatekeeper 候选得分。原生打印间隔不是纯更新耗时，不把它换算成未经证实的训练吞吐；selfplay 吞吐也不是纯 GPU kernel 性能。外层阶段耗时和编译扣除量仍服务调度与预算，但不伪装成这张原生日志性能图的指标。

gatekeeper 的对手随 accepted 模型变化，因此其候选得分不是固定对手棋力或 Elo。独立 Elo 使用单独的比赛记录和图，不混入训练指标。

## 逐局 surprise 诊断

原生 `selfplaysurprisedump` 使用固定模型运行 GameRunner，把每个记录手的搜索统计写入 CSV。它支持六种棋类、普通／MuZero 网络和 PUCT／Gumbel；配置使用原生 selfplay 字段，MuZero 的 `-model` 指向三组件 bundle，并设置 `useMuZero=true`。这是独立诊断，不会向训练目录追加数据：

```bash
bash scripts/run.sh native selfplaysurprisedump \
  -config /absolute/path/to/selfplay.cfg -model /absolute/path/to/model \
  -num-games 10 -output /absolute/path/to/surprise.csv
```

默认关闭重分析，CSV 保留实际出手时的统计。加 `-reanalyze-all` 会重分析所有 cheap-search 位置，保留 `origPolicySurprise`、`origValueSurprise` 和 `origNumVisits`，其余对应列使用重分析后的结果。这些开关行为、cheap/full 搜索、surprise 公式与权重仍由原生流程负责。默认 value surprise 使用向前平滑的搜索／终局结果，因此一手的重分析后数值也受后续重分析影响；设置 `useSearchValueSurprise=true` 才只比较该手搜索值与原始网络预测。

`game` 标识棋类；`outcomeWhite*` 与 `mctsWhite*` 按白方视角分别输出 W/D/L/NR，其中象棋白方对应黑棋一方。非围棋 NR 为 0。`hitTurnLimit=1` 的非围棋对局中，`outcomeWhite*` 是末状态网络 bootstrap，不是已经裁定的和棋；判断实际结果须同时检查截断标志。`turnIdx` 包含导入起点的原始手数，`turnAfterStart` 从记录段起点计数。CSV 逐局加锁写出并刷新；收到停止信号时保留已完成的整局记录，未完成对局不写入。
