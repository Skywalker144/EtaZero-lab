# V4 baseline_az / az_gomoku 与 KataGo、KataGomo 全参数对照

配置快照：2026-10-31。定位：**baseline_az 尽量对齐 KataGo main18b；az_gomoku 面向 Gomoku，参考 KataGomo 形成适合本项目的配方。** “调好的 Gomoku 参数”是目标，当前尚无这套 V4 配方的调参或棋力验证结论。配置入口采用四算法 baseline 与四棋类命名；算法继承关系见 [配置与启动](configuration.md)。

## 1. 来源与覆盖范围

| 对象 | 来源与边界 |
| --- | --- |
| V4 baseline_az | [/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az) |
| V4 az_gomoku | [/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku) |
| KataGo main18b | [/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg) |
| KataGomo 15 路 | [python/scripts/selfplay.cfg](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg) |

KataGo 使用本机 `SkyZero-lab/KataGo` 的 `selfplay8mainb18.cfg`，提交 `d91ea855110dae533f0aada947b2b7d78cc8a4e1`。KataGomo 本机无完整 checkout，使用 V4 `reference_sources.json` 记录的固定提交 `df152116e3787c75c6a3de099d261ca092b7dfc1`，从上游读取 **python/scripts/selfplay.cfg（15 路）**；不是另一份 19 路配置。

KataGomo 的网络、训练和 shuffle 列另参考同提交的 [python/scripts/run_selfplay.sh](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/run_selfplay.sh), [python/train.sh](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.sh), [python/shuffle.sh](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.sh), [python/train.py](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py), [python/modelconfigs.py](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/modelconfigs.py), [python/shuffle.py](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py)。外层脚本进入部署的 `train` 目录执行包装脚本；这里对照仓库配套文件，未验证实际部署目录或预编译 engine 是否与该提交一致。

“所有参数”在本文指：四个指定配置对象的**全部有效声明键**（含 V4 include 展开和空赋值），两份 V4 配置解析后的全部 RunConfig 字段、所选网络预设的全部字段，以及 V4/KataGomo train、shuffle 的全部 argparse 声明。相同值也保留。注释掉的示例不算赋值；未引入其他 profile、比赛或分析配置。原生程序还有与这些配置无关的 GTP/分析等参数，本文不将整个程序的全部接口冒充这四份训练配置。

读表：`—` 为该对象无声明；`未声明` 不等于 false/0；`默认` 只用于已核对源码的值；`空赋值` 在 V4 train/shuffle 中表示不传 CLI。第 3 节展示语义对齐，第 4 节保留原名、原值及最终赋值行，避免把旧字段误算为新字段。KataGo main18b **只提供 selfplay 参数**，不提供 learner、网络型号或门控设置，不能由文件名推断。

## 2. 对齐方向与需要先澄清的差异

这个分工合理：baseline_az 可作为接近 KataGo 的公共起点，az_gomoku 在此之上覆盖五子棋规则、开局、搜索和数据配方。需要按机制语义对齐：Go 的贴目、让子、计分辅助目标不能直接用于 Gomoku；线程数、GPU 编号和缓存容量则应匹配实际硬件。

| 方向 | 当前主要差异 | 后续讨论重点 |
| --- | --- | --- |
| baseline_az → KataGo main18b | maxVisits 600→2000；cheapSearchVisits / reducedVisitsMin 100→350；policyInitAreaProp 0.04→0.08；19 路权重 35→75。 | 这些是实质训练配方差异，不能把当前 baseline_az 称为已经对齐。访问数调整同时影响算力预算和样本生产速度。 |
| baseline_az → KataGo main18b | V4 已移除 score/ownership/seki 等监督及部分动态贴目/让子补偿机制；draw 随机效用仍固定为 0，NR 随机效用已恢复并对齐 main18b。 | 这些属于功能与训练目标差异，不是修改 cfg 数字即可恢复。 |
| az_gomoku → Gomoku | 当前是纯 Freestyle；KataGomo 是 Renju 为主、附带其他规则的混合任务，并使用 VCF/禁手输入。 | 先按所选规则解释参数效果，不能把混合规则模型的配置直接标为纯 Freestyle 最优配方。 |
| az_gomoku → KataGomo 搜索 | visits 100 对 150；cheapSearchProb 均为 0，reducedVisitsMin 均为 20；SVB 0.45 对默认 0；root policy 温度、side 分支比例仍不同。 | 当前已按用户选择改为 100 visits、关闭廉价搜索；不能据此宣称已验证优于之前配方。 |
| 已确认的工程配置 | 两份 V4 均 cudaUseNHWC=true、useNonBuggyLcb=true、maxDataQueueSize=2000、maxRowsPerTrainFile=20000。 | 写盘参数已对齐 main18b；NHWC 的实际吞吐仍取决于设备与网络，本文未测速。 |


### KataGomo 配置与源码并非完全匹配

这份上游文件可以作为参数来源，但**不能认定它与同提交源码组成了已验证可运行的完整训练配方**。下列问题必须与原始数字同时阅读：

| 项目 | 文件声明 | 同提交消费路径 |
| --- | --- | --- |
| policy 初始化长度 | policyInitProp=10 | PlaySettings 读取 policyInitAvgMoveNum，缺省 12；这里不能把 10 当作实际均值。[playsettings.cpp](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/cpp/program/playsettings.cpp#L65) |
| NR 随机化 | noResultStdev=0.4 | GameInitializer 读取 noResultRandRadius，缺省 0；这里不是已生效的高斯随机化。[play.cpp](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/cpp/program/play.cpp#L238) |
| 认输/提前和棋 | 未声明 allowResignation、resignThreshold、resignConsecTurns、allowEarlyDraw、earlyDrawThreshold、earlyDrawConsecTurns、earlyDrawProbSelfplay | loadForSelfplay 无缺省地读取这些键；命令直接调用它，配置单独传入当前源码会缺少必填项。[playsettings.cpp](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/cpp/program/playsettings.cpp#L96) [selfplay.cpp](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/cpp/command/selfplay.cpp#L116) |
| LCB 修正版 | 未声明 useNonBuggyLcb | selfplay 的 SETUP_FOR_OTHER 缺省为 false；V4 与 main18b 显式 true，不建议为了复制旧行为退回 false。[setup.cpp](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/cpp/program/setup.cpp#L575) |
| 后端与设备 | numNNServerThreadsPerModel=2，但还保留 Thread2 及 TRT Thread3–7 的设备键 | 必须按所编译后端及实际服务线程数解释；不能据此认定同时用了八张卡。 |

以上“默认”是修齐启动所需项后该读取路径的缺省逻辑，不表示缺项配置已经成功运行。其他疑似历史遗留的 Go 字段仍完整保留在第 4 节，未逐一证明它们在 KataGomo 中生效。

### 公共参数与并行度

baseline_az 的 bootstrap_games=800、keep-target-rows=20000000、shuffle num-processes=12；az_gomoku 删除这三项同值覆盖并继承。baseline_az min-rows=250000，与 KataGo shuffle.py 的缺省值一致；az_gomoku 保留自己的 150000。公共 cpuctExploration=1.05、cpuctExplorationLog=0.28、rootPolicyTemperatureEarly=1.5，az_gomoku 删除对应覆盖。象棋继承对应算法 baseline 的完整搜索、采样与训练配方；20 个顶层配置的 PUCT 配置均已核对。Hex 和象棋的四算法配置均使用 replay_ratio=2；围棋与五子棋仍为 4。Gumbel 专用温度覆盖保持原值。az_gomoku maxVisits=100、cheapSearchProb=0、reducedVisitsMin=20、chosenMoveTemperatureHalflife=12；cheapSearchVisits=75 暂留，但当前廉价搜索关闭。
numGameThreads 接受 1–16384 的整数，不要求 2 的幂：main18b=800，KataGomo=600。它控制并行对局数，需与 nnMaxBatchSize、numNNServerThreadsPerModel、网络大小和硬件联合测量。当前 baseline_az=128、az_gomoku=512 未改。shuffle workers 采用 az_gomoku 的 12，当前未找到可确认其在本 V4/本机最优的测速证据；后续应在同一数据快照下比较耗时、内存与 I/O，自对弈比较真实游戏/有效样本吞吐，不能仅凭 GPU 利用率挑选。

### Hex／象棋的探索与训练配置检查

Hex 的 half-life 为 15，象棋为 40；普通 AZ/MuZero 的完整自对弈搜索启用 Dirichlet 噪声。[AlphaZero 论文](https://arxiv.org/pdf/1712.01815)的 0.3 是国际象棋的逐动作 α，不是中国象棋的验证结果，也不是 V4 的总浓度；当前保留 KataGo 的总浓度 10.83 和混合权重 0.25。Gumbel 保留自身噪声、禁用 Dirichlet，不启用 cheap search、reduce visits、LCB 或 forced playouts。

| 检查项 | 发现的问题 | 当前设置 |
| --- | --- | --- |
| 象棋根探索 | AZ/MuZero 的 rootNoiseEnabled=false，且保留无噪声时的 root FPU 覆盖 | 继承 baseline：rootNoiseEnabled=true、weight=0.25、rootFpuReductionMax=0 |
| 象棋对局与搜索 | maxMovesPerGame=24；AZ maxVisits=16 | 继承 1600 手上限；AZ 600、MuZero 250、Gumbel 两种均 33 visits |
| 象棋样本多样性 | policy init、early/game fork、side positions 均关闭 | 继承对应 baseline 的 policy 初始化与采样机制 |
| 象棋运行与训练 | bootstrap=4、训练 batch=4、原始文件128行，以及低并行/小缓存覆盖 | 继承 bootstrap=800、batch=128、原始文件20000行及算法推理/并行配置；训练精度继承 baseline |
| 非训练样本的廉价搜索 | 噪声关闭不能一概视为配置错误 | AZ 的零训练权重 cheap search 按 KataGo 临时去掉根噪声；完整训练搜索启用噪声 |
| 尚未对齐的 Hex 配方 | 当前画布19、7–19路分布；平衡开局缺省关闭，与 V3.4 的15路画布、9–15路分布及平衡开局不同 | 这些是训练任务/采样方案的选择，本次未改变；需与小规模检查残留区分 |


### Gatekeeper、SGD 与 LCB

V4 baseline_az 与 az_gomoku 的 gatekeeper_config 均未设置，门控关闭；gatekeeper_required_win_prop=0.5 只是启用门控后的候选得分率阈值，不会自行开启门控。两者当前是 **SGD（momentum=0.9，非 Nesterov）+ Lookahead（6,0.5）+ SWA（80000,8）**，learner FP32、编译开启；CUDA FP16 是推理选项，与训练 AMP 分开。KataGomo 配套 learner 同样构造 SGD momentum=0.9，但脚本启用 FP16，SWA scale=1、缺省周期为 samples-per-epoch/2=1000000，不能说完整优化配方相同。

V4 selfplay 实际出手前临时关闭 LCB、随后恢复以生成训练策略目标，这个调用顺序与 KataGo 对应路径一致，KataGomo 同样有这一处理。无需为了“对齐 KataGo”把 LCB 强行用于 selfplay 实际出手。依据：[play.cpp](/home/sky/RL/EtaZero-lab/EtaZero_V4/cpp/program/play.cpp:1219), [play.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/play.cpp:1245), [play.cpp](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/cpp/program/play.cpp#L1079)。

### 和棋效用与写盘参数

KataGo 的白方和棋等价胜局得分为 d，黑方为 1−d；例如 d=0.8 时白方 0.8、黑方 0.2。相应零和效用为 +(2d−1) 与 −(2d−1)，不是只给白方偏好而遗漏黑方。当 d 在 [0,1] 均匀抽样时，整体对白黑对称，但单局双方对和棋偏好不同。它不直接随机改判，却可以通过改变决策间接影响和棋频率。V4 普通和棋仍固定中性，不实现 draw 随机化。NR 则已恢复 KataGo 的逐局截断高斯机制：白方效用系数均值 0，标准差参数 0.166666666（截断后实际标准差略小），截断在 ±3σ，并拒绝超出 [-1,1] 的样本。az_gomoku 继承该配置，但其 NR 概率被屏蔽，普通和棋效用仍为 0。

maxDataQueueSize 是等待 writer 的**完整对局数上限**，不预先分配满额内存。32 与 2000 在 writer 及时消费时通常没有区别；writer 短暂变慢时，2000 允许积压更多完整对局、减少生产者等待，但峰值内存和未落盘积压可能增加。队列满会阻塞生产者，不丢局。现已统一 2000；若出现持续积压，应看写盘吞吐和内存，继续放大队列不能解决持续瓶颈。maxRowsPerTrainFile 已统一 20000，是原始 selfplay 文件的行数上限；首文件可受 firstFileRandMinProp 提前结束，且与 shuffle 输出文件大小不是同一参数。

## 3. 关键行为与同义参数对照

| 项目 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| 棋类 / 规则 | Go，组合规则 | 纯 Freestyle | Go，组合规则 | basicRules 列表：Renju×8、Freestyle×1、Standard×1；另有 VCN/firstPassWin 规则 |
| 网络 | b5c192nbt-fson-mish | b5c192nbt-fson-mish | 未由文件指定 | b28c512nbt-mish（运行脚本） |
| learner batch | 128 | 128 | 未由文件指定 | 每 GPU 128，脚本 multi-gpus=0,1 |
| learner 精度 | FP32 | FP32 | 未由文件指定 | FP16 AMP（脚本） |
| SWA 周期 / scale | 80000 / 8 | 80000 / 8 | 未由文件指定 | 1000000（源码推导） / 1 |
| 每 epoch 根样本目标 | 256000（128×2000） | 256000 | 未由文件指定 | 2000000 |
| 调度 | bootstrap 800；replay_ratio=4，后续自适应局数/训练额度 | bootstrap 800；replay_ratio=4 | 未由文件指定 | 循环：最多 50000 局→shuffle→1 epoch→export；stopIfNewNet=true 可提前结束采样 |
| shuffle keep-target-rows | 20000000 | 20000000 | 未由文件指定 | train 2100000 / val 51200 |
| shuffle min-rows | 250000 | 150000 | 未由文件指定 | 150000 |
| shuffle expand / taper exponent / scale | 0.4 / 0.65 / 50000 | 0.3 / 0.8 / 150000 | 未由文件指定 | 0.3 / 0.8 / 150000（默认随 min-rows） |
| shuffle workers / 输出文件行数 | 12 / 70000 | 12 / 70000 | 未由文件指定 | 32 / 50000 |
| 数据划分 | 当前外层不生成独立 val 快照 | 同 baseline_az | 未由文件指定 | selfplay validationProp=0；shuffle 后按文件路径哈希 97%/3% 划 train/val |
| 门控 | 关闭 | 关闭 | 未由文件指定 | 所列运行循环没有 gatekeeper 步骤 |

| 参数 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| dataBoardLen | 19 | 15 | 19 | 15 |
| bSizes | 7,9,11,13,15,17,19,  8,10,12,14,16,18 | 15,14,13,12,11 | 7,9,11,13,15,17,19, 8,10,12,14,16,18 | 15,14,13,12,11 |
| bSizeRelProbs | 1,4,3,10,7,9,35, 1,2,4,6,8,10 | 100,10,5,3,1 | 1,4,3,10,7,9,75, 1,2,4,6,8,10 | 100,10,5,3,1 |
| allowRectangleProb | 0.10 | 0 | 0.10 | 0.00 |
| komiMean | 7.5 | 0 | 未声明 | 0.0 |
| handicapProb | 0.10 | 0 | 0.10 | 未声明 |
| drawRandRadius | 0 | 0 | 0.5 | 0.0（该初始化路径未读取此键） |
| noResultStdev | 0.166666666 | 0.166666666 | 0.166666666 | 0.4（旧键；实际读取 RandRadius，默认 0） |
| maxVisits | 600 | 100 | 2000 | 150 |
| cheapSearchProb | 0.75 | 0 | 0.75 | 0.0 |
| cheapSearchVisits | 100 | 75 | 350 | 100（cheapSearchProb=0，未启用） |
| cheapSearchTargetWeight | 0.0 | 0.0 | 0.0 | 0.0 |
| reduceVisits | true | true | true | true |
| reduceVisitsThreshold | 0.9 | 0.9 | 0.9 | 0.9 |
| reduceVisitsThresholdLookback | 3 | 3 | 3 | 3 |
| reducedVisitsMin | 100 | 20 | 350 | 20 |
| reducedVisitsWeight | 0.1 | 0.1 | 0.1 | 0.1 |
| cpuctExploration | 1.05 | 1.05 | 1.05 | 1.0 |
| cpuctExplorationLog | 0.28 | 0.28 | 0.28 | 0.45 |
| cpuctExplorationBase | 500 | 500 | 500.0（默认） | 500 |
| cpuctUtilityStdevPrior | 0.40 | 0.40 | 0.40（默认） | 0.40（默认） |
| cpuctUtilityStdevPriorWeight | 2 | 2 | 2.0（默认） | 2.0（默认） |
| cpuctUtilityStdevScale | 0 | 0 | 0（默认） | 0（缺省逻辑；见第 2 节） |
| numSearchThreads | 1 | 1 | 1 | 1 |
| numVirtualLossesPerThread | 1 | 1 | 1 | 1.0（默认） |
| fpuReductionMax | 0.2 | 0.2 | 0.2 | 0.2（默认） |
| rootFpuReductionMax | 0.0 | 0.0 | 0.0 | 未声明 |
| fpuParentWeightByVisitedPolicy | true | true | true | true（缺省逻辑；见第 2 节） |
| fpuParentWeightByVisitedPolicyPow | 2.0 | 2.0 | 2.0 | 2（缺省逻辑；见第 2 节） |
| valueWeightExponent | 0.5 | 0.5 | 0.5 | 0.25（缺省逻辑；见第 2 节） |
| subtreeValueBiasFactor | 0.30 | 0.45 | 0.30 | 0（缺省逻辑；见第 2 节） |
| subtreeValueBiasWeightExponent | 0.8 | 0.85 | 0.8 | 0.85（缺省逻辑；见第 2 节） |
| subtreeValueBiasFreeProp | 0.8 | 0.8 | 0.8（默认） | 0.8（默认） |
| rootNoiseEnabled | true | true | true | true |
| rootDirichletNoiseWeight | 0.25 | 0.25 | 0.25 | 0.25 |
| rootDirichletNoiseTotalConcentration | 10.83 | 10.83 | 10.83 | 10.83 |
| rootDesiredPerChildVisitsCoeff | 2 | 2 | 2 | 2 |
| chosenMoveTemperatureEarly | 0.75 | 0.75 | 0.75 | 0.75 |
| chosenMoveTemperature | 0.15 | 0.15 | 0.15 | 0.15 |
| chosenMoveTemperatureHalflife | 19 | 12 | 19 | 12 |
| rootPolicyTemperatureEarly | 1.5 | 1.5 | 1.5 | 1.8 |
| rootPolicyTemperature | 1.1 | 1.1 | 1.1 | 1.2 |
| nnPolicyTemperature | 1.0（默认） | 1.0（默认） | 1.0（默认） | 1.1 |
| chosenMoveSubtract | 0 | 0 | 0 | 0 |
| chosenMovePrune | 1 | 1 | 1 | 1 |
| useLcbForSelection | true | true | true | true |
| lcbStdevs | 5.0 | 5.0 | 5.0 | 5.0 |
| minVisitPropForLCB | 0.15 | 0.15 | 0.15 | 0.15 |
| useNonBuggyLcb | true | true | true | false（缺省逻辑；见第 2 节） |
| rootNumSymmetriesToSample | 4 | 4 | 4 | 4 |
| useGraphSearch | true | true | true | true |
| useUncertainty | false | false | false（默认） | false（缺省逻辑；见第 2 节） |
| useNoisePruning | false | false | false（默认） | false（缺省逻辑；见第 2 节） |
| policyOptimism | 0 | 0 | 0（默认） | 未声明 |
| rootPolicyOptimism | 0 | 0 | 0（默认） | 未声明 |
| initGamesWithPolicy | true | true | true | true |
| policyInitAreaProp | 0.04 | 0.04（五子棋用 AvgMoveNum） | 0.08 | 未声明 |
| policyInitAvgMoveNum | 12 | 6 | 未声明 | 12（缺省逻辑；见第 2 节） |
| policyInitAreaTemperature | 1.0 | 1.6 | 未声明 | 1.6 |
| balanceOpeningProb | 未声明 | 0.99 | 未声明 | 0.99（源码固定）（缺省逻辑；见第 2 节） |
| balanceOpeningDropPow | 未声明 | 4 | 未声明 | 未声明 |
| sidePositionProb | 0.020 | 0.020 | 0.020（旧名 forkSidePositionProb） | 0.040（旧名 forkSidePositionProb） |
| earlyForkGameProb | 0.04 | 0.04 | 0.04 | 0.0 |
| earlyForkGameExpectedMoveProp | 0.025 | 0.025 | 0.025 | 0.0 |
| forkGameProb | 0.01 | 0.01 | 0.01 | 0.0 |
| forkGameMinChoices | 3 | 3 | 3 | 3 |
| earlyForkGameMaxChoices | 12 | 12 | 12 | 12 |
| forkGameMaxChoices | 36 | 36 | 36 | 36 |
| normalAsymmetricPlayoutProb | 0.01 | 0.01 | 0.01 | 0.1 |
| maxAsymmetricRatio | 8.0 | 8.0 | 8.0 | 4.0 |
| policySurpriseDataWeight | 0.5 | 0.5 | 0.5 | 0.5 |
| valueSurpriseDataWeight | 0.1 | 0.1 | 0.1 | 0.1 |
| numGameThreads | 128 | 512 | 800 | 600 |
| nnMaxBatchSize | 128 | 64 | 192 | 256 |
| numNNServerThreadsPerModel | 1 | 2 | 8 | 2 |
| nnCacheSizePowerOfTwo | 21 | 23 | 24 | 24 |
| nnMutexPoolSizePowerOfTwo | 15 | 15 | 18 | 20 |
| nnRandomize | true | true | true | true |
| cudaUseFP16 | auto | true | true | 未声明 |
| cudaUseNHWC | true | true | true | 未声明 |
| maxDataQueueSize | 2000 | 2000 | 2000 | 100000 |
| maxRowsPerTrainFile | 20000 | 20000 | 20000 | 5000 |
| firstFileRandMinProp | 0.15 | 0.15 | 0.15 | 0.15 |
| switchNetsMidGame | false | false | true | false |
| maxMovesPerGame | 1600 | 1600 | 1600 | 30000 |

搜索缺省值按各自 selfplay 的 SETUP_FOR_OTHER 路径核对；来源：[setup.cpp](/home/sky/RL/EtaZero-lab/EtaZero_V4/cpp/program/setup.cpp), [setup.cpp](/home/sky/RL/SkyZero-lab/KataGo/cpp/program/setup.cpp), [setup.cpp](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/cpp/program/setup.cpp)。本表未填满所有程序隐式默认；“未声明”保留未知/不适用边界。完整声明（包括 Go 计分、贴目、硬件编号和 KataGomo 特有规则/输入）见下表。

## 4. 全量配置声明表（含继承与来源）

| 覆盖计数 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| include 展开后唯一声明字段数 | 188 | 185 | 115 | 99 |


### net

| 原始字段 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| net.network.architecture | b5c192nbt-fson-mish [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/net.cfg:3) | b5c192nbt-fson-mish [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/net.cfg:3) | — | — |


### run

| 原始字段 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| run.run.auto_plot | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:11) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:11) | — | — |
| run.run.bootstrap_game_multiplier | 1.05 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:6) | 1.05 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:6) | — | — |
| run.run.bootstrap_games | 800 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:3) | 800 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:3) | — | — |
| run.run.gatekeeper_required_win_prop | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:12) | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:12) | — | — |
| run.run.publish_weights | swa [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:7) | swa [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:7) | — | — |
| run.run.replay_ratio | 4 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:5) | 4 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:5) | — | — |
| run.run.train_steps | 2000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:4) | 2000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/run.cfg:4) | — | — |


### selfplay

| 原始字段 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| selfplay.VCNRules | — | — | — | NOVC,NOVC,NOVC,NOVC,NOVC,NOVC,NOVC,NOVC,VC2B,VC2W [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L5) |
| selfplay.agent.algorithm | alphazero [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:3) | alphazero [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:3) | — | — |
| selfplay.agent.nonroot_search_algo | puct [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:5) | puct [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:5) | — | — |
| selfplay.agent.root_search_algo | puct [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:4) | puct [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:4) | — | — |
| selfplay.allowRectangleProb | 0.10 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:12) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/games/gomoku/env.cfg:7) | 0.10 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:78) | 0.00 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L95) |
| selfplay.bSizeRelProbs | 1,4,3,10,7,9,35, 1,2,4,6,8,10 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:11) | 100,10,5,3,1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/games/gomoku/env.cfg:6) | 1,4,3,10,7,9,75, 1,2,4,6,8,10 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:77) | 100,10,5,3,1 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L38) |
| selfplay.bSizes | 7,9,11,13,15,17,19, 8,10,12,14,16,18 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:10) | 15,14,13,12,11 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/games/gomoku/env.cfg:5) | 7,9,11,13,15,17,19, 8,10,12,14,16,18 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:76) | 15,14,13,12,11 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L37) |
| selfplay.balanceOpeningDropPow | — | 4 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:21) | — | — |
| selfplay.balanceOpeningProb | — | 0.99 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:20) | — | — |
| selfplay.basicRules | — | — | — | RENJU,RENJU,RENJU,RENJU,RENJU,RENJU,RENJU,RENJU,FREESTYLE,STANDARD [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L3) |
| selfplay.canvas | 19 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:4) | 15 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/games/gomoku/env.cfg:4) | — | — |
| selfplay.cheapSearchProb | 0.75 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:9) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:6) | 0.75 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:40) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L69) |
| selfplay.cheapSearchTargetWeight | 0.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:11) | 0.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:11) | 0.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:42) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L70) |
| selfplay.cheapSearchVisits | 100 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:10) | 75 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:7) | 350 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:41) | 100 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L32) |
| selfplay.chosenMovePrune | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:29) | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:29) | 1 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:135) | 1 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L125) |
| selfplay.chosenMoveSubtract | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:28) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:28) | 0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:134) | 0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L124) |
| selfplay.chosenMoveTemperature | 0.15 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:23) | 0.15 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:23) | 0.15 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:133) | 0.15 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L123) |
| selfplay.chosenMoveTemperatureEarly | 0.75 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:21) | 0.75 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:21) | 0.75 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:131) | 0.75 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L121) |
| selfplay.chosenMoveTemperatureHalflife | 19 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:22) | 12 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:10) | 19 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:132) | 12 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L122) |
| selfplay.clearBotBeforeSearch | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:14) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:14) | — | — |
| selfplay.compensateAfterPolicyInitProb | — | — | 0.5 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:37) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L59) |
| selfplay.cpuctExploration | 1.05 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:54) | 1.05 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:54) | 1.05 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:164) | 1.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L150) |
| selfplay.cpuctExplorationBase | 500 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:56) | 500 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:56) | — | 500 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L152) |
| selfplay.cpuctExplorationLog | 0.28 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:55) | 0.28 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:55) | 0.28 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:165) | 0.45 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L151) |
| selfplay.cpuctUtilityStdevPrior | 0.40 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:57) | 0.40 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:57) | — | — |
| selfplay.cpuctUtilityStdevPriorWeight | 2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:58) | 2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:58) | — | — |
| selfplay.cpuctUtilityStdevScale | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:59) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:59) | — | — |
| selfplay.cudaDeviceToUseModel0Thread0 | — | — | 0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:117) | — |
| selfplay.cudaDeviceToUseModel0Thread1 | — | — | 1 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:118) | — |
| selfplay.cudaDeviceToUseModel0Thread2 | — | — | 2 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:119) | — |
| selfplay.cudaDeviceToUseModel0Thread3 | — | — | 3 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:120) | — |
| selfplay.cudaDeviceToUseModel0Thread4 | — | — | 4 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:121) | — |
| selfplay.cudaDeviceToUseModel0Thread5 | — | — | 5 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:122) | — |
| selfplay.cudaDeviceToUseModel0Thread6 | — | — | 6 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:123) | — |
| selfplay.cudaDeviceToUseModel0Thread7 | — | — | 7 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:124) | — |
| selfplay.cudaUseFP16 | auto [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:147) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:32) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:126) | — |
| selfplay.cudaUseNHWC | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:148) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:148) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:127) | — |
| selfplay.dataBoardLen | — | — | 19 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:11) | 15 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L36) |
| selfplay.drawEquivalentWinsForWhite | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:51) | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:51) | 0.5 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:156) | — |
| selfplay.drawRandRadius | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:138) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:138) | 0.5 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:94) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L105) |
| selfplay.dynamicScoreCenterScale | — | — | 0.50 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:154) | — |
| selfplay.dynamicScoreCenterZeroWeight | — | — | 0.25 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:153) | — |
| selfplay.dynamicScoreUtilityFactor | — | — | 0.30 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:152) | 0.00 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L142) |
| selfplay.earlyForkGameExpectedMoveProp | 0.025 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:94) | 0.025 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:94) | 0.025 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:18) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L63) |
| selfplay.earlyForkGameMaxChoices | 12 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:97) | 12 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:97) | 12 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:21) | 12 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L66) |
| selfplay.earlyForkGameProb | 0.04 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:93) | 0.04 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:93) | 0.04 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:17) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L62) |
| selfplay.estimateLeadProb | — | — | 0.50 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:58) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L85) |
| selfplay.estimateLeadVisits | — | — | 10 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:59) | — |
| selfplay.fancyKomiVarying | — | — | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:61) | false [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L86) |
| selfplay.firstFileRandMinProp | 0.15 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:90) | 0.15 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:90) | 0.15 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:14) | 0.15 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L52) |
| selfplay.firstPassWinRules | — | — | — | false,false,false,false,false,true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L6) |
| selfplay.forkCompensateKomiProb | — | — | 0.80 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:89) | 0.00 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L100) |
| selfplay.forkGameMaxChoices | 36 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:98) | 36 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:98) | 36 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:22) | 36 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L67) |
| selfplay.forkGameMinChoices | 3 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:96) | 3 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:96) | 3 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:20) | 3 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L65) |
| selfplay.forkGameProb | 0.01 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:95) | 0.01 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:95) | 0.01 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:19) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L64) |
| selfplay.forkSidePositionProb | — | — | 0.020 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:38) | 0.040 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L60) |
| selfplay.fpuParentWeightByVisitedPolicy | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:65) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:65) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:176) | — |
| selfplay.fpuParentWeightByVisitedPolicyPow | 2.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:66) | 2.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:66) | 2.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:177) | — |
| selfplay.fpuReductionMax | 0.2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:63) | 0.2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:63) | 0.2 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:166) | — |
| selfplay.game | go [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:3) | freestyle [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/games/gomoku/env.cfg:3) | — | — |
| selfplay.gpuToUseThread0 | — | — | — | 0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L14) |
| selfplay.gpuToUseThread1 | — | — | — | 1 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L15) |
| selfplay.gpuToUseThread2 | — | — | — | 5 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L16) |
| selfplay.handicapAsymmetricPlayoutProb | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:121) | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:121) | 0.5 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:50) | — |
| selfplay.handicapCompensateKomiProb | — | — | 0.60 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:88) | — |
| selfplay.handicapKomiInterpZeroProb | — | — | 0.05 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:91) | — |
| selfplay.handicapProb | 0.10 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:135) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:35) | 0.10 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:87) | — |
| selfplay.hasButtons | false,false,true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:9) | — | false,false,true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:74) | — |
| selfplay.initGamesWithPolicy | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:101) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:101) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:34) | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L56) |
| selfplay.koRules | SIMPLE,POSITIONAL,SITUATIONAL [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:5) | — | SIMPLE,POSITIONAL,SITUATIONAL [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:70) | — |
| selfplay.komiAuto | — | — | True [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:80) | False [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L97) |
| selfplay.komiBigStdev | — | — | 12.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:83) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L103) |
| selfplay.komiBigStdevProb | — | — | 0.05 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:82) | 0.00 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L102) |
| selfplay.komiBiggerStdev | — | — | 45.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:85) | — |
| selfplay.komiBiggerStdevProb | — | — | 0.005 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:84) | — |
| selfplay.komiMean | 7.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:13) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/games/gomoku/env.cfg:8) | — | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L98) |
| selfplay.komiStdev | — | — | 1.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:81) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L99) |
| selfplay.lcbStdevs | 5.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:44) | 5.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:44) | 5.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:145) | 5.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L135) |
| selfplay.logGamesEvery | 10 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:84) | 10 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:84) | 20 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:6) | 500 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L44) |
| selfplay.logMoves | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:83) | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:83) | false [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:5) | false [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L43) |
| selfplay.logSearchInfo | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:82) | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:82) | false [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:4) | false [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L42) |
| selfplay.logToStdout | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:85) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:85) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:7) | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L45) |
| selfplay.maxAsymmetricRatio | 8.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:123) | 8.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:123) | 8.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:52) | 4.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L78) |
| selfplay.maxDataQueueSize | 2000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:88) | 2000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:88) | 2000 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:12) | 100000 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L49) |
| selfplay.maxMovesPerGame | 1600 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:132) | 1600 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:132) | 1600 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:66) | 30000 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L90) |
| selfplay.maxRowsPerTrainFile | 20000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:89) | 20000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:89) | 20000 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:13) | 5000 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L50) |
| selfplay.maxRowsPerValFile | — | — | — | 5000 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L51) |
| selfplay.maxVisits | 600 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:8) | 100 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:5) | 2000 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:99) | 150 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L30) |
| selfplay.minAsymmetricCompensateKomiProb | — | — | 0.4 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:53) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L79) |
| selfplay.minVisitPropForLCB | 0.15 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:45) | 0.15 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:45) | 0.15 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:146) | 0.15 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L136) |
| selfplay.moveLimitProb | — | — | — | 0.2 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L7) |
| selfplay.multiStoneSuicideLegals | false,true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:8) | — | false,true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:73) | — |
| selfplay.nnCacheSizePowerOfTwo | 21 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:143) | 23 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:31) | 24 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:105) | 24 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L114) |
| selfplay.nnMaxBatchSize | 128 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:142) | 64 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:30) | 192 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:104) | 256 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L28) |
| selfplay.nnMutexPoolSizePowerOfTwo | 15 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:144) | 15 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:144) | 18 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:106) | 20 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L115) |
| selfplay.nnPolicyTemperature | — | — | — | 1.1 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L159) |
| selfplay.nnRandomize | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:146) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:146) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:108) | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L116) |
| selfplay.noResultStdev | 0.166666666 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:139) | 0.166666666 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:139) | 0.166666666 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:95) | 0.4 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L106) |
| selfplay.noResultUtilityForWhite | 0.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:50) | 0.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:50) | 0.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:155) | — |
| selfplay.noResultUtilityReduce | — | — | — | 0.5 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L149) |
| selfplay.normalAsymmetricPlayoutProb | 0.01 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:122) | 0.01 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:122) | 0.01 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:51) | 0.1 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L77) |
| selfplay.numGameThreads | 128 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:131) | 512 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:26) | 800 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:65) | 600 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L27) |
| selfplay.numNNServerThreadsPerModel | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:145) | 2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:29) | 8 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:107) | 2 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L13) |
| selfplay.numSearchThreads | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:12) | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:12) | 1 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:100) | 1 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L110) |
| selfplay.numVirtualLossesPerThread | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:60) | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:60) | 1 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:179) | — |
| selfplay.policyInitAreaProp | 0.04 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:102) | 0.04 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:102) | 0.08 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:35) | — |
| selfplay.policyInitAreaTemperature | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:105) | 1.6 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:23) | — | 1.6 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L57) |
| selfplay.policyInitAvgMoveNum | 12 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:104) | 6 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:22) | — | — |
| selfplay.policyInitProp | — | — | — | 10 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L58) |
| selfplay.policyOptimism | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:17) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:17) | — | — |
| selfplay.policySurpriseDataWeight | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:126) | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:126) | 0.5 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:55) | 0.5 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L82) |
| selfplay.quietSelfplay | — | — | — | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L25) |
| selfplay.reduceVisits | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:114) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:114) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:44) | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L72) |
| selfplay.reduceVisitsThreshold | 0.9 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:115) | 0.9 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:115) | 0.9 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:45) | 0.9 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L73) |
| selfplay.reduceVisitsThresholdLookback | 3 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:116) | 3 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:116) | 3 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:46) | 3 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L74) |
| selfplay.reducedVisitsMin | 100 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:117) | 20 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:17) | 350 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:47) | 20 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L33) |
| selfplay.reducedVisitsWeight | 0.1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:118) | 0.1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:118) | 0.1 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:48) | 0.1 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L75) |
| selfplay.rootDesiredPerChildVisitsCoeff | 2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:37) | 2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:37) | 2 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:141) | 2 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L131) |
| selfplay.rootDirichletNoiseTotalConcentration | 10.83 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:33) | 10.83 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:33) | 10.83 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:138) | 10.83 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L128) |
| selfplay.rootDirichletNoiseWeight | 0.25 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:34) | 0.25 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:34) | 0.25 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:139) | 0.25 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L129) |
| selfplay.rootEndingBonusPoints | — | — | 0.5 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:158) | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L154) |
| selfplay.rootFpuReductionMax | 0.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:64) | 0.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:64) | 0.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:167) | — |
| selfplay.rootNoiseEnabled | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:32) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:32) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:137) | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L127) |
| selfplay.rootNumSymmetriesToSample | 4 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:40) | 4 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:40) | 4 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:142) | 4 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L132) |
| selfplay.rootPolicyOptimism | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:18) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:18) | — | — |
| selfplay.rootPolicyTemperature | 1.1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:25) | 1.1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:25) | 1.1 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:162) | 1.2 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L158) |
| selfplay.rootPolicyTemperatureEarly | 1.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:24) | 1.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:24) | 1.5 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:161) | 1.8 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L157) |
| selfplay.rootPruneUselessMoves | — | — | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:159) | false [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L155) |
| selfplay.scoringRules | AREA,TERRITORY [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:6) | — | AREA,TERRITORY [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:71) | — |
| selfplay.sekiForkHackProb | — | — | 0.01 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:23) | — |
| selfplay.sgfCompensateKomiProb | — | — | 0.85 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:90) | 0.00 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L101) |
| selfplay.sgfKomiInterpZeroProb | — | — | 0.15 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:92) | — |
| selfplay.sidePositionProb | 0.020 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:110) | 0.020 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:110) | — | — |
| selfplay.startPosesPolicyInitAreaProp | — | — | 0.0025 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:36) | — |
| selfplay.staticScoreUtilityFactor | — | — | 0.05 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:151) | 0.00 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L141) |
| selfplay.stopIfNewNet | — | — | — | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L24) |
| selfplay.subtreeValueBiasFactor | 0.30 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:72) | 0.45 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:13) | 0.30 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:170) | — |
| selfplay.subtreeValueBiasFreeProp | 0.8 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:74) | 0.8 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:74) | — | — |
| selfplay.subtreeValueBiasWeightExponent | 0.8 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:73) | 0.85 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg:14) | 0.8 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:171) | — |
| selfplay.switchNetsMidGame | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:130) | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:130) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:60) | false [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L23) |
| selfplay.taxRules | NONE,NONE,SEKI,SEKI,ALL [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/env.cfg:7) | — | NONE,NONE,SEKI,SEKI,ALL [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:72) | — |
| selfplay.trtDeviceToUseThread3 | — | — | — | 4 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L17) |
| selfplay.trtDeviceToUseThread4 | — | — | — | 3 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L18) |
| selfplay.trtDeviceToUseThread5 | — | — | — | 2 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L19) |
| selfplay.trtDeviceToUseThread6 | — | — | — | 1 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L20) |
| selfplay.trtDeviceToUseThread7 | — | — | — | 0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L21) |
| selfplay.useForbiddenInput | — | — | — | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L2) |
| selfplay.useGraphSearch | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:77) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:77) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:174) | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L11) |
| selfplay.useLcbForSelection | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:43) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:43) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:144) | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L134) |
| selfplay.useNoisePruning | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:16) | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:16) | — | — |
| selfplay.useNonBuggyLcb | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:46) | true [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:46) | true [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:173) | — |
| selfplay.useUncertainty | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:15) | false [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:15) | — | — |
| selfplay.useVCFInput | — | — | — | true [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L1) |
| selfplay.validationProp | — | — | — | 0.0 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L9) |
| selfplay.valueSurpriseDataWeight | 0.1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:127) | 0.1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:127) | 0.1 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:56) | 0.1 [↗](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/selfplay.cfg#L83) |
| selfplay.valueWeightExponent | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:69) | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:69) | 0.5 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:168) | — |
| selfplay.winLossUtilityFactor | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:49) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg:49) | 1.0 [↗](/home/sky/RL/SkyZero-lab/KataGo/cpp/configs/training/selfplay8mainb18.cfg:150) | — |


### shuffle

| 原始字段 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| shuffle.filtering.exclude | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:54) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:54) | — | — |
| shuffle.filtering.exclude-basename | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:60) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:60) | — | — |
| shuffle.filtering.exclude-prefix | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:57) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:57) | — | — |
| shuffle.filtering.only-include-md5-path-prop-lbound | 0.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:63) | 0.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:63) | — | — |
| shuffle.filtering.only-include-md5-path-prop-ubound | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:66) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:66) | — | — |
| shuffle.filtering.skip-mtime-range-end | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:68) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:68) | — | — |
| shuffle.filtering.skip-mtime-range-start | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:67) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:67) | — | — |
| shuffle.filtering.summary-file | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:51) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:51) | — | — |
| shuffle.output.approx-rows-per-bucket | 70000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:41) | 70000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:41) | — | — |
| shuffle.output.approx-rows-per-out-file | 70000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:34) | 70000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:34) | — | — |
| shuffle.output.exclude-qvalues | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:47) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:47) | — | — |
| shuffle.output.include-meta | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:44) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:44) | — | — |
| shuffle.replay.add-to-data-rows | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:18) | 0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:18) | — | — |
| shuffle.replay.expand-window-per-row | 0.4 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:6) | 0.3 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/shuffle.cfg:5) | — | — |
| shuffle.replay.keep-target-rows | 20000000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:4) | 20000000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:4) | — | — |
| shuffle.replay.max-rows | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:15) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:15) | — | — |
| shuffle.replay.min-rows | 250000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:5) | 150000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/shuffle.cfg:4) | — | — |
| shuffle.replay.taper-window-exponent | 0.65 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:7) | 0.8 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/shuffle.cfg:6) | — | — |
| shuffle.replay.taper-window-scale | 50000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:8) | 150000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/shuffle.cfg:7) | — | — |
| shuffle.workers.num-processes | 12 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:21) | 12 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:21) | — | — |
| shuffle.workers.num-waves | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:26) | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:26) | — | — |
| shuffle.workers.worker-group-size | 80000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:31) | 80000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/shuffle.cfg:31) | — | — |


### train

| 原始字段 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| train.attention.attn-logit-penalty-batch-frac | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:237) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:237) | — | — |
| train.attention.attn-logit-penalty-cap | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:231) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:231) | — | — |
| train.attention.attn-logit-penalty-coeff | 0.001 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:234) | 0.001 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:234) | — | — |
| train.brenorm.brenorm-adjustment-scale | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:227) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:227) | — | — |
| train.brenorm.brenorm-avg-momentum | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:218) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:218) | — | — |
| train.brenorm.brenorm-target-dmax | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:224) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:224) | — | — |
| train.brenorm.brenorm-target-rmax | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:221) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:221) | — | — |
| train.checkpoints.no-longterm-checkpoints | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:192) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:192) | — | — |
| train.data.data-prefetch-depth | 2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:210) | 2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:210) | — | — |
| train.data.max-train-steps-since-last-reload | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:205) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:205) | — | — |
| train.data.sleep-seconds-per-epoch | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:200) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:200) | — | — |
| train.learning_rate.head-lr-factor | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:69) | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:69) | — | — |
| train.learning_rate.lr-cycle-down-epochs | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:66) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:66) | — | — |
| train.learning_rate.lr-cycle-high | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:60) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:60) | — | — |
| train.learning_rate.lr-cycle-low | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:57) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:57) | — | — |
| train.learning_rate.lr-cycle-up-epochs | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:63) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:63) | — | — |
| train.learning_rate.lr-scale | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:45) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:45) | — | — |
| train.learning_rate.lr-scale-auto | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:48) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:48) | — | — |
| train.learning_rate.lr-scale-auto2 | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:51) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:51) | — | — |
| train.learning_rate.lr-schedule | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:54) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:54) | — | — |
| train.learning_rate.no-lr-warmup | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:77) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:77) | — | — |
| train.learning_rate.noreg-lr-factor | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:72) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:72) | — | — |
| train.lookahead.lookahead-alpha | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:97) | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:97) | — | — |
| train.lookahead.lookahead-k | 6 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:94) | 6 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:94) | — | — |
| train.lookahead.lookahead-print | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:100) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:100) | — | — |
| train.loss.disable-optimistic-policy | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:159) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:159) | — | — |
| train.loss.intermediate-loss-scale | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:177) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:177) | — | — |
| train.loss.main-loss-scale | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:174) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:174) | — | — |
| train.loss.meta-kata-only-soft-policy | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:162) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:162) | — | — |
| train.loss.muzero-hidden-gradient-scale | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:153) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:153) | — | — |
| train.loss.soft-policy-weight-scale | 8.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:156) | 8.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:156) | — | — |
| train.loss.td-value-loss-scales | 0.6,0.6,0.6 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:168) | 0.6,0.6,0.6 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:168) | — | — |
| train.loss.value-loss-scale | 0.6 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:165) | 0.6 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:165) | — | — |
| train.loss.variance-time-loss-scale | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:171) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:171) | — | — |
| train.optimizer.aurora-pp-beta | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:33) | 0.5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:33) | — | — |
| train.optimizer.aurora-pp-iterations | 2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:30) | 2 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:30) | — | — |
| train.optimizer.gnorm-clip-scale | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:15) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:15) | — | — |
| train.optimizer.muon-adam-lr-factor | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:12) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:12) | — | — |
| train.optimizer.ns-steps | 5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:36) | 5 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:36) | — | — |
| train.optimizer.use-adamw | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:18) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:18) | — | — |
| train.optimizer.use-aurora | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:27) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:27) | — | — |
| train.optimizer.use-muon | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:21) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:21) | — | — |
| train.optimizer.use-normuon | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:24) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:24) | — | — |
| train.optimizer.use-polar-express | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:39) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:39) | — | — |
| train.precision.no-compile | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:118) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:118) | — | — |
| train.precision.use-bf16 | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:115) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:115) | — | — |
| train.precision.use-fp16 | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:112) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:112) | — | — |
| train.precision.use-tf32-matmul | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:121) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:121) | — | — |
| train.swa.swa-period-samples | 80000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:103) | 80000 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:103) | — | — |
| train.swa.swa-scale | 8 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:106) | 8 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:106) | — | — |
| train.training.batch-size | 128 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:4) | 128 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:4) | — | — |
| train.training.sub-epochs | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:8) | 1 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:8) | — | — |
| train.validation.max-val-samples | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:181) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:181) | — | — |
| train.validation.randomize-val | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:184) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:184) | — | — |
| train.weight_decay.input-wd-factor | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:81) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:81) | — | — |
| train.weight_decay.normal-attn-wd-factor | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:87) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:87) | — | — |
| train.weight_decay.normal-wd-factor | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:84) | 1.0 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:84) | — | — |
| train.weight_decay.wd-floor-frac | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:90) | 空赋值 [↗](/home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/train.cfg:90) | — | — |


## 5. V4 外层解析结果与网络预设全部字段

RunConfig 表列出正式解析得到的全部字段。native_selfplay 的全部内容已由第 3/4 节覆盖；train_args/shuffle_args 在第 6 节展开，避免把长命令塞入单元格。路径和运行预算由控制器注入时，不能把未写在 cfg 中理解为未设置。

| RunConfig 字段 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| selfplay_config | /home/sky/RL/EtaZero-lab/EtaZero_V4/configs/baseline_az/selfplay.cfg | /home/sky/RL/EtaZero-lab/EtaZero_V4/configs/az_gomoku/selfplay.cfg | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| canvas | 19 | 15 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| batch_size | 128 | 128 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| bootstrap_games | 800 | 800 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| train_steps | 2000 | 2000 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| native_selfplay | 第 3/4 节：翻译后的原生参数与配置声明 | 第 3/4 节：翻译后的原生参数与配置声明 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| model_kind | b5c192nbt-fson-mish | b5c192nbt-fson-mish | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| shuffle_processes | 12 | 12 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| shuffle_rows_per_file | 70000 | 70000 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| shuffle_keep_rows | 20000000 | 20000000 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| train_args | 第 6 节逐项列出 | 第 6 节逐项列出 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| shuffle_args | 第 6 节逐项列出 | 第 6 节逐项列出 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| game | go | freestyle | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| algorithm | alphazero | alphazero | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| muzero_network | 未指定 | 未指定 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| muzero_unroll_steps | 未指定 | 未指定 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| publish_weights | swa | swa | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| replay_ratio | 4.0 | 4.0 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| min_rows | 250000 | 150000 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| bootstrap_game_multiplier | 1.05 | 1.05 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| gatekeeper_config | 未指定 | 未指定 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| gatekeeper_required_win_prop | 0.5 | 0.5 | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |
| auto_plot | true | true | 无 V4 外层接口 | 不同脚本接口，见第 6 节 |

以下为实际选中网络预设的全部字段。az_gomoku 的 version 在落子棋适配后为 18；KataGomo 自有 version=102，不是相同输入/输出协议。KataGo selfplay 文件未指定网络；KataGomo 网络名称来自运行脚本，不能据此宣称当前 V4 小网络的最优 visits 与其相同。

| 预设字段 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 |
| --- | --- | --- | --- | --- |
| activation | mish | mish | 未由文件指定 | mish |
| block_kind | [["rconv1","bottlenest2"],["rconv2","bottlenest2gpool"],["rconv3","bottlenest2"],["rconv4","bottlenest2gpool"],["rconv5","bottlenest2"]] | [["rconv1","bottlenest2"],["rconv2","bottlenest2gpool"],["rconv3","bottlenest2"],["rconv4","bottlenest2gpool"],["rconv5","bottlenest2"]] | 未由文件指定 | [["rconv1","bottlenest2"],["rconv2","bottlenest2"],["rconv3","bottlenest2gpool"],["rconv4","bottlenest2"],["rconv5","bottlenest2"],["rconv6","bottlenest2gpool"],["rconv7","bottlenest2"],["rconv8","bottlenest2"],["rconv9","bottlenest2gpool"],["rconv10","bottlenest2"],["rconv11","bottlenest2"],["rconv12","bottlenest2gpool"],["rconv13","bottlenest2"],["rconv14","bottlenest2"],["rconv15","bottlenest2gpool"],["rconv16","bottlenest2"],["rconv17","bottlenest2"],["rconv18","bottlenest2gpool"],["rconv19","bottlenest2"],["rconv20","bottlenest2"],["rconv21","bottlenest2gpool"],["rconv22","bottlenest2"],["rconv23","bottlenest2"],["rconv24","bottlenest2gpool"],["rconv25","bottlenest2"],["rconv26","bottlenest2"],["rconv27","bottlenest2gpool"],["rconv28","bottlenest2"]] |
| bnorm_epsilon | 0.0001 | 0.0001 | 未由文件指定 | 0.0001 |
| bnorm_running_avg_momentum | 0.001 | 0.001 | 未由文件指定 | 0.001 |
| bnorm_use_gamma | true | true | 未由文件指定 | — |
| g1_num_channels | 32 | 32 | 未由文件指定 | 64 |
| gamma_weight_decay_center_1 | true | true | 未由文件指定 | — |
| gpool_num_channels | 32 | 32 | 未由文件指定 | 64 |
| initial_conv_1x1 | false | false | 未由文件指定 | false |
| mid_num_channels | 96 | 96 | 未由文件指定 | 256 |
| norm_kind | fixscaleonenorm | fixscaleonenorm | 未由文件指定 | fixup |
| num_attention_pool_heads | — | — | 未由文件指定 | 4 |
| num_scorebeliefs | — | — | 未由文件指定 | 8 |
| p1_num_channels | 32 | 32 | 未由文件指定 | 64 |
| sbv2_num_channels | — | — | 未由文件指定 | 128 |
| trunk_num_channels | 192 | 192 | 未由文件指定 | 512 |
| use_attention_pool | — | — | 未由文件指定 | false |
| v1_num_channels | 32 | 32 | 未由文件指定 | 96 |
| v2_size | 80 | 80 | 未由文件指定 | 144 |
| version | 15 | 18 | 未由文件指定 | 102 |

网络字典来源：[modelconfigs.py](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/katago/train/modelconfigs.py)；[modelconfigs.py](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/modelconfigs.py#L1261)。MuZero 网络位于 baseline_mz/net.cfg；Gumbel 搜索位于共享算法配置，均不计入这两份 AlphaZero 配方的声明表。

## 6. train / shuffle 全部 CLI 声明与脚本赋值

此表逐项列出两边源码的全部 argparse 参数，而不是只列 cfg 中出现的参数。`解析默认 None` 表示后续逻辑处理，**不一定是最终生效值**；条件缺省（如 SWA）在前文解释。KataGomo 列是按上述运行脚本参数代入仓库包装脚本后的结果，未实际运行脚本。未由脚本设置的选项保留 argparse 默认；不存在的选项标明无此接口。

### train

| CLI 参数 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 | 声明来源 / 属性 |
| --- | --- | --- | --- | --- | --- |
| always-initial-checkpoint | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:89) store_true |
| attn-logit-penalty-batch-frac | 1.0 | 1.0 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:168) |
| attn-logit-penalty-cap | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:166) |
| attn-logit-penalty-coeff | 0.001 | 0.001 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:167) |
| aurora-pp-beta | 0.5 | 0.5 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:128) |
| aurora-pp-iterations | 2 | 2 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:127) |
| batch-size | 128 | 128 | 未由文件指定 | 128（每 GPU） | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:93)（必填）；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L65)（必填） |
| brenorm-adjustment-scale | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:164)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L100) |
| brenorm-avg-momentum | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:161)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L97) |
| brenorm-target-dmax | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:163)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L99) |
| brenorm-target-rmax | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:162)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L98) |
| data-prefetch-depth | 2 | 2 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:151) |
| datadir | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | ../data/shuffleddata/current/ | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:80)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L59)（必填） |
| disable-optimistic-policy | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:170) store_true |
| epochs-per-export | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:138)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L81) |
| export-prob | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:139)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L82) |
| exportdir | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | ../data/torchmodels_toexport | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:82)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L60) |
| exportprefix | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | b28c512n | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:83)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L61) |
| gnorm-clip-scale | 1.0 | 1.0 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:115)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L70) |
| gnorm-stats-debug | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 不传；解析默认 false | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:159) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L95) store_true |
| h | 不传；解析默认 表达式 argparse.SUPPRESS | 不传；解析默认 表达式 argparse.SUPPRESS | 未由文件指定 | 不传；解析默认 表达式 argparse.SUPPRESS | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:71) help；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L50) help |
| head-lr-factor | 0.5 | 0.5 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:108) |
| initial-checkpoint | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:84)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L62) |
| initial-train-bucket-level | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:147) |
| initial-weights | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:85) |
| initialize-only | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:86) store_true |
| input-features | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:98) |
| input-wd-factor | 1.0 | 1.0 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:111) |
| intermediate-loss-scale | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:177)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L107) |
| latestdatadir | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:81) |
| lookahead-alpha | 0.5 | 0.5 | 未由文件指定 | 0.5 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:120)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L75) |
| lookahead-k | 6 | 6 | 未由文件指定 | 6 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:119)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L74) |
| lookahead-print | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 不传；解析默认 false | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:121) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L76) store_true |
| lr-cycle-down-epochs | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:107) |
| lr-cycle-high | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:105) |
| lr-cycle-low | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:104) |
| lr-cycle-up-epochs | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:106) |
| lr-scale | 1.0 | 1.0 | 未由文件指定 | 1 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:100)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L68) |
| lr-scale-auto | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 不传；解析默认 false | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:101) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L69) store_true |
| lr-scale-auto2 | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:102) |
| lr-schedule | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:103) |
| main-loss-scale | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:176)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L106) |
| max-epochs-this-instance | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 1 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:140)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L83) |
| max-train-bucket-per-new-data | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:145)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L85) |
| max-train-bucket-size | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:146)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L86) |
| max-train-steps-since-last-reload | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:148)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L87) |
| max-training-samples | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:141) |
| max-val-samples | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:150)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L89) |
| meta-kata-only-soft-policy | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:171) store_true |
| model-kind | b5c192nbt-fson-mish | b5c192nbt-fson-mish | 未由文件指定 | b28c512nbt-mish | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:97)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L67) |
| multi-gpus | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 0,1 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:132)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L78) |
| muon-adam-lr-factor | 1.0 | 1.0 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:110) |
| muzero-config | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:95) |
| muzero-hidden-gradient-scale | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:96) |
| no-compile | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:135) store_true |
| no-export | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 不传；解析默认 false | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:153) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L91) store_true |
| no-longterm-checkpoints | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:154) store_true |
| no-lr-warmup | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:158) store_true |
| no-repeat-files | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 不传；解析默认 false | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:155) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L92) store_true |
| noreg-lr-factor | 1.0 | 1.0 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:109) |
| normal-attn-wd-factor | 1.0 | 1.0 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:113) |
| normal-wd-factor | 1.0 | 1.0 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:112) |
| ns-steps | 5 | 5 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:129) |
| pos-len | 19 | 15 | 未由文件指定 | 15 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:92)（必填）；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L64)（必填） |
| quit-if-no-data | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 不传；解析默认 false | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:156) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L93) store_true |
| randomize-val | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 不传；解析默认 false | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:152) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L90) store_true |
| required-initial-checkpoint-train-steps | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:90) |
| samples-per-epoch | 控制器：256000 | 控制器：256000 | 未由文件指定 | 2000000 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:94)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L66) |
| sleep-seconds-per-epoch | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:144)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L84) |
| soft-policy-weight-scale | 8.0 | 8.0 | 未由文件指定 | 8 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:169)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L102) |
| stop-at-samples | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:142) |
| stop-when-train-bucket-limited | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 不传；解析默认 false | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:149) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L88) store_true |
| sub-epochs | 1 | 1 | 未由文件指定 | 不传；解析默认 1 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:116)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L71) |
| swa-period-samples | 80000 | 80000 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:117)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L72) |
| swa-scale | 8 | 8 | 未由文件指定 | 1 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:118)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L73) |
| td-value-loss-scales | 0.6,0.6,0.6 | 0.6,0.6,0.6 | 未由文件指定 | 0.6,0.6,0.6 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:173)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L104) |
| traindir | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | ../data/train/b28c512n | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:79)（必填）；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L58)（必填） |
| use-adamw | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:123) store_true |
| use-aurora | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:126) store_true |
| use-bf16 | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:134) store_true |
| use-fp16 | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | true（旗标） | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:133) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L79) store_true |
| use-muon | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:124) store_true |
| use-normuon | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:125) store_true |
| use-polar-express | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:130) store_true |
| use-tf32-matmul | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:136) store_true |
| value-loss-scale | 0.6 | 0.6 | 未由文件指定 | 0.6 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:172)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/train.py#L103) |
| variance-time-loss-scale | 1.0 | 1.0 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:174) |
| wd-floor-frac | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/train.py:114) |


### shuffle

| CLI 参数 | V4 baseline_az | V4 az_gomoku | KataGo main18b | KataGomo 15 路 | 声明来源 / 属性 |
| --- | --- | --- | --- | --- | --- |
| add-to-data-rows | 0 | 0 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:825)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L353) |
| add-to-window-size | 无此 CLI 接口 | 无此 CLI 接口 | 未由文件指定 | 不传；解析默认 未指定 | [KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L354) |
| approx-rows-per-bucket | 70000 | 70000 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:832) |
| approx-rows-per-out-file | 70000 | 70000 | 未由文件指定 | 50000 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:831)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L358) |
| batch-size | 无此 CLI 接口 | 无此 CLI 接口 | 未由文件指定 | 128 | [KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L360)（必填） |
| dirs | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | ../data/selfplay/ | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:808)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L336) |
| dry-run-print-resource-cost | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:835) |
| ensure-batch-multiple | 无此 CLI 接口 | 无此 CLI 接口 | 未由文件指定 | 不传；解析默认 未指定 | [KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L361) |
| exclude | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:838)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L363) |
| exclude-basename | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 不传；解析默认 false | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:840) store_true；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L365) store_true |
| exclude-prefix | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:839)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L364) |
| exclude-qvalues | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:846) store_true |
| expand-window-per-row | 0.4 | 0.3 | 未由文件指定 | 0.3 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:822)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L350)（必填） |
| h | 不传；解析默认 表达式 argparse.SUPPRESS | 不传；解析默认 表达式 argparse.SUPPRESS | 未由文件指定 | 不传；解析默认 表达式 argparse.SUPPRESS | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:812) help；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L340) help |
| include-meta | 不传；解析默认 false | 不传；解析默认 false | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:845) store_true |
| input-files-stable | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:827) store_true |
| keep-target-rows | 20000000 | 20000000 | 未由文件指定 | train 2100000；val 51200 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:821)（必填）；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L349) |
| max-rows | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:820)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L348) |
| min-rows | 250000 | 150000 | 未由文件指定 | 150000 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:819)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L347) |
| muzero-unroll-steps | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:833) |
| num-processes | 12 | 12 | 未由文件指定 | 32 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:836)（必填）；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L359)（必填） |
| num-waves | 1 | 1 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:834) |
| only-include-md5-path-prop-lbound | 0.0 | 0.0 | 未由文件指定 | train 0；val 0.97 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:841)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L366) |
| only-include-md5-path-prop-ubound | 1.0 | 1.0 | 未由文件指定 | train 0.97；val 1 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:842)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L367) |
| out-dir | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | ../data/shuffleddata/<时间戳>/{train,val} | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:829)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L356)（必填） |
| out-tmp-dir | 控制器按运行阶段管理/注入 | 控制器按运行阶段管理/注入 | 未由文件指定 | ./ktmp/{train,val} | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:830)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L357)（必填） |
| output-npz | 无此 CLI 接口 | 无此 CLI 接口 | 未由文件指定 | true（旗标） | [KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L368) store_true |
| skip-mtime-range-end | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:844) |
| skip-mtime-range-start | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 无此 CLI 接口 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:843) |
| summary-file | 不传；解析默认 未指定 | 不传；解析默认 未指定 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:826)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L355) |
| taper-window-exponent | 0.65 | 0.8 | 未由文件指定 | 0.8 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:823)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L351)（必填） |
| taper-window-scale | 50000 | 150000 | 未由文件指定 | 不传；解析默认 未指定 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:824)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L352) |
| worker-group-size | 80000 | 80000 | 未由文件指定 | 不传；解析默认 80000 | [V4](/home/sky/RL/EtaZero-lab/EtaZero_V4/python/shuffle.py:837)；[KataGomo](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.py#L362) |


### KataGomo 外层脚本的其他参数

| 步骤 | 完整显式参数 / 控制值 | 来源 |
| --- | --- | --- |
| selfplay 命令 | -models-dir data/models；-config selfplay.cfg；-output-dir data/selfplay；-max-games-total 50000 | [run_selfplay.sh](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/run_selfplay.sh) |
| shuffle 包装入口 | BASEDIR=../data；TMPDIR=./ktmp；NTHREADS=32；BATCHSIZE=128 | [run_selfplay.sh](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/run_selfplay.sh) |
| train 包装入口 | BASEDIR=../data；TRAININGNAME=b28c512n；MODELKIND=b28c512nbt-mish；BATCHSIZE=128；EXPORTMODE=main；附加 -multi-gpus 0,1 | [run_selfplay.sh](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/run_selfplay.sh) |
| export 包装入口 | export.sh renju28b ../data 0；此处保留原始位置参数，未核对部署 export.sh 的内部默认值 | [run_selfplay.sh](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/run_selfplay.sh) |
| shuffle 目录维护 | 等待 NFS 10 秒；current 软链接切换；超过 120 分钟的旧目录仅保留最新 5 个 | [shuffle.sh](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/shuffle.sh) |
| 运行循环 | while true；selfplay 后依次 shuffle/train/export/view_loss，无额外门控命令 | [run_selfplay.sh](https://github.com/hzyhhzy/KataGomo/blob/df152116e3787c75c6a3de099d261ca092b7dfc1/python/scripts/run_selfplay.sh) |


## 7. 验证与适用范围

探索配置检查：20 个入口的噪声与算法约束检查通过，Hex／象棋的 half-life 分别为 15／40，其他棋类和四个 baseline 的配置未变。四种象棋算法各运行 2 局 Eigen 小模型自对弈，均产生超过第 24 手的样本及真实终局价值监督；该检查使用 33 visits、较小 cheap/reduced 预算和较小 NN 服务资源，并保留 1600 手上限、policy init、fork 和 side 采样设置。配置回归为 15 passed、4 skipped；未运行正式网络的 CUDA 吞吐或棋力实验。
配置结构验证：20 个入口均通过正式解析；4 个围棋入口与各自 baseline 完全一致；baseline_az 与 az_gomoku 的有效配置与改名前一致。Hex／象棋的 8 个入口均为 replay_ratio=2，4 个五子棋入口的环境一致。配置解析回归为 15 passed、4 skipped（未启用 GPU／训练集成测试）。16 个棋类／算法组合使用小模型通过 Eigen 原生初始化和根搜索检查；该诊断命令固定 20 visits，并缩小了 NN 服务缓存与批大小。另对 8 个 Gumbel 组合运行原生 analysis，实际根访问数均为 33。上述检查不包含正式网络的 CUDA 吞吐或棋力评估。
- 正式 V4 load_config 重新解析 baseline_az 与 az_gomoku；确认当前声明包含已授权的 NHWC=true、队列 2000、原始文件行数 20000。
- 全量配置表共 245 个唯一字段，四列分别 [188, 185, 115, 99] 个；逐列核对每个 include 展开后的声明键均已呈现。
- 静态提取 train/shuffle 的全部 argparse 声明，未执行远端脚本；网络预设由本地当前字典及 KataGomo 静态字典/后缀变换核对。
- KataGomo 的缺项和旧键由同提交 selfplay 入口及读取代码核对；没有运行它的二进制或训练，不能声称该仓库配套脚本已可直接复现。
- NR 随机效用实现此前的验证记录：CUDA/Eigen 构建通过；Eigen runtests 通过（该命令跳过了工作目录不在 Git 根下的配置解析测试）；Python 配置与 selfplay 检查 19 passed、4 skipped。新增原生测试覆盖 NR 正负效用、截断采样、零随机化、真实三劫循环终局、混合概率、效用平方统计和图搜索。真实小模型完成两轮 selfplay→shuffle→训练→导出，第二轮消费第一轮导出模型，总训练 32 个根样本；使用 numGameThreads=3、noResultStdev=0.166666666。沙箱内 GPU 访问失败后在允许 GPU 访问的环境中续跑成功，重试造成第二轮吞吐日志不完整，因此该检查不作为性能数据。没有进行并行度测速或棋力评估，不能据参数相似性宣称 az_gomoku 已调优。
