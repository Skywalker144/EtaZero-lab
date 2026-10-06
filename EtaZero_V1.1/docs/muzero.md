# MuZero

MuZero 使用独立的 latent 搜索、initial/recurrent 推理组批及轨迹展开 learner，接通自对弈、shuffle、训练、模型发布、evaluate/match 和恢复。运行配置为 [configs/muzero](../configs/muzero/)，继承 `baseline`：

```bash
CONFIG_DIR=configs/muzero bash scripts/run.sh
```

该配置保留 baseline 的训练预算、优化器及可共用的搜索启发式；三段主干规模、展开长度和并行局数在其覆盖文件中定义。这些是可运行的起始设置，尚不是经过棋力或等时间实验选择的最优默认值。Gumbel 搜索未实现，对应组合明确报错。

展开后的训练行更大，名义训练分片大小由 [MuZero train.cfg](../configs/muzero/train.cfg) 覆盖；派生配置按配置继承规则读取这些字段。shuffle 保留 baseline 的 worker 数和总数组内存预算，按行大小规划有效桶大小。

## 训练目标与优化器

`[muzero_training] auxiliary_losses=false` 只计算主 policy CE 和终局 WDL CE，系数均为 1。辅助输出结构保留，损失不读取对手、soft、TD、optimistic、误差或 Q 输出；此模式禁止开启 `predict_q_values`。搜索是否使用辅助预测由各自开关独立控制。

`katago_optimizer=false` 使用 `[optimizer] kind` 选择的普通 SGD（momentum 0.9）或 AdamW，学习率与统一 weight decay 由 `[muzero_training]` 指定。它使用 batch 均值反传，不启用 KataGo 分组 LR、自适应衰减、warmup、Lookahead 或 SWA；`gradient_clip=0` 表示不裁剪，发布当前训练权重。checkpoint 仍保存优化器、scaler、计数和 RNG。

没有配置 `[muzero_training]` 时使用共用辅助损失及 KataGo 优化器机制。关闭某组训练增强不会自动关闭搜索、采样或评估增强；这些属于独立条件。该能力不意味着逐项复现原论文或 MuZero_V2。

## 网络与推理

[网络](../python/etazero/muzero/network.py) 的 `network.architecture` 支持 `nbt` 和 `resnet`。NBT 使用 EtaZero NBT block、Mish 与 fson，每段拥有独立参数和归一化统计；[稠密主干](../python/etazero/muzero/resnet.py) 采用 MuZero_V2 的两层全宽 3×3 残差块、逐样本 masked normalization 与 SiLU。`network.channels/blocks` 配置 representation，`muzero.dynamics_*` 和 `muzero.prediction_*` 分别配置另两段；`muzero.latent_channels` 配置潜在宽度。两种主干都保留 EtaZero heads，不是 MuZero_V2 整套网络的逐层复刻。ResNet 的归一化对每个样本的所有通道和有效格点进行 FP32 归约，不维护 BatchNorm 统计，卷积保留来源的默认初始化；prediction 不额外加入末端归一化。ResNet 要求 `katago_optimizer=false`，不套用 NBT/fson 参数分组。相同通道数与块数不保证相同参数量或计算量。

- representation 接收现有五空间平面和六全局特征，包含棋规、执色、禁手特征与 PDA 条件。
- latent 是 FP32 `[B,C+1,H,W]`：前 C 通道按每样本的有效格点与全部通道做 min/max 归一化，常量特征归零；末通道保留有效棋盘 mask。采用 MiniZero 的极小跨度保护：当 `max-min < 1e-5` 时，分母为 `max-min+1e-5`，否则使用原跨度。输出保持在 `[0,1]` 内，极小跨度时最大值可小于 1；该规则同时用于 representation、dynamics、训练和导出推理。FP16 推理仍在 FP32 做归一化。
- dynamics 只接收 latent 和 int64 `[B]` 画布动作，以 one-hot 动作平面生成下一状态；不接收未来观测、占用、禁手或真实终局。
- prediction 复用完整 EtaZero heads：六项策略、主 WDL、三项 TD WDL、短期误差，以及可选纯 W−L Q。MuZero 两种主干均支持 `network.predict_q_values`；AlphaZero 的 Q 架构约束保持原样。

WDL 均表示该 latent 步当前玩家视角。棋类适配无 reward head、无 reward loss，搜索折扣为 1，相邻边翻转价值符号、交换 W/L。训练 heads 和 loss 为 FP32；NBT 导出副本把三段末端 BatchNorm 转为预计算归一化，ResNet 保留逐样本动态归一化。`initial(obs, globals)` / `recurrent(latent, action)` 返回 latent、普通策略 logits、WDL logits、short optimistic logits 和误差标准差。模型元数据独立标识算法与 latent 宽度，导入权重校验架构及完整三段配置。

[推理服务](../cpp/src/muzero/batcher.cpp) 分别组批 initial 与 recurrent 请求。initial 可以交给任一服务，recurrent 路由回创建 latent 的后端，禁止跨模型/设备误用；请求队列有容量限制、失败传播和排空机制。latent 保留在设备上，每个节点拥有独立 tensor 存储。可选 initial cache 按已变换的完整输入缓存 latent 和预测，包含固定朝向及全局条件；`muzero` 默认关闭缓存，避免沿用 AlphaZero 的大容量预测缓存而占用过多显存。

[MuZero 后端](../cpp/src/muzero/torch_backend.cpp) 为各服务建立独立 CUDA stream，复用 pinned observation、global 和 action 输入缓冲区。节点 latent 使用独立分配与 foreach 批量复制，避免单个存活节点保留整批 latent。一次预测回传完成该 stream 上此前的节点复制，结果交给搜索线程后可安全读取和复用父节点。模型加载时核对算法、画布与输入契约；运行时保留输入边界和输出形状检查，不额外归约 latent 有限性、回传 mask 或校验 heads 的实际精度。数值与 mask 正确性由独立开发测试验证。上述执行路径独立于 AlphaZero 的后端、组批队列和搜索实现。

CUDA 服务在加载时为 initial 与 recurrent 准备 batch 为 1、2、4 等二次幂及 `max_batch` 的独立 CUDA Graph，运行时选择能容纳请求的最小 bucket。NBT 的推理归一化使用固定统计，ResNet 按样本归一化；不足 bucket 的部分重复一个有效输入，丢弃额外输出，不增加搜索节点或训练目标。recurrent 通过 foreach 将父 latent 直接组装到 graph 输入，避免中间拼接分配。各 graph 保留自己的输入、输出和内存池；节点复制完成后才允许下一次重放，父节点不会引用可被覆盖的 graph 输出。准备和模型加载的耗时进入实际轮次墙钟；固定模型短测通过预热排除这一启动成本。

FP16 后端在加载时将 Conv2d / Linear 的权重与 bias 转为 autocast 原本使用的 FP16 值，避免每次前向重复转换矩阵。归一化参数、统计、mask 和 latent 保持 FP32；导出文件与 learner 权重保持原有存储精度。

## 搜索与子树加权开关

[MuZero 搜索](../cpp/src/muzero/search.cpp) 的根使用真实合法动作。根以下只使用有效棋盘 mask，允许重复选择同一点，不调用真实落子和终局判断。每次真实落子后丢弃旧树，以新观测的 representation 开始搜索。整棵树固定一个 D4 朝向，动作映射到该朝向，预测映射回画布坐标；不旋转或平均 latent。

每个节点的首次预测记为一次访问，子节点价值转换到父视角后聚合。根访问预算包含初始预测；无图共享、跨手复用或已有访问。时间限制包含根推理，但至少完成两个新 playout（根预测及一次 recurrent）后才生效；显式停止和访问／playout 上限优先，不为满足这个时间下限越过硬上限。并行选择使用虚拟损失，推理在搜索锁外执行，完成后在锁内回传。设备 latent 随清树释放；模型失败唤醒请求与搜索线程。

子树价值加权沿用共用聚合公式，并保留现有开关。在 `selfplay.cfg` 中：

```ini
[value_weighting]
value_weight_exponent = 0
```

`0` 关闭按子树价值偏好的额外加权；正数开启，指数越大偏好越强。指数由所选配置确定。这个开关不同时关闭 uncertainty、noise pruning 或 LCB；在无噪声根且 uncertainty/noise pruning 等备份修正关闭时，指数 0 的普通树回传退化为逐次预测样本的算术平均。eval/match 对应字段位于各自的 `[evaluation]` / `[match]`。

PUCT/FPU、根噪声、温度、forced playout、目标剪枝、LCB、optimistic policy、uncertainty 和 noise pruning 可按配置使用，共用纯数学函数。LCB 只是搜索启发式，短期误差 head 也不代表 dynamics 模型误差或随深度增加的不确定性；这些增强项对 MuZero 的效果需要独立实验。

Forced playout 仅作用于根上已完成 recurrent 评估的子节点。尚无子节点的候选动作使用 FPU 与普通 PUCT 竞争首次访问；正在评估的子节点跳过选点。去除根噪声的搜索不启用 forced playout。

## 连续数据与展开训练

[writer](../cpp/src/selfplay/record.cpp) 保留完整真实对局，新增所有可训练位置的 policy/Q/visits，包括重复次数为零的 cheap 位置。开局前缀不产生监督。原始文件标识 `algorithm=muzero` 和展开长度，不能将 AlphaZero 分片作为 MuZero 数据。

[数据视图](../python/etazero/muzero/data.py) 的一行是一个真实起点，包含一份起点输入、K 个动作和 K+1 份完整 targets。真实棋盘只用于生成监督与校验记录，learner 从不使用未来观测代替 dynamics。分片可以切开重复起点，但完整轨迹保证未来目标、TD 和终局不被分片截断。

- 起点由既有 PCR、Reduce Visits、surprise 和随机重复次数决定，根权重为 1，避免重复计权。真实后续步使用该位置 target weight，除以 snapshot 中所选完整主轨迹的平均权重；同一局跨分片只计一次。零权重步仍保留动作转移。
- 主 value 是终局结果，按每步执色翻转；TD 使用整局搜索 WDL 与终局结果生成的 EtaZero 三时域目标。各真实展开步监督完整 heads，reanalysis 的 outcome gate 保持原有定义。Q 目标保留每动作子节点访问，未搜索动作不参与 Q loss。
- 最后一个真实动作进入吸收态：WDL 继续按步交替，TD 等于终局 WDL，策略/对手策略为棋盘内均匀分布，Q 无访问监督。继续展开的动作在棋盘内均匀抽样，由 reader 的 checkpoint 随机状态控制；吸收态步权重为 1。
- side 独立位置只有根预测目标，后续全部 mask，不能伪造终局。learner 在根预测后移除这些行，不让虚构转移参与 recurrent BatchNorm。
- 一个 batch 共用一次 D4，空间输入、全部步的 policy/Q 和动作同步变换。forbidden dropout 只作用于真实起点输入，监督不随输入 dropout 改变。

[learner](../python/etazero/muzero/training.py) 采用来源中的梯度边界：根 loss 梯度系数 1，未来各步为 `1/K`；第二次及以后 dynamics 前按 `unroll.hidden_gradient_scale` 缩放 hidden 梯度，首次 representation→dynamics 不缩放。前向 loss 日志记录未缩放的各步损失和，反向系数不改变日志值。KataGo 优化器模式保留 batch 总和反传，普通优化器模式使用 batch 均值；consumed samples 计真实起点，不能当作展开状态数与 AlphaZero 比较。

编译训练允许因 side 行筛选而出现动态图边界，神经网络与 loss 仍使用 `torch.compile`。NBT 的 KataGo 优化模式采用独立的 [参数分组](../python/etazero/muzero/optimization.py)，共用 SGD/AdamW、Lookahead 和 SWA；representation/dynamics 的末端归一化属于内部组，prediction 的末端归一化和 heads 属于输出组。普通优化模式对两种主干使用统一 LR/weight decay，不使用上述分组、Lookahead 或 SWA。两种主干均接入 AMP、checkpoint 和验证调度。

## KataGo 工程机制的适用边界

| 机制 | MuZero 行为 |
|---|---|
| 真实规则、开局、实际落子、fork、reanalysis | 在真实环境共用，搜索本身走 latent 路径 |
| 真实棋盘图共享、NOVC 身份与 graph catch-up | 不进入 latent 树；开启 graph_search 明确报错 |
| 跨实际落子的树复用 | 禁止，重新编码真实起点 |
| 每叶随机 D4、根多对称平均 | 改为一棵树固定朝向；根多对称数必须为 1 |
| 真实终局快速回传及 terminal uncertainty 权重 | 不用于 latent 树内 |
| 观测预测缓存 | 使用独立的、保留朝向与 latent 的 initial cache |
| 压缩零重复行、逐行 shuffle | 保存完整后续目标，shuffle 打乱起点序列 |
| side 位置 | 仅根监督；不伪造吸收态 |
| 并发组批、虚拟损失、失败唤醒 | 独立管理 initial/recurrent 请求及设备 latent 生命周期 |
| checkpoint、文件校验、完整轮提交 | 共用运行管理，增加算法/三段模型/展开数据身份校验 |

恢复同时涵盖模型、归一化统计、优化器、Lookahead/SWA、scaler、训练计数、模型 RNG 和 reader 消费游标及吸收态动作 RNG。控制器保留既有整轮提交边界；learner 的轮内 checkpoint 可恢复下一批更新。evaluate/match 和常驻分析服务从实际推理设备上已加载模型的元数据选择搜索实现，并将该模型交给后端使用，不另做用于识别类型的 CPU 完整加载。可以混合 AlphaZero/MuZero 对弈；含 MuZero 的比赛应使用关闭图共享和复用的 profile。

## 训练诊断图

`training.png` 在原有概览面板下增加两项 MuZero 诊断，布局为四行两列，每轮提交自动更新，也可通过 `scripts/run.sh plot --run-dir ...` 重建：

- **Loss by unroll step**：横轴为 0 至 K，0 是 representation 的根预测。展示最新已完成轮次与所有有测量轮次的算术均值。每步包含该步完整 heads 的带权 loss，以整个起点 batch 为分母，保留真实后续步权重、side mask 和吸收态目标；各步之和等于日志总 loss。这里没有乘只作用于反向传播的 `1/K`，曲线不是单步梯度贡献。
- **Gradient norms by module**：沿轮次展示 representation h、dynamics g、prediction f。采用实际反传（已包含 hidden/loss 梯度缩放）经 AMP 反缩放后、裁剪前的参数梯度 L2 范数；总和反传时除以 batch size，均值反传时直接记录，与已有 Network 梯度图保持平均 loss 口径。三段没有共享参数，每次更新的三个范数平方和等于总范数平方；轮次均值不再要求满足该等式。side-only batch 的 dynamics 没有梯度，记录真实零值。

loss 均值包含 AMP 跳步消费，梯度均值只采用成功更新；原始 overflow 仍保存在日志。所有曲线根据 checkpoint 提交链筛选并去重，不计回滚更新。诊断只读取并 detach 已算出的 loss 和梯度，不额外反传或改变优化器更新。旧日志未记录这两类指标时显示缺测，不能事后从总 loss 或总梯度还原。

## 验证入口与限制

```bash
bash scripts/build.sh
conda run -n pytorch ctest --test-dir build --output-on-failure
conda run -n pytorch python -m pytest -q tests/test_muzero_network.py tests/test_muzero_pipeline.py
conda run -n pytorch python tests/reference/check_muzero_network.py
# 在宿主 CUDA 可访问的上下文执行
ETAZERO_GPU_TESTS=1 conda run -n pytorch python -m pytest -q tests/test_muzero_network.py tests/test_muzero_pipeline.py
```

测试覆盖 normalization、手算梯度边界、潜在树视角与访问预算、无真实树内规则、固定 D4、缓存/后端归属、并发失败、完整序列分片、cheap/side/reanalysis/吸收态、reader 与 learner 恢复，以及真实 CUDA 训练与 Python/TorchScript/LibTorch initial/recurrent 对照。初始预测与连续 recurrent 预测的数值对照在独立测试中执行，每轮导出不执行数值校验。短测试说明执行链路和所测算法性质成立，不代表训练收敛、吞吐或棋力收益已验证。
