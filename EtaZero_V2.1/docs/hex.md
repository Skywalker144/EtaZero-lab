# Hex

Hex 支持 AlphaZero / PUCT 与 MuZero / PUCT，并可与三种五子棋规则按每局权重混合训练。棋类由 `env.cfg` 的 `environment.rules` 与 `rule_weights` 选择；尺寸仍使用 `sizes` 与 `size_weights`。完整 Hex 起始 profile 为 [azh](../configs/azh/env.cfg)，训练继承与分析/比赛引用方式见 [配置组织](implementation.md#配置组织)。

## 棋规与动作

方形棋盘每格是一个六邻接顶点，邻居偏移为 `(0,-1),(1,-1),(1,0),(0,1),(-1,1),(-1,0)`。黑先行并连接上下边，白连接左右边。每手在空格放置一子；实际同色路径连通两条目标边即获胜，没有 pass、换边动作和和棋。满盘必有连接胜者，未检测到连接时明确报错。终局原因 `connection` 与五子棋的 `five`、`forbidden` 区分。

采用普通 Hex 的实际连接定义。指定来源 Hex2024 的不限手数模式还将两个空连接点形成的 bridge 纳入提前判胜；EtaZero 不执行这种虚拟连接提前终局，也未接入来源的限手数和棋、特殊模板或随机铺子模式。它们会改变轨迹与监督语义，不能视作同一训练基线。连接几何与来源一致，参考提交与校验值见 [reference_sources.json](../reference_sources.json)。

实际棋盘与原始对局始终使用固定坐标，动作仍是 `canvas` 上的零起始行优先编号；外部 CLI/网页用实际尺寸编号，并显式转换。规则与视角编码见 [schema.py](../python/etazero/schema.py)、[Game](../cpp/include/etazero/game.h)。

## 网络坐标与对称

空间输入沿用有效棋盘、己方、对方和两个禁手平面；Hex 的禁手平面为零。全局契约添加 Hex 标识和 Hex 白方行棋标识，与 Freestyle 区分，既有 PDA 条件继续使用真实执色符号。输入布局唯一维护在 [schema.py](../python/etazero/schema.py)。

网络边界按 KataGomo Hex2024 将白方空间输入转置，使根当前玩家的目标边统一朝上下；policy 和所有空间目标同步转换，输出恢复为实际坐标。WDL 始终是当前玩家 W/D/L，不因空间变换换序。原始轨迹和回放存储实际坐标，learner 在前向前转换；关闭随机增强仍执行白方规范化。

Hex 对称组只有 identity 和 180°旋转，推理配置 `nn_symmetry` 使用组索引 0/1，根集成数量最多为 2。Gomoku 保留 D4。训练/验证的 batch 朝向随机抽样中，各 Hex 行只采用两种有效旋转，再执行白方转置；混合棋类 batch 的 Gomoku 行仍采用 D4。实现见 [symmetry.py](../python/etazero/symmetry.py)、[C++ 请求映射](../cpp/include/etazero/symmetry.h)。

MuZero 树和 K 步训练展开使用根观测确定的同一坐标系，全部 K 个动作与 K+1 组 policy/Q 目标按根映射转换；树内不随交替执色重新转置。latent 搜索不调用真实连接判定；真实终局和监督仍由实际 Game 给出。

## 平衡开局

Hex 使用独立 `[hex_opening]`，字段定义与范围以 [config.py](../python/etazero/config.py) 为准。`selfplay.cfg` 配置自对弈；`match.cfg` 配置比赛，网页主动生成平衡开局复用该比赛 profile。`analysis.cfg` 直接分析给定局面，不自动生成开局。比赛开局环境覆盖使用 `MATCH_HEX_OPENING_<字段大写>`，独立于五子棋的 `[opening]` 和自对弈配置。

按 `probability` 进入初始化分支，再按 `make_fair_probability` 决定是否执行首手拒绝采样；两个门均可保持空盘，此时记录为未尝试，平衡手数与尝试数均为零。

执行采样时，在实际尺寸内均匀抽取黑方首手，复制棋盘落子后，用参考白方网络评估白方胜率 `p = W`。令 `b = 2*p-1`，以 `max((1-b*b)^balance_exponent, min_accept_rate)` 接受该手；拒绝后重新均匀抽点，持续至接受或取消。不得改用 `W-L`，预测 D 非零时两者不同。最低接受率必须大于零；数值异常明确失败，不切换到五子棋 fallback。实际尝试数、参考模型、接受后白方 W-L 诊断值与动作均保留。

自对弈和比赛的来源默认值分别见 [selfplay.cfg](../configs/azh/selfplay.cfg) 与 [match.cfg](../configs/azh/match.cfg)。比赛要求两个门均为 1，每个首手生成一次并供成对交换模型执色使用，开局参考模型随任务交替。普通自对弈黑白共用该轮模型。

平衡首手作为 `train_mask=0` 的可重放前缀保存，预算与采样行数为零。独立 `[policy_init]` 继续控制后续 policy 初始化；它可能改变平衡首手后的胜率。随机 bootstrap 不使用网络开局，已有 hint/fork 前缀跳过空盘平衡机制。

## 验证

[hex_test.cpp](../cpp/tests/hex_test.cpp) 覆盖六邻接、目标边、满盘唯一连接、桥形局面不提前终局、终局视角、白方 W 接受公式、双门、取消及模型坐标恢复。[test_hex.py](../tests/test_hex.py) 覆盖配置范围、独立比赛覆盖、混合棋类与 MuZero 根坐标，并提供真实 CUDA 小规模训练、恢复、分析、比赛和原生网页会话检查。工程闭环不代表棋力、收敛或等时间性能已验证。
