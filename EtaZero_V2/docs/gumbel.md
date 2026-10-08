# Gumbel AlphaZero / MuZero

`agent.algorithm` 仍选择真实规则 AlphaZero 或 latent MuZero。`root_search_algo=gumbel` 使用 Gumbel 根搜索；`nonroot_search_algo=puct` 复用对应模型的现有 PUCT/FPU，`gumbel` 使用论文式 14 的确定性选点。PUCT 根配 Gumbel 非根仍是不允许的项目组合。

四个起始配置分别继承对应的现有配置：

| 配置 | 模型与规则 |
|---|---|
| [baseline_gumbel](../configs/baseline_gumbel/run.cfg) | AlphaZero / 五子棋 |
| [muzero_gumbel](../configs/muzero_gumbel/run.cfg) | MuZero / 五子棋 |
| [baseline_gumbel_hex](../configs/baseline_gumbel_hex/run.cfg) | AlphaZero / Hex |
| [muzero_gumbel_hex](../configs/muzero_gumbel_hex/run.cfg) | MuZero / Hex |

默认根和非根均使用 Gumbel。将 `run.cfg` 的 `agent.nonroot_search_algo` 改为 `puct`，即可复用现有非根 PUCT；评估另在 `engine.cfg` 修改同名字段。在 V2 目录运行 `CONFIG_DIR=configs/baseline_gumbel bash scripts/run.sh`，其他组合替换目录即可。

## 根搜索、预算与落子

算法参考 [Policy improvement by planning with Gumbel](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf) 和作者团队 [Mctx 428bfb7](https://github.com/google-deepmind/mctx/tree/428bfb7e1c931715cd3d74f9c65e9991afd86df8)。固定路径、哈希和 Apache-2.0 许可见 [参考来源](../THIRD_PARTY.md)。实现入口是 [公共数学与调度](../cpp/src/search/gumbel.cpp)、[真实状态搜索](../cpp/src/search/gumbel_search.cpp) 和 [latent 搜索](../cpp/src/muzero/gumbel_search.cpp)。

每手从干净合法域先验 P 生成 log(P)，采样一份根 Gumbel 噪声。首轮按 `log(P)+g` 无放回选择候选，后续按 `log(P)+g+sigma(Q)` 分配访问。使用 Mctx 的 considered-visit 序列实现 Sequential Halving；最多候选数由配置限制，合法动作不足时缩小。没有强制要求预算足够完成全部阶段，余数全部发放；预算不足时从完成访问数最多的动作中按最终分数选胜者。并列取合法域中的第一个动作。

`visits` 保留 V2 口径：包括一次初始根预测，实际根边模拟预算为剩余额度。`max_playouts`、时间限制、显式停止、PCR、Reduce Visits 和 PDA 沿用已有预算与产样机制；返回的计数只包含完成工作。显式停止或零 playout 可以在初始预测前返回 action=-1；只有初始根预测时仍可按先验和根噪声选择动作。真实 AZ 终局直接返回精确结果；MuZero 树内继续仅使用 latent 与有效棋盘动作域。

`[gumbel]` 属于 `selfplay.cfg`，字段及缺省值唯一维护于 [config.py](../python/etazero/config.py)。`max_num_considered_actions` 限制候选数量；`c_visit` / `c_scale` 控制 Q 的访问数缩放；`rescale_q_values` 选择节点内 completed-Q min/max 缩放；`noise_scale` 控制根噪声强度。

`action_selection=gumbel` 使用 SH 胜者，不再对改进策略或访问数抽样；已有落子温度不改变此行为。`visit` 显式选择论文棋类实验中的访问次数温度抽样，使用已有 early/late 温度调度；策略训练目标仍是 completed-Q 改进策略。analysis/match 在自身 profile 配置根/非根策略和 `gumbel_*` 字段，噪声缺省零。cheap 搜索的 remove-root-noise 关闭 Gumbel 噪声，保持当前非根策略。

## Completed Q、非根策略与训练

所有 Q 均在父玩家视角。以已访问动作的干净先验加权 Q 均值，按总完成访问数与初始 NN value 混合得到 v_mix；没有已访问动作时使用初始 NN value。未访问动作以 v_mix 补齐。可选节点内 min/max 归一化后，乘 `(c_visit+max(N))*c_scale`；归一化的跨度下限为 1e-8。

根策略目标是 `softmax(log(P)+sigma(completedQ))`，不包含 Gumbel 噪声，不要求等于访问分布。非根 Gumbel 按 `argmax(pi'-N/(1+sum(N)))` 选点，不在树内加噪声。非根 PUCT 保留原有探索、FPU 和 virtual loss。根 hint 沿用先验向提示动作转移 2% 质量的约定，不叠加 PUCT 根强制配额。

Gumbel 根不执行 Dirichlet、根 policy 温度、forced playout、PUCT target pruning、LCB 或 chosen-move 剪枝。全树 NN policy 温度、optimistic logits 与现有子树价值聚合仍按配置生效，属于 EtaZero 的算法适配。v_mix 仅用于 Q completion，不替换现有 search_wdl；终局、TD 和辅助 Q 监督沿用原有定义，未访问动作不伪造 Q 访问监督。

网络、损失、优化器和 learner 图保持现有 AZ/MZ 定义。固定目标的 policy CE 与论文 policy KL 梯度相同，未额外引入 KL 计算；EtaZero 的辅助 heads 和 WDL 目标不表述为论文整套训练复现。writer 仍保存已有 int16 policy 数组，Gumbel 单独将最大目标映射到 30000，以降低稠密策略尾部的量化误差。极小目标仍可能舍入为零。主轨迹、side、opponent、reanalysis 和 MuZero K+1 真实步使用同一种目标。

原始分片的 Gumbel metadata 记录根/非根策略。数据校验允许未访问合法动作有正目标，同时保留占用点与 padding 为零的边界；原有 PUCT 的校验和打包规则保留。shuffle、预取、增强、checkpoint、发布和恢复消费既有张量布局，无新增每行搜索模式张量。

## 执行与适用边界

搜索工厂只在创建对象时选择策略。Gumbel 使用独立搜索对象，共用已有状态转移、节点展开/回传、推理服务与数据管线；原有 PUCT 节点布局和 learner 不包含 Gumbel 状态。AZ 遍历的编译期特化使 PUCT 实例不执行 Gumbel 选点判断，惰性 child 分配保留。

多线程根调度按完成数与在途预约发放配额，在 considered-visit 层次切换前等待在途任务完成，再使用完成 Q 淘汰。非根 Gumbel 将单位在途预约计入访问缺口，Q completion 和缩放只使用完成数；这是并行适配，单线程退化为论文规则。同步只作用于该局，不阻塞其他对局的共享 GPU 服务。异常释放全部路径和根预约并向调用者传播。

Gumbel 当前要求 `reuse_tree=false`、`use_graph_search=false`，非法配置在启动时明确拒绝。这是 fresh-tree Gumbel 的实现边界，不宣称图共享或跨手复用数学上不可行。MZ 继续要求单一根对称；AZ 可沿用合法的根对称集成。数值使用 double 搜索统计和 V2 已归一化先验，零先验取 double 最小正规数进行 log 保护；不承诺与 JAX 随机数或极端 logits 下逐位一致。

根调度和 Gumbel 数学验证在 [gumbel_test.cpp](../cpp/tests/gumbel_test.cpp)，数据与 CUDA 链路在 [test_gumbel.py](../tests/test_gumbel.py)。工程验收与性能回归不能证明棋力收益；不同非根策略、预算及增强项的实验选择由用户安排。

[Mctx 数值对照](../tests/reference/check_mctx_gumbel.py) 对登记的固定源码执行原始 Q transform、根选点与 considered-visit 函数，以 NumPy/SciPy 提供数组运算，比较 C++ 的访问数、胜者和策略目标；无需安装 JAX。构建后运行 `conda run -n pytorch python tests/reference/check_mctx_gumbel.py --source /path/to/mctx`，源码检出版本必须与登记的哈希一致。
