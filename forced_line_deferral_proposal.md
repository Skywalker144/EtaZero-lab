# 方案：在强制线上推迟网络评估（纵向抽象的第一步）

> 一句话：**网络在一个节点上顺便预测"接下来几手大家大概率怎么走"；搜索沿这条高置信线走下去时先不调用网络，只在真正需要分叉的地方才评估。**

工作名：**DeferZero**（叫法随你定）。核心组件只有两个：**多步策略头** + **延迟评估节点**。

---

## 1. 直观动机

### 1.1 AZ 的浪费在哪
- PUCT 每次 simulation 只扩展 **1 个叶子 = 1 次 NN 调用**。
- 一条 l 手的强制线（冲四→挡、活三→挡……），哪怕每一步 prior 都 ≈ 1，也要 **l 次 simulation、l 次 NN 调用**才能走到线的尽头，才看得到"这条线的结果"。
- 这些中间评估的信息量很低：prior 早就知道该走哪，value 也几乎不变。这就是你最初说的"有些 action 信息量很低"。

### 1.2 为什么能提升等墙钟和极限棋力
- 每次 simulation 的成本基本不变：仍然是 1 次 NN 调用，GPU batch 吞吐也不变。但叶子更深，**NN 调用集中到真正的决策点上**。
- 如果搜索里有比例 f 的 NN 调用落在"可推迟"的位置，等墙钟下决策点得到的评估量大约放大到 **1/(1−f)** 倍。比如 f=30%，大约相当于 1.4 倍 playouts。
- 自对弈的 policy target 质量随之提升，这会进一步影响极限棋力。
- 多步策略头只是给 policy head 多加几个输出通道，额外成本可以忽略。

### 1.3 和 "latent action" 的关系
"沿 prior 走到下一个决策点"本身就是一个**隐式的宏动作**：
- 不需要码本，也不需要单独解码；
- 长度自适应：走到 prior 不再确定的地方就停；
- 对手的着法也由网络给出（见 §3.1）。

以后想做显式 latent / jumpy dynamics，可以在这个框架上自然推广（§5）。

---

## 2. OptionZero 做了什么，为什么不直接照搬

**OptionZero（Huang et al., ICLR 2025）的关键做法**（根据论文原文）：

1. **Dominant option**：取累计概率 ∏P(aᵢ) > 0.5 的最长 argmax 序列，最长 L（Atari 用 3 或 6）。
2. **Option head**：输出 L 个分布 ω₁..ω_L，表示累计概率，外加一个虚拟的 `stop` 动作。
3. **Dynamics**：输入"复合动作"，一次跳到 s_{k+l}，同时预测累计 reward。
4. **树结构**：在 primitive 边之外额外加一条 option 边；option 路径上的中间节点被展开但不评估。
   - primitive 边的统计包含 option 边的统计；
   - 选择时用调整过的 P̃ = max(0, P(a)−P(o)) 和 Q̃ 去掉 option 的贡献；
   - backup 要更新所有可能路径上的边。
5. **执行**：option 可以在环境里整段执行，降低决策频率。实验在 Atari 和 GridWorld 上。
6. 论文 Discussion 里把双人博弈和动态最大长度列为未来工作。

**直接照搬到五子棋会遇到的问题**：

| 问题 | 说明 |
|---|---|
| 对抗下的乐观偏差 | option 边的 Q 只来自线的终点，**绕过了中间对手节点的选择**。如果对手有反驳，option 的 Q 会偏乐观；单人 Atari 没有这个问题。 |
| 统计记账复杂 | 双边统计、Q̃ 调整、多路径 backup，和 EtaZero 的 KataGo 式图搜索 + `recompute` 聚合很难干净地合在一起。 |
| 不能整段执行 | 双人博弈里对手不会配合，根上只能执行第一手。所以收益只来自**搜索效率**，不来自决策频率。 |
| 只有 argmax 加 stop | 中间节点没有完整的 prior，想细化时只能重新评估。 |

---

## 3. 方法

### 3.1 网络：多步策略头
在 [`PolicyHead`](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/python/etazero/network.py#L166-L188) 里增加 L−1 个输出通道，在 s_t 上输出：

- p₁ = 普通 policy（已有）；
- pᵢ (i≥2) = 沿 **dominant 线**走 i−1 手后那个局面的 policy。双方交替，也就是**对手的着法同样由网络给出**。

> 现有的 `opponent_policy` head（KataGo 机制）本质上就是 p₂，只是没有"dominant"条件。**所以 L=2 的版本可以完全不改网络，直接拿它来做原型**（见 §6 的 Step 1）。

**训练目标**：pᵢ(·|s_t) 的目标是 π_{t+i−1}，即轨迹上那个位置的搜索 policy target。逐步加 mask：
- 中间任何一步实际落子 ≠ 该步搜索 policy 的 argmax 时，mask 掉这一步及之后所有步，保证"沿 dominant 线"的语义成立。在强制局面里实际落子几乎总是 argmax，所以有用的数据基本都保留下来。
- 对局在这之前结束，或者该步没有 policy target，也 mask 掉。
- 损失系数可以先参照 opponent policy 的 0.15，再随 i 递减。

### 3.2 搜索：三种节点状态

`EMPTY → PROVISIONAL → READY`

- **READY**：已经做过 NN 评估，和现在一样。
- **PROVISIONAL（暂定节点）**：没有做 NN 评估。
  - prior 取自最近的 READY 祖先（称为 anchor）的 p_i，按真实规则做合法性 mask 后重新归一化。
  - 没有自身 value，`initial_weight = 0`，value 完全由子节点聚合得到。

**推迟条件**：要把节点 s_i（anchor 下第 i 手）建成 PROVISIONAL，以下全部满足：
1. i < L；
2. 到达 s_i 的每一步都是对应 p_j 的 argmax，也就是一直在 dominant 线上；
3. max p_i ≥ τ，即 s_i 本身也"几乎是强制的"；
4. 真实规则判定 s_i 不是终局。线中途如果成五，直接用精确终局值，这正是五子棋 AZ 的优势。

**细化条件**：PROVISIONAL 节点在下面任一情况下做真正的 NN 评估，并升级为 READY：
- 访问数 ≥ n_refine（比如 2–4）；或
- PUCT 在这里选中了非 dominant 的着法。

升级时加入 NN 的 initial value，并用真实 prior 替换暂定 prior。

**这样就保住了"不丢关键反驳"**：每一手仍然是一条真实的树边，对手节点照常做 min 选择。推迟的只是评估，**没有跳过任何选择**。一条线被访问多了，就自动退化回普通 AZ。

一次 simulation 的流程如下：

```text
node = root (READY)
loop:
  if terminal(state):            value = exact;  break
  if node.EMPTY:
      if can_defer(node):        make_provisional(node)      # 不调用 NN，继续往下走
      else:                      nn_eval(node) → READY;  value = v;  break
  elif node.PROVISIONAL and should_refine(node):
                                 nn_eval(node) → READY;  value = v;  break
  a = PUCT(node)                 # 暂定节点用暂定 prior
  state.move(a); node = child
backup(path)                     # 完全不改：逐边按 perspective 乘 ±1
```

每次 simulation 仍然**恰好 1 次 NN 调用**（或者遇到终局），只是叶子更深。

### 3.3 你问的价值回传问题
- 在这个设计里，**每一手都是一条边**，回传逻辑一行都不用改。EtaZero 的 [`AlphaZeroState::move`](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/cpp/include/etazero/algorithm.h#L86-L90) 本来就是用 `parent==player ? +1 : -1` 来确定 perspective 的。
- 如果以后做 OptionZero 式的**单条跨 l 手的跳跃边**，你说的完全正确：**to_play 相同就不乘 −1，不同就乘 −1**。五子棋没有 pass，所以等价于 (−1)^l；线中途成五的情况要单独处理。EtaZero 的边 perspective 本来就是存成 ±1 的，直接适用。

### 3.4 图搜索（transposition）
暂定节点同样以 `graph_key` 放进 node table。如果后来有别的路径到达同一个局面，节点自然共享。所以**不需要 OptionZero 那种多路径 backup**，KataGo 式的 `recompute` 本来就能处理 DAG。

---

## 4. 在 EtaZero（V2.3）里改哪里

| 模块 | 文件 | 改动 |
|---|---|---|
| 数据 | [record.cpp L350-L363](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/cpp/src/selfplay/record.cpp#L350-L363) | 仿照 `opponent_policies`，写出 `future_policies[L-1]` 和逐步权重（按 §3.1 加 dominant mask）。 |
| schema / 读取 | [schema.py](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/python/etazero/schema.py)、[data.py](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/python/etazero/data.py) | 新数组的形状校验；D4 变换和 policy 同步做。 |
| 网络和损失 | [network.py](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/python/etazero/network.py#L166-L188)、[`base_losses`](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/python/etazero/network.py#L291-L315) | policy head 增加 L−1 个通道，加入带权 CE；导出时把它们一并输出。 |
| 推理接口 | [inference.h `Evaluation`](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/cpp/include/etazero/inference.h#L16-L23) 及 backend、cache | 增加 `future_logits`。每步建议**只存 top-k（比如 8 个）**，避免 225×L 的 double 撑爆节点内存。 |
| 搜索节点 | [search.h `Node`](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/cpp/include/etazero/search.h#L126-L153) | `Status` 增加 `PROVISIONAL`；记录 anchor 的 future prior 指针（共享）和 chain 深度 i。 |
| 展开和细化 | [search.cpp `expand`](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/cpp/src/search/search.cpp#L291-L311) | 返回三态：已就绪 / 已推迟（继续走）/ 刚评估（break）；细化时从 PROVISIONAL 升级到 READY。 |
| simulation | [search.cpp L331-L444](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/cpp/src/search/search.cpp#L331-L444) | L347 的 `if(expand(...)) break;` 改成上面的三态逻辑；backup（L435-L441）不改。 |
| 聚合 | [`recompute`](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/cpp/src/search/search.cpp#L283-L290) | 暂定节点 `initial_weight=0`；细化后恢复成 1。 |
| 诊断 | `SearchResult` | 增加统计：推迟节点数、细化数、平均 chain 长度、每次 simulation 的 NN 调用数、叶子深度。 |

**注意事项**：
- 根节点永远是 READY。
- Gumbel 的 interior 分支（`GumbelInterior`）第一版先关闭推迟。
- FPU 在暂定节点上用 anchor 的 value 做参考即可。
- 虚拟损失照常施加在边上。
- **τ > 1 时这套逻辑应该和原版 AZ 逐位一致**，这一条可以写成回归测试。

---

## 5. 第二阶段：MuZero / jumpy dynamics（可选）

- 暂定节点只有在被细化时才需要 latent。可以用一个 **jumpy dynamics** g(h_anchor, prefix) 一次算出任意前缀长度 i 的 h_i：一次 dynamics 调用就能覆盖 i 手。
- **五子棋特有的简化**：没有成五的前提下，一串落子之后的局面只取决于"双方各新增了哪些子"，与顺序无关。所以 prefix 可以编码成两张平面（己方新增、对方新增）加一个奇偶位，不需要序列模型。现有的 [Dynamics](file:///Users/sky/Desktop/DLRL/EtaZero-lab/EtaZero_V2.3/python/etazero/muzero/network.py#L121-L137) 用的是单动作 one-hot 平面，扩展起来很直接。
- 训练：在现有的 K 步展开数据上随机采样 j ≤ K，用 g(h_t, prefix_{t:t+j}) 去预测 t+j 位置的全部 heads。latent 的视角奇偶 = j mod 2。

---

## 6. 实验计划（先判断值不值得做）

| 步骤 | 内容 | 判定 |
|---|---|---|
| **Step 0：可行性测量，不用训练** | 拿现有的 15×15 AZ 模型跑自对弈搜索，统计非根 NN 调用中满足"父节点在 dominant 线上，且 max prior ≥ τ"的比例 f（τ = 0.8 / 0.9 / 0.95）；同时统计 opponent head 在这些位置的 top-1 准确率。 | f 就是可节省 NN 调用的上界。**如果 f < 10–15%，等墙钟收益有限，建议止步或者换思路。** |
| **Step 1：L=2 原型，不改网络** | 直接用 `opponent_policy` head 当 p₂，只改搜索；用固定网络做对局。 | 在相同墙钟、相同 visits 下比较对原版的 Elo；同时看 NN 调用数和叶子深度。 |
| **Step 2：L=4–8** | 加多步策略头，离线用现有数据训练；先测各个 pᵢ 在强制局面上的准确率，再接入搜索，用固定网络对局。 | 看 Elo 随 L、τ、n_refine 的变化。 |
| **Step 3：完整训练** | 从头自对弈训练，使用 EtaZero 现有的等时间 Elo 管线。 | **等墙钟 Elo 曲线**加最终棋力。 |

**诊断指标**：
- **反驳遗漏率**：在推迟版和高预算原版最终选择不同的局面里，有多大比例是推迟版错了；
- 暂定节点的细化比例；
- 平均 chain 长度随训练的变化。网络越强，强制线应该越长。

---

## 7. 风险与诚实评估

1. **f 可能不大**：开局和中盘真正"强制"的步可能不多。Step 0 就是为了尽早看清这一点。不过网络越强 prior 越尖，f 会随训练增加，这对**极限棋力**是利好。
2. **多步头的准确率随 i 下降**：p₂ 已有 KataGo 的经验支撑，p₆ 以后不一定准。好在强制应对通常是局部的、很明显的；判错了也只是让搜索顺序次优，叶子的 value 仍然来自真实棋盘上的 NN 评估，**不存在模型误差累积**（AZ 版本）。
3. **暂定 prior 带来的搜索偏差**：靠 τ 和 n_refine 控制；τ>1 就退化回原版，可以安全地做消融。
4. **新颖性**：思路上和 OptionZero 接近（OptionZero 的中间节点也是"展开不评估"）。可以主张的区别有：
   - 双人对抗下保留每一手的选择，避免乐观偏差；
   - 用"推迟评估 + 按访问细化"代替额外的 option 边和统计调整，更简单；
   - 多步完整 prior，而不只是 argmax 加 stop；
   - 规则版里线中途的终局是精确的；
   - 以等墙钟为目标。

   动手前建议再查一下 "lazy / deferred evaluation MCTS" 方向有没有直接撞车的工作。
5. **和 LLM 的类比**：多步策略头约等于 Medusa 式的多 token 预测头，推迟评估加细化约等于 speculative decoding 的"先起草、需要时再验证"。如果以后想往 LLM 推理搜索上走，这个框架的叙事可以直接迁移。
