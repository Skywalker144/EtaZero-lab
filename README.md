# EtaZero

**EtaZero** 是一个集成了 **AlphaZero**、**Gumbel AlphaZero**、**MuZero** 与 **Gumbel MuZero** 的棋类强化学习框架，深度结合了 **KataGo** 的全部核心算法技巧（tricks），并在底层并行框架与整体训练管线上全面对齐 KataGo 的工业级架构设计。

## 核心特性

- **统一的多算法框架**
  - **AlphaZero**：基于真实棋规与精确状态转移的经典 PUCT 搜索与训练。
  - **Gumbel AlphaZero**：结合 Gumbel-Top-k 采样与 Sequential Halving，支持 Gumbel 根搜索与 Gumbel / PUCT 非根选点，在低模拟预算下依然保证策略提升。
  - **MuZero**：基于隐状态（Latent Space）表征、动态转移（Dynamics）与预测（Prediction）网络的规划与多步展开（Unroll）训练。
  - **Gumbel MuZero**：将 Gumbel 搜索机制与 MuZero 隐空间规划无缝结合。

- **全面集成 KataGo 算法 Tricks**
  - **网络架构与特征**：完整支持 KataGo 风格手写网络预设、全局池化（Global Pooling）与多维度棋盘特征编码。
  - **搜索增强**：集成根节点温度采样、Playout Cap Randomization（机动模拟次数上限）、First Play Urgency (FPU)、Subtree Value Bias（子树价值偏差）、根节点先验噪声与策略剪枝等全套搜索机制。
  - **多目标与辅助训练**：支持短中长期价值预测、辅助策略目标、棋盘所有权/局势辅助头及独立 Policy Init 等训练加速技术。
  - **开局与对称性**：内置平衡开局采样（Balanced Opening）与棋盘几何对称性数据增强。

- **对齐 KataGo 的高性能并行框架与训练管线**
  - **高吞吐 C++ 自对弈引擎**：采用多线程异步对局、树内并行搜索（Virtual Loss）、跨对局共享神经网推理队列、动态组批（Dynamic Batching）与 LibTorch 原生 **CUDA Graph** 加速推理。
  - **工业级端到端训练管线**：完整对齐 KataGo 的 `Self-Play → 两阶段 Shuffle → Learner 训练 → 模型原子发布与门控` 闭环管线，支持按 Replay Ratio 动态规划产样额度、高效预取、断点无损续训与等时间 Elo 自动竞技评估。

## 目录导航

本项目采用多版本演进目录结构，最新主线版本及详细文档请参考：

- **最新主线版本**：[EtaZero_V3](EtaZero_V3/README.md)（包含完整的构建、配置、训练、评估与算法文档路由）
- **Web 交互与分析工作台**：[web](web/README.md)（支持人机对弈、局面研究与搜索树诊断）
- **项目开发与研究规范**：[RULES.md](RULES.md) | [AGENTS.md](AGENTS.md)
