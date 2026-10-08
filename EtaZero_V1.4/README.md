# EtaZero V1.4

EtaZero 面向 Freestyle、Standard、Renju 五子棋及 Hex。支持 AlphaZero / PUCT 与 MuZero / PUCT。AlphaZero 使用真实棋规搜索，MuZero 使用独立 latent 搜索与展开训练；Gumbel 组合尚未实现，配置检查明确拒绝。

训练按完整轮次执行自对弈 → shuffle → learner → 模型发布。各阶段内部支持并行对局、GPU 组批与 CUDA Graph 推理、数据处理和预取；当前为单卡 learner，无 DDP、跨机器调度或跨阶段异步流水线。算法、数据与恢复边界由下方文档分别维护，工程检查不代表棋力或等时间性能收益已验证。

## 构建与小规模验证

以下命令在本版本目录执行，使用已有 Conda `pytorch` 环境。构建依赖 C++17、CMake、zlib、OpenSSL Crypto 和该环境的 LibTorch。

```bash
bash scripts/build.sh
bash scripts/run.sh check-config --config-dir configs/smoke_test
CONFIG_DIR=configs/smoke_test bash scripts/run.sh --run-dir data/my_check --iterations 2
```

`baseline` 提供五子棋起始配置，`baseline_hex` 提供 Hex 起始配置（参见 [Hex](docs/hex.md)），`smoke_test` 用于快速工程验收；测试本身使用独立的 `tests/fixtures/configs/`，不依赖用户运行配置的取值。设备由配置指定，不自动回退到 CPU。隐藏 GPU 的托管沙箱须在可访问宿主 CUDA 的执行环境运行 CUDA 检查。

## 配置与训练

```bash
CONFIG_DIR=configs/baseline bash scripts/run.sh
```

`CONFIG_DIR` 选择配置目录，显式 `--config-dir` 优先。不带子命令时直接训练，也支持 `run`、`check-config`、`analysis`、`match`、`arena` 和 `plot`。配置继承、文件归属、本机覆盖与输出路径见 [配置组织](docs/implementation.md#配置组织)；具体参数以配置文件和解析器为准。

bootstrap 为 iteration 0；iteration 1 完成首轮训练发布，后续每轮使用开始时已发布的模型产样，再训练并发布下一代。新增数据量按基准训练量与 replay ratio 规划，实际训练量受新增数据额度和快照单遍限制；零步轮次保留当前模型。详见 [产样规划](docs/implementation.md#训练额度与自对弈产量)。

MuZero 的网络、展开训练与搜索约束见 [MuZero](docs/muzero.md)。

## 续训与评估

输出目录已有 `.internal/run.json` 时自动恢复，无需 `--resume`。`--iterations` 指累计完成目标，提高目标可以追加训练。控制器复用已完成阶段，自对弈保留完整对局，learner 从已保存的消费游标继续下一批；`--weights` 仅用于新运行的权重初始化。恢复状态与校验范围见 [发布与恢复](docs/implementation.md#模型发布与恢复)。

```bash
CONFIG_DIR=configs/smoke_test bash scripts/run.sh --run-dir data/my_check --iterations 3
bash scripts/run.sh analysis --config-dir configs/smoke_test --run-dir data/my_check --size 5 --rule renju --moves 0,6
bash scripts/run.sh match --config-dir configs/smoke_test --run-dir data/my_check --model-b /path/to/model.pt --size 5 --rule freestyle --games 4 --output data/my_match
bash scripts/run.sh plot --run-dir data/my_check
```

评估显式选择已发布模型；动作使用实际棋盘的零起始行优先编号。比赛按成对开局交换模型执色，支持逐局续测。配置、预算与 Elo 统计见 [评估与 Elo](docs/elo.md)。

## 训练图与自动实验

运行目录包含配置、日志、原始对局、快照、checkpoint 和已发布模型。`training.png` 展示训练概览，`loss.png` 对照训练与验证损失，`logs/performance.png` 展示执行统计；指标定义、缺测和恢复筛选见 [训练图与性能图](docs/implementation.md#训练图与性能图)。

```bash
# 工程验收示例；正式实验配置由用户维护。
CONFIG_DIR=configs/autoexp_example bash scripts/autoexp.sh --dry-run
CONFIG_DIR=configs/autoexp_example bash scripts/autoexp.sh
```

调度器发现伞目录中的配置臂，按预算和 GPU 槽位执行，支持恢复；具体行为见 [自动实验](docs/implementation.md#自动实验)。

## 等时间 Elo

训练预算和 Elo 横轴使用已提交完整轮次扣除编译与绘图后的累计墙钟；可正常结算的中断尝试计入续训轮次，暂停和启动恢复不进入该时间。预算在轮边界检查，可能超过目标。历史缺少编译计时的结果不能精确换算到这一口径。

```bash
bash scripts/run.sh arena --data data/my_experiment --output data/my_elo --dry-run
bash scripts/run.sh arena --data data/my_experiment --output data/my_elo
bash scripts/run.sh arena --data data/my_experiment --output data/my_elo --fit-only
```

计时定义见 [运行观测与证据](docs/implementation.md#运行观测与证据)，模型选择、比赛和拟合见 [评估与 Elo](docs/elo.md)。

## Web 开发工作台

在仓库根目录运行 `bash web/webui.sh`。工作台选择最新主线版本，支持已发布模型选择、人机对弈、手动局面研究、历史回看和搜索诊断，用法见 [Web 文档](../web/README.md)。

## 文档路由

| 任务 | 阅读入口 |
|---|---|
| 查看对局线程、搜索线程、共享组批及 C++ / Python 分工 | [执行架构](docs/implementation.md#执行架构) |
| 测量并调整 selfplay 并行局数、推理 batch 和服务数 | [并行参数短测](docs/implementation.md#selfplay-并行参数短测) |
| 理解原始对局、两阶段 shuffle 和训练预取 | [数据链路](docs/implementation.md#数据链路) |
| 查看 replay ratio、训练额度及 selfplay 局数 | [产样规划](docs/implementation.md#训练额度与自对弈产量) |
| 接入 Hex、理解方向编码、有效对称与首手平衡采样 | [Hex](docs/hex.md) |
| 调整平衡开局及独立 policy init | [开局机制](docs/algorithms.md#平衡开局与-policy-init) |
| 查看实验产物和内部状态位置 | [运行目录](docs/implementation.md#运行目录) |
| 调整配置、继承与本机覆盖 | [配置组织](docs/implementation.md#配置组织) |
| 固定评估预算、成对比赛、续测与 Elo 图 | [评估与 Elo](docs/elo.md) |
| 查看 AlphaZero / KataGo 的局长变化、规模规律与架构对照经验 | [训练经验](../docs/training-experience.md) |
| 网页对弈、手动局面研究与搜索诊断 | [Web 开发工作台](../web/README.md) |
| 查看 checkpoint、模型发布、故障恢复与运行证据 | [发布与恢复](docs/implementation.md#模型发布与恢复) |
| 理解棋规、网络、价值视角、PUCT 和训练目标 | [AlphaZero](docs/algorithms.md#alphazero) |
| 查看算法组合、公共边界和后续 Gumbel 设计 | [算法组合](docs/algorithms.md#算法与搜索组合)、[后续接入](docs/algorithms.md#算法接口与后续设计) |
| 运行 MuZero、配置展开训练与检查 KataGo 机制适用边界 | [MuZero](docs/muzero.md) |
| 查看 V0 历史对齐范围、网络映射和验收证据 | [V0 核查记录](../EtaZero.md#alignment-targets)、[V0 实施记录](../plan.md) |
| 追溯 V0 工程对齐与分批验收 | [V0 工程核查](../EtaZero.md#engineering-audit)、[V0 验收记录](../plan.md) |
| 核对代码来源和许可证 | [参考来源](THIRD_PARTY.md)、[来源校验值](reference_sources.json) |
| 训练图、实验伞目录与 GPU 排队 | [绘图与自动实验](docs/implementation.md#训练图与性能图)、[调度](docs/implementation.md#自动实验) |
| 运行检查和了解验证限制 | [验收与限制](docs/implementation.md#验收与限制) |


本 README 是版本文档总入口。算法语义维护在 `docs/algorithms.md`，工程与数据行为维护在 `docs/implementation.md`；字段归属和校验以 [config.py](python/etazero/config.py) 为准。
