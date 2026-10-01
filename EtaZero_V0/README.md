# EtaZero V0

EtaZero 面向 Freestyle、Standard、Renju 五子棋。当前实现 AlphaZero + PUCT 的完整逐轮训练链路：多局自对弈、共享 GPU 批量推理、完整对局 NPZ、窗口 shuffle、固定训练量、checkpoint、模型校验发布与阶段恢复。并行与主要数据管线参考 KataGo，算法语义见 [算法说明](docs/algorithms.md)。

MuZero 和 Gumbel 属于后续阶段，选择它们会在启动 worker 前报错。当前提供单卡 learner，以及按设备列表启动的独立 selfplay worker；多卡 DDP、跨机器调度和常驻异步 learner 不属于当前能力。

## 构建与小规模验证

以下命令在本版本目录执行，使用已有 Conda `pytorch` 环境。构建依赖 C++17、CMake、zlib 和该环境的 LibTorch；脚本不读取其他研究仓库的代码。

```bash
bash scripts/build.sh
bash scripts/run.sh check-config --config-dir configs/minimal_test
bash scripts/run.sh run --config-dir configs/minimal_test --run-dir runs/my_check --cycles 2 --plot
```

`minimal_test` 是工程验收配置，`baseline` 是完整配置的起点；正式训练参数与预算由用户确定。构建默认 CUDA 架构为 RTX 5090 的 12.0，其他目标可通过 `TORCH_CUDA_ARCH_LIST` 指定。设备由配置明确指定，不自动回退到 CPU；在隐藏 GPU 的托管沙箱中，CUDA 命令须在宿主设备可访问的执行环境运行。

## 续训与评估

```bash
# --cycles 为该运行希望完成的累计轮数，已完成轮次不会重复训练。
bash scripts/run.sh run --config-dir configs/minimal_test --run-dir runs/my_check --resume --cycles 3

# 默认评估已发布模型，动作使用 canvas 上的行优先编号。
bash scripts/run.sh evaluate --config-dir configs/minimal_test --run-dir runs/my_check --size 5 --rule renju --moves 0,6

# model-b 指向另一份已校验发布的 model.pt，比赛交替先后手。
bash scripts/run.sh match --config-dir configs/minimal_test --run-dir runs/my_check --model-b /absolute/path/to/models/model_id/model.pt --size 5 --rule freestyle --games 4

bash scripts/run.sh plot --run-dir runs/my_check
```

新运行必须使用空目录。`--weights /absolute/path/to/checkpoint.pt` 为新运行导入兼容网络的模型状态，重新初始化优化器、计数和随机流；`--resume` 恢复完整训练状态，两者不能合用。恢复要求生效配置一致，改变实验条件时按用户的新运行或续训方案处理。

## 文档路由

| 任务 | 阅读入口 |
|---|---|
| 查看对局线程、搜索线程、共享组批及 C++ / Python 分工 | [执行架构](docs/implementation.md#执行架构) |
| 理解原始对局、两阶段 shuffle 和训练预取 | [数据链路](docs/implementation.md#数据链路) |
| 查看 replay ratio、累计缺口及 selfplay 局数 | [产样规划](docs/implementation.md#固定训练量与自对弈产量) |
| 调整配置、继承与本机覆盖 | [配置组织](docs/implementation.md#配置组织) |
| 查看 checkpoint、模型发布、故障恢复与运行证据 | [发布与恢复](docs/implementation.md#模型发布与恢复) |
| 理解棋规、网络、价值视角、PUCT 和训练目标 | [AlphaZero](docs/algorithms.md#第一轮-alphazero) |
| 查看未来算法的组合约束和接入边界 | [算法组合](docs/algorithms.md#算法与搜索组合)、[后续接入](docs/algorithms.md#后续算法接入) |
| 核对代码来源和许可证 | [参考来源](THIRD_PARTY.md)、[来源校验值](reference_sources.json) |
| 运行检查和了解验证限制 | [验收与限制](docs/implementation.md#验收与限制) |

本 README 是唯一总入口。工程与数据行为维护在 `docs/implementation.md`，算法语义维护在 `docs/algorithms.md`；字段以 [config.py](python/etazero/config.py) 的归属与校验、[baseline](configs/baseline/) 的完整参数为事实源。
