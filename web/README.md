# EtaZero 对弈室

在仓库根目录运行：

```bash
bash web/webui.sh
```

脚本使用 Conda `pytorch`，增量构建 EtaZero V0，然后在 <http://127.0.0.1:8766> 提供单用户对弈界面。默认读取 `EtaZero_V0/configs/minimal_test/eval.cfg` 的继承配置，扫描 `EtaZero_V0/data/minimal_test` 中已发布的历史模型，优先选中 `models/current.json` 指向的模型。点击“开始新局”加载权重。启动时生成模型列表，训练发布新模型后重启服务即可选择。

支持模型选择、模型画布范围内的棋盘尺寸、Freestyle / Standard / Renju、人类执黑或执白、设置每步根访问数、新局、悔棋、手数与对局记录。棋局由服务保存，刷新网页可继续；关闭服务后不保留。AI 执黑时自动下第一手，悔棋撤回到最近一次人类落子前，保留 AI 的开局首手。AI 搜索或加载期间操作锁定，失败显示错误并提供重试。

```bash
# 指定模型；必须是带同目录 manifest.json 的 EtaZero 已发布 TorchScript 模型。
bash web/webui.sh --model EtaZero_V0/data/minimal_test/models/<模型目录>/model.pt
# 扫描其他运行，或数据根目录下的全部运行。
bash web/webui.sh --models-dir EtaZero_V0/data --port 8766
# 评估配置独立于训练，支持现有 EVAL_* 环境覆盖。
EVAL_SEARCH_THREADS=8 EVAL_DEVICE=cuda:0 bash web/webui.sh
```

加载时检查模型输入契约和 SHA-256；不直接读取训练 checkpoint，使用训练已发布的推理权重（包括发布时选用的 SWA）。模型在 C++ 进程中常驻，同一模型新局复用进程，换模型在新局时生效。棋规和 PUCT 使用 V0 的原生实现；每步从新根搜索，根访问数包含一次初始根评估，最少为 2，其余搜索参数来自独立的评估配置。程序按指定设备运行，不自动回退到 CPU。

AI 上一手评估显示思考耗时、根访问数、AI 落子前视角的价值 W−L、搜索 W/D/L 和候选点。策略热力图显示同一个落子前棋盘上的网络先验、原始子节点访问分布或选择权重。网络先验为合法点上归一化的根网络策略（包含配置的全树 NN policy 温度，不含根温度和噪声）；访问分布由原始访问数归一化，选择权重来自评估配置的目标剪枝与 LCB。颜色分别按各图最大值线性缩放，数值为实际百分比。新局、悔棋和人类下一手会清除旧分析。

实现入口：[server.py](server.py)、[app.py](app.py)、[engine.py](engine.py)、[原生 serve 命令](../EtaZero_V0/cpp/src/commands/main.cpp)、[界面](static/)。启动参数以 `python -m web.server --help` 为准。

真实 CUDA 模型检查，在仓库根目录运行：

```bash
PYTHONPATH=EtaZero_V0/python:. \
ETAZERO_WEB_TEST_MODEL="$PWD/EtaZero_V0/data/minimal_test/models/<模型目录>/model.pt" \
conda run --no-capture-output -n pytorch python -m unittest discover -s web/tests -p test_web.py -v
```

浏览器检查需要已有 Playwright 和 Chromium，会新建并完成对局，应对独立测试服务执行：

```bash
ETAZERO_WEB_TEST_URL=http://127.0.0.1:8767 \
conda run --no-capture-output -n pytorch python -m unittest discover -s web/tests -p test_browser.py -v
```
