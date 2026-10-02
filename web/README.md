# EtaZero 开发工作台

在仓库根目录运行：

```bash
bash web/webui.sh
```

脚本使用 Conda `pytorch`，增量构建 EtaZero V0，在 <http://127.0.0.1:8766> 提供单用户工作台。默认读取 `EtaZero_V0/configs/minimal_test/eval.cfg` 的继承配置，扫描 `EtaZero_V0/data/minimal_test` 中已发布的模型，优先选择 `models/current.json` 指向的权重。点击“创建棋局”加载模型；“刷新模型”重新扫描发布目录，无需重启服务，不改变当前棋局或已加载模型。

## 会话与局面研究

桌面采用配置、棋盘、分析三栏。支持明暗主题、手数显示、键盘操作和窄屏布局；主题与手数偏好保存在浏览器。配置区支持按运行名、轮次或模型 ID 筛选模型，并查看 manifest、权重路径、SWA 信息与生效搜索配置。

- **人机对弈**：选择执子方，AI 自动应手；人类执白时 AI 先行。“撤回”退至最近一次人类落子前，保留 AI 开局首手。
- **局面研究**：手动交替落黑白棋，“分析”搜索当前方但不落子，“AI 单步”仅让当前方落一手。“撤回”只退一手。仍执行真实棋规，包括 Renju 黑棋禁手判负。
- **会话配置**：模型、尺寸和棋规需重开棋局生效；模式、执子和预算可点击“应用模式 / 执子 / 预算”更新而不重开。未应用的改动会显示提示。切换模式或执子不会自动落子；若轮到 AI，可点击“重试 AI”或“AI 单步”。
- **历史回看**：点击落子记录、拖动滑块或使用左右键。回看为只读，不改变服务端棋局。“当前”返回最新局面；“从此处继续”撤回后续落子、清除旧分析并进入手动研究模式，不保留被撤回的变化。

快捷键：`A` 分析，`S` AI 单步，`U` 撤回，`←` / `→` 回看，`Home` 初始局面，`End` 当前局面。编辑表单或操作按钮时不触发全局快捷键。

棋局由服务保存，刷新网页可继续；关闭服务后不保留。多个标签页共享同一个棋局，通过状态版本拒绝过期操作。搜索 / 加载期间锁定修改，仍可回看历史；请求失败显示错误，连接中断后自动重连。模型进程退出或超时后重开棋局重新加载。

## 搜索分析

分析面板明确显示搜索前的手数与执子方，区分当前局面和落子后保留的历史分析。“定位局面”将主棋盘切到该搜索对应的局面。只保留最近一次分析，手动落子、撤回或重开清除它。

- 显示实际根访问数、搜索耗时、根访问 / 秒、搜索 W−L、搜索与网络 W/D/L，以及 NN 请求数、推理批次数和平均批量。所有价值均为标注的搜索方视角，W−L 不等同于胜率。
- 候选表包含全部合法点的网络先验、原始访问数、访问占比、选择权重；支持点击表头排序、仅看已访问点、点击点位在棋盘定位。星号表示本次搜索选出的建议点，不保证是先验或选择权重最大值（落子含配置温度）。
- 策略对照显示同一搜索前局面的网络先验与访问分布 / 选择权重；主棋盘也可叠加这些数值。棋盘与分析不匹配时不叠加，并显示提示，避免将旧分布画在新局面上。
- 网络先验在合法点归一化，包含全树 NN policy 温度，不含根温度 / 噪声；原始访问分布由 child visits 归一化，选择权重来自目标剪枝与 LCB。各图颜色分别按最大值线性缩放，数值为实际百分比。
- “原始数据”提供最近一次分析的只读 JSON，便于核对原生返回值。

模型在 C++ 进程中常驻，同一模型重开复用进程。加载时检查输入契约和 SHA-256；只读取已发布的 TorchScript 推理模型（包括发布时选用的 SWA），不读取训练 checkpoint。棋规与 PUCT 使用 V0 原生实现，每次搜索从新根开始；根访问上限最少为 2，包含一次初始根评估，并受评估配置其他停止条件约束。程序按指定设备运行，不自动回退到 CPU。

```bash
# 指定模型：同目录必须包含 manifest.json。
bash web/webui.sh --model EtaZero_V0/data/minimal_test/models/<模型目录>/model.pt
# 扫描其他运行，或数据根目录下的全部运行。
bash web/webui.sh --models-dir EtaZero_V0/data --port 8766
# 独立评估配置，支持 EVAL_* 环境覆盖。
EVAL_SEARCH_THREADS=8 EVAL_DEVICE=cuda:0 bash web/webui.sh
```

实现入口：[server.py](server.py)、[app.py](app.py)、[engine.py](engine.py)、[原生 serve 命令](../EtaZero_V0/cpp/src/commands/main.cpp)、[界面](static/)。启动参数以 `python -m web.server --help` 为准。

## 验证

真实 CUDA 模型检查，在仓库根目录执行：

```bash
PYTHONPATH=EtaZero_V0/python:. \
ETAZERO_WEB_TEST_MODEL="$PWD/EtaZero_V0/data/minimal_test/models/<模型目录>/model.pt" \
conda run --no-capture-output -n pytorch python -m unittest discover -s web/tests -p test_web.py -v
```

浏览器检查需要已有 Playwright 和 Chromium，会重开、回退并完成棋局，应对独立测试服务执行：

```bash
bash web/webui.sh --port 8767
ETAZERO_WEB_TEST_URL=http://127.0.0.1:8767 \
conda run --no-capture-output -n pytorch python -m unittest discover -s web/tests -p test_browser.py -v
```
