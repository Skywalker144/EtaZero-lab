# 配置目录与启动入口

配置组织和启动脚本属于 EtaZero 外层。引擎继续使用 KataGo 的 `ConfigParser`，训练与 shuffle 继续使用来源程序的命令行解析器；外层不维护搜索参数别名、数值转换或另一份默认值表。

## 单臂目录

一个配置目录包含以下文件，也可在 `run.cfg` 中引用共享文件：

| 文件 | 内容与消费者 |
|---|---|
| `run.cfg` 的 `[run]` | 棋类、算法选择、产样额度、发布、绘图、预算和输出路径，由外层读取。 |
| `selfplay.cfg` | 原生 `key = value` 与 `@include`，完整交给原生 selfplay。 |
| `train.cfg` 的 `[train]` | `train.py` 的原生选项名，省略开头的 `-`。 |
| `shuffle.cfg` 的 `[shuffle]` | `shuffle.py` 的原生选项名，省略开头的 `-`。 |
| 可选 `gtp.cfg` | Web／GTP 的独立原生推理配置，不继承或改写 selfplay 搜索参数。 |
| 可选 MuZero JSON | 来源 `MuZeroConfig`，由 `train.cfg` 的 `muzero-config` 引用。 |

`run.cfg` 例如：

```ini
[run]
game = go
algorithm = alphazero
games_per_round = 8
iterations = 2
selfplay_config = selfplay.cfg
train_config = train.cfg
shuffle_config = shuffle.cfg
```

最后三个路径字段省略时使用上述文件名。外层路径字段相对声明它的配置文件解析，包含空格时用引号包围。`run_dir` 可指定相对本目录的输出路径；省略时，位于本版本 `configs/` 下的配置自动对应到 `data/` 下的同一相对路径。外部配置目录需要显式输出路径，避免不同项目写入同一目录。

`[run]` 还接受现有外层控制器的 `publish_weights`、`replay_ratio`、`bootstrap_min_rows`、`gatekeeper_config`、`gatekeeper_required_win_prop`、`auto_plot`，含义见 [调度与恢复](controller.md)。布尔值使用 `true`/`false`。`iterations` 和 `max_hours` 是累计预算，至少指定一项；CLI 中对应参数优先于文件值，恢复时可以增加预算。

`train.cfg` 使用来源选项拼写，例如：

```ini
[train]
model-kind = b2c16-fson-mish
pos-len = 7
batch-size = 8
samples-per-epoch = 32
max-epochs-this-instance = 1
lookahead-k = 6
no-compile
```

象棋使用 `[run] game=xiangqi`，原生 `selfplay.cfg` 使用 `bSizesXY=9-10`，不能同时保留 `bSizes` 或 `allowRectangleProb`；`train.cfg` 的 `pos-len` 至少为 10。learner 的输入族由棋类选择，不另设字段别名。当前可用算法与组合边界见 [象棋](xiangqi.md#selfplay-命令与逐轮调度)。

带值项转换为原生 `-名称 值`；无值项是原生 flag，例如 `no-compile` 转发为 `-no-compile`，不写成 `no-compile = false`。值按命令行引号规则拆分，不执行 shell 展开或命令替换。网络结构继续来自来源 `modelconfigs.py`，不在外层重新定义 backbone 或初始化。MuZero 用 `muzero-config` 替代 `model-kind`，指向实际三组件 JSON；`shuffle.cfg` 用 `muzero-unroll-steps` 指定展开长度。

外层只读取阶段编排必需的 `pos-len`、`batch-size`、`samples-per-epoch`、`max-epochs-this-instance`、模型选择，以及 shuffle 的 `num-processes`、`approx-rows-per-out-file`、`keep-target-rows` 和 MuZero 展开长度。内部继续由 `RunConfig` 连接已有控制器。其他原生参数直接转发，类型、缺省值和算法限制由来源消费者检查。原生输出路径、模型发布目录和单设备边界由控制器管理，不能通过参数文件覆盖；shuffle 使用训练配置中的同一个批大小。

外层 INI 文件允许同名 `.local` 文件覆盖对应键，未知外层字段和重复键会报错；不存在隐式旧版本配置或 baseline 继承。共享外层参数文件使用显式路径。引擎文件的 include、覆盖和注释规则仍由 KataGo 自己处理，外层不会把 `maxVisits`、`cpuctExploration`、`nnMaxBatchSize` 等字段改名。

原生文件中的 `@include` 相对包含它的文件解析；其他原生资源路径保持来源进程的工作目录语义。逐轮 selfplay、shuffle、learner 在本版本 `python/` 下执行，因此额外资源建议填写绝对路径；外层不扫描或重写原生字段中的路径。独立 match 的工作目录规则见 [评估](evaluation.md)。

```bash
bash scripts/run.sh check-config --config-dir /absolute/path/to/profile
CONFIG_DIR=/absolute/path/to/profile bash scripts/run.sh \
  --run-dir /absolute/path/to/run --iterations 2 --dry-run
CONFIG_DIR=/absolute/path/to/profile bash scripts/run.sh \
  --run-dir /absolute/path/to/run --iterations 2
```

省略命令时 `run.sh` 默认执行 `run`；也保留 `native`、`plot`、`match` 等显式子命令。`--config-dir` 与 `--config` 选择同一个配置入口，CLI 路径优先于 `CONFIG_DIR`。`--dry-run` 只输出解析后的计划，不创建运行目录、不启动训练；它不替代原生参数和模型加载检查。`check-config` 同样只检查外层及阶段边界。

## 实验目录

实验根目录放置 `exp.cfg`，每个直接子目录中的 `run.cfg` 对应一个实验臂。`[experiment]` 指定每臂累计预算、GPU 槽位及是否共享初始权重：

```ini
[experiment]
iterations = 2
gpus = 0
shared_init = true
```

`gpus` 为逗号分隔的设备索引或 UUID；省略时继承当前设备可见性并串行执行。预算与输出由实验调度器统一管理，各臂 `run.cfg` 中的单独启动预算和 `run_dir` 不改变实验计划。需要只选择部分臂或引用共享目录时，用可选 `[arms]` 显式指定 `臂名 = 配置目录或运行 JSON 路径`，此时不再自动发现。

```bash
CONFIG_DIR=/absolute/path/to/experiment bash scripts/autoexp.sh \
  --work-dir /absolute/path/to/output --dry-run
CONFIG_DIR=/absolute/path/to/experiment bash scripts/autoexp.sh \
  --work-dir /absolute/path/to/output
```

省略 `--work-dir` 时同样按 `configs/` 到 `data/` 的相对路径映射。训练结果存于 `work-dir/runs/臂名`。可选 `[evaluation]` 中的 `config` 指向独立原生 match 配置，其他字段直接使用 `autoelo` 的选择、拟合和驻留选项名（下划线形式）；所有臂完成后自动评估。具体缓存与评分口径见 [比赛与 Elo](evaluation.md)。

运行内的 `config.json` 与实验的已解析 JSON 保存的是调度证据和子进程输入；程序化调用仍可直接提供这些 JSON。用户配置目录不会改写原生日志。算法与性能绘图读取来源日志，等时间横轴使用明确标注的 EtaZero 调度计时，见 [训练观测](observations.md)。

## Web 与推理配置

Web 使用同版本独立 `gtp.cfg`，内容仍是原生字段和 include。按运行路径映射到 `configs/`，或使用保存的 `selfplay_config` 同目录；可以通过 `web/webui.sh --config`／`--config-dir` 明确覆盖。缺失时显示未配置，不套用其他算法配置。模型发现、独立模型入口、Gumbel／MuZero 的字段与启动命令见 [Web 与原生会话](web.md#启动与原生配置)。
