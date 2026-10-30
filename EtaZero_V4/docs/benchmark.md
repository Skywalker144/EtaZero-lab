# 原生搜索与推理 benchmark

`katago benchmark` 和 GTP `kata-benchmark` 使用来源 `PlayUtils::benchmarkSearchOnPositionsAndPrint`，支持 Go、Freestyle／Standard／Renju、Hex、象棋，以及普通／MuZero 的 PUCT／Gumbel 搜索。独立命令保留来源线程扫描、自动搜索线程数、batch 设置与额外服务线程测试；EtaZero 的 `native` 入口只转发参数。

```bash
bash scripts/run.sh native benchmark -config /absolute/path/to/gtp.cfg \
  -model /absolute/path/to/model -threads 1,4,8 -visits 200 -numpositions 10

bash scripts/run.sh native benchmark -config /absolute/path/to/gtp.cfg \
  -model /absolute/path/to/model -sgf /absolute/path/to/game.sgf -tune
```

普通 `-model` 是模型文件；配置 `useMuZero=true` 时是三组件 bundle 目录。搜索、精度、设备、metadata 和棋类字段沿用原生配置。参数组合必须满足对应算法的搜索约束，不为测速自动关闭 graph、SVB 等设置。独立命令的 `-visits` 覆盖本次搜索访问和 playout 上限，`-numpositions` 限制抽样数；省略 `-threads` 会执行来源自动线程搜索。

## 局面与计时

Go 默认继续使用来源 7～19 路内置 SGF；其采样、按着法构造的确定种子、清树／清缓存位置和计时范围不变。Go 基准仍按来源使用 Tromp-Taylor 类规则并保留棋谱贴目。可用 `-sgf` 指定其他尺寸和棋谱；空棋谱没有可抽样的着法前位置，会明确拒绝。

其他棋类默认使用固定种子生成的合法轨迹，模型和线程数不参与轨迹生成。轨迹按实际棋规终局停止，象棋保留完整规则历史。它们用于测速，并不代表开局分布或棋力测试集。需要匹配研究工作负载时，使用实际同棋类 SGF；显式 `game`／`rules` 的棋类与 SGF 不一致会报错，未指定棋类时采用 SGF 声明。没有 SGF 时从 `game` 或 `rules` 选择棋类，两者都没有时沿用来源 Go。SGF 的初始位置和象棋前史由原生解析器恢复。

`-boardsize` 指定方形棋盘；省略时读取来源 `defaultBoardSize`／`defaultBoardXSize`／`defaultBoardYSize`。象棋默认 9×10，不能用一个方形尺寸代替。GTP `kata-benchmark VISITS` 按当前棋类和棋盘尺寸选择同一组内置局面，使用独立搜索对象，不修改当前棋局；Go 矩形棋盘仍需通过独立命令的 `-sgf` 测试。

每个候选线程数测量同一组局面。模型初始化、预热、棋谱重放和切换服务的准备时间不计入搜索秒数；实际搜索、组批、推理和回传在计时范围内。结果输出来源的 `visits/s`、`nnEvals/s`、`nnBatches/s`、`avgBatchSize`。MuZero 的 NN 行数包含 initial 和 recurrent；它与普通网络的一次棋盘评估不是等价计算单元。随机 evaluator 没有实际 NN 调用，行数和 batch 数为零，不能作为 GPU 吞吐证据。

## 线程与 batch 调整

独立命令沿来源在扫描期间忽略配置中的 `nnMaxBatchSize`，按后端并发策略推导大小；`-fixed-batch-size`、`-half-batch-size` 可以覆盖。默认额外测试半 batch；`-no-half-batch-size-test` 关闭该测试。CUDA／ROCm 普通网络保留来源的每设备双服务线程测试，`-no-server-thread-test` 可关闭。

MuZero 使用现有三组件服务，允许动态降低 batch 上限。测试双服务线程时，先结束全部 benchmark 搜索并释放其 latent，再销毁旧服务、按原生配置创建新服务并预热；没有在活动树上更换 latent 所属设备。普通网络继续沿来源停止／重启 evaluator 工作线程。额外设置仍按来源的实测速度改善阈值决定是否推荐，准备成本不计入搜索速度。

不同线程数、batch 与服务线程数会改变并发、缓存及随机推理分配；工具不承诺相同搜索树或逐局结果。当前真实设备检查使用单张 GPU 上的多个服务线程，未据此验证多物理 GPU 性能。其他来源后端保留普通网络路径，但未在对应设备验证本次接入。

## 与 Elo 的区别

来源 Go PUCT benchmark 自带一个根据速度与线程数粗估 Elo 影响的经验公式。普通 Go PUCT 保留这一来源行为；它不是比赛测得的 Elo，也没有因 V4 的网络改动重新校准。

该公式没有多棋类、Gumbel 或 MuZero 的校准依据，因此这些路径不输出 `EloDiff`，自动调线程使用实际 `visits/s`，只报告吞吐最高的已测配置。来源的候选线程集合、三分搜索和扩大搜索区间流程保留；噪声和非单峰性能曲线仍可能使其错过最优配置。实际棋力比较使用 [原生比赛与 Elo](evaluation.md)，不从测速曲线生成棋力结论。

`tests/test_benchmark.py` 检查六种棋类的独立命令和 GTP、SGF 棋类／尺寸、错误后的继续使用及自动线程扫描；CUDA 检查覆盖普通／MuZero、PUCT／Gumbel、FP32／FP16、双服务线程和固定／动态 batch。测试证明路径和计数可用，不把小规模检查的耗时当成硬件调优结论。其他原生工具仍需按其自身入口核对，不能由 benchmark 的支持范围推断已全部接入。

## 纯推理 benchmarknn

`katago benchmarknn` 直接测量原生后端前向调用，不经过搜索、evaluator 请求队列或缓存。普通网络沿用 `NNEvaluator::benchmarkPureForward`；MuZero 直接调用已有的 `MuZero::Evaluator`。两者共用从来源实现提取的 `cpp/neuralnet/nnbenchmark.h` 计时循环，保留预热后的并发起跑、逐线程中位数与实际墙钟吞吐计算。

```bash
bash scripts/run.sh native benchmarknn -config /absolute/path/to/nn.cfg \
  -model /absolute/path/to/model -boardsize 9,13,19 \
  -batch-size 16 -warmup 20 -iterations 200 -json

# 配置中 useMuZero=true，model 是三组件目录。
bash scripts/run.sh native benchmarknn -config /absolute/path/to/nn.cfg \
  -model /absolute/path/to/muzero -muzero-operation both -json
```

棋类由 `game` 或 `rules` 选择，两者同时存在时必须属于同一棋类；MuZero 还必须与 bundle 的棋类一致。没有显式棋类时，普通网络沿用来源 Go 输入，版本 18 网络使用 Freestyle；MuZero 使用 bundle 的棋类。Go 使用来源空棋盘和 Tromp-Taylor 类规则，落子棋使用相应变体的空棋盘特征，象棋使用完整初始位置。metadata 网络继续使用来源的 dummy warmup profile，普通网络固定为恒等对称；测速输入不代表实际对局分布。

`-boardsize` 接受 `9,13,19` 或 `5x7,9x9`，按 batch 行循环输入尺寸。只有 Go 允许任意矩形，落子棋要求方形，象棋固定 9×10。缺省尺寸为普通网络 19、象棋 9×10、其他 MuZero 棋类的模型画布。普通网络的默认 NN 缓冲区取各方向最大尺寸，仍可由原生 `maxBoard…ForNNBuffer` 覆盖；MuZero 缓冲区必须等于 bundle 的固定方形画布。

batch、服务线程、设备、精度和布局均由原生 `Setup` 解析；每个配置的服务线程对应一个独立工作线程／compute handle。`-batch-size` 优先于 `nnMaxBatchSize`。`-require-exact-nnlen` 要求只列一个尺寸，其值作为来源 `requireExactNNLen` 的默认值，原生 `requireMaxBoardSize` 配置仍可覆盖；开启后实际棋盘必须填满缓冲区。普通后端可能据此省略 mask 运算；MuZero 仍使用带 mask 的现有执行路径，不能将该标志解释为相同优化，象棋也不能填满其方形 MuZero 画布。

MuZero 的 `-muzero-operation` 接受 `initial`、`recurrent`、`both`，默认分别测量两者。initial 包含 representation 与 prediction；recurrent 在计时前生成固定父 latent，反复测量 dynamics 与 prediction，不重复计入 representation，也不把 latent 下载到主机。选用的动作来自各棋类的初始合法动作域；recurrent 输入不随迭代演化，不是完整搜索轨迹。

计时排除加载模型、生成棋盘特征、后端预热、显式 warmup 和 compute handle 销毁。普通网络包含来源 `getOutput` 的主机／设备传输、前向与原始输出整理；MuZero 包含其实际 initial/recurrent API 的输入整理、mask、传输、latent／输出分配与释放及前向。两者都不计入搜索后处理、合法策略归一化或实际调度队列。这不是纯 kernel 时间，也不是端到端自对弈吞吐；一次 initial、recurrent 与普通前向代表不同工作量。

输出的 `actualWallNNEvalsPerSec` 为总计时行数除以共同起跑至最后一个工作线程完成的墙钟时间，是并发吞吐指标。`sumMedianNNEvalsPerSec` 只是各线程中位耗时换算速率之和，不能代替前者。较早完成的工作线程释放资源仍可能影响其他线程的末尾迭代，沿用来源行为。

普通 `-json` 保留来源顶层指标；新增 `game`、`boardShapes` 和实际 `nnBufferSize`，全为方形时保留数值列表 `boardSizes`。MuZero 在 `operations.initial`／`operations.recurrent` 下分别输出相同指标，每次调用只输出一个 JSON 对象；日志写入 stderr。`usingFP16` 报告实际 compute handle 的精度，沿来源表示是否有任一线程使用 FP16，不直接复制配置中的 auto 值。

`tests/test_benchmarknn.py` 覆盖六种棋类、普通／MuZero、FP32／FP16、矩形与填充尺寸、metadata／Transformer、双工作线程、错误输入与工作线程失败退出。原生计时检查验证预热屏障、调用数量和排除销毁耗时；真实 CUDA 检查验证路径与吞吐计数关系，不设性能阈值。当前设备检查使用单张 GPU，未验证其他后端或多物理 GPU 的性能。

## 推理精度与批处理误差

`katago testgpuerror` 使用来源的逐局面 FP32、并发 FP32、当前配置和并发当前配置比较流程，支持六棋类普通模型及 MuZero。它不进行搜索，因此 PUCT／Gumbel 不改变此工具的测量对象。

```bash
bash scripts/run.sh native testgpuerror -config /absolute/path/to/nn.cfg \
  -model /absolute/path/to/model -sgf /absolute/path/to/game.sgf

# MuZero 配置 useMuZero=true；检查 initial 及每个指定深度的 recurrent。
bash scripts/run.sh native testgpuerror -config /absolute/path/to/nn.cfg \
  -model /absolute/path/to/bundle -sgf /absolute/path/to/game.sgf \
  -muzero-recurrent-steps 3
```

Go 缺省继续使用来源 19 路数据，可选择来源 `9`、`13`、`19`、`10x14`、`rectangle` 数据集。其他棋类使用 benchmark 的固定合法轨迹，尺寸由 `-boardsize` 或原生默认尺寸字段选择，象棋为 9×10；这不是训练或棋力测试分布。`-sgf` 可重复，读取同棋类棋谱的唯一未终局位置与变化，包括象棋完整前史、落子棋零气摆子及矩形 Go；它与显式 `-boardsize` 互斥。`-quick` 保留来源固定种子下的 30% 抽样，样本为空会报错。棋类不匹配不会改用其他棋类。

原生 `nnMaxBatchSize`、服务线程、设备、布局与精度决定实际推理设置；缺少 batch 配置时继续使用来源的 `numSearchThreads` 或默认值。比较跳过缓存读取，按同一局面哈希选择有效棋类对称；Go 的八对称映射与来源相同。metadata 使用来源固定 `preaz_5k` 诊断 profile。`policyOptimism`、`playoutDoublingAdvantage` 与 `rootPolicyTemperature` 沿用来源诊断字段，不从搜索结果取值。

输出保留来源平均、90%、99% 和最大误差，以及原有 W−L／NR、最大先验差、策略 KL 与短期价值误差指标和阈值。新增 `drawError` 单独检查 D 概率，使用同组胜率误差阈值，避免 W 和 L 同时偏移而旧指标不变。象棋比较完整 2086 动作；非法动作掩码、布局不一致或非有限值会明确失败。通过表示这些样本在相应数值阈值内一致，不证明棋力，也不能排除各精度共有的实现错误。

普通／MuZero 模型可用 Eigen 生成 `-reference-file`，再由支持相应模型的后端读取同一路径，使用相同模型、位置与诊断参数进行比较。文件保留四结果、实际策略数组及来源输入哈希；旧的三结果参考文件不能使用。读取时检查样本数量、布局和输入哈希，防止把其他局面或参数的文件当作参考。输入哈希不包含模型权重，调用者须为两端指定同一导出模型。Eigen 沿来源行为写入指定文件，使用新的输出路径保存原始参考。当前已验证 Eigen FP32 与 CUDA／OpenCL FP32／FP16；其他后端须在对应设备验证。

MuZero 缺省检查 initial 和 1～3 层 recurrent；`-muzero-recurrent-steps 0` 可只检查 initial。每个深度单独报告同一套误差；每次从该精度自己的 initial latent 开始，按固定种子选择相同静态动作序列，保持根朝向，不读取真实后继棋盘。检查包含随深度累积的数值差异，不跨模型传递 latent。指定 `-reference-file FILE` 时，initial 使用 `FILE`，每个 recurrent 深度使用 `FILE.recurrent-N`，各文件沿用同一原生 JSON-lines 输出格式。Eigen 写出这些参考，CUDA／OpenCL 在对应深度分别读取；所有文件都须来自同一模型、局面与诊断参数。递归参考哈希由根输入哈希按动作序列延伸，只用于本命令的样本匹配；错用其他根、其他深度或缺少文件会报错。未指定参考文件时仍以本后端的逐局面 FP32 为参考。此命令比较后处理后的策略、四结果及来源误差指标；原始 latent 与 logits 的独立 PyTorch 对照见 `tests/test_muzero_eigen.py`。

`tests/test_gpu_error.py` 覆盖六棋类、真实 CUDA FP32／FP16、MuZero 各递推深度、原生数据集及 quick 抽样、普通／MuZero 的 Eigen 参考读写、深度文件错配与缺失拒绝、initial-only、重复 CPU 导出、metadata／Transformer、矩形 Go、象棋前史与输入边界。`tests/test_onnx_backend.py` 另以同一工具对照六棋类普通网络的 Eigen 与 ONNX Runtime CPU FP32 输出。原生规则测试另用可手算分布验证和棋误差检测、象棋动作尾部 KL、掩码拒绝及参考序列化。

## 训练族参考检查

`katago testbackendreference` 比较候选模型与一组训练模型的混合参考，沿用来源的策略 KL、共识动作、期望得分二元 KL 和短期价值误差检查。它与上述同模型精度比较用途不同：混合参考能发现多种精度共有的系统性异常，但合法模型之间本身存在差异，因此参考数据和阈值必须按实际训练族校准。来源内置 Go 数据、阈值、模型大小宽限及极小模型豁免保留；这些数据不能作为 V4 模型、其他棋类或 MuZero 已校准的依据。

工具支持六棋类的普通网络和 MuZero initial。先用 `-dump-only` 在指定局面采集原生输出；此模式明确报告未执行参考检查，不输出 `PASSED`。用新路径保存输出，避免覆盖原始参考。

```bash
bash scripts/run.sh native testbackendreference -config /absolute/path/to/nn.cfg \
  -model /absolute/path/to/model-or-bundle \
  -reference-data-file /absolute/path/to/positions.jsonl \
  -dump-only -dump-candidate /absolute/path/to/new-outputs.jsonl

# 参考数据与阈值须来自待检模型所属训练族的独立校准。
bash scripts/run.sh native testbackendreference -config /absolute/path/to/nn.cfg \
  -model /absolute/path/to/model-or-bundle \
  -reference-data-file /absolute/path/to/calibrated-reference.jsonl \
  -reference-limits-file /absolute/path/to/calibrated-limits.json
```

参考文件每行是 JSON，包含原生 `sample`、`rules`、`komi`，可指定来源的 `policyOptimism`、`pda`、`maxHistory`。`sample` 使用原生 `Sgf::PositionSample` 格式，保留棋类、落子棋零气摆子、原始手数以及象棋完整前史；非围棋终局没有可比较的合法策略，会报错。采集前半局面顺序执行，后半并发执行，沿用来源 batch 16；metadata 使用来源 `preaz_5k`，对称按同一哈希选择棋类有效对称，Go 与来源一致。

采集输出包含实际生效参数、对称、四结果与策略。点动作棋类的策略是物理棋盘行优先数组加最后的 pass 槽，非围棋该槽为 0；象棋使用全部 2086 个原生联合动作。正式参考可包含 `policyMixture`、`winrateAvg`、`shorttermWinlossErrorAvg` 和白方顺序的 `resultAvg=[W,D,L,NR]`。四结果用 `KL(reference || candidate)` 单独检查，避免 W、L 同时变化而期望得分不变；概率下限沿用来源二元 KL 的 `1e-6` 稳定尺度。仅有局面、没有参考指标时只能采集，不能声称检查通过。

显式阈值文件是 JSON 对象，字段名沿用 [`BackendRefLimits`](../cpp/tests/testbackendreference.cpp)：`mixtureSmoothing`、`avgPolicyKL`、`p95PolicyKL`、`consensusTopProb`、`consensusMinProb`、`maxConsensusViolations`、`avgWinrateKL`、`p95WinrateKL`、`avgStWinlossErrorAE`、`p95StWinlossErrorAE`，以及四结果的 `avgResultKL`、`p95ResultKL`。所有字段须明确提供；不回填来源 Go 校准值。使用显式阈值时不应用来源模型大小宽限或极小模型豁免，`-lenience-factor` 仍按来源方向调整阈值。新棋类、MuZero 或四结果参考检查必须指定该文件。

MuZero 配置使用原生 `useMuZero=true`，参考采集只评估真实局面的 initial；recurrent 的数值检查使用 `testgpuerror`。普通／MuZero initial 可在 Eigen 采集后由 CUDA 检查相同参考；OpenCL 支持普通模型对照。`tests/test_backend_reference.py` 验证六棋类、CUDA FP32／FP16、MuZero、Eigen 对照、metadata／Transformer、完整象棋历史与手算四结果偏移；`tests/test_opencl_backend.py` 另验证 Go metadata／Transformer 的 Eigen 与 OpenCL FP32／FP16 对照。测试阈值仅服务构造样例，V4 不附带未经实际训练族校准的生产阈值。
