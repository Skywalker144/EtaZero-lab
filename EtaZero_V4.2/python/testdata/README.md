# 数值测试数据

`etazero_benchmark_1024.npz` 由本版本原生 selfplay 的 random evaluator 生成，供 loss、梯度、编译和优化器路径比较使用。它包含 19×19 棋盘的 1024 行、80 个全局目标通道，以及 两组未来局面空间目标通道。生成时按来源规则处理落子上限，不是棋力或正式训练效果证据。

生成入口是 [`tests/generate_training_fixture.py`](../../tests/generate_training_fixture.py)。在 `pytorch` 环境中运行并指定尚不存在的输出路径；采样使用原生随机种子，不承诺逐位再生成。示例：

```bash
python tests/generate_training_fixture.py --output /tmp/etazero_benchmark.npz
```

来源 `benchmark_data_1024.npz` 原件保留，其旧目标布局不再用于 V4 learner 或当前数值测试。
