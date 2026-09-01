# Lab 06：Partition、Shuffle 与恢复

状态：`可运行`

```bash
make lab06
```

输入：Lab 05 Dataset Version。

输出：DuckDB/Parquet 基线、逻辑计划、两套 Partition、中间任务状态和 `compute_report.json`。

验收：`uv run aide validate-compute --input artifacts/lab05 --output artifacts/lab06`。

本地真实执行 DuckDB、哈希分区、热点 Key 加盐与故障注入；smallpond Runtime 和分布式集群性能不在
当前环境的证据范围内。
