# Lab 06：从 SQL 执行到 Partition、Shuffle 与恢复

这一 Lab 把传统数据工程师熟悉的 SQL、分区和任务重跑，迁移到训练数据加工的执行层。你会真实运行
DuckDB 和 Parquet，并用同一份数据观察热点 Key、哈希重分区、任务失败和中间结果复用。

## 1. 运行实验

```bash
make lab06
```

输入是 Lab 05 的 `mixed_documents.jsonl` 和 `dataset_version_manifest.json`。实验把原创微型文档确定性扩展，
同时执行 Python 逐行聚合与 DuckDB 向量化聚合，结果必须完全一致。

## 2. 检查真实产物

```bash
uv run python -m json.tool artifacts/lab06/compute_report.json
uv run python -m json.tool artifacts/lab06/duckdb_plan.json
ls artifacts/lab06/partitions-baseline
ls artifacts/lab06/partitions-salted
```

重点观察：

- `aggregation_equal`：Python 与 DuckDB 的业务结果是否相同；
- `baseline_partition_distribution`：热点 Key 是否把数据压到少数 Partition；
- `salted_partition_distribution`：加盐后最大分区与平均分区的比例是否下降；
- `retry_count`：指定 Partition 首次失败后是否只重试一次；
- `recovery_pass_reused_partition_count`：恢复阶段是否复用了所有已完成中间结果。

## 3. 对照 smallpond

`duckdb_plan.json` 保存了本地 SQL Plan 和 smallpond 概念映射：Lazy DataFrame 对应逻辑计划，Hash Partition
产生 Shuffle 文件，Task 完成标记提供最小恢复边界。这里没有假装在 macOS 上执行 smallpond 分布式 Runtime；
真正的多 Worker、共享存储和调度器行为属于 Linux 集群进阶支线。

## 4. 做一次最小诊断

修改 `--hot-key-fraction` 或 `--partition-count` 后重跑，再比较两种分区分布。不要只看总耗时：微型数据下
Python 甚至可能更快；本 Lab 的核心证据是执行计划、计算等价性、倾斜位置和恢复语义。

## 验收

- DuckDB 聚合与 Python 聚合一致；
- 生成可读取的 Parquet 和执行计划；
- 加盐后的倾斜比基线更低；
- 故障注入只产生一次重试，恢复阶段复用全部完成分区；
- `validate-compute` 能发现被篡改的 Partition。

完成后，你应能解释 smallpond 为什么建立在强单机引擎和共享中间文件之上，以及这种设计与 Spark
长生命周期 Executor 的差异。
