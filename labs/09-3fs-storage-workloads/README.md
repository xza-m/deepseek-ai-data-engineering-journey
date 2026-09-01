# Lab 09：3FS 与 LLM I/O 工作负载

状态：`可运行`

```bash
make lab09
```

输入：Lab 05 Shard、Lab 06 Shuffle Partition、Lab 08 Checkpoint。

输出：四类本地 I/O 基线、随机读取索引、复制产物、`storage_report.json` 和 3FS Go/No-Go 评审。

当前主线不声称执行 3FS 集群；Linux/RDMA/NVMe/多节点实测属于 Infra 进阶支线。
