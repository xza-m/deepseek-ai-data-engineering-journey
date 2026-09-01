# Lab 09：从 LLM I/O 工作负载理解 3FS

学习 3FS 的正确入口不是先背 CRAQ 或 RDMA，而是先明确训练数据系统到底产生哪些读写，再判断普通对象存储、
共享文件系统或 3FS 分别解决什么问题。

## 1. 运行四类真实本地 I/O

```bash
make lab09
```

实验在当前文件系统真实执行：

- 顺序扫描 Training Shard；
- 利用 Byte Offset 随机读取 Sample；
- 写入 Lab 06 的 Shuffle 中间 Partition；
- 多线程模拟多个 Rank 并行写 Checkpoint。

## 2. 读延迟与吞吐

```bash
uv run python -m json.tool artifacts/lab09/storage_report.json
```

每类工作负载记录 Operation、逻辑 Bytes、P50/P99 延迟与吞吐。稳定的工作负载定义和产物哈希进入
Storage Fingerprint；受缓存、设备和后台负载影响的性能数字只放在 `observations`。

不要把微型文件的高 MB/s 当作存储系统能力。顺序读取、随机小读、并发写的瓶颈完全不同，且 macOS 缓存数字
无法代表 Linux/RDMA/NVMe 集群。

## 3. 做 3FS Go/No-Go 评审

```bash
open artifacts/lab09/storage_review.md
```

当前主线要求如实检查 Linux、RDMA Device、NVMe Device 和多节点实验床。缺任一项就对“本机 3FS 集群实测”
给出 No-Go，同时保留本地文件系统基线。进阶支线可以先用 MinIO/NFS 建立共享存储对照，再到专用环境执行 3FS。

## 4. 映射 3FS 架构

- Metadata Service/FoundationDB：命名空间和元数据一致性；
- Storage Service/CRAQ：数据复制、写入排序与读取扩展；
- Client/RDMA：训练和数据加工进程的数据路径；
- NVMe：高并发吞吐和尾延迟的介质基础。

这些是来自官方设计与源码的架构证据。只有真实部署、节点/网络/设备说明和可复现基准，才能升级为集群证据。

## 验收

- 四类本地工作负载都真实执行且可验证；
- 随机读取使用可审计 Byte Offset 索引；
- Shuffle 与 Checkpoint 写入产物有哈希；
- 报告明确 `threefs_cluster_executed=false`；
- 能基于前置条件给出 Go/No-Go，而不是为了“使用新技术”强行部署。
