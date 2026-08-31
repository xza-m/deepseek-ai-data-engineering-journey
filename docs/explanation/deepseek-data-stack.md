# DeepSeek 数据—计算—存储—训练链路

smallpond 和 3FS 的价值不在于“替换 Spark 和 HDFS”，而在于展示一种围绕 AI 工作负载重新组合数据系统的方式。

## 四个不同责任层

```text
数据构造
  清洗、去重、质量、混合、Tokenization、Packing
        │
        ▼
smallpond
  用 DuckDB 执行分区内计算，用 Ray 扩展任务
        │
        ▼
3FS
  用共享文件命名空间和聚合 SSD/RDMA 带宽承载数据
        │
        ▼
训练/推理框架
  DataLoader、PP/EP/DP、Checkpoint、KV Cache
```

smallpond 不定义“什么是高质量训练数据”，3FS 也不管理模型数据集的业务语义。训练框架则不会自动保证上游数据可追溯。AI 数据工程师的工作，是把这些边界连接成可验证的交付链。

## smallpond 的启发

smallpond 将集群问题拆成两部分：

- 分区内部交给 DuckDB 这样的高性能单机引擎；
- 分区之间通过显式 Shuffle 和任务调度扩展。

中间结果存入共享文件系统，使计算 Worker 不必长期持有 Shuffle 状态。这种设计符合 KISS，但也把分区选择暴露给用户：工程师必须理解数据倾斜、Join Key 和文件组织。

因此，smallpond 最值得学习的不是 DataFrame API，而是“强单机执行引擎 + 简单分布式编排 + 共享存储”的边界选择。

## 3FS 的启发

传统 HDFS 架构常通过数据本地性减少网络访问。3FS 面向高速 NVMe 和 RDMA 网络，将多个存储节点的带宽聚合成共享数据平面，使计算节点可以相对忽略文件的物理位置。

它还保留文件系统语义：目录、原子 Rename、链接和熟悉的文件接口。这些语义适合数据版本发布、临时目录提交、Checkpoint 和大量中间文件管理。

但 3FS 不是数据目录、湖仓表格式或质量平台。它提供高性能文件数据平面，不替代 Catalog、Schema、权限、数据许可和样本血缘。

## 与 DeepSeek-V3 的连接

DeepSeek-V3 报告描述了数据去冗余、保持多样性、Document Packing、FIM 和 Byte-level BPE Tokenizer。报告同时描述了大规模流水线并行、专家并行和数据并行。

这两部分之间存在直接的工程约束：

- 数据混合决定每个领域和语言进入训练的 Token 比例；
- Tokenizer 决定数据压缩效率和模型输入单位；
- Packing 决定有效 Token 比例和文档边界；
- DataLoader 和存储吞吐决定 GPU 是否等待；
- Checkpoint 决定故障恢复成本；
- 训练和评测结果又反过来指导数据版本。

## 不能简单复制的部分

3FS 官方部署需要 Linux、FoundationDB、存储节点和高速网络；官方性能基准也依赖明确的 NVMe/RDMA 集群。大多数团队不应该因为 DeepSeek 开源了项目就立刻替换现有对象存储或 Spark。

正确的迁移方法是：

1. 先测量当前 DataLoader、Shuffle 和 Checkpoint 瓶颈；
2. 判断问题属于数据质量、文件组织、计算还是存储；
3. 用最小实验验证设计；
4. 只有共享存储和高吞吐需求真实存在时，再评估 3FS 类架构。
