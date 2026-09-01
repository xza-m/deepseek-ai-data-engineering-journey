# DeepSeek 源码与论文阅读索引

阅读顺序遵循“先工作负载，再计算，再存储，最后进入实现细节”。不要从 3FS C++ 文件树第一页开始硬读。

第一次阅读请直接完成 [Bridge 01：DeepSeek 源码与论文连接](../tutorials/bridge-01-deepseek-source-study.md)。
本索引用于之后查找准确入口，不替代教程。

## 版本与证据约定

本索引在 2026-08-31 固定到以下官方快照：

| 材料 | 版本 | 说明 |
| --- | --- | --- |
| DeepSeek-V3 Technical Report | arXiv `2412.19437v2` | 论文证据 |
| smallpond | [`52ecc5e`](https://github.com/deepseek-ai/smallpond/tree/52ecc5e45535c7448f848bcb45b0da00d9484f81) | 源码证据 |
| 3FS | [`22fca04`](https://github.com/deepseek-ai/3FS/tree/22fca04564c7cc230fd8b9523b8b92864e1dad47) | 源码证据 |

两个项目当时都没有 GitHub Release，因此使用提交而不是 Release Tag。后续上游更新时，应先复核调用链，再更新此表。

## 第一组：建立全景

1. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
2. [smallpond README](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/README.md)
3. [3FS README](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/README.md)

阅读问题：

- 每个项目或报告的直接用户是谁？
- 它处理的数据单位是什么？
- 它优化的是计算、存储、网络、显存还是模型质量？
- 哪些能力明确不在项目边界内？
- 哪些关系是官方明确描述，哪些只是我们的架构映射？

## 第二组：DeepSeek-V3 数据与训练

重点阅读：

- 2.2 Multi-Token Prediction；
- 3.1 Compute Clusters；
- 3.2 Training Framework；
- 4.1 Data Construction；
- 4.2 中的 Training Hyper-Parameters。

建立以下联系：

```text
去重/多样性 → 训练数据版本
Document Packing / FIM → Sequence 组织
Tokenizer → Token 数和压缩效率
PP/EP/DP → 数据分配与通信
Checkpoint/KV Cache → 存储工作负载
```

重点记录报告明确公开的事实：

- 14.8T Token、减少冗余同时保留多样性；
- Document Packing、0.1 比例的 PSM FIM；
- 128K Byte-level BPE Tokenizer；
- 4K 预训练 Sequence Length；
- 16-way PP、64-way EP、ZeRO-1 DP。

报告没有公开完整数据 Pipeline，也没有直接证明公开 smallpond/3FS 提交与 V3 生产环境的精确对应关系。

## 第三组：smallpond

文档和源码入口：

1. [Getting Started](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/docs/source/getstarted.rst)
2. [Internals](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/docs/source/internals.rst)
3. [`dataframe.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/dataframe.py)
4. [`logical/node.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/logical/node.py)
5. [`logical/optimizer.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/logical/optimizer.py)
6. [`logical/planner.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/logical/planner.py)
7. [`execution/task.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/execution/task.py)
8. [`execution/driver.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/execution/driver.py)
9. [`session.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/session.py)

源码问题：

- DataFrame 操作何时只构建逻辑计划，何时真正执行？
- `repartition` 如何变成生产者和消费者任务？
- 相邻 SQL 为什么可以融合？
- 中间文件在容错和扩展中承担什么作用？
- Ray 路径与 Scheduler/Executor 路径有什么不同？
- DuckDB、smallpond 和运行时各负责什么？

## 第四组：3FS

入口：

1. [3FS README](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/README.md)
2. [Design Notes](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/docs/design_notes.md)
3. [Setup Guide](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/deploy/README.md)
4. [USRBIO API](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/src/lib/api/UsrbIo.md)

设计问题：

- 为什么元数据服务可以无状态？
- FoundationDB 事务解决了哪些文件语义？
- CRAQ 如何做到 Write-all、Read-any？
- 为什么训练数据需要随机读，而 Checkpoint 需要并行写？
- FUSE 的低接入成本和原生接口的性能分别适用于什么场景？
- 失效检测、Lease、Chain 版本和数据恢复如何协作？

源码只定位职责，不要求一次通读实现：

- `src/mgmtd/`：成员、路由、Chain Table 和 Lease；
- `src/meta/`：文件系统语义和 FoundationDB 元数据；
- `src/storage/`：Chunk、Target、复制链与 SSD I/O；
- `src/client/meta/`、`src/client/storage/`：客户端的元数据和文件数据路径。

## 阅读输出模板

每次阅读复制 [源码学习笔记模板](source-study-note-template.md)，只提交一页笔记：

```text
问题：本次要回答什么？
结论：三条以内。
证据：论文章节、官方文档或源码路径。
实验：如何在本地验证其中一部分？
边界：还有哪些结论没有验证？
迁移：它和传统数据工程哪个概念相连？
```

不要大段翻译原文。知识体系来自问题和实验之间的连接。

## 本地映射入口

阅读上游源码后必须回到本项目：

1. `pipeline.py`：Document → Token → Sequence → Dataset Manifest；
2. `training.py`：Dataset → DataLoader → Loss → Checkpoint → Run Manifest；
3. `experiment.py`：固定控制变量的数据版本 A/B；
4. Lab 06：未来验证 smallpond 的 Partition、Shuffle 和物化；
5. Lab 09：未来验证共享存储工作负载和 3FS 证据边界。

如果一条上游源码结论无法连接到本地数据产物、指标或后续实验，它暂时只是背景知识，不是已掌握的工程能力。
