# DeepSeek 源码与论文阅读索引

阅读顺序遵循“先工作负载，再计算，再存储，最后进入实现细节”。不要从 3FS C++ 文件树第一页开始硬读。

## 第一组：建立全景

1. [smallpond README](https://github.com/deepseek-ai/smallpond)
2. [3FS README](https://github.com/deepseek-ai/3FS)
3. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

阅读问题：

- 每个项目的直接用户是谁？
- 它处理的数据单位是什么？
- 它优化的是计算、存储、网络、显存还是模型质量？
- 哪些能力明确不在项目边界内？

## 第二组：DeepSeek-V3 数据与训练

重点阅读：

- 4.1 Data Construction；
- 3.1 Compute Clusters；
- 3.2 Training Framework；
- 3.4 Inference and Deployment。

建立以下联系：

```text
去重/多样性 → 训练数据版本
Document Packing → Sequence 组织
Tokenizer → Token 数和压缩效率
PP/EP/DP → 数据分配与通信
Checkpoint/KV Cache → 存储工作负载
```

## 第三组：smallpond

文档和源码入口：

1. [Getting Started](https://github.com/deepseek-ai/smallpond/blob/main/docs/source/getstarted.rst)
2. [`dataframe.py`](https://github.com/deepseek-ai/smallpond/blob/main/smallpond/dataframe.py)
3. [`logical/node.py`](https://github.com/deepseek-ai/smallpond/blob/main/smallpond/logical/node.py)
4. [`logical/optimizer.py`](https://github.com/deepseek-ai/smallpond/blob/main/smallpond/logical/optimizer.py)
5. [`logical/planner.py`](https://github.com/deepseek-ai/smallpond/blob/main/smallpond/logical/planner.py)
6. [`execution/task.py`](https://github.com/deepseek-ai/smallpond/blob/main/smallpond/execution/task.py)
7. [`session.py`](https://github.com/deepseek-ai/smallpond/blob/main/smallpond/session.py)

源码问题：

- DataFrame 操作何时只构建逻辑计划，何时真正执行？
- `repartition` 如何变成生产者和消费者任务？
- 相邻 SQL 为什么可以融合？
- 中间文件在容错和扩展中承担什么作用？
- Ray 负责什么，DuckDB 又负责什么？

## 第四组：3FS

入口：

1. [Design Notes](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md)
2. [Setup Guide](https://github.com/deepseek-ai/3FS/blob/main/deploy/README.md)
3. [USRBIO API](https://github.com/deepseek-ai/3FS/blob/main/src/lib/api/UsrbIo.md)

设计问题：

- 为什么元数据服务可以无状态？
- FoundationDB 事务解决了哪些文件语义？
- CRAQ 如何做到 Write-all、Read-any？
- 为什么训练数据需要随机读，而 Checkpoint 需要并行写？
- FUSE 的低接入成本和原生接口的性能分别适用于什么场景？
- 失效检测、Lease、Chain 版本和数据恢复如何协作？

## 阅读输出模板

每次阅读只提交一页笔记：

```text
问题：本次要回答什么？
结论：三条以内。
证据：论文章节、官方文档或源码路径。
实验：如何在本地验证其中一部分？
边界：还有哪些结论没有验证？
迁移：它和传统数据工程哪个概念相连？
```

不要大段翻译原文。知识体系来自问题和实验之间的连接。
