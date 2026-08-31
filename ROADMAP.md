# 16 周学习路线

这是一条面向在职数据工程师的标准路线，默认每周投入 8～10 小时。每周都要留下可审阅的代码、数据产物或实验报告，而不是只完成阅读。

## 验收原则

每一阶段必须回答四个问题：

1. 输入和输出分别是什么？
2. 结果如何复现和追溯？
3. 指标证明了什么，又不能证明什么？
4. 这项能力如何影响模型质量、GPU 效率或工程成本？

## 第一阶段：理解 LLM 如何消费数据

### 第 1 周：从文档到 Loss

- 学习 Document、Sample、Token、Sequence、Batch；
- 理解 Next Token Prediction 和标签右移；
- 完成 Lab 00 环境检查；
- 画出一条原始文档进入训练循环的链路。

验收：能够解释 `input_ids`、`labels` 和 `loss_mask` 的关系。

### 第 2 周：Tokenization 与 Packing

- 学习 Byte-level BPE、特殊 Token 和词表；
- 完成 Lab 01；
- 检查文档边界、填充和 Packing 利用率；
- 追溯一个 Sequence 中每段 Token 的来源文档。

验收：相同输入和配置能生成相同 Pipeline Fingerprint。

## 第二阶段：Mini-LLM 数据工厂

### 第 3 周：解析与规范化

- HTML、Markdown、代码和对话的结构差异；
- Unicode、编码、空白、模板和异常文本；
- 建立原始层、规范层和训练候选层。

验收：规范化规则具有单元测试，原始文本仍可追溯。

### 第 4 周：精确与近似去重

- 内容哈希；
- Shingling、MinHash 与 LSH；
- 文档级、段落级和跨数据集去重；
- 训练集与评测集污染检测。

验收：输出重复簇、保留策略、去重率和误判样例。

### 第 5 周：质量、安全与合规

- 结构质量、语言质量、知识密度和可训练性；
- PII、安全、版权与来源许可；
- 规则、分类器和 LLM Judge 的证据差异。

验收：每次过滤都有原因、版本和统计，不以“高质量”替代指标。

### 第 6 周：数据混合与版本

- 领域、语言、难度和来源配比；
- 确定性采样和随机种子；
- Dataset Manifest、Dataset Card 和血缘；
- 数据版本的兼容与回滚边界。

验收：任一训练样本能追溯到来源、规则、Tokenizer 和配置。

## 第三阶段：smallpond 计算引擎

### 第 7 周：DuckDB、Arrow 与 Parquet

- 向量化执行；
- 列式存储、Predicate Pushdown；
- Arrow 内存模型和零拷贝边界；
- 单机强引擎在分布式架构中的位置。

验收：对同一加工任务给出 pandas、DuckDB 的执行与资源对比。

### 第 8 周：逻辑计划、Partition 与 Shuffle

- 阅读 smallpond `dataframe.py`、`logical/node.py`；
- 理解 Lazy DAG、Hash Partition、Shuffle 和 Join；
- 人工制造数据倾斜和小文件问题。

验收：能够解释为什么 Join 两侧必须按相同 Key 分区。

### 第 9 周：计划、调度与容错

- 阅读 `optimizer.py`、`planner.py` 和执行任务；
- 理解 SQL 融合、任务拆分、重试和推测执行；
- 完成 smallpond 版训练数据加工实验。

验收：保留逻辑图、执行时间线、分区指标和失败重试证据。

## 第四阶段：训练系统与数据供给

### 第 10 周：DataLoader

- Map-style 与 Iterable-style Dataset；
- 随机顺序、Worker、预取和缓存；
- Shard 大小、随机读和顺序读；
- DataLoader 等待时间。

验收：能够区分存储、解压、Tokenization 和 Batch 组装瓶颈。

### 第 11 周：Tiny LLM

- 构建最小 Transformer 训练循环；
- 记录 Loss、Tokens/s、样本顺序和数据版本；
- 对比两种 Packing/Sharding 策略。

验收：模型结果与 Dataset Manifest 建立双向关联。

### 第 12 周：分布式训练与 Checkpoint

- DP、TP、PP、EP 和 ZeRO；
- MoE All-to-All；
- 参数、优化器状态、随机状态和数据游标；
- Checkpoint 写入与恢复。

验收：中断后能从 Checkpoint 恢复，并说明数据顺序是否一致。

## 第五阶段：3FS 与 AI 存储

### 第 13 周：3FS 核心架构

- Manager、Metadata、Storage、Client；
- FoundationDB 元数据事务；
- Chunk、Stripe、Chain 和 CRAQ；
- FUSE 与原生客户端。

验收：能够画出一次文件创建、写入、读取和故障切换路径。

### 第 14 周：I/O 与故障实验

- NVMe、NUMA、InfiniBand/RoCE、RDMA；
- DataLoader 随机读、Checkpoint 并行写；
- USRBIO 和 KV Cache；
- 可选 Linux 测试集群。

验收：本地设计实验和真实集群结果必须分开报告。

## 第六阶段：反馈闭环与开源毕业项目

### 第 15 周：数据版本 A/B

- 设计一次数据变量实验；
- 固定模型、训练预算和评测；
- 比较数据质量、训练效率和模型效果；
- 记录反例和不确定性。

验收：结论能够回到具体数据变化，而不是只比较最终分数。

### 第 16 周：毕业与开源

- 整理端到端 Mini-LLM 数据工厂；
- 补齐 Tutorial、How-to、Reference、Explanation；
- 发布可复现报告；
- 邀请另一位数据工程师独立复现。

毕业标准：新学习者仅依赖仓库文档，就能复现实验并解释一条文档如何影响训练数据和模型指标。

## 加速方案

已有机器学习基础且每周可投入 15 小时以上时，可以合并相邻两周，形成 8 周强化路线。不得省略数据血缘、测试、实验指标和证据边界。
