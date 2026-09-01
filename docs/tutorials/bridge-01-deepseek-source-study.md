# Bridge 01：DeepSeek 源码与论文连接

这是语义关和治理关之间的一座源码桥。它不增加新的数据处理功能，而是用
[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)、
[smallpond](https://github.com/deepseek-ai/smallpond) 和
[3FS](https://github.com/deepseek-ai/3FS) 回看已经运行过的 Mini-LLM Data Factory。

完成后，你不只需要记住几个架构名词，还要能从官方证据出发，解释数据如何经过计算、存储和训练影响模型指标。

## 适用对象与目标

本教程面向已经完成 Lab 00～03、熟悉 SQL、ETL、数仓或数据平台的在职工程师。五个任务按知识依赖排列，完成周期由个人决定。

最终要交付四类个人学习证据：

1. DeepSeek-V3 工作负载卡；
2. smallpond 源码调用链；
3. 3FS 元数据与数据路径图；
4. 当前本地项目的架构评审和差距清单。

本关明确不做：

- 在 macOS 上部署或模拟生产级 3FS；
- 逐行通读 smallpond 或 3FS 全部源码；
- 把 DeepSeek 的集群指标改写成本地实测结果；
- 提前实现 Lab 04 的近似去重和污染检测；
- 因为看到新系统就决定替换 Spark、对象存储或现有湖仓。

## 证据标记

笔记中的每个重要结论都要标记证据等级：

| 标记 | 含义 | 可以怎么说 |
| --- | --- | --- |
| `论文` | DeepSeek-V3 技术报告明确描述 | “报告说明……” |
| `源码` | 固定提交中的实现或官方设计文档 | “在提交 `52ecc5e` 中可见……” |
| `本地` | 本仓库命令实际运行得到的结果 | “Lab 03 本地结果显示……” |
| `推演` | 根据证据进行的架构映射，尚未运行 | “可以推演……，仍需实验验证” |

不要把四类证据混写。尤其要注意：DeepSeek-V3 报告、smallpond 和 3FS 是可以相互解释的官方材料，但这不自动证明
公开仓库中的特定提交就是 DeepSeek-V3 训练时使用的生产版本。

## 固定阅读快照

本教程在 2026-08-31 核对以下版本。使用固定提交是为了避免上游 `main` 更新后源码路径和笔记互相矛盾。

| 材料 | 固定版本 | 作用 |
| --- | --- | --- |
| DeepSeek-V3 Technical Report | arXiv `2412.19437v2` | 数据构造、训练框架和计算集群 |
| smallpond | [`52ecc5e`](https://github.com/deepseek-ai/smallpond/tree/52ecc5e45535c7448f848bcb45b0da00d9484f81) | 惰性计划、分区、任务与中间物化 |
| 3FS | [`22fca04`](https://github.com/deepseek-ai/3FS/tree/22fca04564c7cc230fd8b9523b8b92864e1dad47) | 共享文件系统、元数据、CRAQ 与原生 I/O |

## 准备本地证据目录

先重新生成并验证 Lab 03 的受控实验：

```bash
make lab03
uv run aide validate-data-ab --output artifacts/lab03
```

学习笔记放在被 `.gitignore` 忽略的 `artifacts/` 中，避免未完成的个人理解直接进入公共文档：

```bash
mkdir -p artifacts/source-study

for task in 01 02 03 04 05; do
  cp docs/reference/source-study-note-template.md "artifacts/source-study/task-${task}.md"
done
```

## 任务一：从 LLM 工作负载反推数据系统

### 阅读范围

只读技术报告中与数据工程直接相关的部分：

- [2.2 Multi-Token Prediction](https://arxiv.org/html/2412.19437#S2.SS2)：理解一个 Token 可能对应多个训练目标；
- [3.1 Compute Clusters](https://arxiv.org/html/2412.19437#S3.SS1)：理解节点内和节点间通信边界；
- [3.2 Training Framework](https://arxiv.org/html/2412.19437#S3.SS2)：理解 PP、EP、DP 如何改变数据和通信；
- [4.1 Data Construction](https://arxiv.org/html/2412.19437#S4.SS1)：理解去冗余、多样性、Packing、FIM 和 Tokenizer。

技术报告公开的关键事实包括：预训练语料为 14.8T Token；数据 Pipeline 在减少冗余时保留多样性；采用文档 Packing；
Tokenizer 是 128K 词表的 Byte-level BPE；预训练的最大 Sequence Length 为 4K。它们都是工作负载定义，不是让本地项目
照抄参数规模。

### 本地对照

查看当前项目把同类概念记录在哪里：

```bash
jq '{config, metrics, pipeline_fingerprint}' artifacts/lab01/manifest.json

jq '{dataset, config, split, metrics, run_fingerprint}' \
  artifacts/lab02/run_manifest.json

jq '{controls, runs, comparison, evidence_boundary, experiment_fingerprint}' \
  artifacts/lab03/experiment_manifest.json
```

在 `task-01.md` 中完成一张对照表：

| 维度 | DeepSeek-V3 报告 | 本地 Lab 00～03 | 不能外推的结论 |
| --- | --- | --- | --- |
| 数据规模 | 14.8T Token | 读取 Manifest 实际值 | 本地效果不能代表万亿 Token 训练 |
| Tokenizer | 128K Byte-level BPE | 读取 `vocab_size_actual` | 压缩效率和多语言能力不可等同 |
| Sequence | 预训练 4K | 默认 64 | I/O 和显存压力不可等同 |
| 训练并行 | PP、EP、ZeRO-1 DP | 单进程 CPU | 不能声称验证了分布式效率 |

### 任务验收

不看资料回答：为什么 AI 数据工程需要同时记录 Document、Token、Sequence 和训练预算，而不能只记录“处理了多少行”？

## 任务二：追踪 smallpond 从 DataFrame 到 Task

### 获取固定源码

源码只放在被忽略的 `work/` 目录，不作为本项目代码提交：

```bash
mkdir -p work/upstream
git clone https://github.com/deepseek-ai/smallpond.git work/upstream/smallpond
git -C work/upstream/smallpond checkout 52ecc5e45535c7448f848bcb45b0da00d9484f81
```

如果目录已经存在，只需检查版本：

```bash
git -C work/upstream/smallpond rev-parse HEAD
```

### 第一条调用链：惰性计划

依次阅读：

1. [`dataframe.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/dataframe.py)：
   `read_parquet`、`partial_sql`、`repartition`、`write_parquet`、`_get_or_create_tasks`；
2. [`logical/node.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/logical/node.py)：
   `DataSourceNode`、`SqlEngineNode`、`HashPartitionNode`、`DataSinkNode`；
3. [`logical/optimizer.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/logical/optimizer.py)：
   相邻节点如何被改写或融合；
4. [`logical/planner.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/logical/planner.py)：
   逻辑节点如何变成按 Partition 展开的 Task；
5. [`execution/task.py`](https://github.com/deepseek-ai/smallpond/blob/52ecc5e45535c7448f848bcb45b0da00d9484f81/smallpond/execution/task.py)：
   Task 如何运行、物化输出和记录完成状态。

用以下命令快速定位，不必从文件第一行开始读：

```bash
rg -n "def (_get_or_create_tasks|compute|repartition|write_parquet)|class (Optimizer|Planner)" \
  work/upstream/smallpond/smallpond

rg -n "class (DataSourceTask|SqlEngineTask|HashPartitionTask|DataSinkTask)|def run_on_ray" \
  work/upstream/smallpond/smallpond/execution/task.py
```

把调用链写成自己的语言：

```text
DataFrame API
→ 构造 Logical Node
→ Action 触发 compute
→ Optimizer 重写逻辑计划
→ Planner 按分区生成 Task
→ Ray 或 Scheduler/Executor 运行 Task
→ 共享 data root 物化 staging/output 和完成状态
```

`DataFrame` 交互路径和 `Driver` 的 Scheduler/Executor 路径都存在，不能简单概括成“smallpond 就是 Ray”。DuckDB 负责单个
Task 内擅长的 SQL/列式计算，smallpond 负责跨 Partition 的计划、调度、物化和恢复边界。

### 第二条调用链：为什么 Join 要先对齐分区

从 `partial_sql` 示例开始，找到两侧输入必须有兼容 Partition 的检查。回答：

- 数据库中的 Join Key 和 smallpond 的 Hash Partition Key 分别承担什么职责？
- 为什么 SQL 语义正确，不代表分布式执行就高效或可执行？
- Shuffle 中间文件为什么既是 I/O 成本，也是失败恢复边界？

### 任务验收

在 `task-02.md` 画出一张包含 Node、Task、Partition 和中间文件的调用链，并为每个箭头附一个源码路径。

## 任务三：追踪 3FS 的元数据路径和数据路径

### 获取固定源码

3FS 仓库较大，本任务只稀疏检出关键目录，不初始化第三方子模块：

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/deepseek-ai/3FS.git work/upstream/3FS

git -C work/upstream/3FS sparse-checkout init --cone
git -C work/upstream/3FS sparse-checkout set \
  docs deploy src/lib/api src/meta src/storage src/client/meta src/client/storage src/mgmtd

git -C work/upstream/3FS checkout 22fca04564c7cc230fd8b9523b8b92864e1dad47
```

### 先读设计，不先钻 C++

按顺序阅读：

1. [`README.md`](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/README.md)：确认目标工作负载；
2. [`docs/design_notes.md`](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/docs/design_notes.md)：
   组件、文件语义、元数据、数据放置和 CRAQ；
3. [`src/lib/api/UsrbIo.md`](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/src/lib/api/UsrbIo.md)：
   FUSE 兼容路径和异步零拷贝路径；
4. [`deploy/README.md`](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/deploy/README.md)：
   用真实硬件前置条件约束证据边界。

### 路径一：元数据

```text
应用 open/create/rename
→ MetaClient
→ 任一无状态 Metadata Service
→ FoundationDB 事务
→ 返回 inode、chunk size、chain table range、shuffle seed 等布局
```

源码入口只需要定位职责：

```bash
rg -n "class (MetaClient|MetaServer|MetaStore)|FoundationDB|transaction" \
  work/upstream/3FS/src/client/meta work/upstream/3FS/src/meta
```

思考：传统 NameNode 把元数据状态放在服务内存中时，无状态 Metadata Service + 事务 KV 的运维和扩展边界有什么不同？

### 路径二：文件数据

```text
Client 根据文件布局计算 Chunk 和 Chain
→ StorageClient 发起读写
→ Storage Service 管理本地 SSD Target/Chunk
→ CRAQ 写入整条复制链、读取可选择副本
```

```bash
rg -n "class (StorageClient|StorageServer|StorageService)|Reliable(Update|Forwarding)|Chunk(Store|Replica)" \
  work/upstream/3FS/src/client/storage work/upstream/3FS/src/storage
```

不要把 “write-all、read-any” 简化成“随便从副本读”。正确性仍依赖 Chain 版本、脏/净状态、Lease、路由信息和失败恢复。

### FUSE 和 USRBIO 的取舍

在 `task-03.md` 完成下表：

| 场景 | 优先接口 | 原因 | 仍需验证的指标 |
| --- | --- | --- | --- |
| 数据准备的大块 Parquet 写入 | FUSE 可作为起点 | 接入简单，可多文件并行写 | 吞吐、并发写、CPU |
| DataLoader 小块随机读 | 评估 USRBIO | 批处理异步 I/O，减少 FUSE 路径开销 | P99、IOPS、Batch Wait |
| 大模型 Checkpoint | 按写入模式评估 | 需要高吞吐并行写 | 写入时间、恢复时间、失败语义 |

### 任务验收

闭卷解释：为什么 3FS 可以提供共享文件数据平面，但不能替代 Dataset Catalog、许可治理、质量规则和样本血缘？

## 任务四：映射到本地 Mini-LLM Data Factory

先定位当前本地职责：

```bash
rg -n "def (build_dataset|train_tiny_lm|run_data_ab|validate_)" \
  src/ai_data_engineering

jq '{artifacts, config, metrics, pipeline_fingerprint}' artifacts/lab01/manifest.json

jq '{dataset, evaluation_dataset, config, metrics, run_fingerprint}' \
  artifacts/lab02/run_manifest.json

jq '{datasets, controls, runs, comparison, experiment_fingerprint}' \
  artifacts/lab03/experiment_manifest.json
```

完成下面的责任矩阵：

| 能力 | 当前实现 | 未来主要学习锚点 | 不能丢失的契约 |
| --- | --- | --- | --- |
| 规范化、精确去重、Tokenize、Packing | `pipeline.py` | Lab 04/05 | Source Hash、Tokenizer Hash、样本血缘 |
| 固定预算训练与评测 | `training.py` | DataLoader、分布式训练 | Dataset Fingerprint、Seed、Token Budget |
| 数据版本 A/B | `experiment.py` | 正式质量归因 | 控制变量、独立评测集、证据边界 |
| 批量分区、Join、Shuffle | 尚未扩展 | smallpond | Partition、Task、物化产物与失败状态 |
| 共享随机读和并行写 | 尚未验证 | 3FS | 文件布局、I/O 指标、恢复语义 |

然后回答三个架构问题：

1. 如果 Lab 06 把本地 Python 逐行处理替换为 smallpond，为什么 Dataset Manifest 仍应保持稳定？
2. 如果 Lab 09 把本地 SSD 换成共享文件系统，哪些路径可以变化，哪些数据版本语义不能变化？
3. 当训练 Tokens/s 下降时，如何区分数据格式、CPU 处理、DataLoader、存储和 GPU 通信问题？

### 任务验收

在 `task-04.md` 输出一张端到端架构图，每层必须标注“已实现”“源码理解”或“待实验”，禁止用一张图把三者冒充成同一证据等级。

## 任务五：做一次架构评审，而不是读书总结

在 `task-05.md` 完成以下评审：

### 1. 工作负载

- 数据规模、文件数量、平均样本大小和访问模式是什么？
- 瓶颈是计算、Shuffle、随机读、Checkpoint 还是治理？
- 目标指标是有效 Token 产出率、任务 P99、DataLoader Wait、Tokens/s 还是恢复时间？

### 2. 候选架构

- 当前单机实现可以继续承担什么？
- 哪个瓶颈出现后才值得引入 smallpond？
- 哪个瓶颈出现后才值得评估 3FS 或其他共享存储？
- 哪些 Catalog、权限、许可、质量和血缘能力必须由独立治理层承担？

### 3. Go / No-Go

在没有基准数据时，默认结论应是：

- **Go**：继续完成本地质量、Manifest 和 A/B 主线；
- **Go with evidence**：用 smallpond 做受控的 Partition/Shuffle 实验；
- **No-Go**：仅因为官方性能数字就替换生产计算或存储架构；
- **Conditional**：具备 Linux、NVMe、RDMA 和明确 I/O 瓶颈后再做 3FS 集群实验。

## 闭卷问题

每题 2 分，总分 20 分。达到 16 分且没有证据等级混淆，才进入 Lab 04。

1. 为什么训练数据的最小责任单位不只是数据库行？
2. 为什么相同训练 Token 预算不代表相同数据价值？
3. smallpond 的 DataFrame 操作在什么时刻从逻辑计划变成真实执行？
4. DuckDB、smallpond 和 Ray/Scheduler 各承担什么职责？
5. Shuffle 中间文件为什么同时影响性能和恢复？
6. 3FS Metadata Service 为什么可以设计成无状态？
7. CRAQ 的 write-all、read-any 想解决什么问题，又不等于什么？
8. FUSE 和 USRBIO 分别适合从什么工作负载开始评估？
9. 为什么 3FS 不替代 Catalog、湖仓表格式、质量和许可治理？
10. 当前 macOS Tiny LM 实验不能证明哪些 DeepSeek 集群结论？

## 晋级门槛

完成源码桥后，你应当能给另一位数据工程师讲清楚：

```text
模型目标和训练并行定义工作负载
→ 数据 Pipeline 构造并版本化 Token/Sequence/Shard
→ smallpond 类计算层扩展 Partition/Shuffle/物化
→ 3FS 类存储层承载共享随机读和并行写
→ DataLoader/训练框架消费数据并产生 Loss、吞吐和 Checkpoint
→ 评测与血缘把结果反馈到下一版数据
```

如果只能复述组件名、无法给出源码路径、本地 Manifest 或证据边界，就先补齐笔记，不要急着进入下一阶段。
