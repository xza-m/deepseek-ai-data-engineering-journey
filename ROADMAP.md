# 自主节奏能力关卡路线

这是一条面向在职传统数据工程师的 Data + AI 转型路线。它不规定学习周数，也不以打卡次数作为进度。路线只保留必要的知识依赖顺序，用可运行实验、可检查产物和可解释证据判断是否晋级。

先阅读 [从这里开始](START_HERE.md)。如果你已经有相关经验，可以使用“证据定位”找到第一个尚未通过的关卡；如果尚未形成系统认知，建议从起点关依次完成。

## 最终目标

传统数据工程通常对表、任务、指标和 SLA 负责。AI 数据工程师还需要对模型真正消费的数据单位、训练供给效率以及数据变化带来的模型结果负责：

```text
原始文档
→ 可治理的数据版本
→ Token / Packed Sequence / Shard
→ DataLoader / 训练 / Checkpoint
→ 固定评测
→ 失败归因
→ 下一版数据
```

完成路线后，你应当能够独立交付、解释和复现一个 Mini-LLM Data Factory，并能区分以下四类证据：

- `论文`：技术报告或论文明确描述的机制与公开指标；
- `源码`：固定提交中能够定位的设计和实现；
- `本地`：本仓库命令实际运行得到的产物与指标；
- `推演`：由证据映射出的架构判断，尚未经过目标环境验证。

完成路线不等于成为算法研究员、CUDA 工程师或分布式存储内核专家。目标是成为能连接数据、计算、存储、训练和评测，并能基于瓶颈做工程取舍的 AI 数据工程师。

## 路线总览

`Lab 00～10` 是稳定的实验编号，用于定位命令和产物，不表示天数、周数或完成速度。

| 能力关卡 | 核心问题 | 本地实践 | DeepSeek / LLM 锚点 | 晋级证据 |
| --- | --- | --- | --- | --- |
| 起点关 | 什么才算 AI 数据工程证据 | Lab 00 | 训练数据责任边界 | 环境报告与证据分类 |
| 语义关 | 文档怎样变成 Loss，数据变化怎样被模型看见 | Lab 01～03 | DeepSeek-V3 数据构造、Tokenizer、训练预算 | Dataset / Run / Experiment Manifest |
| 源码桥 | 官方论文、smallpond、3FS 如何映射到本地链路 | Bridge 01 | DeepSeek-V3、smallpond、3FS 固定快照 | 工作负载卡、调用链、架构评审 |
| 治理关 | 如何交付可训练、可追溯的数据版本 | Lab 04～05 | 去冗余、多样性、Packing、Shard | 质量报告、Dataset Card、样本血缘 |
| 规模关 | 数据加工怎样扩展并在失败后恢复 | Lab 06 | smallpond 的 Partition、Shuffle、Task、物化 | 执行图、倾斜指标、恢复证据 |
| 供给关 | 训练为何等数据，如何精确续训 | Lab 07～08 | DataLoader、并行策略、Checkpoint | Wait/吞吐 Profile、精确恢复证据 |
| 存储关 | 什么时候值得引入 AI 共享存储 | Lab 09 | 3FS 元数据、CRAQ、FUSE、USRBIO | I/O 基线与 Go / No-Go |
| 反馈关 | 如何证明数据版本影响模型并可被他人复现 | Lab 10 | 固定预算实验、评测与数据迭代 | 多 Seed A/B、失败血缘、复现 Manifest |

## 统一晋级规则

每个关卡都必须留下 [学习证据记录](docs/reference/learning-evidence-template.md)，并同时满足：

1. 对应命令成功，核心产物通过验证器；
2. 亲自检查 Manifest、指标和至少一个失败或反常样例；
3. 能解释它与传统 SQL、数仓、调度、质量或血缘能力的迁移关系；
4. 能指出一个工程取舍和一个结果不能证明的结论；
5. 不看教程也能回答关卡的晋级问题；
6. 输入、配置、代码、数据产物和结论可以追溯。

命令跑通但解释不清，或者理解概念却没有复现证据，都不算通过。

## 起点关：角色、环境与证据边界

### 要解决的问题

传统数据任务“产出正确”与训练数据“对模型有效”有什么区别？本地小数据、官方集群基准和生产验证为什么不能混为一谈？

### 从旧能力迁移

- 从任务成功与表级校验，迁移到数据—训练—评测证据链；
- 从表/行，扩展到 Document、Sample、Token、Sequence 和 Shard；
- 从“结果看起来合理”，迁移到可复现配置、哈希和证据边界。

### 实践与产物

- 完成 [Lab 00](docs/tutorials/00-environment.md)；
- 运行 `make lab00`；
- 检查环境报告，确认哪些能力可以本地验证、哪些需要 Linux、GPU、NVMe 或 RDMA；
- 创建第一份个人学习证据记录。

### 晋级门槛

闭卷解释：为什么在 macOS 上运行的 Tiny LM 不能证明 DeepSeek 生产集群的吞吐，但仍可以验证数据版本、血缘和受控实验方法？

## 语义关：Document → Token → Sequence → Loss

### 要解决的问题

模型消费的训练数据到底是什么？数据版本发生变化时，怎样让模型在受控条件下“看见”这种变化？

### 从旧能力迁移

| 传统数据工程 | AI 数据工程扩展 |
| --- | --- |
| 一行记录 | 有边界、有顺序的 Token Sequence |
| 主键和分区 | Document ID、Content Hash、Dataset Fingerprint |
| 表级血缘 | Source → Token Span → Sequence → Run → Metric |
| 作业参数 | Tokenizer、Seed、Token Budget、Model Config |
| 数据验收 | 固定评测集上的模型反馈 |

### DeepSeek / LLM 锚点

- DeepSeek-V3 Technical Report 的 Data Construction、Tokenizer、Sequence Length 与训练预算；
- Causal Language Modeling 中 Input、Label、Loss Mask 的关系；
- Packing 提升利用率时必须保留的文档边界与血缘。

### 实践与产物

依次完成：

1. [Lab 01：从文档到训练 Token](docs/tutorials/01-document-to-token.md)：规范化、精确去重、Tokenizer、Packing、Dataset Manifest；
2. [Lab 02：Tiny LLM 基线](docs/tutorials/02-tiny-llm-baseline.md)：DataLoader、最小 Transformer、Loss、Checkpoint、Run Manifest；
3. [Lab 03：受控数据 A/B](docs/tutorials/03-controlled-data-ab.md)：固定模型、初始化、Token 预算和评测集，只改变一个数据变量。

### 必须检查的证据

- 同一输入和配置能否生成相同 Dataset Fingerprint；
- 一个 Sequence 能否追溯到来源文档和 Token Span；
- Run Manifest 是否引用准确的数据版本、Seed 和训练预算；
- A/B 是否真正只改变一个变量，是否保留无差异或反常结果。

### 晋级门槛

用自己的话完整解释：一份 JSONL 文档如何变成 Loss；为什么相同 Token 预算不等于相同数据价值；一次 A/B 需要固定哪些变量才能把差异归因到数据。

## 源码桥：DeepSeek 官方证据与本地链路

### 要解决的问题

如何避免只会复述架构名词？怎样把论文机制、源码职责、本地实验和架构推演连成一条有边界的证据链？

### 学习锚点

- DeepSeek-V3：从数据构造与训练并行反推数据系统工作负载；
- smallpond：追踪 `DataFrame → Logical Node → Optimizer → Planner → Task`；
- 3FS：追踪 Metadata Service / FoundationDB 与 Client / Storage / CRAQ 两条路径；
- 本地项目：用 Dataset、Run、Experiment Manifest 映射职责。

### 实践与产物

完成 [Bridge 01：DeepSeek 源码与论文连接](docs/tutorials/bridge-01-deepseek-source-study.md)，产出：

- DeepSeek-V3 工作负载卡；
- smallpond Node、Task、Partition 和中间文件调用链；
- 3FS 元数据路径与文件数据路径图；
- 标注“已实现 / 源码理解 / 待实验”的本地架构图；
- 一份基于真实瓶颈的 Go / No-Go 评审。

### 晋级门槛

闭卷验收至少达到 `16/20`，且没有把官方基准、源码理解或架构推演冒充本地验证。能够说明为什么 smallpond 和 3FS 是数据平面的技术锚点，但都不替代 Dataset Catalog、质量、许可与样本血缘。

## 治理关：质量、去重、混合与数据版本

### 要解决的问题

数据库唯一键、非空和 Schema 正确不足以说明语料可训练。怎样识别语义近似重复、训练/评测污染、低质量内容和不合理分布，并将决策固化成可追溯数据版本？

### 从旧能力迁移

- 从唯一键迁移到 Shingling、MinHash/LSH 和相似簇；
- 从字段质量迁移到内容、语言、安全、PII、许可和可训练性；
- 从表分区迁移到领域混合、Packing、Shard 分布和样本顺序；
- 从 ETL 版本迁移到规则、Tokenizer、混合权重和来源共同定义的 Dataset Version。

### 实践与产物

1. [Lab 04](docs/tutorials/04-quality-dedup-contamination.md)：建立小规模真值集，测量近似去重 Precision/Recall，检查训练/评测污染；
2. [Lab 05](docs/tutorials/05-mixing-packing-sharding.md)：执行确定性混合、Packing 和 Sharding，生成质量报告与 Dataset Card；
3. 对比治理前后的有效 Token、领域分布、Shard 倾斜和血缘覆盖率。

### 晋级门槛

任取一个训练样本，能够追溯到来源、许可、处理规则、Tokenizer、混合权重和 Shard；能够说明阈值误判成本，而不只报告“过滤了多少数据”。

## 规模关：Partition、Shuffle、倾斜与恢复

### 要解决的问题

当单机 Pipeline 不再满足数据规模与时效时，如何扩展计算，同时保留数据版本契约、可观察性和失败恢复边界？

### 从旧能力迁移

- 从 SQL 执行计划迁移到 Lazy DAG、Logical Node 和物理 Task；
- 从数仓分区迁移到 Hash Partition、Shuffle 和训练 Shard；
- 从任务重跑迁移到中间物化、幂等输出和精确失败状态；
- 从平均耗时迁移到分区分布、任务 P50/P99 和热点 Key。

### smallpond 锚点

smallpond 展示了“强单机执行引擎 + 简洁分布式计划”的数据加工思路。DuckDB 负责 Task 内的 SQL/列式计算，smallpond 负责跨 Partition 的计划、调度、物化和恢复边界。它是学习锚点，不是默认替换 Spark 的结论。

### 实践与产物

- 完成 [Lab 06](docs/tutorials/06-partition-shuffle-recovery.md)；
- 保留逻辑计划、物理任务、分区大小和中间文件证据；
- 人工制造热点 Key、小文件或任务失败；
- 只做一个最小修复，针对原问题重新测量。

### 晋级门槛

能够解释为什么 SQL Join 语义正确不等于分布式执行高效；为什么 Shuffle 文件既是 I/O 成本也是恢复边界；为什么扩展计算层不应改变 Dataset Manifest 的语义。

## 供给关：DataLoader、Checkpoint 与训练契约

### 要解决的问题

数据已加工完成，为什么训练仍可能等待？训练中断后“能继续跑”为什么不等于精确恢复？并行策略又如何改变数据和存储压力？

### 从旧能力迁移

- 从数据产出 SLA 迁移到 DataLoader Wait、Samples/s 和 Tokens/s；
- 从文件大小与分区数迁移到随机读、预取、Worker、缓存和 Batch 组装；
- 从任务断点续跑迁移到模型、优化器、随机状态和数据游标共同恢复；
- 从计算资源配置迁移到 DP、TP、PP、EP 与数据分发/通信关系。

### 实践与产物

1. [Lab 07](docs/tutorials/07-dataloader-io.md)：比较数据供给配置，拆分存储、解压、Tokenization 和 Batch 组装等待；
2. [Lab 08](docs/tutorials/08-checkpoint-exact-recovery.md)：注入训练中断，验证 Checkpoint、样本顺序与 Loss 连续性；
3. 建立并行策略—Global/Micro Batch—数据供给—通信—Checkpoint 的关系图。

没有 GPU 也可以完成主线：使用 CPU 训练循环的 Wait 和 Tokens/s 建立本地证据；多 GPU DDP 属于条件化扩展。

### 晋级门槛

能够用指标判断瓶颈来自数据格式、文件组织、CPU 处理还是存储；能够证明恢复后的数据顺序与训练状态是否一致；能够解释训练并行如何改变数据供给和 Checkpoint 压力。

## 存储关：训练数据平面与 3FS 取舍

### 要解决的问题

顺序扫描、DataLoader 随机读、Shuffle 中间文件和并行 Checkpoint 是不同工作负载。怎样先证明存储瓶颈，再判断是否需要 3FS 或其他共享存储？

### 3FS 锚点

- 无状态 Metadata Service 与 FoundationDB 事务；
- 文件布局、Chunk、Chain 与 CRAQ 的写入/读取语义；
- FUSE 兼容路径与 USRBIO 异步路径；
- Linux、NVMe、NUMA、RDMA 等真实部署前提。

### 实践与产物

- 完成 [Lab 09](docs/tutorials/09-3fs-storage-workloads.md)；
- 描述四类 I/O 工作负载并建立本地 SSD 基线；
- 测量吞吐、随机读延迟、并发、DataLoader Wait 和 Checkpoint 时间；
- 输出当前架构与 3FS 的工作负载适配矩阵；
- 区分官方集群基准、本地模拟和自己的目标环境结果。

### Go / No-Go 规则

- **Go**：已有明确共享随机读、并行写或恢复瓶颈，并具备目标环境与基准方案；
- **Conditional**：需要 Linux / NVMe / RDMA 集群才能验证的部分，保留为 Infra 扩展；
- **No-Go**：仅因官方性能数字或技术热度替换现有存储；
- **No-Go**：用 3FS 替代 Catalog、湖仓表格式、质量、权限、许可或样本血缘。

### 晋级门槛

能够画出 3FS 元数据与数据路径，解释 CRAQ 的目标和边界，并用自己的 I/O 指标给出可审查的 Go / No-Go，而不是从架构先进性直接跳到选型结论。

## 反馈关：数据版本—模型结果—下一版改进

### 要解决的问题

怎样把所有能力收敛为一次可信的数据版本实验，并让另一位工程师只依赖仓库和证据完成复现？

### 实践与产物

完成 [Lab 10](docs/tutorials/10-graduation-feedback-loop.md)：

- 选择去重、质量、混合、Packing 或 Sharding 中的一个变量；
- 固定模型、Tokenizer、初始化、训练 Token 预算和评测集；
- 使用多个 Seed，报告均值、方差、反例和不确定性；
- 从失败评测样例追溯到来源文档、处理规则、Dataset、Run 和 Checkpoint；
- 形成下一版数据假设，但不同时修改多个变量；
- 生成毕业复现 Manifest，请另一位工程师独立复现。

### 毕业交付物

1. 可版本化的数据 Pipeline、Dataset Manifest 和样本血缘；
2. Tiny LLM 训练、Run Manifest 和可恢复 Checkpoint；
3. 一次质量、计算、供给或 I/O 的性能诊断；
4. 一次多 Seed 受控数据 A/B 与失败归因；
5. smallpond / 3FS 的源码阅读或架构评审证据；
6. 一篇严格区分论文、源码、本地和生产证据的技术报告；
7. 他人独立复现记录。

### 毕业门槛

运行最终复现与质量门禁：

```bash
make journey
make validate
```

命令通过只是必要条件。你还需要在技术评审、面试或真实工作中解释每个设计选择，并能从一个模型指标反向追溯到具体数据版本和处理决策。

## 完成主线后的纵深方向

先完成反馈关，再根据真实工作瓶颈选择一个方向，不需要同时学习所有底层技术。

### 数据治理与 Data-Centric AI

- 许可、隐私、PII 与删除传播；
- 语义质量、弱监督、LLM Judge 与人工真值集；
- 数据混合、课程学习、难例挖掘和评测污染；
- Dataset Catalog、版本治理与跨模型血缘。

### AI Infra 与训练数据平台

- smallpond / Ray 多 Worker 执行与调度；
- 多 GPU DDP、MoE 数据分发和通信分析；
- MinIO、NFS、对象存储与共享文件系统对比；
- 3FS 测试集群、NVMe、NUMA、RDMA 与故障实验。

### 数据—模型评测与迭代

- 领域评测集与切片指标；
- 失败样例到数据来源和规则的自动回溯；
- 数据版本实验设计、统计显著性和可复现报告；
- 训练前质量信号与训练后模型反馈的联合分析。

## 论文与源码的正确使用方式

论文和源码不是独立的阅读进度，而是当前实验的设计输入。每次只回答六个问题：

```text
问题：当前关卡要回答什么？
机制：论文或源码采用了什么设计？
约束：设计依赖哪些规模、硬件和工作负载？
实验：本地可以验证哪一部分？
边界：哪些结论只能引用，不能声称自己验证过？
迁移：它与传统数据工程的哪个概念相连？
```

默认入口见 [DeepSeek 源码与论文阅读索引](docs/reference/reading-list.md)。不要为了“读完项目”逐行通读源码；先带着当前关卡的问题定位职责，再决定是否深入。

## 这条路线如何保持可持续

- 学习速度由个人决定，关卡标准不变；
- 工作繁忙时可以暂停，但保留 Commit、Manifest 和证据记录，回来后从证据恢复；
- 遇到失败先记录原始现象，只做一个最小修改并复测；
- 没有目标环境时明确标记为推演，不伪造集群结论；
- 所有公开内容先通过密钥、内部路径、数据许可和可复现性检查。

项目进度的真正定义不是“学到了第几个周期”，而是已经能够对哪一段数据—模型链路承担责任。
