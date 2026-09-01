# 从这里开始：自主节奏学习指南

这不是一张按周打卡的课程表，而是一条由**能力证据**驱动的职业转型路径。

你可以一天完成一个实验，也可以用多个工作周期消化一个关卡。是否继续前进只取决于一件事：你能否独立运行、解释并复现当前关卡的产物，而不是已经学习了多久。

## 这条路径适合谁

它面向已经工作的传统数据开发、数仓建模工程师和 SQL 重度使用者。你可能熟悉 ETL、Spark、Flink、DataWorks、调度、质量或血缘，也可能已经做过 Agent 和 LLM 应用，但尚未系统理解训练数据如何影响模型效果、训练吞吐和工程成本。

这条路径不会把你培养成算法研究员或存储内核工程师。它要帮助你把已有的数据工程能力扩展到以下责任范围：

```text
来源文档
→ 可治理的数据版本
→ Token / Sequence / Shard
→ 训练与 Checkpoint
→ 固定评测
→ 数据归因与下一版改进
```

## 两种进入方式

### 方案 A：标准主线（推荐）

从“起点关”开始，依次完成全部关卡。它适合第一次系统建设 Data + AI 知识体系的人，也最适合作为公开学习记录。

### 方案 B：证据定位

如果你已经做过训练数据、分布式计算或模型训练，可以先定位自己的起点：

1. 阅读 [自主节奏能力关卡路线](ROADMAP.md) 中的晋级门槛；
2. 不参考答案，解释该关卡的核心问题；
3. 运行对应验证命令，检查真实产物；
4. 能提交完整学习证据才算通过，否则从该关卡开始。

“了解过”“工作中接触过”和“看过论文”不能代替证据。证据定位只是减少重复学习，不会跳过验收。

## 能力关卡地图

```text
起点关：角色、环境与证据边界
  ↓
语义关：Document → Token → Sequence → Loss → 第一次数据 A/B
  ↓
源码桥：DeepSeek-V3 × smallpond × 3FS × 本地证据
  ↓
治理关：去重、污染、质量、混合、Packing 与 Sharding
  ↓
规模关：Partition、Shuffle、倾斜与失败恢复
  ↓
供给关：DataLoader、I/O、Checkpoint 与训练并行契约
  ↓
存储关：训练 I/O 工作负载与 3FS Go / No-Go
  ↓
反馈关：多 Seed A/B、失败归因、复现与职业作品
```

关卡顺序表达知识依赖，不表达日历周期。完整的目标、实验、产物和验收标准见 [ROADMAP.md](ROADMAP.md)。

## 每个关卡都执行同一份学习合同

1. **提出问题**：本关要解决哪个真实的数据或训练问题？
2. **连接旧知**：它与 SQL、数仓、调度、质量或血缘中的什么概念相连？
3. **读取证据**：只读解决当前问题所需的论文、官方文档或源码路径。
4. **运行实验**：执行对应 Lab，不用阅读代替实操。
5. **检查产物**：查看 Manifest、指标、失败样例和血缘，不只看命令退出码。
6. **写下边界**：明确本地结果不能证明哪些集群或生产结论。
7. **通过验收**：能脱离教程解释、复现并回答晋级问题后再进入下一关。

每次用 [学习证据模板](docs/reference/learning-evidence-template.md) 留下一份记录。个人草稿放在被忽略的 `artifacts/learning-records/`，成熟后再整理为公开文章或案例。

## 第一次启动

本地需要 macOS 或 Linux、Git，以及能够安装 Python 3.11 的 [`uv`](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/xza-m/deepseek-ai-data-engineering-journey.git
cd deepseek-ai-data-engineering-journey

make setup
make lab00

mkdir -p artifacts/learning-records
cp docs/reference/learning-evidence-template.md \
  artifacts/learning-records/entry-gate.md
```

然后完成 [Lab 00：环境与证据边界](docs/tutorials/00-environment.md)，检查 `artifacts/lab00/` 中的真实输出。通过起点关后，再运行 `make lab01`，不要一开始就执行全部实验。

## 学习模式与验收模式

### 学习模式

一次只运行当前关卡对应的 Lab：

```bash
make lab00
make lab01
make lab02
```

每次运行后先读教程、Manifest 和失败样例，再继续下一项。学习的核心不是把命令全部跑绿，而是建立“机制—实验—证据—边界”的解释能力。

### 最终复现模式

`make journey` 和 `make validate` 会连续运行完整主线，适合以下场景：

- 完成全部关卡后的独立复现；
- 维护者验证仓库；
- CI 或发布前质量门禁。

它们不是初学入口。过早批量执行会得到一批产物，却掩盖关卡之间的因果关系。

## 何时可以进入下一关

必须同时满足：

- 对应命令通过，且你检查过核心产物；
- 能解释关键机制及其与传统数据工程的迁移关系；
- 能指出一个失败样例、一个工程取舍和一个证据边界；
- 学习记录包含输入版本、命令、指标和结论；
- 不依赖教程也能回答该关卡的晋级问题。

如果命令通过但解释不清，仍停留在当前关；如果解释正确但没有可复现产物，也仍未通过。

## 接下来读什么

1. [自主节奏能力关卡路线](ROADMAP.md)：确认当前关卡、目标和晋级门槛；
2. [Lab 教程](docs/tutorials/)：完成当前关卡的引导式实践；
3. [Lab 索引](labs/README.md)：查命令、产物和验证器；
4. [DeepSeek 源码与论文阅读索引](docs/reference/reading-list.md)：按问题查证，不做无目标通读；
5. [数据产物契约](docs/reference/project-contracts.md)：理解 Manifest 和血缘的稳定边界。

