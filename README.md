# DeepSeek AI Data Engineering Journey

[![CI](https://github.com/xza-m/deepseek-ai-data-engineering-journey/actions/workflows/ci.yml/badge.svg)](https://github.com/xza-m/deepseek-ai-data-engineering-journey/actions/workflows/ci.yml)

一个面向**在职传统数据工程师**的开源职业转型项目。它服务于仍以 SQL、ETL、数仓建模、
数据开发和任务调度为主要工作的工程师，也服务于已经接触 Agent 或 LLM 应用、但希望继续向
Data + AI 底层深入的人。我们以 DeepSeek 的
[smallpond](https://github.com/deepseek-ai/smallpond)、
[3FS](https://github.com/deepseek-ai/3FS) 和
[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
为主要技术锚点，通过真实、可复现的本地项目，建立从传统数据工程到 LLM 训练数据系统的完整知识体系。

> 项目的目标不是教你调用更多 AI API，而是让你能够对训练数据质量、样本血缘、模型效果和系统吞吐共同负责。

## 为什么做这个项目

传统数据工作正在被 AI 重塑。只会写 SQL、维护 ETL 或交付报表的岗位空间会被压缩，但数据工程师已有的
建模、批处理、质量、血缘、调度和生产化能力并没有过时。真正需要完成的是一次责任升级：

```text
交付一张正确的表
    → 交付一个可训练、可追溯的数据版本
    → 证明这个数据版本如何影响模型质量、训练效率和工程成本
```

这个仓库不是为零基础学生设计的课堂，也不是 RAG、Prompt 或 Agent 应用开发合集。它是一条可以与日常工作
并行推进的工程实践路线：用 16 周完成一个持续演进的 Mini-LLM Data Factory，并把每个阶段沉淀成可以复现、
讲解和开源的职业作品。

## 适合谁

- 以 SQL、ETL、DataWorks、Spark、Flink 或数仓建模为主要技能的数据开发；
- 日常已经使用 AI、甚至开发过 Agent，但对训练数据与训练系统仍缺少系统认识的工程师；
- 希望进入 AI Data、训练数据平台、数据治理或 AI Infra 方向的在职工程师；
- 希望用一个真实开源项目证明能力，而不是只积累课程证书的人。

默认你已经会 SQL，理解基本的数据加工和工程交付。你不需要先掌握 CUDA、RDMA 或大模型训练框架；
项目会从一条文档如何变成训练 Token 开始，再逐步进入模型反馈、规模化计算和 AI 存储。

项目当前不以以下目标为主线：

- 零基础编程或 SQL 入门；
- Agent 编排、Prompt 技巧或应用层框架大全；
- 从零训练具有实际通用能力的大模型；
- 把所有参与者培养成 CUDA、分布式训练或存储内核开发者。

## 学习主线

```text
传统数据工程
    → 文档、样本、Token、Sequence 与 Loss
    → 最小 Tiny LLM 与第一次数据版本 A/B
    → 清洗、去重、质量评估、数据混合与治理
    → Tokenization、Packing、Sharding 与 Dataset Manifest
    → smallpond / DuckDB / Ray 分布式数据加工
    → DataLoader、Checkpoint 与分布式训练
    → 3FS、NVMe、RDMA 与 AI 存储
    → 正式的数据版本—训练—评测—归因—数据改进闭环
```

完整能力地图见 [知识地图](docs/knowledge-map.md)，学习节奏见 [16 周路线](ROADMAP.md)。

路线采用螺旋式学习：先用小数据接通完整链路，再依次增加数据质量、规模、故障和基础设施复杂度。
每个阶段都回到同一个问题——这次数据或系统变化，是否真实影响了模型指标、训练效率或工程成本？

## 开始第一条端到端链路

本地要求：macOS 或 Linux、Git，以及可以安装 Python 3.11 的 `uv`。

```bash
git clone https://github.com/xza-m/deepseek-ai-data-engineering-journey.git
cd deepseek-ai-data-engineering-journey

make setup
make lab00
make lab01
make lab02
make validate
```

`make lab01` 会执行一条真实的最小训练数据链路：

```text
JSONL 文档
→ Unicode/空白规范化
→ 内容哈希与精确去重
→ 训练 Byte-level BPE Tokenizer
→ 添加 BOS/EOS
→ Document Packing
→ 固定长度训练 Sequence
→ Dataset Manifest 与血缘
```

输出位于本地 `artifacts/lab01/`：

- `normalized_documents.jsonl`：规范化、去重后的文档；
- `tokenizer.json`：本次实验训练的 Tokenizer；
- `sequences.jsonl`：固定长度的 `input_ids`、`labels`、`loss_mask` 和来源血缘；
- `manifest.json`：输入、配置、哈希、Token 数和 Packing 指标。

`make lab02` 会继续把这些 Sequence 送入一个本地 Tiny Transformer：

```text
Dataset Manifest
→ PyTorch Dataset / DataLoader
→ Causal Transformer
→ Masked Cross Entropy Loss
→ Checkpoint
→ Run Manifest
```

输出位于本地 `artifacts/lab02/`：

- `checkpoint.pt`：模型和优化器状态；
- `run_manifest.json`：Dataset Fingerprint、训练配置、拆分、Loss、吞吐和模型状态哈希。

开始前建议依次阅读：

1. [Lab 00：环境与证据边界](docs/tutorials/00-environment.md)
2. [Lab 01：从文档到训练 Token](docs/tutorials/01-document-to-token.md)
3. [Lab 02：Tiny LLM 基线训练](docs/tutorials/02-tiny-llm-baseline.md)
4. [数据产物契约](docs/reference/project-contracts.md)

## 项目演进结构

| 阶段 | 主题 | 核心产物 | 状态 |
| --- | --- | --- | --- |
| 0 | 环境、证据等级、项目方法 | 环境报告 | 可运行 |
| 1 | 文档到 Token | Manifest、Tokenizer、Sequence | 可运行 |
| 2 | Tiny LLM 基线 | Loss、Tokens/s、Checkpoint | 可运行 |
| 3 | 最小数据版本 A/B | 受控变量实验报告 | 规划中 |
| 4 | 质量、污染与近似去重 | 质量报告、去重索引 | 规划中 |
| 5 | 数据混合、Packing 与 Sharding | 可复现训练数据版本 | 规划中 |
| 6 | smallpond、Partition 与 Shuffle | 分布式加工 Pipeline | 规划中 |
| 7 | DataLoader 与 I/O | 等待时间和吞吐诊断 | 规划中 |
| 8 | Checkpoint 与恢复 | 可恢复训练实验 | 规划中 |
| 9 | 3FS 架构与 I/O | 设计推演、可选 Linux 集群实验 | 规划中 |
| 10 | 正式反馈闭环与毕业项目 | 数据—模型归因报告、复现脚本 | 规划中 |

“规划中”代表学习目标和验收已定义，不代表代码已经实现。我们不会用占位页面冒充完成。

## 文档导航

项目使用 [Diátaxis](https://diataxis.fr/) 组织文档：

- [Tutorials](docs/tutorials/)：按顺序完成的学习实验；
- [How-to](docs/how-to/)：解决具体问题的操作指南；
- [Reference](docs/reference/)：数据契约、术语和阅读索引；
- [Explanation](docs/explanation/)：DeepSeek 技术栈和能力迁移原理。

## 项目边界

首版明确不做：

- 在 macOS 上伪装完成生产级 3FS 部署；
- 训练一个以参数规模为目标的“大模型”；
- 提前建设 Web 平台、规则引擎或通用任务系统；
- 把 Agent 应用开发包装成 AI 数据工程主线；
- 把本地小数据实验结论包装成 PB 级生产结论。

项目将严格区分四类证据：原理理解、本地实验、集群基准和生产验证。详见
[如何运行质量门禁](docs/how-to/run-quality-gates.md)。

## 工程原则

- **KISS**：每个阶段先完成一条可运行、可解释的主路径；
- **YAGNI**：没有真实需求和实验数据前，不增加平台化抽象；
- **SOLID**：数据读取、质量处理、Tokenization、Packing 和产物验证职责分离；
- **DRY**：数据契约、指标定义和证据等级只维护一份。

## 参与贡献

欢迎数据工程师、算法工程师和 AI Infra 工程师共同完善实验。提交前请阅读
[CONTRIBUTING.md](CONTRIBUTING.md)。所有实验必须包含明确假设、可执行命令、指标、测试和证据边界。

## License

[MIT](LICENSE)
