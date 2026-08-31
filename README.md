# DeepSeek AI Data Engineering Journey

[![CI](https://github.com/xza-m/deepseek-ai-data-engineering-journey/actions/workflows/ci.yml/badge.svg)](https://github.com/xza-m/deepseek-ai-data-engineering-journey/actions/workflows/ci.yml)

一个面向传统数据工程师的开源学习项目。我们以 DeepSeek 的
[smallpond](https://github.com/deepseek-ai/smallpond)、
[3FS](https://github.com/deepseek-ai/3FS) 和
[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
为主要技术依托，通过真实、可复现的本地实验，建立从数据加工到 LLM 训练数据系统的完整知识体系。

> 项目的目标不是追逐新名词，而是让数据工程师能够对训练数据质量、数据血缘、模型效果和系统吞吐共同负责。

## 适合谁

- 熟悉 SQL、ETL、数仓分层、任务调度或数据治理的数据工程师；
- 希望理解 Token、训练样本、DataLoader、Checkpoint 和分布式训练的人；
- 准备进入 AI Data、训练平台、AI Infra 或大模型数据治理方向的人。

你不需要先掌握 CUDA、RDMA 或大模型训练框架。项目会从一条文档如何变成训练 Token 开始。

## 学习主线

```text
传统数据工程
    → 文档、样本、Token、Sequence 与 Loss
    → 清洗、去重、质量评估和数据混合
    → Tokenization、Packing、Sharding 与 Manifest
    → smallpond / DuckDB / Ray 分布式数据加工
    → DataLoader、Checkpoint 与分布式训练
    → 3FS、NVMe、RDMA 与 AI 存储
    → 数据版本—训练—评测—数据改进闭环
```

完整能力地图见 [知识地图](docs/knowledge-map.md)，学习节奏见 [16 周路线](ROADMAP.md)。

## 5 分钟开始第一个实验

本地要求：macOS 或 Linux、Git，以及可以安装 Python 3.11 的 `uv`。

```bash
git clone https://github.com/xza-m/deepseek-ai-data-engineering-journey.git
cd deepseek-ai-data-engineering-journey

make setup
make lab00
make lab01
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

开始前建议依次阅读：

1. [Lab 00：环境与证据边界](docs/tutorials/00-environment.md)
2. [Lab 01：从文档到训练 Token](docs/tutorials/01-document-to-token.md)
3. [数据产物契约](docs/reference/project-contracts.md)

## 课程结构

| 阶段 | 主题 | 核心产物 | 状态 |
| --- | --- | --- | --- |
| 0 | 环境、证据等级、项目方法 | 环境报告 | 可运行 |
| 1 | 文档到 Token | Manifest、Tokenizer、Sequence | 可运行 |
| 2 | 清洗、质量与近似去重 | 质量报告、去重索引 | 规划中 |
| 3 | 数据混合、Packing 与 Sharding | 可复现训练数据版本 | 规划中 |
| 4 | smallpond、Partition 与 Shuffle | 分布式加工 Pipeline | 规划中 |
| 5 | DataLoader 与 Tiny LLM | Loss、Tokens/s、等待时间 | 规划中 |
| 6 | Checkpoint 与恢复 | 可恢复训练实验 | 规划中 |
| 7 | 3FS 架构与 I/O | 设计推演、可选 Linux 集群实验 | 规划中 |
| 8 | 数据—模型反馈闭环 | 数据版本 A/B 评测 | 规划中 |

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
