# DeepSeek AI Data Engineering Journey

一个面向传统数据工程师的开源学习项目，以 DeepSeek 的 smallpond、3FS 与相关论文为主要技术依托，通过真实的本地实验，建立从数据工程到 AI 数据工程的完整知识体系。

## 项目目标

- 理解文档、样本、Token、Sequence、Batch 与训练 Loss 的完整链路。
- 建设可复现的 Mini-LLM 训练数据工厂。
- 通过 smallpond 学习面向共享存储的分布式数据处理。
- 通过 3FS 学习 AI 训练与推理场景下的高性能存储设计。
- 将数据质量、数据血缘、模型效果和系统性能纳入统一验收。
- 沉淀一条其他数据工程师可以独立复现的开源学习路径。

## 学习主线

```text
传统数据工程
    → LLM 数据语义
    → 数据清洗、去重与质量评估
    → Tokenization、Packing 与 Sharding
    → smallpond 分布式计算
    → DataLoader 与分布式训练
    → 3FS AI 存储架构
    → 数据质量与模型效果反馈闭环
```

## 当前状态

项目正在初始化。第一阶段将交付：

1. AI 数据工程知识地图与 16 周学习路线；
2. 可复现的本地开发环境；
3. 从原始文档到训练 Token 的第一个端到端实验；
4. 统一的实验、指标、Manifest 与验收规范。

## 设计原则

- **KISS**：优先完成一条可运行、可解释的端到端主线。
- **YAGNI**：首版不建设通用平台、规则引擎或生产级 3FS 集群。
- **SOLID**：清洗、去重、Tokenization、Packing、Manifest 与训练适配职责分离。
- **DRY**：数据契约、质量指标和版本定义集中维护。

## 参考项目

- [DeepSeek smallpond](https://github.com/deepseek-ai/smallpond)
- [DeepSeek 3FS](https://github.com/deepseek-ai/3FS)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

## 开源计划

仓库目前处于个人学习与验证阶段。完成首轮可复现实验和文档验收后，将开放给更多数据工程师学习与贡献。
