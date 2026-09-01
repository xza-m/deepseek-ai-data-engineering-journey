# 实验索引

这些 Lab 不是彼此独立的课堂作业，而是同一个 Mini-LLM Data Factory 的连续版本。后一个 Lab 必须消费或引用
前一个 Lab 的数据产物、Manifest 或实验基线。

实验状态只使用三种值：

- `可运行`：代码、文档、测试和验收全部存在；
- `规格已定义`：学习目标和验收已定义，代码尚未交付；
- `待设计`：只存在路线方向。

| Lab | 主题 | 状态 | 入口 |
| --- | --- | --- | --- |
| 00 | 环境与证据边界 | 可运行 | [教程](../docs/tutorials/00-environment.md) |
| 01 | 文档到训练 Token | 可运行 | [教程](../docs/tutorials/01-document-to-token.md) |
| 02 | Tiny LLM 基线训练 | 可运行 | [教程](../docs/tutorials/02-tiny-llm-baseline.md) |
| 03 | 最小数据版本 A/B | 可运行 | [教程](../docs/tutorials/03-controlled-data-ab.md) |
| 04 | 质量、污染与近似去重 | 可运行 | [教程](../docs/tutorials/04-quality-dedup-contamination.md) |
| 05 | 数据混合、Packing 与 Sharding | 可运行 | [教程](../docs/tutorials/05-mixing-packing-sharding.md) |
| 06 | smallpond Partition、Shuffle 与故障 | 可运行 | [教程](../docs/tutorials/06-partition-shuffle-recovery.md) |
| 07 | DataLoader 与 I/O | 规格已定义 | [路线第 10、13 周](../ROADMAP.md) |
| 08 | Checkpoint 与训练恢复 | 规格已定义 | [路线第 11～12 周](../ROADMAP.md) |
| 09 | 3FS 架构与 I/O | 规格已定义 | [路线第 13～14 周](../ROADMAP.md) |
| 10 | 正式数据 A/B 与毕业项目 | 规格已定义 | [路线第 15～16 周](../ROADMAP.md) |

后续只在代码和测试可以一起提交时创建新 Lab 目录。

## 当前学习入口

完成 Lab 03 后，先完成
[Bridge 01：DeepSeek 源码与论文强化周](../docs/tutorials/bridge-01-deepseek-source-study.md)，再按编号继续。
Bridge 补充源码证据、架构映射和个人学习产物，不替代后续真实 Lab。

## 当前开发优先级

1. Lab 07：建立 Map/Iterable DataLoader 的供给指标；
2. Lab 08：验证 Checkpoint 精确恢复；
3. Lab 09～10：补齐存储决策与正式多 Seed 数据实验。

这个顺序优先建立反馈闭环。不能因为后续模块更“底层”或更新颖，就跳过训练基线和受控实验。
