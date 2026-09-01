# 实验索引

本页是命令与教程的实验索引，不承担学习路线入口。第一次学习请先读 [START_HERE](../START_HERE.md) 和
[自主节奏能力关卡路线](../ROADMAP.md)。这些 Lab 不是彼此独立的课堂作业，而是同一个 Mini-LLM Data Factory
的连续版本；后一个 Lab 必须消费或引用前一个 Lab 的数据产物、Manifest 或实验基线。

实验状态只使用三种值：

- `可运行`：代码、文档、测试和验收全部存在；
- `规格已定义`：学习目标和验收已定义，代码尚未交付；
- `待设计`：只存在路线方向。

| 能力关卡 | Lab | 主题 | 状态 | 入口 |
| --- | --- | --- | --- | --- |
| 起点关 | 00 | 环境与证据边界 | 可运行 | [教程](../docs/tutorials/00-environment.md) |
| 语义关 | 01 | 文档到训练 Token | 可运行 | [教程](../docs/tutorials/01-document-to-token.md) |
| 语义关 | 02 | Tiny LLM 基线训练 | 可运行 | [教程](../docs/tutorials/02-tiny-llm-baseline.md) |
| 语义关 | 03 | 最小数据版本 A/B | 可运行 | [教程](../docs/tutorials/03-controlled-data-ab.md) |
| 源码桥 | B1 | DeepSeek 论文、smallpond 与 3FS 连接 | 可学习 | [教程](../docs/tutorials/bridge-01-deepseek-source-study.md) |
| 治理关 | 04 | 质量、污染与近似去重 | 可运行 | [教程](../docs/tutorials/04-quality-dedup-contamination.md) |
| 治理关 | 05 | 数据混合、Packing 与 Sharding | 可运行 | [教程](../docs/tutorials/05-mixing-packing-sharding.md) |
| 规模关 | 06 | smallpond Partition、Shuffle 与故障 | 可运行 | [教程](../docs/tutorials/06-partition-shuffle-recovery.md) |
| 供给关 | 07 | DataLoader 与 I/O | 可运行 | [教程](../docs/tutorials/07-dataloader-io.md) |
| 供给关 | 08 | Checkpoint 与训练恢复 | 可运行 | [教程](../docs/tutorials/08-checkpoint-exact-recovery.md) |
| 存储关 | 09 | 3FS 架构与 I/O | 可运行 | [教程](../docs/tutorials/09-3fs-storage-workloads.md) |
| 反馈关 | 10 | 正式数据 A/B 与毕业项目 | 可运行 | [教程](../docs/tutorials/10-graduation-feedback-loop.md) |

全部必修 Lab 都已有代码、教程、测试、验证器与真实本地产物入口。

## 索引使用规则

- 学习进度以能力关卡的晋级证据为准，不以 Lab 编号或用时为准；
- Bridge 在 Lab 03 和 Lab 04 之间完成，补充源码证据和架构映射，不替代真实 Lab；
- `make labNN` 会自动生成所需的前置 Lab，但学习者仍需逐关检查产物；
- `make journey` 只用于完成主线后的全量复现，不是首次学习入口。

## 当前维护优先级

1. 由第一批真实学习者执行独立复现并记录摩擦点；
2. 保持依赖、官方源码阅读索引与 CI 可用；
3. Linux/RDMA/NVMe/多 GPU 条件满足时，再补 Infra 进阶支线。

主线已经闭环，但学习顺序仍不能跳跃：底层系统实验必须建立在训练基线、数据版本和受控反馈之上。
