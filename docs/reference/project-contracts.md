# 项目数据产物契约

本文定义当前实验使用的数据结构。字段含义只在这里维护，教程不重复定义另一套口径。

## 原始文档 `raw_documents.jsonl`

每行格式：

```json
{
  "source_id": "doc-001",
  "text": "需要处理的原始文本",
  "metadata": {"language": "zh", "domain": "data-engineering"}
}
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `source_id` | string | 是 | 来源系统中的稳定标识 |
| `text` | string | 是 | 原始文本 |
| `metadata` | object | 否 | 来源、语言、领域等辅助属性 |

## 规范化文档 `normalized_documents.jsonl`

```json
{
  "doc_id": "doc_0123456789abcdef",
  "source_id": "doc-001",
  "content_sha256": "...",
  "text": "规范化文本",
  "metadata": {},
  "token_count": 42
}
```

`doc_id` 来自规范化内容哈希。内容相同的文档只保留输入中第一次出现的记录。

## 训练序列 `sequences.jsonl`

```json
{
  "sequence_id": "seq_00000000",
  "input_ids": [1, 15, 20, 2],
  "labels": [15, 20, 2, 0],
  "loss_mask": [1, 1, 1, 0],
  "provenance": [
    {
      "doc_id": "doc_0123456789abcdef",
      "sequence_token_start": 0,
      "sequence_token_end": 3,
      "document_token_start": 0,
      "document_token_end": 3
    }
  ]
}
```

约束：

- 三个数组长度都等于 `sequence_length`；
- `labels[i]` 是 `input_ids[i]` 的下一个 Token；
- Padding 对应的 `loss_mask` 为 0；
- Token 区间采用左闭右开 `[start, end)`；
- `provenance` 只描述窗口与文档的交集，不复制原始文本。

## Manifest `manifest.json`

主要字段：

| 字段 | 说明 |
| --- | --- |
| `schema_version` | Manifest 契约版本 |
| `pipeline_version` | 数据加工实现版本 |
| `source_sha256` | 原始输入文件内容哈希 |
| `tokenizer_sha256` | Tokenizer 文件内容哈希 |
| `pipeline_fingerprint` | 输入、Tokenizer 和关键配置的联合指纹 |
| `config` | Tokenizer 模式、词表大小、序列长度和特殊 Token |
| `metrics` | 文档、Token、Sequence、重复和 Packing 指标 |
| `artifacts` | 产物的相对文件名 |
| `artifact_sha256` | 规范化文档、Sequence 和 Tokenizer 的内容哈希 |

Manifest 不包含生成时间、用户名、主机名和绝对路径。这些字段与数据语义无关，会破坏确定性。

`config.tokenizer_mode` 取值：

- `trained`：使用当前文档训练并保存 Tokenizer；
- `reused`：复制并使用外部固定 Tokenizer，`vocab_size_requested` 为 `null`。

Lab 03 使用 `reused` 保证 Baseline、Candidate 和 Evaluation 的 Token 语义完全一致。

## 训练运行 `run_manifest.json`

Lab 02 使用独立的 Run Manifest 把模型训练绑定到 Dataset Manifest：

```json
{
  "schema_version": "0.1",
  "training_version": "0.2.0",
  "run_fingerprint": "...",
  "dataset": {
    "pipeline_fingerprint": "...",
    "sequences_sha256": "...",
    "vocab_size": 320,
    "sequence_length": 64
  },
  "evaluation_dataset": null,
  "config": {
    "steps": 100,
    "batch_size": 4,
    "seed": 42,
    "strict_token_budget": false
  },
  "split": {
    "mode": "internal_sequence_split",
    "train_sequence_ids": ["seq_00000000"],
    "excluded_train_sequence_ids": [],
    "evaluation_sequence_ids": ["seq_00000001"]
  },
  "metrics": {
    "initial_train_loss": 5.9,
    "final_train_loss": 0.1,
    "validation_loss": 6.0,
    "tokens_per_second": 1000.0
  },
  "initial_model_state_sha256": "...",
  "model_state_sha256": "..."
}
```

`run_fingerprint` 由以下语义字段确定：

- `training_version`；
- Dataset 的 `pipeline_fingerprint`；
- 可选外部 Evaluation Dataset 的 `pipeline_fingerprint`；
- 完整训练配置；
- 训练/评测 Sequence 拆分。

运行时间和 Tokens/s 不进入 Run Fingerprint，因为它们随机器负载和硬件变化。模型状态使用独立哈希，
用于判断相同数据、代码与训练配置是否得到相同参数。

`initial_model_state_sha256` 不是直接相信随机种子，而是根据 Seed 重建模型后对所有 Tensor 内容计算哈希。
Lab 03 要求两次 Run 的初始模型状态哈希完全相同。

Lab 02 的拆分单位是 Sequence，只用于本地训练闭环。它不承诺文档级隔离，也不能作为正式泛化评测。

Lab 03 设置 `evaluation_dataset` 和 `split.mode=external_evaluation_dataset`。当
`strict_token_budget=true` 时，训练只使用无 Padding 的完整 Sequence，并通过 `drop_last` 保证每个 Step 消费相同 Token 数。

## Checkpoint `checkpoint.pt`

Checkpoint 包含：

- 模型状态与优化器状态；
- 训练 Step 和训练配置；
- Dataset Fingerprint 与 Run Fingerprint；
- 训练/验证 Sequence 拆分。

Run Manifest 同时记录 Checkpoint 文件哈希与模型状态哈希。前者检测文件是否被修改，后者检测重新加载后的模型参数。
Lab 02 尚未保存调度器、随机状态和数据游标，因此只证明模型状态可以重新加载，不宣称支持精确断点续训。

## 数据实验 `experiment_manifest.json`

Lab 03 使用 Experiment Manifest 连接四个 Dataset 和两个 Training Run：

```json
{
  "schema_version": "0.1",
  "experiment_version": "0.1.0",
  "experiment_fingerprint": "...",
  "controlled_variable": {
    "name": "training_corpus",
    "expected_direction": null
  },
  "controls": {
    "tokenizer_sha256": "...",
    "evaluation_dataset_fingerprint": "...",
    "trained_token_budget": 25600,
    "initial_model_state_sha256": "..."
  },
  "runs": {
    "baseline": {"evaluation_loss": 5.7},
    "candidate": {"evaluation_loss": 6.2}
  },
  "comparison": {
    "candidate_minus_baseline_evaluation_loss": 0.5
  }
}
```

Experiment Fingerprint 覆盖：

- 输入文件哈希；
- 受控变量定义；
- Tokenizer、Evaluation Dataset、训练配置、Token 预算和初始模型状态；
- 四个 Dataset Fingerprint；
- 两个 Run Fingerprint。

运行时间和吞吐不进入 Experiment Fingerprint。比较指标可以由两个 Run Manifest 重新计算，验证器不会相信手工填写的差值。

## 质量报告 `quality_report.json`

Lab 04 的 Quality Fingerprint 覆盖训练、评测和真值输入哈希，Shingle/阈值与质量规则、重算指标，以及全部审计产物哈希。
逐文档 `quality_audit` 必须记录接受/拒绝决策和原因。Precision/Recall 只在显式标注 Pair 上计算，未标注 Pair 不得被
偷偷当成负例。污染候选与质量拒绝是两个不同概念：前者保护评测隔离，后者决定训练数据准入。

## 数据版本 `dataset_version_manifest.json`

Lab 05 把 Quality Fingerprint、接受文档、领域 Mix Spec、Tokenizer、Packed Sequence 和 Training Shard 绑定为同一版本。
Shard 必须按记录顺序无损还原 Dataset 中的全部 Sequence；`sequence_lineage.jsonl` 必须对每个 Sequence 恰好覆盖一次，
并保留 Source ID、Domain、Quality Version 和 Token Span。Dataset Card 是面向人的解释，不能代替机器可重算的 Fingerprint。
Quality Report 的运行耗时属于观测值，不进入 Dataset Version，避免相同语义输入重复执行时产生版本漂移。

## DataLoader 报告 `dataloader_report.json`

Lab 07 绑定 Dataset Version、Compute Fingerprint 与每个 Training Shard 哈希。`profiles` 只记录可重算的 Sample、
有效 Token、唯一 Sequence 和 Batch 数；首批等待、P50/P99 与吞吐属于机器观测，放在 `observations` 中，
不进入 DataLoader Fingerprint。

## 恢复报告 `recovery_report.json`

Lab 08 的 Checkpoint 必须保存 Model、Optimizer、Scheduler、随机数状态、数据游标、Batch Schedule 哈希、
Loss History 和 Dataset Version。恢复验收不是 Loss 近似，而是参考链与恢复链的逐步 Loss、最终 Model、
Optimizer、Scheduler 和学习率全部精确相同。

## 存储报告 `storage_report.json`

Lab 09 绑定 Dataset、Compute 和 Recovery Fingerprint，分别定义顺序 Shard 读、随机 Sample 读、Shuffle 写和
Checkpoint 并行写。操作数和逻辑 Bytes 进入稳定版本；P50/P99、吞吐和环境能力属于观测。未部署 3FS 时，
报告必须显式保存 `cluster_executed=false`，架构理解不能替代集群基准。

## 证据等级

| 等级 | 需要的证据 |
| --- | --- |
| `conceptual` | 原始论文、官方设计文档或源码定位 |
| `local` | 可执行命令、环境、输入、输出和测试结果 |
| `cluster` | 节点、CPU/GPU、存储、网络、数据规模和基准方法 |
| `production` | 真实工作负载、SLO、监控、故障和持续运行记录 |

低等级证据不能直接升级为高等级结论。
