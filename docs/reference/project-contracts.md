# 项目数据产物契约

本文定义 v0.1 实验使用的数据结构。字段含义只在这里维护，教程不重复定义另一套口径。

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
| `config` | 词表大小、序列长度和特殊 Token |
| `metrics` | 文档、Token、Sequence、重复和 Packing 指标 |
| `artifacts` | 产物的相对文件名 |
| `artifact_sha256` | 规范化文档、Sequence 和 Tokenizer 的内容哈希 |

Manifest 不包含生成时间、用户名、主机名和绝对路径。这些字段与数据语义无关，会破坏确定性。

## 训练运行 `run_manifest.json`

Lab 02 使用独立的 Run Manifest 把模型训练绑定到 Dataset Manifest：

```json
{
  "schema_version": "0.1",
  "training_version": "0.1.0",
  "run_fingerprint": "...",
  "dataset": {
    "pipeline_fingerprint": "...",
    "sequences_sha256": "...",
    "vocab_size": 320,
    "sequence_length": 64
  },
  "config": {
    "steps": 100,
    "batch_size": 4,
    "seed": 42
  },
  "split": {
    "train_sequence_ids": ["seq_00000000"],
    "validation_sequence_ids": ["seq_00000001"]
  },
  "metrics": {
    "initial_train_loss": 5.9,
    "final_train_loss": 0.1,
    "validation_loss": 6.0,
    "tokens_per_second": 1000.0
  },
  "model_state_sha256": "..."
}
```

`run_fingerprint` 由以下语义字段确定：

- `training_version`；
- Dataset 的 `pipeline_fingerprint`；
- 完整训练配置；
- 训练/验证 Sequence 拆分。

运行时间和 Tokens/s 不进入 Run Fingerprint，因为它们随机器负载和硬件变化。模型状态使用独立哈希，
用于判断相同数据、代码与训练配置是否得到相同参数。

Lab 02 的拆分单位是 Sequence，只用于本地训练闭环。它不承诺文档级隔离，也不能作为正式泛化评测。

## Checkpoint `checkpoint.pt`

Checkpoint 包含：

- 模型状态与优化器状态；
- 训练 Step 和训练配置；
- Dataset Fingerprint 与 Run Fingerprint；
- 训练/验证 Sequence 拆分。

Run Manifest 同时记录 Checkpoint 文件哈希与模型状态哈希。前者检测文件是否被修改，后者检测重新加载后的模型参数。
Lab 02 尚未保存调度器、随机状态和数据游标，因此只证明模型状态可以重新加载，不宣称支持精确断点续训。

## 证据等级

| 等级 | 需要的证据 |
| --- | --- |
| `conceptual` | 原始论文、官方设计文档或源码定位 |
| `local` | 可执行命令、环境、输入、输出和测试结果 |
| `cluster` | 节点、CPU/GPU、存储、网络、数据规模和基准方法 |
| `production` | 真实工作负载、SLO、监控、故障和持续运行记录 |

低等级证据不能直接升级为高等级结论。
