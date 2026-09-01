# Lab 05：确定性混合、Packing 与 Sharding

Lab 05 只消费 Lab 04 的质量准入文档，把“数据集合”升级为可以被训练、追溯和复现的 Dataset Version。

## 运行

```bash
make lab05
```

```text
Quality Report + Accepted Documents + Mix Spec
→ 确定性领域配比
→ Byte-level BPE / Document Packing
→ 固定长度 Sequence
→ 大小受控 Training Shard
→ Sequence Lineage / Dataset Card / Version Manifest
```

## 关键设计

- `mix_spec.json` 的权重必须为正且总和为 1；
- 使用最大余数法把权重转换为文档配额；
- 相同 Seed、输入和配比产生相同文档顺序与 Fingerprint；
- Shard 是 Dataset Sequence 的无损、顺序一致切分；
- 每条 Sequence Lineage 记录 Shard、Source ID、Domain、Quality Version 和 Token Span；
- Dataset Card 描述来源、规模、用途与限制，不替代机器可校验 Manifest。

默认实验把 8 条准入文档按五个领域混合，生成 10 条 Packed Sequence 和 3 个 JSONL Training Shard，血缘覆盖率为 1.0。

## 产物

- `dataset_version_manifest.json`：上游质量版本、Mix、Dataset、Shard、指标和 Fingerprint；
- `mixed_documents.jsonl`：确定性混合顺序；
- `dataset/`：可由 Lab 01 契约独立验证的 Tokenizer、Sequence 和 Manifest；
- `training_shards/`：大小受控的训练 Shard；
- `sequence_lineage.jsonl`：Sequence 到 Source/Quality/Shard 的血缘；
- `dataset_card.md`：面向人的数据版本说明。

## 验收

- 实际领域计数与目标配额一致；
- 所有 Shard 哈希正确，拼接后与原始 Sequence 完全一致；
- 每条 Sequence 恰好有一条 Lineage；
- Dataset Manifest 仍通过 Lab 01 验证；
- 上游 Quality Fingerprint、输入和 Mix Spec 均绑定 SHA-256；
- 相同输入重复运行得到相同 Dataset Version Fingerprint。

## 边界

当前 JSONL Shard 优先透明和可审计，不代表高吞吐训练格式。领域配比来自项目原创微型数据，只验证确定性机制，
不证明真实模型的最优数据混合。Lab 06 会保持同一 Dataset Version，单独改变计算和分区执行方式。

