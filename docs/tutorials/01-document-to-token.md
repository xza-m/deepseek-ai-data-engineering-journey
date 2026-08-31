# Lab 01：从文档到训练 Token

本教程带你完成第一条真实训练数据链路。重点不是构建强大的 Tokenizer，而是看见传统“数据行”如何变成模型实际消费的训练 Sequence。

## 完成后你将理解

- 文档规范化和精确去重；
- Byte-level BPE 词表训练；
- BOS/EOS 如何表达文档边界；
- `input_ids`、`labels` 和 `loss_mask`；
- Packing 利用率与来源血缘；
- Dataset Manifest 如何保证可复现。

## 1. 查看输入

```bash
sed -n '1,5p' data/sample/raw_documents.jsonl
```

每行是一条 JSON 文档，包含 `source_id`、`text` 和可选 `metadata`。示例数据中故意包含一条重复文档，用于验证精确去重。

## 2. 构建数据集

```bash
make setup
make lab01
```

等价命令：

```bash
uv run aide build-dataset \
  --input data/sample/raw_documents.jsonl \
  --output artifacts/lab01 \
  --vocab-size 320 \
  --sequence-length 64

uv run aide validate-dataset --output artifacts/lab01
```

## 3. 检查 Manifest

```bash
python3 -m json.tool artifacts/lab01/manifest.json
```

回答以下问题：

1. 原始文档数和保留文档数相差多少？
2. `source_sha256` 和 `tokenizer_sha256` 分别保护什么？
3. `packing_efficiency` 为什么不会总是 1？
4. `pipeline_fingerprint` 发生变化时，哪些上游条件可能改变了？

## 4. 理解训练 Sequence

查看第一条 Sequence：

```bash
sed -n '1p' artifacts/lab01/sequences.jsonl | python3 -m json.tool
```

每条记录包含：

- `input_ids`：输入模型的 Token；
- `labels`：向右移动一位后的预测目标；
- `loss_mask`：填充位置不参与 Loss；
- `provenance`：当前窗口与来源文档 Token 区间的交集。

实验会在每个文档前后添加 BOS/EOS，然后把多个文档放入连续 Token 流。文档边界被保留，但没有实现跨样本 Attention Mask。这与“文档边界存在”和“Attention 是否跨文档”是两个不同问题。

## 5. 做一次控制变量实验

保持输入、词表不变，仅修改 Sequence Length：

```bash
uv run aide build-dataset \
  --input data/sample/raw_documents.jsonl \
  --output artifacts/lab01-seq32 \
  --vocab-size 320 \
  --sequence-length 32
```

比较：

```bash
python3 -m json.tool artifacts/lab01/manifest.json
python3 -m json.tool artifacts/lab01-seq32/manifest.json
```

观察 Sequence 数量、Packing 利用率和 Pipeline Fingerprint。不要同时修改词表大小，否则无法归因变化来自哪个变量。

## 6. 验证可复现性

重复生成到另一个目录：

```bash
uv run aide build-dataset \
  --input data/sample/raw_documents.jsonl \
  --output artifacts/lab01-repeat \
  --vocab-size 320 \
  --sequence-length 64

diff artifacts/lab01/manifest.json artifacts/lab01-repeat/manifest.json
```

`diff` 应无输出。Manifest 不写入生成时间和绝对路径，因为它们会制造与数据内容无关的差异。

## 验收

- 数据构建和验证命令成功；
- 精确重复文档被删除；
- 所有 Sequence 长度固定；
- `input_ids` 与 `labels` 保持一位偏移；
- Manifest 可重复生成；
- 能从 Sequence 的 `provenance` 找到来源文档。

## 本实验不能证明什么

- 该示例数据具有真实预训练价值；
- 320 大小的实验词表适合生产模型；
- 精确去重足以解决近似重复和评测污染；
- 本地吞吐代表 smallpond、3FS 或 GPU 集群性能。
