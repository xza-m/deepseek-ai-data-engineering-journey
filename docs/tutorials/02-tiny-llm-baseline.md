# Lab 02：Tiny LLM 基线训练

本教程把 Lab 01 生成的训练 Sequence 真正送入一个 112K 参数左右的 Tiny Transformer，接通
`Dataset Manifest → Loss → Checkpoint → Run Manifest`。目标是理解模型如何消费数据，并建立后续数据 A/B 的固定训练基线，
不是训练具有实用生成能力的模型。

## 完成后你将理解

- Dataset、DataLoader、Batch 和因果 Attention Mask 如何协作；
- `input_ids`、`labels`、`loss_mask` 如何进入交叉熵 Loss；
- 为什么训练运行必须绑定 Dataset Fingerprint；
- Run Fingerprint 与运行时吞吐指标为什么要分开；
- 为什么训练 Loss 下降不代表模型具有泛化能力；
- Checkpoint 验证为什么需要同时检查文件和模型状态。

## 1. 安装训练依赖

```bash
make setup
```

PyTorch 和 NumPy 位于独立的 `train` extra。只运行 Lab 00/01 时可以使用更轻量的安装：

```bash
uv sync --extra dev
```

## 2. 构建上游 Dataset

```bash
make lab01
```

Lab 02 不重新读取原始文档，也不重新训练 Tokenizer。它只消费已经通过验证的：

```text
artifacts/lab01/manifest.json
artifacts/lab01/sequences.jsonl
```

训练前会再次执行 Dataset 验证。Sequence 内容、Tokenizer 或 Manifest 被修改后，训练不能继续使用旧 Fingerprint。

## 3. 训练 Tiny LM

```bash
make lab02
```

等价命令：

```bash
uv run aide train-tiny-lm \
  --dataset artifacts/lab01 \
  --output artifacts/lab02

uv run aide validate-training-run \
  --dataset artifacts/lab01 \
  --output artifacts/lab02
```

默认配置使用 CPU、固定随机种子、两层 Transformer 和 100 个训练 Step。CPU 是本阶段刻意选择的可复现基线，
MPS、CUDA 和多 GPU 留到后续性能与分布式实验。

## 4. 检查 Run Manifest

```bash
python3 -m json.tool artifacts/lab02/run_manifest.json
```

重点观察：

- `dataset.pipeline_fingerprint`：当前训练消费的 Dataset 版本；
- `run_fingerprint`：数据版本、训练配置和拆分策略的联合指纹；
- `split`：训练与验证 Sequence ID；
- `metrics.initial_train_loss` 和 `final_train_loss`：模型是否能够拟合训练数据；
- `metrics.validation_loss`：重新加载 Checkpoint 时需要复算的基线；
- `metrics.tokens_per_second`：当前机器的本地运行指标；
- `model_state_sha256`：与 Checkpoint 容器格式无关的模型状态哈希。

`duration_seconds` 和 `tokens_per_second` 不进入 Run Fingerprint。相同语义运行在不同机器上可以具有相同 Fingerprint，
但不会伪装成相同的性能结果。

## 5. 把一条 Sequence 连接到 Loss

训练时，每个 Batch 执行以下关系：

```text
input_ids
→ Token Embedding + Position Embedding
→ Causal Transformer
→ 每个位置的 Vocabulary Logits
→ 与 labels 计算 Cross Entropy
→ 使用 loss_mask 排除 Padding
→ Batch Loss
```

因果 Mask 禁止当前位置看到未来 Token。`loss_mask` 解决的是 Padding 是否参与 Loss，两者职责不同。

## 6. 理解当前训练结果

示例数据只有 14 条 Sequence。默认训练通常会看到训练 Loss 快速下降，而验证 Loss 仍然较高。这说明模型可以记住小规模训练数据，
不说明它具有语言能力，也不说明验证集足够可靠。

当前拆分按 Sequence 确定性完成。同一文档的 Token 可能跨越多个 Sequence，所以这里的验证集可能存在文档泄漏。
Run Manifest 会明确记录这一边界。Lab 03 将建立固定、独立的数据版本 A/B 评测，不能直接拿 Lab 02 的验证 Loss 宣称数据质量更好。

## 7. 验证 Checkpoint

```bash
uv run aide validate-training-run \
  --dataset artifacts/lab01 \
  --output artifacts/lab02
```

验证器会检查：

1. Dataset Fingerprint 和 Sequence 哈希；
2. Run Fingerprint；
3. Checkpoint 文件哈希；
4. Checkpoint 内的数据版本、训练配置和 Step；
5. 重新加载后的模型状态哈希；
6. 重新计算的 Validation Loss。

Lab 02 只证明 Checkpoint 可以加载并恢复模型状态。优化器、随机状态和数据游标的精确断点续训属于后续 Checkpoint Lab。

## 8. 验证可复现性

```bash
uv run aide train-tiny-lm \
  --dataset artifacts/lab01 \
  --output artifacts/lab02-repeat
```

比较两个 Run Manifest 中的：

- `run_fingerprint`；
- `model_state_sha256`；
- `metrics.loss_history`。

三者应相同。吞吐和运行时间允许不同，因为它们是环境相关观测值。

## 验收

- Dataset 验证、训练和 Training Run 验证全部成功；
- 初始与最终训练 Loss 均为有限值，最终训练 Loss 低于初始值；
- Run Manifest 引用了 Lab 01 的 Pipeline Fingerprint；
- Checkpoint 可以重新加载，并复算出一致的 Validation Loss；
- 相同输入和配置生成相同 Run Fingerprint 与模型状态哈希；
- 能解释为什么当前验证结果不能用于判断真实模型泛化。

## 本实验不能证明什么

- 示例数据具有真实预训练价值；
- 当前验证集无文档泄漏；
- 训练 Loss 下降代表模型获得通用能力；
- CPU Tokens/s 可以代表 GPU 或集群吞吐；
- Checkpoint 已经支持精确断点续训；
- 当前 Tiny Transformer 结构适合生产训练。
