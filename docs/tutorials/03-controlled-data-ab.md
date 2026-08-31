# Lab 03：受控数据版本 A/B

本教程完成项目的第一个数据—模型反馈闭环。Baseline 使用项目原创的 AI 数据工程干净语料；Candidate 保留前一半内容，
把后一半替换为带唯一编号的低信息近重复模板。两次训练共享 Tokenizer、模型结构、初始参数、有效 Token 预算和独立评测集，
只把训练语料作为主要变量。

实验不预设 Candidate 一定更差。验收标准是控制变量真实生效、结果可复现、结论不超出证据，而不是得到期望数字。

## 完成后你将理解

- 为什么 A/B 不能分别训练 Tokenizer；
- 为什么训练/评测必须在文档来源上独立；
- 随机种子相同为什么还要验证初始参数哈希；
- 固定 Step 不等于固定有效 Token 预算；
- 训练 Loss、Evaluation Loss 和吞吐为什么不能互相替代；
- 如何用 Experiment Manifest 保存假设、控制变量、结果与证据边界。

## 1. 检查四类输入

```bash
ls -1 data/lab03
```

| 输入 | 作用 | 是否进入训练 |
| --- | --- | --- |
| `tokenizer_corpus.jsonl` | 单独训练固定 Tokenizer | 否 |
| `train_baseline.jsonl` | 干净训练版本 | 是，Run A |
| `train_candidate.jsonl` | 注入低信息近重复模板 | 是，Run B |
| `evaluation.jsonl` | 固定独立评测集 | 否 |

所有内容均为项目原创样例，Metadata 明确记录 `license=project-original`。

比较两个训练版本：

```bash
diff data/lab03/train_baseline.jsonl data/lab03/train_candidate.jsonl
```

两个版本拥有相同的 `source_id` 集合。前十条语义内容保持不变，后十条只改变文本质量和 Metadata，便于追踪同一来源在两个版本中的变化。

## 2. 运行实验

```bash
make lab03
```

等价命令：

```bash
uv run aide run-data-ab \
  --tokenizer-corpus data/lab03/tokenizer_corpus.jsonl \
  --baseline data/lab03/train_baseline.jsonl \
  --candidate data/lab03/train_candidate.jsonl \
  --evaluation data/lab03/evaluation.jsonl \
  --output artifacts/lab03

uv run aide validate-data-ab --output artifacts/lab03
```

执行过程会：

1. 在独立语料上训练一次 Tokenizer；
2. 用相同 Tokenizer 构建 Baseline、Candidate 和 Evaluation Dataset；
3. 排除训练 Dataset 最后一个可能包含 Padding 的 Sequence；
4. 用相同 Seed 重建两次初始模型；
5. 每次训练 100 Step，每个 Step 消费 4 × 64 个有效 Token；
6. 在同一个外部 Evaluation Dataset 上计算 Loss；
7. 写出并验证 Experiment Manifest、两个 Run Manifest 和两个 Checkpoint。

## 3. 检查控制变量

```bash
python3 -m json.tool artifacts/lab03/experiment_manifest.json
```

在 `controls` 中确认：

- `tokenizer_sha256` 只有一个；
- `evaluation_dataset_fingerprint` 固定；
- `training_config` 完全相同；
- `trained_token_budget` 为 25,600；
- `initial_model_state_sha256` 完全相同。

只写“使用相同 Seed”不够。验证器会用 Seed 重建初始模型，并比较 Tensor 内容哈希。

## 4. 理解固定 Token 预算

Baseline 和 Candidate 的总 Token 数、Sequence 数可以不同，因为训练语料本身就是实验变量。为了避免较大 Dataset 自动获得更多训练计算，
Lab 03 只使用无 Padding 的完整 Sequence，并设置 `drop_last=True`：

```text
100 steps × 4 sequences/step × 64 tokens/sequence = 25,600 effective tokens
```

因此两次运行消费相同数量的有效训练目标。不同 Dataset 大小造成的样本重复频率变化属于“训练语料版本”变量的一部分，必须在结论中说明。

## 5. 阅读本地结果

当前项目样例在 CPU 上得到：

| 指标 | Baseline | Candidate | Candidate - Baseline |
| --- | ---: | ---: | ---: |
| Final Train Loss | 0.08153606 | 0.10258445 | +0.02104839 |
| Evaluation Loss | 5.72893333 | 6.19993825 | +0.47100492 |
| 有效训练 Token | 25,600 | 25,600 | 0 |

Candidate 的 Evaluation Loss 在这次运行中高约 8.22%。这是一个可复现的本地观测，说明低信息近重复模板在当前小模型、语料和预算下
没有改善固定评测 Loss。它不能证明“所有重复数据都会让模型退化”，也不能外推到真实预训练规模。

两次 Tokens/s 也会写入 Manifest，但它们在同一进程中顺序运行，会受到缓存和机器负载影响。本 Lab 不使用该差值解释数据质量。

## 6. 区分三个结论层次

### 可以陈述

在当前原创小语料、固定 Tokenizer、相同初始模型、25,600 Token 训练预算和同一评测集下，Candidate 的 Evaluation Loss 比 Baseline 高
`0.47100492`。

### 只能提出假设

低信息近重复内容可能挤占了模型学习 AI 数据工程概念的有限 Token 预算。

### 不能陈述

- 近似去重一定能让所有模型变好；
- 当前差异具有统计显著性；
- 单次 CPU 实验代表 GPU 集群训练；
- Evaluation Loss 差异等于真实业务能力差异。

## 7. 验证可复现性

重复运行到另一个目录：

```bash
uv run aide run-data-ab \
  --tokenizer-corpus data/lab03/tokenizer_corpus.jsonl \
  --baseline data/lab03/train_baseline.jsonl \
  --candidate data/lab03/train_candidate.jsonl \
  --evaluation data/lab03/evaluation.jsonl \
  --output artifacts/lab03-repeat
```

应保持一致：

- `experiment_fingerprint`；
- 两个 Run Fingerprint；
- 初始和最终模型状态哈希；
- 两条 Loss History 与 Evaluation Loss。

运行时间和 Tokens/s 允许不同。

## 验收

- 四套 Dataset 均通过契约验证并共享同一个 Tokenizer 哈希；
- Evaluation Dataset 未参与训练；
- Baseline 与 Candidate 的初始模型参数哈希一致；
- 两次训练均消费 25,600 个有效 Token；
- 两个 Checkpoint 可以重新加载并复算 Evaluation Loss；
- Experiment Manifest 的输入、Dataset、Run、比较指标和 Fingerprint 通过验证；
- 能区分观测结果、解释假设和不可外推结论。

## 本实验不能证明什么

- 当前人工语料代表真实互联网、代码或企业数据；
- 单个 Seed 足以估计结果方差；
- 低信息模板是唯一的数据质量问题；
- Sequence 级 Loss 足以替代下游能力评测；
- 近似去重算法的 Precision、Recall 或最佳阈值。

最后一项正是 Lab 04 要解决的问题：先建立近似重复真值集和检测指标，再回到本 A/B 框架验证过滤策略。
