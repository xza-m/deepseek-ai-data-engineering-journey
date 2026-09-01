# Lab 10：正式数据 A/B、失败归因与毕业复现

最后一 Lab 把前九个实验连接成一个机器可重算的证据链。目标不是证明“清洗一定提升模型”，而是证明你能
控制变量、报告方差、追踪失败，并基于模型反馈提出下一版数据策略。

## 1. 一条命令运行完整项目

```bash
make journey
```

Lab 10 自动执行三个 Seed 的 A/B。唯一实验变量是 `dataset_admission_policy_version`：

- Baseline `raw-policy-v0`：Lab 04 的全部训练候选；
- Candidate `quality-policy-v1`：只保留质量准入文档，并排除训练—评测污染候选。

每个 Seed 内固定 Tokenizer、模型结构、初始化、独立 Evaluation Dataset 和 7,680 个有效训练 Token。

## 2. 读多 Seed 结论

```bash
open artifacts/lab10/graduation_report.md
uv run python -m json.tool artifacts/lab10/graduation_report.json
```

报告必须同时给出每个 Seed、平均 Delta 和总体标准差。真实默认结果允许 Candidate 改善、退化或方向不一致；
三次 Seed 只能揭示实验方差，不能声称统计显著或生产收益。

## 3. 追踪失败与数据变化

```bash
head artifacts/lab10/training_policy_diff.jsonl
uv run python -m json.tool artifacts/lab10/reproduction_manifest.json
```

`failure_examples` 保存 Candidate 相对 Baseline 退化最大的 Evaluation Sequence，并追踪到 Evaluation Source。
`training_policy_diff.jsonl` 说明每个训练文档在两个策略下是保留还是排除，以及近似重复、PII、许可、安全、
短文本或评测污染原因。

两条血缘能说明“哪些训练数据变化与哪些失败同时发生”，但不能从 Tiny LM 直接证明单文档因果贡献。

## 4. 验证端到端 Fingerprint

Graduation Report 绑定六个上游 Fingerprint：Quality、Dataset Version、Compute、DataLoader、Recovery 和 Storage。
任一上游数据、规则、Shard、Checkpoint 或存储工作负载变化，都必须重新运行毕业实验。

```bash
uv run aide validate-graduation \
  --data data \
  --artifacts artifacts \
  --output artifacts/lab10
```

## 5. 独立复现的诚实边界

`make validate` 是机器门禁；GitHub Actions 在干净 Linux 环境再次执行全链路。人工独立复现必须由真实学习者
只依赖公开仓库完成，因此项目只标记 `ready_for_first_learner`，不会在学习开始前伪造一条“他人已复现”记录。
你完成路线时，把自己的问题、耗时、失败和修正追加为第一份真实学习者复现证据。

## 验收

- 至少两个 Seed，本项目默认三个；
- 每个 A/B 共享 Tokenizer、评测集、初始化和有效 Token 预算；
- 报告平均值、方差、反例和不确定性；
- 最差 Evaluation Sequence 可追溯，训练 Policy Diff 可审计；
- Graduation Fingerprint 绑定 Lab 04～09；
- `make validate` 和公开 CI 都通过。
