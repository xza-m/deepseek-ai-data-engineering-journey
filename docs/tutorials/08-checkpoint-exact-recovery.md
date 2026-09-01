# Lab 08：Checkpoint 精确恢复与并行数据契约

“有 Checkpoint 文件”不等于“能够恢复训练”。真正的恢复必须回到同一数据位置、同一优化器和学习率状态，
并在确定性边界内产生与不中断训练相同的后续步骤。

## 1. 运行故障实验

```bash
make lab08
```

实验先训练一条 12 Step 的不中断参考链；第二条链在 Step 5 保存 Checkpoint、重新创建进程内对象、加载状态，
再从数据游标继续到 Step 12。

Checkpoint 包含：

- Model State；
- Optimizer State；
- Learning-rate Scheduler State；
- Python 与 Torch RNG State；
- Data Cursor 和固定 Batch Schedule 哈希；
- Loss History 与 Dataset Version Fingerprint。

## 2. 检查精确等价

```bash
uv run python -m json.tool artifacts/lab08/recovery_report.json
```

`equivalence` 中五项都必须为 `true`：逐步 Loss、模型参数、优化器状态、调度器状态和最终学习率。
只比较“最终 Loss 差不多”无法发现漏存 Momentum、学习率错位或样本重复消费。

## 3. 把恢复扩展到分布式训练

```bash
uv run python -m json.tool artifacts/lab08/parallelism_data_contract.json
```

文件分别列出 Data Parallel、Tensor Parallel、Pipeline Parallel、Expert Parallel 和 ZeRO 的分区对象、
数据风险与最低血缘要求。它是架构契约，不是多 GPU 实测：例如 Expert Parallel 的关键数据风险是 Router
导致的 Token 倾斜，而 Pipeline Parallel 需要足够 Micro Batch 降低 Bubble。

## 4. 证据边界

本地精确等价只对当前 CPU、PyTorch、确定性算子与固定 Batch Schedule 成立。生产环境还必须验证 World Size
变化、分布式状态分片、远端存储耐久性、原子发布、Checkpoint 保留策略和恢复时间目标。

## 验收

- 恢复链和参考链的逐步 Loss 完全相同；
- 模型、优化器和 Scheduler 状态哈希完全相同；
- Checkpoint 绑定 Dataset、Schedule 和 Data Cursor；
- 篡改 Checkpoint 会被验证器发现；
- 能说明五种并行策略分别给数据系统增加了什么压力。
