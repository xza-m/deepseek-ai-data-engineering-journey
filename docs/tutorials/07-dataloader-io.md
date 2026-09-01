# Lab 07：DataLoader 与训练数据供给

传统数据任务以“分区按时产出”为终点，训练系统还要追问：Batch 是否及时到达训练循环、每秒提供多少有效
Token、等待发生在文件读取还是 CPU 处理。这个 Lab 用 Lab 05 的同一批 Training Shard 对比两种真实 PyTorch
数据入口。

## 1. 运行

```bash
make lab07
```

默认 Profile 三条路径：

- `map-workers-0`：启动时把全部 Shard 加载到内存；
- `iterable-workers-0`：单进程逐 Shard、逐 Epoch 流式读取；
- `iterable-workers-2`：两个 Worker 按 Shard 分工读取。

## 2. 读报告

```bash
uv run python -m json.tool artifacts/lab07/dataloader_report.json
open artifacts/lab07/dataloader_decision.md
```

先检查稳定的完整性指标：每条路径交付的 Sample、有效 Token 和唯一 Sequence 都必须一致。再看本机观测：
首批等待、批次等待 P50/P99、Samples/s、有效 Tokens/s。

## 3. 正确解释 Worker

Worker 更多不等于更快。当前数据只有少量 Shard，启动进程和序列化开销可能大于并行收益；Iterable Dataset
还会在每个 Epoch 重新打开文件，虽然操作系统缓存可能让物理设备读取变少。先用指标定位，再调整 Shard 数量、
Worker、预取或缓存。

## 4. 与训练串联

LLM 训练真正关心的是有效 Tokens/s，而不是 JSON 行数。`loss_mask=0` 的 Padding 不计入供给能力。没有 GPU 时，
本实验只能建立 CPU 侧供给基线；有 GPU 后还需增加 DataLoader Wait 占 Step 时间比例、Pinned Memory、
Host-to-Device 复制和 GPU 利用率。

## 验收

- 三条 Profile 都覆盖全部 Sequence；
- Sample 和有效 Token 数完全一致；
- 报告区分确定性完整性与环境相关耗时；
- 决策树能把问题路由到文件组织、CPU 解析、Worker 或设备复制；
- 篡改任一上游 Shard 后验证失败。
