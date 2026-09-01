# Lab 04：近似去重、评测污染与质量治理

本教程把传统数仓中的“唯一键、空值和规则质量”推进到训练数据的内容相似性、集合隔离、许可、PII 和失败样例。

## 运行

```bash
make lab04
```

主命令使用项目原创输入和人工标注 Pair：

```bash
uv run aide audit-quality \
  --train data/lab04/train.jsonl \
  --evaluation data/lab04/evaluation.jsonl \
  --truth data/lab04/duplicate_truth.jsonl \
  --output artifacts/lab04
```

## 方法

1. 对规范化文本构造词级 3-Shingle；
2. 使用 Jaccard 相似度生成近似重复 Pair 和连通簇；
3. 在人工标注 Pair 上计算 Precision、Recall、F1 和混淆矩阵；
4. 交叉比较训练集与独立评测集，输出污染候选；
5. 在唯一质量边界检查最小长度、许可、安全元数据和确定性 PII；
6. 每个接受或拒绝文档都保留 `quality_audit`，不静默删除失败样例。

默认结果中，14 条训练输入接受 8 条、拒绝 6 条；两组近似重复被聚类，检测到两组训练—评测污染候选。
原创 6 对标注集上的 Precision 和 Recall 均为 1.0。

## 为什么不能只报告去重率

去重率无法说明误删和漏删。阈值过低会把主题相似但语义不同的内容合并，阈值过高则漏掉改写重复。因此报告同时保留：

- 每个标注 Pair 的相似度、预测和真值；
- False Positive 和 False Negative；
- 重复簇及保留成员；
- 每种拒绝原因和原始样例；
- 训练—评测污染候选。

修改阈值时，必须重新运行真值评估和验证命令，不能只追求更高的过滤量。

## 产物

- `quality_report.json`：输入哈希、规则、指标、证据边界和 Quality Fingerprint；
- `accepted_documents.jsonl`：可以进入下一数据版本的文档；
- `rejected_documents.jsonl`：带拒绝原因的失败样例；
- `duplicate_clusters.json`：近似重复簇及保留策略；
- `contamination_pairs.jsonl`：训练—评测相似候选；
- `truth_outcomes.jsonl`：人工标注 Pair 的逐条预测结果。

## 验收

- 真值集非空，报告 Precision、Recall、F1 和四格混淆矩阵；
- 每个拒绝文档至少有一个可解释原因；
- 训练和评测输入分别绑定 SHA-256；
- 所有产物哈希和 Quality Fingerprint 可以重算；
- 修改任一产物后，`validate-quality` 必须失败；
- 能解释误判和漏判的不同业务成本。

## 证据边界

当前 Shingling 是透明、可复现的基线，不是通用语义去重最优算法；小真值集上的 1.0 也不表示生产准确率。
确定性 PII 和安全元数据不能替代分类器、人工复核和正式合规流程。Lab 05 只消费已经保留审计证据的接受文档。

