# 如何运行本地质量门禁

本文用于提交代码前快速确认项目仍然可安装、可测试、可构建数据产物。

## 运行全部门禁

```bash
make validate
```

成功条件：命令退出码为 0，全部测试通过，Lab 00～10 的构建与验证均成功，最后输出
`graduation validation passed`。

## 单独运行

```bash
make lint
make test
make lab00
make lab01
make lab02
make lab03
make lab04
make lab05
make lab06
make lab07
make lab08
make lab09
make lab10
make journey
```

## 常见失败

### `uv: command not found`

先安装 [uv](https://docs.astral.sh/uv/)，再运行：

```bash
make setup
```

### Python 版本不兼容

不要替换系统 Python。执行：

```bash
uv sync --python 3.11 --extra dev --extra train
```

### Manifest 验证失败

删除本地产物并重新生成：

```bash
make clean
make lab01
```

如果仍然失败，检查是否只修改了代码，却没有同步更新数据契约和测试。

### 训练依赖未安装

Lab 02、03、07、08、10 使用独立的训练依赖：

```bash
uv sync --extra train
```

Lab 00/01 不依赖 PyTorch。只安装核心依赖时，数据 Pipeline 仍应可以运行。

### Training Run 验证失败

先确认上游 Dataset 没有被重新生成或修改：

```bash
uv run aide validate-dataset --output artifacts/lab01
uv run aide validate-training-run --dataset artifacts/lab01 --output artifacts/lab02
```

如果 Dataset Fingerprint 已变化，应重新运行 Lab 02，不能把旧 Checkpoint 强行绑定到新数据版本。

### Data A/B 验证失败

```bash
uv run aide validate-data-ab --output artifacts/lab03
```

验证器会逐层检查四个 Dataset、两个 Run、共享 Tokenizer、外部 Evaluation Dataset、初始模型状态和有效 Token 预算。
不要直接编辑 `experiment_manifest.json` 修正数字，应重新运行产生该数字的 Dataset 或 Run。

### 依赖锁变化

提交前检查：

```bash
git diff -- uv.lock pyproject.toml
```

只有显式修改依赖时才应改变 `uv.lock`。

## CI 与本地差异

GitHub Actions 使用 Linux 和 Python 3.11。本地 macOS 成功不等于 Linux 必然成功，因此公开提交必须同时通过 CI。反过来，CI 成功也不证明 3FS、RDMA 或大规模训练可用。
