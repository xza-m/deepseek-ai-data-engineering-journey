# 如何运行本地质量门禁

本文用于提交代码前快速确认项目仍然可安装、可测试、可构建数据产物。

## 运行全部门禁

```bash
make validate
```

成功条件：命令退出码为 0，且 Lab 01 最终输出 `dataset validation passed`。

## 单独运行

```bash
make lint
make test
make lab00
make lab01
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
uv sync --python 3.11 --extra dev
```

### Manifest 验证失败

删除本地产物并重新生成：

```bash
make clean
make lab01
```

如果仍然失败，检查是否只修改了代码，却没有同步更新数据契约和测试。

### 依赖锁变化

提交前检查：

```bash
git diff -- uv.lock pyproject.toml
```

只有显式修改依赖时才应改变 `uv.lock`。

## CI 与本地差异

GitHub Actions 使用 Linux 和 Python 3.11。本地 macOS 成功不等于 Linux 必然成功，因此公开提交必须同时通过 CI。反过来，CI 成功也不证明 3FS、RDMA 或大规模训练可用。
