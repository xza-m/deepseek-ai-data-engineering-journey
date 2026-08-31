# Lab 00：环境与证据边界

本教程帮助你建立可复现的本地环境，并明确后续实验能够证明什么。

## 完成后你将得到

- Python、操作系统和基础命令的机器可读报告；
- 一套统一的本地质量门禁；
- 对“原理理解、本地实验、集群基准、生产验证”的区分。

## 1. 安装依赖

在仓库根目录执行：

```bash
make setup
```

项目使用 `uv` 安装 Python 3.11 和锁定依赖，不修改系统 Python。

## 2. 生成环境报告

```bash
make lab00
```

输出文件：

```text
artifacts/lab00/environment.json
```

查看报告：

```bash
python3 -m json.tool artifacts/lab00/environment.json
```

重点检查：

- Python 是否在项目支持范围内；
- Git、uv、Docker 和 gh 是否可用；
- 当前是否为 Linux；
- 当前环境是否具备 3FS 原生实验条件。

macOS 上 `threefs_native_ready` 应为 `false`。这不是失败，而是提醒你：3FS 原生集群实验需要 Linux，并且真实性能研究还需要 NVMe 与 RDMA 环境。

## 3. 运行质量门禁

```bash
make validate
```

它依次运行：

1. 静态检查；
2. 单元测试；
3. 环境报告；
4. Lab 01 数据构建与产物验证。

## 4. 建立证据语言

后续学习笔记使用以下四个标签：

| 标签 | 含义 | 可以声称 |
| --- | --- | --- |
| `conceptual` | 来自论文、设计文档或源码理解 | 能解释设计和机制 |
| `local` | 在个人电脑实际运行 | 代码和小规模行为已验证 |
| `cluster` | 在明确集群配置完成测试 | 指定配置下的规模和性能 |
| `production` | 有真实业务流量和运维记录 | 指定生产场景下可用 |

本地 Lab 不能证明 smallpond 的 PB 级能力，也不能证明 3FS 的 RDMA 性能。官方基准可以作为 `cluster` 证据引用，但不能改写成我们的实测结果。

## 验收

- `artifacts/lab00/environment.json` 存在且内容可解析；
- `make validate` 通过；
- 能解释为什么本机不能完成生产级 3FS 验证；
- 后续笔记能够使用统一证据标签。
