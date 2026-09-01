"""Lab 09：LLM 数据工作负载、3FS 适配边界与本地存储基线。"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_data_engineering.contracts import STORAGE_SCHEMA_VERSION, STORAGE_VERSION
from ai_data_engineering.pipeline import sha256_file


@dataclass(frozen=True)
class StorageConfig:
    """本地存储工作负载配置。"""

    sequential_repeats: int = 32
    random_read_count: int = 256
    checkpoint_writers: int = 4
    seed: int = 2026

    def validate(self) -> None:
        if self.sequential_repeats < 1 or self.random_read_count < 1 or self.checkpoint_writers < 1:
            raise ValueError("存储工作负载参数必须大于 0")


def _payload_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _observation(durations: list[float], total_bytes: int, wall_seconds: float | None = None) -> dict[str, Any]:
    elapsed = wall_seconds if wall_seconds is not None else sum(durations)
    return {
        "operation_count": len(durations),
        "latency_p50_ms": round(_percentile(durations, 0.5) * 1000, 6),
        "latency_p99_ms": round(_percentile(durations, 0.99) * 1000, 6),
        "wall_seconds": round(elapsed, 6),
        "throughput_mib_per_second": round(total_bytes / max(elapsed, 1e-12) / (1024 * 1024), 3),
    }


def _timed(call: Callable[[], int]) -> tuple[float, int]:
    started = time.perf_counter()
    byte_count = call()
    return time.perf_counter() - started, byte_count


def _build_line_index(shard_paths: list[Path], base_dir: Path) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for path in shard_paths:
        offset = 0
        with path.open("rb") as file:
            for line_number, line in enumerate(file):
                index.append(
                    {
                        "path": str(path.relative_to(base_dir)),
                        "line_number": line_number,
                        "offset": offset,
                        "length": len(line),
                    }
                )
                offset += len(line)
    return index


def _storage_review(capabilities: dict[str, bool], decision: str) -> str:
    return "\n".join(
        [
            "# 3FS 本地架构评审",
            "",
            f"决策：`{decision}`",
            "",
            "## 前置条件",
            "",
            f"- Linux：{capabilities['linux']}",
            f"- RDMA Device：{capabilities['rdma_device']}",
            f"- NVMe Device：{capabilities['nvme_device']}",
            f"- 多节点实验床：{capabilities['multi_node_testbed']}",
            "",
            "## 工作负载到 3FS 的映射",
            "",
            "- 顺序 Training Shard 扫描：Client 并发读与 Storage Service 吞吐；",
            "- 随机 Sample 读取：Metadata/Client 路径、尾延迟和小请求放大；",
            "- Shuffle 中间写：共享文件系统命名空间、并发写与中间结果复用；",
            "- Checkpoint 并行写：多 Rank 大对象写入、耐久性和恢复时间目标；",
            "- CRAQ、FoundationDB、RDMA 与 NVMe 属于 3FS 架构证据，本实验未在本机验证其运行效果。",
            "",
            "## 当前建议",
            "",
            "先保留本地文件系统基线；有共享存储条件时对比 MinIO/NFS；只有具备专用 Linux、RDMA、NVMe 和",
            "多节点实验床后，才进入 3FS 集群支线。",
            "",
        ]
    )


def run_storage_lab(
    input_dir: Path,
    compute_dir: Path,
    recovery_dir: Path,
    output_dir: Path,
    config: StorageConfig | None = None,
) -> dict[str, Any]:
    """执行四类 LLM 数据 I/O，并形成 3FS Go/No-Go 证据。"""

    config = config or StorageConfig()
    config.validate()
    version = json.loads((input_dir / "dataset_version_manifest.json").read_text(encoding="utf-8"))
    compute = json.loads((compute_dir / "compute_report.json").read_text(encoding="utf-8"))
    recovery = json.loads((recovery_dir / "recovery_report.json").read_text(encoding="utf-8"))
    shard_paths = [input_dir / shard["path"] for shard in version["shards"]]
    checkpoint_path = recovery_dir / recovery["artifacts"]["resumed_final"]
    shuffle_sources = sorted((compute_dir / "partitions-salted").glob("*.jsonl"))
    if not shuffle_sources:
        raise ValueError("Lab 06 没有 Shuffle Partition")

    output_dir.mkdir(parents=True, exist_ok=True)
    for directory_name in ("shuffle-writes", "checkpoint-writes"):
        directory = output_dir / directory_name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()

    sequential_durations: list[float] = []
    sequential_bytes = 0
    for _ in range(config.sequential_repeats):
        for path in shard_paths:
            duration, byte_count = _timed(lambda current=path: len(current.read_bytes()))
            sequential_durations.append(duration)
            sequential_bytes += byte_count

    line_index = _build_line_index(shard_paths, input_dir)
    index_path = output_dir / "random_read_index.json"
    index_path.write_text(json.dumps(line_index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generator = random.Random(config.seed)
    random_durations: list[float] = []
    random_bytes = 0
    for _ in range(config.random_read_count):
        entry = line_index[generator.randrange(len(line_index))]

        def read_line(current: dict[str, Any] = entry) -> int:
            with (input_dir / current["path"]).open("rb") as file:
                file.seek(current["offset"])
                return len(file.read(current["length"]))

        duration, byte_count = _timed(read_line)
        random_durations.append(duration)
        random_bytes += byte_count

    shuffle_durations: list[float] = []
    shuffle_bytes = 0
    shuffle_artifacts: list[dict[str, Any]] = []
    for source in shuffle_sources:
        target = output_dir / "shuffle-writes" / source.name

        def write_shuffle(current_source: Path = source, current_target: Path = target) -> int:
            content = current_source.read_bytes()
            current_target.write_bytes(content)
            return len(content)

        duration, byte_count = _timed(write_shuffle)
        shuffle_durations.append(duration)
        shuffle_bytes += byte_count
        shuffle_artifacts.append(
            {"path": str(target.relative_to(output_dir)), "sha256": sha256_file(target), "bytes": byte_count}
        )

    checkpoint_content = checkpoint_path.read_bytes()

    def write_checkpoint(rank: int) -> tuple[float, int, Path]:
        target = output_dir / "checkpoint-writes" / f"rank-{rank:05d}.pt"
        started = time.perf_counter()
        target.write_bytes(checkpoint_content)
        return time.perf_counter() - started, len(checkpoint_content), target

    checkpoint_wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.checkpoint_writers) as executor:
        checkpoint_results = list(executor.map(write_checkpoint, range(config.checkpoint_writers)))
    checkpoint_wall_seconds = time.perf_counter() - checkpoint_wall_started
    checkpoint_durations = [result[0] for result in checkpoint_results]
    checkpoint_bytes = sum(result[1] for result in checkpoint_results)
    checkpoint_artifacts = [
        {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path), "bytes": byte_count}
        for _, byte_count, path in checkpoint_results
    ]

    capabilities = {
        "linux": sys.platform.startswith("linux"),
        "rdma_device": Path("/dev/infiniband").exists(),
        "nvme_device": any(Path("/dev").glob("nvme*n*")),
        "multi_node_testbed": False,
    }
    threefs_ready = all(capabilities.values())
    decision = "GO_FOR_3FS_CLUSTER_EXPERIMENT" if threefs_ready else "NO_GO_FOR_3FS_CLUSTER_ON_CURRENT_HOST"
    review_path = output_dir / "storage_review.md"
    review_path.write_text(_storage_review(capabilities, decision), encoding="utf-8")
    semantic = {
        "storage_version": STORAGE_VERSION,
        "inputs": {
            "dataset_version_fingerprint": version["dataset_version_fingerprint"],
            "compute_fingerprint": compute["compute_fingerprint"],
            "recovery_fingerprint": recovery["recovery_fingerprint"],
            "training_shards": [
                {"path": shard["path"], "sha256": shard["sha256"]} for shard in version["shards"]
            ],
            "checkpoint_model_state_sha256": recovery["metrics"]["resumed_model_state_sha256"],
        },
        "config": asdict(config),
        "workloads": {
            "sequential_shard_scan": {
                "operation_count": len(sequential_durations),
                "logical_bytes": sequential_bytes,
            },
            "random_sample_read": {"operation_count": len(random_durations), "logical_bytes": random_bytes},
            "shuffle_intermediate_write": {
                "operation_count": len(shuffle_durations),
                "logical_bytes": shuffle_bytes,
            },
            "parallel_checkpoint_write": {
                "operation_count": len(checkpoint_durations),
                "logical_bytes": checkpoint_bytes,
            },
        },
        "threefs_review": {
            "cluster_executed": False,
            "capabilities": capabilities,
            "decision": decision,
            "current_baseline_backend": "local-filesystem",
        },
        "artifacts": {
            "random_read_index": index_path.name,
            "storage_review": review_path.name,
            "shuffle_writes": shuffle_artifacts,
            "checkpoint_writes": checkpoint_artifacts,
        },
        "artifact_sha256": {
            "random_read_index": sha256_file(index_path),
            "storage_review": sha256_file(review_path),
        },
    }
    report = {
        "schema_version": STORAGE_SCHEMA_VERSION,
        **semantic,
        "observations": {
            "sequential_shard_scan": _observation(sequential_durations, sequential_bytes),
            "random_sample_read": _observation(random_durations, random_bytes),
            "shuffle_intermediate_write": _observation(shuffle_durations, shuffle_bytes),
            "parallel_checkpoint_write": _observation(
                checkpoint_durations, checkpoint_bytes, wall_seconds=checkpoint_wall_seconds
            ),
            "note": "缓存、文件系统、设备和后台负载都会影响本机数字；只作为当前环境基线",
        },
        "evidence_boundary": [
            "四类 I/O 在本地文件系统真实执行",
            "没有执行 3FS 集群，不提供 RDMA、CRAQ、FoundationDB 或多节点性能结论",
            "本地微型文件的吞吐与尾延迟不能外推训练集群",
        ],
    }
    report["storage_fingerprint"] = _payload_sha256(semantic)
    (output_dir / "storage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def validate_storage_lab(input_dir: Path, compute_dir: Path, recovery_dir: Path, output_dir: Path) -> dict[str, Any]:
    """验证存储报告的上游、工作负载产物、决策边界和稳定指纹。"""

    report = json.loads((output_dir / "storage_report.json").read_text(encoding="utf-8"))
    version = json.loads((input_dir / "dataset_version_manifest.json").read_text(encoding="utf-8"))
    compute = json.loads((compute_dir / "compute_report.json").read_text(encoding="utf-8"))
    recovery = json.loads((recovery_dir / "recovery_report.json").read_text(encoding="utf-8"))
    expected = (
        version["dataset_version_fingerprint"],
        compute["compute_fingerprint"],
        recovery["recovery_fingerprint"],
    )
    actual = (
        report["inputs"]["dataset_version_fingerprint"],
        report["inputs"]["compute_fingerprint"],
        report["inputs"]["recovery_fingerprint"],
    )
    if actual != expected:
        raise ValueError("Storage Report 上游绑定不一致")
    for shard in report["inputs"]["training_shards"]:
        if sha256_file(input_dir / shard["path"]) != shard["sha256"]:
            raise ValueError(f"Storage Shard 哈希不一致: {shard['path']}")
    for name in ("random_read_index", "storage_review"):
        if sha256_file(output_dir / report["artifacts"][name]) != report["artifact_sha256"][name]:
            raise ValueError(f"Storage 产物哈希不一致: {name}")
    for group in ("shuffle_writes", "checkpoint_writes"):
        for artifact in report["artifacts"][group]:
            if sha256_file(output_dir / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Storage 产物哈希不一致: {artifact['path']}")
    if report["threefs_review"]["cluster_executed"]:
        raise ValueError("本地 Lab 不应声称执行 3FS 集群")
    semantic_keys = (
        "storage_version",
        "inputs",
        "config",
        "workloads",
        "threefs_review",
        "artifacts",
        "artifact_sha256",
    )
    semantic = {key: report[key] for key in semantic_keys}
    if _payload_sha256(semantic) != report["storage_fingerprint"]:
        raise ValueError("Storage Fingerprint 不一致")
    return {
        "storage_fingerprint": report["storage_fingerprint"],
        "workload_count": len(report["workloads"]),
        "threefs_cluster_executed": False,
        "decision": report["threefs_review"]["decision"],
    }
