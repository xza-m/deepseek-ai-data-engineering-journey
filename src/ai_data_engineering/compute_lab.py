"""Lab 06：DuckDB/Parquet、Partition、Shuffle、倾斜与失败恢复。"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from ai_data_engineering.contracts import COMPUTE_SCHEMA_VERSION, COMPUTE_VERSION
from ai_data_engineering.pipeline import sha256_file


@dataclass(frozen=True)
class ComputeConfig:
    """本地计算实验配置。"""

    repeat_factor: int = 64
    partition_count: int = 8
    hot_key_fraction: float = 0.75
    failure_partition: int = 3

    def validate(self) -> None:
        if self.repeat_factor < 1:
            raise ValueError("repeat_factor 至少为 1")
        if self.partition_count < 2:
            raise ValueError("partition_count 至少为 2")
        if not 0 < self.hot_key_fraction < 1:
            raise ValueError("hot_key_fraction 必须位于 0 和 1 之间")
        if not 0 <= self.failure_partition < self.partition_count:
            raise ValueError("failure_partition 超出分区范围")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _payload_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _partition(key: str, count: int) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % count


def _distribution(counts: list[int]) -> dict[str, Any]:
    mean = sum(counts) / len(counts)
    return {
        "counts": counts,
        "min": min(counts),
        "max": max(counts),
        "max_to_mean_ratio": round(max(counts) / mean, 6),
        "empty_partition_count": sum(value == 0 for value in counts),
    }


def _aggregate_python(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["domain"] for row in rows)
    chars = Counter()
    for row in rows:
        chars[row["domain"]] += row["char_count"]
    return [
        {"domain": domain, "row_count": counts[domain], "char_count": chars[domain]}
        for domain in sorted(counts)
    ]


def _run_recovery(partition_dir: Path, output_dir: Path, config: ComputeConfig) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    attempts: dict[str, int] = {}
    injected = False

    def execute(partition_index: int) -> None:
        nonlocal injected
        name = f"part-{partition_index:05d}"
        attempts[name] = attempts.get(name, 0) + 1
        if partition_index == config.failure_partition and not injected:
            injected = True
            raise RuntimeError("教学注入故障")
        records = _read_jsonl(partition_dir / f"{name}.jsonl")
        _write_json(output_dir / f"{name}.done.json", {"partition": partition_index, "row_count": len(records)})

    retry_count = 0
    for index in range(config.partition_count):
        try:
            execute(index)
        except RuntimeError:
            retry_count += 1
            execute(index)

    reused = 0
    for index in range(config.partition_count):
        marker = output_dir / f"part-{index:05d}.done.json"
        if marker.is_file():
            reused += 1
        else:
            execute(index)
    return {
        "injected_failure_partition": config.failure_partition,
        "retry_count": retry_count,
        "attempts": attempts,
        "completed_partition_count": len(list(output_dir.glob("*.done.json"))),
        "recovery_pass_reused_partition_count": reused,
    }


def run_compute_lab(input_dir: Path, output_dir: Path, config: ComputeConfig | None = None) -> dict[str, Any]:
    """执行可在笔记本复现的计算、倾斜与恢复实验。"""

    config = config or ComputeConfig()
    config.validate()
    manifest_path = input_dir / "dataset_version_manifest.json"
    documents_path = input_dir / "mixed_documents.jsonl"
    upstream = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_documents = _read_jsonl(documents_path)
    if not source_documents:
        raise ValueError("Lab 05 没有可计算文档")

    output_dir.mkdir(parents=True, exist_ok=True)
    for directory_name in ("partitions-baseline", "partitions-salted", "task-state"):
        directory = output_dir / directory_name
        if directory.exists():
            shutil.rmtree(directory)

    row_count = len(source_documents) * config.repeat_factor
    hot_count = int(row_count * config.hot_key_fraction)
    rows: list[dict[str, Any]] = []
    for row_id in range(row_count):
        document = source_documents[row_id % len(source_documents)]
        rows.append(
            {
                "row_id": row_id,
                "source_id": document["source_id"],
                "domain": document["metadata"]["domain"],
                "text": document["text"],
                "char_count": len(document["text"]),
                "join_key": "hot-key" if row_id < hot_count else document["source_id"],
            }
        )
    rows_path = output_dir / "expanded_documents.jsonl"
    _write_jsonl(rows_path, rows)

    python_started = time.perf_counter()
    python_aggregate = _aggregate_python(rows)
    python_seconds = time.perf_counter() - python_started
    parquet_path = output_dir / "documents.parquet"
    duckdb_started = time.perf_counter()
    connection = duckdb.connect()
    try:
        connection.execute("CREATE TABLE documents AS SELECT * FROM read_json_auto(?)", [str(rows_path)])
        duckdb_aggregate = [
            {"domain": domain, "row_count": count, "char_count": characters}
            for domain, count, characters in connection.execute(
                "SELECT domain, count(*) AS row_count, sum(char_count) AS char_count "
                "FROM documents GROUP BY domain ORDER BY domain"
            ).fetchall()
        ]
        explain_rows = connection.execute(
            "EXPLAIN SELECT domain, count(*), sum(char_count) FROM documents GROUP BY domain"
        ).fetchall()
        connection.execute("COPY documents TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(parquet_path)])
    finally:
        connection.close()
    duckdb_seconds = time.perf_counter() - duckdb_started
    if python_aggregate != duckdb_aggregate:
        raise RuntimeError("Python 与 DuckDB 聚合结果不一致")

    partition_sets: dict[str, list[list[dict[str, Any]]]] = {
        "baseline": [[] for _ in range(config.partition_count)],
        "salted": [[] for _ in range(config.partition_count)],
    }
    for row in rows:
        baseline_index = _partition(row["join_key"], config.partition_count)
        salted_key = (
            f"hot-key:{row['row_id'] % config.partition_count}"
            if row["join_key"] == "hot-key"
            else row["join_key"]
        )
        salted_index = _partition(salted_key, config.partition_count)
        partition_sets["baseline"][baseline_index].append(row)
        partition_sets["salted"][salted_index].append(row)

    partition_artifacts: list[dict[str, Any]] = []
    for strategy, partitions in partition_sets.items():
        directory = output_dir / f"partitions-{strategy}"
        for index, partition_rows in enumerate(partitions):
            path = directory / f"part-{index:05d}.jsonl"
            _write_jsonl(path, partition_rows)
            partition_artifacts.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": sha256_file(path),
                    "row_count": len(partition_rows),
                }
            )

    baseline_distribution = _distribution([len(partition) for partition in partition_sets["baseline"]])
    salted_distribution = _distribution([len(partition) for partition in partition_sets["salted"]])
    recovery = _run_recovery(output_dir / "partitions-salted", output_dir / "task-state", config)
    plan_path = output_dir / "duckdb_plan.json"
    _write_json(
        plan_path,
        {
            "engine": "duckdb",
            "explain": [{"type": row[0], "plan": row[1]} for row in explain_rows],
            "smallpond_concept_mapping": {
                "lazy_dataframe": "DuckDB SQL logical query",
                "partition": "partitions-baseline/*.jsonl",
                "shuffle": "按 join_key 哈希重分区",
                "skew_mitigation": "热点 Key 加盐后重分区",
                "task_recovery": "分区完成标记、单次重试与中间结果复用",
            },
            "smallpond_runtime_executed": False,
        },
    )
    semantic = {
        "compute_version": COMPUTE_VERSION,
        "inputs": {
            "dataset_version_fingerprint": upstream["dataset_version_fingerprint"],
            "dataset_version_manifest_sha256": sha256_file(manifest_path),
            "mixed_documents_sha256": sha256_file(documents_path),
        },
        "config": asdict(config),
        "engine": {"name": "duckdb", "parquet": True, "smallpond_runtime_executed": False},
        "metrics": {
            "source_document_count": len(source_documents),
            "expanded_row_count": len(rows),
            "aggregation_equal": python_aggregate == duckdb_aggregate,
            "python_aggregate": python_aggregate,
            "duckdb_aggregate": duckdb_aggregate,
            "baseline_partition_distribution": baseline_distribution,
            "salted_partition_distribution": salted_distribution,
            "skew_ratio_improvement": round(
                baseline_distribution["max_to_mean_ratio"] - salted_distribution["max_to_mean_ratio"], 6
            ),
            "recovery": recovery,
        },
        "artifacts": {
            "expanded_documents": "expanded_documents.jsonl",
            "parquet": "documents.parquet",
            "duckdb_plan": "duckdb_plan.json",
            "partition_files": partition_artifacts,
        },
        "artifact_sha256": {
            "expanded_documents": sha256_file(rows_path),
            "parquet": sha256_file(parquet_path),
            "duckdb_plan": sha256_file(plan_path),
        },
    }
    report = {
        "schema_version": COMPUTE_SCHEMA_VERSION,
        **semantic,
        "observations": {
            "python_aggregation_seconds": round(python_seconds, 6),
            "duckdb_read_aggregate_and_parquet_seconds": round(duckdb_seconds, 6),
            "note": "微型本地数据的耗时只用于学习 Profile，不能外推生产性能",
        },
        "evidence_boundary": [
            "本实验真实执行 DuckDB、Parquet、哈希分区、倾斜修复和失败注入",
            "smallpond 仅做概念映射，本机没有声称执行 smallpond 分布式 Runtime",
            "加盐只适合可拆分热点工作负载，真实 Join 还需要去盐或二阶段聚合",
        ],
    }
    report["compute_fingerprint"] = _payload_sha256(semantic)
    _write_json(output_dir / "compute_report.json", report)
    return report


def validate_compute_lab(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """验证 Lab 06 上游绑定、计算等价、分区产物和恢复证据。"""

    report = json.loads((output_dir / "compute_report.json").read_text(encoding="utf-8"))
    upstream_path = input_dir / "dataset_version_manifest.json"
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    if report["inputs"]["dataset_version_fingerprint"] != upstream["dataset_version_fingerprint"]:
        raise ValueError("Dataset Version Fingerprint 不一致")
    if report["inputs"]["dataset_version_manifest_sha256"] != sha256_file(upstream_path):
        raise ValueError("Dataset Version Manifest 哈希不一致")
    for name, relative_path in (
        ("expanded_documents", report["artifacts"]["expanded_documents"]),
        ("parquet", report["artifacts"]["parquet"]),
        ("duckdb_plan", report["artifacts"]["duckdb_plan"]),
    ):
        if sha256_file(output_dir / relative_path) != report["artifact_sha256"][name]:
            raise ValueError(f"Lab 06 产物哈希不一致: {name}")
    for artifact in report["artifacts"]["partition_files"]:
        path = output_dir / artifact["path"]
        if sha256_file(path) != artifact["sha256"] or len(_read_jsonl(path)) != artifact["row_count"]:
            raise ValueError(f"Partition 产物不一致: {artifact['path']}")
    metrics = report["metrics"]
    if not metrics["aggregation_equal"] or metrics["python_aggregate"] != metrics["duckdb_aggregate"]:
        raise ValueError("聚合等价性未通过")
    if metrics["skew_ratio_improvement"] <= 0:
        raise ValueError("热点 Key 加盐没有改善分区倾斜")
    recovery = metrics["recovery"]
    if recovery["retry_count"] != 1 or recovery["completed_partition_count"] != report["config"]["partition_count"]:
        raise ValueError("失败重试证据不完整")
    if recovery["recovery_pass_reused_partition_count"] != report["config"]["partition_count"]:
        raise ValueError("中间结果复用证据不完整")
    semantic_keys = ("compute_version", "inputs", "config", "engine", "metrics", "artifacts", "artifact_sha256")
    semantic = {key: report[key] for key in semantic_keys}
    if _payload_sha256(semantic) != report["compute_fingerprint"]:
        raise ValueError("Compute Fingerprint 不一致")
    return {
        "compute_fingerprint": report["compute_fingerprint"],
        "expanded_row_count": metrics["expanded_row_count"],
        "skew_ratio_improvement": metrics["skew_ratio_improvement"],
        "retry_count": recovery["retry_count"],
    }
