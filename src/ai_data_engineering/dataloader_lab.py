"""Lab 07：Map/Iterable DataLoader 与本地 I/O Profiling。"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from ai_data_engineering.contracts import DATALOADER_SCHEMA_VERSION, DATALOADER_VERSION
from ai_data_engineering.pipeline import sha256_file


@dataclass(frozen=True)
class DataLoaderConfig:
    """DataLoader 实验配置。"""

    batch_size: int = 4
    repeat_epochs: int = 32
    worker_counts: tuple[int, ...] = (0, 2)

    def validate(self) -> None:
        if self.batch_size < 1 or self.repeat_epochs < 1:
            raise ValueError("batch_size 和 repeat_epochs 必须大于 0")
        if not self.worker_counts or any(count < 0 for count in self.worker_counts):
            raise ValueError("worker_counts 必须包含非负整数")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _payload_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _tensor_record(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, str]:
    return (
        torch.tensor(record["input_ids"], dtype=torch.long),
        torch.tensor(record["loss_mask"], dtype=torch.long),
        record["sequence_id"],
    )


class MapSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """启动时把全部 Shard 加载到内存的 Map-style Dataset。"""

    def __init__(self, shard_paths: list[Path], repeat_epochs: int) -> None:
        self.records = [record for path in shard_paths for record in _read_jsonl(path)]
        self.repeat_epochs = repeat_epochs

    def __len__(self) -> int:
        return len(self.records) * self.repeat_epochs

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return _tensor_record(self.records[index % len(self.records)])


class IterableSequenceDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """逐 Shard 流式读取，并按 Worker 分配文件。"""

    def __init__(self, shard_paths: list[Path], repeat_epochs: int) -> None:
        self.shard_paths = shard_paths
        self.repeat_epochs = repeat_epochs

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, str]]:
        worker = get_worker_info()
        paths = self.shard_paths if worker is None else self.shard_paths[worker.id :: worker.num_workers]
        for _ in range(self.repeat_epochs):
            for path in paths:
                for record in _read_jsonl(path):
                    yield _tensor_record(record)


def _profile(loader: DataLoader[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    iterator = iter(loader)
    wait_seconds: list[float] = []
    sample_count = 0
    trainable_tokens = 0
    sequence_ids: set[str] = set()
    while True:
        wait_started = time.perf_counter()
        try:
            _, loss_mask, identifiers = next(iterator)
        except StopIteration:
            break
        wait_seconds.append(time.perf_counter() - wait_started)
        sample_count += int(loss_mask.shape[0])
        trainable_tokens += int(loss_mask.sum().item())
        sequence_ids.update(identifiers)
    duration = time.perf_counter() - started
    stable = {
        "sample_count": sample_count,
        "trainable_token_count": trainable_tokens,
        "unique_sequence_count": len(sequence_ids),
        "batch_count": len(wait_seconds),
    }
    observations = {
        "duration_seconds": round(duration, 6),
        "first_batch_wait_seconds": round(wait_seconds[0], 6),
        "batch_wait_p50_seconds": round(statistics.median(wait_seconds), 6),
        "batch_wait_p99_seconds": round(_percentile(wait_seconds, 0.99), 6),
        "samples_per_second": round(sample_count / duration, 3),
        "trainable_tokens_per_second": round(trainable_tokens / duration, 3),
    }
    return stable, observations


def _decision_markdown(source_bytes: int, sequence_count: int) -> str:
    return "\n".join(
        [
            "# DataLoader 瓶颈决策树",
            "",
            f"当前 Shard：{source_bytes} Bytes，{sequence_count} Sequences。",
            "",
            "1. 首批等待高、后续稳定：检查 Worker 启动、文件枚举和预热；",
            "2. 每批等待都高：分别测存储读取、解压/解析、Tokenization 与 Batch Collate；",
            "3. 数据可安全放入内存且反复训练：优先 Map-style，减少重复解析；",
            "4. 数据无法放入内存或持续到达：使用 Iterable-style，并按 Shard 分配 Worker；",
            "5. Worker 数大于 Shard 数：先增大 Shard 数或调整文件组织，盲目加 Worker 不会增加并行度；",
            "6. CPU DataLoader 已快但 GPU 仍等待：再检查 Host-to-Device、Pinned Memory 和训练计算重叠。",
            "",
            "本地 CPU 微型实验只定位数据供给路径，不证明 GPU 集群最优参数。",
            "",
        ]
    )


def run_dataloader_lab(
    input_dir: Path,
    compute_report_path: Path,
    output_dir: Path,
    config: DataLoaderConfig | None = None,
) -> dict[str, Any]:
    """对同一 Training Shard 执行 Map/Iterable DataLoader Profile。"""

    config = config or DataLoaderConfig()
    config.validate()
    version_path = input_dir / "dataset_version_manifest.json"
    version = json.loads(version_path.read_text(encoding="utf-8"))
    compute = json.loads(compute_report_path.read_text(encoding="utf-8"))
    shard_paths = [input_dir / shard["path"] for shard in version["shards"]]
    source_bytes = sum(path.stat().st_size for path in shard_paths)
    expected_samples = version["metrics"]["sequence_count"] * config.repeat_epochs
    expected_tokens = version["metrics"]["trainable_token_count"] * config.repeat_epochs

    profiles: list[dict[str, Any]] = []
    observations: dict[str, Any] = {}
    map_started = time.perf_counter()
    map_dataset = MapSequenceDataset(shard_paths, config.repeat_epochs)
    map_preload_seconds = time.perf_counter() - map_started
    map_stable, map_observations = _profile(DataLoader(map_dataset, batch_size=config.batch_size, num_workers=0))
    profiles.append(
        {
            "name": "map-workers-0",
            "style": "map",
            "workers": 0,
            "physical_file_read_cycles": 1,
            **map_stable,
        }
    )
    observations["map-workers-0"] = {"preload_seconds": round(map_preload_seconds, 6), **map_observations}

    for workers in sorted(set(config.worker_counts)):
        iterable_dataset = IterableSequenceDataset(shard_paths, config.repeat_epochs)
        loader_kwargs: dict[str, Any] = {"batch_size": config.batch_size, "num_workers": workers}
        if workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        stable, profile_observations = _profile(DataLoader(iterable_dataset, **loader_kwargs))
        name = f"iterable-workers-{workers}"
        profiles.append(
            {
                "name": name,
                "style": "iterable",
                "workers": workers,
                "physical_file_read_cycles": config.repeat_epochs,
                **stable,
            }
        )
        observations[name] = profile_observations

    for profile in profiles:
        if profile["sample_count"] != expected_samples or profile["trainable_token_count"] != expected_tokens:
            raise RuntimeError(f"DataLoader 丢失或重复样本: {profile['name']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "dataloader_decision.md"
    decision_path.write_text(_decision_markdown(source_bytes, version["metrics"]["sequence_count"]), encoding="utf-8")
    semantic = {
        "dataloader_version": DATALOADER_VERSION,
        "inputs": {
            "dataset_version_fingerprint": version["dataset_version_fingerprint"],
            "compute_fingerprint": compute["compute_fingerprint"],
            "shards": [
                {"path": shard["path"], "sha256": shard["sha256"], "sequence_count": shard["sequence_count"]}
                for shard in version["shards"]
            ],
        },
        "config": {**asdict(config), "worker_counts": list(config.worker_counts)},
        "dataset": {
            "source_bytes": source_bytes,
            "sequence_count": version["metrics"]["sequence_count"],
            "trainable_token_count": version["metrics"]["trainable_token_count"],
            "expected_delivered_sample_count": expected_samples,
            "expected_delivered_trainable_tokens": expected_tokens,
        },
        "profiles": profiles,
        "artifacts": {"decision_tree": "dataloader_decision.md"},
        "artifact_sha256": {"decision_tree": sha256_file(decision_path)},
    }
    report = {
        "schema_version": DATALOADER_SCHEMA_VERSION,
        **semantic,
        "observations": observations,
        "evidence_boundary": [
            "Samples 与有效 Token 完整性是确定性证据，耗时和吞吐是本机观测",
            "Map-style 预加载一次；Iterable-style 每个 Epoch 重新读取 Shard，操作系统缓存仍可能命中",
            "没有 GPU 时不能把 CPU DataLoader 吞吐解释成 GPU 利用率",
        ],
    }
    report["dataloader_fingerprint"] = _payload_sha256(semantic)
    (output_dir / "dataloader_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def validate_dataloader_lab(input_dir: Path, compute_report_path: Path, output_dir: Path) -> dict[str, Any]:
    """验证 DataLoader 上游、Shard、样本完整性和稳定指纹。"""

    report = json.loads((output_dir / "dataloader_report.json").read_text(encoding="utf-8"))
    version = json.loads((input_dir / "dataset_version_manifest.json").read_text(encoding="utf-8"))
    compute = json.loads(compute_report_path.read_text(encoding="utf-8"))
    if report["inputs"]["dataset_version_fingerprint"] != version["dataset_version_fingerprint"]:
        raise ValueError("Dataset Version Fingerprint 不一致")
    if report["inputs"]["compute_fingerprint"] != compute["compute_fingerprint"]:
        raise ValueError("Compute Fingerprint 不一致")
    for shard in report["inputs"]["shards"]:
        if sha256_file(input_dir / shard["path"]) != shard["sha256"]:
            raise ValueError(f"DataLoader Shard 哈希不一致: {shard['path']}")
    expected_samples = report["dataset"]["expected_delivered_sample_count"]
    expected_tokens = report["dataset"]["expected_delivered_trainable_tokens"]
    for profile in report["profiles"]:
        if profile["sample_count"] != expected_samples or profile["trainable_token_count"] != expected_tokens:
            raise ValueError(f"DataLoader 样本完整性失败: {profile['name']}")
        if profile["unique_sequence_count"] != report["dataset"]["sequence_count"]:
            raise ValueError(f"DataLoader Sequence 覆盖失败: {profile['name']}")
    decision_path = output_dir / report["artifacts"]["decision_tree"]
    if sha256_file(decision_path) != report["artifact_sha256"]["decision_tree"]:
        raise ValueError("DataLoader 决策树哈希不一致")
    semantic_keys = ("dataloader_version", "inputs", "config", "dataset", "profiles", "artifacts", "artifact_sha256")
    semantic = {key: report[key] for key in semantic_keys}
    if _payload_sha256(semantic) != report["dataloader_fingerprint"]:
        raise ValueError("DataLoader Fingerprint 不一致")
    return {
        "dataloader_fingerprint": report["dataloader_fingerprint"],
        "profile_count": len(report["profiles"]),
        "sequence_count": report["dataset"]["sequence_count"],
        "all_profiles_complete": True,
    }
