"""Lab 05：确定性数据混合、Packing、Sharding 与 Dataset Card。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_data_engineering.contracts import DATASET_VERSION_SCHEMA_VERSION, DATASET_VERSION_VERSION
from ai_data_engineering.pipeline import build_dataset, sha256_file, validate_dataset


@dataclass(frozen=True)
class VersionConfig:
    """训练数据版本构建配置。"""

    vocab_size: int = 320
    sequence_length: int = 64
    max_sequences_per_shard: int = 4
    seed: int = 42

    def validate(self) -> None:
        if self.vocab_size < 260:
            raise ValueError("vocab_size 至少为 260")
        if self.sequence_length < 2:
            raise ValueError("sequence_length 至少为 2")
        if self.max_sequences_per_shard < 1:
            raise ValueError("max_sequences_per_shard 至少为 1")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_mix_spec(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    weights = value.get("domain_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("mix spec 缺少 domain_weights")
    parsed = {str(key): float(weight) for key, weight in weights.items()}
    if any(weight <= 0 for weight in parsed.values()) or abs(sum(parsed.values()) - 1.0) > 1e-9:
        raise ValueError("domain_weights 必须全部大于 0 且总和为 1")
    return parsed


def _quotas(weights: dict[str, float], total: int) -> dict[str, int]:
    exact = {domain: weight * total for domain, weight in weights.items()}
    quota = {domain: int(value) for domain, value in exact.items()}
    remaining = total - sum(quota.values())
    order = sorted(weights, key=lambda domain: (-(exact[domain] - quota[domain]), domain))
    for domain in order[:remaining]:
        quota[domain] += 1
    return quota


def _deterministic_mix(
    records: list[dict[str, Any]], weights: dict[str, float], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {domain: [] for domain in weights}
    for record in records:
        domain = record.get("metadata", {}).get("domain")
        if domain not in groups:
            raise ValueError(f"文档 {record.get('source_id')} 的 domain 不在 mix spec: {domain}")
        groups[domain].append(record)
    target = _quotas(weights, len(records))
    shortages = {
        domain: target[domain] - len(groups[domain]) for domain in weights if target[domain] > len(groups[domain])
    }
    if shortages:
        raise ValueError(f"领域样本不足以满足确定性配比: {shortages}")
    selected: list[dict[str, Any]] = []
    for domain in sorted(groups):
        ranked = sorted(
            groups[domain],
            key=lambda record: hashlib.sha256(f"{seed}:{record['source_id']}".encode()).hexdigest(),
        )
        selected.extend(ranked[: target[domain]])
    selected.sort(key=lambda record: hashlib.sha256(f"mix:{seed}:{record['source_id']}".encode()).hexdigest())
    actual = Counter(record["metadata"]["domain"] for record in selected)
    return selected, {
        "target_document_count": len(records),
        "selected_document_count": len(selected),
        "target_counts": target,
        "actual_counts": dict(sorted(actual.items())),
        "actual_weights": {domain: round(actual[domain] / len(selected), 6) for domain in sorted(actual)},
    }


def _dataset_card(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    lines = [
        "# Dataset Card：Lab 05 Training Version",
        "",
        "## 来源与治理",
        "",
        "- 输入只来自 Lab 04 接受文档；",
        "- 每个文档保留许可、安全与 `quality_audit`；",
        "- 数据仅用于项目教学和本地 Tiny LM 实验。",
        "",
        "## 配比与规模",
        "",
        f"- 文档数：{metrics['document_count']}",
        f"- Sequence 数：{metrics['sequence_count']}",
        f"- 可训练 Token：{metrics['trainable_token_count']}",
        f"- Packing 利用率：{metrics['packing_efficiency']}",
        f"- Shard 数：{metrics['shard_count']}",
        f"- 领域分布：`{json.dumps(metrics['domain_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## 已知边界",
        "",
        "- 领域配比是项目原创微型语料的确定性演示，不代表生产最优混合；",
        "- Byte-level BPE 在本数据上训练，不代表 DeepSeek-V3 Tokenizer；",
        "- Shard 使用 JSONL 便于审计，不是训练吞吐最优格式。",
        "",
    ]
    return "\n".join(lines)


def build_dataset_version(
    input_path: Path,
    quality_report_path: Path,
    mix_spec_path: Path,
    output_dir: Path,
    config: VersionConfig | None = None,
) -> dict[str, Any]:
    """从质量准入文档构建可追溯训练数据版本。"""

    config = config or VersionConfig()
    config.validate()
    quality_report = json.loads(quality_report_path.read_text(encoding="utf-8"))
    if quality_report.get("quality_fingerprint") is None:
        raise ValueError("Quality Report 缺少 quality_fingerprint")
    records = _load_jsonl(input_path)
    if not records:
        raise ValueError("没有可用于构建版本的接受文档")
    if any(record.get("metadata", {}).get("quality_audit", {}).get("decision") != "accept" for record in records):
        raise ValueError("输入包含未通过质量准入的文档")
    weights = _load_mix_spec(mix_spec_path)
    mixed, mix_metrics = _deterministic_mix(records, weights, config.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    mixed_path = output_dir / "mixed_documents.jsonl"
    _write_jsonl(mixed_path, mixed)
    dataset_dir = output_dir / "dataset"
    dataset_manifest = build_dataset(
        mixed_path,
        dataset_dir,
        vocab_size=config.vocab_size,
        sequence_length=config.sequence_length,
    )
    sequences = _load_jsonl(dataset_dir / dataset_manifest["artifacts"]["sequences"])
    normalized = _load_jsonl(dataset_dir / dataset_manifest["artifacts"]["normalized_documents"])
    by_doc_id = {record["doc_id"]: record for record in normalized}

    shard_dir = output_dir / "training_shards"
    shard_dir.mkdir(exist_ok=True)
    shards: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(sequences), config.max_sequences_per_shard)):
        shard_records = sequences[start : start + config.max_sequences_per_shard]
        shard_name = f"shard-{index:05d}.jsonl"
        shard_path = shard_dir / shard_name
        _write_jsonl(shard_path, shard_records)
        source_ids = sorted(
            {by_doc_id[span["doc_id"]]["source_id"] for record in shard_records for span in record["provenance"]}
        )
        domains = Counter(
            by_doc_id[span["doc_id"]]["metadata"]["domain"] for record in shard_records for span in record["provenance"]
        )
        shards.append(
            {
                "path": f"training_shards/{shard_name}",
                "sha256": sha256_file(shard_path),
                "sequence_count": len(shard_records),
                "trainable_token_count": sum(sum(record["loss_mask"]) for record in shard_records),
                "source_ids": source_ids,
                "provenance_span_domain_counts": dict(sorted(domains.items())),
            }
        )
        for record in shard_records:
            lineage.append(
                {
                    "sequence_id": record["sequence_id"],
                    "shard": shard_name,
                    "sources": [
                        {
                            "doc_id": span["doc_id"],
                            "source_id": by_doc_id[span["doc_id"]]["source_id"],
                            "domain": by_doc_id[span["doc_id"]]["metadata"]["domain"],
                            "quality_version": by_doc_id[span["doc_id"]]["metadata"]["quality_audit"][
                                "quality_version"
                            ],
                            "sequence_token_start": span["sequence_token_start"],
                            "sequence_token_end": span["sequence_token_end"],
                        }
                        for span in record["provenance"]
                    ],
                }
            )

    lineage_path = output_dir / "sequence_lineage.jsonl"
    _write_jsonl(lineage_path, lineage)
    domain_counts = Counter(record["metadata"]["domain"] for record in mixed)
    shard_sizes = [shard["sequence_count"] for shard in shards]
    metrics = {
        "document_count": len(mixed),
        "domain_counts": dict(sorted(domain_counts.items())),
        "sequence_count": len(sequences),
        "trainable_token_count": dataset_manifest["metrics"]["trainable_token_count"],
        "packing_efficiency": dataset_manifest["metrics"]["packing_efficiency"],
        "shard_count": len(shards),
        "shard_sequence_min": min(shard_sizes),
        "shard_sequence_max": max(shard_sizes),
        "lineage_coverage": round(len(lineage) / len(sequences), 6),
    }
    card_path = output_dir / "dataset_card.md"
    card_path.write_text(_dataset_card({"metrics": metrics}), encoding="utf-8")
    semantic = {
        "dataset_version": DATASET_VERSION_VERSION,
        "inputs": {
            "accepted_documents_sha256": sha256_file(input_path),
            "quality_fingerprint": quality_report["quality_fingerprint"],
            "mix_spec_sha256": sha256_file(mix_spec_path),
        },
        "config": asdict(config),
        "mix": {"domain_weights": weights, **mix_metrics},
        "dataset": {
            "pipeline_fingerprint": dataset_manifest["pipeline_fingerprint"],
            "tokenizer_sha256": dataset_manifest["tokenizer_sha256"],
            "manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
        "metrics": metrics,
        "shards": shards,
        "artifacts": {
            "mixed_documents": "mixed_documents.jsonl",
            "dataset": "dataset",
            "sequence_lineage": "sequence_lineage.jsonl",
            "dataset_card": "dataset_card.md",
        },
        "artifact_sha256": {
            "mixed_documents": sha256_file(mixed_path),
            "sequence_lineage": sha256_file(lineage_path),
            "dataset_card": sha256_file(card_path),
        },
    }
    manifest = {
        "schema_version": DATASET_VERSION_SCHEMA_VERSION,
        **semantic,
        "evidence_boundary": [
            "领域权重只用于原创微型语料的确定性演示",
            "JSONL Shard 优先可审计性，不代表生产训练格式",
            "本地 Shard 均衡不证明分布式 DataLoader 吞吐",
        ],
    }
    manifest["dataset_version_fingerprint"] = _sha256_payload(semantic)
    _write_json(output_dir / "dataset_version_manifest.json", manifest)
    return manifest


def validate_dataset_version(
    input_path: Path, quality_report_path: Path, mix_spec_path: Path, output_dir: Path
) -> dict[str, Any]:
    """验证 Dataset Version 的上游绑定、Shard 完整性与血缘覆盖。"""

    manifest_path = output_dir / "dataset_version_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != DATASET_VERSION_SCHEMA_VERSION
        or manifest.get("dataset_version") != DATASET_VERSION_VERSION
    ):
        raise ValueError("不支持的 Dataset Version 版本")
    expected_inputs = {
        "accepted_documents_sha256": sha256_file(input_path),
        "quality_fingerprint": json.loads(quality_report_path.read_text(encoding="utf-8"))["quality_fingerprint"],
        "mix_spec_sha256": sha256_file(mix_spec_path),
    }
    if manifest["inputs"] != expected_inputs:
        raise ValueError("Dataset Version 未绑定当前上游输入")
    dataset_dir = output_dir / manifest["artifacts"]["dataset"]
    dataset_result = validate_dataset(dataset_dir)
    dataset_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if dataset_result["pipeline_fingerprint"] != manifest["dataset"]["pipeline_fingerprint"]:
        raise ValueError("Dataset Pipeline Fingerprint 不一致")
    if sha256_file(dataset_dir / "manifest.json") != manifest["dataset"]["manifest_sha256"]:
        raise ValueError("Dataset Manifest 哈希不一致")
    all_shard_records: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        shard_path = output_dir / shard["path"]
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError("Training Shard 哈希不一致")
        records = _load_jsonl(shard_path)
        if len(records) != shard["sequence_count"]:
            raise ValueError("Training Shard Sequence 数不一致")
        all_shard_records.extend(records)
    original_sequences = _load_jsonl(dataset_dir / dataset_manifest["artifacts"]["sequences"])
    if all_shard_records != original_sequences:
        raise ValueError("Training Shard 无法无损还原 Dataset Sequence")
    lineage = _load_jsonl(output_dir / manifest["artifacts"]["sequence_lineage"])
    if {record["sequence_id"] for record in lineage} != {record["sequence_id"] for record in original_sequences}:
        raise ValueError("Sequence Lineage 覆盖不完整")
    for name, expected_hash in manifest["artifact_sha256"].items():
        relative = manifest["artifacts"][name]
        if sha256_file(output_dir / relative) != expected_hash:
            raise ValueError(f"{name} 产物哈希不一致")
    semantic = {
        key: manifest[key]
        for key in (
            "dataset_version",
            "inputs",
            "config",
            "mix",
            "dataset",
            "metrics",
            "shards",
            "artifacts",
            "artifact_sha256",
        )
    }
    if _sha256_payload(semantic) != manifest["dataset_version_fingerprint"]:
        raise ValueError("Dataset Version Fingerprint 不一致")
    return {
        "dataset_version_fingerprint": manifest["dataset_version_fingerprint"],
        "document_count": manifest["metrics"]["document_count"],
        "sequence_count": manifest["metrics"]["sequence_count"],
        "shard_count": manifest["metrics"]["shard_count"],
        "lineage_coverage": manifest["metrics"]["lineage_coverage"],
    }
