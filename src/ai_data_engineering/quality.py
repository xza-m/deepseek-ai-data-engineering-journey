"""Lab 04：近似去重、评测污染与确定性质量治理。"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_data_engineering.contracts import QUALITY_SCHEMA_VERSION, QUALITY_VERSION
from ai_data_engineering.pipeline import normalize_text, sha256_file

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")


@dataclass(frozen=True)
class QualityConfig:
    """Lab 04 唯一质量规则入口。"""

    similarity_threshold: float = 0.45
    shingle_size: int = 3
    contamination_threshold: float = 0.45
    min_characters: int = 40
    allowed_licenses: tuple[str, ...] = ("project-original", "cc-by-4.0")

    def validate(self) -> None:
        if not 0 < self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold 必须位于 0 和 1 之间")
        if not 0 < self.contamination_threshold <= 1:
            raise ValueError("contamination_threshold 必须位于 0 和 1 之间")
        if self.shingle_size < 1:
            raise ValueError("shingle_size 至少为 1")
        if self.min_characters < 1:
            raise ValueError("min_characters 至少为 1")
        if not self.allowed_licenses:
            raise ValueError("allowed_licenses 不能为空")


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


def _load_documents(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = record.get("source_id")
            text = record.get("text")
            metadata = record.get("metadata", {})
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"{path.name} 第 {line_number} 行缺少 source_id")
            if source_id in seen:
                raise ValueError(f"source_id 重复: {source_id}")
            if not isinstance(text, str) or not normalize_text(text):
                raise ValueError(f"{path.name} 第 {line_number} 行缺少有效 text")
            if not isinstance(metadata, dict):
                raise ValueError(f"{path.name} 第 {line_number} 行 metadata 必须是对象")
            seen.add(source_id)
            records.append({"source_id": source_id, "text": normalize_text(text), "metadata": metadata})
    if not records:
        raise ValueError(f"输入中没有文档: {path}")
    return records


def _word_shingles(text: str, size: int) -> set[str]:
    words = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _quality_reasons(record: dict[str, Any], config: QualityConfig) -> list[str]:
    text = record["text"]
    metadata = record["metadata"]
    reasons: list[str] = []
    if len(text) < config.min_characters:
        reasons.append("too_short")
    if metadata.get("license") not in config.allowed_licenses:
        reasons.append("license_not_allowed")
    if metadata.get("safety", "safe") != "safe":
        reasons.append("unsafe_metadata")
    if EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text):
        reasons.append("pii_detected")
    return reasons


def _find_pairs(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    config: QualityConfig,
    threshold: float,
    same_collection: bool,
) -> list[dict[str, Any]]:
    left_shingles = {
        record["source_id"]: _word_shingles(record["text"], config.shingle_size) for record in left_records
    }
    right_shingles = {
        record["source_id"]: _word_shingles(record["text"], config.shingle_size) for record in right_records
    }
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(left_records):
        for right_index, right in enumerate(right_records):
            if same_collection and right_index <= left_index:
                continue
            similarity = _jaccard(left_shingles[left["source_id"]], right_shingles[right["source_id"]])
            if similarity >= threshold:
                pairs.append(
                    {
                        "left_source_id": left["source_id"],
                        "right_source_id": right["source_id"],
                        "similarity": round(similarity, 6),
                    }
                )
    return sorted(pairs, key=lambda item: (item["left_source_id"], item["right_source_id"]))


def _clusters(source_ids: list[str], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent = {source_id: source_id for source_id in source_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        union(pair["left_source_id"], pair["right_source_id"])
    grouped: dict[str, list[str]] = {}
    for source_id in source_ids:
        grouped.setdefault(find(source_id), []).append(source_id)
    return [
        {"cluster_id": f"cluster-{index:04d}", "members": sorted(members), "keep_source_id": min(members)}
        for index, members in enumerate(sorted(grouped.values()))
        if len(members) > 1
    ]


def _load_truth(path: Path, known_ids: set[str]) -> list[dict[str, Any]]:
    truth: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            left, right, duplicate = (
                record.get("left_source_id"),
                record.get("right_source_id"),
                record.get("is_duplicate"),
            )
            if left not in known_ids or right not in known_ids or left == right or not isinstance(duplicate, bool):
                raise ValueError(f"Truth 第 {line_number} 行不符合契约")
            pair = tuple(sorted((left, right)))
            if pair in seen_pairs:
                raise ValueError(f"Truth Pair 重复: {pair}")
            seen_pairs.add(pair)
            truth.append({"left_source_id": pair[0], "right_source_id": pair[1], "is_duplicate": duplicate})
    if not truth:
        raise ValueError("Truth 集不能为空")
    return truth


def _benchmark(truth: list[dict[str, Any]], records: list[dict[str, Any]], config: QualityConfig) -> dict[str, Any]:
    by_id = {record["source_id"]: record for record in records}
    outcomes: list[dict[str, Any]] = []
    tp = fp = fn = tn = 0
    for item in truth:
        similarity = _jaccard(
            _word_shingles(by_id[item["left_source_id"]]["text"], config.shingle_size),
            _word_shingles(by_id[item["right_source_id"]]["text"], config.shingle_size),
        )
        predicted = similarity >= config.similarity_threshold
        actual = item["is_duplicate"]
        tp += int(predicted and actual)
        fp += int(predicted and not actual)
        fn += int(not predicted and actual)
        tn += int(not predicted and not actual)
        outcomes.append({**item, "predicted_duplicate": predicted, "similarity": round(similarity, 6)})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
        "labeled_pair_count": len(truth),
        "outcomes": outcomes,
    }


def _analyze(train_path: Path, evaluation_path: Path, truth_path: Path, config: QualityConfig) -> dict[str, Any]:
    config.validate()
    train_records = _load_documents(train_path)
    evaluation_records = _load_documents(evaluation_path)
    truth = _load_truth(truth_path, {record["source_id"] for record in train_records})
    started_at = time.perf_counter()
    duplicate_pairs = _find_pairs(
        train_records, train_records, config, config.similarity_threshold, same_collection=True
    )
    clusters = _clusters([record["source_id"] for record in train_records], duplicate_pairs)
    contamination = _find_pairs(
        train_records,
        evaluation_records,
        config,
        config.contamination_threshold,
        same_collection=False,
    )
    benchmark = _benchmark(truth, train_records, config)
    duplicate_rejections = {
        member for cluster in clusters for member in cluster["members"] if member != cluster["keep_source_id"]
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for record in train_records:
        reasons = _quality_reasons(record, config)
        if record["source_id"] in duplicate_rejections:
            reasons.append("near_duplicate")
        reasons = sorted(set(reasons))
        audit = {
            "quality_version": QUALITY_VERSION,
            "decision": "reject" if reasons else "accept",
            "reasons": reasons,
        }
        output_record = {
            "source_id": record["source_id"],
            "text": record["text"],
            "metadata": {**record["metadata"], "quality_audit": audit},
        }
        if reasons:
            rejected.append(output_record)
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            accepted.append(output_record)
    duration = time.perf_counter() - started_at
    return {
        "train_records": train_records,
        "evaluation_records": evaluation_records,
        "accepted": accepted,
        "rejected": rejected,
        "duplicate_pairs": duplicate_pairs,
        "clusters": clusters,
        "contamination": contamination,
        "benchmark": benchmark,
        "reason_counts": dict(sorted(reason_counts.items())),
        "duration_seconds": duration,
    }


def run_quality_audit(
    train_path: Path,
    evaluation_path: Path,
    truth_path: Path,
    output_dir: Path,
    config: QualityConfig | None = None,
) -> dict[str, Any]:
    """运行质量审计并写出可训练文档和质量报告。"""

    config = config or QualityConfig()
    analysis = _analyze(train_path, evaluation_path, truth_path, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "accepted_documents": "accepted_documents.jsonl",
        "rejected_documents": "rejected_documents.jsonl",
        "duplicate_clusters": "duplicate_clusters.json",
        "contamination_pairs": "contamination_pairs.jsonl",
        "truth_outcomes": "truth_outcomes.jsonl",
    }
    _write_jsonl(output_dir / artifacts["accepted_documents"], analysis["accepted"])
    _write_jsonl(output_dir / artifacts["rejected_documents"], analysis["rejected"])
    _write_json(output_dir / artifacts["duplicate_clusters"], analysis["clusters"])
    _write_jsonl(output_dir / artifacts["contamination_pairs"], analysis["contamination"])
    _write_jsonl(output_dir / artifacts["truth_outcomes"], analysis["benchmark"]["outcomes"])
    artifact_sha256 = {name: sha256_file(output_dir / path) for name, path in artifacts.items()}
    semantic = {
        "quality_version": QUALITY_VERSION,
        "inputs": {
            "train_sha256": sha256_file(train_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "truth_sha256": sha256_file(truth_path),
        },
        "config": asdict(config),
        "metrics": {
            "input_document_count": len(analysis["train_records"]),
            "accepted_document_count": len(analysis["accepted"]),
            "rejected_document_count": len(analysis["rejected"]),
            "duplicate_pair_count": len(analysis["duplicate_pairs"]),
            "duplicate_cluster_count": len(analysis["clusters"]),
            "contamination_pair_count": len(analysis["contamination"]),
            "reason_counts": analysis["reason_counts"],
            "benchmark": {key: value for key, value in analysis["benchmark"].items() if key != "outcomes"},
        },
        "artifacts": artifacts,
        "artifact_sha256": artifact_sha256,
    }
    report = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        **semantic,
        "observations": {
            "duration_seconds": round(analysis["duration_seconds"], 6),
            "documents_per_second": round(len(analysis["train_records"]) / analysis["duration_seconds"], 3),
        },
        "evidence_boundary": [
            "Precision/Recall 只来自项目原创的小规模标注 Pair",
            "词级 Shingling 和阈值不是生产语义去重的通用最优方案",
            "PII 与安全规则只提供确定性教学证据，不替代合规审查",
        ],
    }
    report["quality_fingerprint"] = _sha256_payload(semantic)
    _write_json(output_dir / "quality_report.json", report)
    return report


def validate_quality_audit(
    train_path: Path, evaluation_path: Path, truth_path: Path, output_dir: Path
) -> dict[str, Any]:
    """重算 Lab 04 语义结果并验证产物哈希。"""

    report_path = output_dir / "quality_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"找不到 Quality Report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != QUALITY_SCHEMA_VERSION or report.get("quality_version") != QUALITY_VERSION:
        raise ValueError("不支持的质量报告版本")
    config_data = report["config"]
    config_data["allowed_licenses"] = tuple(config_data["allowed_licenses"])
    analysis = _analyze(train_path, evaluation_path, truth_path, QualityConfig(**config_data))
    expected_metrics = {
        "input_document_count": len(analysis["train_records"]),
        "accepted_document_count": len(analysis["accepted"]),
        "rejected_document_count": len(analysis["rejected"]),
        "duplicate_pair_count": len(analysis["duplicate_pairs"]),
        "duplicate_cluster_count": len(analysis["clusters"]),
        "contamination_pair_count": len(analysis["contamination"]),
        "reason_counts": analysis["reason_counts"],
        "benchmark": {key: value for key, value in analysis["benchmark"].items() if key != "outcomes"},
    }
    if expected_metrics != report["metrics"]:
        raise ValueError("质量指标与输入重算结果不一致")
    expected_inputs = {
        "train_sha256": sha256_file(train_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "truth_sha256": sha256_file(truth_path),
    }
    if expected_inputs != report["inputs"]:
        raise ValueError("质量报告未绑定当前输入")
    for name, relative_path in report["artifacts"].items():
        if sha256_file(output_dir / relative_path) != report["artifact_sha256"][name]:
            raise ValueError(f"{name} 产物哈希与报告不一致")
    semantic = {
        key: report[key] for key in ("quality_version", "inputs", "config", "metrics", "artifacts", "artifact_sha256")
    }
    if _sha256_payload(semantic) != report["quality_fingerprint"]:
        raise ValueError("Quality Fingerprint 与报告语义不一致")
    return {
        "quality_fingerprint": report["quality_fingerprint"],
        "accepted_document_count": report["metrics"]["accepted_document_count"],
        "precision": report["metrics"]["benchmark"]["precision"],
        "recall": report["metrics"]["benchmark"]["recall"],
        "contamination_pair_count": report["metrics"]["contamination_pair_count"],
    }
