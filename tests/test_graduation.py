from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.graduation import GraduationConfig, run_graduation_lab, validate_graduation_lab


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    tokenizer_records = [
        {
            "source_id": f"tokenizer-{index}",
            "text": (
                f"Tokenizer corpus {index} explains documents tokens lineage "
                "evaluation and model training evidence."
            ),
            "metadata": {"domain": "tokenizer"},
        }
        for index in range(8)
    ]
    train_records = [
        {
            "source_id": f"train-{index}",
            "text": (
                f"Training document {index} explains reliable data quality lineage "
                "batches and model evaluation behavior."
            ),
            "metadata": {"domain": "data", "license": "project-original", "safety": "safe"},
        }
        for index in range(10)
    ]
    evaluation_records = [
        {
            "source_id": f"evaluation-{index}",
            "text": f"Evaluation document {index} measures whether governed data changes reliable model behavior.",
            "metadata": {"domain": "evaluation"},
        }
        for index in range(4)
    ]
    _write_jsonl(data_root / "lab03/tokenizer_corpus.jsonl", tokenizer_records)
    _write_jsonl(data_root / "lab04/train.jsonl", train_records)
    _write_jsonl(data_root / "lab04/evaluation.jsonl", evaluation_records)
    accepted = []
    for record in train_records[:8]:
        accepted.append(
            {
                **record,
                "metadata": {
                    **record["metadata"],
                    "quality_audit": {"decision": "accept", "quality_version": "0.1.0", "reasons": []},
                },
            }
        )
    rejected = []
    for record in train_records[8:]:
        rejected.append(
            {
                **record,
                "metadata": {
                    **record["metadata"],
                    "quality_audit": {
                        "decision": "reject",
                        "quality_version": "0.1.0",
                        "reasons": ["too_short"],
                    },
                },
            }
        )
    _write_jsonl(artifacts_root / "lab04/accepted_documents.jsonl", accepted)
    _write_jsonl(artifacts_root / "lab04/rejected_documents.jsonl", rejected)
    _write_jsonl(
        artifacts_root / "lab04/contamination_pairs.jsonl",
        [{"left_source_id": "train-0", "right_source_id": "evaluation-0", "similarity": 0.5}],
    )
    reports = {
        "lab04/quality_report.json": {"quality_fingerprint": "quality-test"},
        "lab05/dataset_version_manifest.json": {"dataset_version_fingerprint": "version-test"},
        "lab06/compute_report.json": {"compute_fingerprint": "compute-test"},
        "lab07/dataloader_report.json": {"dataloader_fingerprint": "dataloader-test"},
        "lab08/recovery_report.json": {"recovery_fingerprint": "recovery-test"},
        "lab09/storage_report.json": {"storage_fingerprint": "storage-test"},
    }
    for relative, value in reports.items():
        path = artifacts_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    (tmp_path / "uv.lock").write_text("test-lock", encoding="utf-8")
    return data_root, artifacts_root


def _config() -> GraduationConfig:
    return GraduationConfig(
        seeds=(3, 5),
        steps=4,
        batch_size=2,
        learning_rate=0.01,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        vocab_size=280,
        sequence_length=16,
    )


def test_graduation_runs_multi_seed_policy_ab_and_failure_lineage(tmp_path: Path) -> None:
    data_root, artifacts_root = _inputs(tmp_path)
    output_dir = artifacts_root / "lab10"
    report = run_graduation_lab(data_root, artifacts_root, output_dir, _config())
    assert report["aggregate"]["seed_count"] == 2
    assert report["policy_metrics"]["excluded_document_count"] == 3
    assert report["failure_examples"]
    assert validate_graduation_lab(data_root, artifacts_root, output_dir)["machine_reproduction_ready"] is True


def test_graduation_validator_detects_tampered_report_artifact(tmp_path: Path) -> None:
    data_root, artifacts_root = _inputs(tmp_path)
    output_dir = artifacts_root / "lab10"
    run_graduation_lab(data_root, artifacts_root, output_dir, _config())
    (output_dir / "graduation_report.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="产物哈希"):
        validate_graduation_lab(data_root, artifacts_root, output_dir)
