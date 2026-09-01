from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.quality import QualityConfig, run_quality_audit, validate_quality_audit


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    truth = tmp_path / "truth.jsonl"
    common = {"license": "project-original", "safety": "safe", "domain": "test"}
    _write(
        train,
        [
            {
                "source_id": "a",
                "text": "data pipelines create reliable training samples with lineage",
                "metadata": common,
            },
            {
                "source_id": "b",
                "text": "data pipelines create traceable training samples with lineage",
                "metadata": common,
            },
            {"source_id": "c", "text": "storage systems serve random model reads and checkpoints", "metadata": common},
            {
                "source_id": "d",
                "text": "private contact test@example.com should be rejected safely",
                "metadata": common,
            },
        ],
    )
    _write(
        evaluation,
        [
            {
                "source_id": "eval-a",
                "text": "storage systems serve random model reads and checkpoints",
                "metadata": common,
            }
        ],
    )
    _write(
        truth,
        [
            {"left_source_id": "a", "right_source_id": "b", "is_duplicate": True},
            {"left_source_id": "a", "right_source_id": "c", "is_duplicate": False},
        ],
    )
    return train, evaluation, truth


def test_quality_audit_measures_truth_filters_and_contamination(tmp_path: Path) -> None:
    train, evaluation, truth = _inputs(tmp_path)
    output = tmp_path / "quality"
    report = run_quality_audit(
        train,
        evaluation,
        truth,
        output,
        QualityConfig(similarity_threshold=0.3, contamination_threshold=0.9, min_characters=20),
    )

    assert report["metrics"]["benchmark"]["precision"] == 1.0
    assert report["metrics"]["benchmark"]["recall"] == 1.0
    assert report["metrics"]["contamination_pair_count"] == 1
    assert report["metrics"]["reason_counts"]["near_duplicate"] == 1
    assert report["metrics"]["reason_counts"]["pii_detected"] == 1
    assert (
        validate_quality_audit(train, evaluation, truth, output)["quality_fingerprint"] == report["quality_fingerprint"]
    )


def test_quality_validator_detects_tampered_artifact(tmp_path: Path) -> None:
    train, evaluation, truth = _inputs(tmp_path)
    output = tmp_path / "quality"
    run_quality_audit(
        train,
        evaluation,
        truth,
        output,
        QualityConfig(similarity_threshold=0.3, contamination_threshold=0.9, min_characters=20),
    )
    (output / "accepted_documents.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="产物哈希"):
        validate_quality_audit(train, evaluation, truth, output)
