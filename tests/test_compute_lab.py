from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.compute_lab import ComputeConfig, run_compute_lab, validate_compute_lab


def _input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "lab05"
    input_dir.mkdir()
    documents = [
        {
            "source_id": f"source-{index}",
            "text": f"document {index} contains enough deterministic content for compute testing",
            "metadata": {"domain": "data" if index % 2 == 0 else "llm"},
        }
        for index in range(4)
    ]
    (input_dir / "mixed_documents.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in documents), encoding="utf-8"
    )
    (input_dir / "dataset_version_manifest.json").write_text(
        json.dumps({"dataset_version_fingerprint": "test-version"}), encoding="utf-8"
    )
    return input_dir


def test_compute_lab_proves_equivalence_skew_fix_and_recovery(tmp_path: Path) -> None:
    input_dir = _input(tmp_path)
    output_dir = tmp_path / "output"
    report = run_compute_lab(
        input_dir,
        output_dir,
        ComputeConfig(repeat_factor=16, partition_count=4, hot_key_fraction=0.75, failure_partition=1),
    )

    assert report["metrics"]["aggregation_equal"] is True
    assert report["metrics"]["skew_ratio_improvement"] > 0
    assert report["metrics"]["recovery"]["retry_count"] == 1
    assert validate_compute_lab(input_dir, output_dir)["expanded_row_count"] == 64


def test_compute_lab_detects_tampered_partition(tmp_path: Path) -> None:
    input_dir = _input(tmp_path)
    output_dir = tmp_path / "output"
    report = run_compute_lab(
        input_dir,
        output_dir,
        ComputeConfig(repeat_factor=8, partition_count=4, hot_key_fraction=0.75, failure_partition=1),
    )
    first_partition = output_dir / report["artifacts"]["partition_files"][0]["path"]
    first_partition.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Partition"):
        validate_compute_lab(input_dir, output_dir)
