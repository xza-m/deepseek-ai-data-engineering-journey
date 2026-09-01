from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.pipeline import sha256_file
from ai_data_engineering.storage_lab import StorageConfig, run_storage_lab, validate_storage_lab


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_dir = tmp_path / "lab05"
    shard_dir = input_dir / "training_shards"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "shard-00000.jsonl"
    shard_path.write_text(
        "".join(json.dumps({"sequence_id": f"seq-{index}", "value": index}) + "\n" for index in range(4)),
        encoding="utf-8",
    )
    (input_dir / "dataset_version_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version_fingerprint": "dataset-test",
                "shards": [
                    {
                        "path": "training_shards/shard-00000.jsonl",
                        "sha256": sha256_file(shard_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    compute_dir = tmp_path / "lab06"
    partition_dir = compute_dir / "partitions-salted"
    partition_dir.mkdir(parents=True)
    (partition_dir / "part-00000.jsonl").write_text("shuffle-data\n", encoding="utf-8")
    (compute_dir / "compute_report.json").write_text(
        json.dumps({"compute_fingerprint": "compute-test"}), encoding="utf-8"
    )
    recovery_dir = tmp_path / "lab08"
    recovery_dir.mkdir()
    (recovery_dir / "resumed_final.pt").write_bytes(b"checkpoint-data")
    (recovery_dir / "recovery_report.json").write_text(
        json.dumps(
            {
                "recovery_fingerprint": "recovery-test",
                "artifacts": {"resumed_final": "resumed_final.pt"},
                "metrics": {"resumed_model_state_sha256": "model-test"},
            }
        ),
        encoding="utf-8",
    )
    return input_dir, compute_dir, recovery_dir


def test_storage_lab_runs_four_workloads_with_honest_3fs_boundary(tmp_path: Path) -> None:
    input_dir, compute_dir, recovery_dir = _inputs(tmp_path)
    output_dir = tmp_path / "output"
    report = run_storage_lab(
        input_dir,
        compute_dir,
        recovery_dir,
        output_dir,
        StorageConfig(sequential_repeats=2, random_read_count=4, checkpoint_writers=2),
    )
    assert len(report["workloads"]) == 4
    assert report["threefs_review"]["cluster_executed"] is False
    assert validate_storage_lab(input_dir, compute_dir, recovery_dir, output_dir)["workload_count"] == 4


def test_storage_validator_detects_tampered_output(tmp_path: Path) -> None:
    input_dir, compute_dir, recovery_dir = _inputs(tmp_path)
    output_dir = tmp_path / "output"
    report = run_storage_lab(
        input_dir,
        compute_dir,
        recovery_dir,
        output_dir,
        StorageConfig(sequential_repeats=2, random_read_count=4, checkpoint_writers=2),
    )
    target = output_dir / report["artifacts"]["shuffle_writes"][0]["path"]
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="产物哈希"):
        validate_storage_lab(input_dir, compute_dir, recovery_dir, output_dir)
