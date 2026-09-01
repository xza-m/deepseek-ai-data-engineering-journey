from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.dataloader_lab import DataLoaderConfig, run_dataloader_lab, validate_dataloader_lab
from ai_data_engineering.pipeline import sha256_file


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "lab05"
    shard_dir = input_dir / "training_shards"
    shard_dir.mkdir(parents=True)
    shards = []
    for shard_index in range(2):
        path = shard_dir / f"shard-{shard_index:05d}.jsonl"
        records = [
            {
                "sequence_id": f"seq-{shard_index}-{index}",
                "input_ids": [1, index + 4, 2, 0],
                "loss_mask": [1, 1, 1, 0],
            }
            for index in range(2)
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        shards.append(
            {
                "path": str(path.relative_to(input_dir)),
                "sha256": sha256_file(path),
                "sequence_count": len(records),
            }
        )
    (input_dir / "dataset_version_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version_fingerprint": "dataset-test",
                "shards": shards,
                "metrics": {"sequence_count": 4, "trainable_token_count": 12},
            }
        ),
        encoding="utf-8",
    )
    compute_path = tmp_path / "compute.json"
    compute_path.write_text(json.dumps({"compute_fingerprint": "compute-test"}), encoding="utf-8")
    return input_dir, compute_path


def test_dataloader_profiles_preserve_all_sequences_and_tokens(tmp_path: Path) -> None:
    input_dir, compute_path = _inputs(tmp_path)
    output_dir = tmp_path / "output"
    report = run_dataloader_lab(
        input_dir,
        compute_path,
        output_dir,
        DataLoaderConfig(batch_size=2, repeat_epochs=3, worker_counts=(0,)),
    )
    assert {profile["style"] for profile in report["profiles"]} == {"map", "iterable"}
    assert all(profile["sample_count"] == 12 for profile in report["profiles"])
    assert all(profile["trainable_token_count"] == 36 for profile in report["profiles"])
    assert validate_dataloader_lab(input_dir, compute_path, output_dir)["all_profiles_complete"] is True


def test_dataloader_validation_detects_tampered_shard(tmp_path: Path) -> None:
    input_dir, compute_path = _inputs(tmp_path)
    output_dir = tmp_path / "output"
    run_dataloader_lab(
        input_dir,
        compute_path,
        output_dir,
        DataLoaderConfig(batch_size=2, repeat_epochs=2, worker_counts=(0,)),
    )
    (input_dir / "training_shards/shard-00000.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Shard 哈希"):
        validate_dataloader_lab(input_dir, compute_path, output_dir)
