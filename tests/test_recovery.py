from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.pipeline import build_dataset
from ai_data_engineering.recovery import RecoveryConfig, run_recovery_lab, validate_recovery_lab


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "lab05"
    input_dir.mkdir()
    source = input_dir / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": f"source-{index}",
                    "text": f"recovery sample {index} contains deterministic model training evidence",
                    "metadata": {"domain": "training"},
                }
            )
            + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    build_dataset(source, input_dir / "dataset", vocab_size=280, sequence_length=16)
    (input_dir / "dataset_version_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version_fingerprint": "dataset-version-test",
                "artifacts": {"dataset": "dataset"},
            }
        ),
        encoding="utf-8",
    )
    dataloader_path = tmp_path / "dataloader.json"
    dataloader_path.write_text(json.dumps({"dataloader_fingerprint": "dataloader-test"}), encoding="utf-8")
    return input_dir, dataloader_path


def test_checkpoint_resume_is_step_exact(tmp_path: Path) -> None:
    input_dir, dataloader_path = _inputs(tmp_path)
    output_dir = tmp_path / "output"
    report = run_recovery_lab(
        input_dir,
        dataloader_path,
        output_dir,
        RecoveryConfig(total_steps=8, interrupt_step=3, batch_size=2, seed=7),
    )
    assert all(report["metrics"]["equivalence"].values())
    assert report["metrics"]["reference_loss_history"] == report["metrics"]["resumed_loss_history"]
    assert validate_recovery_lab(input_dir, dataloader_path, output_dir)["exact_recovery"] is True


def test_recovery_validator_detects_tampered_checkpoint(tmp_path: Path) -> None:
    input_dir, dataloader_path = _inputs(tmp_path)
    output_dir = tmp_path / "output"
    run_recovery_lab(
        input_dir,
        dataloader_path,
        output_dir,
        RecoveryConfig(total_steps=6, interrupt_step=2, batch_size=2, seed=7),
    )
    with (output_dir / "resumed_final.pt").open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(ValueError, match="产物哈希"):
        validate_recovery_lab(input_dir, dataloader_path, output_dir)
