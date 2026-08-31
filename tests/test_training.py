from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.pipeline import build_dataset
from ai_data_engineering.training import TrainingConfig, train_tiny_lm, validate_training_run


def _build_test_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "source.jsonl"
    records = [
        {
            "source_id": f"doc-{index}",
            "text": f"AI data engineering sample {index}. Tokens need reproducible lineage and training evidence.",
            "metadata": {"language": "en", "group": index % 2},
        }
        for index in range(8)
    ]
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "dataset"
    build_dataset(source, dataset_dir, vocab_size=280, sequence_length=16)
    return dataset_dir


def _test_config() -> TrainingConfig:
    return TrainingConfig(
        steps=16,
        batch_size=2,
        learning_rate=1e-2,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        validation_fraction=0.25,
        seed=7,
    )


def test_training_is_reproducible_and_checkpoint_can_be_reloaded(tmp_path: Path) -> None:
    dataset_dir = _build_test_dataset(tmp_path)
    first_output = tmp_path / "first-run"
    second_output = tmp_path / "second-run"

    first = train_tiny_lm(dataset_dir, first_output, _test_config())
    second = train_tiny_lm(dataset_dir, second_output, _test_config())

    assert first["run_fingerprint"] == second["run_fingerprint"]
    assert first["model_state_sha256"] == second["model_state_sha256"]
    assert first["metrics"]["loss_history"] == second["metrics"]["loss_history"]
    assert first["metrics"]["final_train_loss"] < first["metrics"]["initial_train_loss"]
    assert first["metrics"]["validation_loss"] > 0

    result = validate_training_run(dataset_dir, first_output)
    assert result["run_fingerprint"] == first["run_fingerprint"]
    assert result["model_state_sha256"] == first["model_state_sha256"]


def test_training_validator_detects_corrupted_checkpoint(tmp_path: Path) -> None:
    dataset_dir = _build_test_dataset(tmp_path)
    output_dir = tmp_path / "run"
    train_tiny_lm(dataset_dir, output_dir, _test_config())

    with (output_dir / "checkpoint.pt").open("ab") as checkpoint:
        checkpoint.write(b"corrupted")

    with pytest.raises(ValueError, match="Checkpoint 文件哈希"):
        validate_training_run(dataset_dir, output_dir)


def test_training_config_rejects_incompatible_attention_heads() -> None:
    config = TrainingConfig(embedding_dim=10, num_heads=4)

    with pytest.raises(ValueError, match="num_heads"):
        config.validate()
