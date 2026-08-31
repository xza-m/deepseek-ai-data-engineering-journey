from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.experiment import run_data_ab, validate_data_ab
from ai_data_engineering.training import TrainingConfig


def _write_documents(path: Path, prefix: str, texts: list[str]) -> None:
    records = [
        {
            "source_id": f"{prefix}-{index}",
            "text": text,
            "metadata": {"group": prefix},
        }
        for index, text in enumerate(texts)
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _build_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tokenizer_corpus = tmp_path / "tokenizer.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    _write_documents(
        tokenizer_corpus,
        "tokenizer",
        [f"Tokenizer corpus {index} explains data lineage, model loss, batches, and manifests." for index in range(8)],
    )
    _write_documents(
        baseline,
        "baseline",
        [f"Clean training sample {index} connects documents, tokens, models, and evidence." for index in range(12)],
    )
    _write_documents(
        candidate,
        "candidate",
        [f"Repeated low information template {index}: click click click click click." for index in range(12)],
    )
    _write_documents(
        evaluation,
        "evaluation",
        [f"Held out evaluation {index} asks how reliable data changes model behavior." for index in range(6)],
    )
    return tokenizer_corpus, baseline, candidate, evaluation


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        steps=8,
        batch_size=2,
        learning_rate=1e-2,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        seed=11,
        strict_token_budget=True,
    )


def test_data_ab_controls_tokenizer_initialization_and_token_budget(tmp_path: Path) -> None:
    tokenizer_corpus, baseline, candidate, evaluation = _build_inputs(tmp_path)
    output_dir = tmp_path / "experiment"

    manifest = run_data_ab(
        tokenizer_corpus,
        baseline,
        candidate,
        evaluation,
        output_dir,
        vocab_size=280,
        sequence_length=16,
        training_config=_training_config(),
    )

    assert manifest["controlled_variable"]["name"] == "training_corpus"
    assert manifest["runs"]["baseline"]["initial_model_state_sha256"] == manifest["runs"]["candidate"][
        "initial_model_state_sha256"
    ]
    assert manifest["runs"]["baseline"]["trained_token_count"] == manifest["runs"]["candidate"][
        "trained_token_count"
    ]
    tokenizer_hashes = {dataset["tokenizer_sha256"] for dataset in manifest["datasets"].values()}
    assert len(tokenizer_hashes) == 1

    result = validate_data_ab(output_dir)
    assert result["experiment_fingerprint"] == manifest["experiment_fingerprint"]
    assert result["trained_token_budget"] == 8 * 2 * 16


def test_data_ab_validator_detects_tampered_comparison(tmp_path: Path) -> None:
    tokenizer_corpus, baseline, candidate, evaluation = _build_inputs(tmp_path)
    output_dir = tmp_path / "experiment"
    run_data_ab(
        tokenizer_corpus,
        baseline,
        candidate,
        evaluation,
        output_dir,
        vocab_size=280,
        sequence_length=16,
        training_config=_training_config(),
    )

    manifest_path = output_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["comparison"]["candidate_minus_baseline_evaluation_loss"] = 0
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="比较指标"):
        validate_data_ab(output_dir)
