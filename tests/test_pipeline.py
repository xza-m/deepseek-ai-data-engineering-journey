from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.contracts import MIN_BYTE_BPE_VOCAB_SIZE
from ai_data_engineering.pipeline import build_dataset, normalize_text, validate_dataset


def _write_source(path: Path) -> None:
    records = [
        {"source_id": "a", "text": "ＡＩ   data\r\nengineering", "metadata": {"language": "mixed"}},
        {"source_id": "b", "text": "AI data\nengineering", "metadata": {"language": "mixed"}},
        {"source_id": "c", "text": "Token sequences need lineage.", "metadata": {"language": "en"}},
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_normalize_text_handles_unicode_and_whitespace() -> None:
    assert normalize_text("ＡＩ   data\r\n\r\n engineering ") == "AI data\nengineering"


def test_dataset_is_deduplicated_valid_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = build_dataset(source, first_output, vocab_size=280, sequence_length=16)
    second = build_dataset(source, second_output, vocab_size=280, sequence_length=16)

    assert first == second
    assert first["metrics"]["raw_document_count"] == 3
    assert first["metrics"]["document_count"] == 2
    assert first["metrics"]["duplicate_document_count"] == 1
    assert (first_output / "manifest.json").read_bytes() == (second_output / "manifest.json").read_bytes()

    result = validate_dataset(first_output)
    assert result["document_count"] == 2
    assert result["sequence_count"] > 0
    assert result["trainable_token_count"] > 0


def test_build_dataset_rejects_too_small_vocab(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)

    with pytest.raises(ValueError, match="vocab_size"):
        build_dataset(source, tmp_path / "output", MIN_BYTE_BPE_VOCAB_SIZE - 1, 16)


def test_validator_detects_broken_label_shift(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    output = tmp_path / "output"
    build_dataset(source, output, vocab_size=280, sequence_length=16)

    sequences_path = output / "sequences.jsonl"
    records = [json.loads(line) for line in sequences_path.read_text(encoding="utf-8").splitlines()]
    records[0]["labels"][0] = -1
    sequences_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="labels 未正确右移"):
        validate_dataset(output)
