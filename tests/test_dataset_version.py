from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_data_engineering.dataset_version import VersionConfig, build_dataset_version, validate_dataset_version


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    accepted = tmp_path / "accepted.jsonl"
    quality = tmp_path / "quality.json"
    mix = tmp_path / "mix.json"
    records = []
    for domain in ("data", "model"):
        for index in range(3):
            records.append(
                {
                    "source_id": f"{domain}-{index}",
                    "text": (
                        f"{domain} sample {index} explains tokens manifests lineage "
                        "batches and reliable training evidence."
                    ),
                    "metadata": {
                        "domain": domain,
                        "license": "project-original",
                        "quality_audit": {"decision": "accept", "quality_version": "0.1.0", "reasons": []},
                    },
                }
            )
    _write_jsonl(accepted, records)
    quality.write_text(json.dumps({"quality_fingerprint": "quality-test"}), encoding="utf-8")
    mix.write_text(json.dumps({"domain_weights": {"data": 0.5, "model": 0.5}}), encoding="utf-8")
    return accepted, quality, mix


def test_dataset_version_is_sharded_traceable_and_deterministic(tmp_path: Path) -> None:
    accepted, quality, mix = _inputs(tmp_path)
    config = VersionConfig(vocab_size=280, sequence_length=16, max_sequences_per_shard=2, seed=7)
    first = build_dataset_version(accepted, quality, mix, tmp_path / "first", config)
    second = build_dataset_version(accepted, quality, mix, tmp_path / "second", config)

    assert first["dataset_version_fingerprint"] == second["dataset_version_fingerprint"]
    assert first["metrics"]["lineage_coverage"] == 1.0
    assert max(shard["sequence_count"] for shard in first["shards"]) <= 2
    result = validate_dataset_version(accepted, quality, mix, tmp_path / "first")
    assert result["sequence_count"] == first["metrics"]["sequence_count"]


def test_dataset_version_detects_tampered_shard(tmp_path: Path) -> None:
    accepted, quality, mix = _inputs(tmp_path)
    output = tmp_path / "output"
    manifest = build_dataset_version(
        accepted,
        quality,
        mix,
        output,
        VersionConfig(vocab_size=280, sequence_length=16, max_sequences_per_shard=2),
    )
    (output / manifest["shards"][0]["path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Shard 哈希"):
        validate_dataset_version(accepted, quality, mix, output)
