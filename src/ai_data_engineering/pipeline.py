"""Lab 01：从 JSONL 文档构建可追溯的训练 Sequence。"""

from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from ai_data_engineering.contracts import (
    BOS_TOKEN,
    EOS_TOKEN,
    MIN_BYTE_BPE_VOCAB_SIZE,
    PAD_TOKEN,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    SPECIAL_TOKENS,
    UNK_TOKEN,
)


@dataclass(frozen=True)
class NormalizedDocument:
    """经过规范化和精确去重的文档。"""

    doc_id: str
    source_id: str
    content_sha256: str
    text: str
    metadata: dict[str, Any]
    token_count: int = 0


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """计算文件内容哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """执行最小且可解释的 Unicode 与空白规范化。"""

    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def load_normalized_documents(path: Path) -> tuple[list[NormalizedDocument], int, int]:
    """读取 JSONL，按规范化内容哈希保留第一条文档。"""

    documents: list[NormalizedDocument] = []
    seen_hashes: set[str] = set()
    seen_source_ids: set[str] = set()
    raw_document_count = 0
    duplicate_document_count = 0

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw_document_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第 {line_number} 行不是有效 JSON: {error.msg}") from error

            source_id = record.get("source_id")
            text = record.get("text")
            metadata = record.get("metadata", {})
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"第 {line_number} 行缺少非空 source_id")
            if source_id in seen_source_ids:
                raise ValueError(f"source_id 重复: {source_id}")
            seen_source_ids.add(source_id)
            if not isinstance(text, str):
                raise ValueError(f"第 {line_number} 行的 text 必须是字符串")
            if not isinstance(metadata, dict):
                raise ValueError(f"第 {line_number} 行的 metadata 必须是对象")

            normalized_text = normalize_text(text)
            if not normalized_text:
                raise ValueError(f"第 {line_number} 行规范化后为空文本")
            content_sha256 = _sha256_bytes(normalized_text.encode("utf-8"))
            if content_sha256 in seen_hashes:
                duplicate_document_count += 1
                continue
            seen_hashes.add(content_sha256)

            documents.append(
                NormalizedDocument(
                    doc_id=f"doc_{content_sha256[:16]}",
                    source_id=source_id,
                    content_sha256=content_sha256,
                    text=normalized_text,
                    metadata=metadata,
                )
            )

    if not documents:
        raise ValueError("输入中没有可用文档")
    return documents, raw_document_count, duplicate_document_count


def train_byte_bpe(documents: Iterable[NormalizedDocument], vocab_size: int) -> Tokenizer:
    """在当前数据上训练教学用途的 Byte-level BPE Tokenizer。"""

    if vocab_size < MIN_BYTE_BPE_VOCAB_SIZE:
        raise ValueError(f"vocab_size 至少为 {MIN_BYTE_BPE_VOCAB_SIZE}，才能覆盖字节表和特殊 Token")

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        show_progress=False,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator((document.text for document in documents), trainer=trainer)
    return tokenizer


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _pipeline_fingerprint(source_sha256: str, tokenizer_sha256: str, config: dict[str, Any]) -> str:
    payload = json.dumps(
        {"source_sha256": source_sha256, "tokenizer_sha256": tokenizer_sha256, "config": config},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _document_provenance(
    document_spans: list[dict[str, Any]], sequence_start: int, sequence_end: int
) -> list[dict[str, Any]]:
    provenance = []
    for span in document_spans:
        overlap_start = max(sequence_start, span["global_start"])
        overlap_end = min(sequence_end, span["global_end"])
        if overlap_start >= overlap_end:
            continue
        provenance.append(
            {
                "doc_id": span["doc_id"],
                "sequence_token_start": overlap_start - sequence_start,
                "sequence_token_end": overlap_end - sequence_start,
                "document_token_start": overlap_start - span["global_start"],
                "document_token_end": overlap_end - span["global_start"],
            }
        )
    return provenance


def build_dataset(
    input_path: Path,
    output_dir: Path,
    vocab_size: int,
    sequence_length: int,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    """构建规范化文档、Tokenizer、训练 Sequence 和确定性 Manifest。"""

    if sequence_length < 2:
        raise ValueError("sequence_length 至少为 2")
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    documents, raw_count, duplicate_count = load_normalized_documents(input_path)
    output_tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_path is None:
        tokenizer = train_byte_bpe(documents, vocab_size)
        tokenizer.save(str(output_tokenizer_path), pretty=True)
        tokenizer_mode = "trained"
        vocab_size_requested: int | None = vocab_size
    else:
        tokenizer_source_path = tokenizer_path.resolve()
        if not tokenizer_source_path.is_file():
            raise FileNotFoundError(f"找不到复用 Tokenizer: {tokenizer_source_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_source_path))
        if tokenizer_source_path != output_tokenizer_path.resolve():
            shutil.copyfile(tokenizer_source_path, output_tokenizer_path)
        tokenizer_mode = "reused"
        vocab_size_requested = None

    token_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(token_id is None for token_id in token_ids.values()):
        raise RuntimeError("Tokenizer 缺少项目要求的特殊 Token")
    pad_id = int(token_ids[PAD_TOKEN])
    bos_id = int(token_ids[BOS_TOKEN])
    eos_id = int(token_ids[EOS_TOKEN])

    token_stream: list[int] = []
    document_spans: list[dict[str, Any]] = []
    tokenized_documents: list[NormalizedDocument] = []
    for document in documents:
        encoded = tokenizer.encode(document.text, add_special_tokens=False).ids
        document_tokens = [bos_id, *encoded, eos_id]
        start = len(token_stream)
        token_stream.extend(document_tokens)
        document_spans.append(
            {"doc_id": document.doc_id, "global_start": start, "global_end": len(token_stream)}
        )
        tokenized_documents.append(replace(document, token_count=len(document_tokens)))

    if len(token_stream) < 2:
        raise ValueError("Token 流不足以构建训练标签")

    sequences = []
    trainable_token_count = 0
    window_size = sequence_length + 1
    for index, start in enumerate(range(0, len(token_stream) - 1, sequence_length)):
        window = token_stream[start : start + window_size]
        padded_window = window + [pad_id] * (window_size - len(window))
        input_ids = padded_window[:-1]
        labels = padded_window[1:]
        loss_mask = [int(label != pad_id) for label in labels]
        trainable_token_count += sum(loss_mask)
        input_end = min(start + sequence_length, len(token_stream))
        sequences.append(
            {
                "sequence_id": f"seq_{index:08d}",
                "input_ids": input_ids,
                "labels": labels,
                "loss_mask": loss_mask,
                "provenance": _document_provenance(document_spans, start, input_end),
            }
        )

    normalized_path = output_dir / "normalized_documents.jsonl"
    sequences_path = output_dir / "sequences.jsonl"
    _write_jsonl(normalized_path, (asdict(document) for document in tokenized_documents))
    _write_jsonl(sequences_path, sequences)

    source_sha256 = sha256_file(input_path)
    tokenizer_sha256 = sha256_file(output_tokenizer_path)
    artifact_sha256 = {
        "normalized_documents": sha256_file(normalized_path),
        "sequences": sha256_file(sequences_path),
        "tokenizer": tokenizer_sha256,
    }
    config = {
        "normalization": "NFKC+line-whitespace-v1",
        "tokenizer_mode": tokenizer_mode,
        "vocab_size_requested": vocab_size_requested,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "sequence_length": sequence_length,
        "special_token_ids": {key: int(value) for key, value in token_ids.items()},
    }
    capacity = len(sequences) * sequence_length
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source_sha256": source_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "pipeline_fingerprint": _pipeline_fingerprint(source_sha256, tokenizer_sha256, config),
        "config": config,
        "metrics": {
            "raw_document_count": raw_count,
            "document_count": len(tokenized_documents),
            "duplicate_document_count": duplicate_count,
            "total_token_count": len(token_stream),
            "trainable_token_count": trainable_token_count,
            "sequence_count": len(sequences),
            "packing_efficiency": round(trainable_token_count / capacity, 6),
        },
        "artifacts": {
            "normalized_documents": normalized_path.name,
            "sequences": sequences_path.name,
            "tokenizer": output_tokenizer_path.name,
        },
        "artifact_sha256": artifact_sha256,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_dataset(output_dir: Path) -> dict[str, Any]:
    """验证 Lab 01 产物契约、数组长度、右移标签和内容哈希。"""

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 Manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的 Manifest schema_version")

    config = manifest["config"]
    metrics = manifest["metrics"]
    artifacts = manifest["artifacts"]
    artifact_sha256 = manifest["artifact_sha256"]
    sequence_length = config["sequence_length"]
    pad_id = config["special_token_ids"][PAD_TOKEN]
    tokenizer_path = output_dir / artifacts["tokenizer"]
    sequences_path = output_dir / artifacts["sequences"]
    normalized_path = output_dir / artifacts["normalized_documents"]

    if sha256_file(tokenizer_path) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer 文件哈希与 Manifest 不一致")

    sequence_count = 0
    trainable_token_count = 0
    with sequences_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            input_ids = record["input_ids"]
            labels = record["labels"]
            loss_mask = record["loss_mask"]
            if not (len(input_ids) == len(labels) == len(loss_mask) == sequence_length):
                raise ValueError(f"第 {line_number} 条 Sequence 长度不符合契约")
            if any(mask not in (0, 1) for mask in loss_mask):
                raise ValueError(f"第 {line_number} 条 Sequence 的 loss_mask 非法")
            if any(mask != int(label != pad_id) for label, mask in zip(labels, loss_mask, strict=True)):
                raise ValueError(f"第 {line_number} 条 Sequence 的 Padding Mask 非法")
            if any(labels[index] != input_ids[index + 1] for index in range(sequence_length - 1)):
                raise ValueError(f"第 {line_number} 条 Sequence 的 labels 未正确右移")
            sequence_count += 1
            trainable_token_count += sum(loss_mask)

    normalized_count = sum(1 for line in normalized_path.read_text(encoding="utf-8").splitlines() if line)
    if sequence_count != metrics["sequence_count"]:
        raise ValueError("Sequence 数量与 Manifest 不一致")
    if normalized_count != metrics["document_count"]:
        raise ValueError("文档数量与 Manifest 不一致")
    if trainable_token_count != metrics["trainable_token_count"]:
        raise ValueError("可训练 Token 数与 Manifest 不一致")
    for name, path in {
        "normalized_documents": normalized_path,
        "sequences": sequences_path,
        "tokenizer": tokenizer_path,
    }.items():
        if sha256_file(path) != artifact_sha256[name]:
            raise ValueError(f"{name} 文件哈希与 Manifest 不一致")
    expected_fingerprint = _pipeline_fingerprint(
        manifest["source_sha256"], manifest["tokenizer_sha256"], config
    )
    if expected_fingerprint != manifest["pipeline_fingerprint"]:
        raise ValueError("Pipeline Fingerprint 与 Manifest 不一致")

    return {
        "document_count": normalized_count,
        "sequence_count": sequence_count,
        "trainable_token_count": trainable_token_count,
        "pipeline_fingerprint": manifest["pipeline_fingerprint"],
    }
