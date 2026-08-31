"""Lab 02：使用 Lab 01 Sequence 训练可复现的 Tiny Causal LM。"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ai_data_engineering.contracts import RUN_SCHEMA_VERSION, TRAINING_VERSION
from ai_data_engineering.pipeline import sha256_file, validate_dataset


@dataclass(frozen=True)
class TrainingConfig:
    """Tiny LM 的最小模型与训练配置。"""

    steps: int = 100
    batch_size: int = 4
    learning_rate: float = 3e-3
    embedding_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    feed_forward_dim: int = 128
    validation_fraction: float = 0.2
    seed: int = 42

    def validate(self) -> None:
        """在唯一可信边界检查训练配置。"""

        if self.steps < 1:
            raise ValueError("steps 至少为 1")
        if self.batch_size < 1:
            raise ValueError("batch_size 至少为 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if self.embedding_dim < 4:
            raise ValueError("embedding_dim 至少为 4")
        if self.num_layers < 1:
            raise ValueError("num_layers 至少为 1")
        if self.num_heads < 1 or self.embedding_dim % self.num_heads != 0:
            raise ValueError("num_heads 必须大于 0 且能整除 embedding_dim")
        if self.feed_forward_dim < self.embedding_dim:
            raise ValueError("feed_forward_dim 不能小于 embedding_dim")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction 必须位于 0 和 1 之间")


class SequenceDataset(Dataset):
    """将 Lab 01 JSONL Sequence 暴露为 PyTorch Dataset。"""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record = self._records[index]
        return (
            torch.tensor(record["input_ids"], dtype=torch.long),
            torch.tensor(record["labels"], dtype=torch.long),
            torch.tensor(record["loss_mask"], dtype=torch.float32),
        )


class TinyCausalLM(nn.Module):
    """教学用途的最小 Decoder-only 因果语言模型。"""

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        feed_forward_dim: int,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim),
            enable_nested_tensor=False,
        )
        self.language_model_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        _, current_length = input_ids.shape
        if current_length > self.sequence_length:
            raise ValueError("输入长度超过模型配置的 sequence_length")
        positions = torch.arange(current_length, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones((current_length, current_length), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        return self.language_model_head(hidden)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_sha256(state_dict: dict[str, Tensor]) -> str:
    """对模型 Tensor 内容计算与 Checkpoint 容器无关的哈希。"""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(bytes(tensor.view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def _load_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_dataset(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequences_path = dataset_dir / manifest["artifacts"]["sequences"]
    records = [json.loads(line) for line in sequences_path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) < 2:
        raise ValueError("Tiny LM 至少需要 2 条 Sequence，才能建立训练/验证拆分")
    sequence_ids = [record["sequence_id"] for record in records]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("Sequence ID 必须唯一")
    return manifest, records


def _split_records(
    records: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(records), generator=generator).tolist()
    validation_count = max(1, min(len(records) - 1, round(len(records) * validation_fraction)))
    validation_indices = set(indices[:validation_count])
    train_records = [record for index, record in enumerate(records) if index not in validation_indices]
    validation_records = [record for index, record in enumerate(records) if index in validation_indices]
    return train_records, validation_records


def _masked_cross_entropy(logits: Tensor, labels: Tensor, loss_mask: Tensor) -> Tensor:
    token_losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
    flat_mask = loss_mask.flatten()
    trainable_tokens = flat_mask.sum()
    if trainable_tokens.item() <= 0:
        raise ValueError("Batch 中没有可训练 Token")
    return (token_losses * flat_mask).sum() / trainable_tokens


def _evaluate(model: TinyCausalLM, records: list[dict[str, Any]], batch_size: int) -> float:
    loader = DataLoader(SequenceDataset(records), batch_size=batch_size, shuffle=False)
    weighted_loss = 0.0
    token_count = 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels, loss_mask in loader:
            loss = _masked_cross_entropy(model(input_ids), labels, loss_mask)
            batch_tokens = int(loss_mask.sum().item())
            weighted_loss += float(loss.item()) * batch_tokens
            token_count += batch_tokens
    if token_count == 0:
        raise ValueError("评估集合中没有可训练 Token")
    return weighted_loss / token_count


def _build_model(dataset_manifest: dict[str, Any], config: TrainingConfig) -> TinyCausalLM:
    dataset_config = dataset_manifest["config"]
    return TinyCausalLM(
        vocab_size=dataset_config["vocab_size_actual"],
        sequence_length=dataset_config["sequence_length"],
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        feed_forward_dim=config.feed_forward_dim,
    )


def train_tiny_lm(dataset_dir: Path, output_dir: Path, config: TrainingConfig) -> dict[str, Any]:
    """训练 Tiny LM 并写出 Checkpoint 与可追溯 Run Manifest。"""

    config.validate()
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"找不到 Dataset 目录: {dataset_dir}")
    dataset_manifest, records = _load_dataset(dataset_dir)
    train_records, validation_records = _split_records(records, config.validation_fraction, config.seed)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    model = _build_model(dataset_manifest, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    initial_train_loss = _evaluate(model, train_records, config.batch_size)

    loader_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        SequenceDataset(train_records),
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    iterator = iter(train_loader)
    loss_history: list[float] = []
    trained_token_count = 0
    started_at = time.perf_counter()
    model.train()
    for _ in range(config.steps):
        try:
            input_ids, labels, loss_mask = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            input_ids, labels, loss_mask = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = _masked_cross_entropy(model(input_ids), labels, loss_mask)
        if not torch.isfinite(loss):
            raise ValueError("训练 Loss 出现非有限值")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(float(loss.item()))
        trained_token_count += int(loss_mask.sum().item())
    duration_seconds = time.perf_counter() - started_at

    final_train_loss = _evaluate(model, train_records, config.batch_size)
    validation_loss = _evaluate(model, validation_records, config.batch_size)
    split = {
        "train_sequence_ids": [record["sequence_id"] for record in train_records],
        "validation_sequence_ids": [record["sequence_id"] for record in validation_records],
        "note": "Lab 02 按 Sequence 做确定性拆分，只用于本地训练闭环，不代表无文档泄漏的正式评测",
    }
    semantic_run = {
        "training_version": TRAINING_VERSION,
        "dataset_fingerprint": dataset_manifest["pipeline_fingerprint"],
        "config": asdict(config),
        "split": split,
    }
    run_fingerprint = _sha256_payload(semantic_run)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    model_state = model.state_dict()
    model_state_sha256 = _state_sha256(model_state)
    torch.save(
        {
            "training_version": TRAINING_VERSION,
            "run_fingerprint": run_fingerprint,
            "dataset_fingerprint": dataset_manifest["pipeline_fingerprint"],
            "step": config.steps,
            "training_config": asdict(config),
            "split": split,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    sequences_name = dataset_manifest["artifacts"]["sequences"]
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "training_version": TRAINING_VERSION,
        "run_fingerprint": run_fingerprint,
        "dataset": {
            "pipeline_fingerprint": dataset_manifest["pipeline_fingerprint"],
            "sequences_sha256": dataset_manifest["artifact_sha256"]["sequences"],
            "vocab_size": dataset_manifest["config"]["vocab_size_actual"],
            "sequence_length": dataset_manifest["config"]["sequence_length"],
            "sequence_count": dataset_manifest["metrics"]["sequence_count"],
            "artifact": sequences_name,
        },
        "config": asdict(config),
        "split": split,
        "metrics": {
            "initial_train_loss": round(initial_train_loss, 8),
            "final_train_loss": round(final_train_loss, 8),
            "validation_loss": round(validation_loss, 8),
            "loss_history": [round(value, 8) for value in loss_history],
            "trained_token_count": trained_token_count,
            "duration_seconds": round(duration_seconds, 6),
            "tokens_per_second": round(trained_token_count / duration_seconds, 3),
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "environment": {"device": "cpu", "torch_version": torch.__version__},
        "artifacts": {"checkpoint": checkpoint_path.name},
        "artifact_sha256": {"checkpoint": sha256_file(checkpoint_path)},
        "model_state_sha256": model_state_sha256,
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    return run_manifest


def validate_training_run(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """验证 Run Manifest、Dataset 绑定、Checkpoint 和重新加载后的 Loss。"""

    run_manifest_path = output_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"找不到 Run Manifest: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("不支持的 Run Manifest schema_version")
    if run_manifest.get("training_version") != TRAINING_VERSION:
        raise ValueError("不支持的 training_version")

    dataset_manifest, records = _load_dataset(dataset_dir)
    dataset_section = run_manifest["dataset"]
    if dataset_section["pipeline_fingerprint"] != dataset_manifest["pipeline_fingerprint"]:
        raise ValueError("Run Manifest 未绑定到当前 Dataset Fingerprint")
    if dataset_section["sequences_sha256"] != dataset_manifest["artifact_sha256"]["sequences"]:
        raise ValueError("Run Manifest 中的 Sequence 哈希与 Dataset 不一致")

    config = TrainingConfig(**run_manifest["config"])
    config.validate()
    split = run_manifest["split"]
    expected_fingerprint = _sha256_payload(
        {
            "training_version": TRAINING_VERSION,
            "dataset_fingerprint": dataset_manifest["pipeline_fingerprint"],
            "config": asdict(config),
            "split": split,
        }
    )
    if expected_fingerprint != run_manifest["run_fingerprint"]:
        raise ValueError("Run Fingerprint 与数据、配置或拆分不一致")

    checkpoint_path = output_dir / run_manifest["artifacts"]["checkpoint"]
    if sha256_file(checkpoint_path) != run_manifest["artifact_sha256"]["checkpoint"]:
        raise ValueError("Checkpoint 文件哈希与 Run Manifest 不一致")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["run_fingerprint"] != run_manifest["run_fingerprint"]:
        raise ValueError("Checkpoint 未绑定到当前 Run Fingerprint")
    if checkpoint["dataset_fingerprint"] != dataset_manifest["pipeline_fingerprint"]:
        raise ValueError("Checkpoint 未绑定到当前 Dataset Fingerprint")
    if checkpoint["step"] != config.steps or checkpoint["training_config"] != asdict(config):
        raise ValueError("Checkpoint 的训练配置与 Run Manifest 不一致")
    if checkpoint["split"] != split:
        raise ValueError("Checkpoint 的数据拆分与 Run Manifest 不一致")

    model = _build_model(dataset_manifest, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    if _state_sha256(model.state_dict()) != run_manifest["model_state_sha256"]:
        raise ValueError("Checkpoint 模型状态哈希与 Run Manifest 不一致")

    records_by_id = {record["sequence_id"]: record for record in records}
    train_ids = split["train_sequence_ids"]
    validation_ids = split["validation_sequence_ids"]
    if set(train_ids) & set(validation_ids):
        raise ValueError("训练集与验证集 Sequence 重叠")
    if set(train_ids) | set(validation_ids) != set(records_by_id):
        raise ValueError("Run Manifest 的数据拆分未完整覆盖 Dataset")
    validation_records = [records_by_id[sequence_id] for sequence_id in validation_ids]
    reloaded_validation_loss = _evaluate(model, validation_records, config.batch_size)
    expected_validation_loss = run_manifest["metrics"]["validation_loss"]
    if not math.isclose(reloaded_validation_loss, expected_validation_loss, rel_tol=0, abs_tol=1e-6):
        raise ValueError("重新加载 Checkpoint 后的 Validation Loss 不一致")

    return {
        "run_fingerprint": run_manifest["run_fingerprint"],
        "dataset_fingerprint": dataset_manifest["pipeline_fingerprint"],
        "step": config.steps,
        "initial_train_loss": run_manifest["metrics"]["initial_train_loss"],
        "final_train_loss": run_manifest["metrics"]["final_train_loss"],
        "validation_loss": expected_validation_loss,
        "model_state_sha256": run_manifest["model_state_sha256"],
    }
