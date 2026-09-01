"""Lab 08：Checkpoint 精确恢复与训练并行的数据压力映射。"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ai_data_engineering.contracts import RECOVERY_SCHEMA_VERSION, RECOVERY_VERSION
from ai_data_engineering.pipeline import sha256_file, validate_dataset
from ai_data_engineering.training import TinyCausalLM, _masked_cross_entropy, _state_sha256


@dataclass(frozen=True)
class RecoveryConfig:
    """精确恢复实验配置。"""

    total_steps: int = 12
    interrupt_step: int = 5
    batch_size: int = 2
    learning_rate: float = 0.01
    embedding_dim: int = 16
    num_layers: int = 1
    num_heads: int = 4
    feed_forward_dim: int = 32
    seed: int = 2026

    def validate(self) -> None:
        if self.total_steps < 2 or not 0 < self.interrupt_step < self.total_steps:
            raise ValueError("interrupt_step 必须位于 0 和 total_steps 之间")
        if self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("batch_size 和 learning_rate 必须大于 0")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("num_heads 必须整除 embedding_dim")


def _payload_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_value(value: Any, digest: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(bytes(tensor.reshape(-1).view(torch.uint8).tolist()))
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=str):
            _hash_value(str(key), digest)
            _hash_value(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _hash_value(item, digest)
    else:
        digest.update(repr(value).encode("utf-8"))


def _state_value_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(value, digest)
    return digest.hexdigest()


def _load_records(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_dataset(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    path = dataset_dir / manifest["artifacts"]["sequences"]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return manifest, records


def _build_model(manifest: dict[str, Any], config: RecoveryConfig) -> TinyCausalLM:
    return TinyCausalLM(
        vocab_size=manifest["config"]["vocab_size_actual"],
        sequence_length=manifest["config"]["sequence_length"],
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        feed_forward_dim=config.feed_forward_dim,
    )


def _schedule(record_count: int, config: RecoveryConfig) -> list[list[int]]:
    generator = torch.Generator().manual_seed(config.seed)
    indices: list[int] = []
    required = config.total_steps * config.batch_size
    while len(indices) < required:
        indices.extend(torch.randperm(record_count, generator=generator).tolist())
    return [
        indices[start : start + config.batch_size]
        for start in range(0, required, config.batch_size)
    ]


def _batch(records: list[dict[str, Any]], indices: list[int]) -> tuple[Tensor, Tensor, Tensor]:
    selected = [records[index] for index in indices]
    return (
        torch.tensor([record["input_ids"] for record in selected], dtype=torch.long),
        torch.tensor([record["labels"] for record in selected], dtype=torch.long),
        torch.tensor([record["loss_mask"] for record in selected], dtype=torch.float32),
    )


def _initialize(
    manifest: dict[str, Any], config: RecoveryConfig
) -> tuple[TinyCausalLM, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = _build_model(manifest, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=config.total_steps,
    )
    return model, optimizer, scheduler


def _train_range(
    model: TinyCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    records: list[dict[str, Any]],
    schedule: list[list[int]],
    start_step: int,
    end_step: int,
    loss_history: list[float],
) -> None:
    model.train()
    for step in range(start_step, end_step):
        input_ids, labels, loss_mask = _batch(records, schedule[step])
        optimizer.zero_grad(set_to_none=True)
        loss = _masked_cross_entropy(model(input_ids), labels, loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.item()))


def _checkpoint(
    model: TinyCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    loss_history: list[float],
    dataset_version_fingerprint: str,
    schedule_sha256: str,
) -> dict[str, Any]:
    return {
        "recovery_version": RECOVERY_VERSION,
        "dataset_version_fingerprint": dataset_version_fingerprint,
        "schedule_sha256": schedule_sha256,
        "step": step,
        "data_cursor": step,
        "loss_history": loss_history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }


def _parallelism_contract() -> dict[str, Any]:
    return {
        "data_parallel": {
            "partitioned": "Batch/Sequence",
            "replicated": "模型参数",
            "data_risk": "Rank 间样本重复或 Shard 不均",
            "required_lineage": "global_step + rank + sequence_id",
        },
        "tensor_parallel": {
            "partitioned": "层内 Tensor",
            "replicated": "同一 Micro Batch 的 Token",
            "data_risk": "Sequence Shape 不一致导致 Collective 阻塞",
            "required_lineage": "global_step + micro_batch + sequence_shape",
        },
        "pipeline_parallel": {
            "partitioned": "模型层",
            "replicated": "跨 Stage 流动的 Micro Batch 语义",
            "data_risk": "Micro Batch 数不足产生 Pipeline Bubble",
            "required_lineage": "global_step + micro_batch + stage",
        },
        "expert_parallel": {
            "partitioned": "MoE Expert",
            "replicated": "Router 输入",
            "data_risk": "Token 路由倾斜造成 Expert 负载不均",
            "required_lineage": "token + route + expert + capacity_drop",
        },
        "zero": {
            "partitioned": "优化器状态/梯度/参数",
            "replicated": "训练数据全局顺序契约",
            "data_risk": "恢复时 World Size 与状态分片不兼容",
            "required_lineage": "checkpoint + world_size + shard_owner",
        },
    }


def run_recovery_lab(
    input_dir: Path,
    dataloader_report_path: Path,
    output_dir: Path,
    config: RecoveryConfig | None = None,
) -> dict[str, Any]:
    """比较不中断训练与中断后恢复训练的逐步等价性。"""

    config = config or RecoveryConfig()
    config.validate()
    version = json.loads((input_dir / "dataset_version_manifest.json").read_text(encoding="utf-8"))
    dataloader = json.loads(dataloader_report_path.read_text(encoding="utf-8"))
    dataset_dir = input_dir / version["artifacts"]["dataset"]
    dataset_manifest, records = _load_records(dataset_dir)
    schedule = _schedule(len(records), config)
    schedule_sha256 = _payload_sha256(schedule)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_model, reference_optimizer, reference_scheduler = _initialize(dataset_manifest, config)
    reference_losses: list[float] = []
    _train_range(
        reference_model,
        reference_optimizer,
        reference_scheduler,
        records,
        schedule,
        0,
        config.total_steps,
        reference_losses,
    )
    reference_checkpoint = _checkpoint(
        reference_model,
        reference_optimizer,
        reference_scheduler,
        config.total_steps,
        reference_losses,
        version["dataset_version_fingerprint"],
        schedule_sha256,
    )
    reference_path = output_dir / "reference_final.pt"
    torch.save(reference_checkpoint, reference_path)

    interrupted_model, interrupted_optimizer, interrupted_scheduler = _initialize(dataset_manifest, config)
    resumed_losses: list[float] = []
    _train_range(
        interrupted_model,
        interrupted_optimizer,
        interrupted_scheduler,
        records,
        schedule,
        0,
        config.interrupt_step,
        resumed_losses,
    )
    interruption_checkpoint = _checkpoint(
        interrupted_model,
        interrupted_optimizer,
        interrupted_scheduler,
        config.interrupt_step,
        resumed_losses,
        version["dataset_version_fingerprint"],
        schedule_sha256,
    )
    interruption_path = output_dir / "interruption_checkpoint.pt"
    torch.save(interruption_checkpoint, interruption_path)

    restored = torch.load(interruption_path, map_location="cpu", weights_only=False)
    resumed_model, resumed_optimizer, resumed_scheduler = _initialize(dataset_manifest, config)
    resumed_model.load_state_dict(restored["model_state_dict"])
    resumed_optimizer.load_state_dict(restored["optimizer_state_dict"])
    resumed_scheduler.load_state_dict(restored["scheduler_state_dict"])
    random.setstate(restored["python_random_state"])
    torch.set_rng_state(restored["torch_random_state"])
    resumed_losses = list(restored["loss_history"])
    _train_range(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        records,
        schedule,
        restored["data_cursor"],
        config.total_steps,
        resumed_losses,
    )
    resumed_checkpoint = _checkpoint(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        config.total_steps,
        resumed_losses,
        version["dataset_version_fingerprint"],
        schedule_sha256,
    )
    resumed_path = output_dir / "resumed_final.pt"
    torch.save(resumed_checkpoint, resumed_path)

    reference_model_hash = _state_sha256(reference_model.state_dict())
    resumed_model_hash = _state_sha256(resumed_model.state_dict())
    reference_optimizer_hash = _state_value_sha256(reference_optimizer.state_dict())
    resumed_optimizer_hash = _state_value_sha256(resumed_optimizer.state_dict())
    reference_scheduler_hash = _state_value_sha256(reference_scheduler.state_dict())
    resumed_scheduler_hash = _state_value_sha256(resumed_scheduler.state_dict())
    equivalence = {
        "loss_history_exact": reference_losses == resumed_losses,
        "model_state_exact": reference_model_hash == resumed_model_hash,
        "optimizer_state_exact": reference_optimizer_hash == resumed_optimizer_hash,
        "scheduler_state_exact": reference_scheduler_hash == resumed_scheduler_hash,
        "final_learning_rate_exact": reference_scheduler.get_last_lr() == resumed_scheduler.get_last_lr(),
    }
    if not all(equivalence.values()):
        raise RuntimeError(f"Checkpoint 恢复未达到逐步精确等价: {equivalence}")

    parallelism_path = output_dir / "parallelism_data_contract.json"
    parallelism_path.write_text(
        json.dumps(_parallelism_contract(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    semantic = {
        "recovery_version": RECOVERY_VERSION,
        "inputs": {
            "dataset_version_fingerprint": version["dataset_version_fingerprint"],
            "dataloader_fingerprint": dataloader["dataloader_fingerprint"],
            "dataset_pipeline_fingerprint": dataset_manifest["pipeline_fingerprint"],
        },
        "config": asdict(config),
        "schedule": {"sha256": schedule_sha256, "batch_count": len(schedule)},
        "checkpoint_contract": [
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "python_random_state",
            "torch_random_state",
            "data_cursor",
            "loss_history",
            "dataset_version_fingerprint",
            "schedule_sha256",
        ],
        "metrics": {
            "reference_loss_history": reference_losses,
            "resumed_loss_history": resumed_losses,
            "reference_model_state_sha256": reference_model_hash,
            "resumed_model_state_sha256": resumed_model_hash,
            "reference_optimizer_state_sha256": reference_optimizer_hash,
            "resumed_optimizer_state_sha256": resumed_optimizer_hash,
            "reference_scheduler_state_sha256": reference_scheduler_hash,
            "resumed_scheduler_state_sha256": resumed_scheduler_hash,
            "equivalence": equivalence,
            "recovered_from_step": config.interrupt_step,
            "final_step": config.total_steps,
        },
        "artifacts": {
            "interruption_checkpoint": interruption_path.name,
            "reference_final": reference_path.name,
            "resumed_final": resumed_path.name,
            "parallelism_data_contract": parallelism_path.name,
        },
    }
    report = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        **semantic,
        "artifact_sha256": {
            "interruption_checkpoint": sha256_file(interruption_path),
            "reference_final": sha256_file(reference_path),
            "resumed_final": sha256_file(resumed_path),
            "parallelism_data_contract": sha256_file(parallelism_path),
        },
        "evidence_boundary": [
            "精确等价只在当前 CPU、PyTorch 版本、确定性算子与固定 Batch Schedule 下成立",
            "并行策略文件是数据压力契约，不是多 GPU 性能实测",
            "生产 Checkpoint 还需验证分布式拓扑、原子发布、保留策略和远端耐久性",
        ],
    }
    report["recovery_fingerprint"] = _payload_sha256(semantic)
    (output_dir / "recovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def validate_recovery_lab(input_dir: Path, dataloader_report_path: Path, output_dir: Path) -> dict[str, Any]:
    """验证 Checkpoint 文件、状态哈希、上游绑定和精确恢复证据。"""

    report = json.loads((output_dir / "recovery_report.json").read_text(encoding="utf-8"))
    version = json.loads((input_dir / "dataset_version_manifest.json").read_text(encoding="utf-8"))
    dataloader = json.loads(dataloader_report_path.read_text(encoding="utf-8"))
    if report["inputs"]["dataset_version_fingerprint"] != version["dataset_version_fingerprint"]:
        raise ValueError("Recovery 未绑定当前 Dataset Version")
    if report["inputs"]["dataloader_fingerprint"] != dataloader["dataloader_fingerprint"]:
        raise ValueError("Recovery 未绑定当前 DataLoader Profile")
    for name, relative_path in report["artifacts"].items():
        if sha256_file(output_dir / relative_path) != report["artifact_sha256"][name]:
            raise ValueError(f"Recovery 产物哈希不一致: {name}")
    reference = torch.load(output_dir / report["artifacts"]["reference_final"], map_location="cpu", weights_only=False)
    resumed = torch.load(output_dir / report["artifacts"]["resumed_final"], map_location="cpu", weights_only=False)
    if reference["loss_history"] != resumed["loss_history"]:
        raise ValueError("恢复后的 Loss History 不一致")
    if _state_sha256(reference["model_state_dict"]) != _state_sha256(resumed["model_state_dict"]):
        raise ValueError("恢复后的模型状态不一致")
    if not all(report["metrics"]["equivalence"].values()):
        raise ValueError("恢复等价性证据不完整")
    semantic_keys = (
        "recovery_version",
        "inputs",
        "config",
        "schedule",
        "checkpoint_contract",
        "metrics",
        "artifacts",
    )
    semantic = {key: report[key] for key in semantic_keys}
    if _payload_sha256(semantic) != report["recovery_fingerprint"]:
        raise ValueError("Recovery Fingerprint 不一致")
    return {
        "recovery_fingerprint": report["recovery_fingerprint"],
        "recovered_from_step": report["metrics"]["recovered_from_step"],
        "final_step": report["metrics"]["final_step"],
        "exact_recovery": True,
    }
