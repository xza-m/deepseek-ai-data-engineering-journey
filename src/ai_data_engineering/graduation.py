"""Lab 10：多 Seed 数据策略 A/B、失败血缘与毕业复现报告。"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ai_data_engineering.contracts import GRADUATION_SCHEMA_VERSION, GRADUATION_VERSION
from ai_data_engineering.experiment import run_data_ab, validate_data_ab
from ai_data_engineering.pipeline import sha256_file
from ai_data_engineering.training import TrainingConfig, _build_model, _masked_cross_entropy


@dataclass(frozen=True)
class GraduationConfig:
    """正式 A/B 与毕业报告配置。"""

    seeds: tuple[int, ...] = (11, 29, 47)
    steps: int = 60
    batch_size: int = 2
    learning_rate: float = 0.005
    embedding_dim: int = 32
    num_layers: int = 1
    num_heads: int = 4
    feed_forward_dim: int = 64
    vocab_size: int = 384
    sequence_length: int = 64

    def validate(self) -> None:
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("毕业实验至少需要两个不重复 Seed")
        if self.steps < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("训练配置必须为正数")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("num_heads 必须整除 embedding_dim")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _payload_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_policy_candidate(
    train_path: Path,
    accepted_path: Path,
    rejected_path: Path,
    contamination_path: Path,
    candidate_path: Path,
    policy_diff_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = _read_jsonl(train_path)
    accepted = _read_jsonl(accepted_path)
    rejected = {record["source_id"]: record for record in _read_jsonl(rejected_path)}
    contaminated = {record["left_source_id"] for record in _read_jsonl(contamination_path)}
    candidate = [record for record in accepted if record["source_id"] not in contaminated]
    candidate_ids = {record["source_id"] for record in candidate}
    if len(candidate) < 2:
        raise ValueError("质量策略后的 Candidate 文档不足")
    policy_diff: list[dict[str, Any]] = []
    for record in train:
        source_id = record["source_id"]
        reasons: list[str] = []
        if source_id in rejected:
            reasons.extend(rejected[source_id]["metadata"]["quality_audit"]["reasons"])
        if source_id in contaminated:
            reasons.append("evaluation_contamination")
        policy_diff.append(
            {
                "source_id": source_id,
                "baseline_policy": "include",
                "candidate_policy": "include" if source_id in candidate_ids else "exclude",
                "reasons": sorted(set(reasons)) if reasons else ["quality_policy_accept"],
            }
        )
    _write_jsonl(candidate_path, candidate)
    _write_jsonl(policy_diff_path, policy_diff)
    return candidate, policy_diff


def _training_config(config: GraduationConfig, seed: int) -> TrainingConfig:
    return TrainingConfig(
        steps=config.steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        feed_forward_dim=config.feed_forward_dim,
        seed=seed,
        strict_token_budget=True,
    )


def _sequence_losses(experiment_dir: Path, run_name: str) -> list[dict[str, Any]]:
    run_dir = experiment_dir / "runs" / run_name
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    dataset_dir = experiment_dir / "datasets" / run_name
    dataset_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation_dir = experiment_dir / "datasets" / "evaluation"
    evaluation_manifest = json.loads((evaluation_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation_records = _read_jsonl(evaluation_dir / evaluation_manifest["artifacts"]["sequences"])
    normalized = _read_jsonl(evaluation_dir / evaluation_manifest["artifacts"]["normalized_documents"])
    source_by_doc = {record["doc_id"]: record["source_id"] for record in normalized}
    training_config = TrainingConfig(**run["config"])
    model = _build_model(dataset_manifest, training_config)
    checkpoint = torch.load(run_dir / run["artifacts"]["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in evaluation_records:
            input_ids = torch.tensor([record["input_ids"]], dtype=torch.long)
            labels = torch.tensor([record["labels"]], dtype=torch.long)
            loss_mask = torch.tensor([record["loss_mask"]], dtype=torch.float32)
            loss = _masked_cross_entropy(model(input_ids), labels, loss_mask)
            results.append(
                {
                    "sequence_id": record["sequence_id"],
                    "source_ids": sorted({source_by_doc[span["doc_id"]] for span in record["provenance"]}),
                    "loss": round(float(loss.item()), 8),
                    "trainable_token_count": int(loss_mask.sum().item()),
                }
            )
    return results


def _failure_examples(experiment_dirs: list[Path]) -> list[dict[str, Any]]:
    by_sequence: dict[str, dict[str, Any]] = {}
    for experiment_dir in experiment_dirs:
        baseline = {item["sequence_id"]: item for item in _sequence_losses(experiment_dir, "baseline")}
        candidate = {item["sequence_id"]: item for item in _sequence_losses(experiment_dir, "candidate")}
        if baseline.keys() != candidate.keys():
            raise RuntimeError("A/B Evaluation Sequence 不一致")
        for sequence_id in baseline:
            target = by_sequence.setdefault(
                sequence_id,
                {
                    "sequence_id": sequence_id,
                    "evaluation_source_ids": baseline[sequence_id]["source_ids"],
                    "trainable_token_count": baseline[sequence_id]["trainable_token_count"],
                    "baseline_losses": [],
                    "candidate_losses": [],
                },
            )
            target["baseline_losses"].append(baseline[sequence_id]["loss"])
            target["candidate_losses"].append(candidate[sequence_id]["loss"])
    examples: list[dict[str, Any]] = []
    for value in by_sequence.values():
        baseline_mean = statistics.mean(value.pop("baseline_losses"))
        candidate_mean = statistics.mean(value.pop("candidate_losses"))
        examples.append(
            {
                **value,
                "baseline_mean_loss": round(baseline_mean, 8),
                "candidate_mean_loss": round(candidate_mean, 8),
                "candidate_minus_baseline_loss": round(candidate_mean - baseline_mean, 8),
                "attribution_boundary": "Sequence 差异与 Policy Diff 可追溯，但微型实验不能证明单文档因果贡献",
            }
        )
    return sorted(examples, key=lambda item: (-item["candidate_minus_baseline_loss"], item["sequence_id"]))[:3]


def _aggregate_seed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [item["candidate_minus_baseline_evaluation_loss"] for item in seed_results]
    baseline_losses = [item["baseline_evaluation_loss"] for item in seed_results]
    candidate_losses = [item["candidate_evaluation_loss"] for item in seed_results]
    return {
        "seed_count": len(seed_results),
        "baseline_evaluation_loss_mean": round(statistics.mean(baseline_losses), 8),
        "baseline_evaluation_loss_population_std": round(statistics.pstdev(baseline_losses), 8),
        "candidate_evaluation_loss_mean": round(statistics.mean(candidate_losses), 8),
        "candidate_evaluation_loss_population_std": round(statistics.pstdev(candidate_losses), 8),
        "candidate_minus_baseline_evaluation_loss_mean": round(statistics.mean(deltas), 8),
        "candidate_minus_baseline_evaluation_loss_population_std": round(statistics.pstdev(deltas), 8),
        "candidate_better_seed_count": sum(delta < 0 for delta in deltas),
        "candidate_worse_seed_count": sum(delta > 0 for delta in deltas),
        "equal_seed_count": sum(delta == 0 for delta in deltas),
    }


def _graduation_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    rows = [
        f"| {item['seed']} | {item['baseline_evaluation_loss']} | {item['candidate_evaluation_loss']} | "
        f"{item['candidate_minus_baseline_evaluation_loss']} |"
        for item in report["seed_results"]
    ]
    failures = [
        f"- `{item['sequence_id']}` / {', '.join(item['evaluation_source_ids'])}: "
        f"Candidate-Baseline = {item['candidate_minus_baseline_loss']}"
        for item in report["failure_examples"]
    ]
    return "\n".join(
        [
            "# Lab 10 毕业实验报告",
            "",
            f"Graduation Fingerprint：`{report['graduation_fingerprint']}`",
            "",
            "## 实验问题",
            "",
            "只改变训练数据准入策略：Baseline 包含全部候选文档；Candidate 使用 Lab 04 质量准入，并排除评测污染候选。",
            "Tokenizer、模型结构、评测集、每个 Seed 的初始化和有效 Token 预算保持一致。",
            "",
            "## 多 Seed 结果",
            "",
            "| Seed | Baseline Eval Loss | Candidate Eval Loss | Delta |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
            f"平均 Delta：{aggregate['candidate_minus_baseline_evaluation_loss_mean']}；"
            f"总体标准差：{aggregate['candidate_minus_baseline_evaluation_loss_population_std']}。",
            "",
            "三次 Seed 只能暴露方差，不能提供生产统计显著性结论。结果方向不作为毕业标准，控制变量和可复现性才是。",
            "",
            "## 失败 Sequence 与血缘",
            "",
            *failures,
            "",
            "这些 Sequence 可回到 Evaluation Source；训练数据变化可回到 `training_policy_diff.jsonl`。",
            "关联不等于单文档因果。",
            "",
            "## 复现",
            "",
            "```bash",
            "make setup",
            "make journey",
            "make validate",
            "```",
            "",
            "CI 负责干净 Linux 环境的机器复现；首位学习者的人工复现记录是学习过程的毕业证据，不在仓库初始化时伪造。",
            "",
        ]
    )


def _upstream_reports(artifacts_root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "quality": artifacts_root / "lab04/quality_report.json",
        "dataset_version": artifacts_root / "lab05/dataset_version_manifest.json",
        "compute": artifacts_root / "lab06/compute_report.json",
        "dataloader": artifacts_root / "lab07/dataloader_report.json",
        "recovery": artifacts_root / "lab08/recovery_report.json",
        "storage": artifacts_root / "lab09/storage_report.json",
    }
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}


def run_graduation_lab(
    data_root: Path,
    artifacts_root: Path,
    output_dir: Path,
    config: GraduationConfig | None = None,
) -> dict[str, Any]:
    """执行质量策略多 Seed A/B，并生成端到端毕业证据。"""

    config = config or GraduationConfig()
    config.validate()
    upstream = _upstream_reports(artifacts_root)
    train_path = data_root / "lab04/train.jsonl"
    evaluation_path = data_root / "lab04/evaluation.jsonl"
    tokenizer_corpus_path = data_root / "lab03/tokenizer_corpus.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = output_dir / "experiments"
    if experiments_dir.exists():
        shutil.rmtree(experiments_dir)
    candidate_path = output_dir / "quality_policy_candidate.jsonl"
    policy_diff_path = output_dir / "training_policy_diff.jsonl"
    candidate, policy_diff = _build_policy_candidate(
        train_path,
        artifacts_root / "lab04/accepted_documents.jsonl",
        artifacts_root / "lab04/rejected_documents.jsonl",
        artifacts_root / "lab04/contamination_pairs.jsonl",
        candidate_path,
        policy_diff_path,
    )

    seed_results: list[dict[str, Any]] = []
    experiment_dirs: list[Path] = []
    for seed in config.seeds:
        experiment_dir = experiments_dir / f"seed-{seed}"
        experiment = run_data_ab(
            tokenizer_corpus_path,
            train_path,
            candidate_path,
            evaluation_path,
            experiment_dir,
            config.vocab_size,
            config.sequence_length,
            _training_config(config, seed),
            controlled_variable={
                "name": "dataset_admission_policy_version",
                "baseline": "raw-policy-v0：全部项目候选文档",
                "candidate": "quality-policy-v1：质量准入且排除评测污染候选",
                "expected_direction": None,
            },
            hypothesis="固定训练与评测控制时，数据准入策略版本可能改变 Evaluation Loss",
        )
        validate_data_ab(experiment_dir)
        seed_results.append(
            {
                "seed": seed,
                "experiment_fingerprint": experiment["experiment_fingerprint"],
                "initial_model_state_sha256": experiment["controls"]["initial_model_state_sha256"],
                "trained_token_budget": experiment["controls"]["trained_token_budget"],
                "tokenizer_sha256": experiment["controls"]["tokenizer_sha256"],
                "evaluation_dataset_fingerprint": experiment["controls"]["evaluation_dataset_fingerprint"],
                "baseline_evaluation_loss": experiment["runs"]["baseline"]["evaluation_loss"],
                "candidate_evaluation_loss": experiment["runs"]["candidate"]["evaluation_loss"],
                "candidate_minus_baseline_evaluation_loss": experiment["comparison"][
                    "candidate_minus_baseline_evaluation_loss"
                ],
            }
        )
        experiment_dirs.append(experiment_dir)
    if len({item["tokenizer_sha256"] for item in seed_results}) != 1:
        raise RuntimeError("多 Seed 实验没有共享 Tokenizer")
    if len({item["evaluation_dataset_fingerprint"] for item in seed_results}) != 1:
        raise RuntimeError("多 Seed 实验没有共享 Evaluation Dataset")
    if len({item["trained_token_budget"] for item in seed_results}) != 1:
        raise RuntimeError("多 Seed 实验的有效 Token 预算不一致")

    upstream_fingerprints = {
        "quality": upstream["quality"]["quality_fingerprint"],
        "dataset_version": upstream["dataset_version"]["dataset_version_fingerprint"],
        "compute": upstream["compute"]["compute_fingerprint"],
        "dataloader": upstream["dataloader"]["dataloader_fingerprint"],
        "recovery": upstream["recovery"]["recovery_fingerprint"],
        "storage": upstream["storage"]["storage_fingerprint"],
    }
    semantic = {
        "graduation_version": GRADUATION_VERSION,
        "controlled_variable": {
            "name": "dataset_admission_policy_version",
            "baseline": "raw-policy-v0",
            "candidate": "quality-policy-v1",
            "candidate_rule": "Lab 04 accept AND not evaluation contamination candidate",
        },
        "upstream_fingerprints": upstream_fingerprints,
        "config": {**asdict(config), "seeds": list(config.seeds)},
        "inputs": {
            "tokenizer_corpus_sha256": sha256_file(tokenizer_corpus_path),
            "baseline_train_sha256": sha256_file(train_path),
            "candidate_train_sha256": sha256_file(candidate_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "policy_diff_sha256": sha256_file(policy_diff_path),
        },
        "policy_metrics": {
            "baseline_document_count": len(_read_jsonl(train_path)),
            "candidate_document_count": len(candidate),
            "excluded_document_count": sum(item["candidate_policy"] == "exclude" for item in policy_diff),
        },
        "seed_results": seed_results,
        "aggregate": _aggregate_seed_results(seed_results),
        "failure_examples": _failure_examples(experiment_dirs),
        "artifacts": {
            "candidate_train": candidate_path.name,
            "training_policy_diff": policy_diff_path.name,
            "experiments": "experiments",
            "graduation_report": "graduation_report.md",
            "reproduction_manifest": "reproduction_manifest.json",
        },
    }
    report = {
        "schema_version": GRADUATION_SCHEMA_VERSION,
        **semantic,
        "evidence_boundary": [
            "三次 Seed 和 Tiny LM 只估计当前原创微型语料下的方差",
            "失败 Sequence 与训练 Policy Diff 的血缘关联不是单文档因果归因",
            "CI 是机器干净环境复现；人工独立复现必须由真实学习者完成",
            "smallpond 与 3FS 的分布式集群证据仍属于条件化 Infra 支线",
        ],
    }
    report["graduation_fingerprint"] = _payload_sha256(semantic)
    markdown_path = output_dir / report["artifacts"]["graduation_report"]
    markdown_path.write_text(_graduation_markdown(report), encoding="utf-8")
    lock_path = data_root.parent / "uv.lock"
    reproduction = {
        "schema_version": "0.1",
        "graduation_fingerprint": report["graduation_fingerprint"],
        "commands": ["make setup", "make journey", "make validate"],
        "python": "3.11",
        "dependency_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "upstream_fingerprints": upstream_fingerprints,
        "automated_clean_environment": "github-actions-linux-configured",
        "human_independent_reproduction": "ready_for_first_learner",
        "boundary": "不在仓库初始化时伪造人工复现记录",
    }
    reproduction_path = output_dir / report["artifacts"]["reproduction_manifest"]
    _write_json(reproduction_path, reproduction)
    report["artifact_sha256"] = {
        "candidate_train": sha256_file(candidate_path),
        "training_policy_diff": sha256_file(policy_diff_path),
        "graduation_report": sha256_file(markdown_path),
        "reproduction_manifest": sha256_file(reproduction_path),
    }
    _write_json(output_dir / "graduation_report.json", report)
    return report


def _validate_upstream(report: dict[str, Any], artifacts_root: Path) -> None:
    upstream = _upstream_reports(artifacts_root)
    current = {
        "quality": upstream["quality"]["quality_fingerprint"],
        "dataset_version": upstream["dataset_version"]["dataset_version_fingerprint"],
        "compute": upstream["compute"]["compute_fingerprint"],
        "dataloader": upstream["dataloader"]["dataloader_fingerprint"],
        "recovery": upstream["recovery"]["recovery_fingerprint"],
        "storage": upstream["storage"]["storage_fingerprint"],
    }
    if report["upstream_fingerprints"] != current:
        raise ValueError("Graduation Report 上游 Fingerprint 不一致")


def validate_graduation_lab(data_root: Path, artifacts_root: Path, output_dir: Path) -> dict[str, Any]:
    """重算毕业实验的运行、聚合、血缘产物和复现契约。"""

    report = json.loads((output_dir / "graduation_report.json").read_text(encoding="utf-8"))
    _validate_upstream(report, artifacts_root)
    recomputed_seeds: list[dict[str, Any]] = []
    experiment_dirs: list[Path] = []
    for expected in report["seed_results"]:
        experiment_dir = output_dir / "experiments" / f"seed-{expected['seed']}"
        validate_data_ab(experiment_dir)
        experiment = json.loads((experiment_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
        actual = {
            "seed": expected["seed"],
            "experiment_fingerprint": experiment["experiment_fingerprint"],
            "initial_model_state_sha256": experiment["controls"]["initial_model_state_sha256"],
            "trained_token_budget": experiment["controls"]["trained_token_budget"],
            "tokenizer_sha256": experiment["controls"]["tokenizer_sha256"],
            "evaluation_dataset_fingerprint": experiment["controls"]["evaluation_dataset_fingerprint"],
            "baseline_evaluation_loss": experiment["runs"]["baseline"]["evaluation_loss"],
            "candidate_evaluation_loss": experiment["runs"]["candidate"]["evaluation_loss"],
            "candidate_minus_baseline_evaluation_loss": experiment["comparison"][
                "candidate_minus_baseline_evaluation_loss"
            ],
        }
        recomputed_seeds.append(actual)
        experiment_dirs.append(experiment_dir)
    if recomputed_seeds != report["seed_results"]:
        raise ValueError("Graduation Seed Result 与实验不一致")
    if _aggregate_seed_results(recomputed_seeds) != report["aggregate"]:
        raise ValueError("Graduation 多 Seed 聚合不一致")
    if _failure_examples(experiment_dirs) != report["failure_examples"]:
        raise ValueError("Graduation 失败 Sequence 归因不一致")
    for name in ("candidate_train", "training_policy_diff", "graduation_report", "reproduction_manifest"):
        path = output_dir / report["artifacts"][name]
        if sha256_file(path) != report["artifact_sha256"][name]:
            raise ValueError(f"Graduation 产物哈希不一致: {name}")
    reproduction = json.loads(
        (output_dir / report["artifacts"]["reproduction_manifest"]).read_text(encoding="utf-8")
    )
    if reproduction["graduation_fingerprint"] != report["graduation_fingerprint"]:
        raise ValueError("Reproduction Manifest 未绑定 Graduation Fingerprint")
    semantic_keys = (
        "graduation_version",
        "controlled_variable",
        "upstream_fingerprints",
        "config",
        "inputs",
        "policy_metrics",
        "seed_results",
        "aggregate",
        "failure_examples",
        "artifacts",
    )
    semantic = {key: report[key] for key in semantic_keys}
    if _payload_sha256(semantic) != report["graduation_fingerprint"]:
        raise ValueError("Graduation Fingerprint 不一致")
    if not all(math.isfinite(item["candidate_minus_baseline_evaluation_loss"]) for item in report["seed_results"]):
        raise ValueError("Graduation A/B 出现非有限值")
    if sha256_file(output_dir / report["artifacts"]["candidate_train"]) != report["inputs"][
        "candidate_train_sha256"
    ]:
        raise ValueError("Graduation Candidate 输入哈希不一致")
    return {
        "graduation_fingerprint": report["graduation_fingerprint"],
        "seed_count": report["aggregate"]["seed_count"],
        "mean_evaluation_loss_delta": report["aggregate"]["candidate_minus_baseline_evaluation_loss_mean"],
        "failure_example_count": len(report["failure_examples"]),
        "machine_reproduction_ready": True,
    }
