"""Lab 03：只改变训练语料的数据版本 A/B 实验。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai_data_engineering.contracts import EXPERIMENT_SCHEMA_VERSION, EXPERIMENT_VERSION
from ai_data_engineering.pipeline import build_dataset, sha256_file, validate_dataset
from ai_data_engineering.training import TrainingConfig, train_tiny_lm, validate_training_run


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline_fingerprint": manifest["pipeline_fingerprint"],
        "source_sha256": manifest["source_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "sequence_count": manifest["metrics"]["sequence_count"],
        "sequence_length": manifest["config"]["sequence_length"],
        "trainable_token_count": manifest["metrics"]["trainable_token_count"],
        "packing_efficiency": manifest["metrics"]["packing_efficiency"],
    }


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    return {
        "run_fingerprint": run["run_fingerprint"],
        "initial_model_state_sha256": run["initial_model_state_sha256"],
        "model_state_sha256": run["model_state_sha256"],
        "initial_train_loss": metrics["initial_train_loss"],
        "final_train_loss": metrics["final_train_loss"],
        "evaluation_loss": metrics["validation_loss"],
        "trained_token_count": metrics["trained_token_count"],
        "tokens_per_second": metrics["tokens_per_second"],
    }


def _comparison(baseline_run: dict[str, Any], candidate_run: dict[str, Any]) -> dict[str, float]:
    baseline_metrics = baseline_run["metrics"]
    candidate_metrics = candidate_run["metrics"]
    baseline_evaluation_loss = baseline_metrics["validation_loss"]
    candidate_evaluation_loss = candidate_metrics["validation_loss"]
    evaluation_delta = candidate_evaluation_loss - baseline_evaluation_loss
    return {
        "candidate_minus_baseline_evaluation_loss": round(evaluation_delta, 8),
        "candidate_to_baseline_evaluation_loss_ratio": round(
            candidate_evaluation_loss / baseline_evaluation_loss, 8
        ),
        "candidate_minus_baseline_final_train_loss": round(
            candidate_metrics["final_train_loss"] - baseline_metrics["final_train_loss"], 8
        ),
        "candidate_minus_baseline_tokens_per_second": round(
            candidate_metrics["tokens_per_second"] - baseline_metrics["tokens_per_second"], 3
        ),
    }


def _experiment_semantics(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_version": manifest["experiment_version"],
        "controlled_variable": manifest["controlled_variable"],
        "inputs": manifest["inputs"],
        "controls": manifest["controls"],
        "datasets": manifest["datasets"],
        "run_fingerprints": {
            name: run["run_fingerprint"] for name, run in manifest["runs"].items()
        },
    }


def run_data_ab(
    tokenizer_corpus_path: Path,
    baseline_input_path: Path,
    candidate_input_path: Path,
    evaluation_input_path: Path,
    output_dir: Path,
    vocab_size: int,
    sequence_length: int,
    training_config: TrainingConfig,
    controlled_variable: dict[str, Any] | None = None,
    hypothesis: str | None = None,
) -> dict[str, Any]:
    """构建三个固定 Tokenizer Dataset，运行两次训练并输出 A/B Manifest。"""

    training_config.validate()
    if not training_config.strict_token_budget:
        raise ValueError("数据 A/B 必须启用 strict_token_budget")

    input_paths = {
        "tokenizer_corpus": tokenizer_corpus_path.resolve(),
        "baseline": baseline_input_path.resolve(),
        "candidate": candidate_input_path.resolve(),
        "evaluation": evaluation_input_path.resolve(),
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {name} 输入: {path}")
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    if input_hashes["baseline"] == input_hashes["candidate"]:
        raise ValueError("Baseline 与 Candidate 输入不能相同")
    if input_hashes["evaluation"] in (input_hashes["baseline"], input_hashes["candidate"]):
        raise ValueError("Evaluation 输入必须独立于训练输入")

    datasets_dir = output_dir / "datasets"
    tokenizer_dataset_dir = datasets_dir / "tokenizer-corpus"
    baseline_dataset_dir = datasets_dir / "baseline"
    candidate_dataset_dir = datasets_dir / "candidate"
    evaluation_dataset_dir = datasets_dir / "evaluation"
    tokenizer_manifest = build_dataset(
        input_paths["tokenizer_corpus"], tokenizer_dataset_dir, vocab_size, sequence_length
    )
    tokenizer_path = tokenizer_dataset_dir / tokenizer_manifest["artifacts"]["tokenizer"]
    baseline_manifest = build_dataset(
        input_paths["baseline"], baseline_dataset_dir, vocab_size, sequence_length, tokenizer_path
    )
    candidate_manifest = build_dataset(
        input_paths["candidate"], candidate_dataset_dir, vocab_size, sequence_length, tokenizer_path
    )
    evaluation_manifest = build_dataset(
        input_paths["evaluation"], evaluation_dataset_dir, vocab_size, sequence_length, tokenizer_path
    )

    tokenizer_hashes = {
        manifest["tokenizer_sha256"]
        for manifest in (tokenizer_manifest, baseline_manifest, candidate_manifest, evaluation_manifest)
    }
    if len(tokenizer_hashes) != 1:
        raise RuntimeError("A/B Dataset 未共享同一个 Tokenizer")

    runs_dir = output_dir / "runs"
    baseline_run_dir = runs_dir / "baseline"
    candidate_run_dir = runs_dir / "candidate"
    baseline_run = train_tiny_lm(
        baseline_dataset_dir,
        baseline_run_dir,
        training_config,
        evaluation_dataset_dir=evaluation_dataset_dir,
    )
    candidate_run = train_tiny_lm(
        candidate_dataset_dir,
        candidate_run_dir,
        training_config,
        evaluation_dataset_dir=evaluation_dataset_dir,
    )
    validate_training_run(baseline_dataset_dir, baseline_run_dir, evaluation_dataset_dir)
    validate_training_run(candidate_dataset_dir, candidate_run_dir, evaluation_dataset_dir)

    if baseline_run["initial_model_state_sha256"] != candidate_run["initial_model_state_sha256"]:
        raise RuntimeError("A/B 两次训练没有使用相同的初始模型状态")
    baseline_token_count = baseline_run["metrics"]["trained_token_count"]
    candidate_token_count = candidate_run["metrics"]["trained_token_count"]
    if baseline_token_count != candidate_token_count:
        raise RuntimeError("A/B 两次训练没有使用相同的有效 Token 预算")

    controlled_variable = controlled_variable or {
        "name": "training_corpus",
        "baseline": "项目原创的 AI 数据工程干净样例",
        "candidate": "注入低信息重复模板的训练样例",
        "expected_direction": None,
    }
    hypothesis = hypothesis or "在固定 Tokenizer、模型、初始化、Token 预算和评测集时，训练语料变化可能改变评测 Loss"
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "hypothesis": hypothesis,
        "controlled_variable": controlled_variable,
        "inputs": input_hashes,
        "controls": {
            "tokenizer_sha256": tokenizer_manifest["tokenizer_sha256"],
            "evaluation_dataset_fingerprint": evaluation_manifest["pipeline_fingerprint"],
            "training_config": asdict(training_config),
            "trained_token_budget": baseline_token_count,
            "initial_model_state_sha256": baseline_run["initial_model_state_sha256"],
        },
        "datasets": {
            "tokenizer_corpus": _dataset_summary(tokenizer_manifest),
            "baseline": _dataset_summary(baseline_manifest),
            "candidate": _dataset_summary(candidate_manifest),
            "evaluation": _dataset_summary(evaluation_manifest),
        },
        "runs": {
            "baseline": _run_summary(baseline_run),
            "candidate": _run_summary(candidate_run),
        },
        "comparison": _comparison(baseline_run, candidate_run),
        "artifacts": {
            "tokenizer_dataset": "datasets/tokenizer-corpus",
            "baseline_dataset": "datasets/baseline",
            "candidate_dataset": "datasets/candidate",
            "evaluation_dataset": "datasets/evaluation",
            "baseline_run": "runs/baseline",
            "candidate_run": "runs/candidate",
        },
        "evidence_boundary": [
            "该实验只验证当前原创小语料、Tiny LM 和固定训练预算",
            "Candidate 与 Baseline 的 Token 分布不同，这是受控训练语料变量的一部分",
            "吞吐差异是顺序执行的本地观测，不能单独解释为数据质量差异",
            "单次 Seed 结果不能估计实验方差",
        ],
    }
    manifest["experiment_fingerprint"] = _sha256_payload(_experiment_semantics(manifest))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "experiment_manifest.json", manifest)
    return manifest


def validate_data_ab(output_dir: Path) -> dict[str, Any]:
    """验证 A/B 的数据、训练运行、公平性控制和比较指标。"""

    manifest_path = output_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 Experiment Manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("不支持的 Experiment Manifest schema_version")
    if manifest.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("不支持的 experiment_version")

    artifacts = manifest["artifacts"]
    tokenizer_dataset_dir = output_dir / artifacts["tokenizer_dataset"]
    baseline_dataset_dir = output_dir / artifacts["baseline_dataset"]
    candidate_dataset_dir = output_dir / artifacts["candidate_dataset"]
    evaluation_dataset_dir = output_dir / artifacts["evaluation_dataset"]
    baseline_run_dir = output_dir / artifacts["baseline_run"]
    candidate_run_dir = output_dir / artifacts["candidate_run"]

    dataset_manifests = {}
    for name, path in {
        "tokenizer_corpus": tokenizer_dataset_dir,
        "baseline": baseline_dataset_dir,
        "candidate": candidate_dataset_dir,
        "evaluation": evaluation_dataset_dir,
    }.items():
        validate_dataset(path)
        dataset_manifests[name] = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if _dataset_summary(dataset_manifests[name]) != manifest["datasets"][name]:
            raise ValueError(f"{name} Dataset 与 Experiment Manifest 不一致")
        if dataset_manifests[name]["source_sha256"] != manifest["inputs"][name]:
            raise ValueError(f"{name} 输入哈希与 Dataset 不一致")

    tokenizer_hashes = {item["tokenizer_sha256"] for item in dataset_manifests.values()}
    if tokenizer_hashes != {manifest["controls"]["tokenizer_sha256"]}:
        raise ValueError("A/B Dataset 没有共享受控 Tokenizer")
    if (
        dataset_manifests["evaluation"]["pipeline_fingerprint"]
        != manifest["controls"]["evaluation_dataset_fingerprint"]
    ):
        raise ValueError("受控 Evaluation Dataset Fingerprint 不一致")

    baseline_result = validate_training_run(
        baseline_dataset_dir, baseline_run_dir, evaluation_dataset_dir
    )
    candidate_result = validate_training_run(
        candidate_dataset_dir, candidate_run_dir, evaluation_dataset_dir
    )
    baseline_run = json.loads((baseline_run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    candidate_run = json.loads((candidate_run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if _run_summary(baseline_run) != manifest["runs"]["baseline"]:
        raise ValueError("Baseline Run 与 Experiment Manifest 不一致")
    if _run_summary(candidate_run) != manifest["runs"]["candidate"]:
        raise ValueError("Candidate Run 与 Experiment Manifest 不一致")
    if baseline_result["initial_model_state_sha256"] != candidate_result["initial_model_state_sha256"]:
        raise ValueError("A/B 初始模型状态不一致")
    if baseline_result["initial_model_state_sha256"] != manifest["controls"]["initial_model_state_sha256"]:
        raise ValueError("受控初始模型状态与 Run 不一致")

    baseline_token_count = baseline_run["metrics"]["trained_token_count"]
    candidate_token_count = candidate_run["metrics"]["trained_token_count"]
    if baseline_token_count != candidate_token_count:
        raise ValueError("A/B 有效 Token 预算不一致")
    config = TrainingConfig(**manifest["controls"]["training_config"])
    if baseline_run["config"] != asdict(config) or candidate_run["config"] != asdict(config):
        raise ValueError("A/B Run 没有共享受控训练配置")
    expected_token_budget = (
        config.steps * config.batch_size * manifest["datasets"]["baseline"]["sequence_length"]
    )
    if not config.strict_token_budget:
        raise ValueError("A/B 未启用 strict_token_budget")
    if baseline_token_count != manifest["controls"]["trained_token_budget"]:
        raise ValueError("Experiment Manifest 的 Token 预算与 Run 不一致")
    if baseline_token_count != expected_token_budget:
        raise ValueError("A/B Run 未满足严格 Token 预算")

    if _comparison(baseline_run, candidate_run) != manifest["comparison"]:
        raise ValueError("A/B 比较指标与 Run 不一致")
    expected_fingerprint = _sha256_payload(_experiment_semantics(manifest))
    if expected_fingerprint != manifest["experiment_fingerprint"]:
        raise ValueError("Experiment Fingerprint 与受控输入不一致")
    if not all(math.isfinite(run["evaluation_loss"]) for run in manifest["runs"].values()):
        raise ValueError("A/B Evaluation Loss 出现非有限值")

    return {
        "experiment_fingerprint": manifest["experiment_fingerprint"],
        "baseline_evaluation_loss": manifest["runs"]["baseline"]["evaluation_loss"],
        "candidate_evaluation_loss": manifest["runs"]["candidate"]["evaluation_loss"],
        "candidate_minus_baseline_evaluation_loss": manifest["comparison"][
            "candidate_minus_baseline_evaluation_loss"
        ],
        "trained_token_budget": baseline_token_count,
    }
