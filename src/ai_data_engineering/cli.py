"""项目命令行入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ai_data_engineering.environment import build_environment_report
from ai_data_engineering.pipeline import build_dataset, validate_dataset


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aide", description="AI 数据工程学习实验 CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    environment_parser = commands.add_parser("environment", help="生成本地环境能力报告")
    environment_parser.add_argument("--output", type=Path, help="JSON 输出路径")

    build_parser = commands.add_parser("build-dataset", help="执行 Lab 01 文档到训练 Token 流水线")
    build_parser.add_argument("--input", type=Path, required=True, help="输入 JSONL")
    build_parser.add_argument("--output", type=Path, required=True, help="产物目录")
    build_parser.add_argument("--vocab-size", type=int, default=320)
    build_parser.add_argument("--sequence-length", type=int, default=64)
    build_parser.add_argument("--tokenizer", type=Path, help="复用已有 Tokenizer，不重新训练词表")

    validate_parser = commands.add_parser("validate-dataset", help="验证 Lab 01 数据产物")
    validate_parser.add_argument("--output", type=Path, required=True, help="产物目录")

    train_parser = commands.add_parser("train-tiny-lm", help="执行 Lab 02 Tiny LM 基线训练")
    train_parser.add_argument("--dataset", type=Path, required=True, help="Lab 01 Dataset 目录")
    train_parser.add_argument("--evaluation-dataset", type=Path, help="独立 Evaluation Dataset 目录")
    train_parser.add_argument("--output", type=Path, required=True, help="训练产物目录")
    train_parser.add_argument("--steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--embedding-dim", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--num-layers", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--num-heads", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--feed-forward-dim", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--validation-fraction", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--strict-token-budget", action="store_true", default=argparse.SUPPRESS)

    validate_training_parser = commands.add_parser("validate-training-run", help="验证 Lab 02 训练产物")
    validate_training_parser.add_argument("--dataset", type=Path, required=True, help="Lab 01 Dataset 目录")
    validate_training_parser.add_argument("--evaluation-dataset", type=Path, help="独立 Evaluation Dataset 目录")
    validate_training_parser.add_argument("--output", type=Path, required=True, help="训练产物目录")

    experiment_parser = commands.add_parser("run-data-ab", help="执行 Lab 03 受控数据版本 A/B")
    experiment_parser.add_argument("--tokenizer-corpus", type=Path, required=True)
    experiment_parser.add_argument("--baseline", type=Path, required=True)
    experiment_parser.add_argument("--candidate", type=Path, required=True)
    experiment_parser.add_argument("--evaluation", type=Path, required=True)
    experiment_parser.add_argument("--output", type=Path, required=True)
    experiment_parser.add_argument("--vocab-size", type=int, default=384)
    experiment_parser.add_argument("--sequence-length", type=int, default=64)
    experiment_parser.add_argument("--steps", type=int, default=argparse.SUPPRESS)
    experiment_parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    experiment_parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    experiment_parser.add_argument("--embedding-dim", type=int, default=argparse.SUPPRESS)
    experiment_parser.add_argument("--num-layers", type=int, default=argparse.SUPPRESS)
    experiment_parser.add_argument("--num-heads", type=int, default=argparse.SUPPRESS)
    experiment_parser.add_argument("--feed-forward-dim", type=int, default=argparse.SUPPRESS)
    experiment_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    validate_experiment_parser = commands.add_parser("validate-data-ab", help="验证 Lab 03 A/B 实验")
    validate_experiment_parser.add_argument("--output", type=Path, required=True)

    quality_parser = commands.add_parser("audit-quality", help="执行 Lab 04 近似去重、污染与质量审计")
    quality_parser.add_argument("--train", type=Path, required=True)
    quality_parser.add_argument("--evaluation", type=Path, required=True)
    quality_parser.add_argument("--truth", type=Path, required=True)
    quality_parser.add_argument("--output", type=Path, required=True)
    quality_parser.add_argument("--similarity-threshold", type=float, default=0.45)
    quality_parser.add_argument("--contamination-threshold", type=float, default=0.45)
    quality_parser.add_argument("--shingle-size", type=int, default=3)
    quality_parser.add_argument("--min-characters", type=int, default=40)

    validate_quality_parser = commands.add_parser("validate-quality", help="验证 Lab 04 质量审计")
    validate_quality_parser.add_argument("--train", type=Path, required=True)
    validate_quality_parser.add_argument("--evaluation", type=Path, required=True)
    validate_quality_parser.add_argument("--truth", type=Path, required=True)
    validate_quality_parser.add_argument("--output", type=Path, required=True)

    version_parser = commands.add_parser("build-version", help="执行 Lab 05 混合、Packing 与 Sharding")
    version_parser.add_argument("--input", type=Path, required=True)
    version_parser.add_argument("--quality-report", type=Path, required=True)
    version_parser.add_argument("--mix-spec", type=Path, required=True)
    version_parser.add_argument("--output", type=Path, required=True)
    version_parser.add_argument("--vocab-size", type=int, default=320)
    version_parser.add_argument("--sequence-length", type=int, default=64)
    version_parser.add_argument("--max-sequences-per-shard", type=int, default=4)
    version_parser.add_argument("--seed", type=int, default=42)

    validate_version_parser = commands.add_parser("validate-version", help="验证 Lab 05 Dataset Version")
    validate_version_parser.add_argument("--input", type=Path, required=True)
    validate_version_parser.add_argument("--quality-report", type=Path, required=True)
    validate_version_parser.add_argument("--mix-spec", type=Path, required=True)
    validate_version_parser.add_argument("--output", type=Path, required=True)

    compute_parser = commands.add_parser("run-compute", help="执行 Lab 06 DuckDB、分区、倾斜与恢复实验")
    compute_parser.add_argument("--input", type=Path, required=True, help="Lab 05 产物目录")
    compute_parser.add_argument("--output", type=Path, required=True)
    compute_parser.add_argument("--repeat-factor", type=int, default=64)
    compute_parser.add_argument("--partition-count", type=int, default=8)
    compute_parser.add_argument("--hot-key-fraction", type=float, default=0.75)
    compute_parser.add_argument("--failure-partition", type=int, default=3)

    validate_compute_parser = commands.add_parser("validate-compute", help="验证 Lab 06 计算实验")
    validate_compute_parser.add_argument("--input", type=Path, required=True, help="Lab 05 产物目录")
    validate_compute_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)
    if args.command == "environment":
        report = build_environment_report()
        if args.output:
            _write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "build-dataset":
        manifest = build_dataset(args.input, args.output, args.vocab_size, args.sequence_length, args.tokenizer)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-dataset":
        result = validate_dataset(args.output)
        print("dataset validation passed")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "train-tiny-lm":
        try:
            from ai_data_engineering.training import TrainingConfig, train_tiny_lm
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise SystemExit("训练依赖未安装，请运行: uv sync --extra train") from error
            raise

        config_fields = {
            name: getattr(args, name)
            for name in (
                "steps",
                "batch_size",
                "learning_rate",
                "embedding_dim",
                "num_layers",
                "num_heads",
                "feed_forward_dim",
                "validation_fraction",
                "seed",
                "strict_token_budget",
            )
            if hasattr(args, name)
        }
        run_manifest = train_tiny_lm(
            args.dataset,
            args.output,
            TrainingConfig(**config_fields),
            evaluation_dataset_dir=args.evaluation_dataset,
        )
        print(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-training-run":
        try:
            from ai_data_engineering.training import validate_training_run
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise SystemExit("训练依赖未安装，请运行: uv sync --extra train") from error
            raise

        result = validate_training_run(args.dataset, args.output, args.evaluation_dataset)
        print("training run validation passed")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "run-data-ab":
        try:
            from ai_data_engineering.experiment import run_data_ab
            from ai_data_engineering.training import TrainingConfig
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise SystemExit("训练依赖未安装，请运行: uv sync --extra train") from error
            raise

        training_fields = {
            name: getattr(args, name)
            for name in (
                "steps",
                "batch_size",
                "learning_rate",
                "embedding_dim",
                "num_layers",
                "num_heads",
                "feed_forward_dim",
                "seed",
            )
            if hasattr(args, name)
        }
        training_config = TrainingConfig(**training_fields, strict_token_budget=True)
        result = run_data_ab(
            args.tokenizer_corpus,
            args.baseline,
            args.candidate,
            args.evaluation,
            args.output,
            args.vocab_size,
            args.sequence_length,
            training_config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-data-ab":
        try:
            from ai_data_engineering.experiment import validate_data_ab
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise SystemExit("训练依赖未安装，请运行: uv sync --extra train") from error
            raise

        result = validate_data_ab(args.output)
        print("data A/B validation passed")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "audit-quality":
        from ai_data_engineering.quality import QualityConfig, run_quality_audit

        result = run_quality_audit(
            args.train,
            args.evaluation,
            args.truth,
            args.output,
            QualityConfig(
                similarity_threshold=args.similarity_threshold,
                contamination_threshold=args.contamination_threshold,
                shingle_size=args.shingle_size,
                min_characters=args.min_characters,
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-quality":
        from ai_data_engineering.quality import validate_quality_audit

        result = validate_quality_audit(args.train, args.evaluation, args.truth, args.output)
        print("quality audit validation passed")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "build-version":
        from ai_data_engineering.dataset_version import VersionConfig, build_dataset_version

        result = build_dataset_version(
            args.input,
            args.quality_report,
            args.mix_spec,
            args.output,
            VersionConfig(
                vocab_size=args.vocab_size,
                sequence_length=args.sequence_length,
                max_sequences_per_shard=args.max_sequences_per_shard,
                seed=args.seed,
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-version":
        from ai_data_engineering.dataset_version import validate_dataset_version

        result = validate_dataset_version(args.input, args.quality_report, args.mix_spec, args.output)
        print("dataset version validation passed")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "run-compute":
        from ai_data_engineering.compute_lab import ComputeConfig, run_compute_lab

        result = run_compute_lab(
            args.input,
            args.output,
            ComputeConfig(
                repeat_factor=args.repeat_factor,
                partition_count=args.partition_count,
                hot_key_fraction=args.hot_key_fraction,
                failure_partition=args.failure_partition,
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-compute":
        from ai_data_engineering.compute_lab import validate_compute_lab

        result = validate_compute_lab(args.input, args.output)
        print("compute lab validation passed")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"未处理的命令: {args.command}")
