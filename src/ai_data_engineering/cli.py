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

    validate_parser = commands.add_parser("validate-dataset", help="验证 Lab 01 数据产物")
    validate_parser.add_argument("--output", type=Path, required=True, help="产物目录")
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
        manifest = build_dataset(args.input, args.output, args.vocab_size, args.sequence_length)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-dataset":
        result = validate_dataset(args.output)
        print("dataset validation passed")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"未处理的命令: {args.command}")
