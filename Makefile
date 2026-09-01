PYTHON := uv run python

.PHONY: setup lint test lab00 lab01 lab02 lab03 lab04 lab05 lab06 lab07 lab08 lab09 validate clean

setup:
	uv sync --python 3.11 --extra dev --extra train

lint:
	uv run ruff check .

test:
	uv run pytest

lab00:
	uv run aide environment --output artifacts/lab00/environment.json

lab01:
	uv run aide build-dataset \
		--input data/sample/raw_documents.jsonl \
		--output artifacts/lab01 \
		--vocab-size 320 \
		--sequence-length 64
	uv run aide validate-dataset --output artifacts/lab01

lab02: lab01
	uv run aide train-tiny-lm \
		--dataset artifacts/lab01 \
		--output artifacts/lab02
	uv run aide validate-training-run \
		--dataset artifacts/lab01 \
		--output artifacts/lab02

lab03: lab02
	uv run aide run-data-ab \
		--tokenizer-corpus data/lab03/tokenizer_corpus.jsonl \
		--baseline data/lab03/train_baseline.jsonl \
		--candidate data/lab03/train_candidate.jsonl \
		--evaluation data/lab03/evaluation.jsonl \
		--output artifacts/lab03
	uv run aide validate-data-ab --output artifacts/lab03

lab04: lab03
	uv run aide audit-quality \
		--train data/lab04/train.jsonl \
		--evaluation data/lab04/evaluation.jsonl \
		--truth data/lab04/duplicate_truth.jsonl \
		--output artifacts/lab04
	uv run aide validate-quality \
		--train data/lab04/train.jsonl \
		--evaluation data/lab04/evaluation.jsonl \
		--truth data/lab04/duplicate_truth.jsonl \
		--output artifacts/lab04

lab05: lab04
	uv run aide build-version \
		--input artifacts/lab04/accepted_documents.jsonl \
		--quality-report artifacts/lab04/quality_report.json \
		--mix-spec data/lab05/mix_spec.json \
		--output artifacts/lab05
	uv run aide validate-version \
		--input artifacts/lab04/accepted_documents.jsonl \
		--quality-report artifacts/lab04/quality_report.json \
		--mix-spec data/lab05/mix_spec.json \
		--output artifacts/lab05

lab06: lab05
	uv run aide run-compute \
		--input artifacts/lab05 \
		--output artifacts/lab06
	uv run aide validate-compute \
		--input artifacts/lab05 \
		--output artifacts/lab06

lab07: lab06
	uv run aide profile-dataloader \
		--input artifacts/lab05 \
		--compute-report artifacts/lab06/compute_report.json \
		--output artifacts/lab07
	uv run aide validate-dataloader \
		--input artifacts/lab05 \
		--compute-report artifacts/lab06/compute_report.json \
		--output artifacts/lab07

lab08: lab07
	uv run aide run-recovery \
		--input artifacts/lab05 \
		--dataloader-report artifacts/lab07/dataloader_report.json \
		--output artifacts/lab08
	uv run aide validate-recovery \
		--input artifacts/lab05 \
		--dataloader-report artifacts/lab07/dataloader_report.json \
		--output artifacts/lab08

lab09: lab08
	uv run aide benchmark-storage \
		--input artifacts/lab05 \
		--compute artifacts/lab06 \
		--recovery artifacts/lab08 \
		--output artifacts/lab09
	uv run aide validate-storage \
		--input artifacts/lab05 \
		--compute artifacts/lab06 \
		--recovery artifacts/lab08 \
		--output artifacts/lab09

validate: lint test lab00 lab01 lab02 lab03 lab04 lab05 lab06 lab07 lab08 lab09

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
