PYTHON := uv run python

.PHONY: setup lint test lab00 lab01 lab02 lab03 validate clean

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

validate: lint test lab00 lab01 lab02 lab03

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
