PYTHON := uv run python

.PHONY: setup lint test lab00 lab01 validate clean

setup:
	uv sync --python 3.11 --extra dev

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

validate: lint test lab00 lab01

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
