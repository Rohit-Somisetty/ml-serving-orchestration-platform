POETRY ?= poetry
PYTHON ?= python

.PHONY: install lint typecheck test format api data train batch

install:
	$(POETRY) install

lint:
	$(POETRY) run ruff check src tests

format:
	$(POETRY) run ruff format src tests

typecheck:
	$(POETRY) run mypy src tests

test:
	$(POETRY) run pytest

data:
	$(POETRY) run mlp data-generate --out data/dataset.csv --n 5000 --seed 42

train:
	$(POETRY) run mlp train --data data/dataset.csv --model-dir artifacts/model --register

api:
	$(POETRY) run uvicorn ml_platform.serving.api:app --reload

batch:
	$(POETRY) run mlp batch --input data/sample_requests.jsonl --out outputs/preds.jsonl --model-version latest
