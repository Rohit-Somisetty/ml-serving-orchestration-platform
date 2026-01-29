#!/usr/bin/env bash
set -euo pipefail

poetry install
poetry run mlp data-generate --out data/dataset.csv --n 2000 --seed 42
poetry run mlp train --data data/dataset.csv --model-dir artifacts/model --register
poetry run uvicorn ml_platform.serving.api:app --reload
