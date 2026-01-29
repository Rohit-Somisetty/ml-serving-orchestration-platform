# ML Serving Orchestration Platform

A lightweight, local-first ML platform that trains a synthetic product-category classifier, serves it through FastAPI, orchestrates batch + drift jobs, and emits structured monitoring signals.

## Features
- Deterministic synthetic data generator (`mlp data-generate`) for self-contained datasets.
- Scikit-learn pipeline with text + price + brand features, registry-backed artifact management, and Typer CLI.
- FastAPI service with production-grade guardrails (request IDs, batch limits, soft timeouts, truncation warnings) plus `GET /health`, `/model`, `/registry`, `/metrics/summary`, `/monitoring/drift/latest`, `/jobs/latest`.
- Batch inference runner + orchestration DAGs (nightly drift, daily batch) with job-state tracking under `outputs/jobs/`.
- Model rollout controls: aliases (`latest`, `stable`, `candidate`), promotions, rollbacks, and canary routing (via `CANARY_ALIAS` / `CANARY_PERCENT`).
- Structured JSONL logging with optional DuckDB sink, drift monitoring (price PSI + text-length delta), and summarizer reporting latency/error percentiles.
- Tooling: Poetry, Ruff, MyPy, Pytest, GitHub Actions CI, Dockerfile, docker-compose, Typer CLI.
- `mlp demo` command captures a full local workflow (train → serve → canary → batch → drift) with artifacts under `outputs/demo/`.

## Quickstart
```bash
# 1. Install
poetry install

# 2. Generate data
poetry run mlp data-generate --out data/dataset.csv --n 5000 --seed 42

# 3. Train + register
poetry run mlp train --data data/dataset.csv --model-dir artifacts/model --register

# 4. Promote latest to the stable alias (serving defaults to `stable` when available)
poetry run mlp registry promote --from latest --to stable

# 5. Start API (or `make api`)
poetry run uvicorn ml_platform.serving.api:app --reload

# 6. Call predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Modern sofa","description":"Sectional couch","price":950,"brand":"FurniCo"}'

# 7. Run batch job
poetry run mlp batch --input data/sample_requests.jsonl --out outputs/preds.jsonl --model-version latest

# 8. (Optional) Run a scheduled DAG locally & inspect job history
poetry run mlp schedule run nightly
poetry run mlp jobs list

# 9. Inspect outputs & monitoring signals
cat outputs/logs/inference.jsonl
cat outputs/monitoring/drift_report.json
curl http://localhost:8000/metrics/summary
```

## Windows Quickstart (Poetry)
```
py -3.11 --version
cd "E:/Github projects/ml-serving-orchestration-platform"
poetry env remove --all
py -3.11 -m pip install --upgrade pip
$pyExe = py -3.11 -c "import sys; print(sys.executable)"
poetry env use $pyExe
poetry install
poetry run python scripts/doctor.py
```
See [docs/windows_poetry_troubleshooting.md](docs/windows_poetry_troubleshooting.md) if you hit stdlib import errors or need to recreate the venv.

## One-command Demo
```
poetry install
poetry run mlp demo --n 5000 --seed 42 --canary-percent 10
cat outputs/demo/demo_summary.json
```
See [docs/demo_runbook.md](docs/demo_runbook.md) for details plus Docker/kind smoke scripts.

## CLI Overview
| Command | Description |
| --- | --- |
| `mlp data-generate --out data/dataset.csv --n 5000 --seed 42` | Create synthetic dataset |
| `mlp train --data data/dataset.csv --model-dir artifacts/model --register` | Train model, log metrics, register artifacts |
| `mlp batch --input data/sample_requests.jsonl --out outputs/preds.jsonl --model-version latest` | Batch inference with structured outputs |
| `mlp registry list` | Show versions + alias targets |
| `mlp registry promote --from latest --to stable` | Promote latest trained model to `stable` |
| `mlp registry rollback --alias stable --steps 1` | Roll back an alias to a previous version |
| `mlp schedule list` / `mlp schedule run nightly` | Inspect + execute local DAGs (nightly drift, daily batch) |
| `mlp jobs list` / `mlp jobs show <id>` | Inspect orchestration job history |
| `mlp demo --n 5000 --seed 42 --mode local` | One-command E2E demo (see docs/demo_runbook.md) |
| `mlp version` | Print package version and current git SHA |

## Project Layout
```
src/ml_platform/
  config.py              # Paths & settings
  schemas.py             # Pydantic request/response contracts
  data/                  # Synthetic generator + splits
  training/              # Train loop, metrics, registry
  serving/               # FastAPI app, predictor, middleware
  orchestration/         # Minimal DAG engine, scheduler, DAG definitions
  batch/                 # Batch inference runner
  monitoring/            # Logger, drift, health probes
  utils/                 # IO helpers & hashing
```
Other key files: `Makefile`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `scripts/bootstrap_demo.sh`, `scripts/smoke_docker.sh`, `scripts/smoke_kind.sh`, `data/sample_requests.jsonl`, `docs/demo_runbook.md`, `docs/k8s_runbook.md`, `infra/` (Kubernetes manifests + overlays).

## Docker
```
docker build -t mlp-api .
docker run -p 8000:8000 mlp-api
# or docker compose
docker compose up --build
```

The image bundles `seed_artifacts/` so the registry seeds itself automatically on first boot, and the container startup scripts create `/app/artifacts` and `/app/outputs` before serving traffic—no local artifacts or outputs are required in the build context.

## CI/CD
GitHub Actions workflow (`.github/workflows/ci.yml`) runs Ruff, MyPy, and Pytest on pushes/PRs to `main`/`master`.

## Orchestration & Jobs
- DAGs live under `ml_platform/orchestration/dags.py`:
  - `nightly`: load latest/stable model, sample recent inference payloads, recompute drift, update `outputs/monitoring/drift_report.json`.
  - `daily_batch`: run batch inference on `data/sample_requests.jsonl` and drop outputs under `outputs/batch/<date>/preds.jsonl`.
- Run locally via Typer: `poetry run mlp schedule list`, `poetry run mlp schedule run nightly`.
- Job state transitions (`queued -> running -> success/failed`) are recorded in `outputs/jobs/<job_id>.json` and surfaced through `mlp jobs list` / `mlp jobs show <job_id>` or the `GET /jobs/latest` API endpoint.

## Model Rollout & Canary Controls
- Registry aliases (`latest`, `stable`, `candidate`, etc.) are stored locally. Key commands:
  - `poetry run mlp registry list`
  - `poetry run mlp registry promote --from latest --to stable`
  - `poetry run mlp registry set-alias --alias candidate --version v20260101010101`
  - `poetry run mlp registry rollback --alias stable --steps 1`
- Serving defaults to the `stable` alias if it exists, otherwise `latest`. Override behavior with env vars:
  - `MODEL_ALIAS=stable` (pin to alias)
  - `MODEL_VERSION=v20260101010101` (pin to exact version)
  - `CANARY_ALIAS=candidate CANARY_PERCENT=10` (route ~10% of traffic to the `candidate` alias and surface `canary_used` in responses/logs)

## Observability & Guardrails
- Key API endpoints:
  - `GET /metrics/summary` → latency percentiles, error rate, top predicted categories computed from `outputs/logs/inference.jsonl`.
  - `GET /monitoring/drift/latest` → most recent drift report emitted by scheduled jobs or online requests.
  - `GET /jobs/latest` → last N job run records.
  - `GET /registry` → versions + alias mappings.
- Structured logging: every request/response + middleware timing is appended to `outputs/logs/inference.jsonl` (and optional DuckDB sink).
- Guardrails:
  - `MLP_MAX_BATCH` (default 100) enforces batch payload size limits (`422` with structured error if exceeded).
  - `MLP_MAX_TEXT_CHARS` (default 10k) truncates long descriptions with a `warnings` flag in predictions.
  - `MLP_SOFT_TIMEOUT_MS` (default 800ms) triggers a structured timeout error if inference exceeds the threshold.
  - Shared error schema `{request_id, error_type, message, stage}` returned for all API guardrails.

## Roadmap
- [ ] Expand monitoring to include latency percentiles and failure counters persisted in DuckDB.
- [ ] Add feature-store abstraction + async batch orchestration.
- [ ] Provide Kubernetes manifests within `infra/` for deployment.
- [ ] Extend registry metadata API + drift dashboards.

## Kubernetes Deployment (kind)
Follow the local cluster guide in [docs/k8s_runbook.md](docs/k8s_runbook.md) to build/load the Docker image into kind, apply the base manifests, flip on the canary overlay, and smoke-test `/health`, `/predict`, and `/metrics/summary` via `kubectl port-forward`.
The provided manifests mount `emptyDir` volumes for `/app/artifacts` + `/app/outputs` and use an initContainer to copy bundled seed models into place so no external registry is required.
Adjust env vars (batch caps, soft timeouts, canary percent) via the `mlp-serving-config` ConfigMap before applying overlays.
## License
MIT
