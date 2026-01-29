# Demo Runbook

This runbook shows how to exercise the entire platform in a few minutes, from local Typer automation to Docker and kind-based smoke tests. Pick the workflow that matches your environment.

## 1. Single-command local demo
```
poetry install
poetry run mlp demo --n 5000 --seed 42
cat outputs/demo/demo_summary.json | jq
```
What the command does:
1. Generates `data/dataset_demo.csv` with the requested row count and seed (skip via `--no-train`).
2. Trains + registers a model under `artifacts/model_demo`, promotes `candidate=latest`, and ensures `stable` is set (or left on the previous stable version).
3. Starts the FastAPI service via `uvicorn` with `MODEL_ALIAS=stable`, `CANARY_ALIAS=candidate`, and `CANARY_PERCENT` from the flag.
4. Issues `POST /predict`, `POST /predict/batch` (with one malformed record to produce a structured failure), and `GET /metrics/summary` while capturing requests/responses to `outputs/demo/demo_{requests,responses}.jsonl`.
5. Runs the `nightly` (drift) and `daily_batch` DAGs so `outputs/monitoring/drift_report.json` and batch outputs refresh.
6. Copies monitoring, metrics, and job history into `outputs/demo/` and emits a concise `demo_summary.json` with dataset rows, alias versions, traffic split, metrics, drift path, and recent jobs.

Key flags:
- `--mode docker|kind` → prints the relevant smoke-test instructions instead of running locally.
- `--no-train` → skip data generation/training and rely on the bundled seed registry.
- `--canary-percent 25` → change how much traffic routes to the `candidate` alias while the demo is running.

Artifacts written under `outputs/demo/`:
- `demo_summary.json` – overall status, versions, metrics excerpt, job snapshot.
- `demo_requests.jsonl` / `demo_responses.jsonl` – raw payloads + responses (including batch failures).
- `metrics_summary.json` – copy of `/metrics/summary`.
- `drift_report.json` – latest drift computation (if available).
- `jobs_snapshot.json` – most recent orchestration jobs (e.g., nightly + daily_batch) plus `api.log` streamed from uvicorn.

## 2. Docker smoke test
Use the helper script (requires Docker + curl):
```
./scripts/smoke_docker.sh
```
What it does:
- Builds the Docker image (tag `mlp:local` by default).
- Runs the container publishing port 8000 with `MODEL_ALIAS=stable` and optional `CANARY_PERCENT` env.
- Polls `/health`, then curls `/metrics/summary` and a sample `/predict` request before stopping the container.
Customize via env vars:
- `IMAGE`, `CONTAINER` – override names.
- `CANARY_PERCENT` – adjust traffic split exposed by the container.

Prefer manual control? Run the same commands inline (documented in comments inside the script) and review `docker logs mlp-demo` to inspect inference traces.

## 3. kind-based smoke test
For a local Kubernetes experience without a cloud cluster:
```
./scripts/smoke_kind.sh
```
The script assumes `kind`, `kubectl`, and Docker are installed. It will:
1. Create (or reuse) a cluster named `mlp-demo`.
2. Build the Docker image and `kind load docker-image` into the cluster.
3. Apply `infra/k8s/base` (namespace, ConfigMap, Deployment + emptyDir volumes, Service, HPA).
4. Wait for the `mlp-serving` deployment to become available.
5. Port-forward `svc/mlp-serving` on 8000 and run the same health/metrics/predict curls.
6. Clean up the port-forward session. (Cluster removal is optional; run `kind delete cluster --name mlp-demo` when you are done.)

To flip on canary routing inside the cluster, run:
```
kubectl apply -k infra/k8s/overlays/canary
```
Then re-run the script or manually curl `/predict` to verify the response includes `"canary_used": true` about 10% of the time.

## Related docs
- [docs/k8s_runbook.md](k8s_runbook.md) – deeper dive into the Kubernetes deployment, manual port-forwarding, and troubleshooting tips.
- [README.md](../README.md) – project overview and CLI reference (training, registry, orchestration, demo command).
```