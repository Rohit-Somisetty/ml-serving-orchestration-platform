# Kubernetes Runbook (kind demo)

This guide provisions the FastAPI serving tier on a local [kind](https://kind.sigs.k8s.io/) cluster and runs the same guardrailed prediction API that exists locally. Prefer an automated smoke test? Run `scripts/smoke_kind.sh` which follows the same steps end-to-end.

## Prerequisites
- Docker Desktop or another container runtime running locally.
- `kind` >= 0.20 and `kubectl` >= 1.28 available on your `PATH`.
- This repository cloned locally (commands below assume repo root).

## 1. Create a kind cluster
```bash
kind create cluster --name mlp-demo
```

## 2. Build and load the container image
```bash
# Build the image from the provided Dockerfile
docker build -t mlp:local .

# Load it into the kind cluster (avoids pushing to a remote registry)
kind load docker-image mlp:local --name mlp-demo
```

> The image already contains rule-based seed artifacts under `/seed_artifacts`. The deployment initContainer copies them into `/app/artifacts` so the API can load a default `stable` model without hitting object storage.

## 3. Deploy the base stack
```bash
kubectl apply -k infra/k8s/base
```

This creates the `mlp` namespace, ConfigMap-driven env vars, the Deployment + Service, an emptyDir-backed volume for logs/artifacts, and an HPA stub.

## 4. (Optional) Enable canary routing
The application already supports request-level canary routing. To route ~10% of calls to the `candidate` alias:
```bash
kubectl apply -k infra/k8s/overlays/canary
```
Adjust the percentage by editing `infra/k8s/overlays/canary/configmap-patch.yaml` before re-applying, e.g. set `CANARY_PERCENT: "25"` for 25% traffic.

## 5. Port-forward and smoke test
```bash
kubectl port-forward -n mlp svc/mlp-serving 8000:8000
```
In a separate terminal:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics/summary
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Modern sofa", "description": "Sectional couch", "price": 950, "brand": "FurniCo"}'
```
Responses should include model metadata plus any guardrail warnings (batch limits, truncation, etc.).

## 6. Inspect logs and jobs
```bash
kubectl logs -n mlp deploy/mlp-serving -f
kubectl get pods -n mlp
kubectl get events -n mlp
```
Outputs and job history live on the pod-local emptyDirs under `/app/outputs`, so tail the logs to observe inference traces.

## 7. Clean up
```bash
kubectl delete namespace mlp
kind delete cluster --name mlp-demo
```

## Troubleshooting
- **Pod CrashLoopBackOff**: run `kubectl logs -n mlp pod/<name> -c seed-artifacts` to confirm the init container copied `/seed_artifacts` correctly. Ensure the Docker image was rebuilt after updating artifacts.
- **Model not found**: verify the ConfigMap still points `MODEL_ALIAS=stable` and that the registry contains alias metadata in `/app/artifacts/registry/aliases/stable.json` inside the pod.
- **Hitting observability endpoints**: `kubectl port-forward` first, then curl `GET /jobs/latest` or `/monitoring/drift/latest` to confirm background jobs are writing logs.
