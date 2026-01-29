#!/usr/bin/env bash
set -euo pipefail

KIND_CLUSTER="${KIND_CLUSTER:-mlp-demo}"
IMAGE="${IMAGE:-mlp:local}"
CANARY_PERCENT="${CANARY_PERCENT:-10}"
NAMESPACE="mlp"
PORT="${PORT:-8000}"
PORT_FORWARD_LOG="$(mktemp)"
PORT_FORWARD_PID=""

cleanup() {
  if [[ -n "${PORT_FORWARD_PID}" ]] && ps -p "${PORT_FORWARD_PID}" >/dev/null 2>&1; then
    kill "${PORT_FORWARD_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${PORT_FORWARD_LOG}"
}
trap cleanup EXIT

pretty() {
  if command -v jq >/dev/null 2>&1; then
    jq "$@"
  else
    cat
  fi
}

echo "[smoke-kind] Ensuring kind cluster ${KIND_CLUSTER} exists…"
if ! kind get clusters | grep -q "${KIND_CLUSTER}"; then
  kind create cluster --name "${KIND_CLUSTER}"
fi

echo "[smoke-kind] Building image ${IMAGE} and loading into kind…"
docker build -t "${IMAGE}" .
kind load docker-image "${IMAGE}" --name "${KIND_CLUSTER}"

echo "[smoke-kind] Applying base manifests…"
kubectl apply -k infra/k8s/base
kubectl wait --namespace "${NAMESPACE}" --for=condition=available deploy/mlp-serving --timeout=180s

echo "[smoke-kind] (Optional) Apply canary overlay for ${CANARY_PERCENT}% traffic"
echo "kubectl apply -k infra/k8s/overlays/canary"

kubectl port-forward -n "${NAMESPACE}" svc/mlp-serving "${PORT}:8000" >"${PORT_FORWARD_LOG}" 2>&1 &
PORT_FORWARD_PID=$!
sleep 5

curl -s "http://127.0.0.1:${PORT}/health" | pretty
curl -s "http://127.0.0.1:${PORT}/metrics/summary" | pretty '.top_categories[:2]'
curl -s -X POST "http://127.0.0.1:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d '{"title":"Modern sofa","description":"Sectional couch","price":950,"brand":"FurniCo"}' | pretty

echo "[smoke-kind] Port-forward logs at ${PORT_FORWARD_LOG}"
echo "[smoke-kind] Run 'kubectl delete namespace ${NAMESPACE}' or 'kind delete cluster --name ${KIND_CLUSTER}' to clean up when finished."
