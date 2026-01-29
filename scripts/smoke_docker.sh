#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-mlp:local}"
CONTAINER="${CONTAINER:-mlp-demo}"
PORT="${PORT:-8000}"
CANARY_PERCENT="${CANARY_PERCENT:-10}"

cleanup() {
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    docker stop "${CONTAINER}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

pretty() {
  if command -v jq >/dev/null 2>&1; then
    jq "$@"
  else
    cat
  fi
}

echo "[smoke-docker] Building image ${IMAGE}…"
docker build -t "${IMAGE}" .

echo "[smoke-docker] Starting container ${CONTAINER} on port ${PORT}…"
docker run \
  --rm \
  -d \
  --name "${CONTAINER}" \
  -p "${PORT}:8000" \
  -e MODEL_ALIAS=stable \
  -e CANARY_ALIAS=candidate \
  -e CANARY_PERCENT="${CANARY_PERCENT}" \
  "${IMAGE}"

for _ in {1..15}; do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 2
fi

echo "[smoke-docker] /health"
curl -s "http://127.0.0.1:${PORT}/health" | pretty

echo "[smoke-docker] /metrics/summary"
curl -s "http://127.0.0.1:${PORT}/metrics/summary" | pretty '.top_categories[:2]'

echo "[smoke-docker] /predict"
curl -s -X POST "http://127.0.0.1:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d '{"title":"Modern sofa","description":"Sectional couch","price":950,"brand":"FurniCo"}' | jq

echo "[smoke-docker] Logs (tail)"
docker logs "${CONTAINER}" --tail 20 || true
