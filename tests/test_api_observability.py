from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from ml_platform.config import get_settings


def test_metrics_summary_endpoint(api_client: TestClient) -> None:
    payload = {"title": "Sofa", "description": "Comfortable", "price": 999, "brand": "FurniCo"}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    metrics = api_client.get("/metrics/summary").json()
    required_keys = {
        "total_requests",
        "error_rate",
        "p50_latency_ms",
        "p95_latency_ms",
        "top_categories",
    }
    assert required_keys.issubset(metrics)


def test_batch_size_guard(
    monkeypatch: pytest.MonkeyPatch,
    training_result: dict[str, object],
) -> None:
    import ml_platform.serving.api as api_module

    monkeypatch.setenv("MODEL_ALIAS", "latest")
    monkeypatch.setenv("CANARY_ALIAS", "latest")
    monkeypatch.setenv("CANARY_PERCENT", "0")
    monkeypatch.setenv("MLP_MAX_BATCH", "1")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    api_module = importlib.reload(api_module)
    client = TestClient(api_module.app)
    try:
        records = [
            {"title": "Chair", "description": "", "price": 200, "brand": "Comfy"},
            {"title": "Table", "description": "", "price": 400, "brand": "Oak"},
        ]
        resp = client.post("/predict/batch", json=records)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["error_type"] == "batch_limit"
    finally:
        client.close()
