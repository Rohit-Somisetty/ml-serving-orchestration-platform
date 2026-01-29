from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_predict_endpoint(api_client: TestClient) -> None:
    payload = {
        "title": "Modern desk",
        "description": "Compact workstation",
        "price": 450,
        "brand": "UrbanNest",
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"]
    assert data["confidence"] > 0
    assert "model_version" in data
    assert "model_alias" in data
    assert data["canary_used"] in {True, False}
