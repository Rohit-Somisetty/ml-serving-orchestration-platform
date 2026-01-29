from __future__ import annotations

from pathlib import Path


def test_training_outputs(training_result: dict) -> None:
    model_path = Path(training_result["model_path"])
    metrics_path = Path(training_result["metrics_path"])
    manifest_path = Path(training_result["manifest_path"])
    assert model_path.exists()
    assert metrics_path.exists()
    assert manifest_path.exists()
    assert training_result["version"] is not None
