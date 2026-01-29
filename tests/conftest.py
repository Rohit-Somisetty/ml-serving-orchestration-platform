from __future__ import annotations

import importlib
import json
import os
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_platform.config import get_settings
from ml_platform.data.synth import generate_dataset, save_dataset
from ml_platform.serving.predictor import ModelPredictor
from ml_platform.training.train import train_model


@pytest.fixture(scope="session")
def base_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("mlp-tests")
    os.environ["MLP_BASE_DIR"] = str(base)
    get_settings.cache_clear()  # type: ignore[attr-defined]
    return base


@pytest.fixture(scope="session")
def dataset_path(base_dir: Path) -> Path:
    path = base_dir / "data" / "dataset.csv"
    df = generate_dataset(n_samples=600, seed=7)
    save_dataset(df, path)
    sample_path = path.parent / "sample_requests.jsonl"
    sample_records = [
        {"title": "Modern sofa", "description": "Sectional", "price": 950, "brand": "FurniCo"},
        {"title": "Desk", "description": "Writing desk", "price": 320, "brand": "UrbanNest"},
    ]
    serialized = "\n".join(json.dumps(record) for record in sample_records) + "\n"
    sample_path.write_text(serialized, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def training_result(base_dir: Path, dataset_path: Path) -> dict[str, object]:
    model_dir = base_dir / "artifacts" / "model"
    return train_model(data_path=dataset_path, model_dir=model_dir, register=True)


@pytest.fixture(scope="session")
def model_dir(training_result: dict[str, object]) -> Path:
    return Path(training_result["model_path"]).parent


@pytest.fixture(scope="session")
def predictor(model_dir: Path) -> ModelPredictor:
    return ModelPredictor(model_dir)


@pytest.fixture()
def api_client(training_result: dict[str, object]) -> Generator[TestClient, None, None]:
    import ml_platform.serving.api as api_module

    prev_model_alias = os.environ.get("MODEL_ALIAS")
    prev_canary_alias = os.environ.get("CANARY_ALIAS")
    prev_canary_percent = os.environ.get("CANARY_PERCENT")
    os.environ["MODEL_ALIAS"] = "latest"
    os.environ["CANARY_ALIAS"] = "latest"
    os.environ["CANARY_PERCENT"] = "0"
    get_settings.cache_clear()  # type: ignore[attr-defined]
    api_module = importlib.reload(api_module)
    try:
        with TestClient(api_module.app) as client:
            yield client
    finally:
        if prev_model_alias is None:
            os.environ.pop("MODEL_ALIAS", None)
        else:
            os.environ["MODEL_ALIAS"] = prev_model_alias
        if prev_canary_alias is None:
            os.environ.pop("CANARY_ALIAS", None)
        else:
            os.environ["CANARY_ALIAS"] = prev_canary_alias
        if prev_canary_percent is None:
            os.environ.pop("CANARY_PERCENT", None)
        else:
            os.environ["CANARY_PERCENT"] = prev_canary_percent
        get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture()
def _base_dir(base_dir: Path) -> Path:
    """Alias fixture to satisfy legacy test names."""

    return base_dir
