from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2]


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings derived from environment variables with sensible defaults."""

    base_dir: Path = Path(os.environ.get("MLP_BASE_DIR", _DEFAULT_BASE_DIR))
    data_dir: Path = base_dir / "data"
    artifacts_dir: Path = base_dir / "artifacts"
    model_dir: Path = artifacts_dir / "model"
    registry_dir: Path = artifacts_dir / "registry"
    outputs_dir: Path = base_dir / "outputs"
    logs_dir: Path = outputs_dir / "logs"
    monitoring_dir: Path = outputs_dir / "monitoring"
    jobs_dir: Path = outputs_dir / "jobs"
    batch_outputs_dir: Path = outputs_dir / "batch"
    logs_file: Path = logs_dir / "inference.jsonl"
    drift_report: Path = monitoring_dir / "drift_report.json"
    default_dataset: Path = data_dir / "dataset.csv"
    model_version: str | None = os.environ.get("MODEL_VERSION")
    model_alias: str | None = os.environ.get("MODEL_ALIAS")
    canary_alias: str | None = os.environ.get("CANARY_ALIAS")
    canary_percent: int = int(os.environ.get("CANARY_PERCENT", "0"))
    max_batch_size: int = int(os.environ.get("MLP_MAX_BATCH", "100"))
    max_text_chars: int = int(os.environ.get("MLP_MAX_TEXT_CHARS", "10000"))
    soft_timeout_ms: int = int(os.environ.get("MLP_SOFT_TIMEOUT_MS", "800"))
    recent_request_window: int = int(os.environ.get("MLP_RECENT_REQUEST_WINDOW", "200"))

    def __post_init__(self) -> None:
        self._refresh_paths()
        self.canary_percent = max(0, min(100, self.canary_percent))
        self.max_batch_size = max(1, self.max_batch_size)
        self.max_text_chars = max(500, self.max_text_chars)
        self.soft_timeout_ms = max(50, self.soft_timeout_ms)
        self.recent_request_window = max(10, self.recent_request_window)

    def resolve_registry_version(self) -> str:
        return self.model_version or "latest"

    def _refresh_paths(self) -> None:
        self.data_dir = self.base_dir / "data"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.model_dir = self.artifacts_dir / "model"
        self.registry_dir = self.artifacts_dir / "registry"
        self.outputs_dir = self.base_dir / "outputs"
        self.logs_dir = self.outputs_dir / "logs"
        self.monitoring_dir = self.outputs_dir / "monitoring"
        self.jobs_dir = self.outputs_dir / "jobs"
        self.batch_outputs_dir = self.outputs_dir / "batch"
        self.logs_file = self.logs_dir / "inference.jsonl"
        self.drift_report = self.monitoring_dir / "drift_report.json"
        self.default_dataset = self.data_dir / "dataset.csv"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    base_dir = Path(os.environ.get("MLP_BASE_DIR", _DEFAULT_BASE_DIR))
    settings = Settings(
        base_dir=base_dir,
        model_version=os.environ.get("MODEL_VERSION"),
        model_alias=os.environ.get("MODEL_ALIAS"),
        canary_alias=os.environ.get("CANARY_ALIAS"),
        canary_percent=_env_int("CANARY_PERCENT", 0),
        max_batch_size=_env_int("MLP_MAX_BATCH", 100),
        max_text_chars=_env_int("MLP_MAX_TEXT_CHARS", 10000),
        soft_timeout_ms=_env_int("MLP_SOFT_TIMEOUT_MS", 800),
        recent_request_window=_env_int("MLP_RECENT_REQUEST_WINDOW", 200),
    )
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.monitoring_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.batch_outputs_dir.mkdir(parents=True, exist_ok=True)
    settings.registry_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    return settings
