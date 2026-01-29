from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

from ml_platform.batch.runner import run_batch
from ml_platform.config import Settings, get_settings
from ml_platform.monitoring.drift import DriftMonitor
from ml_platform.monitoring.summary import recent_prediction_payloads
from ml_platform.orchestration.dag import DAG, Task
from ml_platform.serving.predictor import ModelPredictor
from ml_platform.training import registry
from ml_platform.utils.io import read_jsonl

JobContext = dict[str, object]


def _get_settings(context: JobContext) -> Settings:
    existing = context.get("settings")
    if isinstance(existing, Settings):
        return existing
    settings = get_settings()
    context["settings"] = settings
    return settings


def _load_model_task(context: JobContext) -> None:
    settings = _get_settings(context)
    registry_dir = settings.registry_dir
    alias = registry.preferred_serving_alias(registry_dir)
    version = registry.resolve_version_reference(registry_dir, alias)
    model_path = registry.resolve_model_path(registry_dir, version)
    context["predictor"] = ModelPredictor(model_path)


def _collect_recent_requests_task(context: JobContext) -> None:
    settings = _get_settings(context)
    requests = recent_prediction_payloads(settings.logs_file, settings.recent_request_window)
    if not requests:
        fallback_path = settings.data_dir / "sample_requests.jsonl"
        requests = list(read_jsonl(fallback_path))
    context["recent_requests"] = requests


def _compute_drift_task(context: JobContext) -> None:
    settings = _get_settings(context)
    predictor = cast(ModelPredictor, context["predictor"])
    recent_requests = cast(list[dict[str, object]], context.get("recent_requests", []))
    if not recent_requests:
        raise ValueError("No recent requests available for drift computation")
    monitor = DriftMonitor(predictor.manifest.get("baseline_stats", {}), settings.drift_report)
    monitor.evaluate(recent_requests)


def _prepare_batch_output_task(context: JobContext) -> None:
    settings = _get_settings(context)
    date_stamp = datetime.utcnow().strftime("%Y%m%d")
    out_dir = settings.batch_outputs_dir / date_stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    context["batch_output"] = out_dir / "preds.jsonl"


def _run_daily_batch_task(context: JobContext) -> None:
    settings = _get_settings(context)
    output_path = cast(Path, context["batch_output"])
    input_path = settings.data_dir / "sample_requests.jsonl"
    run_batch(input_path=input_path, output_path=output_path, version="latest")


def build_registered_dags() -> dict[str, DAG]:
    nightly = DAG(
        name="nightly",
        tasks=[
            Task("load_model", _load_model_task),
            Task("collect_recent_requests", _collect_recent_requests_task, deps=["load_model"]),
            Task("compute_drift", _compute_drift_task, deps=["collect_recent_requests"]),
        ],
    )
    daily_batch = DAG(
        name="daily_batch",
        tasks=[
            Task("prepare_batch_output", _prepare_batch_output_task),
            Task("run_daily_batch", _run_daily_batch_task, deps=["prepare_batch_output"]),
        ],
    )
    return {"nightly": nightly, "daily_batch": daily_batch}
