from __future__ import annotations

import uuid
from pathlib import Path
from typing import cast

from ml_platform.config import get_settings
from ml_platform.monitoring.drift import DriftMonitor
from ml_platform.monitoring.logger import StructuredLogger
from ml_platform.serving.predictor import ModelPredictor
from ml_platform.training import registry
from ml_platform.utils.io import read_jsonl, write_jsonl


def _resolve_model(version: str | None) -> tuple[Path, str, str]:
    settings = get_settings()
    if version == "local":
        return settings.model_dir, "local", "local"
    reference = None if version in (None, "latest") else version
    if reference is None:
        reference = registry.preferred_serving_alias(settings.registry_dir)
    model_path = registry.resolve_model_path(settings.registry_dir, reference)
    resolved_version = registry.resolve_version_reference(settings.registry_dir, reference)
    alias = reference if reference in registry.list_aliases(settings.registry_dir) else reference
    return model_path, alias, resolved_version


def run_batch(
    input_path: Path,
    output_path: Path,
    version: str | None = "latest",
) -> dict[str, object]:
    model_dir, alias, resolved_version = _resolve_model(version)
    predictor = ModelPredictor(model_dir)
    settings = get_settings()
    logger = StructuredLogger(settings.logs_file)
    drift_monitor = DriftMonitor(
        predictor.manifest.get("baseline_stats", {}),
        settings.drift_report,
    )

    results: list[dict[str, object]] = []
    successes = 0
    failures = 0
    processed_records: list[dict[str, object]] = []
    for record in read_jsonl(input_path):
        record_obj = cast(dict[str, object], record)
        processed_records.append(record_obj)
        request_id = str(uuid.uuid4())
        try:
            pred = predictor.predict_one(record_obj)
            output_payload: dict[str, object] = {
                **pred,
                "model_alias": alias,
                "model_version": resolved_version,
            }
            success_payload: dict[str, object] = {
                "request_id": request_id,
                "status": "ok",
                "input": record_obj,
                "output": output_payload,
            }
            results.append(success_payload)
            logger.log_event(
                {
                    "event": "batch_predict",
                    "request_id": request_id,
                    "model_alias": alias,
                    "model_version": resolved_version,
                    **pred,
                },
            )
            successes += 1
        except Exception as exc:  # pragma: no cover - defensive
            failures += 1
            error_payload: dict[str, object] = {
                "request_id": request_id,
                "status": "error",
                "input": record_obj,
                "error": str(exc),
                "model_alias": alias,
                "model_version": resolved_version,
            }
            results.append(error_payload)
            logger.log_event({"event": "batch_predict_error", **error_payload})
    write_jsonl(results, output_path)
    if processed_records:
        drift_monitor.evaluate(processed_records)
    return {
        "output_path": str(output_path),
        "processed": successes + failures,
        "succeeded": successes,
        "failed": failures,
    }
