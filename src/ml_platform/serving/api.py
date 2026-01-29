from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, NoReturn, cast

from fastapi import Depends, FastAPI, HTTPException, Request, status

from ml_platform.config import Settings, get_settings
from ml_platform.monitoring.drift import DriftMonitor
from ml_platform.monitoring.health import health_probe
from ml_platform.monitoring.logger import StructuredLogger
from ml_platform.monitoring.summary import compute_metrics_summary
from ml_platform.orchestration.state import JobRecord, JobStateStore
from ml_platform.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    JobStatus,
    MetricsSummary,
    ModelSummary,
    PredictionResponse,
    ProductRecord,
    RegistrySummary,
)
from ml_platform.serving.middleware import RequestContextMiddleware
from ml_platform.serving.router import ModelHandle, ModelRouter
from ml_platform.training import registry

settings = get_settings()
logger = StructuredLogger(settings.logs_file)
job_store = JobStateStore()
app = FastAPI(title="ML Serving Orchestration Platform", version="0.2.0")
app.add_middleware(RequestContextMiddleware, logger=logger)

_model_router: ModelRouter | None = None
_drift_monitor: DriftMonitor | None = None


class PredictionError(Exception):
    def __init__(
        self,
        detail: dict[str, object],
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
    ) -> None:
        super().__init__(detail.get("message"))
        self.detail = detail
        self.status_code = status_code


def _structured_error(
    request_id: str,
    error_type: str,
    message: str,
    stage: str,
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "error_type": error_type,
        "message": message,
        "stage": stage,
    }


def _ensure_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
    return request_id


def _coerce_str(value: object | None) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _coerce_float(value: object | None, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _job_status_from_record(record: JobRecord) -> JobStatus:
    queued_at = datetime.fromisoformat(record["queued_at"])
    return JobStatus(
        job_id=record["job_id"],
        dag_name=record["dag_name"],
        status=record["status"],
        queued_at=queued_at,
        started_at=_parse_datetime(record["started_at"]),
        finished_at=_parse_datetime(record["finished_at"]),
        duration_ms=record["duration_ms"],
        error=record["error"],
    )


def _sanitize_payload(
    record: dict[str, object],
    config: Settings,
    request_id: str,
) -> tuple[dict[str, object], list[str]]:
    payload = dict(record)
    warnings: list[str] = []
    title = _coerce_str(payload.get("title"))
    description = _coerce_str(payload.get("description"))
    if not title.strip() and not description.strip() and payload.get("price") is None:
        detail = _structured_error(
            request_id,
            "validation",
            "Record must include text or price",
            "validation",
        )
        raise PredictionError(detail, status.HTTP_422_UNPROCESSABLE_ENTITY)
    max_desc = max(0, config.max_text_chars - len(title))
    if len(description) > max_desc >= 0:
        payload["description"] = description[:max_desc] if max_desc > 0 else ""
        warnings.append("text_truncated")
    return payload, warnings


def _init_router() -> ModelRouter:
    global _model_router, _drift_monitor
    if _model_router is None:
        _model_router = ModelRouter(settings)
        baseline = _model_router.primary_handle.predictor.manifest.get("baseline_stats", {})
        _drift_monitor = DriftMonitor(baseline, settings.drift_report)
    return _model_router


def get_model_router() -> ModelRouter:
    try:
        return _init_router()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.on_event("startup")
async def startup_event() -> None:
    try:
        _init_router()
    except FileNotFoundError:
        pass


def _log_success_event(
    event: str,
    response: PredictionResponse,
    payload: dict[str, object],
    duration_ms: float,
) -> None:
    logger.log_event(
        {
            "event": event,
            **response.model_dump(),
            "duration_ms": round(duration_ms, 2),
            "input": payload,
        },
    )


def _run_prediction(
    router: ModelRouter,
    payload: dict[str, object],
    request_id: str,
    warnings: list[str],
    event_name: str = "predict",
    request: Request | None = None,
) -> PredictionResponse:
    handle, canary_used = router.choose_handle(request_id)
    start = time.perf_counter()
    result = handle.predictor.predict_one(payload)
    duration_ms = (time.perf_counter() - start) * 1000
    if duration_ms > settings.soft_timeout_ms:
        detail = _structured_error(
            request_id,
            "timeout",
            f"Inference exceeded {settings.soft_timeout_ms}ms",
            "inference",
        )
        logger.log_event(
            {
                "event": "predict_timeout",
                **detail,
                "duration_ms": round(duration_ms, 2),
            },
        )
        raise PredictionError(detail, status.HTTP_503_SERVICE_UNAVAILABLE)
    category = _coerce_str(result.get("category"))
    confidence = _coerce_float(result.get("confidence"))
    response = PredictionResponse(
        request_id=request_id,
        category=category,
        confidence=confidence,
        model_version=handle.version,
        model_alias=handle.alias,
        canary_used=canary_used,
        warnings=list(warnings),
    )
    if request is not None and canary_used:
        request.state.canary_used = True
    _log_success_event(event_name, response, payload, duration_ms)
    if _drift_monitor:
        _drift_monitor.evaluate([payload])
    return response


@app.get("/health")
def health() -> dict[str, object]:
    base = health_probe()
    router = _model_router
    base["api_model_loaded"] = router is not None
    if router:
        base["model_alias"] = router.primary_handle.alias
        base["model_version"] = router.primary_handle.version
        base["canary"] = router.canary_handle.alias if router.canary_handle else None
    return base


@app.get("/model", response_model=ModelSummary)
def model_details(router: ModelRouter = Depends(get_model_router)) -> ModelSummary:  # noqa: B008
    handle: ModelHandle = router.primary_handle
    manifest = handle.predictor.manifest
    created_at = manifest.get("created_at")
    trained_at = datetime.fromisoformat(created_at) if created_at else None
    return ModelSummary(
        version=handle.version,
        trained_at=trained_at,
        metrics=handle.predictor.metrics,
    )


@app.get("/registry", response_model=RegistrySummary)
def registry_summary() -> RegistrySummary:
    versions, aliases = registry.available_versions_with_aliases(settings.registry_dir)
    return RegistrySummary(versions=versions, aliases=aliases)


@app.get("/metrics/summary", response_model=MetricsSummary)
def metrics_summary() -> MetricsSummary:
    return compute_metrics_summary(settings.logs_file)


@app.get("/monitoring/drift/latest")
def drift_latest() -> dict[str, object]:
    if not settings.drift_report.exists():
        raise HTTPException(status_code=404, detail="Drift report not found")
    data = json.loads(settings.drift_report.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Invalid drift report format")
    return cast(dict[str, object], data)


@app.get("/jobs/latest", response_model=list[JobStatus])
def jobs_latest(limit: int = 5) -> list[JobStatus]:
    jobs = job_store.list_jobs(limit=limit)
    return [_job_status_from_record(job) for job in jobs]


def _handle_prediction_error(error: PredictionError) -> NoReturn:
    logger.log_event({"event": "predict_error", **error.detail})
    raise HTTPException(status_code=error.status_code, detail=error.detail)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: Request,
    record: ProductRecord,
    router: ModelRouter = Depends(get_model_router),  # noqa: B008
) -> PredictionResponse:
    request_id = _ensure_request_id(request)
    try:
        payload = cast(dict[str, object], record.model_dump(exclude_none=True))
        sanitized, warnings = _sanitize_payload(payload, settings, request_id)
        return _run_prediction(router, sanitized, request_id, warnings, request=request)
    except PredictionError as error:
        _handle_prediction_error(error)


def _validate_batch_size(count: int, request_id: str) -> None:
    if count > settings.max_batch_size:
        detail = _structured_error(
            request_id,
            "batch_limit",
            f"Batch size {count} exceeds limit {settings.max_batch_size}",
            "validation",
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: Request,
    records: list[ProductRecord],
    router: ModelRouter = Depends(get_model_router),  # noqa: B008
) -> BatchPredictionResponse:
    base_request_id = _ensure_request_id(request)
    _validate_batch_size(len(records), base_request_id)
    successes: list[PredictionResponse] = []
    failures: list[ErrorResponse] = []
    for idx, record in enumerate(records):
        item_request_id = f"{base_request_id}:{idx}"
        payload = cast(dict[str, object], record.model_dump(exclude_none=True))
        try:
            sanitized, warnings = _sanitize_payload(payload, settings, item_request_id)
            response = _run_prediction(
                router,
                sanitized,
                item_request_id,
                warnings,
                event_name="predict_batch_item",
                request=request,
            )
            successes.append(response)
        except PredictionError as error:
            detail = error.detail
            failures.append(ErrorResponse(**cast(dict[str, Any], detail)))
            logger.log_event({"event": "predict_batch_error", **detail})
    return BatchPredictionResponse(results=successes, failures=failures)
