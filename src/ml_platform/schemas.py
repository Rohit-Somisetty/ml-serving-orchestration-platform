from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProductRecord(BaseModel):
    title: str | None = Field(default=None, max_length=256)
    description: str | None = Field(default=None, max_length=1024)
    price: float | None = Field(default=None, ge=0)
    brand: str | None = Field(default=None, max_length=128)


class PredictionResponse(BaseModel):
    request_id: str
    category: str
    confidence: float
    model_version: str
    model_alias: str
    canary_used: bool = False
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    request_id: str
    error_type: str
    message: str
    stage: str


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    failures: list[ErrorResponse]


class ModelSummary(BaseModel):
    version: str
    trained_at: datetime | None
    metrics: dict[str, float]


class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    registry_available: bool
    latest_version: str | None


class DriftReport(BaseModel):
    generated_at: datetime
    price_psi: float
    text_length_delta: float
    alerts: list[str]


class MetricsSummary(BaseModel):
    total_requests: int
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    top_categories: list[dict[str, Any]]


class TrainingManifest(BaseModel):
    model_version: str
    created_at: datetime
    dataset_path: str
    dataset_hash: str
    hyperparams: dict[str, Any]
    metrics: dict[str, float]
    baseline_stats: dict[str, Any]


class BatchJobResult(BaseModel):
    output_path: str
    processed: int
    succeeded: int
    failed: int


class JobStatus(BaseModel):
    job_id: str
    dag_name: str
    status: str
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: float | None
    error: str | None


class RegistrySummary(BaseModel):
    versions: list[str]
    aliases: dict[str, str]
