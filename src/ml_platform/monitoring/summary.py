from __future__ import annotations

import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import TypeAlias, cast

from ml_platform.schemas import MetricsSummary

LogRecord: TypeAlias = dict[str, object]


def _read_log_tail(path: Path, limit: int) -> list[LogRecord]:
    if not path.exists() or limit <= 0:
        return []
    window: deque[LogRecord] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    window.append(cast(LogRecord, data))
            except json.JSONDecodeError:
                continue
    return list(window)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * percentile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _as_float(value: object | None, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_int(value: object | None, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def compute_metrics_summary(path: Path, limit: int = 1000) -> MetricsSummary:
    entries = _read_log_tail(path, limit)
    request_events = [e for e in entries if e.get("event") == "request_complete"]
    latencies = [_as_float(e.get("duration_ms")) for e in request_events]
    errors = [e for e in request_events if _as_int(e.get("status")) >= 400]
    category_events = [
        e
        for e in entries
        if e.get("event") in {"predict", "predict_batch_item"} and e.get("category")
    ]
    counter = Counter(
        str(e["category"])
        for e in category_events
        if isinstance(e.get("category"), str)
    )
    top_categories = [
        {"category": name, "count": count}
        for name, count in counter.most_common(5)
    ]
    total_requests = len(request_events)
    error_rate = (len(errors) / total_requests) if total_requests else 0.0
    return MetricsSummary(
        total_requests=total_requests,
        error_rate=round(error_rate, 4),
        p50_latency_ms=round(_percentile(latencies, 0.5), 2),
        p95_latency_ms=round(_percentile(latencies, 0.95), 2),
        top_categories=top_categories,
    )


def recent_prediction_payloads(path: Path, limit: int) -> list[LogRecord]:
    entries = _read_log_tail(path, limit * 3)
    payloads: list[LogRecord] = []
    for event in reversed(entries):
        payload = event.get("input")
        if isinstance(payload, dict):
            payloads.append(cast(LogRecord, payload))
        if len(payloads) >= limit:
            break
    payloads.reverse()
    return payloads
