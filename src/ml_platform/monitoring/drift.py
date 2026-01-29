from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from ml_platform.utils.io import dump_json


def _coerce_float(value: object | None, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_str(value: object | None) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _psi(expected: Sequence[float], actual: Sequence[float]) -> float:
    eps = 1e-9
    return float(
        sum(
            (a - e) * np.log((a + eps) / (e + eps))
            for e, a in zip(expected, actual, strict=False)
        ),
    )


class DriftMonitor:
    def __init__(self, baseline: dict[str, object], report_path: Path) -> None:
        self.baseline = baseline
        self.report_path = report_path
        raw_bins = baseline.get("price_bins")
        bins_source = (
            raw_bins
            if isinstance(raw_bins, list | tuple)
            else [0, 200, 400, 600, 800, 1000, 2000]
        )
        self.price_bins = [float(bin_edge) for bin_edge in bins_source]
        raw_hist = baseline.get("price_hist")
        hist_source = (
            raw_hist
            if isinstance(raw_hist, list | tuple)
            else [1.0] + [0.0] * (len(self.price_bins) - 1)
        )
        self.baseline_hist = [float(val) for val in hist_source]
        self.text_mean = _coerce_float(baseline.get("text_length_mean"), 10.0)

    def evaluate(self, records: Sequence[dict[str, object]]) -> dict[str, object]:
        if not records:
            raise ValueError("No records provided for drift evaluation")
        prices = [_coerce_float(r.get("price")) for r in records]
        text_lengths = [
            len((_coerce_str(r.get("title")) + " " + _coerce_str(r.get("description"))).split())
            for r in records
        ]
        hist, _ = np.histogram(prices, bins=self.price_bins)
        hist = hist / hist.sum() if hist.sum() else np.zeros_like(hist, dtype=float)
        psi_val = _psi(self.baseline_hist, hist.tolist())
        current_mean = float(np.mean(text_lengths) if text_lengths else 0)
        delta = current_mean - self.text_mean
        alerts = []
        if psi_val > 0.2:
            alerts.append(f"price_psi_high:{psi_val:.2f}")
        if abs(delta) > max(2.0, 0.3 * self.text_mean):
            alerts.append(f"text_length_drift:{delta:.2f}")
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "price_psi": round(psi_val, 4),
            "text_length_delta": round(delta, 4),
            "alerts": alerts,
        }
        dump_json(report, self.report_path)
        return report
