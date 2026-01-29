from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd

from ml_platform.config import get_settings
from ml_platform.training.registry import resolve_model_path
from ml_platform.utils.io import load_json

FEATURE_COLUMNS = ["text", "price", "brand"]


class ModelPredictor:
    def __init__(self, model_dir: Path) -> None:
        model_path = model_dir / "model.pkl"
        manifest_path = model_dir / "training_manifest.json"
        metrics_path = model_dir / "metrics.json"
        rule_path = model_dir / "rule_based.json"

        self.model_dir = model_dir
        self._backend: str = "sklearn"
        self._pipeline: Any | None = None
        self._rule_config: dict[str, Any] | None = None

        if model_path.exists():
            self._pipeline = joblib.load(model_path)
        elif rule_path.exists():
            self._backend = "rule_based"
            self._rule_config = cast(dict[str, Any], load_json(rule_path))
        else:
            raise FileNotFoundError(
                (
                    "Missing model artifacts. Provide a trained pipeline (model.pkl) "
                    "or a rule_based.json seed."
                ),
            )

        self.manifest: dict[str, Any] = (
            cast(dict[str, Any], load_json(manifest_path)) if manifest_path.exists() else {}
        )
        self.metrics: dict[str, Any] = (
            cast(dict[str, Any], load_json(metrics_path)) if metrics_path.exists() else {}
        )

        manifest_version = self.manifest.get("model_version") or (
            self._rule_config.get("model_version") if self._rule_config else None
        )
        self.version = str(manifest_version or model_dir.name)

    @staticmethod
    def _prepare_frame(records: Iterable[dict[str, object]]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for record in records:
            title = record.get("title") or ""
            description = record.get("description") or ""
            rows.append(
                {
                    "text": f"{str(title)} {str(description)}".strip(),
                    "price": record.get("price"),
                    "brand": str(record.get("brand") or "unknown"),
                },
            )
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    def predict_one(self, record: dict[str, object]) -> dict[str, object]:
        if self._backend == "rule_based":
            return self._predict_rule_based(record)
        assert self._pipeline is not None
        df = self._prepare_frame([record])
        proba = self._pipeline.predict_proba(df)[0]
        idx = int(proba.argmax())
        label = str(self._pipeline.classes_[idx])
        confidence = float(proba[idx])
        return cast(dict[str, object], {"category": label, "confidence": confidence})

    def predict_batch(self, records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        if not records:
            return []
        if self._backend == "rule_based":
            return [self._predict_rule_based(record) for record in records]
        assert self._pipeline is not None
        df = self._prepare_frame(records)
        proba = self._pipeline.predict_proba(df)
        preds = self._pipeline.classes_[proba.argmax(axis=1)]
        return [
            cast(
                dict[str, object],
                {"category": str(label), "confidence": float(prob.max())},
            )
            for label, prob in zip(preds, proba, strict=False)
        ]

    def _predict_rule_based(self, record: dict[str, object]) -> dict[str, object]:
        assert self._rule_config is not None
        title = str(record.get("title") or "")
        description = str(record.get("description") or "")
        text = (title + " " + description).lower()
        brand = str(record.get("brand") or "").lower()
        text_blob = f"{text} {brand}".strip()
        rules = self._rule_config.get("rules", [])
        default_cfg = self._rule_config.get("default", {})
        default_category = str(default_cfg.get("category", "home"))
        default_conf = float(default_cfg.get("confidence", 0.55))

        for rule in rules:
            keywords = [kw.lower() for kw in rule.get("keywords", [])]
            if not keywords:
                continue
            if any(keyword in text_blob for keyword in keywords):
                return cast(
                    dict[str, object],
                    {
                        "category": str(rule.get("category", default_category)),
                        "confidence": float(rule.get("confidence", default_conf)),
                    },
                )

        # Lightweight numeric heuristic fallback
        price = record.get("price")
        if isinstance(price, int | float):
            if price >= 800:
                return cast(dict[str, object], {"category": "furniture", "confidence": 0.65})
            if price <= 50:
                return cast(dict[str, object], {"category": "kitchen", "confidence": 0.6})

        return cast(
            dict[str, object],
            {"category": default_category, "confidence": default_conf},
        )


def load_predictor(version: str | None = None) -> ModelPredictor:
    settings = get_settings()
    if version in (None, "latest"):
        model_dir = resolve_model_path(settings.registry_dir, None)
    else:
        model_dir = resolve_model_path(settings.registry_dir, version)
    return ModelPredictor(model_dir)
