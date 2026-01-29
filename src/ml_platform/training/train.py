from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_platform.config import get_settings
from ml_platform.data.splits import baseline_stats, load_dataset, stratified_split
from ml_platform.training.metrics import compute_metrics
from ml_platform.training.registry import register_model
from ml_platform.utils.hashing import file_sha256
from ml_platform.utils.io import dump_json

TEXT_COL = "text"
NUMERIC_COLS = ["price"]
CAT_COLS = ["brand"]


def _build_pipeline() -> Pipeline:
    text_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
        ],
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ],
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, TEXT_COL),
            ("price", numeric_pipeline, NUMERIC_COLS),
            ("brand", categorical_pipeline, CAT_COLS),
        ],
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    return pipeline


def train_model(data_path: Path, model_dir: Path, register: bool = False) -> dict[str, object]:
    settings = get_settings()
    df = load_dataset(data_path)
    train_df, val_df = stratified_split(df)

    features = [TEXT_COL] + NUMERIC_COLS + CAT_COLS
    pipeline = _build_pipeline()
    pipeline.fit(train_df[features], train_df["category"])

    preds = pipeline.predict(val_df[features])
    metrics = compute_metrics(val_df["category"], preds)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    joblib.dump(pipeline, model_path)

    metrics_path = dump_json(metrics, model_dir / "metrics.json")

    stats = baseline_stats(train_df)
    dataset_hash = file_sha256(data_path)

    manifest = {
        "model_version": "unregistered",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "dataset_path": str(data_path),
        "dataset_hash": dataset_hash,
        "hyperparams": {
            "max_features": 1500,
            "ngram_range": [1, 2],
            "clf": "logistic_regression",
        },
        "metrics": metrics,
        "baseline_stats": stats,
    }

    manifest_path = dump_json(manifest, model_dir / "training_manifest.json")

    registry_path = None
    version = None
    if register:
        version, registry_path = register_model(model_dir, settings.registry_dir)
        manifest["model_version"] = version
        dump_json(manifest, manifest_path)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "registry_path": str(registry_path) if registry_path else None,
        "version": version,
    }
