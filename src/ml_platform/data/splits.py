from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["brand"] = df["brand"].fillna("unknown")
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text"] = (df["title"] + " " + df["description"]).str.strip()
    return df


def stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["category"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def baseline_stats(df: pd.DataFrame) -> dict[str, object]:
    text_lengths = df["text"].str.split().apply(len)
    price_series = df["price"].fillna(0)
    price_bins = [0, 200, 400, 600, 800, 1000, 2000]
    price_hist, _ = np.histogram(price_series, bins=price_bins)
    price_probs = (price_hist / price_hist.sum()).tolist()
    return {
        "text_length_mean": float(text_lengths.mean()),
        "price_bins": price_bins,
        "price_hist": price_probs,
    }
