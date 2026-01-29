from __future__ import annotations

from ml_platform.data.synth import generate_dataset


def test_generate_dataset_shape() -> None:
    df = generate_dataset(n_samples=200, seed=3)
    assert len(df) == 200
    assert set(["title", "description", "price", "brand", "category"]).issubset(df.columns)
    assert df["category"].nunique() >= 4
