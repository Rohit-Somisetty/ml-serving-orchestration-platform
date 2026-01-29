from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

CATEGORIES = ["sofa", "bed", "table", "chair"]
BRANDS = ["FurniCo", "HomeCraft", "UrbanNest", "Oak&Co", "ComfyMade"]
ADJECTIVES = ["modern", "vintage", "compact", "luxury", "ergonomic", "eco"]
MATERIALS = ["oak", "pine", "steel", "bamboo", "walnut"]


def _sample_category(rng: random.Random) -> str:
    weights = [0.3, 0.2, 0.25, 0.25]
    return rng.choices(CATEGORIES, weights=weights, k=1)[0]


def _build_text(category: str, rng: random.Random) -> tuple[str, str]:
    adjective = rng.choice(ADJECTIVES)
    material = rng.choice(MATERIALS)
    title = f"{adjective.title()} {category.title()}"
    desc = (
        f"{adjective} {category} crafted with {material}. "
        f"Designed for {rng.choice(['studio', 'family', 'office'])} spaces."
    )
    if rng.random() < 0.15:
        desc += " Minor cosmetic imperfections add character."
    return title, desc


def _price_for_category(category: str, rng: random.Random) -> float:
    base = {"sofa": 900, "bed": 700, "table": 400, "chair": 250}[category]
    noise = rng.gauss(0, base * 0.15)
    price = max(50.0, base + noise)
    if rng.random() < 0.05:
        price *= rng.uniform(0.5, 1.5)
    return round(price, 2)


def generate_dataset(n_samples: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for _ in range(n_samples):
        category = _sample_category(rng)
        raw_title, raw_description = _build_text(category, rng)
        title: str | None = raw_title
        description: str | None = raw_description
        price = _price_for_category(category, rng)
        brand: str | None = rng.choice(BRANDS)
        if rng.random() < 0.1:
            brand = None
        if rng.random() < 0.05:
            description = None
        if rng.random() < 0.03:
            title = None
        rows.append(
            {
                "title": title,
                "description": description,
                "price": price,
                "brand": brand,
                "category": category,
            },
        )
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_dataset(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
