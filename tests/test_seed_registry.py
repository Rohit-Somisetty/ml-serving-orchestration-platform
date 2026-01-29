from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_platform.config import get_settings
from ml_platform.training.seed_registry import ensure_seed_registry


def test_ensure_seed_registry_copies_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path
    seed_registry = base_dir / "seed_artifacts" / "registry"
    version = "vseed99"
    version_dir = seed_registry / version
    version_dir.mkdir(parents=True)
    (version_dir / "rule_based.json").write_text(
        json.dumps({"model_version": version, "default": {"category": "demo", "confidence": 0.6}}),
        encoding="utf-8",
    )
    (seed_registry / "latest.txt").write_text(f"{version}\n", encoding="utf-8")

    monkeypatch.setenv("MLP_BASE_DIR", str(base_dir))
    get_settings.cache_clear()  # type: ignore[attr-defined]
    settings = get_settings()

    ensure_seed_registry(settings)

    copied_dir = settings.registry_dir / version
    assert copied_dir.exists()
    assert (copied_dir / "rule_based.json").exists()
    assert (settings.registry_dir / "latest.txt").read_text(encoding="utf-8").strip() == version

    get_settings.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.delenv("MLP_BASE_DIR")
