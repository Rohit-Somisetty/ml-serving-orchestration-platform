from __future__ import annotations

import shutil
from pathlib import Path

from ml_platform.config import Settings


def _has_registered_versions(registry_dir: Path) -> bool:
    if not registry_dir.exists():
        return False
    for entry in registry_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("v"):
            return True
    return False


def ensure_seed_registry(settings: Settings) -> None:
    """Copy bundled seed artifacts into the active registry if it is empty."""

    registry_dir = settings.registry_dir
    if _has_registered_versions(registry_dir):
        return

    seed_registry = settings.base_dir / "seed_artifacts" / "registry"
    if not seed_registry.exists():
        return

    registry_dir.mkdir(parents=True, exist_ok=True)

    for src in seed_registry.iterdir():
        dest = registry_dir / src.name
        if src.is_dir():
            if dest.exists():
                continue
            shutil.copytree(src, dest)
        else:
            if dest.exists():
                continue
            shutil.copy2(src, dest)
