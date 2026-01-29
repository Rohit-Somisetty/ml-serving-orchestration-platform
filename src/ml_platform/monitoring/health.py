from __future__ import annotations

from ml_platform.config import get_settings
from ml_platform.training.registry import get_latest_version, resolve_model_path


def health_probe() -> dict[str, object]:
    settings = get_settings()
    model_loaded = False
    latest_version: str | None = None
    try:
        latest_version = get_latest_version(settings.registry_dir)
        if latest_version:
            path = resolve_model_path(settings.registry_dir, latest_version)
            model_loaded = path.exists()
    except FileNotFoundError:
        model_loaded = False
    registry_available = settings.registry_dir.exists()
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "registry_available": registry_available,
        "latest_version": latest_version,
    }
