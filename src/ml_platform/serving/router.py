from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ml_platform.config import Settings, get_settings
from ml_platform.serving.predictor import ModelPredictor
from ml_platform.training import registry
from ml_platform.training.seed_registry import ensure_seed_registry


@dataclass
class ModelHandle:
    alias: str
    version: str
    predictor: ModelPredictor


def should_route_canary(request_id: str, percent: int) -> bool:
    if percent <= 0:
        return False
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return bucket < percent


class ModelRouter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.canary_percent = self.settings.canary_percent
        ensure_seed_registry(self.settings)
        self.primary_handle = self._load_primary_handle()
        self.canary_handle = self._load_canary_handle()

    def _load_primary_handle(self) -> ModelHandle:
        registry_dir = self.settings.registry_dir
        if self.settings.model_version:
            reference = self.settings.model_version
            alias = self.settings.model_alias or "pinned"
        else:
            alias = self.settings.model_alias or registry.preferred_serving_alias(registry_dir)
            reference = alias
        model_path = registry.resolve_model_path(registry_dir, reference)
        version = registry.resolve_version_reference(registry_dir, reference)
        predictor = ModelPredictor(model_path)
        # Use predictor manifest version if available to keep metadata aligned
        resolved_version = predictor.version or version
        if resolved_version == "unregistered":
            resolved_version = version
        return ModelHandle(alias=alias, version=resolved_version, predictor=predictor)

    def _load_canary_handle(self) -> ModelHandle | None:
        if not self.settings.canary_alias:
            return None
        registry_dir = self.settings.registry_dir
        alias_ref = self.settings.canary_alias
        model_path = registry.resolve_model_path(registry_dir, alias_ref)
        version = registry.resolve_version_reference(registry_dir, alias_ref)
        predictor = ModelPredictor(model_path)
        resolved_version = predictor.version or version
        if resolved_version == "unregistered":
            resolved_version = version
        return ModelHandle(alias=alias_ref, version=resolved_version, predictor=predictor)

    def choose_handle(self, request_id: str) -> tuple[ModelHandle, bool]:
        if self.canary_handle and should_route_canary(request_id, self.canary_percent):
            return self.canary_handle, True
        return self.primary_handle, False

    def metadata(self) -> dict[str, object]:
        return {
            "primary": {
                "alias": self.primary_handle.alias,
                "version": self.primary_handle.version,
            },
            "canary": (
                {
                    "alias": self.canary_handle.alias,
                    "version": self.canary_handle.version,
                    "percent": self.canary_percent,
                }
                if self.canary_handle
                else None
            ),
        }
