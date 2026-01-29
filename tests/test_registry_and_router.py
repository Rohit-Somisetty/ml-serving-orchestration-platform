from __future__ import annotations

from ml_platform.config import get_settings
from ml_platform.serving.router import ModelRouter, should_route_canary
from ml_platform.training import registry


def test_registry_promotion_sets_stable_alias(training_result: dict[str, object]) -> None:
    settings = get_settings()
    versions = registry.list_versions(settings.registry_dir)
    assert versions
    target_version = versions[-1]
    registry.set_alias(settings.registry_dir, "stable", target_version)
    router = ModelRouter(settings)
    assert router.primary_handle.alias == "stable"
    assert router.primary_handle.version == target_version


def test_canary_routing_is_deterministic() -> None:
    rid = "request-123"
    first = should_route_canary(rid, 25)
    second = should_route_canary(rid, 25)
    assert first == second
    assert should_route_canary(rid, 0) is False
