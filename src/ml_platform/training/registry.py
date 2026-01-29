from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast

LATEST_FILE = "latest.txt"
ALIASES_DIR = "aliases"


def _latest_file(registry_dir: Path) -> Path:
    return registry_dir / LATEST_FILE


def _aliases_dir(registry_dir: Path) -> Path:
    return registry_dir / ALIASES_DIR


def register_model(source_dir: Path, registry_dir: Path) -> tuple[str, Path]:
    version = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    target = registry_dir / version
    if target.exists():
        raise FileExistsError(f"Registry version {version} already exists")
    shutil.copytree(source_dir, target)
    _latest_file(registry_dir).write_text(version, encoding="utf-8")
    return version, target


def list_versions(registry_dir: Path) -> list[str]:
    versions = [p.name for p in registry_dir.iterdir() if p.is_dir() and p.name.startswith("v")]
    return sorted(versions)


def get_latest_version(registry_dir: Path) -> str | None:
    latest_path = _latest_file(registry_dir)
    if latest_path.exists():
        return latest_path.read_text(encoding="utf-8").strip()
    versions = list_versions(registry_dir)
    return versions[-1] if versions else None


def _alias_state_path(registry_dir: Path, alias: str) -> Path:
    alias_dir = _aliases_dir(registry_dir)
    alias_dir.mkdir(parents=True, exist_ok=True)
    return alias_dir / f"{alias}.json"


class AliasState(TypedDict, total=False):
    alias: str
    current: str
    history: list[str]
    updated_at: str


def _load_alias_state(registry_dir: Path, alias: str) -> AliasState:
    path = _alias_state_path(registry_dir, alias)
    if not path.exists():
        return {"alias": alias, "history": []}
    return cast(AliasState, json.loads(path.read_text(encoding="utf-8")))


def list_aliases(registry_dir: Path) -> dict[str, str]:
    alias_dir = _aliases_dir(registry_dir)
    if not alias_dir.exists():
        return {}
    aliases: dict[str, str] = {}
    for file_path in alias_dir.glob("*.json"):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        current = data.get("current")
        if current:
            aliases[file_path.stem] = current
    return aliases


def get_alias_version(registry_dir: Path, alias: str) -> str | None:
    return list_aliases(registry_dir).get(alias)


def resolve_version_reference(registry_dir: Path, reference: str | None) -> str:
    if reference in (None, "latest"):
        resolved = get_latest_version(registry_dir)
        if not resolved:
            raise FileNotFoundError("No registered models found")
        return resolved
    aliases = list_aliases(registry_dir)
    if reference in aliases:
        return aliases[reference]
    if reference is None:
        raise ValueError("Version reference cannot be None after alias resolution")
    candidate_path = registry_dir / reference
    if not candidate_path.exists():
        raise FileNotFoundError(f"Model reference {reference} not found")
    return reference


def resolve_model_path(registry_dir: Path, reference: str | None) -> Path:
    version = resolve_version_reference(registry_dir, reference)
    path = registry_dir / version
    if not path.exists():
        raise FileNotFoundError(f"Model version {version} not found in registry")
    return path


def set_alias(registry_dir: Path, alias: str, reference: str) -> str:
    version = resolve_version_reference(registry_dir, reference)
    state = _load_alias_state(registry_dir, alias)
    history = list(state.get("history", []))
    history.append(version)
    state.update(
        {
            "alias": alias,
            "current": version,
            "history": history,
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    path = _alias_state_path(registry_dir, alias)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return version


def promote_alias(registry_dir: Path, source: str, target_alias: str) -> str:
    version = resolve_version_reference(registry_dir, source)
    return set_alias(registry_dir, target_alias, version)


def rollback_alias(registry_dir: Path, alias: str, steps: int = 1) -> str:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    state = _load_alias_state(registry_dir, alias)
    history: list[str] = state.get("history", [])
    if len(history) <= steps:
        raise ValueError(f"Alias {alias} does not have {steps} previous versions")
    target_version = history[-(steps + 1)]
    return set_alias(registry_dir, alias, target_version)


def preferred_serving_alias(registry_dir: Path) -> str:
    aliases = list_aliases(registry_dir)
    if "stable" in aliases:
        return "stable"
    return "latest"


def available_versions_with_aliases(registry_dir: Path) -> tuple[list[str], dict[str, str]]:
    return list_versions(registry_dir), list_aliases(registry_dir)
