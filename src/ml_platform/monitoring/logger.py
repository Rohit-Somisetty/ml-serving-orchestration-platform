from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from ml_platform.utils.io import append_jsonl

_duckdb: ModuleType | None
try:
    import duckdb as _duckdb
except ImportError:  # pragma: no cover
    _duckdb = None

DUCKDB: ModuleType | None = _duckdb


class StructuredLogger:
    def __init__(self, jsonl_path: Path, duckdb_path: Path | None = None) -> None:
        self.jsonl_path = jsonl_path
        self.duckdb_path = duckdb_path
        self._conn: Any = None
        if duckdb_path and DUCKDB is not None:
            duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = DUCKDB.connect(str(duckdb_path))
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_logs (
                    timestamp TIMESTAMP,
                    event VARCHAR
                )
                """,
            )

    def log_event(self, payload: dict[str, Any]) -> None:
        enriched = {"timestamp": datetime.utcnow().isoformat(), **payload}
        append_jsonl(enriched, self.jsonl_path)
        if self._conn is not None:
            self._conn.execute(
                "INSERT INTO inference_logs VALUES (?, ?)",
                [enriched["timestamp"], json_dumps(enriched)],
            )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


def json_dumps(obj: dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
