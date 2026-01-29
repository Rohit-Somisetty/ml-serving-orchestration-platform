from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast

from ml_platform.config import get_settings

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


class JobRecord(TypedDict):
    job_id: str
    dag_name: str
    status: str
    queued_at: str
    started_at: str | None
    finished_at: str | None
    duration_ms: float | None
    error: str | None


class JobStateStore:
    def __init__(self, jobs_dir: Path | None = None) -> None:
        settings = get_settings()
        self.jobs_dir = jobs_dir or settings.jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _job_file(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _write_state(self, state: JobRecord) -> None:
        job_id = state["job_id"]
        self._job_file(job_id).write_text(json.dumps(state, indent=2), encoding="utf-8")

    def create_job(self, dag_name: str) -> JobRecord:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        job_id = f"{dag_name}-{timestamp}"
        state: JobRecord = {
            "job_id": job_id,
            "dag_name": dag_name,
            "status": STATUS_QUEUED,
            "queued_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "finished_at": None,
            "duration_ms": None,
            "error": None,
        }
        self._write_state(state)
        return state

    def mark_running(self, job_id: str) -> JobRecord:
        state = self.get_job(job_id)
        now = datetime.utcnow().isoformat()
        state["status"] = STATUS_RUNNING
        state["started_at"] = now
        self._write_state(state)
        return state

    def mark_finished(
        self,
        job_id: str,
        status: str,
        error: str | None = None,
    ) -> JobRecord:
        state = self.get_job(job_id)
        finished = datetime.utcnow()
        state["status"] = status
        state["finished_at"] = finished.isoformat()
        started_at = state.get("started_at")
        if started_at:
            start_dt = datetime.fromisoformat(started_at)
            state["duration_ms"] = (finished - start_dt).total_seconds() * 1000
        if error:
            state["error"] = error
        self._write_state(state)
        return state

    def get_job(self, job_id: str) -> JobRecord:
        path = self._job_file(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Job {job_id} not found")
        return cast(JobRecord, json.loads(path.read_text(encoding="utf-8")))

    def list_jobs(self, limit: int = 10) -> list[JobRecord]:
        records: list[JobRecord] = []
        for file_path in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            records.append(cast(JobRecord, json.loads(file_path.read_text(encoding="utf-8"))))
        return records[:limit]
