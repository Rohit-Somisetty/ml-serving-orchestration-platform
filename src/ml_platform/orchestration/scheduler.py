from __future__ import annotations

from ml_platform.config import get_settings
from ml_platform.orchestration.dag import DAG
from ml_platform.orchestration.state import STATUS_FAILED, STATUS_SUCCESS, JobStateStore


class Scheduler:
    def __init__(self, dags: dict[str, DAG], store: JobStateStore | None = None) -> None:
        self.dags = dags
        self.store = store or JobStateStore()

    def list_dags(self) -> dict[str, list[str]]:
        return {name: dag.task_names() for name, dag in self.dags.items()}

    def run(self, dag_name: str) -> str:
        if dag_name not in self.dags:
            raise ValueError(f"Unknown DAG {dag_name}")
        dag = self.dags[dag_name]
        job_state = self.store.create_job(dag_name)
        job_id = job_state["job_id"]
        context = {
            "job_id": job_id,
            "settings": get_settings(),
        }
        self.store.mark_running(job_id)
        try:
            dag.execute(context)
            self.store.mark_finished(job_id, STATUS_SUCCESS)
        except Exception as exc:
            self.store.mark_finished(job_id, STATUS_FAILED, error=str(exc))
            raise
        return job_id


def build_scheduler() -> Scheduler:
    from ml_platform.orchestration.dags import build_registered_dags

    dags = build_registered_dags()
    return Scheduler(dags)
