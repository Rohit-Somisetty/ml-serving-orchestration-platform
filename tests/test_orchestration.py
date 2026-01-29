from __future__ import annotations

from pathlib import Path

from ml_platform.orchestration.dag import DAG, Task
from ml_platform.orchestration.scheduler import Scheduler
from ml_platform.orchestration.state import STATUS_SUCCESS, JobStateStore


def test_dag_runner_records_job_state(_base_dir: Path) -> None:
    execution_order: list[str] = []

    def task_a(context: dict[str, object]) -> None:
        execution_order.append("task_a")

    def task_b(context: dict[str, object]) -> None:
        execution_order.append("task_b")

    dag = DAG(
        "demo",
        [Task("task_a", task_a), Task("task_b", task_b, deps=["task_a"])],
    )
    store = JobStateStore()
    scheduler = Scheduler({"demo": dag}, store=store)
    job_id = scheduler.run("demo")
    job = store.get_job(job_id)
    assert execution_order == ["task_a", "task_b"]
    assert job["status"] == STATUS_SUCCESS
    assert job["started_at"] and job["finished_at"]
    assert job["duration_ms"] is not None
