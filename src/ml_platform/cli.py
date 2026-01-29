from __future__ import annotations

# ruff: noqa: B008
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Generator
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import typer

from ml_platform.batch.runner import run_batch
from ml_platform.config import get_settings
from ml_platform.data.synth import generate_dataset, save_dataset
from ml_platform.orchestration.scheduler import build_scheduler
from ml_platform.orchestration.state import JobStateStore
from ml_platform.training import registry
from ml_platform.training.train import train_model
from ml_platform.utils.io import write_jsonl

app = typer.Typer(help="ML Serving Orchestration Platform CLI")
registry_app = typer.Typer(help="Model registry controls")
schedule_app = typer.Typer(help="Local scheduler controls")
jobs_app = typer.Typer(help="Job insights")

app.add_typer(registry_app, name="registry")
app.add_typer(schedule_app, name="schedule")
app.add_typer(jobs_app, name="jobs")


class DemoMode(str, Enum):
    LOCAL = "local"
    DOCKER = "docker"
    KIND = "kind"


DEMO_PORT = 8000
PACKAGE_NAME = "ml-serving-orchestration-platform"


@app.command("data-generate")
def data_generate(
    out: Path = typer.Option(..., help="Output CSV path"),
    n: int = typer.Option(5000, min=100, help="Number of samples"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Generate a deterministic synthetic dataset."""
    df = generate_dataset(n_samples=n, seed=seed)
    save_dataset(df, out)
    typer.echo(f"Dataset with {len(df)} rows written to {out}")


@app.command("train")
def train(
    data: Path = typer.Option(..., help="Training dataset CSV"),
    model_dir: Path = typer.Option(Path("artifacts/model"), help="Directory for artifacts"),
    register: bool = typer.Option(
        False,
        "--register",
        help="Register artifacts after training",
        is_flag=True,
    ),
) -> None:
    """Train the classifier and optionally register it."""
    results = train_model(data_path=data, model_dir=model_dir, register=register)
    typer.echo(f"Model artifacts created at {results['model_path']}")
    if results.get("version"):
        typer.echo(f"Registered model version: {results['version']}")


@app.command("batch")
def batch(
    input: Path = typer.Option(..., help="JSONL input requests"),
    out: Path = typer.Option(..., help="JSONL output path"),
    model_version: str = typer.Option("latest", help="Registry version, alias, or 'local'"),
) -> None:
    """Run batch inference against stored requests."""
    result = run_batch(input_path=input, output_path=out, version=model_version)
    summary = (
        "Batch finished: processed={processed} succeeded={succeeded} "
        "failed={failed} -> {output_path}"
    ).format(**result)
    typer.echo(summary)


@registry_app.command("list")
def registry_list() -> None:
    """Show available versions and aliases."""
    settings = get_settings()
    versions, aliases = registry.available_versions_with_aliases(settings.registry_dir)
    payload = {"versions": versions, "aliases": aliases}
    typer.echo(json.dumps(payload, indent=2))


@registry_app.command("promote")
def registry_promote(
    source: str = typer.Option("latest", "--from", help="Version or alias to promote"),
    target_alias: str = typer.Option(..., "--to", help="Alias to assign"),
) -> None:
    settings = get_settings()
    version = registry.promote_alias(settings.registry_dir, source, target_alias)
    typer.echo(f"Alias {target_alias} -> {version}")


@registry_app.command("set-alias")
def registry_set_alias(
    alias: str = typer.Option(..., "--alias", help="Alias name"),
    version: str = typer.Option(..., "--version", help="Target version or alias"),
) -> None:
    settings = get_settings()
    resolved = registry.set_alias(settings.registry_dir, alias, version)
    typer.echo(f"Alias {alias} now targets {resolved}")


@registry_app.command("rollback")
def registry_rollback(
    alias: str = typer.Option(..., "--alias", help="Alias to roll back"),
    steps: int = typer.Option(1, "--steps", min=1, help="Steps to roll back"),
) -> None:
    settings = get_settings()
    resolved = registry.rollback_alias(settings.registry_dir, alias, steps)
    typer.echo(f"Alias {alias} rolled back to {resolved}")


@schedule_app.command("list")
def schedule_list() -> None:
    scheduler = build_scheduler()
    for name, tasks in scheduler.list_dags().items():
        typer.echo(f"{name}: {', '.join(tasks)}")


@schedule_app.command("run")
def schedule_run(
    dag_name: str = typer.Argument(..., help="DAG name, e.g. nightly or daily_batch"),
) -> None:
    scheduler = build_scheduler()
    job_id = scheduler.run(dag_name)
    typer.echo(f"Scheduled {dag_name} -> job {job_id}")


@jobs_app.command("list")
def jobs_list(limit: int = typer.Option(5, help="How many jobs to display")) -> None:
    store = JobStateStore()
    jobs = store.list_jobs(limit=limit)
    typer.echo(json.dumps(jobs, indent=2))


@jobs_app.command("show")
def jobs_show(job_id: str = typer.Argument(..., help="Job identifier")) -> None:
    store = JobStateStore()
    job = store.get_job(job_id)
    typer.echo(json.dumps(job, indent=2))


@app.command("version")
def version() -> None:
    info = _version_metadata()
    if info["git_sha"]:
        typer.echo(f"mlp {info['version']} ({info['git_sha']})")
    else:
        typer.echo(f"mlp {info['version']}")


@app.command("demo")
def demo(  # noqa: PLR0913 - CLI needs tuneable knobs
    n: int = typer.Option(5000, min=50, help="Number of synthetic samples to generate"),
    seed: int = typer.Option(42, help="Dataset seed"),
    mode: DemoMode = typer.Option(
        DemoMode.LOCAL,
        help="Execution mode: local (default), docker, kind",
    ),
    canary_percent: int = typer.Option(
        10,
        min=0,
        max=100,
        help="Percent of traffic sent to canary",
    ),
    outdir: Path = typer.Option(
        Path("outputs/demo"),
        help="Directory for demo artifacts",
    ),
    no_train: bool = typer.Option(
        False,
        "--no-train",
        help="Skip training and reuse existing registry",
        is_flag=True,
    ),
) -> None:
    """Run an end-to-end demo: train → serve → canary → batch → drift."""

    try:
        if mode is DemoMode.LOCAL:
            summary_path = _run_local_demo(
                n=n,
                seed=seed,
                canary_percent=canary_percent,
                outdir=outdir,
                no_train=no_train,
            )
            typer.echo(f"Demo completed. Summary available at {summary_path}")
        else:
            _print_demo_instructions(mode, canary_percent)
    except Exception as exc:  # pragma: no cover - surface friendly message
        typer.echo(f"Demo failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _run_local_demo(n: int, seed: int, canary_percent: int, outdir: Path, no_train: bool) -> Path:
    settings = get_settings()
    outdir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _ensure_sample_requests(settings.data_dir)
    version_info = _version_metadata()

    dataset_rows = 0
    dataset_path = settings.data_dir / "dataset_demo.csv"
    model_dir = settings.artifacts_dir / "model_demo"

    if not no_train:
        typer.echo("[demo] Generating dataset…")
        df = generate_dataset(n_samples=n, seed=seed)
        dataset_rows = len(df)
        save_dataset(df, dataset_path)
        typer.echo("[demo] Training + registering model…")
        train_model(data_path=dataset_path, model_dir=model_dir, register=True)
    else:
        typer.echo("[demo] --no-train supplied; reusing existing registry artifacts.")

    stable_version, candidate_version = _configure_demo_aliases(settings.registry_dir)

    env = os.environ.copy()
    env.update(
        {
            "MLP_BASE_DIR": str(settings.base_dir),
            "MODEL_ALIAS": "stable",
            "CANARY_ALIAS": "candidate",
            "CANARY_PERCENT": str(canary_percent),
        },
    )

    log_path = outdir / "api.log"
    request_records: list[dict[str, Any]] = []
    response_records: list[dict[str, Any]] = []

    typer.echo("[demo] Launching FastAPI (uvicorn)…")
    with _api_process(DEMO_PORT, env, log_path) as base_url:
        client = httpx.Client(base_url=base_url, timeout=5.0)
        try:
            single_payload = {
                "title": "Modern sofa",
                "description": "Sectional couch with storage chaise",
                "price": 1299,
                "brand": "FurniCo",
            }
            request_records.append({"endpoint": "/predict", "payload": single_payload})
            single_resp = client.post("/predict", json=single_payload)
            single_resp.raise_for_status()
            response_records.append(
                {
                    "endpoint": "/predict",
                    "status": single_resp.status_code,
                    "body": single_resp.json(),
                },
            )

            batch_payload = [
                {
                    "title": "Outdoor grill",
                    "description": "Stainless steel 4-burner grill",
                    "price": 799,
                    "brand": "HeatMaster",
                },
                {
                    "title": "",
                    "description": "",
                    "price": None,
                    "brand": "",
                },
            ]
            request_records.append({"endpoint": "/predict/batch", "payload": batch_payload})
            batch_resp = client.post("/predict/batch", json=batch_payload)
            batch_resp.raise_for_status()
            response_records.append(
                {
                    "endpoint": "/predict/batch",
                    "status": batch_resp.status_code,
                    "body": batch_resp.json(),
                },
            )

            metrics_resp = client.get("/metrics/summary")
            metrics_resp.raise_for_status()
            metrics_summary = metrics_resp.json()
            response_records.append(
                {
                    "endpoint": "/metrics/summary",
                    "status": metrics_resp.status_code,
                    "body": metrics_summary,
                },
            )
        finally:
            client.close()

    typer.echo("[demo] Running nightly + daily_batch DAGs…")
    scheduler = build_scheduler()
    nightly_job = scheduler.run("nightly")
    daily_job = scheduler.run("daily_batch")
    jobs_ran = [nightly_job, daily_job]

    metrics_path = outdir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(response_records[-1]["body"], indent=2), encoding="utf-8")

    requests_path = outdir / "demo_requests.jsonl"
    responses_path = outdir / "demo_responses.jsonl"
    write_jsonl(request_records, requests_path)
    write_jsonl(response_records, responses_path)

    drift_dest = None
    if settings.drift_report.exists():
        drift_dest = outdir / "drift_report.json"
        shutil.copy2(settings.drift_report, drift_dest)

    jobs_snapshot = JobStateStore().list_jobs(limit=10)
    jobs_snapshot_path = outdir / "jobs_snapshot.json"
    jobs_snapshot_path.write_text(json.dumps(jobs_snapshot, indent=2), encoding="utf-8")

    summary = {
        "mode": DemoMode.LOCAL.value,
        "dataset_rows": dataset_rows,
        "model_trained": not no_train,
        "stable_version": stable_version,
        "candidate_version": candidate_version,
        "app_version": version_info["version"],
        "git_sha": version_info["git_sha"],
        "canary_percent": canary_percent,
        "requests_sent": len(request_records),
        "requests_path": str(requests_path),
        "responses_path": str(responses_path),
        "metrics_summary_path": str(metrics_path),
        "drift_report_path": str(drift_dest) if drift_dest else None,
        "jobs_snapshot_path": str(jobs_snapshot_path),
        "jobs_ran": jobs_ran,
        "api_log_path": str(log_path),
        "metrics_summary": response_records[-1]["body"],
        "jobs_snapshot": jobs_snapshot,
    }

    summary_path = outdir / "demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _configure_demo_aliases(registry_dir: Path) -> tuple[str, str]:
    latest_version = registry.get_latest_version(registry_dir)
    if not latest_version:
        raise RuntimeError("No registered models found. Train a model or remove --no-train.")

    previous_stable = registry.get_alias_version(registry_dir, "stable")
    candidate_version = registry.set_alias(registry_dir, "candidate", "latest")
    if previous_stable:
        stable_version = previous_stable
    else:
        stable_version = candidate_version
        registry.set_alias(registry_dir, "stable", stable_version)
    return stable_version, candidate_version


@contextlib.contextmanager
def _api_process(port: int, env: dict[str, str], log_path: Path) -> Generator[str, None, None]:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "ml_platform.serving.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)
        try:
            _wait_for_api(f"http://127.0.0.1:{port}/health")
            yield f"http://127.0.0.1:{port}"
        finally:
            _stop_process(process)


def _ensure_sample_requests(data_dir: Path) -> Path:
    sample_path = data_dir / "sample_requests.jsonl"
    if sample_path.exists():
        return sample_path
    examples = [
        {
            "title": "Modern sofa",
            "description": "Sectional couch with storage chaise",
            "price": 1299,
            "brand": "FurniCo",
        },
        {
            "title": "Outdoor grill",
            "description": "Stainless steel 4-burner grill",
            "price": 799,
            "brand": "HeatMaster",
        },
        {
            "title": "Desk",
            "description": "Writing desk",
            "price": 320,
            "brand": "UrbanNest",
        },
    ]
    payload = "\n".join(json.dumps(record) for record in examples) + "\n"
    sample_path.write_text(payload, encoding="utf-8")
    return sample_path


def _wait_for_api(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(1)
        else:
            time.sleep(1)
    raise RuntimeError("API failed to start within timeout")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            process.kill()


def _print_demo_instructions(mode: DemoMode, canary_percent: int) -> None:
    if mode is DemoMode.DOCKER:
        typer.echo(
            "[demo] Docker mode: run scripts/smoke_docker.sh to build, run, "
            "and verify the API container.",
        )
        typer.echo(
            "[demo] Override CANARY_PERCENT with docker run -e CANARY_PERCENT=VALUE "
            "if needed.",
        )
    elif mode is DemoMode.KIND:
        typer.echo("[demo] kind mode: run scripts/smoke_kind.sh or follow docs/k8s_runbook.md.")
        typer.echo(f"[demo] Apply the canary overlay to route {canary_percent}% of traffic.")
    else:  # pragma: no cover
        typer.echo("[demo] Local mode already executed.")


def _version_metadata() -> dict[str, str | None]:
    try:
        version = metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:  # pragma: no cover - local dev
        version = "0.0.0"
    git_sha: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_sha = result.stdout.strip() or None
    except Exception:  # pragma: no cover - git missing
        git_sha = None
    return {"version": version, "git_sha": git_sha}


if __name__ == "__main__":
    app()
