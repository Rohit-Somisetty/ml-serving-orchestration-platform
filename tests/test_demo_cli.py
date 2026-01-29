from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ml_platform.cli import app
from ml_platform.config import get_settings


def test_demo_command_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv("MLP_BASE_DIR", str(tmp_path))
    get_settings.cache_clear()  # type: ignore[attr-defined]

    outdir = tmp_path / "demo"
    result = runner.invoke(
        app,
        [
            "demo",
            "--n",
            "50",
            "--seed",
            "1",
            "--mode",
            "local",
            "--outdir",
            str(outdir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    summary_path = outdir / "demo_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in [
        "dataset_rows",
        "stable_version",
        "candidate_version",
        "metrics_summary",
        "jobs_ran",
        "app_version",
        "git_sha",
    ]:
        assert key in summary
    assert summary["dataset_rows"] == 50
    assert (outdir / "demo_requests.jsonl").exists()
    assert (outdir / "demo_responses.jsonl").exists()

    get_settings.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.delenv("MLP_BASE_DIR", raising=False)
