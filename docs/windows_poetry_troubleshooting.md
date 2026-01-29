# Windows Poetry Troubleshooting

Use a dedicated, in-project Poetry environment to avoid shared-venv path issues.

## Known-good checklist
1. Install Python 3.11 from the Windows Store or python.org and ensure `py -3.11 --version` works.
2. From PowerShell in the repo root:
   ```powershell
   poetry env list
   poetry env remove --all
   py -3.11 -m pip install --upgrade pip
   poetry env use (py -3.11 -c "import sys; print(sys.executable)")
   poetry install
   poetry run python scripts/doctor.py
   ```
3. If the stdlib import check fails, delete the `.venv` folder in the repo and rerun the steps above.

## Quick commands (copy/paste)
```powershell
cd "E:/Github projects/ml-serving-orchestration-platform"
poetry env list
poetry env remove --all
py -3.11 -m pip install --upgrade pip
$pyExe = py -3.11 -c "import sys; print(sys.executable)"
poetry env use $pyExe
poetry install
poetry run python -c "import sys, site; print(sys.version); print(site.getsitepackages())"
poetry run python scripts/doctor.py
```

If any command fails, rerun after closing shells that might have the old environment activated.
