# Contributing

Thanks for taking time to improve the ML Serving Orchestration Platform!

## Prerequisites
- Python 3.11
- Poetry 1.8+
- `poetry install`

## Workflow
1. Run `poetry run python scripts/doctor.py` to ensure your environment is healthy.
2. Format/lint/type-check: `poetry run ruff check src tests` and `poetry run mypy src`.
3. Execute the full test suite: `poetry run pytest -q` and `poetry run mlp demo --n 50 --seed 1` for smoke coverage.
4. Open a pull request against `main` with a concise description of changes and any new docs/tests.

## Code style
- Prefer small, focused commits with descriptive messages.
- Keep functions short and add docstrings or inline comments only when logic is non-obvious.
- Update docs (README, runbooks) when behavior changes.

## Reporting issues
- Use GitHub Issues with reproduction steps, expected behavior, and logs.
- Security issues should follow [SECURITY.md](SECURITY.md).
