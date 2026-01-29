from __future__ import annotations

import importlib
import sys
from typing import List

REQUIRED_MODULES = ["json", "ssl", "pathlib"]


def _print_header() -> None:
    print("MLP Environment Doctor")
    print("========================")


def _check_stdlib() -> List[str]:
    failures: List[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append(f"{module_name}: {exc}")
    return failures


def main() -> None:
    _print_header()
    print(f"sys.executable : {sys.executable}")
    print(f"sys.version    : {sys.version}")
    print(f"sys.prefix     : {sys.prefix}")
    base_prefix = getattr(sys, "base_prefix", None)
    if base_prefix:
        print(f"sys.base_prefix: {base_prefix}")

    if sys.version_info < (3, 11):
        print("Python 3.11+ is required", file=sys.stderr)
        sys.exit(1)

    failures = _check_stdlib()
    if failures:
        print("Stdlib import failures detected:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        sys.exit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()
