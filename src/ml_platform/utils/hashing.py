from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()
