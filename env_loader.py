"""Load variables from a local .env file into the process environment.

Kept dependency-free so the examples need nothing beyond the SDK. Variables
already present in the environment always win over values in the file.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path | None = None) -> int:
    """Populate os.environ from KEY=VALUE lines. Returns the number of keys loaded."""
    env_path = path or Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return 0

    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded
