"""Load local environment variables from a .env file (optional, no extra deps)."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> bool:
    """Parse a .env file into os.environ if it exists. Returns True if loaded."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return False

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True
