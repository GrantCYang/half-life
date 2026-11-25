"""Checkout resources; override HALF_LIFE_ROOT for a noneditable installation."""

import os
from pathlib import Path

ROOT = Path(
    os.environ.get("HALF_LIFE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
