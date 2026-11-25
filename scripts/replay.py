"""Validate public aggregates and regenerate the two data-driven figures."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("make_figures.py")), run_name="__main__")
