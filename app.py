"""Entrypoint shim. Hugging Face Spaces requires app.py at the repo root."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "app" / "main.py"), run_name="__main__")
