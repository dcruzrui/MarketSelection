"""Entry point: python -m nyc_site_selection

This simply delegates to scripts/run_all.py.
"""

from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[2]
runpy.run_path(str(root / "scripts" / "run_all.py"), run_name="__main__")
