#!/usr/bin/env python
"""Run the public demo pipeline end-to-end.

Default folders:
  - DATA_DIR:    data/sample
  - OUTPUTS_DIR: outputs

Usage:
  python scripts/run_all.py
  python scripts/run_all.py --steps cluster_map monte_carlo stress_test publish
  python scripts/run_all.py --data-dir path/to/data --outputs-dir path/to/outputs
"""

import os
import argparse
import runpy
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _set_env(data_dir: Path, outputs_dir: Path) -> None:
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["OUTPUTS_DIR"] = str(outputs_dir)


def _run(script_path: Path) -> None:
    print(f"\n▶ Running: {script_path.name}")
    runpy.run_path(str(script_path), run_name="__main__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to input data folder (default: data/sample)",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Path to outputs folder (default: outputs)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["cluster_map", "monte_carlo", "stress_test", "publish"],
        choices=["optimal_selection", "cluster_map", "monte_carlo", "stress_test", "publish"],
        help="Which steps to run (default: cluster_map monte_carlo stress_test publish)",
    )
    args = parser.parse_args()

    root = _repo_root()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else (root / "data" / "sample")
    outputs_dir = Path(args.outputs_dir).resolve() if args.outputs_dir else (root / "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    _set_env(data_dir, outputs_dir)

    scripts_dir = root / "scripts"

    if "optimal_selection" in args.steps:
        _run(scripts_dir / "GitHubOptimalSelection.py")

    if "cluster_map" in args.steps:
        _run(scripts_dir / "GitHubClusterMapNYC.py")

    if "monte_carlo" in args.steps:
        _run(scripts_dir / "GitHubMonteCarlo.py")

    if "stress_test" in args.steps:
        _run(scripts_dir / "GitHubStressTest.py")

    if "publish" in args.steps:
        _run(scripts_dir / "publish_maps.py")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
