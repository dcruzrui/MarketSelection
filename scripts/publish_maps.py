#!/usr/bin/env python
"""Copy generated HTML maps into docs/ for GitHub Pages.

GitHub repository view shows HTML as source text, so the recommended way to share
interactive Folium maps is GitHub Pages (served as real HTML).

This script:
- finds the latest maps in outputs/
- falls back to bundled examples in examples/outputs/
- copies maps into docs/maps/
- ensures docs/index.html exists
"""

import os
import shutil
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pick_existing(*candidates: Path) -> Path | None:
    for p in candidates:
        if p and p.exists():
            return p
    return None


def newest_run_stress_map(outputs_dir: Path) -> Path | None:
    # Find outputs/RUN_*/STRESS_TEST_LADDER_BG/ALL_T_aggregated_map_bg.html and pick newest RUN_*
    run_dirs = sorted(outputs_dir.glob("RUN_*/STRESS_TEST_LADDER_BG/ALL_T_aggregated_map_bg.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0] if run_dirs else None


def main():
    root = repo_root()
    data_dir = Path(os.environ.get("DATA_DIR", root / "data" / "sample"))
    outputs_dir = Path(os.environ.get("OUTPUTS_DIR", root / "outputs"))
    examples_dir = root / "examples" / "outputs"

    docs_maps = root / "docs" / "maps"
    docs_maps.mkdir(parents=True, exist_ok=True)

    # Cluster map (ranked BGs)
    out_cluster = outputs_dir / "BG_ECON_CLUSTER_OUT" / "bg_rank_map.html"
    ex_cluster = examples_dir / "BG_ECON_CLUSTER_OUT" / "bg_rank_map.html"
    src1 = pick_existing(out_cluster, ex_cluster)

    # Stress test map (aggregated across ladder)
    out_stress = newest_run_stress_map(outputs_dir)
    ex_stress = examples_dir / "RUN_20260202_202016" / "STRESS_TEST_LADDER_BG" / "ALL_T_aggregated_map_bg.html"
    src2 = pick_existing(out_stress, ex_stress)

    copied = []
    for src in [src1, src2]:
        if not src:
            continue
        dst = docs_maps / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

    print("✅ Published maps to docs/maps:")
    for p in copied:
        print("  -", p.relative_to(root))

    if not copied:
        print("⚠️ No maps found to publish. Run the pipeline first.")


if __name__ == "__main__":
    main()
