# Code Guide

This guide explains what each script does, the expected inputs, and the main outputs.
For a step-by-step walkthrough, see `GUIDE.md`. For a parameter reference, see `PARAMETERS.md`.

---

## Quick map of the pipeline

1) **Cluster + Rank** → `scripts/GitHubClusterMapNYC.py`  
2) **Monte Carlo stability** → `scripts/GitHubMonteCarlo.py`  
3) **Stress-Test Ladder + Twins + Aggregated map** → `scripts/GitHubStressTest.py`  
4) **Publish maps to GitHub Pages** → `scripts/publish_maps.py`  
5) **Run everything** → `scripts/run_all.py`

---

## Configuration (how to tune without editing code)

Preferred:
- edit `config.yaml`

Quick experiment:
- override with environment variables (env vars win over `config.yaml`)

Examples:
```bash
# move outputs somewhere else
OUTPUTS_DIR=outputs python scripts/run_all.py

# run more Monte Carlo iterations
N_RUNS=200 python scripts/GitHubMonteCarlo.py
```

---

## `scripts/GitHubClusterMapNYC.py`

**Purpose**
- Respect the user-provided `Exclusion` flag: cluster only on rows where `Exclusion == "No"`
- Optionally exclude park/open-space dominated BGs
- Exclude water-dominant BGs from ranking (but keep polygons so water is not shaded)
- Cluster BGs and rank clusters using a LocationScore
- Render `bg_rank_map.html`

**Inputs (defaults)**
- `data/sample/bg_features_nyc.csv`
- optional: open-space GeoJSON (parks/open areas)
- optional: block or BG geometry (improves polygon accuracy)

**Key outputs**
- `outputs/BG_ECON_CLUSTER_OUT/bg_rank_map.html`
- `outputs/BG_ECON_CLUSTER_OUT/excluded_blockgroups.csv`
- `outputs/BG_ECON_CLUSTER_OUT/bg_econ_results.csv` (rank table)

**Key tunables**
- Open-space exclusion threshold: `OPENSPACE_SHARE_MIN_TO_EXCLUDE`
- Water dominance threshold: `LAND_SHARE_BG_MIN`
- KMeans settings: `K_VALUES`, `RANDOM_SEED`, `PCA_COMPONENTS_MAX`
- Scoring: `BG_WEIGHTS`

See `PARAMETERS.md` for the full list.

---

## `scripts/GitHubMonteCarlo.py`

**Purpose**
- Re-fit clustering under perturbations to measure stability.
- Produces selection frequency / pass-rate tables.

**Inputs**
- `data/sample/bg_features_nyc.csv`
- optional: `EXCLUDED_BG_CSV` (drops BGs from *everything*)

**Key outputs**
- `outputs/BG_FINALWITHCOUNT_OUT/mc_counts_bg.csv`
- `outputs/BG_FINALWITHCOUNT_OUT/MC_SIM/` (checkpoints)

**Key tunables**
- `N_RUNS` (how many Monte Carlo iterations)
- `TOP_RANKS` (what counts as “selected”)
- `BOOTSTRAP_FRAC`, `NOISE_STD`
- Baseline clustering knobs: `K_RANGE`, `SOBOL_SAMPLES`

See `PARAMETERS.md`.

---

## `scripts/GitHubStressTest.py`

**Purpose**
- Applies a strict→relaxed ladder of feasibility regimes.
- Aggregates candidates across regimes (frequency + earliest regime).
- Enforces spacing to anchors and computes “twin” similarity.
- Renders the aggregated map.

**Inputs**
- Monte Carlo outputs: `mc_counts_bg.csv`
- BG features table for coordinates + z-features
- optional BG shapefile (enables queen adjacency + twin shading)

**Key outputs**
- `outputs/RUN_*/STRESS_TEST_LADDER_BG/ALL_T_aggregated_map_bg.html`
- `outputs/RUN_*/STRESS_TEST_LADDER_BG/summary_table.csv`
- `outputs/RUN_*/run_manifest.json`

**Key tunables**
- Ladder size: number of `t` steps
- Schedules: min pass-rate / min select-count / buffer distance / search radius
- Adjacency: queen (if shapefile) vs proximity radius
- Twins: alpha/q threshold and adjacency requirements
- Reverse geocoding: enable/disable + timeouts

See `PARAMETERS.md`.

---

## `scripts/publish_maps.py`

Copies the most recent maps into `docs/maps/` so they render on GitHub Pages.

**Outputs**
- `docs/maps/*.html`

---

## `scripts/run_all.py`

One command to run the pipeline end-to-end and then publish maps to `docs/`.

---

## Portable paths (environment variables)

All scripts default to repo-relative paths, but you can override them:

- `DATA_DIR` — default: `data/sample`
- `OUTPUTS_DIR` — default: `outputs`
- `BG_FEATURES_CSV` / `BG_DATA` — override the BG features CSV
- `EXCLUDED_BG_CSV` — force a specific exclusion list file

Example:
```bash
DATA_DIR=data/sample OUTPUTS_DIR=outputs python scripts/run_all.py
```
