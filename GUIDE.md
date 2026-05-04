# Guided Walkthrough

This guide is for someone opening the repository for the first time (a reviewer, hiring manager, or you later).

Design goal: you should always know:
- what stage you are running
- what files it reads/writes
- what to tune (and how)

---

## What you are producing

You are generating two interactive outputs:

1) **Cluster Ranking Map** (`bg_rank_map.html`)
   - BGs are colored by **cluster rank** (Rank 1 = most promising cluster group).
   - BGs excluded from modeling are shaded grey.
   - Water-dominant BG polygons are rendered transparent so the ocean stays unshaded.

2) **Stress-Test Ladder Aggregated Map** (`ALL_T_aggregated_map_bg.html`)
   - Candidates are aggregated across strict→relaxed regimes.
   - Tooltips emphasize *earliest regime* and *selection frequency*.
   - Rays/links (when enabled) show candidate-to-anchor “twin” relationships.

---

## Before you run anything

### Install dependencies
- `pip install -r requirements.txt` (quickest)
- or `conda env create -f environment.yml` if geospatial libraries are hard to install on your OS

### Confirm the demo data exists
Minimum required file:
- `data/sample/bg_features_nyc.csv`

Required columns in the features CSV:
- IDs: `GEOID_BG`
- Coordinates: `intpt_lat`, `intpt_lon`
- Engineered features (raw):  
  `share_college_plus`, `share_commute_60p`, `pop_density`, `occ_units_density`, `potbus_per_1k`, `median_income`

Recommended columns (if available):
- Robust z-scores: `<feature>_z` for each engineered feature
- `Exclusion` flag: `Yes`/`No` (used to mark rows with missing data)

Optional layers:
- Open-space/parks GeoJSON (used to compute open_space_share and exclude park-dominant BGs)
- Block/BG boundary geometry (improves polygon rendering and enables queen adjacency)

---

## Stage 1: Cluster + Rank

Command:
```bash
python scripts/GitHubClusterMapNYC.py
```

Reads (defaults):
- `data/sample/bg_features_nyc.csv`
- optional open-space file (see `OPENSPACE_PATH` or `config.yaml`)
- optional blocks/BG geometry, if present in `data/sample/`

Writes:
- `outputs/BG_ECON_CLUSTER_OUT/bg_rank_map.html`
- `outputs/BG_ECON_CLUSTER_OUT/bg_econ_results.csv`
- `outputs/BG_ECON_CLUSTER_OUT/excluded_blockgroups.csv`

What to check:
- The map loads and shows a set of ranked colors.
- Excluded BGs are grey (and water stays unshaded).

Interpretation:
- “Rank” is a cluster-level label (not a continuous score) derived from the LocationScore ordering of clusters.

---

## Stage 2: Monte Carlo stability

Command:
```bash
python scripts/GitHubMonteCarlo.py
```

Reads:
- `data/sample/bg_features_nyc.csv`
- optional `outputs/BG_ECON_CLUSTER_OUT/excluded_blockgroups.csv` (or bundled example)

Writes:
- `outputs/BG_FINALWITHCOUNT_OUT/mc_counts_bg.csv`
- checkpoints under `outputs/BG_FINALWITHCOUNT_OUT/MC_SIM/`

How to interpret `mc_counts_bg.csv`:
- `eligible_runs_bg`: number of runs where the BG was in the model universe
- `select_count_bg`: number of runs where the BG landed in a top-ranked cluster group
- `pass_rate_bg = select_count_bg / eligible_runs_bg`

---

## Stage 3: Stress-Test Ladder + Twins

Command:
```bash
python scripts/GitHubStressTest.py
```

Reads:
- `outputs/BG_FINALWITHCOUNT_OUT/mc_counts_bg.csv` (or bundled example)
- `data/sample/bg_features_nyc.csv`

Writes (inside a timestamped run folder):
- `outputs/RUN_*/STRESS_TEST_LADDER_BG/ALL_T_aggregated_map_bg.html`
- `outputs/RUN_*/STRESS_TEST_LADDER_BG/summary_table.csv`
- `outputs/RUN_*/STRESS_TEST_LADDER_BG/final_shortlist_bg.csv`
- `outputs/RUN_*/run_manifest.json`

How to interpret the ladder:
- The ladder sweeps `t` from strict to relaxed (more permissive as t increases).
- Candidates are summarized by:
  - earliest `t` where they survive, and
  - frequency across the ladder.

Twins:
- “Twin” logic matches candidate BG profiles to anchor profiles using a distance metric (often Mahalanobis).
- In the public demo, anchors are synthetic placeholders.

---

## Stage 4: Publish maps for GitHub Pages

Command:
```bash
python scripts/publish_maps.py
```

This copies your newest maps into:
- `docs/maps/`

Commit those `docs/` changes to GitHub, enable GitHub Pages, and open the site URL.

---

## How to tune parameters (recommended workflow)

1) Edit `config.yaml` (preferred, repo-wide)
2) Use environment variables for quick experiments

Example: increase Monte Carlo runs and change selection definition
```bash
N_RUNS=200 TOP_RANKS=3 python scripts/GitHubMonteCarlo.py
```

Full parameter reference:
- `PARAMETERS.md`

---

## Troubleshooting

- If `geopandas` install fails on Windows: use `conda` (it typically resolves compiled deps cleanly).
- If maps show as HTML source on GitHub: you must use GitHub Pages (`/docs`), not the repo browser.
- If you do not have an open-space file: set `APPLY_OPENSPACE_MASK=0` (or turn it off in `config.yaml`).
