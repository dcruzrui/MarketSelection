# Methodology (Public Demo)

This repository implements a **block‑group–level site‑selection workflow** for New York City.
The public version is designed to be reproducible and resume‑ready: it uses **portable paths**,
ships with **sample inputs**, and publishes **interactive Folium maps** via GitHub Pages.

## 1) Problem framing

Goal: identify **candidate census block groups (BGs)** that look similar to strong “anchor” locations and
score well on demand‑proxy variables, while remaining operationally feasible under spacing constraints.

This is a **public demo**:
- It uses **synthetic “anchor” points** by default (instead of proprietary store coordinates).
- It focuses on **socio‑economic + intensity proxies** (no proprietary KPIs).
- Every major output is written to disk as a CSV/HTML artifact so the workflow is auditable.

## 2) Inputs and engineered features

### Primary input table
Most scripts read a BG‑level features file:

- `data/sample/bg_features_nyc.csv`

Required columns (public demo):
- `GEOID_BG` (block group ID)
- `intpt_lat`, `intpt_lon` (representative point / centroid coordinates)
- engineered variables, e.g.
  - `share_college_plus`
  - `share_commute_60p`
  - `pop_density`
  - `occ_units_density`
  - `potbus_per_1k`
  - `median_income`

### Robust z‑scores
Most modeling steps operate on robust z‑scores (suffix `_z`) of the engineered variables:

- `share_college_plus_z`, `share_commute_60p_z`, `pop_density_z`, `occ_units_density_z`,
  `potbus_per_1k_z`, `median_income_z`

These are either:
1) **precomputed in the input file**, or
2) computed inside scripts using **RobustScaler** (median / IQR scaling) to reduce sensitivity to outliers.

## 3) Exclusion logic (what gets removed and why)

The workflow explicitly separates:
- **the full universe of NYC BGs** (what you can see on the map), from
- **the modeling universe** (what is allowed into clustering / Monte Carlo / stress testing).

Common exclusion sources in this repo:

1) **User-provided Exclusion flag**
   - If `Exclusion != "No"` the BG is excluded from clustering/MC and written to `excluded_blockgroups.csv`.

2) **Open-space dominance (parks / large open areas)**
   - `GitHubClusterMapNYC.py` can exclude BGs dominated by open-space polygons by computing:
     `open_space_share = area(BG ∩ open_space) / area(BG)` (computed in EPSG:2263).
   - If `open_space_share >= threshold` (default 0.60) → excluded from ranking.

3) **Water-dominance**
   - BGs that are mostly water are treated specially so the *ocean does not get shaded*.
   - In maps, water‑dominant BGs are typically rendered **transparent**.

4) **External exclusion list (one file to rule them all)**
   - Many scripts optionally read an exclusion file:
     `outputs/BG_ECON_CLUSTER_OUT/excluded_blockgroups.csv`
   - If present, those BGs are removed **before** modeling steps.

The exclusion list is also used to **shade excluded BGs in grey** (while keeping water transparent).

## 4) Clustering + ranking (ClusterMap + MonteCarlo Part 1)

### Purpose
Group BGs into “behaviorally similar” clusters based on the robust z‑score feature space, then convert
clusters into **ranks** (Rank 1 = most promising cluster group under the scoring model).

### Feature preprocessing
Typical flow:
1) assemble the Z feature matrix `Z`
2) optionally apply **PCA with whitening** (stabilizes clustering in correlated feature spaces)
3) fit **KMeans** on either `Z` or PCA scores

### Sobol-weight search (public demo)
Some scripts search feature weights using a Sobol low‑discrepancy sampler. Conceptually:
- propose a candidate weight vector `w` (non‑negative, normalized)
- compute a weighted representation (or weighted score)
- fit clustering and evaluate internal metrics (e.g., silhouette / Davies‑Bouldin)
- keep the best weight setting under a combined score

This is a practical way to explore weighting without brute‑forcing a dense grid.

### Cluster ranking via LocationScore
After clusters are assigned, clusters are ranked using a **LocationScore**:

`LocationScore = Σ_i (w_i * z_i)`

where `z_i` are robust z‑scores for the engineered variables and `w_i` are configured weights.
Cluster rank is based on the average LocationScore of its members.

Outputs include:
- BG → `Cluster_ID`
- BG → `Cluster_Rank` (Rank 1…K)
- a map that colors BGs by rank and shades excluded BGs.

## 5) Monte Carlo stability (GitHubMonteCarlo.py)

### Purpose
Measure how often each BG is selected as “top‑rank eligible” when the clustering is perturbed.

Each Monte Carlo run typically:
1) bootstraps a subset of BGs as the fit set
2) adds small noise to the z‑features (stress test against measurement variability)
3) fits PCA(whiten) + KMeans for that run
4) re‑ranks clusters via LocationScore
5) flags BGs in clusters `rank ≤ TOP_RANKS` as “selected” for that run

Counters per BG:
- `eligible_runs_bg`: number of runs where the BG was in the evaluation universe
- `select_count_bg`: number of runs where the BG was selected
- `pass_rate_bg = select_count_bg / eligible_runs_bg`

This creates a **stability signal**: high pass_rate suggests the BG is robustly favorable.

## 6) Stress‑Test Ladder + spacing + “twin” similarity (GitHubStressTest.py)

### 6.1 Stress‑Test ladder (strict → relaxed)
The ladder runs a parameter `t` from 0.0 → 1.0, moving from **strict constraints** to **relaxed constraints**.
Each regime produces a set of BGs that pass the current feasibility filters.

For each `t`, the script writes:
- `ladder_t_<t>/kept_blocks.csv` (BG rows; kept name for parity with the block-level code)
- `ladder_t_<t>/metrics.json`

The ladder helps answer: “Which BGs survive even under strict assumptions?”

### 6.2 Aggregation across regimes
Across all `t`, the script aggregates:
- `freq`: how many regimes selected the BG
- `earliest_t`: the earliest regime where the BG appears (lower = survives stricter assumptions)

This yields a shortlist with both **robustness** (high freq) and **strict-feasibility** (low earliest_t).

### 6.3 Spacing / anti-cannibalization via anchors
To avoid selecting BGs that are too close to existing footprint, the public demo uses **anchors**:
- synthetic lat/lon points are snapped to BGs
- spacing constraints enforce minimum distance between selected BGs and anchor BGs

This mirrors the business logic of avoiding cannibalization or redundancy.

### 6.4 “Twins” (similarity nests)
The stress test also builds “twin” relationships:
- candidates are compared in feature space using a configured distance (e.g., Mahalanobis)
- BGs can be linked to the most similar anchor cluster/profile

Interpretation:
- a low twin distance means “this candidate BG looks statistically similar to a strong anchor profile”
- twin links are visualized in the aggregated map (candidate ↔ anchor rays).

## 7) Maps and interpretation

### Map 1 — Cluster ranking map
`bg_rank_map.html` shows:
- colored BGs by cluster rank
- excluded BGs shaded grey (except water-dominant BGs, which are transparent)
- anchor BGs as star markers

Use this to understand the **baseline ranking**.

### Map 2 — Stress‑test aggregated map
`ALL_T_aggregated_map_bg.html` shows:
- candidate BGs with tooltips for `earliest_t`, `freq`, and related diagnostics
- twin relationships / anchor clusters (if enabled)
- excluded BG shading for context

Use this to understand **robustness under constraints**.

## 8) Reproducibility and configuration

All scripts are GitHub‑friendly and default to repo‑relative paths:
- `DATA_DIR` (default `data/sample`)
- `OUTPUTS_DIR` (default `outputs`)

Many scripts also write a manifest JSON so results are traceable:
- inputs used
- parameters / thresholds
- timestamps and output locations

If you want a fully deterministic run, keep seeds fixed and avoid changing the input ordering.
