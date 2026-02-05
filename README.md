All the interactive maps and guides: https://dcruzrui.github.io/MarketSelection/

# NYC Site Selection (Public Demo)

A reproducible, public-facing demo of a **NYC block‑group site selection** workflow.

This repo is written as a **guided walkthrough**. At any point you should be able to answer:
1) **What stage am I running?**
2) **What files does it read/write?**
3) **What can I tune (and where)?**

Interactive maps are published via **GitHub Pages** (so they render as real maps, not HTML source).

---

## What this project does

**Goal:** rank NYC census **block groups (BGs)** by similarity to a “good” profile using socio‑economic variables and a business‑intensity proxy, then stress-test candidate lists under stricter/looser constraints.

**Pipeline stages**

1) **Cluster + Rank** (`scripts/GitHubClusterMapNYC.py`)
   - applies exclusions (missing-data flag, open‑space dominance, water dominance)
   - clusters BGs using robust z‑features
   - ranks clusters with a weighted LocationScore
   - produces a **Cluster Ranking Map**

2) **Monte Carlo stability** (`scripts/GitHubMonteCarlo.py`)
   - repeats clustering under bootstrap + noise perturbations
   - counts how often each BG lands in a “top” cluster
   - produces **pass‑rate / stability** tables

3) **Stress‑Test Ladder + Twins** (`scripts/GitHubStressTest.py`)
   - runs a strict→relaxed ladder of feasibility regimes
   - aggregates candidates across regimes (earliest regime + frequency)
   - enforces spacing to anchors (public demo uses synthetic anchors)
   - computes “twin” similarity (candidate ↔ most similar anchor)
   - produces an **Aggregated Stress‑Test Map**

4) **Publish maps** (`scripts/publish_maps.py`)
   - copies the latest generated maps into `docs/maps/` for GitHub Pages

If you want the full narrative explanation, start here:
- `GUIDE.md` (step‑by‑step “what to run, what to look for”)
- `METHODOLOGY.md` (why each step exists)
- `CODE_GUIDE.md` (script-by-script inputs/outputs)

---

## Data (what’s in `data/sample/`)

This repo ships with a **small sample dataset** so a reviewer can run the pipeline end‑to‑end.

### Core table
- `data/sample/bg_features_nyc.csv`
  - One row per BG (`GEOID_BG`)
  - Must include:
    - geometry centroids: `intpt_lat`, `intpt_lon`
    - engineered features (raw):
      - `share_college_plus`, `share_commute_60p`, `pop_density`, `occ_units_density`, `potbus_per_1k`, `median_income`
    - optional robust z-scores: `<feature>_z` (recommended)
    - an `Exclusion` flag (`Yes`/`No`) if you want missing-data exclusion handled upstream

### Optional geospatial layers
- Open-space / parks GeoJSON (optional)
  - Used to compute `open_space_share` and exclude BGs that are mostly parks/open land.
  - Configure via `OPENSPACE_PATH` (env) or `config.yaml`.

- Block or BG boundaries (optional)
  - If present, enables more accurate BG polygons and (in StressTest) **queen adjacency** and twin shading.

---

## Quickstart (recommended)

### 1) Install

**pip**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**conda** (recommended if geospatial wheels are tricky)
```bash
conda env create -f environment.yml
conda activate nyc-site-selection
```

### 2) Run everything

```bash
python scripts/run_all.py
```

This runs: Cluster → Monte Carlo → StressTest → Publish maps.

### 3) Open the results locally

- `outputs/BG_ECON_CLUSTER_OUT/bg_rank_map.html`
- `outputs/RUN_*/STRESS_TEST_LADDER_BG/ALL_T_aggregated_map_bg.html`

---

## Tunable parameters (very important)

All key knobs are documented in **one place**:

- `PARAMETERS.md` — human-readable parameter reference (defaults + what they do)
- `config.yaml` — edit this file to tune the pipeline without touching code

You can also override many values with environment variables.

Example: increase Monte Carlo runs and change top ranks
```bash
N_RUNS=200 TOP_RANKS=3 python scripts/GitHubMonteCarlo.py
```

---

## Publish the maps so they render on GitHub

GitHub does **not** render interactive HTML maps inside the repo browser.
Use **GitHub Pages** so the maps open normally.

### Enable Pages
1. Repo → **Settings → Pages**
2. Build and deployment: “Deploy from a branch”
3. Branch: `main` (or `master`), Folder: `/docs`

### After running locally
```bash
python scripts/publish_maps.py
```

Then commit the updated `docs/maps/*.html`.

---

## Repo structure

```
data/
  sample/                 # portable demo inputs (small)
examples/
  outputs/                # bundled example outputs (for previews)
outputs/                  # YOUR local outputs (gitignored)
scripts/                  # runnable pipeline scripts
src/                      # minimal package stub

docs/                     # GitHub Pages site (index + maps + guide)
```

---

## Documentation (start here)

- `GUIDE.md` – step-by-step “what we’re doing” walkthrough
- `METHODOLOGY.md` – conceptual explanation of the pipeline
- `PARAMETERS.md` – what you can tune + safe ranges
- `CODE_GUIDE.md` – script-by-script inputs/outputs + where to edit

---

## License
MIT (see `LICENSE`).
