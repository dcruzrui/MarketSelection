All the interactive maps and guides: https://dcruzrui.github.io/MarketSelection/

# MarketSelection: NYC Site Selection (Public Demo)

A reproducible, public-facing **experiment** demonstrating a full site-selection pipeline for a citywide business network using NYC census **block groups (BGs)**. The workflow converts neighborhood-level signals into a ranked opportunity surface, validates the stability of top areas under repeated perturbations, and produces an anchor-aware shortlist designed to expand beyond an existing footprint.

**Interactive maps (GitHub Pages):** https://dcruzrui.github.io/MarketSelection/

---

## What this experiment demonstrates

This experiment answers a practical expansion question:

> Given an existing network footprint, which NYC block groups look most suitable for **new locations**, and how robust are those recommendations?

The pipeline is designed to be:
- **Reproducible** (end-to-end scripts, pinned dependencies, deterministic outputs where possible)
- **Configurable** (key assumptions are centralized and tunable)
- **Interpretable** (anchor-aware selection and candidate↔anchor similarity reporting)
- **Reviewable** (interactive maps + clear stage-by-stage artifacts)

---

## Anchors: representing the existing business footprint

Anchors represent **existing locations for the business type** (stores/service centers/branches). They define the baseline footprint that already serves the market and are incorporated directly into candidate selection.

- In production, anchors correspond to the business’s real locations.
- In this public demo, anchors are **synthetic** to avoid exposing proprietary coordinates, while preserving identical logic.

Anchors are used in two ways:

### 1) Coverage expansion via spacing constraints
High-scoring areas are not automatically good expansion targets if they sit too close to existing locations. The pipeline applies spacing rules relative to anchors (and often between selected candidates) to encourage **net-new coverage** and reduce overlap/cannibalization.

### 2) Candidate↔anchor “twin” matching for interpretability
Each candidate BG is matched to its most similar anchor in feature space (its “twin”). This provides an explanation layer: selected candidates resemble the profile of neighborhoods where existing locations operate successfully, while being positioned to extend the footprint into new areas.

---

## Pipeline stages (what each script does)

### Stage 1 — Cluster + Rank neighborhoods  
**Script:** `scripts/GitHubClusterMapNYC.py`

**Purpose:** Build a citywide opportunity surface by clustering comparable BGs and ranking clusters by a weighted score.

**Key steps:**
- Applies exclusion rules (missing data, open-space dominance, water dominance)
- Standardizes features robustly (z-features)
- Clusters BGs into peer groups and ranks clusters using **LocationScore**
- Produces an interactive **Cluster Ranking Map**

**Outputs (examples):**
- `outputs/BG_ECON_CLUSTER_OUT/`
- `outputs/BG_ECON_CLUSTER_OUT/bg_rank_map.html`

---

### Stage 2 — Monte Carlo stability testing  
**Script:** `scripts/GitHubMonteCarlo.py`

**Purpose:** Quantify how consistently BGs appear in top-ranked groups across repeated perturbations.

**Key steps:**
- Reruns clustering under bootstrap sampling and controlled noise
- Computes pass-rate / frequency summaries for each BG
- Produces stability tables used downstream for candidate screening

**Outputs (examples):**
- `outputs/.../MC_OUT/`

---

### Stage 3 — Stress-Test Ladder + anchor-aware selection + twins  
**Script:** `scripts/GitHubStressTest.py`

**Purpose:** Convert stable candidates into an operationally realistic shortlist across a strict→relaxed feasibility ladder, then enforce anchor-aware selection and interpretability.

**Key steps:**
- Evaluates feasibility filters across multiple regimes (strict → relaxed)
- Aggregates candidates using **earliest regime** and **frequency**
- Enforces spacing relative to the anchor footprint
- Assigns a “twin” anchor to each candidate and records similarity
- Produces an interactive **Aggregated Stress-Test Map** with anchors and candidate relationships

**Outputs (examples):**
- `outputs/RUN_*/STRESS_TEST_LADDER_BG/ALL_T_aggregated_map_bg.html`

---

### Stage 4 — Publish maps to GitHub Pages  
**Script:** `scripts/publish_maps.py`

**Purpose:** Publish generated HTML maps to a browser-viewable site.

**Key steps:**
- Copies latest maps into `docs/maps/`
- Enables viewing through GitHub Pages (GitHub repo browser does not render Folium maps)

**Outputs:**
- `docs/maps/*.html`

---

## Data included (what’s in `data/sample/`)

A small sample dataset is included so the pipeline runs end-to-end without external downloads.

**Core dataset:** `data/sample/bg_features_nyc.csv` (one row per BG) includes:
- BG identifier: `GEOID_BG`
- centroid geometry: `intpt_lat`, `intpt_lon`
- engineered features (raw):
  - `share_college_plus`, `share_commute_60p`, `pop_density`, `occ_units_density`, `potbus_per_1k`, `median_income`
- optional robust z-scores: `<feature>_z` (recommended)
- optional missing-data exclusion flag: `Exclusion` (`Yes`/`No`)

Optional geospatial layers (parks/open-space and boundary polygons) can be configured via `config.yaml` for richer exclusion logic and more accurate polygon rendering.

---

## Tunable parameters (centralized and documented)

Key assumptions are intended to be tuned without editing core scripts.

- `config.yaml` — central configuration for thresholds, weights, spacing rules, and run counts  
- `PARAMETERS.md` — parameter reference (defaults + what each knob changes)

Typical tunable categories include:
- **Feature weights** used in LocationScore (how strongly each signal influences ranking)
- **Exclusion thresholds** (missing-data handling, open-space share, water share, density cutoffs)
- **Clustering settings** (number of clusters, scaling choices, seed/reproducibility)
- **Monte Carlo settings** (number of runs, noise level, bootstrap fraction, top-rank definition)
- **Stress-test ladder regimes** (how strict filters are at each step, minimum selection counts)
- **Anchor spacing rules** (minimum distance from anchors and/or between candidates)
- **Twin-matching settings** (distance metric and feature set used for candidate↔anchor similarity)

Many parameters can also be overridden via environment variables.

Example:
```bash
N_RUNS=200 TOP_RANKS=3 python scripts/GitHubMonteCarlo.py
