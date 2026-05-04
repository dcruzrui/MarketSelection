# Tunable Parameters Reference

This repo is designed so reviewers (and you) can quickly understand what can be tuned and what the knobs do.

**Recommended workflow**
1) Edit `config.yaml` (preferred)
2) Use environment variables for quick experiments
3) Only edit script constants if you are changing core logic

Environment variables override `config.yaml`.

---

## Global path knobs

These determine where data is read from and where outputs go.

| Parameter | Default | Where used | What it does |
|---|---:|---|---|
| `DATA_DIR` | `data/sample` | all scripts | Base directory for inputs |
| `OUTPUTS_DIR` | `outputs` | all scripts | Where your run outputs are written |
| `BG_FEATURES_CSV` / `BG_DATA` | `data/sample/bg_features_nyc.csv` | ClusterMap / MonteCarlo / StressTest | Main BG features table |
| `EXCLUDED_BG_CSV` | auto | MonteCarlo / StressTest | External list of BGs removed from the universe |

---

## Stage 1: Cluster + Rank (`scripts/GitHubClusterMapNYC.py`)

### Data exclusion + masking

| Parameter | Default | What it does |
|---|---:|---|
| `APPLY_OPENSPACE_MASK` | `1` | If 1, computes `open_space_share` and excludes park/open-space dominated BGs |
| `OPENSPACE_SHARE_MIN_TO_EXCLUDE` | `0.60` | Exclude if open-space share ≥ threshold |
| `REQUIRE_OPENSPACE_MASK` | `1` | If 1 and open-space file missing, script will warn/stop (safety check) |
| `APPLY_WATER_MASK` | `1` | If 1, treats water-dominant BGs as excluded from ranking |
| `LAND_SHARE_BG_MIN` | `0.10` | Water-dominant if land_share_bg < threshold (rendered transparent on map) |

### Clustering / search

| Parameter | Default | What it does |
|---|---:|---|
| `RANDOM_SEED` | `42` | Reproducibility for splits, PCA, KMeans |
| `VAL_FRAC` | `0.25` | Holdout fraction used to score candidate clustering settings |
| `PCA_COMPONENTS_MAX` | `6` | Upper bound on PCA components |
| `K_VALUES` | `[5]` | KMeans K values to test |
| `N_SOBOL` | `80` | Number of Sobol samples for weighting search |
| `MIN_CLUSTER_FRAC` | `0.01` | Small-cluster penalty threshold |
| `DBI_PENALTY_W` | `0.50` | Penalize high Davies–Bouldin index solutions |
| `SMALLCLUST_PENALTY_W` | `0.50` | Penalize tiny clusters |
| `IMBALANCE_PENALTY_W` | `0.10` | Penalize imbalance in cluster sizes |

### Scoring / map appearance

| Parameter | Default | What it does |
|---|---:|---|
| `MAP_TITLE` | `"NYC Block Group Suitability Map (Cluster-Based Ranking)"` | Map title shown in the HTML |
| `BG_WEIGHTS` | see script/config | Weights used to compute LocationScore and rank clusters |

---

## Stage 2: Monte Carlo stability (`scripts/GitHubMonteCarlo.py`)

### Baseline clustering (Part 1)

| Parameter | Default | What it does |
|---|---:|---|
| `HAVE_Z_COLS` | `True` | If False, robust z-scores are created from raw engineered features |
| `SOBOL_SAMPLES` | `32` | Sobol samples for weighting search |
| `K_RANGE` | `[6,8,10,12]` | Candidate K values for KMeans |
| `TRAIN_FRAC` | `0.75` | Fit fraction (rest used for validation scoring) |
| `MIN_CLUSTER_SIZE_FRAC` | `0.012` | Minimum cluster fraction threshold |
| `MIN_CLUSTER_SIZE_MIN` | `40` | Absolute minimum cluster size |

### Monte Carlo stability (Part 2)

| Parameter | Default | What it does |
|---|---:|---|
| `N_RUNS` | `100` | Number of perturbation runs |
| `TOP_RANKS` | `2` | BG is “selected” if it lands in clusters ranked 1..TOP_RANKS |
| `BOOTSTRAP_FRAC` | `0.75` | Bootstrap sample fraction per run |
| `NOISE_STD` | `0.03` | Std dev of noise injected into z-features |
| `CHECKPOINT_EVERY` | `10` | Write checkpoint every N runs |

### Anchors (public demo)

| Parameter | Default | What it does |
|---|---:|---|
| `EXCLUDE_ANCHOR_BGS` | `True` | Removes the (synthetic) anchor BGs from the candidate universe |

---

## Stage 3: Stress-Test Ladder + Twins (`scripts/GitHubStressTest.py`)

This stage runs a strict→relaxed sweep over `t` and aggregates results.

### Ladder size + global caps

| Parameter | Default | What it does |
|---|---:|---|
| `T_STEPS` (count) | `100` | Number of ladder scenarios |
| `MAX_SITES_PER_SCENARIO` | `15` | Max candidates kept per scenario |

### Schedule knobs (strict at t=0, relaxed as t increases)

The code uses simple schedules such as:
- `MIN_PASS_RATE_BG(t)` (decreases with t)
- `MIN_SELECT_COUNT_BG(t)` (decreases with t)
- `MIN_STORE_BUFFER_MI(t)` (decreases with t)
- `MAX_SEARCH_MI(t)` (increases with t)

Tune these via `config.yaml` (preferred) or by editing the top-of-file schedule section.

### Twins + adjacency

| Parameter | Default | What it does |
|---|---:|---|
| `TWINS_ALPHA` | `0.05` | Tail probability used to define a twin threshold (q = 1 - alpha) |
| `REQUIRE_ADJACENT_TWIN` | `True` | If True, candidate must have adjacent twin support |
| `MIN_ADJACENT_TWINS` | `1` | Minimum adjacent twins required |
| `BG_SHP_PATH` | `data/sample/nyc_block_groups_2021.shp` | If present, enables queen adjacency and twin shading |

### Reverse geocoding (optional)

| Parameter | Default | What it does |
|---|---:|---|
| `DO_REVERSE_GEOCODE` | `1` | If 1, look up human-readable addresses for tooltips (cached) |
| `GEOCODE_TIMEOUT_S` | `6` | Request timeout |
| `GEOCODE_SLEEP_S` | `1.0` | Delay between requests |

---

## Quick tuning recipes

### Make Monte Carlo more convincing (slower)
```bash
N_RUNS=300 NOISE_STD=0.02 python scripts/GitHubMonteCarlo.py
```

### Make selection stricter (fewer candidates)
```bash
TOP_RANKS=1 python scripts/GitHubMonteCarlo.py
```

### Turn off open-space exclusions (if you do not have parks data)
```bash
APPLY_OPENSPACE_MASK=0 python scripts/GitHubClusterMapNYC.py
```

---

## Where to change defaults

- Prefer: edit `config.yaml`
- Fast test: use environment variables
- Deep changes: edit the “TUNABLE PARAMETERS” blocks at the top of each script
