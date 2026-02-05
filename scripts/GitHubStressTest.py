# -*- coding: utf-8 -*-
"""
GitHubStressTest_UPDATED.py
==========================

Goal
----
Make the *GitHubStressTest* (BG) script mirror the behavior and outputs of the
block-level stress-test ladder + aggregated map + twin clusters + A↔B composite
similarity workflow from:

    NewUpdatedStressTest11082025.py

The ONLY intended differences vs the block-level script are:
1) We operate at Block Group level (BG_ID),
2) We use BG centroid coordinates (default: intpt_lat/intpt_lon),
3) Existing-store spacing uses "anchors" (synthetic by default) instead of FinalData-fedex detection.

What this script now does (mirrors the block-level flow)
--------------------------------------------------------
Phase 1: 100-regime ladder (strict → relaxed)
- Per-t output folders:
    ladder_t_<t>/
        kept_blocks.csv      (BG rows; name preserved for parity)
        metrics.json
- Summary:
    summary_table.csv

Phase 2: Aggregation across all T
- ALL_T_aggregated_blockgroups.csv
- final_shortlist_bg.csv

Phase 3A/3B: Twin nests (Mahalanobis) for
- Section A: candidate anchors (aggregated kept BGs)
- Section B: store anchors (existing footprint; synthetic anchors snapped to BGs by default)

Phase 4: A↔B Composite Similarity
- sectionA_profiles.csv
- sectionB_profiles.csv
- AB_component_breakdown.csv
- AB_similarity_matches.csv

Map
---
- ALL_T_aggregated_map_bg.html
  - dots for aggregated candidates, sized by frequency
  - color encodes earliest t (strictness)
  - popups show Top-1 twin store + similarity + component breakdown
  - optional: twin clusters overlay with rays

Notes
-----
- If you have BG polygon geometries and geopandas installed, you can set BG_SHP_PATH to either a .shp or a .zip.
  Queen adjacency is used if available; otherwise we use proximity adjacency (BallTree radius).
- Reverse geocoding is optional (disabled by default for GitHub portability).

"""

import os, re, json, math, time, glob, argparse
from pathlib import Path
from datetime import datetime
from statistics import NormalDist

import numpy as np
import pandas as pd

# sklearn is required for BallTree + scaling/covariance models
from sklearn.neighbors import BallTree
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import MinCovDet, LedoitWolf

# folium is required for the HTML map
import folium
import branca
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
from matplotlib.colors import LinearSegmentedColormap, Normalize

try:
    import requests
except Exception:
    requests = None

# Optional (better adjacency if you have a shapefile)
try:
    import geopandas as gpd
except Exception:
    gpd = None


# ============================================================
# USER CONFIG
# ============================================================

# Paths (GitHub-friendly defaults)
# Override via env vars if you want to point at your own data/output folders:
#   DATA_DIR=/path/to/data  OUTPUTS_DIR=/path/to/outputs  python scripts/GitHubStressTest.py
REPO_ROOT   = Path(__file__).resolve().parents[1]
CFG         = load_repo_config(REPO_ROOT)

DATA_DIR    = resolve_repo_path(REPO_ROOT, env_or_cfg("DATA_DIR", CFG, "paths.data_dir", "data/sample"), REPO_ROOT / "data" / "sample")
OUTPUTS_DIR = resolve_repo_path(REPO_ROOT, env_or_cfg("OUTPUTS_DIR", CFG, "paths.outputs_dir", "outputs"), REPO_ROOT / "outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep BASE_DIR as the *data* directory for backwards compatibility with other variables below.
BASE_DIR = os.environ.get("BASE_DIR", str(DATA_DIR))

BG_FEATURES_CSV = os.environ.get("BG_FEATURES_CSV", str(DATA_DIR / "bg_features_nyc.csv"))



# BG MonteCarlo output folder (must contain mc_counts_bg.csv)
# Priority:
#  1) env var BG_MC_DIR (explicit)
#  2) outputs/BG_FINALWITHCOUNT_OUT (if it exists)
#  3) examples/outputs/BG_FINALWITHCOUNT_OUT (bundled sample)
_examples_mc = REPO_ROOT / "examples" / "outputs" / "BG_FINALWITHCOUNT_OUT"
_default_mc  = OUTPUTS_DIR / "BG_FINALWITHCOUNT_OUT"
BG_MC_DIR    = os.environ.get("BG_MC_DIR", str(_default_mc if _default_mc.exists() else _examples_mc))
MC_COUNTS_CSV = os.path.join(BG_MC_DIR, "mc_counts_bg.csv")

# (Optional) BG exclusion list:
#   Any BG listed here is removed from the universe at the very beginning (features + MC + everything downstream).
# Priority:
#  1) env var EXCLUDED_BG_CSV (explicit)
#  2) outputs/BG_ECON_CLUSTER_OUT/excluded_blockgroups.csv (if it exists)
#  3) examples/outputs/BG_ECON_CLUSTER_OUT/excluded_blockgroups.csv (bundled sample)
_default_excl = OUTPUTS_DIR / "BG_ECON_CLUSTER_OUT" / "excluded_blockgroups.csv"
_examples_excl = REPO_ROOT / "examples" / "outputs" / "BG_ECON_CLUSTER_OUT" / "excluded_blockgroups.csv"
EXCLUDED_BG_CSV = os.environ.get(
    "EXCLUDED_BG_CSV",
    str(_default_excl if _default_excl.exists() else _examples_excl),
)


# (Optional) BG polygon geometry source to enable Queen adjacency + shaded twin areas.
# You can point to either:
#   - a .shp path, OR
#   - a .zip containing a shapefile set (.shp/.dbf/.shx/.prj)
# If None or missing, proximity adjacency is used instead and twin areas cannot be shaded.
BG_SHP_PATH = str(resolve_repo_path(REPO_ROOT, env_or_cfg("BG_SHP_PATH", CFG, "paths.bg_shp_path", DATA_DIR / "nyc_block_groups_2021.shp"), DATA_DIR / "nyc_block_groups_2021.shp"))

# --- Water-handling for twin shading ---
# BG polygons (especially near coasts/rivers) can include large water portions. Since we may not have
# a separate land-mask layer in the GitHub sample package, we optionally *exclude* mostly-water BGs
# from the *twin shading overlay* using ALAND/AWATER attributes in the BG shapefile DBF.
TWIN_SHADE_EXCLUDE_WATER   = True
TWIN_SHADE_LAND_SHARE_MIN  = 0.20  # keep if ALAND/(ALAND+AWATER) >= this (and ALAND>0)


# Output root
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT  = os.path.join(str(OUTPUTS_DIR), f"RUN_{RUN_STAMP}")
LADDER_DIR_ROOT = os.path.join(RUN_ROOT, "STRESS_TEST_LADDER_BG")
os.makedirs(LADDER_DIR_ROOT, exist_ok=True)

# ID + coordinates
BG_ID   = "GEOID_BG"
LAT_COL = "intpt_lat"
LON_COL = "intpt_lon"

EARTH_RADIUS_MI = 3958.7613

# -------------------------
# Ladder steps (mirror block-level: endpoint=False)
# -------------------------
T_STEPS_N = env_or_cfg_int("T_STEPS_N", CFG, "stress_test.t_steps", 100)
T_STEPS = np.linspace(0.0, 1.0, int(T_STEPS_N), endpoint=False)
TOTAL_SCENARIOS = len(T_STEPS)


# Inferred from mc_counts_bg.csv at runtime (used to scale MIN_SELECT_COUNT_BG)
ELIGIBLE_RUNS_REF_BG = None

# -------------------------
# Schedule parameters (strict at t=0, relaxed as t increases)
# -------------------------
FDR_ALPHA_START = env_or_cfg_float("FDR_ALPHA_START", CFG, "stress_test.schedules.fdr_alpha_start", 0.01)
FDR_ALPHA_END   = env_or_cfg_float("FDR_ALPHA_END",   CFG, "stress_test.schedules.fdr_alpha_end",   0.10)

MIN_PASS_RATE_START = env_or_cfg_float("MIN_PASS_RATE_START", CFG, "stress_test.schedules.min_pass_rate_start", 0.70)
MIN_PASS_RATE_END   = env_or_cfg_float("MIN_PASS_RATE_END",   CFG, "stress_test.schedules.min_pass_rate_end",   0.40)

MIN_SELECT_START = env_or_cfg_int("MIN_SELECT_START", CFG, "stress_test.schedules.min_select_start", 20)
MIN_SELECT_END   = env_or_cfg_int("MIN_SELECT_END",   CFG, "stress_test.schedules.min_select_end",   10)

DEMAND_TOP   = env_or_cfg_int("DEMAND_TOP",   CFG, "stress_test.schedules.demand_top",   10000)
DEMAND_FLOOR = env_or_cfg_int("DEMAND_FLOOR", CFG, "stress_test.schedules.demand_floor", 4000)
INV_P        = env_or_cfg_float("INV_P", CFG, "stress_test.schedules.inv_p", 1.0)
INV_K        = env_or_cfg_float("INV_K", CFG, "stress_test.schedules.inv_k", 3.0)

MAX_SEARCH_MI_BASE  = env_or_cfg_float("MAX_SEARCH_MI_BASE",  CFG, "stress_test.schedules.max_search_mi_base",  1.5)
MAX_SEARCH_MI_SCALE = env_or_cfg_float("MAX_SEARCH_MI_SCALE", CFG, "stress_test.schedules.max_search_mi_scale", 1.0)

MIN_STORE_BUFFER_START = env_or_cfg_float("MIN_STORE_BUFFER_START", CFG, "stress_test.schedules.min_store_buffer_start", 2.0)
MIN_STORE_BUFFER_END   = env_or_cfg_float("MIN_STORE_BUFFER_END",   CFG, "stress_test.schedules.min_store_buffer_end",   0.8)

MIN_RING_FALLBACK_START = env_or_cfg_float("MIN_RING_FALLBACK_START", CFG, "stress_test.schedules.min_ring_fallback_start", 1.0)
MIN_RING_FALLBACK_END   = env_or_cfg_float("MIN_RING_FALLBACK_END",   CFG, "stress_test.schedules.min_ring_fallback_end",   0.5)

# Mirror-style schedules (BG analogs; shape mirrors the original ladder)
def FDR_ALPHA(t):
    return float(FDR_ALPHA_START + (FDR_ALPHA_END - FDR_ALPHA_START) * t)
def MIN_PASS_RATE_BG(t):
    return float(MIN_PASS_RATE_START + (MIN_PASS_RATE_END - MIN_PASS_RATE_START) * t)
def MIN_SELECT_COUNT_BG(t):
    """Absolute minimum select-count threshold (BG level).

    With 20 Monte Carlo runs, the select_count_bg ranges 0..20. We mirror the user's
    schedule request: a linear floor that relaxes from 20 → 10 across t∈[0,1].
    """
    return int(round(MIN_SELECT_START + (MIN_SELECT_END - MIN_SELECT_START) * t))

# demand schedule constants come from config.yaml (see above)
def TARGET_DEMAND(t):  return int(round(DEMAND_FLOOR + (DEMAND_TOP - DEMAND_FLOOR) / ((1 + INV_K * t) ** INV_P)))
def MAX_SEARCH_MI(t):
    return float(MAX_SEARCH_MI_BASE + MAX_SEARCH_MI_SCALE * math.sqrt(t))
def MIN_STORE_BUFFER_MI(t):
    return float(MIN_STORE_BUFFER_START + (MIN_STORE_BUFFER_END - MIN_STORE_BUFFER_START) * t)
def MIN_RING_MI_FALLBACK(t):
    return float(MIN_RING_FALLBACK_START + (MIN_RING_FALLBACK_END - MIN_RING_FALLBACK_START) * t)

MAX_SITES_PER_SCENARIO = env_or_cfg_int("MAX_SITES_PER_SCENARIO", CFG, "stress_test.max_sites_per_scenario", 15)

# -------------------------
# Map styling toggles (mirror block-level knobs)
# -------------------------
MAKE_PER_T_LAYERS   = False
SHOW_COUNT_LABELS   = False
SCALE_RING_BY_FREQ  = True
RING_MAX_SCALE      = 2.0
DOT_MIN_RADIUS      = 4.0
DOT_MAX_RADIUS      = 12.0

# Twin layer toggles
INCLUDE_TWIN_CLUSTERS_ALL = True
TWINS_ALPHA = env_or_cfg_float("TWINS_ALPHA", CFG, "stress_test.twins.alpha", 0.05)  # q = 1-alpha
TWINS_LOCAL_BY_BORO = True
MAX_TWIN_ANCHORS    = None      # e.g., 100
MIN_TWIN_CLUSTER_UNITS = 1

# In the original block-level script, the aggregated map can optionally require
# that candidates have at least one adjacent twin (by the twin threshold q).
REQUIRE_ADJACENT_TWIN = env_or_cfg_bool("REQUIRE_ADJACENT_TWIN", CFG, "stress_test.twins.require_adjacent_twin", True)
MIN_ADJACENT_TWINS = env_or_cfg_int("MIN_ADJACENT_TWINS", CFG, "stress_test.twins.min_adjacent_twins", 1)

# Composite similarity weights (same as block script)
AB_WEIGHTS = {
    "comp":  0.35,
    "disp":  0.20,
    "scale": 0.15,
    "shape": 0.15,
    "anchor":0.15
}
BORO_DIFFERENT_PENALTY = 0.02

# Colormap (same palette as block script)
STRICT_COLORS = ["#0b3d91", "#2c7fb8", "#41b6c4", "#a1dab4", "#ffffbf", "#fdae61", "#d7191c"]
MPL_CMAP = LinearSegmentedColormap.from_list("strictness", STRICT_COLORS)
MPL_NORM = Normalize(vmin=0.0, vmax=1.0)

# Reverse geocoding (optional; used to display addresses in map tooltips)
# Set DO_REVERSE_GEOCODE=0 to disable.
DO_REVERSE_GEOCODE = env_or_cfg_bool("DO_REVERSE_GEOCODE", CFG, "stress_test.reverse_geocode.enabled", True)
GEOCODE_TIMEOUT_S  = env_or_cfg_int("GEOCODE_TIMEOUT_S", CFG, "stress_test.reverse_geocode.timeout_s", 6)
GEOCODE_SLEEP_S    = env_or_cfg_float("GEOCODE_SLEEP_S", CFG, "stress_test.reverse_geocode.sleep_s", 1.0)
GEOCODE_CACHE_JSON = os.path.join(BASE_DIR, "geocode_cache.json")

# If no shapefile, proximity adjacency is used:
PROX_ADJ_RADIUS_MI = env_or_cfg_float("PROX_ADJ_RADIUS_MI", CFG, "stress_test.adjacency.prox_adj_radius_mi", 0.35)   # tweak if you want denser/sparser neighbor graph for "contiguity"
MAX_CLUSTER_RADIUS_MI = 1.20
RAY_MAX_RADIUS_MI     = MAX_CLUSTER_RADIUS_MI

# Covariance stabilization (mirrors the block-level defaults closely)
BORO_GLOBAL_MIX        = 0.10
TWINS_COV_RIDGE        = 0.15
TWINS_COV_DIAGFLOOR    = 1e-3
TWINS_USE_DIAG         = True
DEFAULT_Q              = 0.85
TWINS_CHI_CAP          = None    # if set, caps threshold by chi-square quantile
GAMMA_LOCAL            = 0.95

# Synthetic anchors (Option A) — used both for spacing (existing footprint) and Section B profiles.
ANCHORS_15 = [
    (40.7000, -73.9900),
    (40.7075, -73.9550),
    (40.7150, -73.9750),
    (40.7220, -73.9300),
    (40.7300, -73.9950),
    (40.7370, -73.9600),
    (40.7450, -73.9100),
    (40.7520, -73.9850),
    (40.7600, -73.9450),
    (40.7680, -73.9050),
    (40.7760, -73.9800),
    (40.7840, -73.9400),
    (40.7920, -73.9000),
    (40.8000, -73.9650),
    (40.8080, -73.9250),
]


# ============================================================
# Helpers
# ============================================================

def _haversine_miles(lat, lon, lat2, lon2):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    phi1 = np.radians(lat); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat)
    dl   = np.radians(lon2 - lon)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2.0)**2
    return 2.0 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))

def _haversine_pair(lat1, lon1, lat2, lon2) -> float:
    return float(_haversine_miles(lat1, lon1, np.array([lat2]), np.array([lon2]))[0])

def _bearing_deg(lat1, lon1, lat2, lon2):
    y = math.sin(math.radians(lon2-lon1))*math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1))*math.sin(math.radians(lat2)) - math.sin(math.radians(lat1))*math.cos(math.radians(lat2))*math.cos(math.radians(lon2-lon1))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

_OCTANTS = ["N","NE","E","SE","S","SW","W","NW"]


# Center bearing (degrees) for each octant (used for ray-walk projections)
_OCT_CENTER = {'N':0.0,'NE':45.0,'E':90.0,'SE':135.0,'S':180.0,'SW':225.0,'W':270.0,'NW':315.0}

def _octant_of(bearing):
    edges = [22.5,67.5,112.5,157.5,202.5,247.5,292.5,337.5]
    for edge, name in zip(edges, _OCTANTS):
        if bearing <= edge:
            return name
    return "N"

def _ensure_bg_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if BG_ID not in out.columns:
        if "GEOID" in out.columns:
            out[BG_ID] = out["GEOID"].astype(str)
        elif "GEOIDFQ" in out.columns:
            out[BG_ID] = out["GEOIDFQ"].astype(str).str.replace("1500000US", "", regex=False)
        else:
            raise ValueError(f"Missing {BG_ID} (or GEOID/GEOIDFQ).")
    out[BG_ID] = out[BG_ID].astype(str)
    return out


def _canon_bg_id(v) -> str:
    """
    Canonicalize a BG id into a 12-digit GEOID string (digits only).
    Handles common variants like GEOIDFQ (1500000US...), floats, or scientific notation artifacts.
    """
    if v is None:
        return ""
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return ""
    # common Census prefix
    if s.startswith("1500000US"):
        s = s.replace("1500000US", "")
    # keep digits only (this also removes commas/spaces/etc.)
    digits = re.sub(r"\D", "", s)
    if not digits:
        return ""
    # if we accidentally captured the prefix digits too, keep the last 12 (the actual GEOID)
    if len(digits) > 12:
        digits = digits[-12:]
    # if it's shorter (rare), left-pad to 12
    if len(digits) < 12:
        digits = digits.zfill(12)
    return digits


def _load_excluded_bg_ids(path_csv: str) -> set:
    """
    Load excluded BG ids from a CSV (any reasonable id column name).
    Returns a set of canonical 12-digit GEOID strings.
    """
    try:
        ex = pd.read_csv(path_csv, dtype=str)
    except Exception as e:
        print(f"(note) Could not read EXCLUDED_BG_CSV at {path_csv}: {e}")
        return set()

    if ex is None or getattr(ex, "empty", False) or len(ex.columns) == 0:
        return set()

    cand_cols = [
        BG_ID, "GEOID_BG", "geoid_bg",
        "GEOID", "geoid",
        "GEOIDFQ", "geoidfq",
        "block_group", "bg", "BG",
    ]
    id_col = next((c for c in cand_cols if c in ex.columns), ex.columns[0])

    out = set()
    for v in ex[id_col].dropna().tolist():
        cv = _canon_bg_id(v)
        if cv:
            out.add(cv)
    return out


def _coerce_latlon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LAT_COL] = pd.to_numeric(out.get(LAT_COL, np.nan), errors="coerce")
    out[LON_COL] = pd.to_numeric(out.get(LON_COL, np.nan), errors="coerce")
    out = out.dropna(subset=[LAT_COL, LON_COL]).reset_index(drop=True)
    return out

def _infer_borough_from_geoid_bg(geoid: str) -> str:
    """
    NYC county FIPS in GEOID (BG):
      36005 Bronx, 36047 Brooklyn, 36061 Manhattan, 36081 Queens, 36085 Staten Island
    In GEOID_BG: first 2=state, next 3=county.
    """
    s = str(geoid)
    if len(s) >= 5:
        county = s[2:5]
        return {"005":"Bronx","047":"Brooklyn","061":"Manhattan","081":"Queens","085":"Staten Island"}.get(county, "Unknown")
    return "Unknown"

def _ensure_borough_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for cand in ["borough","BoroName","BORONAME","boro","boro_name","BORO_NM"]:
        if cand in out.columns:
            out = out.rename(columns={cand:"borough"})
            out["borough"] = out["borough"].astype(str)
            return out
    out["borough"] = out[BG_ID].astype(str).apply(_infer_borough_from_geoid_bg)
    return out


def _filter_bg_polys_for_land_shading(polys_gdf: "pd.DataFrame", land_share_min: float = 0.20):
    """Filter BG polygons to reduce water shading.

    Uses ALAND/AWATER (or common variants) if present. This is *not* a true land clip; it simply
    removes polygons that are pure-water or mostly-water from the *twin overlay shading layer*.

    - Always removes ALAND==0 when a land area field exists.
    - If AWATER exists too, also enforces land_share >= land_share_min.
    - If expected fields are missing, returns polys_gdf unchanged.
    """
    if polys_gdf is None:
        return None

    cols = list(getattr(polys_gdf, "columns", []))
    land_candidates = ["ALAND", "ALAND20", "ALAND_BG", "ALANDCE", "aland", "aland20", "aland_bg"]
    water_candidates = ["AWATER", "AWATER20", "AWATER_BG", "AWATERCE", "awater", "awater20", "awater_bg"]

    land_col = next((c for c in land_candidates if c in cols), None)
    water_col = next((c for c in water_candidates if c in cols), None)

    # No usable fields -> no filtering.
    if land_col is None and water_col is None:
        return polys_gdf

    df = polys_gdf.copy()

    land = pd.to_numeric(df[land_col], errors="coerce").fillna(0.0) if land_col else pd.Series(0.0, index=df.index)
    water = pd.to_numeric(df[water_col], errors="coerce").fillna(0.0) if water_col else pd.Series(0.0, index=df.index)

    if water_col:
        total = (land + water)
        denom = total.where(total > 0, 1.0)
        land_share = (land / denom)
        mask = (land > 0) & (land_share >= float(land_share_min))
    else:
        # If no AWATER, we can still safely drop pure-water units (ALAND==0).
        mask = (land > 0)

    try:
        out = df.loc[mask].copy()
        # If we accidentally filtered everything, fall back to original (better to shade than show nothing).
        if getattr(out, "empty", False):
            return polys_gdf
        return out
    except Exception:
        return polys_gdf


def _get_potential_bus(df: pd.DataFrame) -> pd.Series:
    cols = ["PotentialBus","PotentBus","potential_bus","potbus","potbus_count","Potential_Bus"]
    for c in cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # last resort: potbus_per_1k * pop/1000
    if "potbus_per_1k" in df.columns:
        pop_cols = ["P1_total","P1_001N","total_population","pop_total","population","pop"]
        for pc in pop_cols:
            if pc in df.columns:
                pop = pd.to_numeric(df[pc], errors="coerce").fillna(0.0)
                return pd.to_numeric(df["potbus_per_1k"], errors="coerce").fillna(0.0) * (pop/1000.0)
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

def _snap_anchors_to_bgs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Snap the synthetic anchors to the nearest BG centroid in (lat,lon).
    Returns a unique BG list (duplicates removed).
    """
    pts = df[[BG_ID, LAT_COL, LON_COL]].dropna().copy()
    A = pts[[LAT_COL, LON_COL]].astype(float).values
    ids = pts[BG_ID].astype(str).values
    chosen = []
    for i, (alat, alon) in enumerate(ANCHORS_15):
        d2 = (A[:, 0] - alat)**2 + (A[:, 1] - alon)**2
        chosen.append((i, ids[int(np.argmin(d2))], alat, alon))
    out = pd.DataFrame(chosen, columns=["anchor_idx", BG_ID, "anchor_lat", "anchor_lon"])
    out = out.drop_duplicates(subset=[BG_ID]).reset_index(drop=True)
    return out

def _precompute_neighbors(lat_s: pd.Series, lon_s: pd.Series, w_s: pd.Series, max_radius_mi: float):
    """
    For each BG i, precompute neighbor distances (mi) and cumulative weights (PotentialBus)
    within max_radius_mi. This lets us recompute demand rings quickly for every t.
    """
    lat = lat_s.astype(float).to_numpy()
    lon = lon_s.astype(float).to_numpy()
    w   = w_s.astype(float).to_numpy()

    coords = np.deg2rad(np.c_[lat, lon])
    tree = BallTree(coords, metric="haversine")
    rad = max_radius_mi / EARTH_RADIUS_MI

    ind, dist = tree.query_radius(coords, r=rad, return_distance=True, sort_results=True)

    nei_dists, nei_cumw = [], []
    for i in range(len(lat)):
        idx = ind[i]
        di  = dist[i] * EARTH_RADIUS_MI
        # drop self if present
        if len(idx) > 0 and idx[0] == i:
            idx = idx[1:]
            di  = di[1:]
        wi = w[idx] if len(idx) else np.array([], dtype=float)
        cw = np.cumsum(wi) if len(wi) else np.array([], dtype=float)
        nei_dists.append(di.astype(float))
        nei_cumw.append(cw.astype(float))
    return nei_dists, nei_cumw

def _rings_from_precomputed(target_demand: float, cap_mi: float, nei_dists, nei_cumw):
    """
    Find r such that cumulative PotentialBus reaches target_demand within cap_mi.
    """
    r = np.full(len(nei_dists), np.nan, dtype=float)
    for i in range(len(nei_dists)):
        d  = nei_dists[i]
        cw = nei_cumw[i]
        if d.size == 0:
            continue
        j_cap = np.searchsorted(d, cap_mi, side="right") - 1
        if j_cap < 0:
            continue
        cw_cap = cw[:j_cap+1]
        k = np.searchsorted(cw_cap, target_demand, side="left")
        if k < len(cw_cap):
            r[i] = float(d[k])
    return r

def _write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _write_run_manifest(extra: dict):
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "BASE_DIR": BASE_DIR,
        "RUN_ROOT": RUN_ROOT,
        "LADDER_DIR_ROOT": LADDER_DIR_ROOT,
        "BG_FEATURES_CSV": BG_FEATURES_CSV,
        "MC_COUNTS_CSV": MC_COUNTS_CSV,
        "BG_SHP_PATH": BG_SHP_PATH,
        "TOTAL_SCENARIOS": TOTAL_SCENARIOS,
        "LAT_COL": LAT_COL,
        "LON_COL": LON_COL,
        "anchors_15_latlon": ANCHORS_15,
    }
    meta.update(extra or {})
    _write_json(os.path.join(LADDER_DIR_ROOT, "run_manifest.json"), meta)


# ============================================================
# Twin features + adjacency
# ============================================================

_TWIN_FEAT_COLS = [
    "share_college_plus",
    "share_commute_60p",
    "pop_density",
    "occ_units_density",
    "comp_per_km2",
    "potbus_per_1k",
]

def _prep_twin_features_bg(bg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the six twin features used in the block script (BG analog).
    If the engineered cols already exist, we use them; otherwise we attempt to compute from raw ACS-like columns.
    """
    df = bg_df.copy()
    df = _ensure_bg_id(df)
    df = _ensure_borough_col(df)

    # Ensure PotentialBus exists
    if "PotentialBus" not in df.columns:
        df["PotentialBus"] = _get_potential_bus(df).astype(float)
    else:
        df["PotentialBus"] = pd.to_numeric(df["PotentialBus"], errors="coerce").fillna(0.0).astype(float)

    # If all engineered columns exist, just proceed to z-scores
    have_all = all(c in df.columns for c in _TWIN_FEAT_COLS)

    if not have_all:
        # Education: college+
        edu_num_cols = ["B15003_022","B15003_023","B15003_024","B15003_025"]
        for c in edu_num_cols + ["B15003_001"]:
            if c not in df.columns:
                df[c] = 0
        edu_college = sum(pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in edu_num_cols)
        edu_denom   = pd.to_numeric(df["B15003_001"], errors="coerce").replace(0, np.nan)
        df["share_college_plus"] = (edu_college / edu_denom).fillna(0.0)

        # Commute: 60+ minutes share (B08303_013 is often 60-89)
        for c in ["B08303_001","B08303_013"]:
            if c not in df.columns:
                df[c] = 0
        cm_denom = pd.to_numeric(df["B08303_001"], errors="coerce").replace(0, np.nan)
        df["share_commute_60p"] = (pd.to_numeric(df["B08303_013"], errors="coerce") / cm_denom).fillna(0.0)

        # Land area km^2
        land_cols = ["ALAND","ALAND20","ALAND_BG","aland","aland20"]
        land = None
        for lc in land_cols:
            if lc in df.columns:
                land = pd.to_numeric(df[lc], errors="coerce").fillna(0.0).astype(float)
                break
        if land is None:
            # fallback: treat each BG as 1 km^2 to avoid blow-ups
            land_km2 = pd.Series(np.ones(len(df)), index=df.index, dtype=float)
        else:
            land_km2 = land / 1e6
            land_km2 = land_km2.replace(0, np.nan)

        # Pop / occupied units
        pop_cols = ["P1_total","P1_001N","total_population","population","pop"]
        pop = None
        for pc in pop_cols:
            if pc in df.columns:
                pop = pd.to_numeric(df[pc], errors="coerce").fillna(0.0).astype(float)
                break
        if pop is None:
            if "pop_total" in df.columns:
                pop = pd.to_numeric(df["pop_total"], errors="coerce").fillna(0.0).astype(float)
            else:
                pop = pd.Series(0.0, index=df.index, dtype=float)

        occ_cols = ["H1_002N","occupied_units","occ_units","H1_occ"]
        occ = None
        for oc in occ_cols:
            if oc in df.columns:
                occ = pd.to_numeric(df[oc], errors="coerce").fillna(0.0).astype(float)
                break
        if occ is None:
            if "occ_total" in df.columns:
                occ = pd.to_numeric(df["occ_total"], errors="coerce").fillna(0.0).astype(float)
            else:
                occ = pd.Series(0.0, index=df.index, dtype=float)

        # Competition counts
        comp_cols = ["DualComp","PrintComp","ShipComp","dual_comp","print_comp","ship_comp","comp_total"]
        comp_total = None
        for cc in comp_cols:
            if cc in df.columns:
                # If comp_total exists, use directly; else sum the three if present.
                if cc == "comp_total":
                    comp_total = pd.to_numeric(df[cc], errors="coerce").fillna(0.0).astype(float)
                break
        if comp_total is None:
            comp_total = 0.0
            for cc in ["DualComp","PrintComp","ShipComp","dual_comp","print_comp","ship_comp"]:
                if cc in df.columns:
                    comp_total = comp_total + pd.to_numeric(df[cc], errors="coerce").fillna(0.0).astype(float)
            comp_total = pd.Series(comp_total, index=df.index, dtype=float)

        # Densities (log1p)
        df["pop_density"]       = np.log1p((pop / land_km2).fillna(0.0))
        df["occ_units_density"] = np.log1p((occ / land_km2).fillna(0.0))
        df["comp_per_km2"]      = np.log1p((comp_total / land_km2).fillna(0.0))

        # PotentialBus per 1k pop (log1p)
        denom_pop = pop.replace(0, np.nan)
        df["potbus_per_1k"] = (1000.0 * df["PotentialBus"] / denom_pop).fillna(0.0)
        df["potbus_per_1k"] = np.log1p(df["potbus_per_1k"])

    # z-score (robust scaling) for Mahalanobis space
    scaler = RobustScaler()
    Z = scaler.fit_transform(df[_TWIN_FEAT_COLS].astype(float))
    for j, c in enumerate(_TWIN_FEAT_COLS):
        df[c + "_z"] = Z[:, j]
    return df

def _stabilize_cov(cov, lam=TWINS_COV_RIDGE, diag_floor=TWINS_COV_DIAGFLOOR):
    p = cov.shape[0]
    tau = float(np.trace(cov)) / max(p, 1)
    cov = (1.0 - lam) * cov + lam * tau * np.eye(p)
    cov = cov + (diag_floor * tau) * np.eye(p)
    return cov

def _fit_cov_models(feat_df: pd.DataFrame, by_borough=True):
    """
    Fit GLOBAL + (optionally) borough-level covariance models in z-space.
    """
    zcols = [c + "_z" for c in _TWIN_FEAT_COLS]
    models = {}

    def _fit(X):
        try:
            mcd = MinCovDet().fit(X)
            cov = mcd.covariance_.astype(float)
        except Exception:
            try:
                cov = LedoitWolf().fit(X).covariance_.astype(float)
            except Exception:
                cov = np.cov(X, rowvar=False).astype(float)

        if TWINS_USE_DIAG:
            cov = np.diag(np.clip(np.diag(cov), 1e-12, None))

        cov = np.atleast_2d(cov) + 1e-9 * np.eye(cov.shape[0])
        cov = _stabilize_cov(cov)
        icov = np.linalg.inv(cov)
        return {"cov": cov, "icov": icov}

    Xg = feat_df[zcols].to_numpy(float)
    models["_GLOBAL_"] = _fit(Xg)

    if by_borough and "borough" in feat_df.columns:
        for b, g in feat_df.groupby("borough", dropna=False):
            X = g[zcols].to_numpy(float)
            if X.shape[0] >= 50:
                models[str(b)] = _fit(X)

        # mix borough cov with global cov for stability
        Cg = models["_GLOBAL_"]["cov"]
        for b in list(models.keys()):
            if b == "_GLOBAL_":
                continue
            Cb = models[b]["cov"]
            Cmix = (1.0 - BORO_GLOBAL_MIX) * Cb + BORO_GLOBAL_MIX * Cg
            Cmix = np.atleast_2d(Cmix) + 1e-9 * np.eye(Cmix.shape[0])
            Cmix = _stabilize_cov(Cmix)
            models[b] = {"cov": Cmix, "icov": np.linalg.inv(Cmix)}
    return models

def _queen_neighbors_dict_from_shp(bg_shp: "gpd.GeoDataFrame") -> dict:
    """
    Build Queen adjacency dict from polygons.

    We try a fast geopandas.sjoin (requires a spatial index like rtree/pygeos).
    If that fails, we fall back to a pure-shapely STRtree implementation so the
    script works in environments without a geopandas spatial index.
    """
    if gpd is None:
        raise RuntimeError("geopandas not available; cannot compute queen adjacency")

    WORK_CRS = "EPSG:2263"
    g = bg_shp[[BG_ID, "geometry"]].copy()
    g[BG_ID] = g[BG_ID].astype(str)

    # Project for robust planar intersection tests when CRS is known.
    try:
        gp = g.to_crs(WORK_CRS).copy() if getattr(g, "crs", None) is not None else g.copy()
    except Exception:
        gp = g.copy()

    # Fix invalid geometries
    try:
        gp["geometry"] = gp.geometry.buffer(0)
    except Exception:
        pass

    # Drop empties
    try:
        gp = gp[gp.geometry.notnull() & (~gp.geometry.is_empty)].copy()
    except Exception:
        gp = gp.dropna(subset=["geometry"]).copy()

    # 1) Fast path: sjoin (if spatial index is available)
    try:
        right = gp.rename(columns={BG_ID: "_RID"})
        pairs = gpd.sjoin(gp, right[["_RID", "geometry"]], how="inner", predicate="intersects")
        pairs = pairs[pairs[BG_ID] != pairs["_RID"]].copy()

        adj = {}
        for bid, sub in pairs.groupby(BG_ID):
            adj[str(bid)] = sorted(set(sub["_RID"].astype(str).tolist()))

        for bid in gp[BG_ID].astype(str):
            adj.setdefault(str(bid), [])

        degs = [len(v) for v in adj.values()]
        if degs:
            print(f"[Adjacency] Queen: nodes={len(adj)} | mean_deg={np.mean(degs):.2f} | median_deg={np.median(degs):.0f}")
        return adj
    except Exception as e_sjoin:
        # 2) Fallback: STRtree (no geopandas spatial index required)
        try:
            from shapely.strtree import STRtree
        except Exception as e_tree:
            raise RuntimeError(f"Queen adjacency failed (sjoin: {e_sjoin}; STRtree import: {e_tree})")

        geoms = list(gp.geometry.values)
        ids = gp[BG_ID].astype(str).tolist()
        tree = STRtree(geoms)

        # Needed for Shapely <2 where query() returns geometries
        id_to_idx = {id(gm): i for i, gm in enumerate(geoms)}

        adj = {ids[i]: [] for i in range(len(ids))}
        for i, geom in enumerate(geoms):
            try:
                hits = tree.query(geom)
            except Exception:
                hits = []

            if hits is None or len(hits) == 0:
                continue

            idxs = []
            first = hits[0]
            if isinstance(first, (int, np.integer)):
                idxs = list(hits)
            else:
                # Shapely 1.x: hits are geometries
                idxs = [id_to_idx.get(id(h), None) for h in hits]
                idxs = [j for j in idxs if j is not None]

            neigh = []
            for j in idxs:
                if j == i:
                    continue
                try:
                    if geom.intersects(geoms[j]):
                        neigh.append(ids[j])
                except Exception:
                    pass

            adj[ids[i]] = sorted(set(neigh))

        degs = [len(v) for v in adj.values()]
        if degs:
            print(f"[Adjacency] Queen: nodes={len(adj)} | mean_deg={np.mean(degs):.2f} | median_deg={np.median(degs):.0f}")
        return adj

def _proximity_neighbors_dict(df: pd.DataFrame, radius_mi: float) -> dict:
    """
    Build adjacency by proximity within radius_mi using BallTree haversine metric.
    """
    coords = np.deg2rad(df[[LAT_COL, LON_COL]].astype(float).to_numpy())
    tree = BallTree(coords, metric="haversine")
    rad = radius_mi / EARTH_RADIUS_MI
    inds = tree.query_radius(coords, r=rad, return_distance=False)

    ids = df[BG_ID].astype(str).to_numpy()
    adj = {}
    for i, neigh in enumerate(inds):
        n = [ids[j] for j in neigh if j != i]
        adj[ids[i]] = sorted(set(n))
    degs = [len(v) for v in adj.values()]
    if degs:
        print(f"[Adjacency] Proximity({radius_mi:.2f} mi): nodes={len(adj)} | mean_deg={np.mean(degs):.2f} | median_deg={np.median(degs):.0f}")
    return adj

def _mahal_d2(u, v, icov):
    d = v - u
    return float(d.T @ icov @ d)

def _chi2_ppf_wh(p, k):
    """
    Wilson-Hilferty approx to chi-square quantile (matches block script)
    """
    z = NormalDist().inv_cdf(p)
    a = 2.0 / (9.0 * k)
    return k * (1 - a + z * math.sqrt(a))**3


# ============================================================
# Twin cluster extraction + profiling
# ============================================================

def _broaden_distance_sample(anchor_id, base, per_borough, icov, zcols, max_n=1500, seed=42):
    import random
    rng = random.Random(seed)
    if per_borough and "borough" in base.columns:
        boro = str(base.loc[anchor_id, "borough"]) if anchor_id in base.index else None
        pool = base.index[base["borough"] == boro] if boro is not None else base.index
    else:
        pool = base.index
    pool = [bid for bid in pool if bid != anchor_id]
    if not pool:
        return []
    if len(pool) > max_n:
        pool = rng.sample(pool, max_n)
    a = base.loc[anchor_id, zcols].to_numpy(float)
    vals = []
    for nid in pool:
        b = base.loc[nid, zcols].to_numpy(float)
        vals.append(_mahal_d2(a, b, icov))
    return vals

def _contiguous_twins_for_anchor(anchor_id: str,
                                base_feat: pd.DataFrame,
                                models: dict,
                                adj: dict,
                                centroids: dict,
                                q: float = DEFAULT_Q,
                                per_borough: bool = TWINS_LOCAL_BY_BORO):
    """
    Block-script parity contiguous twin cluster:
      - Compute threshold thr from adjacency-neighbor Mahalanobis distances at quantile q
        (broaden sample if too few neighbors).
      - Grow cluster with BFS:
          ok_anchor: d(anchor, nid) <= thr
          ok_front : d(cur,    nid) <= GAMMA_LOCAL*thr (if GAMMA_LOCAL is not None)
      - Enforce MAX_CLUSTER_RADIUS_MI spatial cap from anchor centroid.
    """
    zcols = [c + "_z" for c in _TWIN_FEAT_COLS]
    base = base_feat[[BG_ID, "borough"] + zcols].copy().set_index(BG_ID)

    if anchor_id not in base.index:
        return {"cluster": [], "d_map": {}, "thr": None, "tested": 0}

    boro = str(base.loc[anchor_id, "borough"]) if per_borough else "_GLOBAL_"
    icov = models.get(boro, models["_GLOBAL_"])["icov"]

    # neighbor distances
    a_anchor = base.loc[anchor_id, zcols].to_numpy(float)
    vals = []
    for nid in adj.get(anchor_id, []):
        if nid not in base.index:
            continue
        b_vec = base.loc[nid, zcols].to_numpy(float)
        vals.append(_mahal_d2(a_anchor, b_vec, icov))

    if len(vals) < 8:
        vals = vals + _broaden_distance_sample(anchor_id, base, per_borough, icov, zcols, max_n=1500)

    thr_q = float(np.quantile(vals, q)) if len(vals) else float("inf")
    thr   = thr_q

    if TWINS_CHI_CAP is not None:
        try:
            thr = min(thr_q, float(_chi2_ppf_wh(float(TWINS_CHI_CAP), len(zcols))))
        except Exception:
            thr = thr_q

    if not np.isfinite(thr):
        thr = 0.0

    # BFS growth
    cluster = {anchor_id}
    d_map = {anchor_id: 0.0}
    queue = [anchor_id]
    seen = {anchor_id}
    tested = 0

    alat, alon = centroids.get(anchor_id, (None, None))

    def _within_cap(nid: str) -> bool:
        if alat is None or nid not in centroids or not MAX_CLUSTER_RADIUS_MI:
            return True
        blat, blon = centroids[nid]
        try:
            return _haversine_pair(alat, alon, blat, blon) <= float(MAX_CLUSTER_RADIUS_MI)
        except Exception:
            return True

    while queue:
        cur = queue.pop(0)
        for nid in adj.get(cur, []):
            if nid in seen or nid not in base.index:
                continue
            if not _within_cap(nid):
                continue

            seen.add(nid)
            tested += 1

            a_cur = base.loc[cur, zcols].to_numpy(float)
            b = base.loc[nid, zcols].to_numpy(float)

            dA = _mahal_d2(a_anchor, b, icov)
            dC = _mahal_d2(a_cur, b, icov)

            ok_anchor = (dA <= thr)
            ok_front = (dC <= (GAMMA_LOCAL * thr)) if GAMMA_LOCAL is not None else True

            if ok_anchor and ok_front:
                cluster.add(nid)
                d_map[nid] = float(dA)
                queue.append(nid)

    return {"cluster": sorted(cluster), "d_map": d_map, "thr": float(thr), "tested": int(tested)}


def _ray_walk_anchor_only(anchor_id: str,
                          octant: str,
                          thr: float,
                          base: pd.DataFrame,
                          zcols: list,
                          adj: dict,
                          centroids: dict,
                          icov=None,
                          max_steps: int = 200):
    """Ray-walk diagnostics (not drawn on map): block-script parity."""
    if anchor_id not in centroids or anchor_id not in base.index:
        return [], 0.0, ""

    alat, alon = centroids[anchor_id]
    dir_center = float(_OCT_CENTER.get(octant, 0.0))
    visited = {anchor_id}
    path = []
    cur = anchor_id
    prev_proj = 0.0
    eps = 1e-6

    base_ids = set(base.index)

    def _proj(bid: str):
        blat, blon = centroids[bid]
        dist = _haversine_pair(alat, alon, blat, blon)
        br = _bearing_deg(alat, alon, blat, blon)
        ang = abs(((br - dir_center + 180.0) % 360.0) - 180.0)
        proj = dist * math.cos(math.radians(ang))
        return proj, ang, dist, br

    a_vec = base.loc[anchor_id, zcols].to_numpy(float)

    for _ in range(int(max_steps)):
        cands = []
        for nid in adj.get(cur, []):
            if nid in visited or nid not in base_ids or nid not in centroids:
                continue
            blat, blon = centroids[nid]
            if RAY_MAX_RADIUS_MI and _haversine_pair(alat, alon, blat, blon) > float(RAY_MAX_RADIUS_MI):
                continue
            _, _, _, br = _proj(nid)
            if _octant_of(br) != octant:
                continue
            b_vec = base.loc[nid, zcols].to_numpy(float)
            if _mahal_d2(a_vec, b_vec, icov) > thr:
                continue
            proj, ang, dist, _ = _proj(nid)
            if proj > prev_proj + eps:
                cands.append((proj, -dist, -(180.0 - ang), nid))
        if not cands:
            break
        cands.sort()
        proj, _, _, chosen = cands[-1]
        path.append(chosen)
        visited.add(chosen)
        cur = chosen
        prev_proj = proj

    furthest = path[-1] if path else ""
    return path, float(prev_proj), str(furthest)


def _has_adjacent_twin(anchor_id: str,
                       base_feat: pd.DataFrame,
                       models: dict,
                       adj: dict,
                       centroids: dict,
                       q: float,
                       per_borough: bool = True,
                       min_needed: int = 1) -> bool:
    """Fast gate: does anchor have >= min_needed adjacent twin(s) under quantile q?"""
    zcols = [c + "_z" for c in _TWIN_FEAT_COLS]
    base = base_feat[[BG_ID, "borough"] + zcols].copy().set_index(BG_ID)
    if anchor_id not in base.index:
        return False

    neighs = adj.get(anchor_id, [])
    if not neighs:
        return False

    boro = str(base.loc[anchor_id, "borough"]) if per_borough else "_GLOBAL_"
    icov = models.get(boro, models["_GLOBAL_"])["icov"]

    a = base.loc[anchor_id, zcols].to_numpy(float)
    alat, alon = centroids.get(anchor_id, (None, None))

    def _within_cap(nid: str) -> bool:
        if alat is None or nid not in centroids or not MAX_CLUSTER_RADIUS_MI:
            return True
        blat, blon = centroids[nid]
        return _haversine_pair(alat, alon, blat, blon) <= float(MAX_CLUSTER_RADIUS_MI)

    vals = []
    for nid in neighs:
        if nid not in base.index or not _within_cap(nid):
            continue
        b = base.loc[nid, zcols].to_numpy(float)
        vals.append(_mahal_d2(a, b, icov))
    if len(vals) < 8:
        vals = vals + _broaden_distance_sample(anchor_id, base, per_borough, icov, zcols, max_n=1500)
    if not vals:
        return False

    thr_q = float(np.quantile(vals, q))
    thr = thr_q
    if TWINS_CHI_CAP is not None:
        try:
            thr = min(thr_q, float(_chi2_ppf_wh(float(TWINS_CHI_CAP), len(zcols))))
        except Exception:
            thr = thr_q

    hits = 0
    for nid in neighs:
        if nid not in base.index or not _within_cap(nid):
            continue
        b = base.loc[nid, zcols].to_numpy(float)
        dA = _mahal_d2(a, b, icov)
        local_ok = (dA <= (GAMMA_LOCAL * thr)) if GAMMA_LOCAL is not None else True
        if (dA <= thr) and local_ok:
            hits += 1
            if hits >= int(min_needed):
                return True
    return False


def _add_all_twin_clusters_layer_bg(m,
                                   anchors_df: pd.DataFrame,
                                   cmap,
                                   base_feat: pd.DataFrame,
                                   models: dict,
                                   adj: dict,
                                   centroids: dict,
                                   polys_gdf=None,
                                   q: float = DEFAULT_Q,
                                   per_borough: bool = TWINS_LOCAL_BY_BORO,
                                   max_anchors: int | None = None,
                                   min_units: int = 1,
                                   out_dir: str | None = None):
    """Block-script parity map layer: shaded contiguous twin clusters + summary CSV + bottom-right legend."""
    if anchors_df is None or anchors_df.empty:
        return

    anchors_df = anchors_df.copy()
    anchors_df[BG_ID] = anchors_df[BG_ID].astype(str)
    anchors = anchors_df[BG_ID].tolist()
    if max_anchors is not None:
        anchors = anchors[:int(max_anchors)]

    if "t_first" in anchors_df.columns:
        t_series = pd.to_numeric(anchors_df["t_first"], errors="coerce")
    else:
        t_series = pd.Series(np.nan, index=anchors_df.index, dtype=float)
    t_map = dict(zip(anchors_df[BG_ID], t_series))
    color_map = {bid: (cmap(float(t0)) if np.isfinite(t0) else "#666666") for bid, t0 in t_map.items()}

    poly_index = None

    # Optional: avoid shading mostly-water BG polygons in the twin overlay.
    polys_for_shade = polys_gdf
    if polys_for_shade is not None and TWIN_SHADE_EXCLUDE_WATER:
        try:
            polys_for_shade = _filter_bg_polys_for_land_shading(polys_for_shade, TWIN_SHADE_LAND_SHARE_MIN)
        except Exception:
            polys_for_shade = polys_gdf

    if polys_for_shade is not None and hasattr(polys_for_shade, "set_index"):
        try:
            polys = polys_for_shade[[BG_ID, "geometry"]].copy()
            polys[BG_ID] = polys[BG_ID].astype(str)
            poly_index = polys.set_index(BG_ID)
        except Exception:
            poly_index = None

    fg = folium.FeatureGroup(name="Candidate twin clusters (mahal)", show=True)
    m.add_child(fg)

    zcols = [c + "_z" for c in _TWIN_FEAT_COLS]
    base = base_feat[[BG_ID, "borough"] + zcols].copy().set_index(BG_ID)

    rows = []
    for anchor in anchors:
        if anchor not in base.index:
            continue

        boro = str(base.loc[anchor, "borough"]) if per_borough else "_GLOBAL_"
        icov = models.get(boro, models["_GLOBAL_"])["icov"]

        res = _contiguous_twins_for_anchor(anchor, base_feat, models, adj, centroids, q=q, per_borough=per_borough)
        cluster_ids = res.get("cluster", []) or []
        d_map = res.get("d_map", {}) or {}
        thr = float(res.get("thr", np.nan)) if res.get("thr", None) is not None else float("nan")
        tested = int(res.get("tested", 0))
        cluster_size = int(len(cluster_ids))
        adjacent_twins = max(0, cluster_size - 1)

        col = color_map.get(anchor, "#666666")

        if cluster_size >= int(min_units) and poly_index is not None:
            for bid in cluster_ids:
                if bid not in poly_index.index:
                    continue
                geom = poly_index.loc[bid, "geometry"]
                gj = folium.GeoJson(
                    data=getattr(geom, "__geo_interface__", geom),
                    style_function=lambda _feat, c=col: dict(color=c, weight=0.8, fill=True, fillColor=c, fillOpacity=0.40),
                    highlight_function=lambda _g: {"weight": 1.5},
                )
                dval = d_map.get(bid, np.nan)
                tip = (f"<b>Anchor:</b> {anchor}<br><b>BG:</b> {bid}"
                       f"<br><b>distance:</b> {float(dval):.3f}<br><b>cutoff (q):</b> {float(q):.2f} → {float(thr):.3f}"
                       f"<br><b>Mode:</b> mahal")
                folium.Tooltip(tip, sticky=True).add_to(gj)
                gj.add_to(fg)

        # Ray-walk diagnostics (CSV only)
        ray_counts = {}
        ray_miles = {}
        for oc in _OCTANTS:
            path, proj_mi, _ = _ray_walk_anchor_only(anchor, oc, thr, base, zcols, adj, centroids, icov=icov)
            ray_counts[oc] = int(len(path))
            ray_miles[oc] = float(proj_mi)

        rows.append({
            "anchor_bg_id": str(anchor),
            "t_first": float(t_map.get(anchor, np.nan)) if np.isfinite(t_map.get(anchor, np.nan)) else np.nan,
            "cluster_size": cluster_size,
            "adjacent_twins": adjacent_twins,
            "tested_candidates_cluster": tested,
            **{f"ray_count_{oc}": int(ray_counts[oc]) for oc in _OCTANTS},
            **{f"ray_miles_{oc}": float(ray_miles[oc]) for oc in _OCTANTS},
        })

    if out_dir:
        try:
            df_out = pd.DataFrame(rows)
            if not df_out.empty:
                df_out = df_out.sort_values(["adjacent_twins","t_first"], ascending=[False, True]).reset_index(drop=True)
                out_csv = os.path.join(out_dir, "twin_cluster_sizes_mahal.csv")
                df_out.to_csv(out_csv, index=False)
                print(f"✅ Twin summary (incl. rays) saved: {out_csv}")
        except Exception as e:
            print(f"[WARN] Could not write twin_cluster_sizes_mahal.csv: {e}")

    # Bottom-right legend (block-script parity)
    legend_html = ("{% macro html(this, kwargs) %}\n"
    "    <div style=\"position: fixed; bottom: 20px; right: 20px; z-index: 999999;\n"
    "                background: rgba(255,255,255,0.95); padding: 8px 10px; border: 1px solid #888;\n"
    "                font-size: 12px; line-height: 1.2; box-shadow: 0 1px 4px rgba(0,0,0,0.2);\">\n"
    "      <div style=\"font-weight:600;\">Candidate Twin Clusters</div>\n"
    "      <div>Shade = contiguous units indistinguishable from the anchor</div>\n"
    "      <div>Gate = quantile q (Mahalanobis)</div>\n"
    "      <div>Rays = outward per octant while indifference holds</div>\n"
    "    </div>\n"
    "    {% endmacro %}")
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

def _cluster_rays(anchor_id: str, cluster_ids: list, centroids: dict):
    """
    Choose at most one 'ray' per octant: farthest twin within RAY_MAX_RADIUS_MI in that octant.
    Returns list of dicts with ray info.
    """
    alat, alon = centroids.get(anchor_id, (np.nan, np.nan))
    if np.isnan(alat):
        return []

    best = {o: None for o in _OCTANTS}
    for nid in cluster_ids:
        if nid == anchor_id:
            continue
        nlat, nlon = centroids.get(nid, (np.nan, np.nan))
        if np.isnan(nlat):
            continue
        dist = _haversine_pair(alat, alon, nlat, nlon)
        if dist > RAY_MAX_RADIUS_MI:
            continue
        bear = _bearing_deg(alat, alon, nlat, nlon)
        octa = _octant_of(bear)
        cur = best.get(octa)
        if cur is None or dist > cur["dist_mi"]:
            best[octa] = {"to_id": nid, "dist_mi": dist, "bearing": bear, "octant": octa, "to_lat": nlat, "to_lon": nlon}

    out = [v for v in best.values() if v is not None]
    out.sort(key=lambda x: x["dist_mi"], reverse=True)
    return out

def _cluster_profile(anchor_id: str,
                     cluster_ids: list,
                     base_feat: pd.DataFrame,
                     centroids: dict,
                     adj: dict) -> dict:
    """
    Build a compact numerical profile similar in spirit to the block script, used for A↔B similarity.
    """
    zcols = [c + "_z" for c in _TWIN_FEAT_COLS]
    base = base_feat.set_index(BG_ID)

    rows = base.loc[cluster_ids, zcols].to_numpy(float) if cluster_ids else np.zeros((0, len(zcols)))
    if rows.size == 0:
        rows = np.zeros((1, len(zcols)))

    mu = rows.mean(axis=0)
    cov = np.cov(rows.T) if rows.shape[0] > 1 else np.diag(np.ones(rows.shape[1]))
    cov = np.atleast_2d(cov).astype(float)

    # anchor representativeness: distance between anchor z and cluster mean
    a = base.loc[anchor_id, zcols].to_numpy(float) if anchor_id in base.index else mu
    anch_gap = float(np.linalg.norm(a - mu))

    # cluster spatial radii
    alat, alon = centroids.get(anchor_id, (np.nan, np.nan))
    dists = []
    for nid in cluster_ids:
        nlat, nlon = centroids.get(nid, (np.nan, np.nan))
        if np.isnan(alat) or np.isnan(nlat):
            continue
        dists.append(_haversine_pair(alat, alon, nlat, nlon))
    max_r = float(np.max(dists)) if dists else 0.0
    mean_r = float(np.mean(dists)) if dists else 0.0

    # adjacency density within cluster (using provided adj graph)
    cluster_set = set(cluster_ids)
    edges = 0
    poss = 0
    for u in cluster_set:
        nu = set(adj.get(u, []))
        for v in cluster_set:
            if v == u:
                continue
            poss += 1
            if v in nu:
                edges += 1
    adj_density = float(edges / poss) if poss > 0 else 0.0

    boro = str(base.loc[anchor_id, "borough"]) if "borough" in base.columns and anchor_id in base.index else "Unknown"

    return {
        "anchor_id": anchor_id,
        "borough": boro,
        "n_twins": int(len(cluster_ids) - 1),      # excluding anchor
        "cluster_size": int(len(cluster_ids)),
        "mu_vec": mu.tolist(),
        "cov_flat": cov.flatten().tolist(),
        "anch_gap": anch_gap,
        "max_radius_mi": max_r,
        "mean_radius_mi": mean_r,
        "adj_density": adj_density,
        "anchor_lat": float(alat) if not np.isnan(alat) else np.nan,
        "anchor_lon": float(alon) if not np.isnan(alon) else np.nan,
    }


# ============================================================
# A↔B similarity
# ============================================================

def _vector_from_list(x, n=None):
    arr = np.asarray(x, dtype=float)
    if n is not None and arr.size != n:
        arr = np.resize(arr, n)
    return arr

def _matrix_from_flat(flat, k):
    a = np.asarray(flat, dtype=float)
    if a.size != k*k:
        a = np.resize(a, k*k)
    return a.reshape(k, k)

def _pair_component_dist(a_prof: dict, b_prof: dict):
    """
    Component distances before normalization.
    """
    k = len(_TWIN_FEAT_COLS)

    muA = _vector_from_list(a_prof["mu_vec"], k)
    muB = _vector_from_list(b_prof["mu_vec"], k)
    comp = float(np.linalg.norm(muA - muB))

    covA = _matrix_from_flat(a_prof["cov_flat"], k)
    covB = _matrix_from_flat(b_prof["cov_flat"], k)
    disp = float(np.linalg.norm(covA - covB, ord="fro"))

    scaleA = np.array([math.log1p(a_prof["cluster_size"]), math.log1p(a_prof["max_radius_mi"]), math.log1p(a_prof["mean_radius_mi"])], dtype=float)
    scaleB = np.array([math.log1p(b_prof["cluster_size"]), math.log1p(b_prof["max_radius_mi"]), math.log1p(b_prof["mean_radius_mi"])], dtype=float)
    scale = float(np.linalg.norm(scaleA - scaleB))

    shapeA = np.array([a_prof["adj_density"]], dtype=float)
    shapeB = np.array([b_prof["adj_density"]], dtype=float)
    shape = float(np.linalg.norm(shapeA - shapeB))

    anchor = float(abs(a_prof["anch_gap"] - b_prof["anch_gap"]))

    return {"comp": comp, "disp": disp, "scale": scale, "shape": shape, "anchor": anchor}

def compute_AB_similarity(sectionA_profiles: pd.DataFrame,
                          sectionB_profiles: pd.DataFrame,
                          weights: dict = AB_WEIGHTS,
                          boro_penalty: float = BORO_DIFFERENT_PENALTY):
    """
    Compute component-normalized distances and composite similarity %, returning:
    - matches_df: top-1 match per candidate
    - breakdown_df: all pair distances + sims
    """
    if sectionA_profiles.empty or sectionB_profiles.empty:
        return (pd.DataFrame(), pd.DataFrame())

    # Convert profiles into dicts for speed
    A = sectionA_profiles.to_dict(orient="records")
    B = sectionB_profiles.to_dict(orient="records")

    # First pass: compute raw component distances for all pairs
    rows = []
    for a in A:
        for b in B:
            comp = _pair_component_dist(a, b)
            rows.append({
                "A_anchor_id": a["anchor_id"],
                "B_anchor_id": b["anchor_id"],
                "A_borough": a.get("borough", "Unknown"),
                "B_borough": b.get("borough", "Unknown"),
                **{f"dist_{k}": float(v) for k, v in comp.items()}
            })

    br = pd.DataFrame(rows)

    # Normalize each component by its p90 to reduce scale dominance
    p90 = {}
    for k in ["comp", "disp", "scale", "shape", "anchor"]:
        col = f"dist_{k}"
        val = float(np.nanpercentile(br[col].to_numpy(float), 90)) if br.shape[0] else 1.0
        p90[k] = max(val, 1e-9)

    for k in ["comp", "disp", "scale", "shape", "anchor"]:
        br[f"norm_{k}"] = br[f"dist_{k}"] / p90[k]

    # Composite distance (weighted) + borough penalty
    compD = np.zeros(len(br), dtype=float)
    for k, w in weights.items():
        compD += float(w) * br[f"norm_{k}"].to_numpy(float)

    bdiff = (br["A_borough"].astype(str) != br["B_borough"].astype(str)).to_numpy(bool)
    compD = compD + (bdiff.astype(float) * float(boro_penalty))
    br["composite_dist"] = compD

    # Similarity % from composite distance p90
    p90D = float(np.nanpercentile(compD, 90)) if compD.size else 1.0
    p90D = max(p90D, 1e-9)
    br["similarity_pct"] = np.clip(100.0 * (1.0 - (br["composite_dist"] / p90D)), 0.0, 100.0)

    # Also expose component similarity % (normalized)
    for k in ["comp", "disp", "scale", "shape", "anchor"]:
        br[f"sim_{k}_pct"] = np.clip(100.0 * (1.0 - (br[f"norm_{k}"])), 0.0, 100.0)

    # Top-1 match per A
    idx = br.groupby("A_anchor_id")["similarity_pct"].idxmax()
    matches = br.loc[idx].copy().sort_values("similarity_pct", ascending=False).reset_index(drop=True)

    return matches, br


# ============================================================
# Reverse geocoding (optional)
# ============================================================

def _load_geocode_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_geocode_cache(path: str, cache: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def _reverse_geocode(lat: float, lon: float, cache: dict):
    """
    Nominatim reverse geocode; returns (label, updated_cache).
    Disabled by default; requires requests.
    """
    if not DO_REVERSE_GEOCODE or requests is None:
        return "", cache
    key = f"{lat:.6f},{lon:.6f}"
    if key in cache:
        return str(cache[key]), cache
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"format":"jsonv2", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 0}
        headers = {"User-Agent": "bg-stress-test/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=GEOCODE_TIMEOUT_S)
        if r.status_code == 200:
            js = r.json()
            label = js.get("display_name", "")
            cache[key] = label
            time.sleep(GEOCODE_SLEEP_S)
            return str(label), cache
    except Exception:
        pass
    cache[key] = ""
    return "", cache


# ============================================================
# Map
# ============================================================

def _color_for_t(t_first: float) -> str:
    t = float(np.clip(t_first if np.isfinite(t_first) else 1.0, 0.0, 1.0))
    rgba = MPL_CMAP(MPL_NORM(t))
    r, g, b = [int(255*x) for x in rgba[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"

def _radius_for_freq(freq: float) -> float:
    f = float(np.clip(freq, 0.0, 1.0))
    return float(DOT_MIN_RADIUS + (DOT_MAX_RADIUS - DOT_MIN_RADIUS) * (f ** 0.5))

def _legend_html() -> str:
    html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 12px 14px; border: 1px solid #bbb;
                border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.2);
                width: 320px; font-size: 12px;">
      <div style="font-weight: 700; margin-bottom: 6px;">NYC Block Group Stress-Test Ladder (Aggregated)</div>
      <div style="margin-bottom: 6px;">
        <span style="font-weight:600;">Dot color</span>: earliest <code>t</code> where BG first appears (strictness).<br/>
        <span style="font-weight:600;">Dot size</span>: frequency across 100 ladder regimes.
      </div>
      <div style="margin-bottom: 6px;">
        <span style="font-weight:600;">Popup</span>: Top-1 Twin Store + Composite Similarity % and component sims.
      </div>
      <div style="opacity:0.8;">
        Twin clusters (if enabled) are approximate for BGs unless a BG polygon shapefile is provided.
      </div>
    </div>
    """
    return html

def build_aggregated_map_bg(
    agg_bg: pd.DataFrame,
    anchors_bg: pd.DataFrame,
    best_match: pd.DataFrame,
    out_html: str,
    base_feat: pd.DataFrame,
    models: dict,
    adj: dict,
    centroids: dict = None,
    bg_centroids: dict = None,
    twin_clusters: dict = None,
    twin_rays: dict = None,
    require_adjacent_twin: bool = REQUIRE_ADJACENT_TWIN,
    min_adjacent_twins: int = MIN_ADJACENT_TWINS,
):
    """
    BG version of the aggregated ladder map that MATCHES the *block* map styling:
      - color encodes strictness (earliest T a BG appears)
      - ring outlines show median ring (optionally scaled by frequency)
      - black dots show selection frequency
      - FedEx stars are purple "★" DivIcons (same CSS)
      - optional shaded twin clusters layer (requires BG polygon shapefile)

    Notes:
      - This function is intentionally defensive about column naming.
      - If BG polygons are available, we (1) compute bounds for centering,
        (2) enable shaded twin clusters, and (3) prefer queen adjacency for twin logic.
    """

    # Backward/forward-compatible aliasing for callers:
    if centroids is None:
        centroids = bg_centroids
    _ = (twin_clusters, twin_rays)

    if agg_bg is None or agg_bg.empty:
        raise ValueError("agg_bg is empty; nothing to map.")

    dfm = agg_bg.copy()
    dfm[BG_ID] = dfm[BG_ID].astype(str)

    # ---- Column normalization (match block-script expectations) ----
    if "lat_med" not in dfm.columns:
        if LAT_COL in dfm.columns:
            dfm["lat_med"] = pd.to_numeric(dfm[LAT_COL], errors="coerce")
        elif "bg_lat" in dfm.columns:
            dfm["lat_med"] = pd.to_numeric(dfm["bg_lat"], errors="coerce")
        else:
            dfm["lat_med"] = np.nan

    if "lon_med" not in dfm.columns:
        if LON_COL in dfm.columns:
            dfm["lon_med"] = pd.to_numeric(dfm[LON_COL], errors="coerce")
        elif "bg_lon" in dfm.columns:
            dfm["lon_med"] = pd.to_numeric(dfm["bg_lon"], errors="coerce")
        else:
            dfm["lon_med"] = np.nan

    if "ring_med" not in dfm.columns:
        for cand in ["r_use_bg", "r_use", "ring", "ring_mi"]:
            if cand in dfm.columns:
                dfm["ring_med"] = pd.to_numeric(dfm[cand], errors="coerce")
                break
        if "ring_med" not in dfm.columns:
            dfm["ring_med"] = np.nan

    if "freq" not in dfm.columns:
        if "ladder_count" in dfm.columns:
            dfm["freq"] = pd.to_numeric(dfm["ladder_count"], errors="coerce").fillna(0).astype(int)
        elif "sel_count" in dfm.columns:
            dfm["freq"] = pd.to_numeric(dfm["sel_count"], errors="coerce").fillna(0).astype(int)
        else:
            dfm["freq"] = 0

    if "freq_frac" not in dfm.columns:
        if "ladder_freq" in dfm.columns:
            dfm["freq_frac"] = pd.to_numeric(dfm["ladder_freq"], errors="coerce")
        elif "sel_frac" in dfm.columns:
            dfm["freq_frac"] = pd.to_numeric(dfm["sel_frac"], errors="coerce")
        else:
            dfm["freq_frac"] = np.nan
    dfm["freq_frac"] = pd.to_numeric(dfm["freq_frac"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    if "t_first" not in dfm.columns:
        dfm["t_first"] = np.nan
    dfm["t_first"] = pd.to_numeric(dfm["t_first"], errors="coerce")

    if "t_last" not in dfm.columns:
        dfm["t_last"] = np.nan
    dfm["t_last"] = pd.to_numeric(dfm["t_last"], errors="coerce")

    # Ensure we can do haversine caps / ray-walks even if centroids wasn't passed.
    if centroids is None:
        centroids = {}
        try:
            if base_feat is not None and (LAT_COL in base_feat.columns) and (LON_COL in base_feat.columns):
                tmp = base_feat[[BG_ID, LAT_COL, LON_COL]].copy()
                tmp[BG_ID] = tmp[BG_ID].astype(str)
                tmp[LAT_COL] = pd.to_numeric(tmp[LAT_COL], errors="coerce")
                tmp[LON_COL] = pd.to_numeric(tmp[LON_COL], errors="coerce")
                centroids = {str(r[BG_ID]): (float(r[LAT_COL]), float(r[LON_COL]))
                             for _, r in tmp.dropna(subset=[LAT_COL, LON_COL]).iterrows()}
        except Exception:
            centroids = {}

    # ---- Try to load BG polygons so twin areas can be shaded ----
    polys = None
    polys_path_used = None
    polys_zip_used = None
    polys_extract_dir = None

    def _score_bg_poly_name(fn: str) -> int:
        low = (fn or "").lower()
        # Prefer explicit NYC BG files; allow broad keyword matches.
        keywords = [
            "nyc", "newyork", "new_york",
            "block_group", "blockgroup", "block_groups",
            "bgrp", "blkgrp",
        ]
        return sum(1 for k in keywords if k in low)

    def _pick_best_shp_in_tree(root_dir: str) -> str | None:
        """Pick the most likely BG polygon .shp under root_dir."""
        if not root_dir or not os.path.isdir(root_dir):
            return None
        candidates = []
        for root, dirs, files in os.walk(root_dir):
            # Skip huge folders
            skip = {"venv", ".venv", ".git", "__pycache__", "site-packages"}
            dirs[:] = [d for d in dirs if d not in skip]
            for fn in files:
                if fn.lower().endswith(".shp"):
                    score = _score_bg_poly_name(fn)
                    candidates.append((score, os.path.join(root, fn)))
        if not candidates:
            return None
        # Highest score, then shortest path
        candidates.sort(key=lambda x: (-x[0], len(x[1])))
        return candidates[0][1]

    def _auto_find_bg_shp(start_dir: str) -> str | None:
        """Ranked search for BG polygon shapefile (.shp); keep shallow to avoid huge scans."""
        if not start_dir or not os.path.isdir(start_dir):
            return None

        # Search BASE_DIR and its parent
        search_roots = [start_dir]
        parent = os.path.dirname(start_dir.rstrip(os.sep))
        if parent and parent != start_dir:
            search_roots.append(parent)

        max_depth = 4
        candidates = []
        for root0 in search_roots:
            root0 = os.path.abspath(root0)
            for root, dirs, files in os.walk(root0):
                rel = os.path.relpath(root, root0)
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if depth > max_depth:
                    dirs[:] = []
                    continue
                skip = {"venv", ".venv", ".git", "__pycache__", "site-packages"}
                dirs[:] = [d for d in dirs if d not in skip]
                for fn in files:
                    if not fn.lower().endswith(".shp"):
                        continue
                    score = _score_bg_poly_name(fn)
                    if score <= 0:
                        continue
                    candidates.append((score, os.path.join(root, fn)))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], len(x[1])))
        return candidates[0][1]

    def _auto_find_bg_zip(start_dir: str) -> str | None:
        """Ranked search for BG polygon zip (contains .shp/.dbf/.shx)."""
        if not start_dir or not os.path.isdir(start_dir):
            return None

        # Search BASE_DIR and its parent
        search_roots = [start_dir]
        parent = os.path.dirname(start_dir.rstrip(os.sep))
        if parent and parent != start_dir:
            search_roots.append(parent)

        max_depth = 4
        candidates = []
        for root0 in search_roots:
            root0 = os.path.abspath(root0)
            for root, dirs, files in os.walk(root0):
                rel = os.path.relpath(root, root0)
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if depth > max_depth:
                    dirs[:] = []
                    continue
                skip = {"venv", ".venv", ".git", "__pycache__", "site-packages"}
                dirs[:] = [d for d in dirs if d not in skip]
                for fn in files:
                    if not fn.lower().endswith(".zip"):
                        continue
                    score = _score_bg_poly_name(fn)
                    # Also accept common explicit names like "nyc_block_groups_2021.zip"
                    if score <= 0 and "block" not in fn.lower():
                        continue
                    candidates.append((score, os.path.join(root, fn)))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], len(x[1])))
        return candidates[0][1]

    def _safe_extract_zip(zip_path: str, dest_dir: str) -> None:
        """Extract zip safely (prevents path traversal)."""
        import zipfile
        os.makedirs(dest_dir, exist_ok=True)
        base = os.path.abspath(dest_dir) + os.sep
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # Normalize path separators in zip members
                name = member.filename.replace("\\", "/")
                if name.endswith("/"):
                    continue
                out_path = os.path.abspath(os.path.join(dest_dir, name))
                if not out_path.startswith(base):
                    # Skip unsafe member
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

    def _shp_from_zip(zip_path: str, dest_root: str) -> str | None:
        """Extract zip (once) and return best .shp path inside."""
        if not zip_path or not os.path.exists(zip_path):
            return None
        zname = os.path.splitext(os.path.basename(zip_path))[0]
        dest_dir = os.path.join(dest_root, f"_bg_polys_{zname}")
        # If already extracted, reuse.
        if os.path.isdir(dest_dir):
            shp = _pick_best_shp_in_tree(dest_dir)
            if shp and os.path.exists(shp):
                return shp
        try:
            _safe_extract_zip(zip_path, dest_dir)
            shp = _pick_best_shp_in_tree(dest_dir)
            return shp
        except Exception:
            return None

    if gpd is not None:
        try:
            # Prefer user-provided BG_SHP_PATH; allow it to be either a .shp or a .zip.
            if BG_SHP_PATH and os.path.exists(BG_SHP_PATH):
                if BG_SHP_PATH.lower().endswith(".zip"):
                    polys_zip_used = BG_SHP_PATH
                    polys_extract_dir = RUN_ROOT
                    polys_path_used = _shp_from_zip(polys_zip_used, polys_extract_dir)
                else:
                    polys_path_used = BG_SHP_PATH
            else:
                # Auto-discovery: prefer .shp, else fall back to a BG zip.
                polys_path_used = _auto_find_bg_shp(BASE_DIR)
                if not polys_path_used:
                    polys_zip_used = _auto_find_bg_zip(BASE_DIR)
                    if polys_zip_used and os.path.exists(polys_zip_used):
                        polys_extract_dir = RUN_ROOT
                        polys_path_used = _shp_from_zip(polys_zip_used, polys_extract_dir)

            if polys_path_used and os.path.exists(polys_path_used):
                polys = gpd.read_file(polys_path_used)
                polys = _ensure_bg_id(polys)
                # merge/derive borough if needed
                polys = _ensure_borough_col(polys)
                # project to WGS84 for folium
                if getattr(polys, "crs", None) is not None:
                    polys = polys.to_crs(4326)
        except Exception:
            polys = None
            polys_path_used = None
            polys_zip_used = None
            polys_extract_dir = None

# Prefer queen adjacency for twin computations if polygons exist (parity with block script).
    adj_for_twins = adj
    if polys is not None:
        try:
            adj_for_twins = _queen_neighbors_dict_from_shp(polys)
        except Exception:
            adj_for_twins = adj

    # ---- Map center from polygons bounds (same as block script) ----
    try:
        if polys is not None:
            minx, miny, maxx, maxy = polys.total_bounds
            center = [(miny + maxy) / 2, (minx + maxx) / 2]
        else:
            raise RuntimeError("no polys bounds")
    except Exception:
        center = [40.73, -73.94]

    m = folium.Map(location=center, zoom_start=11, tiles="cartodb positron", control_scale=True)

    # Color scale (strictness)
    cmap = branca.colormap.LinearColormap(colors=STRICT_COLORS, vmin=0.0, vmax=1.0).to_step(11)
    cmap.caption = "Strictness (color = earliest T a block appears; 0 = strict, 1 = relaxed)"
    cmap.add_to(m)

    # Helper functions (match reference)
    def ring_scale(frac):
        frac = float(frac) if frac is not None else 0.0
        frac = max(0.0, min(1.0, frac))
        return 1.0 + (float(RING_MAX_SCALE) - 1.0) * frac

    def dot_radius(frac):
        frac = float(frac) if frac is not None else 0.0
        frac = max(0.0, min(1.0, frac))
        return float(DOT_MIN_RADIUS) + (float(DOT_MAX_RADIUS) - float(DOT_MIN_RADIUS)) * frac

    # Aggregated candidates (rings + dots in ONE layer, like block script)
    fg_agg = folium.FeatureGroup(name="Aggregated (rings = median ring; dot = selection frequency)", show=True)
    m.add_child(fg_agg)

    # Optional: enforce adjacent-twin gate (same as block script)
    if require_adjacent_twin:
        if min_adjacent_twins is None:
            min_adjacent_twins = 1
        keep_mask = []
        for bid in dfm[BG_ID].astype(str).tolist():
            keep_mask.append(_has_adjacent_twin(
                bid, base_feat=base_feat, models=models, adj=adj_for_twins, centroids=centroids,
                q=(1.0 - float(globals().get("TWINS_ALPHA", 0.05))),  # same as block script
                per_borough=TWINS_LOCAL_BY_BORO, min_needed=int(min_adjacent_twins)
            ))
        dfm = dfm.loc[keep_mask].copy()
        print(f"[INFO] After adjacent-twin gate (min={int(min_adjacent_twins)}), remaining: {len(dfm)}")

    # Candidate → best match lookup (for tooltip enrichment)
    bm_lookup = {}
    if best_match is not None and not best_match.empty:
        key_col = None
        if BG_ID in best_match.columns:
            key_col = BG_ID
        elif "A_anchor_id" in best_match.columns:
            key_col = "A_anchor_id"
        try:
            if key_col is not None:
                for _, r in best_match.iterrows():
                    k = str(r.get(key_col, "")).strip()
                    if k:
                        bm_lookup[k] = r
        except Exception:
            bm_lookup = {}


    # Reverse geocode cache + anchor coordinate lookup (for tooltip addresses)
    geocode_cache = _load_geocode_cache(GEOCODE_CACHE_JSON) if DO_REVERSE_GEOCODE else {}
    anchor_xy = {}
    if DO_REVERSE_GEOCODE and anchors_bg is not None and not anchors_bg.empty and BG_ID in anchors_bg.columns:
        try:
            a = anchors_bg.copy()
            a[BG_ID] = a[BG_ID].astype(str)

            # Prefer snapped BG centroid coords if present; otherwise fall back to raw anchor coords.
            lat_col = None
            lon_col = None
            for cand in ["bg_lat", "anchor_lat", LAT_COL, "lat"]:
                if cand in a.columns:
                    lat_col = cand
                    break
            for cand in ["bg_lon", "anchor_lon", LON_COL, "lon"]:
                if cand in a.columns:
                    lon_col = cand
                    break

            if lat_col and lon_col:
                a["_alat"] = pd.to_numeric(a[lat_col], errors="coerce")
                a["_alon"] = pd.to_numeric(a[lon_col], errors="coerce")
                a = a.dropna(subset=["_alat", "_alon"])
                anchor_xy = {str(rr[BG_ID]): (float(rr["_alat"]), float(rr["_alon"])) for _, rr in a.iterrows()}
        except Exception:
            anchor_xy = {}
    # Draw rings + dots
    for _, r in dfm.iterrows():
        bid = str(r.get(BG_ID, ""))
        lat = float(r.get("lat_med", np.nan))
        lon = float(r.get("lon_med", np.nan))
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue

        t0 = float(r.get("t_first", np.nan)) if np.isfinite(r.get("t_first", np.nan)) else np.nan
        tL = float(r.get("t_last", np.nan)) if np.isfinite(r.get("t_last", np.nan)) else np.nan
        color = (cmap(t0) if np.isfinite(t0) else "#666666")

        freq = int(r.get("freq", 0))
        frac = float(r.get("freq_frac", 0.0))
        r_use = float(r.get("ring_med", np.nan)) if np.isfinite(r.get("ring_med", np.nan)) else np.nan

        # Build tooltip HTML (same structure as block script; fill what we can)
        pbus = r.get("pbus_sum", r.get("PotentialBus", r.get("Potential Business", np.nan)))
        try:
            pbus = float(pd.to_numeric(pbus, errors="coerce"))
        except Exception:
            pbus = np.nan

        # Optional best-match info
        mr = bm_lookup.get(bid, None)
        match_id = ""
        sim = ""
        if mr is not None:
            try:
                match_id = str(mr.get("B_anchor_id", mr.get("anchor_id", "")))
            except Exception:
                match_id = ""
            sim = mr.get("similarity_pct_disp", mr.get("similarity_pct", ""))

        # Reverse-geocoded addresses (BG centroid labels). These are approximate and
        # are intended for human-readable map hover tooltips.
        cand_addr = ""
        if DO_REVERSE_GEOCODE:
            cand_addr, geocode_cache = _reverse_geocode(lat, lon, geocode_cache)

        anch_addr = ""
        if DO_REVERSE_GEOCODE and match_id and match_id in anchor_xy:
            alat, alon = anchor_xy[match_id]
            anch_addr, geocode_cache = _reverse_geocode(alat, alon, geocode_cache)

        t0_disp = f"{t0:.2f}" if np.isfinite(t0) else "N/A"
        tL_disp = f"{tL:.2f}" if np.isfinite(tL) else "N/A"
        r_disp  = f"{r_use:.2f}" if np.isfinite(r_use) else "—"

        # similarity may already be formatted; try to normalize to a percent display
        sim_disp = ""
        if sim is not None and str(sim).strip() != "":
            try:
                sim_num = float(sim)
                sim_disp = f"{sim_num:.1f}%"
            except Exception:
                sim_disp = str(sim)

        tooltip_html = (
            f"<b>{BG_ID}:</b> {bid}<br>"
            f"<b>Candidate Address</b>: {cand_addr or '—'}<br>"
            f"<b>Earliest-T</b>: {t0_disp}<br>"
            f"<b>Last-T</b>: {tL_disp}<br>"
            f"<b>Ladder selections</b>: {freq} / {TOTAL_SCENARIOS} ({frac:.0%})<br>"
            f"<b>ring (mi)</b>: {r_disp}<br>"
            f"<b>Nearby Businesses</b>: {(pbus if np.isfinite(pbus) else 0):,.0f}"
        )

        if match_id or sim_disp:
            tooltip_html += (
                f"<br><b>Top-1 Twin Anchor</b>: {match_id or '—'}"
                f"<br><b>Twin Anchor Address</b>: {anch_addr or '—'}"
            )
            if sim_disp:
                tooltip_html += f"<br><b>Similarity</b>: {sim_disp}"

        # Ring (outline only)
        if np.isfinite(r_use) and r_use > 0:
            draw_radius_m = r_use * 1609.344 * (ring_scale(frac) if SCALE_RING_BY_FREQ else 1.0)
            ring = folium.Circle([lat, lon], radius=draw_radius_m, color=color, weight=1.4, fill=False, opacity=0.95)
            ring.add_to(fg_agg)
            folium.Tooltip(tooltip_html, sticky=True).add_to(ring)

        # Dot (black)
        dot = folium.CircleMarker(
            [lat, lon],
            radius=dot_radius(frac),
            color="#000000",
            weight=1.0,
            fill=True,
            fill_opacity=0.95,
            fill_color="#000000",
        )
        dot.add_to(fg_agg)
        folium.Tooltip(tooltip_html, sticky=True).add_to(dot)

    # Existing FedEx stars (same CSS as block script)
    try:
        fed = anchors_bg.copy() if anchors_bg is not None else pd.DataFrame()
        if not fed.empty:
            # Accept either bg_lat/bg_lon or raw LAT_COL/LON_COL
            if "bg_lat" not in fed.columns and LAT_COL in fed.columns:
                fed["bg_lat"] = pd.to_numeric(fed[LAT_COL], errors="coerce")
            if "bg_lon" not in fed.columns and LON_COL in fed.columns:
                fed["bg_lon"] = pd.to_numeric(fed[LON_COL], errors="coerce")

            fed = fed.dropna(subset=["bg_lat", "bg_lon"]).copy()
            if not fed.empty:
                STAR_HTML = ("<div style='font-size:14px; line-height:14px; color:#800080;"
                             "text-shadow:-1px 0 #000, 0 1px #000, 1px 0 #000, 0 -1px #000;'>★</div>")
                fg_fed = folium.FeatureGroup(name="FedEx Office Print & Ship Center (NYC only)", show=True)
                m.add_child(fg_fed)
                for _, s in fed.iterrows():
                    lat = float(s["bg_lat"]); lon = float(s["bg_lon"])
                    folium.Marker(
                        location=[lat, lon],
                        icon=folium.DivIcon(html=STAR_HTML),
                        tooltip="FedEx Office Print & Ship Center"
                    ).add_to(fg_fed)
    except Exception:
        pass

    # Candidate twin clusters + rays (shaded polygons) — parity with block script.
    if INCLUDE_TWIN_CLUSTERS_ALL:
        try:
            q_for_layer = 1.0 - float(globals().get("TWINS_ALPHA", DEFAULT_Q))
            _add_all_twin_clusters_layer_bg(
                m=m,
                anchors_df=dfm[[BG_ID, "t_first"]].copy() if "t_first" in dfm.columns else dfm[[BG_ID]].copy(),
                cmap=cmap,
                base_feat=base_feat,
                models=models,
                adj=adj_for_twins,
                centroids=centroids,
                polys_gdf=polys,
                q=q_for_layer,
                per_borough=TWINS_LOCAL_BY_BORO,
                max_anchors=MAX_TWIN_ANCHORS,
                min_units=MIN_TWIN_CLUSTER_UNITS,
                out_dir=os.path.dirname(out_html) if out_html else None,
            )
            if polys is None:
                print("[WARN] Twin cluster shading is enabled, but no BG polygon shapefile was found/loaded. "
                      "Set BG_SHP_PATH to your BG .shp (or a .zip containing the shapefile) to shade twin areas.")
            else:
                print(f"[INFO] Twin clusters shaded using polygons from: {polys_path_used}")
        except Exception as e:
            print(f"[WARN] Twin-cluster layer failed: {e}")

    # Bottom-left legend (aggregated ladder view)
    try:
        legend_html = f"""{{% macro html(this, kwargs) %}}
    <div style="
     position: fixed; bottom: 20px; left: 20px; z-index: 999999;
     background: rgba(255,255,255,0.95); padding: 8px 10px; border: 1px solid #888;
     font-size: 12px; line-height: 1.2; box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    ">
      <div style="font-weight:600;">Aggregated ladder view (100 scenarios)</div>
      <div><b>Color</b>: earliest T a block appears (strict→relaxed)</div>
      <div><b>Dot size</b>: frequency across {TOTAL_SCENARIOS} scenarios</div>
      <div><b>Ring radius</b>: {'scaled by frequency (visual)' if SCALE_RING_BY_FREQ else 'true demand radius (mi)'} </div>
      <div><b>Hover</b>: times selected, earliest/last T, Top-1 Twin Store</div>
      <div><b>Similarity %</b>: A↔B context similarity (0–100; higher is more similar)</div>
      <div><b>Stars</b>: Anchor Locations</div>
    </div>
    {{% endmacro %}}"""
        legend = MacroElement()
        legend._template = Template(legend_html)
        m.get_root().add_child(legend)
    except Exception:
        pass

    folium.LayerControl(collapsed=False).add_to(m)

    # Save
    if DO_REVERSE_GEOCODE:
        _save_geocode_cache(GEOCODE_CACHE_JSON, geocode_cache)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    print(f"✅ Map saved: {out_html}")

def run_ladder_and_map():
    # --- validate inputs
    if not os.path.exists(BG_FEATURES_CSV):
        raise FileNotFoundError(f"Missing BG_FEATURES_CSV: {BG_FEATURES_CSV}")
    if not os.path.exists(MC_COUNTS_CSV):
        # GitHub/sample runs often place mc_counts_bg.csv next to this script.
        alt = os.path.join(BASE_DIR, "mc_counts_bg.csv")
        if os.path.exists(alt):
            print(f"(note) MC_COUNTS_CSV not found at {MC_COUNTS_CSV}; using {alt}")
            mc_path = alt
        else:
            raise FileNotFoundError(f"Missing MC_COUNTS_CSV: {MC_COUNTS_CSV}")
    else:
        mc_path = MC_COUNTS_CSV

    feat = _ensure_bg_id(pd.read_csv(BG_FEATURES_CSV, dtype={BG_ID: str}))
    mc   = _ensure_bg_id(pd.read_csv(mc_path, dtype={BG_ID: str}))

    # --- Drop excluded BGs from the universe at the very beginning (features + MC + everything downstream)
    excluded_bg_ids = set()
    if EXCLUDED_BG_CSV and os.path.exists(str(EXCLUDED_BG_CSV)):
        excluded_bg_ids = _load_excluded_bg_ids(EXCLUDED_BG_CSV)
        print(f"🧹 Loaded excluded BGs: {len(excluded_bg_ids)}")

    if excluded_bg_ids:
        # Filter using a canonical id so formats like GEOIDFQ / floats still match.
        feat["_bgid_canon"] = feat[BG_ID].map(_canon_bg_id)
        mc["_bgid_canon"]   = mc[BG_ID].map(_canon_bg_id)

        before_feat = len(feat)
        before_mc   = len(mc)

        feat = feat[~feat["_bgid_canon"].isin(excluded_bg_ids)].drop(columns=["_bgid_canon"]).reset_index(drop=True)
        mc   = mc[~mc["_bgid_canon"].isin(excluded_bg_ids)].drop(columns=["_bgid_canon"]).reset_index(drop=True)

        print(f"✅ Exclusion applied: removed {before_feat - len(feat)} BGs from features | remaining {len(feat)}")
        print(f"✅ Exclusion applied: removed {before_mc - len(mc)} BGs from MC counts | remaining {len(mc)}")

    # normalize MC column names to *_bg
    rename_map = {}
    if "eligible_runs" in mc.columns and "eligible_runs_bg" not in mc.columns: rename_map["eligible_runs"] = "eligible_runs_bg"
    if "select_count" in mc.columns and "select_count_bg" not in mc.columns:   rename_map["select_count"]  = "select_count_bg"
    if "pass_rate" in mc.columns and "pass_rate_bg" not in mc.columns:        rename_map["pass_rate"]     = "pass_rate_bg"
    if "eligible_runs_blk" in mc.columns and "eligible_runs_bg" not in mc.columns: rename_map["eligible_runs_blk"] = "eligible_runs_bg"
    if "select_count_blk" in mc.columns and "select_count_bg" not in mc.columns:   rename_map["select_count_blk"]  = "select_count_bg"
    if "pass_rate_blk" in mc.columns and "pass_rate_bg" not in mc.columns:        rename_map["pass_rate_blk"]     = "pass_rate_bg"
    if rename_map:
        mc = mc.rename(columns=rename_map)

    df = feat.merge(mc, on=BG_ID, how="left")
    df = _coerce_latlon(df)
    df = _ensure_borough_col(df)
    df["PotentialBus"] = _get_potential_bus(df).astype(float)

    if "eligible_runs_bg" in df.columns:
        df["eligible_runs_bg"] = pd.to_numeric(df["eligible_runs_bg"], errors="coerce").fillna(0).astype(int)
    else:
        df["eligible_runs_bg"] = 0

    if "select_count_bg" in df.columns:
        df["select_count_bg"]  = pd.to_numeric(df["select_count_bg"], errors="coerce").fillna(0).astype(int)
    else:
        df["select_count_bg"]  = 0

    if "pass_rate_bg" in df.columns:
        df["pass_rate_bg"]     = pd.to_numeric(df["pass_rate_bg"], errors="coerce").fillna(0.0).astype(float)
    else:
        df["pass_rate_bg"]     = 0.0

    # Scale select-count ladder thresholds to the observed eligible-run count.
    global ELIGIBLE_RUNS_REF_BG
    try:
        ELIGIBLE_RUNS_REF_BG = int(pd.to_numeric(df["eligible_runs_bg"], errors="coerce").max())
    except Exception:
        ELIGIBLE_RUNS_REF_BG = 1
    if ELIGIBLE_RUNS_REF_BG < 10:
        print(f"(note) eligible_runs_bg max is {ELIGIBLE_RUNS_REF_BG}; scaling MIN_SELECT_COUNT_BG accordingly.")

    # FDR column (optional; bypass if missing like original fallback)
    if "p_fdr_bg" not in df.columns:
        for cand in ["p_fdr_blk", "p_fdr", "p_fdr_bg"]:
            if cand in df.columns:
                df["p_fdr_bg"] = pd.to_numeric(df[cand], errors="coerce")
                break
    if "p_fdr_bg" not in df.columns:
        df["p_fdr_bg"] = np.nan
    if df["p_fdr_bg"].isna().all():
        print("(note) FDR p-values unavailable; bypassing FDR filter.")
        df["p_fdr_bg"] = 0.0
    else:
        df["p_fdr_bg"] = pd.to_numeric(df["p_fdr_bg"], errors="coerce").fillna(1.0).astype(float)

    # Snap anchors → BG IDs (existing footprint)
    anchors_bg = _snap_anchors_to_bgs(df)
    anchor_ids = set(anchors_bg[BG_ID].astype(str).tolist())

    # store anchor coords (BG centroid coords)
    anchors_bg["bg_lat"] = anchors_bg[BG_ID].map(df.set_index(BG_ID)[LAT_COL].astype(float).to_dict())
    anchors_bg["bg_lon"] = anchors_bg[BG_ID].map(df.set_index(BG_ID)[LON_COL].astype(float).to_dict())
    anchors_bg = anchors_bg.dropna(subset=["bg_lat","bg_lon"]).reset_index(drop=True)
    fed_lat = anchors_bg["bg_lat"].astype(float).to_numpy()
    fed_lon = anchors_bg["bg_lon"].astype(float).to_numpy()
    print(f"Anchor BGs used for spacing / Section-B: {len(fed_lat)}")

    # Neighbor precompute for rings
    max_cap_across_T = float(max(MAX_SEARCH_MI(t) for t in T_STEPS))
    nei_dists, nei_cumw = _precompute_neighbors(df[LAT_COL], df[LON_COL], df["PotentialBus"], max_cap_across_T)
    print("Precomputed neighbor lists for fast ring recomputation.")

    # ------------------------------------------------------------
    # Phase 1: evaluate ladder; write per-T folders + summary_table
    # ------------------------------------------------------------
    def evaluate_one_scenario(t: float, scenario_idx: int):
        alpha      = float(FDR_ALPHA(t))
        tau        = float(MIN_PASS_RATE_BG(t))
        cmin       = int(MIN_SELECT_COUNT_BG(t))
        demand     = float(TARGET_DEMAND(t))
        cap_mi     = float(MAX_SEARCH_MI(t))
        store_buf  = float(MIN_STORE_BUFFER_MI(t))
        r_fallback = float(max(0.01, MIN_RING_MI_FALLBACK(t)))

        dfx = df.copy()

        r = _rings_from_precomputed(demand, cap_mi, nei_dists, nei_cumw)
        dfx["r_demand_miles_bg"] = r
        dfx["hit_target_within_cap"] = dfx["r_demand_miles_bg"].notna() & (dfx["r_demand_miles_bg"] <= cap_mi)

        filt = (
            (dfx["p_fdr_bg"].fillna(1.0) <= alpha) &
            (dfx["pass_rate_bg"].fillna(0.0) >= tau) &
            (dfx["select_count_bg"].fillna(0) >= cmin) &
            (dfx["hit_target_within_cap"] == True)
        )
        cand = dfx[filt].copy()

        # exclude store anchors from being candidates
        cand = cand[~cand[BG_ID].isin(anchor_ids)].copy()

        cand["r_use_bg"] = np.where(
            cand["r_demand_miles_bg"].notna() & (cand["r_demand_miles_bg"] > 0),
            cand["r_demand_miles_bg"].astype(float),
            float(r_fallback),
        )

        cand = cand.sort_values(["pass_rate_bg", "select_count_bg"], ascending=[False, False]).reset_index(drop=True)

        kept = []
        kept_lat, kept_lon, kept_r = [], [], []

        for _, rrow in cand.iterrows():
            lat, lon = float(rrow[LAT_COL]), float(rrow[LON_COL])
            r_bg     = float(rrow["r_use_bg"])

            # anti-cannibal spacing vs existing footprint
            if fed_lat.size > 0:
                d = _haversine_miles(lat, lon, fed_lat, fed_lon)
                if d.size and float(np.min(d)) < max(r_bg, store_buf):
                    continue

            # spacing among selected candidates within this scenario
            feasible = True
            if kept:
                dists = _haversine_miles(lat, lon, np.array(kept_lat), np.array(kept_lon))
                sep_needed = np.maximum(r_bg, np.array(kept_r))
                if np.any(dists < sep_needed):
                    feasible = False

            if feasible:
                kept.append(rrow)
                kept_lat.append(lat); kept_lon.append(lon); kept_r.append(r_bg)

            if len(kept) >= MAX_SITES_PER_SCENARIO:
                break

        kept_df = pd.DataFrame(kept) if kept else cand.iloc[0:0].copy()

        # scenario metrics (mirror the block script's summary)
        metrics = {
            "scenario_idx": int(scenario_idx),
            "t": float(t),
            "FDR_ALPHA": alpha,
            "MIN_PASS_RATE_BG": tau,
            "MIN_SELECT_COUNT_BG": cmin,
            "TARGET_DEMAND": float(demand),
            "MAX_SEARCH_MI": float(cap_mi),
            "MIN_STORE_BUFFER_MI": float(store_buf),
            "MIN_RING_FALLBACK": float(r_fallback),
            "kept_count": int(len(kept_df)),
            "total_potentialbus": float(kept_df["PotentialBus"].sum()) if len(kept_df) else 0.0,
            "avg_passrate": float(kept_df["pass_rate_bg"].mean()) if len(kept_df) else 0.0,
            "med_ring": float(kept_df["r_use_bg"].median()) if len(kept_df) else 0.0,
        }
        boro_counts = kept_df["borough"].value_counts().to_dict() if len(kept_df) else {}
        for boro in ["Bronx","Brooklyn","Manhattan","Queens","Staten Island","Unknown"]:
            metrics[f"borough_{boro.replace(' ','_').lower()}"] = int(boro_counts.get(boro, 0))

        # write per-t folder
        scen_dir = os.path.join(LADDER_DIR_ROOT, f"ladder_t_{t:0.3f}")
        os.makedirs(scen_dir, exist_ok=True)

        kept_out = kept_df[[BG_ID, "borough", LAT_COL, LON_COL,
                            "PotentialBus", "pass_rate_bg", "select_count_bg",
                            "r_demand_miles_bg", "r_use_bg", "p_fdr_bg"]].copy() if len(kept_df) else kept_df.copy()
        if not kept_out.empty:
            kept_out["t"] = float(t)
            kept_out["scenario_idx"] = int(scenario_idx)
        kept_out.to_csv(os.path.join(scen_dir, "kept_blocks.csv"), index=False)

        _write_json(os.path.join(scen_dir, "metrics.json"), metrics)
        return metrics, kept_df

    rows = []
    all_kept = []
    t0 = time.time()
    for i, t in enumerate(T_STEPS):
        print(f"[LADDER] Evaluating t={t:0.3f} ({i+1}/{TOTAL_SCENARIOS})")
        m, kept_df = evaluate_one_scenario(float(t), int(i))
        rows.append(m)
        if kept_df is not None and not kept_df.empty:
            k = kept_df.copy()
            k["t"] = float(t)
            k["scenario_idx"] = int(i)
            all_kept.append(k)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  ...{i+1} scenarios done | elapsed {elapsed/60.0:.1f} min")

    summary = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    summary_path = os.path.join(LADDER_DIR_ROOT, "summary_table.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✅ Ladder complete. Summary saved: {summary_path}")

    # ------------------------------------------------------------
    # Phase 2: aggregate across all T
    # ------------------------------------------------------------
    agg = pd.concat(all_kept, ignore_index=True) if all_kept else pd.DataFrame()
    agg_out = os.path.join(LADDER_DIR_ROOT, "ALL_T_aggregated_blockgroups.csv")
    agg.to_csv(agg_out, index=False)

    # aggregated-by-BG stats
    if not agg.empty and BG_ID in agg.columns:
        freq = agg.groupby(BG_ID).size().rename("ladder_count").reset_index()
        first = agg.groupby(BG_ID)["t"].min().rename("t_first").reset_index()
        last  = agg.groupby(BG_ID)["t"].max().rename("t_last").reset_index()
        ring_med = agg.groupby(BG_ID)["r_use_bg"].median().rename("ring_med").reset_index()
        lat_med = agg.groupby(BG_ID)[LAT_COL].median().rename(LAT_COL).reset_index()
        lon_med = agg.groupby(BG_ID)[LON_COL].median().rename(LON_COL).reset_index()
        boro = agg.groupby(BG_ID)["borough"].agg(lambda x: x.value_counts().index[0] if len(x) else "Unknown").rename("borough").reset_index()

        pass_med = None
        if "pass_rate_bg" in agg.columns:
            pass_med = agg.groupby(BG_ID)["pass_rate_bg"].median().rename("pass_rate_med").reset_index()

        pbus_sum = None
        if "PotentialBus" in agg.columns:
            try:
                pbus_sum = agg.groupby(BG_ID)["PotentialBus"].sum().rename("pbus_sum").reset_index()
            except Exception:
                # if PotentialBus is non-numeric
                pbus_sum = agg.groupby(BG_ID)["PotentialBus"].apply(lambda x: pd.to_numeric(x, errors="coerce").sum()).rename("pbus_sum").reset_index()


        scored = freq.merge(first, on=BG_ID).merge(last, on=BG_ID).merge(ring_med, on=BG_ID).merge(lat_med, on=BG_ID).merge(lon_med, on=BG_ID).merge(boro, on=BG_ID)
        scored["ladder_freq"] = scored["ladder_count"] / float(TOTAL_SCENARIOS)
        scored["robustness"] = scored["ladder_freq"] * (1.0 - scored["t_first"].fillna(1.0))

        if pass_med is not None:
            scored = scored.merge(pass_med, on=BG_ID, how="left")
        if pbus_sum is not None:
            scored = scored.merge(pbus_sum, on=BG_ID, how="left")


        key_cols = [BG_ID, "pass_rate_bg", "select_count_bg", "eligible_runs_bg", "p_fdr_bg", "PotentialBus"]
        key_cols = [c for c in key_cols if c in df.columns]
        scored = scored.merge(df[key_cols], on=BG_ID, how="left")

        final = scored.sort_values(["robustness", "ladder_count", "pass_rate_bg"], ascending=[False, False, False]).head(10).copy()
    else:
        scored = pd.DataFrame()
        final = pd.DataFrame(columns=[BG_ID, "ladder_count", "t_first", "ladder_freq", "robustness"])

    final_out = os.path.join(LADDER_DIR_ROOT, "final_shortlist_bg.csv")
    final.to_csv(final_out, index=False)

    # ------------------------------------------------------------
    # Phase 3-4: twin clusters + AB similarity
    # ------------------------------------------------------------
    if scored.empty:
        print("[TWINS] No aggregated candidates; skipping twins + similarity + map.")
        _write_run_manifest({
            "outputs": {
                "summary_table": summary_path,
                "ALL_T_aggregated_blockgroups": agg_out,
                "final_shortlist_bg": final_out,
            },
            "anchors_snapped_bg_ids": sorted(list(anchor_ids)),
        })
        return

    # Prepare twin feature table (z-space) for the full BG universe
    base_feat = _prep_twin_features_bg(df)
    models = _fit_cov_models(base_feat, by_borough=TWINS_LOCAL_BY_BORO)

    # centroid dict
    centroids = {str(r[BG_ID]): (float(r[LAT_COL]), float(r[LON_COL])) for _, r in df[[BG_ID, LAT_COL, LON_COL]].iterrows()}

    # adjacency: queen if shapefile available else proximity
    adj = None
    if BG_SHP_PATH and os.path.exists(BG_SHP_PATH) and gpd is not None:
        try:
            shp = gpd.read_file(BG_SHP_PATH)
            # normalize ID
            if BG_ID not in shp.columns:
                for cand in ["GEOID","GEOIDFQ"]:
                    if cand in shp.columns:
                        shp[BG_ID] = shp[cand].astype(str)
                        break
            shp[BG_ID] = shp[BG_ID].astype(str)
            adj = _queen_neighbors_dict_from_shp(shp[[BG_ID, "geometry"]].dropna())
        except Exception as e:
            print(f"[Adjacency] Queen failed ({e}); using proximity adjacency.")
            adj = _proximity_neighbors_dict(df, PROX_ADJ_RADIUS_MI)
    else:
        adj = _proximity_neighbors_dict(df, PROX_ADJ_RADIUS_MI)

    q = 1.0 - float(TWINS_ALPHA)

    # Section B profiles = store anchors
    secB_rows = []
    twin_sizes_rows = []

    for _, ar in anchors_bg.iterrows():
        bid = str(ar[BG_ID])
        res = _contiguous_twins_for_anchor(bid, base_feat, models, adj, centroids, q=q, per_borough=TWINS_LOCAL_BY_BORO)
        cl = res["cluster"]
        rays = _cluster_rays(bid, cl, centroids)
        prof = _cluster_profile(bid, cl, base_feat, centroids, adj)
        secB_rows.append(prof)

        twin_sizes_rows.append({
            "anchor_id": bid,
            "section": "B_store",
            "cluster_size": int(len(cl)),
            "n_twins": int(len(cl) - 1),
            "thr": res.get("thr", np.nan),
            "tested": res.get("tested", 0),
            "n_rays": int(len(rays)),
        })

    sectionB = pd.DataFrame(secB_rows)
    sectionB_out = os.path.join(LADDER_DIR_ROOT, "sectionB_profiles.csv")
    sectionB.to_csv(sectionB_out, index=False)

    # Section A profiles = candidate anchors (from scored list)
    # Optionally cap number of anchors for speed.
    scored_sorted = scored.sort_values(["robustness", "ladder_count"], ascending=[False, False]).reset_index(drop=True)
    if MAX_TWIN_ANCHORS is not None:
        scored_sorted = scored_sorted.head(int(MAX_TWIN_ANCHORS)).copy()

    secA_rows = []
    a_clusters = {}
    a_rays = {}
    for _, cr in scored_sorted.iterrows():
        bid = str(cr[BG_ID])
        res = _contiguous_twins_for_anchor(bid, base_feat, models, adj, centroids, q=q, per_borough=TWINS_LOCAL_BY_BORO)
        cl = res["cluster"]
        rays = _cluster_rays(bid, cl, centroids)
        a_clusters[bid] = cl
        a_rays[bid] = rays
        prof = _cluster_profile(bid, cl, base_feat, centroids, adj)
        secA_rows.append(prof)

        twin_sizes_rows.append({
            "anchor_id": bid,
            "section": "A_candidate",
            "cluster_size": int(len(cl)),
            "n_twins": int(len(cl) - 1),
            "thr": res.get("thr", np.nan),
            "tested": res.get("tested", 0),
            "n_rays": int(len(rays)),
        })

    sectionA = pd.DataFrame(secA_rows)
    sectionA_out = os.path.join(LADDER_DIR_ROOT, "sectionA_profiles.csv")
    sectionA.to_csv(sectionA_out, index=False)

    twin_sizes = pd.DataFrame(twin_sizes_rows)
    twin_sizes_out = os.path.join(LADDER_DIR_ROOT, "twin_cluster_sizes_mahal.csv")
    twin_sizes.to_csv(twin_sizes_out, index=False)

    # A↔B similarity
    matches, breakdown = compute_AB_similarity(sectionA, sectionB)
    matches_out = os.path.join(LADDER_DIR_ROOT, "AB_similarity_matches.csv")
    breakdown_out = os.path.join(LADDER_DIR_ROOT, "AB_component_breakdown.csv")
    matches.to_csv(matches_out, index=False)
    breakdown.to_csv(breakdown_out, index=False)

    # ------------------------------------------------------------
    # Map: aggregated candidates + best match popups
    # ------------------------------------------------------------
    map_out = os.path.join(LADDER_DIR_ROOT, "ALL_T_aggregated_map_bg.html")
    map_cols = [BG_ID, "t_first", "t_last", "ladder_count", "ladder_freq", "ring_med",
                LAT_COL, LON_COL, "borough", "pass_rate_med", "pbus_sum"]
    map_cols = [c for c in map_cols if c in scored_sorted.columns]
    build_aggregated_map_bg(scored[map_cols].copy(),
                            anchors_bg,
                            matches,
                            map_out,
                            bg_centroids=centroids,
                            twin_clusters=a_clusters,
                            twin_rays=a_rays,
                            base_feat=base_feat,
                            models=models,
                            adj=adj,
                            require_adjacent_twin=REQUIRE_ADJACENT_TWIN,
                            min_adjacent_twins=MIN_ADJACENT_TWINS)

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------
    _write_run_manifest({
        "outputs": {
            "summary_table": summary_path,
            "ALL_T_aggregated_blockgroups": agg_out,
            "final_shortlist_bg": final_out,
            "sectionA_profiles": sectionA_out,
            "sectionB_profiles": sectionB_out,
            "AB_component_breakdown": breakdown_out,
            "AB_similarity_matches": matches_out,
            "twin_cluster_sizes_mahal": twin_sizes_out,
            "ALL_T_aggregated_map_bg": map_out,
        },
        "anchors_snapped_bg_ids": sorted(list(anchor_ids)),
        "adjacency_mode": "queen" if (BG_SHP_PATH and os.path.exists(str(BG_SHP_PATH)) and gpd is not None) else f"proximity_{PROX_ADJ_RADIUS_MI:.2f}mi",
        "reverse_geocode": bool(DO_REVERSE_GEOCODE),
    })

    print("✅ Saved outputs in:", LADDER_DIR_ROOT)


def main():
    parser = argparse.ArgumentParser(description="BG Stress-Test Ladder + Aggregated Map + Twins + A↔B Similarity (GitHub version)")
    parser.add_argument("--ladder-only", action="store_true", help="Run ladder and aggregation only (skip twins + map).")
    parser.add_argument("--map-only", action="store_true", help="Build map only from existing ladder output (not implemented; for parity keep full run).")
    args = parser.parse_args()

    if args.map_only:
        raise NotImplementedError("map-only mode is not implemented in this GitHub BG version. Run without flags.")

    if args.ladder_only:
        # For parity, ladder-only still writes per-t folders, summary_table, aggregated csv, final shortlist.
        # It will skip twins/map by temporarily disabling scored -> empty.
        global INCLUDE_TWIN_CLUSTERS_ALL
        INCLUDE_TWIN_CLUSTERS_ALL = False
        # easiest: run full pipeline but stop before twins/map; handled inside run_ladder_and_map if scored empty
        # We'll just run it, but forcibly disable twinning by setting MAX_TWIN_ANCHORS=0.
        global MAX_TWIN_ANCHORS
        MAX_TWIN_ANCHORS = 0

    run_ladder_and_map()


if __name__ == "__main__":
    main()
