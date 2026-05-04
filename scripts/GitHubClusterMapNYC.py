# -*- coding: utf-8 -*-
"""
ClusterMap.py (BLOCK-GROUP MIRROR, INCOME INCLUDED)
===================================================

UPDATED per your request:
0) Respect the user-provided Exclusion flag:
   - We ONLY cluster on rows where Exclusion == "No".
   - BGs with Exclusion != "No" are treated as excluded-from-clustering, written to excluded_blockgroups.csv,
     and rendered as excluded on the map.

1) Removed the density/business eligibility thresholds:
   - No density/business threshold gate.
   - We rank every BG that has features, except those excluded by parks/open-space or water-dominance.

2) Ocean stays uncolored:
   - BG geometry is built by dissolving ALL blocks (including water blocks).
   - Water-dominated BGs are excluded from ranking and rendered transparent on the map (no grey ocean).

3) Excluded BGs are visible:
   - Any BG inside NYC not used in ranking (parks, water-dominated, or missing features) is rendered grey
     (water-dominated BGs are transparent to keep the ocean white).

4) Anchor stars:
   - No raw-lat/lon anchor stars.
   - We add a simple purple star at the representative point of each snapped anchor BG (guaranteed inside the polygon).

Open-space filter:
- Uses an open-space GeoJSON (parks/open areas) placed in the same folder as this script.
- Computes open_space_share = area(BG ∩ open_space) / area(BG) in EPSG:2263
- Excludes if open_space_share >= OPENSPACE_SHARE_MIN_TO_EXCLUDE (default 0.60)
"""

import os
import json
import warnings
from pathlib import Path
from decimal import Decimal, InvalidOperation
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import folium
except Exception:
    folium = None

try:
    from folium.plugins import BeautifyIcon
except Exception:
    BeautifyIcon = None

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from scipy.stats import qmc

warnings.filterwarnings("ignore")

from _config import (
    load_repo_config, cfg_get, resolve_repo_path,
    env_or_cfg, env_or_cfg_bool, env_or_cfg_float, env_or_cfg_int, env_or_cfg_list,
)


# =======================
# ============ PATHS =====
# =======================
# GitHub-friendly defaults (relative paths). Override with env vars if needed:
#   DATA_DIR=/path/to/data  OUTPUTS_DIR=/path/to/outputs  python scripts/GitHubClusterMapNYC.py
REPO_ROOT   = Path(__file__).resolve().parents[1]
CFG         = load_repo_config(REPO_ROOT)

# Repo-relative paths with config.yaml support (env vars override config)
DATA_DIR    = resolve_repo_path(REPO_ROOT, env_or_cfg("DATA_DIR", CFG, "paths.data_dir", "data/sample"), REPO_ROOT / "data" / "sample")
OUTPUTS_DIR = resolve_repo_path(REPO_ROOT, env_or_cfg("OUTPUTS_DIR", CFG, "paths.outputs_dir", "outputs"), REPO_ROOT / "outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep legacy variable name for minimal changes below:
SCRIPT_DIR = DATA_DIR

BG_FEATURES_PATH = str(resolve_repo_path(REPO_ROOT, env_or_cfg("BG_FEATURES_CSV", CFG, "paths.bg_features_csv", SCRIPT_DIR / "bg_features_nyc.csv"), SCRIPT_DIR / "bg_features_nyc.csv"))
OUT_DIR = str(Path(os.environ.get("OUT_DIR", OUTPUTS_DIR / "BG_ECON_CLUSTER_OUT")))
os.makedirs(OUT_DIR, exist_ok=True)

# Optional: fallback BG layer (only used if we cannot find a blocks file)
BG_GEOM_FALLBACK_PATH = os.environ.get("BG_GEOM_PATH", str(SCRIPT_DIR / "filtered_blockgroups.geojson")).strip()

# If you don't have a BLOCKS file to dissolve, we'll try to load a BG boundary file directly.
# (Common GitHub/public setup: a BG layer + features CSV.)
BG_GEOM_CANDIDATES = [
    "nyc_block_groups_2021.zip",
    "nyc_block_groups_2021.shp",
    "nyc_block_groups_2021.geojson",
    "filtered_blockgroups.geojson",
    "filtered_block_groups.geojson",
    "block_groups.geojson",
    "blockgroups.geojson",
    "nyc_block_groups.geojson",
]

def _resolve_bg_geom_path() -> Optional[str]:
    # If user set BG_GEOM_PATH, honor it.
    if BG_GEOM_FALLBACK_PATH:
        p = Path(BG_GEOM_FALLBACK_PATH)
        if not p.is_absolute():
            p = SCRIPT_DIR / p
        if p.exists():
            return str(p)

    # Otherwise, try common filenames in the script folder.
    for rel in BG_GEOM_CANDIDATES:
        fp = SCRIPT_DIR / rel
        if fp.exists():
            return str(fp)

    return None

# =======================
# ===== MAP TITLE =======
# =======================
MAP_TITLE = str(env_or_cfg("MAP_TITLE", CFG, "cluster_map.map_title", "NYC Block Group Suitability Map (Cluster-Based Ranking)")).strip()

# =======================
# == BLOCKS GEOMETRY =====
# =======================
# We want FULL BG geometry (including water), so we dissolve ALL blocks.
BLOCKS_GEOM_PATH = os.environ.get("BLOCKS_GEOM_PATH", "").strip()

BLOCKS_GEOM_CANDIDATES = [
    "nyc_blocks_only.shp",
    "nyc_blocks_only.geojson",
    "nyc_blocks_only.gpkg",
    "nyc_blocks.shp",
    "nyc_blocks.geojson",
    "filtered_blocks.geojson",
    "filtered_blocks.shp",
]

def _find_first_existing(rel_names):
    for rel in rel_names:
        fp = SCRIPT_DIR / rel
        if fp.exists():
            return str(fp)
    return None

def _resolve_blocks_path():
    if BLOCKS_GEOM_PATH:
        p = Path(BLOCKS_GEOM_PATH)
        if not p.is_absolute():
            p = SCRIPT_DIR / p
        if p.exists():
            return str(p)
    return _find_first_existing(BLOCKS_GEOM_CANDIDATES)

def _guess_block_id_col(cols):
    preferred = ["GEOID20", "GEOID", "GEOID10", "BLOCK_ID", "geoid20", "geoid"]
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        if str(c).upper().startswith("GEOID"):
            return c
    return None

def _guess_aland_col(cols):
    preferred = ["ALAND20", "ALAND", "aland20", "aland"]
    for c in preferred:
        if c in cols:
            return c
    return None

def _guess_awater_col(cols):
    preferred = ["AWATER20", "AWATER", "awater20", "awater"]
    for c in preferred:
        if c in cols:
            return c
    return None


# =======================
# ====== FEATURES =======
# =======================
FEATURE_COLS = [
    "share_college_plus",
    "share_commute_60p",
    "pop_density",
    "occ_units_density",
    "potbus_per_1k",
    "median_income",
]

LOG1P_COLS = ["pop_density", "occ_units_density", "potbus_per_1k", "median_income"]

# =========================
# == LocationScore weights
# =========================
BG_WEIGHTS = {
    "share_college_plus": +0.1181,
    "share_commute_60p":  -0.1725,
    "pop_density":        +0.1198,
    "occ_units_density":  +0.1305,
    "potbus_per_1k":      +0.3039,
    "median_income":      +0.1552,
}

# Optional: override LocationScore weights from config.yaml
_cfg_w = cfg_get(CFG, "cluster_map.weights", None)
if isinstance(_cfg_w, dict):
    for _k in list(BG_WEIGHTS.keys()):
        if _k in _cfg_w:
            try:
                BG_WEIGHTS[_k] = float(_cfg_w[_k])
            except Exception:
                pass

# =======================
# == BG ELIGIBILITY =====
# =======================
# NOTE: We do NOT filter BGs by density/business thresholds.
# The only exclusions are:
#   - parks/open areas: open_space_share >= OPENSPACE_SHARE_MIN_TO_EXCLUDE
#   - water-dominated BGs: land_share_bg < LAND_SHARE_BG_MIN (geometry kept; ocean stays uncolored)
#   - missing features (shown grey)

# =========================
# == OPEN SPACE / PARKS MASK
# =========================
# Put your open-space file next to this script.
OPENSPACE_GEOJSON_DEFAULT_NAME = "6fc98808-ab15-43a8-bc0d-d3a906e2937e.geojson"
OPENSPACE_PATH_ENV = os.environ.get("OPENSPACE_PATH", "").strip()

OPENSPACE_SHARE_MIN_TO_EXCLUDE = env_or_cfg_float("OPENSPACE_SHARE_MIN_TO_EXCLUDE", CFG, "cluster_map.open_space_share_min_to_exclude", 0.60)
APPLY_OPENSPACE_MASK = env_or_cfg_bool("APPLY_OPENSPACE_MASK", CFG, "cluster_map.apply_open_space_mask", True)
REQUIRE_OPENSPACE_MASK = env_or_cfg_bool("REQUIRE_OPENSPACE_MASK", CFG, "cluster_map.require_open_space_mask", True)

# =========================
# == WATER DOMINANCE MASK ==
# =========================
# Exclude BGs that are mostly water, but DO NOT drop geometry.
# Default: exclude if land_share_bg < 0.10 (i.e., <10% land).
LAND_SHARE_BG_MIN = env_or_cfg_float("LAND_SHARE_BG_MIN", CFG, "cluster_map.land_share_bg_min", 0.10)
APPLY_WATER_MASK = env_or_cfg_bool("APPLY_WATER_MASK", CFG, "cluster_map.apply_water_mask", True)

# =======================
# ==== SEARCH PARAMS =====
# =======================
RANDOM_SEED = env_or_cfg_int("RANDOM_SEED", CFG, "cluster_map.random_seed", 42)
VAL_FRAC = env_or_cfg_float("VAL_FRAC", CFG, "cluster_map.val_frac", 0.25)
WINSOR_P = env_or_cfg_float("WINSOR_P", CFG, "cluster_map.winsor_p", 0.01)

PCA_COMPONENTS_MAX = env_or_cfg_int("PCA_COMPONENTS_MAX", CFG, "cluster_map.pca_components_max", 6)
K_VALUES = env_or_cfg_list("K_VALUES", CFG, "cluster_map.k_values", [5])
N_SOBOL = env_or_cfg_int("N_SOBOL", CFG, "cluster_map.n_sobol", 80)

DBI_PENALTY_W = env_or_cfg_float("DBI_PENALTY_W", CFG, "cluster_map.penalty.dbi_w", 0.50)
SMALLCLUST_PENALTY_W = env_or_cfg_float("SMALLCLUST_PENALTY_W", CFG, "cluster_map.penalty.small_cluster_w", 0.50)
IMBALANCE_PENALTY_W = env_or_cfg_float("IMBALANCE_PENALTY_W", CFG, "cluster_map.penalty.imbalance_w", 0.10)
MIN_CLUSTER_FRAC = env_or_cfg_float("MIN_CLUSTER_FRAC", CFG, "cluster_map.penalty.min_cluster_frac", 0.01)

# ==========================
# == 15 PLACEHOLDER ANCHORS
# ==========================
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

# =======================
# ====== COLORS =========
# =======================
RANK_COLOR_LIST = [
    "#b3de69",
    "#80b1d3",
    "#bebada",
    "#fb8072",
    "#fdb462",
    "#8dd3c7",
    "#ccebc5",
    "#fccde5",
]

EXCLUDED_BG_COLOR  = "#bdbdbd"
EXCLUDED_BG_BORDER = "#bdbdbd"

def rank_to_color(rank: Any) -> str:
    try:
        r = int(rank)
    except Exception:
        return EXCLUDED_BG_COLOR
    if r <= 0:
        return EXCLUDED_BG_COLOR
    return RANK_COLOR_LIST[(r - 1) % len(RANK_COLOR_LIST)]


# =========================================================
# ===================== HELPERS ===========================
# =========================================================
def _normalize_geoid_val(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.startswith("1500000US"):
        s = s.replace("1500000US", "")
    if s.endswith(".0"):
        s = s[:-2]
    if "e" in s.lower():
        try:
            s = str(int(Decimal(s)))
        except (InvalidOperation, ValueError):
            pass
    s = s.strip()
    if s.isdigit() and len(s) < 12:
        s = s.zfill(12)
    return s

def _normalize_geoid_bg(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    if "GEOID_BG" not in g.columns:
        if "GEOID" in g.columns:
            g["GEOID_BG"] = g["GEOID"]
        elif "GEOIDFQ" in g.columns:
            g["GEOID_BG"] = g["GEOIDFQ"]
        else:
            raise ValueError("Features must contain GEOID_BG (or GEOID / GEOIDFQ).")

    g["GEOID_BG"] = g["GEOID_BG"].map(_normalize_geoid_val)
    g = g[g["GEOID_BG"].astype(str).str.len() > 0].copy()
    g["GEOID_BG"] = g["GEOID_BG"].astype(str)
    return g

def _winsorize_df(X: pd.DataFrame, p: float) -> pd.DataFrame:
    Xw = X.copy()
    for c in Xw.columns:
        lo = Xw[c].quantile(p)
        hi = Xw[c].quantile(1 - p)
        Xw[c] = Xw[c].clip(lo, hi)
    return Xw

def _safe_log1p(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            v = pd.to_numeric(out[c], errors="coerce")
            out[c] = np.log1p(v.clip(lower=0))
    return out

def _robust_z_inplace(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    X = out[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(med)
    rs = RobustScaler()
    Z = rs.fit_transform(X.values)
    for i, c in enumerate(cols):
        out[c + "_z"] = Z[:, i]
    return out

def _location_score(df_with_z: pd.DataFrame) -> pd.Series:
    s = np.zeros(len(df_with_z), dtype=float)
    for feat, w in BG_WEIGHTS.items():
        col = feat + "_z"
        if col in df_with_z.columns:
            z = df_with_z[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
            s += float(w) * z
    return pd.Series(s, index=df_with_z.index)


# =========================================================
# ============ GEOMETRY + LAND/WATER SHARES ===============
# =========================================================
def _build_bg_geom_from_blocks(blocks_path: str) -> "gpd.GeoDataFrame":
    """
    Build FULL BG geometries by dissolving ALL blocks to BG (including water blocks).
    """
    if gpd is None:
        raise RuntimeError("geopandas not installed.")

    b0 = gpd.read_file(blocks_path)
    bid = _guess_block_id_col(b0.columns)
    if bid is None:
        raise ValueError(f"Could not find a block GEOID column in {list(b0.columns)}")

    b0 = b0.dropna(subset=[bid]).copy()
    b0[bid] = b0[bid].astype(str)

    b0["GEOID_BG"] = b0[bid].astype(str).str.slice(0, 12).map(_normalize_geoid_val)
    b0 = b0[b0["GEOID_BG"].astype(str).str.len() > 0].copy()
    b0["GEOID_BG"] = b0["GEOID_BG"].astype(str)

    bg = b0[["GEOID_BG", "geometry"]].dissolve(by="GEOID_BG", as_index=False)
    try:
        bg["geometry"] = bg["geometry"].buffer(0)
    except Exception:
        pass
    return bg

def _add_landwater_shares_from_blocks(bg_gdf: "gpd.GeoDataFrame", blocks_path: str) -> "gpd.GeoDataFrame":
    """
    Add land_share_bg and water_share_bg using ALAND/AWATER summed from blocks.

    land_share_bg = ALAND / (ALAND + AWATER) aggregated to BG
    water_share_bg = AWATER / (ALAND + AWATER) aggregated to BG

    If ALAND/AWATER not present, falls back to land_share_bg=1, water_share_bg=0.
    """
    if gpd is None:
        raise RuntimeError("geopandas not installed.")

    out = bg_gdf.copy()

    # If columns already exist, drop them so merge doesn't create _x/_y suffixes
    for c in ["land_share_bg", "water_share_bg"]:
        if c in out.columns:
            out = out.drop(columns=[c])

    b0 = gpd.read_file(blocks_path)
    bid = _guess_block_id_col(b0.columns)
    aland_col = _guess_aland_col(b0.columns)
    awater_col = _guess_awater_col(b0.columns)

    if bid is None or aland_col is None or awater_col is None:
        print("⚠️ Blocks file missing GEOID/ALAND/AWATER columns; cannot compute land/water shares from blocks.")
        # Leave as NaN so we can fall back to BG boundary attributes (ALAND/AWATER) if available.
        out["land_share_bg"] = np.nan
        out["water_share_bg"] = np.nan
        return out

    b0 = b0.dropna(subset=[bid]).copy()
    b0[bid] = b0[bid].astype(str)

    # BG id = first 12 digits of block GEOID
    b0["GEOID_BG"] = b0[bid].astype(str).str.slice(0, 12).map(_normalize_geoid_val)
    b0 = b0[b0["GEOID_BG"].astype(str).str.len() > 0].copy()
    b0["GEOID_BG"] = b0["GEOID_BG"].astype(str)

    aland = pd.to_numeric(b0[aland_col], errors="coerce").fillna(0.0)
    awater = pd.to_numeric(b0[awater_col], errors="coerce").fillna(0.0)

    stats = (
        pd.DataFrame({"GEOID_BG": b0["GEOID_BG"], "aland": aland, "awater": awater})
        .groupby("GEOID_BG", as_index=False)[["aland", "awater"]]
        .sum()
    )

    denom = (stats["aland"] + stats["awater"]).replace(0, np.nan)
    stats["land_share_bg"] = (stats["aland"] / denom).fillna(0.0).clip(0.0, 1.0)
    stats["water_share_bg"] = (stats["awater"] / denom).fillna(0.0).clip(0.0, 1.0)

    out = out.merge(stats[["GEOID_BG", "land_share_bg", "water_share_bg"]], on="GEOID_BG", how="left")

    # Defaults if missing (e.g., BG not present in stats)
    out["land_share_bg"] = pd.to_numeric(out["land_share_bg"], errors="coerce")
    out["water_share_bg"] = pd.to_numeric(out["water_share_bg"], errors="coerce")

    return out


def _add_landwater_shares_from_bg_attrs(bg_gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Compute land_share_bg/water_share_bg using ALAND/AWATER on the BG layer itself."""
    if gpd is None:
        raise RuntimeError("geopandas not installed.")

    out = bg_gdf.copy()
    aland_c = _guess_aland_col(out.columns)
    awater_c = _guess_awater_col(out.columns)

    if aland_c is None or awater_c is None:
        return out

    aland_v = pd.to_numeric(out[aland_c], errors="coerce").fillna(0.0)
    awater_v = pd.to_numeric(out[awater_c], errors="coerce").fillna(0.0)
    denom = (aland_v + awater_v).replace(0, np.nan)

    out["land_share_bg"] = (aland_v / denom).fillna(0.0).clip(0.0, 1.0).astype(float)
    out["water_share_bg"] = (awater_v / denom).fillna(0.0).clip(0.0, 1.0).astype(float)
    return out


def _ensure_landwater_shares(bg_gdf: "gpd.GeoDataFrame", blocks_path: Optional[str]) -> "gpd.GeoDataFrame":
    """Best-effort: compute land/water shares from blocks if available, else from BG attrs."""
    if gpd is None:
        raise RuntimeError("geopandas not installed.")
    out = bg_gdf.copy()

    # If already present with at least some non-null values, keep them.
    if ("land_share_bg" in out.columns) and out["land_share_bg"].notna().any():
        if "water_share_bg" not in out.columns:
            out["water_share_bg"] = 1.0 - pd.to_numeric(out["land_share_bg"], errors="coerce").fillna(1.0)
        return out

    # Prefer blocks aggregation when possible (most accurate when you dissolved from blocks)
    if blocks_path:
        try:
            out = _add_landwater_shares_from_blocks(out, blocks_path)
            # If any BGs still lack shares (common when block file lacks ALAND/AWATER or merge misses a BG),
            # try to fill from BG boundary attributes (ALAND/AWATER) when present.
            if ("land_share_bg" in out.columns) and out["land_share_bg"].isna().any():
                try:
                    tmp = _add_landwater_shares_from_bg_attrs(out)
                    if "land_share_bg" in tmp.columns:
                        out["land_share_bg"] = out["land_share_bg"].fillna(tmp["land_share_bg"])
                    if "water_share_bg" in tmp.columns:
                        if "water_share_bg" not in out.columns:
                            out["water_share_bg"] = tmp["water_share_bg"]
                        else:
                            out["water_share_bg"] = out["water_share_bg"].fillna(tmp["water_share_bg"])
                except Exception:
                    pass
            if out["land_share_bg"].notna().any():
                return out
        except Exception as e:
            print("⚠️ Could not compute land/water shares from blocks:", repr(e))

    # Fall back to BG attrs (ALAND/AWATER on BG boundaries)
    try:
        out = _add_landwater_shares_from_bg_attrs(out)
    except Exception as e:
        print("⚠️ Could not compute land/water shares from BG attrs:", repr(e))

    # If still missing, set NaNs
    if "land_share_bg" not in out.columns:
        out["land_share_bg"] = np.nan
    if "water_share_bg" not in out.columns:
        out["water_share_bg"] = np.nan

    return out



# =========================================================
# ================== OPEN SPACE MASK ======================
# =========================================================
def _resolve_open_space_path() -> Optional[str]:
    if OPENSPACE_PATH_ENV:
        p = Path(OPENSPACE_PATH_ENV)
        if not p.is_absolute():
            p = SCRIPT_DIR / p
        if p.exists():
            return str(p)

    p0 = SCRIPT_DIR / OPENSPACE_GEOJSON_DEFAULT_NAME
    if p0.exists():
        return str(p0)

    # lightweight heuristic scan
    try:
        for fp in SCRIPT_DIR.iterdir():
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in (".geojson", ".shp", ".gpkg"):
                continue
            name = fp.name.lower()
            if ("park" in name) or ("open" in name) or ("openspace" in name) or ("open-space" in name):
                return str(fp)
    except Exception:
        pass

    return None

def _add_open_space_share(bg_gdf: "gpd.GeoDataFrame", open_path: str) -> "gpd.GeoDataFrame":
    """
    Add open_space_share = area(BG ∩ open_space_union) / area(BG)
    Computed in EPSG:2263 (NYC-friendly CRS).
    """
    if gpd is None:
        raise RuntimeError("geopandas not installed.")

    out = bg_gdf.copy()
    out["open_space_share"] = 0.0

    if (not open_path) or (not Path(open_path).exists()):
        msg = f"Open-space layer not found: {open_path}"
        if REQUIRE_OPENSPACE_MASK:
            raise FileNotFoundError(msg)
        print("⚠️ " + msg + " (skipping open-space masking)")
        return out

    open_gdf = gpd.read_file(open_path)
    if open_gdf.empty:
        msg = f"Open-space layer is empty: {open_path}"
        if REQUIRE_OPENSPACE_MASK:
            raise ValueError(msg)
        print("⚠️ " + msg + " (skipping open-space masking)")
        return out

    if open_gdf.crs is None:
        print("⚠️ Open-space layer has no CRS; assuming EPSG:4326.")
        open_gdf = open_gdf.set_crs(4326, allow_override=True)

    if out.crs is None:
        print("⚠️ BG geometry has no CRS; assuming EPSG:4326.")
        out = out.set_crs(4326, allow_override=True)

    target_crs = 2263
    bg_m = out.to_crs(target_crs).copy()
    open_m = open_gdf.to_crs(target_crs).copy()

    try:
        bg_m["geometry"] = bg_m["geometry"].buffer(0)
    except Exception:
        pass
    try:
        open_m["geometry"] = open_m["geometry"].buffer(0)
    except Exception:
        pass

    open_union = open_m.geometry.unary_union

    bg_area = bg_m.geometry.area.replace(0, np.nan)
    inter_area = bg_m.geometry.intersection(open_union).area
    share = (inter_area / bg_area).fillna(0.0).clip(0.0, 1.0)

    out["open_space_share"] = share.to_numpy()
    return out


# =========================================================
# ================== PCA + SOBOL SEARCH ===================
# =========================================================
def _prep_matrix_for_clustering(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(med)

    X = _safe_log1p(X, LOG1P_COLS)
    X = _winsorize_df(X, WINSOR_P)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X.values)

    n_comp = min(PCA_COMPONENTS_MAX, Xs.shape[1])
    pca = PCA(n_components=n_comp, whiten=True, random_state=RANDOM_SEED)
    Z = pca.fit_transform(Xs)

    meta = {"scaler": scaler, "pca": pca, "n_comp": int(n_comp)}
    return Z, meta

def _sobol_weights(d: int, n: int) -> np.ndarray:
    sampler = qmc.Sobol(d=d, scramble=True, seed=RANDOM_SEED)
    W = sampler.random(n)
    W = np.clip(W, 1e-12, None)
    W = W / W.sum(axis=1, keepdims=True)
    return W

def _score_cfg(Z: np.ndarray, k: int, w: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    idx = np.arange(Z.shape[0])
    tr, va = train_test_split(idx, test_size=VAL_FRAC, random_state=RANDOM_SEED, shuffle=True)

    if k < 2 or k >= len(tr):
        return -1e9, {"reason": "k_invalid"}

    w = np.asarray(w, dtype=float)
    w = w / (w.sum() + 1e-12)
    s = np.sqrt(w + 1e-12)

    Ztr = Z[tr] * s
    Zva = Z[va] * s

    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
    ytr = km.fit_predict(Ztr)
    yva = km.predict(Zva)

    if len(np.unique(yva)) < 2:
        return -1e9, {"reason": "degenerate_validation"}

    sil = float(silhouette_score(Zva, yva))
    ch = float(calinski_harabasz_score(Zva, yva))
    dbi = float(davies_bouldin_score(Zva, yva))

    counts = np.bincount(ytr, minlength=k).astype(float)
    min_size = max(2, int(MIN_CLUSTER_FRAC * len(ytr)))
    small_frac = float((counts < min_size).mean())
    imbalance = float(np.std(counts) / (np.mean(counts) + 1e-12))

    score = (
        sil
        + 0.0005 * ch
        - DBI_PENALTY_W * dbi
        - SMALLCLUST_PENALTY_W * small_frac
        - IMBALANCE_PENALTY_W * imbalance
    )

    return score, {
        "silhouette_val": sil,
        "calinski_harabasz_val": ch,
        "dbi_val": dbi,
        "small_cluster_frac": small_frac,
        "imbalance": imbalance,
    }

def _search_best(Z: np.ndarray) -> Tuple[Dict[str, Any], pd.DataFrame]:
    d = Z.shape[1]
    W = _sobol_weights(d, N_SOBOL)

    rows = []
    best = {"score": -1e18, "k": None, "w": None, "details": None}

    for i, w in enumerate(W):
        for k in K_VALUES:
            sc, det = _score_cfg(Z, k, w)
            row = {"iter": int(i), "k": int(k), "score": float(sc), **det}
            for j in range(d):
                row[f"w_pca{j+1}"] = float(w[j])
            rows.append(row)

            if sc > best["score"]:
                best = {"score": float(sc), "k": int(k), "w": w.copy(), "details": det}

    hist = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return best, hist

def _fit_final_kmeans(Z: np.ndarray, best: Dict[str, Any]) -> np.ndarray:
    k = int(best["k"])
    w = np.asarray(best["w"], dtype=float)
    w = w / (w.sum() + 1e-12)
    s = np.sqrt(w + 1e-12)
    Zs = Z * s
    km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_SEED)
    return km.fit_predict(Zs)


# =========================================================
# =================== MAP OUTPUT ==========================
# =========================================================
def _folium_map(bg_gdf: "gpd.GeoDataFrame", out_html: str) -> None:
    if folium is None:
        print("⚠️ folium not installed; skipping HTML map.")
        return

    g = bg_gdf.to_crs(4326).copy()

    bounds = g.total_bounds
    center = [(bounds[1] + bounds[3]) / 2.0, (bounds[0] + bounds[2]) / 2.0]

    m = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True)

    # Title overlay
    if MAP_TITLE:
        title_html = f"""
        <div style="
            position: fixed;
            top: 10px; left: 50%;
            transform: translateX(-50%);
            z-index: 999999;
            background: rgba(255,255,255,0.95);
            padding: 10px 14px;
            border: 1px solid #888;
            border-radius: 6px;
            font-size: 18px;
            font-weight: 600;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        ">
          {MAP_TITLE}
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

    # White canvas
    m.get_root().header.add_child(folium.Element("""
    <style>
      .leaflet-container { background: #ffffff; }
      .anchor-star { background: transparent !important; border: none !important; }
    </style>
    """))

    # Optional basemap layer (hidden by default)
    try:
        folium.TileLayer("cartodb positron", name="Basemap (CartoDB Positron)", show=False).add_to(m)
    except Exception:
        pass



    # Fit bounds
    pad_lat = (bounds[3] - bounds[1]) * 0.20
    pad_lon = (bounds[2] - bounds[0]) * 0.20
    try:
        m.fit_bounds([
            [bounds[1] - pad_lat, bounds[0] - pad_lon],
            [bounds[3] + pad_lat, bounds[2] + pad_lon],
        ])
    except Exception:
        pass

    # Excluded BGs = those excluded from clustering (preferred), else Cluster_Rank is NaN (fallback)
    if "is_excluded_bg" in g.columns:
        ex_mask = g["is_excluded_bg"].fillna(False).astype(bool)
    elif "Cluster_Rank" in g.columns:
        ex_mask = g["Cluster_Rank"].isna()
    else:
        ex_mask = pd.Series(False, index=g.index)

    excluded = g[ex_mask].copy()
    if not excluded.empty:
        # Keep land_share_bg/open_space_share so we can avoid painting the ocean grey.
        excl_cols = ["geometry"]
        for c in ["land_share_bg", "water_share_bg", "ALAND", "AWATER", "ALAND20", "AWATER20", "open_space_share"]:
            if c in excluded.columns and c not in excl_cols:
                excl_cols.insert(0, c)
        excluded_view = excluded[excl_cols].copy()

        def _excl_style(feat):
            land = feat["properties"].get("land_share_bg", None)
            try:
                land = float(land)
                if np.isnan(land):
                    land = None
            except Exception:
                land = None

            # If land_share_bg is missing, try to compute from ALAND/AWATER first,
            # then fall back to water_share_bg.
            if land is None:
                aland = feat["properties"].get("ALAND", None)
                if aland is None:
                    aland = feat["properties"].get("ALAND20", None)
                awater = feat["properties"].get("AWATER", None)
                if awater is None:
                    awater = feat["properties"].get("AWATER20", None)
                try:
                    aland = float(aland) if aland is not None else None
                except Exception:
                    aland = None
                try:
                    awater = float(awater) if awater is not None else None
                except Exception:
                    awater = None

                if (aland is not None) and (awater is not None) and (aland + awater) > 0:
                    land = float(aland) / float(aland + awater)
                else:
                    water = feat["properties"].get("water_share_bg", None)
                    try:
                        water = float(water)
                        if np.isnan(water):
                            water = None
                    except Exception:
                        water = None
                    if (water is not None) and (water > (1.0 - LAND_SHARE_BG_MIN)):
                        land = 0.0
                    else:
                        land = None

            # If we still couldn't infer land share (common when using a GeoJSON that lacks
            # ALAND/AWATER and we don't have a blocks file), DO NOT paint it grey.
            # This prevents grey ocean when water-only BGs are present but lack attributes.
            if land is None:
                open_share = feat["properties"].get("open_space_share", None)
                try:
                    open_share = float(open_share)
                    if np.isnan(open_share):
                        open_share = None
                except Exception:
                    open_share = None

                # If it looks like a park/open-space dominated BG, show it grey.
                if (open_share is not None) and (open_share >= OPENSPACE_SHARE_MIN_TO_EXCLUDE):
                    return {
                        "fillColor": EXCLUDED_BG_COLOR,
                        "color": EXCLUDED_BG_BORDER,
                        "weight": 0.4,
                        "fillOpacity": 0.6,
                    }

                # Otherwise keep transparent (white background shows through).
                return {"fillColor": "#ffffff", "color": "#ffffff", "weight": 0.0, "fillOpacity": 0.0}


            # If it's water-dominated, render transparent (so ocean stays uncolored).
            if APPLY_WATER_MASK and (land < LAND_SHARE_BG_MIN):
                return {"fillColor": "#ffffff", "color": "#ffffff", "weight": 0.0, "fillOpacity": 0.0}

            return {
                "fillColor": EXCLUDED_BG_COLOR,
                "color": EXCLUDED_BG_BORDER,
                "weight": 0.4,
                "fillOpacity": 0.6,
            }

        folium.GeoJson(
            data=excluded_view.to_json(),
            name="Excluded Block Groups",
            style_function=_excl_style,
        ).add_to(m)

    if "Cluster_Rank" in g.columns:
        ranked = g[(~ex_mask) & (~g["Cluster_Rank"].isna())].copy()
    else:
        ranked = g.iloc[0:0].copy()
    if ranked.empty:
        folium.LayerControl(collapsed=True).add_to(m)
        m.save(out_html)
        print("✅ Saved HTML map:", out_html)
        return

    max_rank = int(pd.to_numeric(ranked["Cluster_Rank"], errors="coerce").max())

    def style_fn(feat):
        # Never paint water-dominated polygons (keeps ocean from being shaded).
        land = feat["properties"].get("land_share_bg", None)
        try:
            land = float(land)
            if np.isnan(land):
                land = None
        except Exception:
            land = None
        if land is None:
            aland = feat["properties"].get("ALAND", None)
            if aland is None:
                aland = feat["properties"].get("ALAND20", None)
            awater = feat["properties"].get("AWATER", None)
            if awater is None:
                awater = feat["properties"].get("AWATER20", None)
            try:
                aland = float(aland) if aland is not None else None
            except Exception:
                aland = None
            try:
                awater = float(awater) if awater is not None else None
            except Exception:
                awater = None

            if (aland is not None) and (awater is not None) and (aland + awater) > 0:
                land = float(aland) / float(aland + awater)
            else:
                water = feat["properties"].get("water_share_bg", None)
                try:
                    water = float(water)
                    if np.isnan(water):
                        water = None
                except Exception:
                    water = None
                if (water is not None) and (water > (1.0 - LAND_SHARE_BG_MIN)):
                    land = 0.0
                else:
                    land = 1.0
        if APPLY_WATER_MASK and (land < LAND_SHARE_BG_MIN):
            return {"fillColor": "#ffffff", "color": "#ffffff", "weight": 0.0, "fillOpacity": 0.0}
        r = feat["properties"].get("Cluster_Rank")
        col = rank_to_color(r)

        cid = feat["properties"].get("Cluster_ID")
        try:
            cid_i = int(cid)
        except Exception:
            cid_i = None
        w = 1.2 if cid_i == 0 else 0.25

        return {"fillColor": col, "color": "#000000", "weight": w, "fillOpacity": 0.70}

    keep_cols = [c for c in [
        "GEOID_BG", "Cluster_ID", "Cluster_Rank", "LocationScore",
        "open_space_share", "land_share_bg", "water_share_bg", "ALAND", "AWATER", "ALAND20", "AWATER20",
        "geometry"
    ] if c in ranked.columns]
    layer_df = ranked[keep_cols].copy()

    tooltip_fields = [f for f in [
        "GEOID_BG", "Cluster_ID", "Cluster_Rank", "LocationScore",
        "open_space_share", "land_share_bg"
    ] if f in layer_df.columns]

    aliases_map = {
        "GEOID_BG": "Block Group GEOID",
        "Cluster_ID": "Cluster ID (0=Anchor)",
        "Cluster_Rank": "Rank (1=Best)",
        "LocationScore": "LocationScore",
    }
    tooltip_aliases = [aliases_map.get(f, f) for f in tooltip_fields]

    folium.GeoJson(
        data=layer_df.to_json(),
        name="Ranked Block Groups",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True, sticky=False),
    ).add_to(m)


    # Anchor BG stars (purple) — placed on the snapped anchor BG polygons (not raw lat/lon).
    if "is_anchor_bg" in g.columns:
        anch = g[g["is_anchor_bg"].fillna(False).astype(bool)].copy()
        if not anch.empty:
            anchor_fg = folium.FeatureGroup(name="Anchor BGs (stars)", show=True)
            for _, row in anch.iterrows():
                geom = row.geometry
                try:
                    pt = geom.representative_point()
                except Exception:
                    pt = geom.centroid
                star_svg = """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="22" height="22">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87L18.18 22 12 18.77 5.82 22 7 14.14 2 9.27l6.91-1.01z"
                        fill="#6a3d9a" stroke="#ffffff" stroke-width="1.4"/>
                </svg>
                """
                folium.Marker(
                    location=[pt.y, pt.x],
                    tooltip=f"Anchor BG: {row.get('GEOID_BG','')}",
                    z_index_offset=1000000,
                    icon=folium.DivIcon(
                        html=star_svg,
                        class_name="anchor-star",
                        icon_size=(22, 22),
                        icon_anchor=(11, 11),
                    ),
                ).add_to(anchor_fg)
            anchor_fg.add_to(m)

    legend_swatches = "".join(
        f"<div><span style='display:inline-block;width:12px;height:12px;background:{rank_to_color(i)};border:1px solid #333;margin-right:6px;'></span>Rank {i}</div>"
        for i in range(1, max_rank + 1)
    )

    legend_html = f"""
    <div style="
         position: fixed; bottom: 20px; left: 20px; z-index: 9999;
         background: rgba(255,255,255,0.95); padding: 10px 12px; border: 1px solid #888;
         font-size: 13px; line-height: 1.3; max-width: 380px;">
      <div style="font-weight:600; margin-bottom:6px;">Legend</div>
      {legend_swatches}
      <div>
        <span style="display:inline-block;width:12px;height:12px;background:{EXCLUDED_BG_COLOR};border:1px solid {EXCLUDED_BG_BORDER};margin-right:6px;"></span>
        Excluded BGs (Exclusion!=No / parks / water-dominated / missing features)
      </div>
      <div style="margin-top:6px; font-size:12px; color:#444;">
        Parks excluded if open_space_share ≥ {OPENSPACE_SHARE_MIN_TO_EXCLUDE:.2f}.<br>
        Water-dominated BGs (land_share_bg &lt; {LAND_SHARE_BG_MIN:.2f}) are excluded and rendered transparent (ocean stays uncolored).<br>
        BGs with Exclusion!=No and BGs missing features are shown grey.
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=True).add_to(m)

    m.save(out_html)
    print("✅ Saved HTML map:", out_html)


# =========================================================
# ======================= MAIN PIPELINE ===================
# =========================================================
def run_bg_econ_cluster_pipeline(
    bg_features: pd.DataFrame,
    bg_geom: Optional["gpd.GeoDataFrame"] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    if gpd is None:
        raise RuntimeError("geopandas not installed.")

    # ---- Load features ----
    df = _normalize_geoid_bg(bg_features).drop_duplicates("GEOID_BG").copy()

    # ---- User-provided Exclusion flag (STRICT) ----
    # Only rows where Exclusion == "No" are eligible to be clustered.
    # Anything else ("Yes", blank, NaN, etc.) is treated as excluded-from-clustering.
    if "Exclusion" not in df.columns:
        df["Exclusion"] = "No"
    df["Exclusion"] = df["Exclusion"].astype(str).str.strip()
    df["Exclusion_norm"] = df["Exclusion"].str.lower()

    # Ensure feature columns exist (on the full feature table)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Strict include set for clustering
    df_cluster = df[df["Exclusion_norm"] == "no"].copy()
    excluded_by_flag_set = set(df.loc[df["Exclusion_norm"] != "no", "GEOID_BG"].astype(str))

    # ---- Build full BG geometry (including water) ----
    if bg_geom is None:
        blocks_path = _resolve_blocks_path()
        if blocks_path:
            print(f"🧩 Building FULL BG geometry from blocks (includes water): {blocks_path}")
            bg_geom = _build_bg_geom_from_blocks(blocks_path)
        else:
            bg_path = _resolve_bg_geom_path()
            if bg_path:
                print("🧩 Loading BG geometry (no blocks file found):", bg_path)
                bg_geom = gpd.read_file(bg_path)
            else:
                print("⚠️ No BG geometry file found. Falling back to:", BG_GEOM_FALLBACK_PATH)
                bg_geom = gpd.read_file(BG_GEOM_FALLBACK_PATH)

    gg = bg_geom.copy()

    # Normalize GEOID_BG in geometry
    if "GEOID_BG" in gg.columns:
        gg["GEOID_BG"] = gg["GEOID_BG"].map(_normalize_geoid_val)
    elif "GEOID" in gg.columns:
        gg["GEOID_BG"] = gg["GEOID"].map(_normalize_geoid_val)
    elif "GEOIDFQ" in gg.columns:
        gg["GEOID_BG"] = gg["GEOIDFQ"].map(_normalize_geoid_val)
    else:
        raise ValueError("BG geometry must contain GEOID_BG (or GEOID / GEOIDFQ).")

    gg = gg[gg["GEOID_BG"].astype(str).str.len() > 0].copy()
    gg["GEOID_BG"] = gg["GEOID_BG"].astype(str)

    # ---- Add land/water shares (so we can EXCLUDE water BGs without dropping geometry) ----
    # Order:
    #   1) from blocks (if you provided a blocks layer)
    #   2) from BG boundaries ALAND/AWATER (e.g., nyc_block_groups_2021.zip)
    landwater_blocks_path = _resolve_blocks_path()
    if APPLY_WATER_MASK:
        gg = _ensure_landwater_shares(gg, landwater_blocks_path)

        # If still missing (rare), warn once.
        try:
            n_ok = pd.to_numeric(gg["land_share_bg"], errors="coerce").notna().sum()
        except Exception:
            n_ok = 0
        if n_ok == 0:
            print("⚠️ APPLY_WATER_MASK=1 but land_share_bg could not be computed; ocean masking may be inaccurate.")
    else:
        gg["land_share_bg"] = 1.0
        gg["water_share_bg"] = 0.0

    # ---- Add open-space share (parks/open areas) ----
    if APPLY_OPENSPACE_MASK:
        open_path = _resolve_open_space_path()
        if open_path is None:
            msg = "APPLY_OPENSPACE_MASK=1 but no open-space file found next to script."
            if REQUIRE_OPENSPACE_MASK:
                raise FileNotFoundError(msg)
            print("⚠️ " + msg + " (skipping open-space mask)")
            gg["open_space_share"] = 0.0
        else:
            print(f"🌳 Open-space layer: {open_path}")
            gg = _add_open_space_share(gg, open_path)
    else:
        gg["open_space_share"] = 0.0

    # ---- Merge *cluster-eligible* features (Exclusion == "No") onto geometry ----
    bg_gdf = gg.merge(df_cluster, on="GEOID_BG", how="inner")
    if len(bg_gdf) == 0:
        raise RuntimeError(
            "Merge produced 0 rows after applying Exclusion==No. "
            "Check that GEOID_BG formats match between features and geometry, and that Exclusion is populated."
        )

    # ============================
    # Eligibility / exclusions (for CLUSTERING)
    # ============================
    # NOTE: No density/business gating. For clustering we exclude:
    #   - Exclusion != "No" (handled upstream by df_cluster)
    #   - parks/open-space dominated BGs
    #   - water-dominated BGs (geometry kept; rendered transparent so ocean stays uncolored)

    # Parks/open-space exclusion (applies to cluster-eligible BGs)
    open_share = pd.to_numeric(bg_gdf.get("open_space_share", 0.0), errors="coerce").fillna(0.0)
    open_ok = (open_share < OPENSPACE_SHARE_MIN_TO_EXCLUDE) if APPLY_OPENSPACE_MASK else pd.Series(True, index=bg_gdf.index)

    # Water exclusion (water-dominated BGs). If land_share_bg is missing, treat as pass (prevents accidental ocean-grey).
    land_share = pd.to_numeric(bg_gdf.get("land_share_bg", np.nan), errors="coerce")
    land_share_filled = land_share.fillna(1.0)
    water_ok = (land_share_filled >= LAND_SHARE_BG_MIN) if APPLY_WATER_MASK else pd.Series(True, index=bg_gdf.index)

    eligible = open_ok & water_ok

    # ---- Rank universe (this is what actually gets clustered) ----
    bg_gdf_rank = bg_gdf[eligible].copy()
    if len(bg_gdf_rank) == 0:
        raise RuntimeError(
            "All block groups were filtered out after applying Exclusion==No and the geometry filters. "
            "Try increasing OPENSPACE_SHARE_MIN_TO_EXCLUDE, or lowering LAND_SHARE_BG_MIN."
        )

    # ======================================================
    # Build the MASTER excluded list (ALL BGs excluded from clustering)
    # ======================================================
    geom_ids = gg["GEOID_BG"].astype(str)
    clustered_ids = bg_gdf_rank["GEOID_BG"].astype(str)
    excluded_ids_set = set(geom_ids) - set(clustered_ids)

    # Start from geometry to ensure everything shaded can be drawn
    excl_geom = gg.loc[gg["GEOID_BG"].isin(excluded_ids_set)].copy()

    # Merge diagnostics from the FULL feature table (includes Exclusion==Yes rows)
    diag_cols = ["GEOID_BG", "Exclusion"] + [c for c in FEATURE_COLS if c in df.columns]
    diag = df[diag_cols].copy()
    excl = excl_geom.merge(diag, on="GEOID_BG", how="left")

    # Reason flags
    feat_all_set = set(df["GEOID_BG"].astype(str))
    excl_open_share = pd.to_numeric(excl.get("open_space_share", 0.0), errors="coerce").fillna(0.0)
    excl_land_share = pd.to_numeric(excl.get("land_share_bg", np.nan), errors="coerce")

    excl["excluded_by_flag"] = excl["GEOID_BG"].isin(excluded_by_flag_set).astype(int)
    excl["excluded_open_space"] = (
        (APPLY_OPENSPACE_MASK) & (excl_open_share >= OPENSPACE_SHARE_MIN_TO_EXCLUDE)
    ).astype(int)
    excl["excluded_water_dominated"] = (
        (APPLY_WATER_MASK) & excl_land_share.notna() & (excl_land_share < LAND_SHARE_BG_MIN)
    ).astype(int)
    excl["excluded_missing_features"] = (~excl["GEOID_BG"].isin(feat_all_set)).astype(int)
    excl["excluded_missing_geometry"] = 0

    # Ensure the CSV includes ALL Exclusion!=No BGs, even if they are missing in the geometry layer (rare).
    missing_geom_flagged = sorted(list(excluded_by_flag_set - set(geom_ids)))
    if missing_geom_flagged:
        extra = (
            df.loc[df["GEOID_BG"].isin(missing_geom_flagged), diag_cols]
            .drop_duplicates("GEOID_BG")
            .copy()
        )
        extra["open_space_share"] = np.nan
        extra["land_share_bg"] = np.nan
        extra["water_share_bg"] = np.nan
        extra["excluded_by_flag"] = 1
        extra["excluded_open_space"] = 0
        extra["excluded_water_dominated"] = 0
        extra["excluded_missing_features"] = 0
        extra["excluded_missing_geometry"] = 1
        excl = pd.concat([excl, extra], ignore_index=True, sort=False)

    # ---- Write excluded CSV (no geometry) ----
    out_excl = os.path.join(OUT_DIR, "excluded_blockgroups.csv")

    # Column ordering (only include cols that exist)
    cols_out = [
        "GEOID_BG",
        "Exclusion",
        "excluded_by_flag",
        "excluded_open_space",
        "excluded_water_dominated",
        "excluded_missing_features",
        "excluded_missing_geometry",
        "open_space_share",
        "land_share_bg",
        "water_share_bg",
    ]
    # add feature diagnostics if available
    cols_out += [c for c in FEATURE_COLS if c in excl.columns]
    # de-duplicate while preserving order
    seen = set()
    cols_out = [c for c in cols_out if (c in excl.columns) and (not (c in seen or seen.add(c)))]

    excl_out = excl[cols_out].copy()
    # Make Exclusion readable for missing-feature rows
    if "Exclusion" in excl_out.columns:
        excl_out["Exclusion"] = excl_out["Exclusion"].fillna("").astype(str)

    # Sort so Exclusion!=No cases appear first
    sort_cols = [
        "excluded_by_flag",
        "excluded_missing_features",
        "excluded_open_space",
        "excluded_water_dominated",
        "excluded_missing_geometry",
        "GEOID_BG",
    ]
    sort_cols = [c for c in sort_cols if c in excl_out.columns]
    excl_out = excl_out.sort_values(
        by=sort_cols,
        ascending=[False] * (len(sort_cols) - 1) + [True],
        kind="mergesort",
    )

    excl_out.to_csv(out_excl, index=False)

    print(f"🧹 Cluster universe (Exclusion==No & passed geometry filters): {len(bg_gdf_rank):,}")
    print(
        f"🧹 Excluded BGs total (incl. Exclusion!=No + filter fails + missing features): {len(excl_out):,}"
    )
    print(f"   ✅ Wrote excluded CSV: {out_excl}")

    # ---- Anchors (simple: snap anchors by nearest centroid) ----
    # If you prefer your prior anchor snapping function, you can re-add it.
    gg4326 = bg_gdf_rank.to_crs(4326).copy()
    pts_on_surface = gg4326.geometry.representative_point()
    gg4326["intpt_lat"] = pts_on_surface.y
    gg4326["intpt_lon"] = pts_on_surface.x

    pts = gg4326[["GEOID_BG", "intpt_lat", "intpt_lon"]].dropna().copy()
    A = pts[["intpt_lat", "intpt_lon"]].values
    bg_ids = pts["GEOID_BG"].astype(str).values

    picked = []
    for (a_lat, a_lon) in ANCHORS_15:
        d2 = (A[:, 0] - a_lat) ** 2 + (A[:, 1] - a_lon) ** 2
        picked.append(bg_ids[int(np.argmin(d2))])

    anchor_bgs = pd.Index(pd.unique(picked)).astype(str)
    bg_gdf_rank["is_anchor_bg"] = bg_gdf_rank["GEOID_BG"].isin(anchor_bgs)

    anchors = bg_gdf_rank[bg_gdf_rank["is_anchor_bg"]].copy()
    candidates = bg_gdf_rank[~bg_gdf_rank["is_anchor_bg"]].copy()

    # ---- Clustering ----
    Z, meta = _prep_matrix_for_clustering(candidates)
    best, hist = _search_best(Z)
    cand_labels = _fit_final_kmeans(Z, best)

    candidates = candidates.copy()
    candidates["Cluster_ID"] = cand_labels.astype(int) + 1
    anchors = anchors.copy()
    anchors["Cluster_ID"] = 0

    merged = pd.concat([candidates, anchors], axis=0, ignore_index=True)

    # ---- LocationScore ----
    merged = _robust_z_inplace(merged, FEATURE_COLS)
    merged["LocationScore"] = _location_score(merged)

    z_cols = [c + "_z" for c in FEATURE_COLS]

    # ---- Rank clusters by mean LocationScore ----
    means = merged.groupby("Cluster_ID")["LocationScore"].mean().sort_values(ascending=False)
    rank_map = {int(cid): int(r) for r, cid in enumerate(means.index.tolist(), start=1)}
    merged["Cluster_Rank"] = merged["Cluster_ID"].map(rank_map).astype(int)

    # ---- Save outputs ----
    results_cols = [
        "GEOID_BG", "is_anchor_bg", "Cluster_ID", "Cluster_Rank",
        "LocationScore", "open_space_share", "land_share_bg", "water_share_bg"
    ] + z_cols
    for c in results_cols:
        if c not in merged.columns and c in bg_gdf_rank.columns:
            merged[c] = bg_gdf_rank[c]

    results = merged[results_cols].copy()

    out_results = os.path.join(OUT_DIR, "bg_econ_results.csv")
    out_hist = os.path.join(OUT_DIR, "bg_search_history.csv")
    out_manifest = os.path.join(OUT_DIR, "bg_manifest.json")

    results.to_csv(out_results, index=False)
    hist.to_csv(out_hist, index=False)

    manifest = {
        "unit": "block_group",
        "random_seed": RANDOM_SEED,
        "features": FEATURE_COLS,
        "log1p_cols": LOG1P_COLS,
        "winsor_p": WINSOR_P,
        "pca_components": int(meta["n_comp"]),
        "k_values": [int(k) for k in K_VALUES],
        "n_sobol": int(N_SOBOL),
        "eligibility": {
            "density_business_thresholds_used": False,
            "exclusion_flag_column": "Exclusion",
            "exclusion_flag_rule": "only Exclusion==No is clustered",
            "open_space_share_exclude_ge": OPENSPACE_SHARE_MIN_TO_EXCLUDE,
            "land_share_bg_min": LAND_SHARE_BG_MIN,
            "excluded_csv": "excluded_blockgroups.csv",
            "cluster_universe_count": int(len(bg_gdf_rank)),
            "excluded_total": int(len(excl_out)),
            "excluded_by_flag": int(excl_out.get("excluded_by_flag", pd.Series(dtype=float)).sum()) if "excluded_by_flag" in excl_out.columns else None,
            "excluded_open_space": int(excl_out.get("excluded_open_space", pd.Series(dtype=float)).sum()) if "excluded_open_space" in excl_out.columns else None,
            "excluded_water_dominated": int(excl_out.get("excluded_water_dominated", pd.Series(dtype=float)).sum()) if "excluded_water_dominated" in excl_out.columns else None,
            "excluded_missing_features": int(excl_out.get("excluded_missing_features", pd.Series(dtype=float)).sum()) if "excluded_missing_features" in excl_out.columns else None,
            "excluded_missing_geometry": int(excl_out.get("excluded_missing_geometry", pd.Series(dtype=float)).sum()) if "excluded_missing_geometry" in excl_out.columns else None,
        },
        "best": {
            "score": float(best["score"]),
            "k": int(best["k"]),
            "details": best["details"],
            "w_pca": [float(x) for x in best["w"]],
        },
        "rank_map": rank_map,
        "location_score_weights": BG_WEIGHTS,
        "open_space_file": str(_resolve_open_space_path()),
        "blocks_file": str(_resolve_blocks_path()),
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ---- Map layer (left merge results onto ALL geometry) ----
    merged_gdf = gg.merge(
        results[["GEOID_BG", "is_anchor_bg", "Cluster_ID", "Cluster_Rank", "LocationScore"]],
        on="GEOID_BG",
        how="left",
    )
    # Drive the excluded shading from the master excluded list (not just NaN ranks)
    merged_gdf["is_excluded_bg"] = merged_gdf["GEOID_BG"].astype(str).isin(excluded_ids_set)

    out_html = os.path.join(OUT_DIR, "bg_rank_map.html")
    try:
        _folium_map(merged_gdf, out_html)
    except Exception as e:
        print("⚠️ Could not write folium map:", repr(e))

    print("✅ Saved:")
    print("  -", out_results)
    print("  -", out_hist)
    print("  -", out_manifest)
    print("  -", out_excl)

    return results, hist, manifest


if __name__ == "__main__":
    if gpd is None:
        raise RuntimeError("Install geopandas to run this script.")

    if not os.path.exists(BG_FEATURES_PATH):
        raise FileNotFoundError(f"Missing features CSV: {BG_FEATURES_PATH}")

    bg_features = pd.read_csv(BG_FEATURES_PATH, dtype=str)
    run_bg_econ_cluster_pipeline(bg_features=bg_features, bg_geom=None)
