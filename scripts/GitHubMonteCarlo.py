# -*- coding: utf-8 -*-
"""
FinalwithCount.py (MIRROR @ BLOCK-GROUP LEVEL) — STOP AFTER MONTE CARLO
======================================================================

You asked:
- Same logic as original FinalwithCount
- But run at BLOCK GROUP level (GEOID_BG)
- No code after Monte Carlo simulation (so: NO thresholding/spacing/maps/audits after MC)
- Assume engineered variables already exist (including z-cols if you have them)

What this script does (same flow as original, up to MC end):
1) (Optional) Snap 15 placeholder anchor points to nearest BGs and EXCLUDE those BGs (mirror of "exclude FedEx BGs")
2) Part 1: PCA + Sobol-weighted KMeans on BGs (features = z_cols), assign Cluster_ID
3) Rank clusters by LocationScore (weighted sum of z-cols)
4) Part 2: Monte Carlo stability loop with checkpoint/resume + STOP file
   - bootstrap subset for fit
   - add noise to z-cols
   - PCA(whiten) per run
   - Sobol-weighted KMeans per run
   - select BGs in clusters ranked 1..TOP_RANKS
   - update counters: elig_runs, select_count
5) Save:
   - bg_results.csv (Part 1 clustering result)
   - mc_counts_bg.csv (eligible_runs_bg, select_count_bg, pass_rate_bg)
   - search_history_part1.csv (Part 1 Sobol search)
   - (MC checkpoints written during runs)

IMPORTANT:
- This script assumes you already have a BG table with engineered variables.
- If you *don’t* already have z-cols, set HAVE_Z_COLS=False and it will robust-scale them here.
"""

import os, time, json, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
from scipy.stats import qmc

warnings.filterwarnings("ignore")

from _config import (
    load_repo_config, cfg_get, resolve_repo_path,
    env_or_cfg, env_or_cfg_bool, env_or_cfg_float, env_or_cfg_int, env_or_cfg_list,
)

# =========================
# ============ PATHS =======
# =========================
# GitHub-friendly defaults (relative paths). Override with env vars if needed:
#   DATA_DIR=/path/to/data  OUTPUTS_DIR=/path/to/outputs  python scripts/GitHubMonteCarlo.py
REPO_ROOT   = Path(__file__).resolve().parents[1]
CFG         = load_repo_config(REPO_ROOT)

DATA_DIR    = resolve_repo_path(REPO_ROOT, env_or_cfg("DATA_DIR", CFG, "paths.data_dir", "data/sample"), REPO_ROOT / "data" / "sample")
OUTPUTS_DIR = resolve_repo_path(REPO_ROOT, env_or_cfg("OUTPUTS_DIR", CFG, "paths.outputs_dir", "outputs"), REPO_ROOT / "outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

BG_DATA = str(resolve_repo_path(REPO_ROOT, env_or_cfg("BG_DATA", CFG, "paths.bg_features_csv", DATA_DIR / "bg_features_nyc.csv"), DATA_DIR / "bg_features_nyc.csv"))   # engineered BG table
OUT_DIR = str(Path(os.environ.get("OUT_DIR", OUTPUTS_DIR / "BG_FINALWITHCOUNT_OUT")))
os.makedirs(OUT_DIR, exist_ok=True)

SIM_ROOT   = os.path.join(OUT_DIR, "MC_SIM")
os.makedirs(SIM_ROOT, exist_ok=True)

CHECKPOINT_FN        = os.path.join(SIM_ROOT, "mc_checkpoint.npz")
CHECKPOINT_META_JSON = os.path.join(SIM_ROOT, "mc_checkpoint_meta.json")
STOP_FILE            = os.path.join(SIM_ROOT, "STOP")  # create this file to stop safely


# =========================
# ====== External exclusion list (BGs to drop from *everything*)
# =========================
# If this file exists, every GEOID_BG listed in it is removed BEFORE Part 1 and therefore
# never participates in clustering, Monte Carlo, or any outputs.
#
# Priority:
#  1) env var EXCLUDED_BG_CSV (explicit)
#  2) outputs/BG_ECON_CLUSTER_OUT/excluded_blockgroups.csv (if it exists)
#  3) examples/outputs/BG_ECON_CLUSTER_OUT/excluded_blockgroups.csv (bundled sample)
EXCLUDE_BG_FILE = True
_default_excl = OUTPUTS_DIR / "BG_ECON_CLUSTER_OUT" / "excluded_blockgroups.csv"
_examples_excl = REPO_ROOT / "examples" / "outputs" / "BG_ECON_CLUSTER_OUT" / "excluded_blockgroups.csv"
EXCLUDED_BG_CSV = os.environ.get(
    "EXCLUDED_BG_CSV",
    str(_default_excl if _default_excl.exists() else _examples_excl),
)


# Output controls
# FinalwithCount writes OUT_DIR/block_results.csv and uses MC_SIM/mc_checkpoint(.npz/.json).
# This BG mirror ALWAYS writes block_results.csv and uses the same checkpoint filenames for compatibility.
SAVE_EXTRA_PART1_FILES = False   # if True, also write bg_results.csv + search_history_part1.csv
SAVE_MC_COUNTS_BG      = True    # writes OUT_DIR/mc_counts_bg.csv (columns use *_bg suffix)
SAVE_MC_COUNTS_BLK     = True    # writes OUT_DIR/mc_counts_blk_style.csv (columns use *_blk suffix for reuse)

# =========================
# ====== BG columns
# =========================
BG_ID = "GEOID_BG"

# If your engineered file already contains robust z-scores:
HAVE_Z_COLS = env_or_cfg_bool("HAVE_Z_COLS", CFG, "monte_carlo.have_z_cols", True)  # set False if you only have raw vars and need z created here

# Engineered (non-competition) vars (BG-level)
ENG_FEATS = [
    "share_college_plus",
    "share_commute_60p",
    "pop_density",
    "occ_units_density",
    "potbus_per_1k",
    "median_income",
]

# If HAVE_Z_COLS=True, the model uses these:
Z_COLS = [f"{c}_z" for c in ENG_FEATS]

# =========================
# ====== LocationScore weights (competition removed)
# (original weights minus comp_per_km2; you can keep or renormalize later)
# =========================
BG_WEIGHTS = {
    "share_college_plus": +0.1181,
    "share_commute_60p":  -0.1725,
    "pop_density":        +0.1198,
    "occ_units_density":  +0.1305,
    "potbus_per_1k":      +0.3039,
    "median_income":      +0.1552,
}

# Optional: override LocationScore weights from config.yaml (cluster_map.weights)
_cfg_w = cfg_get(CFG, "cluster_map.weights", None)
if isinstance(_cfg_w, dict):
    for _k in list(BG_WEIGHTS.keys()):
        if _k in _cfg_w:
            try:
                BG_WEIGHTS[_k] = float(_cfg_w[_k])
            except Exception:
                pass
# =========================
# ====== Part 1 clustering params (mirror)
# =========================
RANDOM_SEED = env_or_cfg_int("RANDOM_SEED", CFG, "monte_carlo.random_seed", 42)
PCA_COMPONENTS_MAX = env_or_cfg_int("PCA_COMPONENTS_MAX", CFG, "monte_carlo.pca_components_max", 6)

SOBOL_SAMPLES = env_or_cfg_int("SOBOL_SAMPLES", CFG, "monte_carlo.sobol_samples", 32)
K_RANGE = env_or_cfg_list("K_RANGE", CFG, "monte_carlo.k_range", list(range(6, 13, 2)))
TRAIN_FRAC = env_or_cfg_float("TRAIN_FRAC", CFG, "monte_carlo.train_frac", 0.75)

MIN_CLUSTER_SIZE_FRAC = env_or_cfg_float("MIN_CLUSTER_SIZE_FRAC", CFG, "monte_carlo.min_cluster_size_frac", 0.012)
MIN_CLUSTER_SIZE_MIN = env_or_cfg_int("MIN_CLUSTER_SIZE_MIN", CFG, "monte_carlo.min_cluster_size_min", 40)    # NOTE: original used 500 (blocks). BGs are fewer; keep logic, adjust minimum.

# =========================
# ====== Part 2 Monte Carlo params (mirror)
# =========================
N_RUNS = env_or_cfg_int("N_RUNS", CFG, "monte_carlo.n_runs", 100)
TOP_RANKS = env_or_cfg_int("TOP_RANKS", CFG, "monte_carlo.top_ranks", 2)
BOOTSTRAP_FRAC = env_or_cfg_float("BOOTSTRAP_FRAC", CFG, "monte_carlo.bootstrap_frac", 0.75)
NOISE_STD = env_or_cfg_float("NOISE_STD", CFG, "monte_carlo.noise_std", 0.03)

CHECKPOINT_EVERY = env_or_cfg_int("CHECKPOINT_EVERY", CFG, "monte_carlo.checkpoint_every", 10)

# =========================
# ====== Anchor placeholders (mirror of "exclude FedEx BGs")
# =========================
EXCLUDE_ANCHOR_BGS = env_or_cfg_bool("EXCLUDE_ANCHOR_BGS", CFG, "monte_carlo.exclude_anchor_bgs", True)

# Option A placeholders — hardcoded lat/lon anchors
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
# If you have centroid columns, use these for snapping anchors:
LAT_COL = "intpt_lat"
LON_COL = "intpt_lon"


# =========================
# ========= Helpers
# =========================
def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 1e-12, None)
    return w / (w.sum() + 1e-12)

def location_score_from_weights(std_df: pd.DataFrame) -> pd.Series:
    val = np.zeros(len(std_df), dtype=float)
    for feat, w in BG_WEIGHTS.items():
        col = feat + "_z"
        if col in std_df.columns:
            z = std_df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
            val += w * z
    return pd.Series(val, index=std_df.index)

def _maybe_build_z(df: pd.DataFrame) -> pd.DataFrame:
    """If HAVE_Z_COLS=False, create robust z-cols from ENG_FEATS."""
    out = df.copy()
    if HAVE_Z_COLS:
        return out
    for c in ENG_FEATS:
        if c not in out.columns:
            out[c] = 0.0
    X = out[ENG_FEATS].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    rs = RobustScaler()
    Z = rs.fit_transform(X.values)
    for i, c in enumerate(ENG_FEATS):
        out[c + "_z"] = Z[:, i]
    return out

def _pick_anchor_bgs(df: pd.DataFrame) -> set:
    """
    Snap 15 anchor points to nearest BG using centroid lat/lon columns.
    If lat/lon missing, deterministically sample 15 BGs (seeded).
    """
    if LAT_COL in df.columns and LON_COL in df.columns:
        tmp = df[[BG_ID, LAT_COL, LON_COL]].dropna().copy()
        if len(tmp) > 50:
            A = tmp[[LAT_COL, LON_COL]].astype(float).values
            bg_ids = tmp[BG_ID].astype(str).values
            picked = []
            for (a_lat, a_lon) in ANCHORS_15:
                d2 = (A[:, 0] - a_lat) ** 2 + (A[:, 1] - a_lon) ** 2
                picked.append(bg_ids[int(np.argmin(d2))])
            return set(pd.unique(picked))
    rng = np.random.default_rng(RANDOM_SEED)
    u = pd.Index(df[BG_ID].astype(str).unique())
    if len(u) < 15:
        return set(u.tolist())
    return set(rng.choice(u, size=15, replace=False).tolist())

def prepare_matrix_for_clustering_bg(df: pd.DataFrame):
    """
    Mirror original: PCA(whiten) on z-cols.
    """
    X = df[Z_COLS].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    n_comp = min(PCA_COMPONENTS_MAX, X.shape[1])
    pca = PCA(n_components=n_comp, whiten=True, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X.values)
    meta = {"feature_cols": Z_COLS, "pca_n_components": n_comp, "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
    return Z, pca, meta

def sobol_weight_k_grid(dim: int, n_samples: int, k_range: list, seed: int):
    """
    Mirror original: Sobol weights in [0,1]^dim + k grid.
    """
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    W = sampler.random(n_samples)
    W = np.clip(W, 1e-12, None)
    W = W / W.sum(axis=1, keepdims=True)

    K = np.array([k for k in k_range for _ in range(n_samples)], dtype=int)
    W2 = np.vstack([W for _ in k_range])
    return W2, K

def score_config(Z: np.ndarray, k: int, weights: np.ndarray, n_total: int):
    """
    Mirror original: train/val split, KMeans, silhouette, DBI, size penalties.
    """
    idx = np.arange(Z.shape[0])
    tr, va = train_test_split(idx, train_size=TRAIN_FRAC, random_state=RANDOM_SEED, shuffle=True)

    w = _normalize_weights(weights)
    Zw = Z * w

    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
    km.fit(Zw[tr])

    cen = km.cluster_centers_
    lab_tr = np.argmin(cdist(Zw[tr], cen), axis=1)
    lab_va = np.argmin(cdist(Zw[va], cen), axis=1)

    sil_tr = silhouette_score(Zw[tr], lab_tr) if k > 1 else -np.inf
    sil_va = silhouette_score(Zw[va], lab_va) if k > 1 else -np.inf
    db_tr  = davies_bouldin_score(Zw[tr], lab_tr) if k > 1 else np.inf

    sizes = np.bincount(lab_tr, minlength=k)
    min_size = max(MIN_CLUSTER_SIZE_MIN, int(MIN_CLUSTER_SIZE_FRAC * n_total))
    small = int((sizes < min_size).sum())
    imbalance = float((sizes.max() - sizes.min()) / max(int(sizes.max()), 1)) if k > 1 else 0.0

    penalty = 0.05 * small + 0.10 * imbalance
    score = float(sil_va) - 0.5 * float(db_tr) - float(penalty)

    details = {
        "sil_train": float(sil_tr),
        "sil_val": float(sil_va),
        "db_train": float(db_tr),
        "sizes_train": sizes.tolist(),
        "penalty": float(penalty),
        "min_size": int(min_size),
    }
    return score, details, km

def hyperparam_search(Z: np.ndarray, n_total: int, seed: int):
    dim = Z.shape[1]
    W, K = sobol_weight_k_grid(dim, SOBOL_SAMPLES, K_RANGE, seed=seed)
    best = {"score": -np.inf, "k": None, "weights": None, "details": None, "km": None}
    hist = []
    for i in range(len(K)):
        w = W[i]; k = int(K[i])
        s, det, km = score_config(Z, k, w, n_total=n_total)
        hist.append({"score": float(s), "k": k, "weights": w.tolist(), **det})
        if s > best["score"]:
            best = {"score": float(s), "k": k, "weights": w, "details": det, "km": km}
    hist_df = pd.DataFrame(hist).sort_values("score", ascending=False).reset_index(drop=True)
    return best, hist_df

def assign_all_labels(Z: np.ndarray, km: KMeans, weights: np.ndarray) -> np.ndarray:
    w = _normalize_weights(weights)
    Zw = Z * w
    return np.argmin(cdist(Zw, km.cluster_centers_), axis=1)

# ---------- checkpoint helpers (same logic, BG renamed) ----------
def _ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _replace_with_retries(src: str, dst: str, attempts: int = 6, delay_sec: float = 0.07):
    for i in range(attempts):
        try:
            if os.path.exists(dst):
                try: os.remove(dst)
                except Exception: pass
            os.replace(src, dst)
            return
        except Exception:
            time.sleep(delay_sec)
    os.replace(src, dst)

def _atomic_save_checkpoint(path_npz: str, path_meta: str,
                            sel_arr: np.ndarray, elig_arr: np.ndarray,
                            runs_done: int, total_runs: int, block_ids: list[str]):
    _ensure_parent_dir(path_npz)
    _ensure_parent_dir(path_meta)

    pid = os.getpid()
    ts  = int(time.time() * 1000)
    tmp_npz  = f"{path_npz}.tmp.{pid}.{ts}"
    tmp_meta = f"{path_meta}.tmp.{pid}.{ts}"

    # NPZ
    with open(tmp_npz, "wb") as f:
        np.savez_compressed(
            f,
            sel=sel_arr.astype(np.int32),
            elig=elig_arr.astype(np.int32),
            runs_done=np.array([runs_done], dtype=np.int32),
            total_runs=np.array([total_runs], dtype=np.int32),
        )
        try:
            f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
    _replace_with_retries(tmp_npz, path_npz)

    # META (store actual order for later realignment)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs_done": int(runs_done),
        "total_runs": int(total_runs),
        "block_ids": block_ids,
    }
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        try:
            f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
    _replace_with_retries(tmp_meta, path_meta)

def _align_checkpoint_arrays(sel_ck: np.ndarray, elig_ck: np.ndarray,
                            old_ids: list[str], cur_ids: list[str]):
    pos = {bid: i for i, bid in enumerate(old_ids)}
    sel_new  = np.zeros(len(cur_ids), dtype=int)
    elig_new = np.zeros(len(cur_ids), dtype=int)
    for j, bid in enumerate(cur_ids):
        i = pos.get(bid)
        if i is not None and i < len(sel_ck):
            sel_new[j]  = int(sel_ck[i])
            elig_new[j] = int(elig_ck[i])
    return sel_new, elig_new

def _load_checkpoint(path_npz: str, path_meta: str, cur_ids: list[str], total_runs_expected: int):
    if not (os.path.exists(path_npz) and os.path.exists(path_meta)):
        return None
    try:
        with open(path_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        old_ids = meta.get("block_ids", meta.get("bg_ids", None))
        runs_done = int(meta.get("runs_done", 0))
        total_runs_ck = int(meta.get("total_runs", 0))
        if old_ids is None or total_runs_ck != total_runs_expected:
            return None

        ck = np.load(path_npz, allow_pickle=True)
        sel_ck  = ck["sel"].astype(int)
        elig_ck = ck["elig"].astype(int)
        sel_aligned, elig_aligned = _align_checkpoint_arrays(sel_ck, elig_ck, old_ids, cur_ids)

        return {"sel": sel_aligned, "elig": elig_aligned, "runs_done": runs_done}
    except Exception:
        return None

def _remove_checkpoint(path_npz: str, path_meta: str):
    try:
        if os.path.exists(path_npz): os.remove(path_npz)
        if os.path.exists(path_meta): os.remove(path_meta)
    except Exception:
        pass

def _print_progress(i_run: int, total: int, t_start: float):
    elapsed = time.time() - t_start
    avg = elapsed / max(1, i_run)
    remain = max(0.0, (total - i_run) * avg)
    def _fmt(sec):
        m = int(sec // 60); s = int(sec % 60)
        return f"{m:02d}:{s:02d}"
    print(f"MC run {i_run:>4d}/{total} | elapsed {_fmt(elapsed)} | est remaining {_fmt(remain)}")

def _stop_requested(stop_path: str) -> bool:
    if os.path.exists(stop_path):
        return True
    base = os.path.splitext(stop_path)[0]
    for cand in glob.glob(base + ".*"):
        if os.path.basename(cand).startswith(os.path.basename(base)):
            return True
    return False


# =========================
# ========== RUN
# =========================
# Load engineered BG table
bg_df = pd.read_csv(BG_DATA)
if BG_ID not in bg_df.columns:
    raise ValueError(f"Expected {BG_ID} in {BG_DATA}.")

bg_df[BG_ID] = bg_df[BG_ID].astype(str)

# Exclude BGs listed in external exclusion file (applies to Part 1 + Monte Carlo)
excluded_set = set()
if EXCLUDE_BG_FILE:
    if os.path.exists(EXCLUDED_BG_CSV):
        ex_df = pd.read_csv(EXCLUDED_BG_CSV, dtype=str)
        if len(ex_df.columns) == 0:
            excluded_set = set()
        else:
            if BG_ID in ex_df.columns:
                id_col = BG_ID
            else:
                # Best-effort: find a GEOID-like column
                id_col = None
                for c in ex_df.columns:
                    lc = str(c).strip().lower()
                    if ("geoid" in lc) and ("bg" in lc):
                        id_col = c
                        break
                if id_col is None:
                    for c in ex_df.columns:
                        lc = str(c).strip().lower()
                        if "geoid" in lc:
                            id_col = c
                            break
                if id_col is None:
                    id_col = ex_df.columns[0]
                print(f"⚠ '{BG_ID}' not found in exclusion file; using column '{id_col}' as IDs.")
            excluded_set = set(
                ex_df[id_col].astype(str).str.strip().replace("", np.nan).dropna().tolist()
            )
        if excluded_set:
            bg_df = bg_df[~bg_df[BG_ID].isin(excluded_set)].copy()
            print(f"🚫 External exclusions applied: removed {len(excluded_set)} BGs from {EXCLUDED_BG_CSV}")
    else:
        print(f"⚠ Exclusion file not found: {EXCLUDED_BG_CSV} (continuing without external exclusions).")


# Ensure z-cols exist if needed
bg_df = _maybe_build_z(bg_df)

for c in Z_COLS:
    if c not in bg_df.columns:
        raise ValueError(f"Missing required z-col: {c}. Set HAVE_Z_COLS=False or fix your engineered table.")

# Exclude 15 anchor BGs (mirror “exclude FedEx BGs entirely”)
if EXCLUDE_ANCHOR_BGS:
    anchor_set = _pick_anchor_bgs(bg_df)
    candidates = bg_df[~bg_df[BG_ID].isin(anchor_set)].copy()
else:
    anchor_set = set()
    candidates = bg_df.copy()

# ========= Part 1: PCA + Sobol KMeans =========
if len(candidates) >= 2:
    Z_cand, pca, meta = prepare_matrix_for_clustering_bg(candidates)
    best, hist_df = hyperparam_search(Z_cand, n_total=len(candidates), seed=RANDOM_SEED)
    cand_labels = assign_all_labels(Z_cand, best["km"], best["weights"]).astype(int)
    candidates["Cluster_ID"] = cand_labels + 1
else:
    pca = PCA(n_components=min(PCA_COMPONENTS_MAX, len(Z_COLS)), whiten=True, random_state=RANDOM_SEED)
    meta = {"feature_cols": Z_COLS, "pca_n_components": 0, "explained_variance_ratio": []}
    best = {"score": float("nan"), "k": 1, "weights": np.ones(min(PCA_COMPONENTS_MAX, len(Z_COLS))), "details": {}, "km": None}
    candidates["Cluster_ID"] = 1
    hist_df = pd.DataFrame()

# Rank clusters by LocationScore
candidates["LocationScore"] = location_score_from_weights(candidates)
means = candidates.groupby("Cluster_ID")["LocationScore"].mean().sort_values(ascending=False)
rank_map = {cluster_id: rank for rank, cluster_id in enumerate(means.index, start=1)}
candidates["Cluster_Rank"] = candidates["Cluster_ID"].map(rank_map).astype(int)

# Persist Part 1 results (MATCH FinalwithCount naming/schema as closely as possible)
# Original FinalwithCount writes:
#   results_cols = [BLOCK_ID, BG_ID, "borough", "LocationScore", "Cluster_ID", "Cluster_Rank"] + z_cols
# At BG level we don't have BLOCK_ID, so we mirror the rest.
results_cols = [BG_ID, "borough", "LocationScore", "Cluster_ID", "Cluster_Rank"] + Z_COLS
results_cols = [c for c in results_cols if c in candidates.columns]
candidates[results_cols].to_csv(os.path.join(OUT_DIR, "block_results.csv"), index=False)

# Optional: keep the extra BG-named duplicates you were writing before
if SAVE_EXTRA_PART1_FILES:
    part1_cols = [BG_ID, "LocationScore", "Cluster_ID", "Cluster_Rank"] + Z_COLS
    candidates[part1_cols].to_csv(os.path.join(OUT_DIR, "bg_results.csv"), index=False)
    hist_df.to_csv(os.path.join(OUT_DIR, "search_history_part1.csv"), index=False)

# ========= Part 2: Monte-Carlo @ BG level =========
cand_bgs = candidates.sort_values(BG_ID).reset_index(drop=True).copy()
bg_ids = cand_bgs[BG_ID].astype(str).tolist()

# --- Pre-clean Z matrix for Monte Carlo (prevents NaN/inf from crashing PCA) ---
X_base = cand_bgs[Z_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
X_base = X_base.fillna(X_base.median()).fillna(0.0)
X_base_np = X_base.to_numpy(dtype=float)

# initialize counters (resume if possible)
resume_state = _load_checkpoint(CHECKPOINT_FN, CHECKPOINT_META_JSON, bg_ids, N_RUNS)
if resume_state is not None:
    sel  = pd.Series(resume_state["sel"],  index=bg_ids, dtype=int)
    elig = pd.Series(resume_state["elig"], index=bg_ids, dtype=int)
    runs_start = int(resume_state["runs_done"])
else:
    sel  = pd.Series(0, index=bg_ids, dtype=int)
    elig = pd.Series(0, index=bg_ids, dtype=int)
    runs_start = 0

t0 = time.time()

try:
    for run in range(runs_start, N_RUNS):
        _print_progress(run, N_RUNS, t0)

        if _stop_requested(STOP_FILE):
            print("🛑 STOP file detected. Saving checkpoint and exiting cleanly...")
            _atomic_save_checkpoint(CHECKPOINT_FN, CHECKPOINT_META_JSON,
                                    sel.values, elig.values,
                                    runs_done=run, total_runs=N_RUNS, block_ids=bg_ids)
            raise SystemExit(0)

        # --- Monte-Carlo iteration ---
        idx = np.random.choice(len(cand_bgs), size=int(max(2, BOOTSTRAP_FRAC * len(cand_bgs))), replace=False)
        fit_df = cand_bgs.iloc[idx].copy()

        X_noisy = X_base_np + np.random.normal(
            0.0, NOISE_STD, size=X_base_np.shape
        )
        fit_noisy = fit_df.copy()
        fit_noisy[Z_COLS] = X_noisy[fit_df.index]

        X_fit = fit_noisy[Z_COLS].to_numpy(dtype=float)
        n_comp = min(PCA_COMPONENTS_MAX, X_fit.shape[1])
        pca_run = PCA(n_components=n_comp, whiten=True, random_state=(RANDOM_SEED + run))
        Z_fit = pca_run.fit_transform(X_fit)

        best_run, _ = hyperparam_search(Z_fit, n_total=len(fit_noisy), seed=(RANDOM_SEED + 81 * run + 7))

        Z_all = pca_run.transform(X_noisy)
        w_norm = _normalize_weights(best_run["weights"]) if best_run.get("weights") is not None else np.ones(Z_all.shape[1])
        Zw_all = Z_all * w_norm
        labels_all = np.argmin(cdist(Zw_all, best_run["km"].cluster_centers_), axis=1)

        tmp = cand_bgs[[BG_ID]].copy()
        tmp["Cluster_ID"] = labels_all + 1
        tmp["LocationScore"] = location_score_from_weights(pd.DataFrame(X_noisy, columns=Z_COLS, index=cand_bgs.index))

        means_run = tmp.groupby("Cluster_ID")["LocationScore"].mean().sort_values(ascending=False)
        rank_map_run = {cid: rank for rank, cid in enumerate(means_run.index, start=1)}
        tmp["Cluster_Rank"] = tmp["Cluster_ID"].map(rank_map_run).astype(int)

        # Selection rule: clusters ranked 1..TOP_RANKS
        top_bgs = tmp[tmp["Cluster_Rank"] <= TOP_RANKS]

        # Update counters
        elig.loc[cand_bgs[BG_ID].astype(str).tolist()] += 1
        sel.loc[top_bgs[BG_ID].astype(str).tolist()] += 1

        # checkpoint
        if (run + 1) % CHECKPOINT_EVERY == 0:
            _atomic_save_checkpoint(CHECKPOINT_FN, CHECKPOINT_META_JSON,
                                    sel.values, elig.values,
                                    runs_done=run + 1, total_runs=N_RUNS, block_ids=bg_ids)
            print(f"💾 Checkpoint saved at run {run + 1}.")

    _remove_checkpoint(CHECKPOINT_FN, CHECKPOINT_META_JSON)
    print("✅ Monte-Carlo complete. Removed checkpoint.")

except KeyboardInterrupt:
    print("\n🛑 KeyboardInterrupt. Saving checkpoint and exiting...")
    runs_done_now = max(0, min(N_RUNS, 'run' in locals() and run or 0))
    _atomic_save_checkpoint(CHECKPOINT_FN, CHECKPOINT_META_JSON,
                            sel.values, elig.values,
                            runs_done=runs_done_now, total_runs=N_RUNS, block_ids=bg_ids)
    raise SystemExit(0)

# ========= END OF MONTE CARLO (STOP HERE per your instruction) =========

# Save raw Monte-Carlo counts (BG-native)
if SAVE_MC_COUNTS_BG:
    mc_counts_bg = pd.DataFrame({
        BG_ID: elig.index.astype(str),
        "eligible_runs_bg": elig.values.astype(int),
        "select_count_bg": sel.values.astype(int),
    })
    mc_counts_bg["pass_rate_bg"] = np.where(
        mc_counts_bg["eligible_runs_bg"] > 0,
        mc_counts_bg["select_count_bg"] / mc_counts_bg["eligible_runs_bg"],
        0.0
    )
    mc_counts_bg.to_csv(os.path.join(OUT_DIR, "mc_counts_bg.csv"), index=False)

# Also write a "blk-style" schema so any downstream code expecting *_blk fields works unchanged.
# (This mirrors the in-memory 'block_counts' created right after the MC loop in FinalwithCount.)
if SAVE_MC_COUNTS_BLK:
    mc_counts_blk = pd.DataFrame({
        BG_ID: elig.index.astype(str),
        "eligible_runs_blk": elig.values.astype(int),
        "select_count_blk": sel.values.astype(int),
    })
    mc_counts_blk["pass_rate_blk"] = np.where(
        mc_counts_blk["eligible_runs_blk"] > 0,
        mc_counts_blk["select_count_blk"] / mc_counts_blk["eligible_runs_blk"],
        0.0
    )
    mc_counts_blk.to_csv(os.path.join(OUT_DIR, "mc_counts_blk_style.csv"), index=False)

# Minimal manifest (optional)
manifest = {
    "unit": "block_group",
    "excluded_anchor_bgs": sorted(list(anchor_set)) if EXCLUDE_ANCHOR_BGS else [],
    "part1": {
        "pca_components_max": PCA_COMPONENTS_MAX,
        "sobol_samples": SOBOL_SAMPLES,
        "k_range": K_RANGE,
        "best": {
            "score": float(best.get("score", np.nan)),
            "k": int(best.get("k", -1)) if best.get("k") is not None else None,
        },
    },
    "monte_carlo": {
        "n_runs": N_RUNS,
        "top_ranks": TOP_RANKS,
        "bootstrap_frac": BOOTSTRAP_FRAC,
        "noise_std": NOISE_STD,
        "checkpoint_every": CHECKPOINT_EVERY,
    },
    "z_cols": Z_COLS,
}
with open(os.path.join(OUT_DIR, "manifest_bg_mc_only.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("✅ Saved:")
print("  -", os.path.join(OUT_DIR, "bg_results.csv"))
print("  -", os.path.join(OUT_DIR, "search_history_part1.csv"))
print("  -", os.path.join(OUT_DIR, "mc_counts_bg.csv"))
print("  -", os.path.join(OUT_DIR, "manifest_bg_mc_only.json"))
