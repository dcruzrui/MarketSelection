
import os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.stats import qmc

warnings.filterwarnings("ignore")

# ================================
# ============ PATHS =============
# ================================
# GitHub-friendly defaults (relative paths). Override with env vars if needed:
#   DATA_DIR=/path/to/data  OUTPUTS_DIR=/path/to/outputs  python scripts/GitHubOptimalSelection.py
REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_DIR   = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data" / "sample"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", REPO_ROOT / "outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

BG_FEATURES_CSV = str(Path(os.environ.get("BG_FEATURES_CSV", DATA_DIR / "bg_features_nyc.csv")))
OUT_DIR = str(Path(os.environ.get("OUT_DIR", OUTPUTS_DIR / f"CLUSTER_OUT_BG_{time.strftime('%Y%m%d_%H%M%S')}")))
os.makedirs(OUT_DIR, exist_ok=True)

# ================================
# ======= CONFIG PARAMETERS ======
# ================================
RANDOM_SEED             = 42
BG_ID                   = "GEOID_BG"

PCA_COMPONENTS_MAX      = 10            # <= #features
K_RANGE                 = (5, 8)        # inclusive
VAL_SIZE                = 0.5
SOBOL_SAMPLES           = 128          # power of 2 avoids Sobol warning
MIN_CLUSTER_SIZE        = 30            # BG-appropriate (blocks often used larger)
WEIGHT_LOW, WEIGHT_HIGH = 0.0, 3.0

np.random.seed(RANDOM_SEED)

# ================================
# ===== FEATURE DEFINITIONS ======
# ================================
FEATURE_COLS = [
    "share_college_plus",
    "share_commute_60p",
    "pop_density",
    "occ_units_density",
    "potbus_per_1k",
    "median_income",          # <--- included
]

# Columns to log1p + winsorize
SKEWED_COLS = [
    "pop_density",
    "occ_units_density",
    "potbus_per_1k",
    "median_income",          # <--- included
]

# ================================
# =========== HELPERS ============
# ================================
def _KMeans(**kwargs) -> KMeans:
    return KMeans(**kwargs)

def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(weights)
    return weights / (norm + 1e-9)

def prepare_matrix_for_clustering(bg_df: pd.DataFrame):
    """
    Match original pipeline style:
    log1p(skewed) -> winsorize 1-99% -> median impute -> robust scale -> PCA(whiten)
    """
    df = bg_df.copy()

    # coerce numeric
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # log1p skewed (guard negatives)
    for c in SKEWED_COLS:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))

    # winsorize 1–99% on skewed
    for c in SKEWED_COLS:
        if c in df.columns and df[c].notna().sum() > 0:
            q01, q99 = df[c].quantile(0.01), df[c].quantile(0.99)
            if pd.notna(q01) and pd.notna(q99) and q01 < q99:
                df[c] = df[c].clip(q01, q99)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df[FEATURE_COLS])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)

    n_comp = min(PCA_COMPONENTS_MAX, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, whiten=True, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X_scaled)

    meta = {
        "unit": "block_group",
        "bg_id_col": BG_ID,
        "feature_cols": FEATURE_COLS,
        "skewed_cols": SKEWED_COLS,
        "pca_n_components": int(n_comp),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return Z, imputer, scaler, pca, meta

def sobol_weight_k_grid(dim: int, samples: int, k_range: tuple[int,int]):
    """
    Sobol in [0,1]^(dim+1)
      - weights in [WEIGHT_LOW, WEIGHT_HIGH] (then L2-normalize when used)
      - k sampled from last dimension
    """
    sampler = qmc.Sobol(d=dim + 1, scramble=True, seed=RANDOM_SEED)
    U = sampler.random(samples)

    W = U[:, :dim] * (WEIGHT_HIGH - WEIGHT_LOW) + WEIGHT_LOW
    k_min, k_max = k_range
    K = (U[:, -1] * (k_max - k_min + 1)).astype(int) + k_min
    return W, K

def score_config(Z: np.ndarray, k: int, weights: np.ndarray, val_size: float = VAL_SIZE):
    """
    Match original scoring structure:
      score = sil_val - 0.5*db_train - (0.05*small + 0.10*imbalance)

    small = number of clusters with size < MIN_CLUSTER_SIZE (train)
    imbalance = (max-min)/max (train)
    """
    if k < 2 or k > len(Z):
        return -np.inf, {"sil_train": np.nan, "sil_val": np.nan, "db_train": np.nan,
                         "sizes_train": [], "penalty": 0.0}, None

    Zw = Z * _normalize_weights(weights)
    Ztr, Zva = train_test_split(Zw, test_size=val_size, random_state=RANDOM_SEED)

    km = _KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit(Ztr)
    lab_tr = km.labels_
    cen = km.cluster_centers_
    lab_va = np.argmin(cdist(Zva, cen), axis=1)

    if len(np.unique(lab_tr)) < 2 or len(np.unique(lab_va)) < 2:
        return -np.inf, {"sil_train": np.nan, "sil_val": np.nan, "db_train": np.nan,
                         "sizes_train": [], "penalty": 0.0}, km

    sil_tr = float(silhouette_score(Ztr, lab_tr))
    sil_va = float(silhouette_score(Zva, lab_va))
    db_tr  = float(davies_bouldin_score(Ztr, lab_tr))

    sizes = np.bincount(lab_tr, minlength=k)
    small = int((sizes < MIN_CLUSTER_SIZE).sum())
    imbalance = float((sizes.max() - sizes.min()) / max(int(sizes.max()), 1))

    penalty = 0.05 * small + 0.10 * imbalance
    score = sil_va - 0.5 * db_tr - penalty

    details = {
        "sil_train": sil_tr,
        "sil_val": sil_va,
        "db_train": db_tr,
        "sizes_train": sizes.tolist(),
        "penalty": float(penalty),
    }
    return float(score), details, km

def hyperparam_search(Z: np.ndarray):
    dim = Z.shape[1]
    W, K = sobol_weight_k_grid(dim, SOBOL_SAMPLES, K_RANGE)

    best = {"score": -np.inf, "k": None, "weights": None, "details": None, "km": None}
    history = []

    for i in range(len(K)):
        w = W[i]
        k = int(K[i])
        s, det, km = score_config(Z, k, w)
        history.append({"score": float(s), "k": k, "weights": w.tolist(), **det})
        if s > best["score"]:
            best = {"score": float(s), "k": k, "weights": w, "details": det, "km": km}

    hist_df = pd.DataFrame(history).sort_values("score", ascending=False).reset_index(drop=True)
    return best, hist_df

def assign_all_labels(Z: np.ndarray, km: KMeans, weights: np.ndarray) -> np.ndarray:
    Zw = Z * _normalize_weights(weights)
    return np.argmin(cdist(Zw, km.cluster_centers_), axis=1)

def compute_effective_feature_weights(pca: PCA, meta: dict, weights: np.ndarray):
    """
    w_norm = L2-normalized PCA weights
    M = P.T @ diag(w_norm^2) @ P
    effective_weight = sqrt(diag(M))
    importance_share = effective_weight / sum(effective_weight)
    """
    w = _normalize_weights(np.asarray(weights, dtype=float))
    P = pca.components_
    M = P.T @ np.diag(w**2) @ P

    feat_influence_sq = np.diag(M)
    feat_influence = np.sqrt(np.maximum(feat_influence_sq, 0))
    importance_share = feat_influence / (feat_influence.sum() + 1e-12)

    eff_df = pd.DataFrame({
        "feature": meta["feature_cols"],
        "effective_weight": feat_influence.astype(float),
        "importance_share": importance_share.astype(float),
    }).sort_values("importance_share", ascending=False).reset_index(drop=True)

    return eff_df, M

def save_artifacts(out_dir: str,
                   bg_df: pd.DataFrame,
                   labels: np.ndarray,
                   imputer: SimpleImputer,
                   scaler: RobustScaler,
                   pca: PCA,
                   best: dict,
                   meta: dict,
                   eff_weights_df: pd.DataFrame,
                   M_equiv: np.ndarray,
                   hist_df: pd.DataFrame) -> None:

    out = bg_df.copy()
    out["Cluster"] = labels.astype(int)
    labels_csv = os.path.join(out_dir, "block_clusters.csv")
    out.to_csv(labels_csv, index=False)

    hist_path = os.path.join(out_dir, "search_history.csv")
    hist_df.to_csv(hist_path, index=False)

    prof_cols = meta["feature_cols"]
    prof = out[["Cluster"] + prof_cols].groupby("Cluster").agg(["mean", "median", "count"])
    prof.to_csv(os.path.join(out_dir, "cluster_profiles.csv"))

    eff_path = os.path.join(out_dir, "effective_feature_weights.csv")
    eff_weights_df.to_csv(eff_path, index=False)

    M_path = os.path.join(out_dir, "equivalent_metric_matrix.csv")
    np.savetxt(M_path, M_equiv, delimiter=",")

    artifacts = {
        "imputer": {"strategy": imputer.strategy, "statistics_": imputer.statistics_.tolist()},
        "scaler": {"type": "RobustScaler", "center_": scaler.center_.tolist(), "scale_": scaler.scale_.tolist()},
        "pca": {
            "n_components": int(pca.n_components_),
            "components": pca.components_.tolist(),
            "mean": pca.mean_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "whiten": True,
        },
        "kmeans": {
            "n_clusters": int(best["k"]),
            "cluster_centers": best["km"].cluster_centers_.tolist(),
            "inertia": float(best["km"].inertia_),
        },
        "weights_pca": _normalize_weights(best["weights"]).tolist(),
        "search": {"best_score": float(best["score"]), **best["details"]},
        "meta": meta,
        "effective_feature_weights": eff_weights_df.to_dict(orient="records"),
        "equivalent_metric_matrix_shape": list(M_equiv.shape),
        "notes": "effective_feature_weights are in the scaled (RobustScaler) feature space."
    }

    with open(os.path.join(out_dir, "model_artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    print("✅ Wrote:")
    print("  -", labels_csv)
    print("  -", hist_path)
    print("  -", os.path.join(out_dir, "cluster_profiles.csv"))
    print("  -", eff_path)
    print("  -", M_path)
    print("  -", os.path.join(out_dir, "model_artifacts.json"))

def main():
    print("▶ Loading BG features:", BG_FEATURES_CSV)
    bg_df = pd.read_csv(BG_FEATURES_CSV, dtype={BG_ID: str})  # :contentReference[oaicite:1]{index=1}

    # NEW: keep only rows where Exclusion == "No"
    if "Exclusion" not in bg_df.columns:
        raise ValueError(
            "Expected column 'Exclusion' in BG_FEATURES_CSV. Add it when you build bg_features_nyc.csv.")

    keep = bg_df["Exclusion"].astype(str).str.strip().str.lower().eq("no")
    dropped = int((~keep).sum())
    print(f"🧹 Exclusion filter: dropped {dropped:,} rows | kept {int(keep.sum()):,}")

    bg_df = bg_df.loc[keep].copy()

    # (optional safety)
    if bg_df.empty:
        raise ValueError("After filtering Exclusion=='No', there are 0 rows left to cluster.")

    missing = [c for c in [BG_ID] + FEATURE_COLS if c not in bg_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in BG_FEATURES_CSV: {missing}")

    print("▶ Preparing matrix (log1p → winsorize → impute → robust scale → PCA whiten)")
    Z, imputer, scaler, pca, meta = prepare_matrix_for_clustering(bg_df)

    print("▶ Sobol search over weights & k ...")
    best, hist_df = hyperparam_search(Z)
    print("✅ Best:", json.dumps({"score": best["score"], "k": best["k"], "details": best["details"]}, indent=2))

    print("▶ Computing effective feature weights ...")
    eff_df, M = compute_effective_feature_weights(pca, meta, best["weights"])
    print(eff_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("▶ Assigning labels ...")
    labels = assign_all_labels(Z, best["km"], best["weights"])

    print("▶ Saving artifacts to:", OUT_DIR)
    save_artifacts(OUT_DIR, bg_df, labels, imputer, scaler, pca, best, meta, eff_df, M, hist_df)

    print("📁 Done. Outputs in:", OUT_DIR)

# =========================================================
# 15 PLACEHOLDER ANCHORS (MTA stations + coordinates)
# =========================================================
ANCHORS_15_MTA = [
    {"station": "137 St-City College",     "lat": 40.822008, "lon": -73.953676},
    {"station": "3 Av-149 St",            "lat": 40.816109, "lon": -73.917757},
    {"station": "Aqueduct-N Conduit Av",  "lat": 40.668234, "lon": -73.834058},
    {"station": "Brook Av",               "lat": 40.807566, "lon": -73.919240},
    {"station": "Castle Hill Av",         "lat": 40.834255, "lon": -73.851222},
    {"station": "Freeman St",             "lat": 40.829993, "lon": -73.891865},
    {"station": "Gates Av",               "lat": 40.689630, "lon": -73.922270},
    {"station": "Grand Army Plaza",       "lat": 40.675235, "lon": -73.971046},
    {"station": "Gun Hill Rd",            "lat": 40.877850, "lon": -73.866256},
    {"station": "Knickerbocker Av",       "lat": 40.698664, "lon": -73.919711},
    {"station": "Mets-Willets Point",     "lat": 40.754622, "lon": -73.845625},
    {"station": "Newkirk Plaza",          "lat": 40.635082, "lon": -73.962793},
    {"station": "Prince St",              "lat": 40.724329, "lon": -73.997702},
    {"station": "St George",              "lat": 40.643748, "lon": -74.073643},
    {"station": "Stapleton",              "lat": 40.627915, "lon": -74.075162},
]

ANCHORS_15_PLACEHOLDER = [(a["lat"], a["lon"]) for a in ANCHORS_15_MTA]

if __name__ == "__main__":
    main()
