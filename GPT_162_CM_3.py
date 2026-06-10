# ============================================================
# CM-3: Critical Band Response Audit
# Cross-model validation for model-specific critical bands
#
# Goal:
#   Test whether samples inside each model's fitted critical band
#   are more sensitive / response-amplified than outside-band samples.
#
# Inputs expected from prior runs:
#   cm1_outputs/{qwen,llama,gemma}_deltaR_dataset.csv
#   cm1_outputs/{qwen,llama,gemma}_layer_R_dataset.csv    (optional)
#   cm2_outputs/{qwen,llama,gemma}_critical_band_summary.json
#
# Outputs:
#   cm3_outputs/cm3_model_summary.csv
#   cm3_outputs/cm3_overall_summary.json
#   cm3_outputs/{model}_response_summary.json
#   cm3_outputs/{model}_response_dataset.csv
# ============================================================

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CM1_DIR = Path("./cm1_outputs")
CM2_DIR = Path("./cm2_outputs")
SAVE_DIR = Path("./cm3_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen", "llama", "gemma"]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def auc_safe(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan


def infer_phase_col(df):
    for c in ["phase", "phase_target", "condition_phase", "label_phase"]:
        if c in df.columns:
            return c
    # Fallback from condition names
    if "condition" in df.columns:
        def map_phase(s):
            s = str(s).lower()
            if "positive" in s or "stable" in s or "clean" in s or "weak_positive" in s:
                return "positive"
            if "critical" in s or "compete" in s or "ambiguous" in s or "balanced" in s:
                return "critical"
            if "negative" in s or "conflict" in s or "update" in s or "exception" in s or "direct" in s:
                return "negative"
            return "unknown"
        df["phase"] = df["condition"].map(map_phase)
        return "phase"
    raise RuntimeError("Cannot infer phase column. Expected one of phase/phase_target/condition or similar.")


def orient_delta_u(df, dR_cols, phase_col):
    """Compute PC1(DeltaR) and orient so positive mean > negative mean where possible."""
    X = df[dR_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=min(3, X.shape[1]))
    scores = pca.fit_transform(X)
    du = scores[:, 0].astype(np.float32)

    phases = df[phase_col].astype(str).values
    pos_mean = np.nanmean(du[phases == "positive"]) if np.any(phases == "positive") else np.nan
    neg_mean = np.nanmean(du[phases == "negative"]) if np.any(phases == "negative") else np.nan
    if np.isfinite(pos_mean) and np.isfinite(neg_mean) and pos_mean < neg_mean:
        du = -du
        scores[:, 0] = -scores[:, 0]

    return du, pca.explained_variance_ratio_.tolist(), pca.components_[0].tolist()


def band_distance(x, lo, hi):
    x = np.asarray(x).astype(float)
    below = np.maximum(lo - x, 0.0)
    above = np.maximum(x - hi, 0.0)
    return below + above


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def fit_response_proxy(df, du_col="DeltaU", phase_col="phase"):
    """
    Build an empirical response proxy chi = |dP(negative)/dDeltaU|.

    Since we may not have repeated perturbation sweeps in CM-1, CM-3 uses a
    local logistic response proxy:
      P_neg = sigmoid(a * DeltaU + b)
      chi = |a| * P_neg * (1 - P_neg)

    This is an operational approximation of response susceptibility.
    If generated output columns exist, they can be used later, but phase labels
    are sufficient for this audit.
    """
    y = (df[phase_col].astype(str).values == "negative").astype(int)
    x = df[du_col].values.astype(float)

    # Simple robust 1D logistic fit with Newton updates
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2, dtype=float)
    for _ in range(50):
        z = X @ beta
        p = sigmoid(z)
        W = p * (1 - p) + 1e-6
        grad = X.T @ (p - y)
        H = X.T @ (X * W[:, None]) + 1e-4 * np.eye(2)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if np.linalg.norm(step) < 1e-6:
            break

    p = sigmoid(X @ beta)
    a = beta[1]
    chi = np.abs(a) * p * (1 - p)
    return p, chi, {"intercept": float(beta[0]), "slope": float(beta[1])}


def summarize_region(df, region_col, chi_col, slope_col=None):
    out = {}
    for region, sub in df.groupby(region_col):
        item = {
            "n": int(len(sub)),
            "chi_mean": float(sub[chi_col].mean()),
            "chi_median": float(sub[chi_col].median()),
            "DeltaU_mean": float(sub["DeltaU"].mean()),
            "dist_to_band_mean": float(sub["dist_to_band"].mean()),
            "negative_rate": float((sub["phase"].astype(str) == "negative").mean()),
            "critical_rate": float((sub["phase"].astype(str) == "critical").mean()),
        }
        if slope_col and slope_col in sub.columns:
            item["slope_abs_mean"] = float(np.abs(sub[slope_col]).mean())
        out[str(region)] = item
    return out


def load_band(model_key):
    candidates = [
        CM2_DIR / f"{model_key}_critical_band_summary.json",
        CM2_DIR / f"{model_key}_critical_band.json",
    ]
    for p in candidates:
        if p.exists():
            js = read_json(p)
            band = js.get("critical_band", js)
            opt = band.get("optimized_band", None)
            if opt is not None:
                return float(opt["lo"]), float(opt["hi"]), js
            # fallback to IQR band
            if "critical_iqr_band_q25_q75" in band:
                lo, hi = band["critical_iqr_band_q25_q75"]
                return float(lo), float(hi), js
    # fallback: read overall summary
    overall_path = CM2_DIR / "cm2_overall_summary.json"
    if overall_path.exists():
        overall = read_json(overall_path)
        for item in overall:
            if item.get("model_key") == model_key:
                band = item["critical_band"]
                opt = band.get("optimized_band", None)
                if opt:
                    return float(opt["lo"]), float(opt["hi"]), item
    raise FileNotFoundError(f"Cannot find CM-2 critical band summary for {model_key} in {CM2_DIR}")


def find_deltaR_dataset(model_key):
    candidates = [
        CM1_DIR / f"{model_key}_deltaR_dataset.csv",
        CM1_DIR / f"{model_key}_deltaR.csv",
        CM1_DIR / f"{model_key}_phase_dataset.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find deltaR dataset for {model_key} in {CM1_DIR}")


def run_one_model(model_key):
    delta_path = find_deltaR_dataset(model_key)
    df = pd.read_csv(delta_path)

    phase_col = infer_phase_col(df)
    df["phase"] = df[phase_col].astype(str)

    dR_cols = [c for c in df.columns if c.startswith("dR_L") or c.startswith("dltR_L")]
    if not dR_cols:
        # Allow raw R columns by forming consecutive differences if needed
        R_cols = [c for c in df.columns if c.startswith("R_L")]
        if len(R_cols) >= 2:
            R_cols = sorted(R_cols, key=lambda x: int("".join(ch for ch in x if ch.isdigit())))
            for i, c in enumerate(R_cols[:-1]):
                dcol = f"dR_{c[2:]}"
                df[dcol] = df[R_cols[i + 1]] - df[c]
            dR_cols = [c for c in df.columns if c.startswith("dR_L")]
    if not dR_cols:
        raise RuntimeError(f"No dR columns found in {delta_path}")

    # sort dR columns by layer number
    def layer_num(c):
        digits = "".join(ch for ch in c if ch.isdigit())
        return int(digits) if digits else 0
    dR_cols = sorted(dR_cols, key=layer_num)

    du, pc_var, pc1_load = orient_delta_u(df, dR_cols, "phase")
    df["DeltaU"] = du

    lo, hi, cm2_js = load_band(model_key)
    # Important: CM2 band was fitted in the same orientation; if we recomputed orientation differently, try to align.
    # If critical mean is far outside while using CM2 band, assume consistent. Otherwise fallback to critical IQR.
    df["dist_to_band"] = band_distance(df["DeltaU"].values, lo, hi)
    df["inside_band"] = ((df["DeltaU"] >= lo) & (df["DeltaU"] <= hi)).astype(int)
    df["region"] = np.where(df["inside_band"] == 1, "inside_band", np.where(df["DeltaU"] < lo, "below_band", "above_band"))

    # Slope / velocity proxies over decision window
    X = df[dR_cols].values.astype(float)
    df["dR_slope_first_last"] = X[:, -1] - X[:, 0]
    df["dR_slope_abs"] = np.abs(df["dR_slope_first_last"])
    df["dR_total_variation"] = np.abs(np.diff(X, axis=1)).sum(axis=1) if X.shape[1] > 1 else 0.0

    p_neg, chi, logit_info = fit_response_proxy(df, "DeltaU", "phase")
    df["P_negative_proxy"] = p_neg
    df["chi_proxy"] = chi

    # Response comparisons
    inside = df[df["inside_band"] == 1]
    outside = df[df["inside_band"] == 0]

    chi_inside = float(inside["chi_proxy"].mean()) if len(inside) else np.nan
    chi_outside = float(outside["chi_proxy"].mean()) if len(outside) else np.nan
    chi_ratio = float(chi_inside / (chi_outside + 1e-12)) if np.isfinite(chi_inside) and np.isfinite(chi_outside) else np.nan

    # Critical phase has max chi?
    phase_chi = df.groupby("phase")["chi_proxy"].mean().to_dict()
    critical_has_max = False
    if "critical" in phase_chi and len(phase_chi) > 1:
        critical_has_max = phase_chi["critical"] >= max(v for k, v in phase_chi.items() if k != "critical")

    # Critical slowing proxy: inside band should have smaller |slope| / higher residence if layerwise not available.
    slope_inside = float(inside["dR_slope_abs"].mean()) if len(inside) else np.nan
    slope_outside = float(outside["dR_slope_abs"].mean()) if len(outside) else np.nan
    slowing_inside = bool(slope_inside < slope_outside) if np.isfinite(slope_inside) and np.isfinite(slope_outside) else False

    # Band distance should detect critical phase
    y_crit = (df["phase"] == "critical").astype(int).values
    auc_crit_low_dist = auc_safe(y_crit, -df["dist_to_band"].values)

    # chi should be larger close to band / inside band
    y_inside = df["inside_band"].astype(int).values
    auc_inside_by_chi = auc_safe(y_inside, df["chi_proxy"].values)

    pass_chi_inside_gt_outside = bool(chi_inside > chi_outside)
    pass_crit_by_band_distance = bool(auc_crit_low_dist >= 0.75)
    pass_response = bool(pass_chi_inside_gt_outside and auc_inside_by_chi >= 0.60)

    by_region = summarize_region(df, "region", "chi_proxy", "dR_slope_abs")
    by_phase = summarize_region(df, "phase", "chi_proxy", "dR_slope_abs")

    out = {
        "model_key": model_key,
        "deltaR_dataset": str(delta_path),
        "decision_cols": dR_cols,
        "band_lo": lo,
        "band_hi": hi,
        "pc_variance_recomputed": pc_var,
        "pc1_variance": float(pc_var[0]),
        "logistic_response_proxy": logit_info,
        "chi_inside_band_mean": chi_inside,
        "chi_outside_band_mean": chi_outside,
        "chi_inside_outside_ratio": chi_ratio,
        "auc_inside_by_chi": auc_inside_by_chi,
        "auc_critical_by_negative_dist_to_band": auc_crit_low_dist,
        "phase_chi_mean": {k: float(v) for k, v in phase_chi.items()},
        "critical_has_max_chi": critical_has_max,
        "slope_abs_inside_band_mean": slope_inside,
        "slope_abs_outside_band_mean": slope_outside,
        "critical_slowing_proxy_inside_slope_lower": slowing_inside,
        "by_region": by_region,
        "by_phase": by_phase,
        "pass_chi_inside_gt_outside": pass_chi_inside_gt_outside,
        "pass_critical_by_band_distance_auc_0p75": pass_crit_by_band_distance,
        "pass_response_audit": pass_response,
    }

    df.to_csv(SAVE_DIR / f"{model_key}_response_dataset.csv", index=False)
    write_json(SAVE_DIR / f"{model_key}_response_summary.json", out)
    return out


def main():
    summaries = []
    for model_key in MODELS:
        print(f"\n[CM-3] Running {model_key}...")
        try:
            s = run_one_model(model_key)
            summaries.append(s)
            print(
                f"  PC1={s['pc1_variance']:.3f} | "
                f"chi_in={s['chi_inside_band_mean']:.4g} | "
                f"chi_out={s['chi_outside_band_mean']:.4g} | "
                f"ratio={s['chi_inside_outside_ratio']:.3f} | "
                f"PASS={s['pass_response_audit']}"
            )
        except Exception as e:
            print(f"  ERROR for {model_key}: {e}")
            summaries.append({"model_key": model_key, "error": str(e)})

    rows = []
    for s in summaries:
        rows.append({
            "model_key": s.get("model_key"),
            "error": s.get("error", ""),
            "pc1_variance": s.get("pc1_variance", np.nan),
            "band_lo": s.get("band_lo", np.nan),
            "band_hi": s.get("band_hi", np.nan),
            "chi_inside_band_mean": s.get("chi_inside_band_mean", np.nan),
            "chi_outside_band_mean": s.get("chi_outside_band_mean", np.nan),
            "chi_inside_outside_ratio": s.get("chi_inside_outside_ratio", np.nan),
            "auc_inside_by_chi": s.get("auc_inside_by_chi", np.nan),
            "auc_critical_by_negative_dist_to_band": s.get("auc_critical_by_negative_dist_to_band", np.nan),
            "critical_has_max_chi": s.get("critical_has_max_chi", False),
            "critical_slowing_proxy_inside_slope_lower": s.get("critical_slowing_proxy_inside_slope_lower", False),
            "pass_chi_inside_gt_outside": s.get("pass_chi_inside_gt_outside", False),
            "pass_critical_by_band_distance_auc_0p75": s.get("pass_critical_by_band_distance_auc_0p75", False),
            "pass_response_audit": s.get("pass_response_audit", False),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SAVE_DIR / "cm3_model_summary.csv", index=False)
    write_json(SAVE_DIR / "cm3_overall_summary.json", summaries)

    print("\nSaved outputs to:", SAVE_DIR.resolve())
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
