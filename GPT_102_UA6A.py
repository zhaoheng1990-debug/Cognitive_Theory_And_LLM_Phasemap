import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import r2_score
from scipy.stats import linregress

INPUT_CSV = r"C:\Windows\System32\ua5b_outputs\ua5b_dataset_with_susceptibility.csv"
CONFIG_JSON = "./ua5b_outputs/ua5b_config.json"

SAVE_DIR = Path("./ua6a_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

U_COL = "DeltaU_global"
CHI_COL = "chi_DeltaU"
EPS = 1e-9

df = pd.read_csv(INPUT_CSV)

with open(CONFIG_JSON, "r", encoding="utf-8") as f:
    cfg = json.load(f)

U_MINUS = cfg["critical_band"]["U_minus"]
U_CENTER = cfg["critical_band"]["U_center"]
U_PLUS = cfg["critical_band"]["U_plus"]

# ============================================================
# 1. Distance definitions
# ============================================================

df["dist_center"] = np.abs(df[U_COL] - U_CENTER)

def dist_to_band_edge(u):
    if U_MINUS <= u <= U_PLUS:
        return min(abs(u - U_MINUS), abs(u - U_PLUS))
    return min(abs(u - U_MINUS), abs(u - U_PLUS))

df["dist_edge"] = df[U_COL].apply(dist_to_band_edge)
df["inside_band"] = ((df[U_COL] >= U_MINUS) & (df[U_COL] <= U_PLUS)).astype(int)

# avoid log(0)
df["log_chi"] = np.log(df[CHI_COL].clip(EPS))
df["log_dist_center"] = np.log(df["dist_center"].clip(EPS))
df["log_dist_edge"] = np.log(df["dist_edge"].clip(EPS))

# ============================================================
# 2. Bin-level scaling
# ============================================================

df["center_bin"] = pd.qcut(df["dist_center"], q=12, duplicates="drop")
df["edge_bin"] = pd.qcut(df["dist_edge"], q=12, duplicates="drop")

def make_bin_summary(bin_col, dist_col):
    out = (
        df.groupby(bin_col, observed=False)
        .agg(
            n=(dist_col, "count"),
            dist_mean=(dist_col, "mean"),
            dist_min=(dist_col, "min"),
            dist_max=(dist_col, "max"),
            chi_mean=(CHI_COL, "mean"),
            chi_std=(CHI_COL, "std"),
            gen_negative_rate=("gen_negative_proxy", "mean"),
            inside_band=("inside_band", "mean"),
        )
        .reset_index()
    )
    out["log_dist"] = np.log(out["dist_mean"].clip(EPS))
    out["log_chi"] = np.log(out["chi_mean"].clip(EPS))
    return out

center_bins = make_bin_summary("center_bin", "dist_center")
center_bins["distance_type"] = "center"

edge_bins = make_bin_summary("edge_bin", "dist_edge")
edge_bins["distance_type"] = "edge"

bin_summary = pd.concat([center_bins, edge_bins], ignore_index=True)

# ============================================================
# 3. Fit candidate scaling laws
# ============================================================

def fit_power_law(bin_df):
    """
    log chi = a - gamma log dist
    gamma = -slope
    """
    x = bin_df["log_dist"].values
    y = bin_df["log_chi"].values

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 4:
        return None

    res = linregress(x, y)
    yhat = res.intercept + res.slope * x

    return {
        "model": "power_law",
        "formula": "chi ~ A * dist^(-gamma)",
        "slope": res.slope,
        "gamma": -res.slope,
        "intercept": res.intercept,
        "r2": r2_score(y, yhat),
        "corr": np.corrcoef(y, yhat)[0, 1],
        "p_value": res.pvalue,
        "n": len(x),
    }

def fit_exponential(bin_df):
    """
    log chi = a - lambda dist
    """
    x = bin_df["dist_mean"].values
    y = bin_df["log_chi"].values

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 4:
        return None

    res = linregress(x, y)
    yhat = res.intercept + res.slope * x

    return {
        "model": "exponential",
        "formula": "chi ~ A * exp(-lambda * dist)",
        "slope": res.slope,
        "lambda": -res.slope,
        "intercept": res.intercept,
        "r2": r2_score(y, yhat),
        "corr": np.corrcoef(y, yhat)[0, 1],
        "p_value": res.pvalue,
        "n": len(x),
    }

def fit_linear_chi(bin_df):
    """
    chi = a + b dist
    """
    x = bin_df["dist_mean"].values
    y = bin_df["chi_mean"].values

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 4:
        return None

    res = linregress(x, y)
    yhat = res.intercept + res.slope * x

    return {
        "model": "linear",
        "formula": "chi ~ a + b * dist",
        "slope": res.slope,
        "intercept": res.intercept,
        "r2": r2_score(y, yhat),
        "corr": np.corrcoef(y, yhat)[0, 1],
        "p_value": res.pvalue,
        "n": len(x),
    }

fit_rows = []

for dtype, sub in bin_summary.groupby("distance_type"):
    for fitter in [fit_power_law, fit_exponential, fit_linear_chi]:
        row = fitter(sub)
        if row:
            row["distance_type"] = dtype
            fit_rows.append(row)

model_comparison = pd.DataFrame(fit_rows)

# ============================================================
# 4. Inside-band vs outside-band scaling
# ============================================================

subset_rows = []

for name, sub in {
    "all": df,
    "inside_band": df[df["inside_band"] == 1],
    "outside_band": df[df["inside_band"] == 0],
    "critical_phase": df[df["phase"] == "critical"],
}.items():
    for dist_col in ["dist_center", "dist_edge"]:
        tmp = sub[[dist_col, CHI_COL]].copy()
        tmp = tmp[(tmp[dist_col] > 0) & (tmp[CHI_COL] > 0)]
        if len(tmp) < 12:
            continue

        x = np.log(tmp[dist_col].values)
        y = np.log(tmp[CHI_COL].values)

        res = linregress(x, y)
        yhat = res.intercept + res.slope * x

        subset_rows.append({
            "subset": name,
            "distance_type": dist_col,
            "n": len(tmp),
            "gamma": -res.slope,
            "slope": res.slope,
            "r2": r2_score(y, yhat),
            "corr": np.corrcoef(y, yhat)[0, 1],
            "p_value": res.pvalue,
        })

scaling_fit_summary = pd.DataFrame(subset_rows)

# ============================================================
# 5. Diagnostics
# ============================================================

best_power = (
    model_comparison[model_comparison["model"] == "power_law"]
    .sort_values("r2", ascending=False)
    .head(1)
)

best_any = (
    model_comparison
    .sort_values("r2", ascending=False)
    .head(1)
)

diagnostics = pd.DataFrame([{
    "best_model": best_any["model"].iloc[0],
    "best_distance_type": best_any["distance_type"].iloc[0],
    "best_r2": float(best_any["r2"].iloc[0]),
    "best_power_distance_type": best_power["distance_type"].iloc[0] if len(best_power) else None,
    "best_power_r2": float(best_power["r2"].iloc[0]) if len(best_power) else np.nan,
    "best_power_gamma": float(best_power["gamma"].iloc[0]) if len(best_power) else np.nan,
    "power_law_supported_lite": bool(len(best_power) and best_power["r2"].iloc[0] > 0.50),
    "power_law_supported_strong": bool(len(best_power) and best_power["r2"].iloc[0] > 0.75),
    "note": "Strong support requires power-law fit to beat exponential/linear alternatives, not merely have positive slope."
}])

# ============================================================
# 6. Save
# ============================================================

df.to_csv(SAVE_DIR / "ua6a_dataset_with_scaling.csv", index=False, encoding="utf-8-sig")
bin_summary.to_csv(SAVE_DIR / "ua6a_bin_scaling_summary.csv", index=False, encoding="utf-8-sig")
model_comparison.to_csv(SAVE_DIR / "ua6a_model_comparison.csv", index=False, encoding="utf-8-sig")
scaling_fit_summary.to_csv(SAVE_DIR / "ua6a_scaling_fit_summary.csv", index=False, encoding="utf-8-sig")
diagnostics.to_csv(SAVE_DIR / "ua6a_diagnostics.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "ua6a_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "input_csv": INPUT_CSV,
        "hypothesis": "Susceptibility follows critical scaling: chi ~ |DeltaU - Uc|^{-gamma}",
        "critical_band": {
            "U_minus": U_MINUS,
            "U_center": U_CENTER,
            "U_plus": U_PLUS,
        },
        "success_criteria": {
            "PASS_Lite": "power-law R2 > 0.50",
            "PASS": "power-law R2 > 0.75",
            "PASS_Strong": "power-law beats exponential and linear alternatives"
        }
    }, f, ensure_ascii=False, indent=2)

print("\n========== UA-6A MODEL COMPARISON ==========")
print(model_comparison.to_string(index=False))

print("\n========== UA-6A SCALING FIT SUMMARY ==========")
print(scaling_fit_summary.to_string(index=False))

print("\n========== UA-6A BIN SCALING SUMMARY ==========")
print(bin_summary.to_string(index=False))

print("\n========== UA-6A DIAGNOSTICS ==========")
print(diagnostics.to_string(index=False))

print("\n[DONE] Saved to:", SAVE_DIR)