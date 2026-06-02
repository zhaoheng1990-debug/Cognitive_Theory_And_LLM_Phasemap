# ============================================================
# Neighborhood-Audit-1b
# Final-Layer Exclusion and Corpus-Robustness Diagnostics
#
# Goal:
#   Harden DHRF Axiom 1 by testing whether Neighborhood-Audit-1
#   is merely a final-layer / lm_head readout artifact.
#
# Input:
#   ./neighborhood_audit1_outputs/
#
# Required files:
#   neighborhood_audit1_metrics.csv
#   neighborhood_audit1_summary_by_topk_mode.csv
#   neighborhood_audit1_layerwise_best.csv
#   neighborhood_audit1_global_best.csv
#
# Main checks:
#   1. Final-layer exclusion:
#        Are L0-L26 still strong after removing L27?
#
#   2. Non-final global best:
#        What is the best real_topk result excluding L27?
#
#   3. Layer-band summary:
#        L0-L2, L3-L17, L18-L23, L24-L26, L27
#
#   4. TopK scaling excluding L27:
#        Does TopK -> topology scaling remain after removing final layer?
#
#   5. Baseline separation:
#        real_topk vs random / shuffled / input_token
#
#   6. Optional corpus comparison:
#        If you later run Neighborhood-Audit-1 on real natural texts
#        and save outputs to another folder, this script can compare folders.
#
# Outputs:
#   ./neighborhood_audit1b_outputs/
#
# Key files:
#   neighborhood_audit1b_final_exclusion_summary.csv
#   neighborhood_audit1b_nonfinal_layerwise_best.csv
#   neighborhood_audit1b_nonfinal_global_best.csv
#   neighborhood_audit1b_layer_band_summary.csv
#   neighborhood_audit1b_scaling_nonfinal.csv
#   neighborhood_audit1b_robustness_decision.csv
#   neighborhood_audit1b_*.png
# ============================================================

from pathlib import Path
import math
import numpy as np
import pandas as pd

# matplotlib is now expected to be installed.
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("./neighborhood_audit1_outputs")
SAVE_DIR = Path("./neighborhood_audit1b_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FINAL_LAYER = 27

# Optional:
# If you later run Neighborhood-Audit-1 on real natural texts and save it here,
# set this path to compare default-template corpus vs real-text corpus.
# Example:
#   COMPARISON_INPUT_DIRS = {
#       "default_template": Path("./neighborhood_audit1_outputs"),
#       "real_text": Path("./neighborhood_audit1_realtext_outputs"),
#   }
COMPARISON_INPUT_DIRS = {
    "default_template": INPUT_DIR,
}

# Band definitions.
# Adjust if you use a model with a different number of layers.
LAYER_BANDS = {
    "L0_2_input_boundary": list(range(0, 3)),
    "L3_17_latent_bulk": list(range(3, 18)),
    "L18_23_recoupling_ramp": list(range(18, 24)),
    "L24_26_pre_final_peak": list(range(24, 27)),
    "L27_final_readout": [27],
}

# A conservative threshold for "non-final still strong".
# This is not a universal constant; it is a local audit threshold.
STRONG_TOPO_THRESHOLD = 0.50
STRONG_LIFT_RANDOM_THRESHOLD = 0.30
STRONG_LIFT_SHUFFLED_THRESHOLD = 0.30
STRONG_LIFT_INPUT_THRESHOLD = 0.15

# ============================================================
# LOAD
# ============================================================

metrics_path = INPUT_DIR / "neighborhood_audit1_metrics.csv"
summary_path = INPUT_DIR / "neighborhood_audit1_summary_by_topk_mode.csv"
layerwise_path = INPUT_DIR / "neighborhood_audit1_layerwise_best.csv"
global_best_path = INPUT_DIR / "neighborhood_audit1_global_best.csv"

required = [metrics_path, summary_path, layerwise_path, global_best_path]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing required Neighborhood-Audit-1 output files:\n"
        + "\n".join(missing)
    )

metrics = pd.read_csv(metrics_path)
summary = pd.read_csv(summary_path)
layerwise_best = pd.read_csv(layerwise_path)
global_best = pd.read_csv(global_best_path)

# Standardize types.
for df in [metrics, summary, layerwise_best, global_best]:
    for c in ["layer", "topk"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# Only real TopK rows for most analyses.
real = metrics[metrics["baseline"] == "real_topk"].copy()
nonfinal = real[real["layer"] != FINAL_LAYER].copy()
final = real[real["layer"] == FINAL_LAYER].copy()

if len(nonfinal) == 0:
    raise RuntimeError("No non-final rows found. Check FINAL_LAYER or metrics file.")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_mean(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(x.mean()) if len(x) else np.nan

def safe_max(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(x.max()) if len(x) else np.nan

def summarize_block(df, label):
    if len(df) == 0:
        return {
            "block": label,
            "n_rows": 0,
            "mean_topo": np.nan,
            "max_topo": np.nan,
            "mean_lift_random": np.nan,
            "mean_lift_shuffled": np.nan,
            "mean_lift_input_token": np.nan,
            "max_lift_random": np.nan,
            "max_lift_shuffled": np.nan,
            "max_lift_input_token": np.nan,
        }

    return {
        "block": label,
        "n_rows": int(len(df)),
        "mean_topo": safe_mean(df["Topo"]),
        "max_topo": safe_max(df["Topo"]),
        "mean_lift_random": safe_mean(df["LiftTopo_vs_random"]),
        "mean_lift_shuffled": safe_mean(df["LiftTopo_vs_shuffled"]),
        "mean_lift_input_token": safe_mean(df["LiftTopo_vs_input_token"]),
        "max_lift_random": safe_max(df["LiftTopo_vs_random"]),
        "max_lift_shuffled": safe_max(df["LiftTopo_vs_shuffled"]),
        "max_lift_input_token": safe_max(df["LiftTopo_vs_input_token"]),
    }

def pass_fail(value, threshold, direction="ge"):
    if not np.isfinite(value):
        return False
    if direction == "ge":
        return bool(value >= threshold)
    if direction == "le":
        return bool(value <= threshold)
    raise ValueError(direction)

def add_layer_band(df):
    df = df.copy()
    df["layer_band"] = "unassigned"
    for band, layers in LAYER_BANDS.items():
        df.loc[df["layer"].isin(layers), "layer_band"] = band
    return df

# ============================================================
# 1. FINAL-LAYER EXCLUSION SUMMARY
# ============================================================

blocks = []

blocks.append(summarize_block(real, "all_layers_L0_27"))
blocks.append(summarize_block(nonfinal, "nonfinal_L0_26"))
blocks.append(summarize_block(final, "final_L27_only"))

# Focus on each mode.
for mode in sorted(real["mode"].unique()):
    blocks.append(summarize_block(real[real["mode"] == mode], f"all_layers_{mode}"))
    blocks.append(summarize_block(nonfinal[nonfinal["mode"] == mode], f"nonfinal_{mode}"))
    blocks.append(summarize_block(final[final["mode"] == mode], f"final_{mode}"))

final_exclusion_summary = pd.DataFrame(blocks)

# Add ratios comparing non-final best with final best.
final_best_topo = safe_max(final["Topo"])
nonfinal_best_topo = safe_max(nonfinal["Topo"])

final_exclusion_summary["final_best_topo"] = final_best_topo
final_exclusion_summary["nonfinal_best_topo"] = nonfinal_best_topo
final_exclusion_summary["nonfinal_to_final_best_ratio"] = (
    nonfinal_best_topo / final_best_topo
    if np.isfinite(final_best_topo) and final_best_topo > 0
    else np.nan
)

# ============================================================
# 2. NON-FINAL BEST TABLES
# ============================================================

nonfinal_layerwise_best = (
    nonfinal
    .sort_values("Topo", ascending=False)
    .groupby("layer", as_index=False)
    .first()
    .sort_values("layer")
)

nonfinal_global_best = (
    nonfinal
    .sort_values("Topo", ascending=False)
    .head(40)
    .copy()
)

final_global_best = (
    final
    .sort_values("Topo", ascending=False)
    .head(20)
    .copy()
)

# ============================================================
# 3. LAYER BAND SUMMARY
# ============================================================

real_banded = add_layer_band(real)

layer_band_rows = []
for band in LAYER_BANDS.keys():
    sub = real_banded[real_banded["layer_band"] == band]
    layer_band_rows.append(summarize_block(sub, band))

    for mode in sorted(real["mode"].unique()):
        sub_mode = sub[sub["mode"] == mode]
        layer_band_rows.append(summarize_block(sub_mode, f"{band}__{mode}"))

layer_band_summary = pd.DataFrame(layer_band_rows)

# Also compute band best rows.
layer_band_best = (
    real_banded
    .sort_values("Topo", ascending=False)
    .groupby(["layer_band"], as_index=False)
    .first()
    .sort_values("Topo", ascending=False)
)

# ============================================================
# 4. TOPK SCALING EXCLUDING L27
# ============================================================

scaling_nonfinal = (
    nonfinal
    .groupby(["topk", "mode"], as_index=False)
    .agg(
        MeanSp=("DistSpearman", "mean"),
        MeanKNN=("KNN", "mean"),
        MeanCKA=("CKA", "mean"),
        MeanTopo=("Topo", "mean"),
        MeanRandTopo=("RandTopo", "mean"),
        MeanShuffleTopo=("ShuffleTopo", "mean"),
        MeanInputTokenTopo=("InputTokenTopo", "mean"),
        MeanLiftRandom=("LiftTopo_vs_random", "mean"),
        MeanLiftShuffled=("LiftTopo_vs_shuffled", "mean"),
        MeanLiftInputToken=("LiftTopo_vs_input_token", "mean"),
    )
    .sort_values(["mode", "topk"])
)

# Check monotonic-ish scaling.
# Strict monotonic can be too harsh; we count positive adjacent steps.
scaling_checks = []

for mode in sorted(scaling_nonfinal["mode"].unique()):
    sub = scaling_nonfinal[scaling_nonfinal["mode"] == mode].sort_values("topk")
    vals = sub["MeanTopo"].values
    topks = sub["topk"].values

    if len(vals) >= 2:
        diffs = np.diff(vals)
        positive_steps = int((diffs > 0).sum())
        total_steps = int(len(diffs))
        slope_ratio = positive_steps / total_steps if total_steps else np.nan
        overall_gain = float(vals[-1] - vals[0])
    else:
        positive_steps = 0
        total_steps = 0
        slope_ratio = np.nan
        overall_gain = np.nan

    scaling_checks.append({
        "mode": mode,
        "min_topk": int(topks[0]) if len(topks) else np.nan,
        "max_topk": int(topks[-1]) if len(topks) else np.nan,
        "mean_topo_at_min_topk": float(vals[0]) if len(vals) else np.nan,
        "mean_topo_at_max_topk": float(vals[-1]) if len(vals) else np.nan,
        "overall_gain": overall_gain,
        "positive_adjacent_steps": positive_steps,
        "total_adjacent_steps": total_steps,
        "positive_step_ratio": slope_ratio,
    })

scaling_check_df = pd.DataFrame(scaling_checks)

# ============================================================
# 5. BASELINE SEPARATION DIAGNOSTICS
# ============================================================

baseline_sep_rows = []

for scope_name, scope_df in [
    ("all_layers", real),
    ("nonfinal_L0_26", nonfinal),
    ("final_L27", final),
]:
    for mode in sorted(real["mode"].unique()):
        sub = scope_df[scope_df["mode"] == mode]
        if len(sub) == 0:
            continue

        baseline_sep_rows.append({
            "scope": scope_name,
            "mode": mode,
            "n_rows": int(len(sub)),
            "mean_topo": safe_mean(sub["Topo"]),
            "mean_random": safe_mean(sub["RandTopo"]),
            "mean_shuffled": safe_mean(sub["ShuffleTopo"]),
            "mean_input_token": safe_mean(sub["InputTokenTopo"]),
            "mean_lift_random": safe_mean(sub["LiftTopo_vs_random"]),
            "mean_lift_shuffled": safe_mean(sub["LiftTopo_vs_shuffled"]),
            "mean_lift_input_token": safe_mean(sub["LiftTopo_vs_input_token"]),
            "min_lift_random": float(pd.to_numeric(sub["LiftTopo_vs_random"], errors="coerce").min()),
            "min_lift_shuffled": float(pd.to_numeric(sub["LiftTopo_vs_shuffled"], errors="coerce").min()),
            "min_lift_input_token": float(pd.to_numeric(sub["LiftTopo_vs_input_token"], errors="coerce").min()),
        })

baseline_separation = pd.DataFrame(baseline_sep_rows)

# ============================================================
# 6. ROBUSTNESS DECISION TABLE
# ============================================================

# Pick non-final best row.
best_nonfinal_row = nonfinal.sort_values("Topo", ascending=False).iloc[0].to_dict()
best_final_row = final.sort_values("Topo", ascending=False).iloc[0].to_dict() if len(final) else {}

nonfinal_mean_topo = safe_mean(nonfinal["Topo"])
nonfinal_mean_lift_random = safe_mean(nonfinal["LiftTopo_vs_random"])
nonfinal_mean_lift_shuffled = safe_mean(nonfinal["LiftTopo_vs_shuffled"])
nonfinal_mean_lift_input = safe_mean(nonfinal["LiftTopo_vs_input_token"])

nonfinal_pass_topo = pass_fail(nonfinal_mean_topo, STRONG_TOPO_THRESHOLD)
nonfinal_pass_rand = pass_fail(nonfinal_mean_lift_random, STRONG_LIFT_RANDOM_THRESHOLD)
nonfinal_pass_shuffle = pass_fail(nonfinal_mean_lift_shuffled, STRONG_LIFT_SHUFFLED_THRESHOLD)
nonfinal_pass_input = pass_fail(nonfinal_mean_lift_input, STRONG_LIFT_INPUT_THRESHOLD)

# Scaling pass if all modes have positive overall gain and at least 70% positive adjacent steps.
scaling_pass = bool(
    len(scaling_check_df) > 0
    and (scaling_check_df["overall_gain"] > 0).all()
    and (scaling_check_df["positive_step_ratio"] >= 0.70).all()
)

ratio = (
    nonfinal_best_topo / final_best_topo
    if np.isfinite(nonfinal_best_topo) and np.isfinite(final_best_topo) and final_best_topo > 0
    else np.nan
)

# If non-final best is at least 75% of final best, we say final is not the only source.
nonfinal_not_collapsed_vs_final = pass_fail(ratio, 0.75)

decision_rows = [
    {
        "criterion": "nonfinal_mean_topo_above_threshold",
        "value": nonfinal_mean_topo,
        "threshold": STRONG_TOPO_THRESHOLD,
        "pass": nonfinal_pass_topo,
        "interpretation": "L0-L26 still preserve hidden topology after removing L27.",
    },
    {
        "criterion": "nonfinal_mean_lift_vs_random_above_threshold",
        "value": nonfinal_mean_lift_random,
        "threshold": STRONG_LIFT_RANDOM_THRESHOLD,
        "pass": nonfinal_pass_rand,
        "interpretation": "Non-final signal is not explained by random vocabulary embeddings.",
    },
    {
        "criterion": "nonfinal_mean_lift_vs_shuffled_above_threshold",
        "value": nonfinal_mean_lift_shuffled,
        "threshold": STRONG_LIFT_SHUFFLED_THRESHOLD,
        "pass": nonfinal_pass_shuffle,
        "interpretation": "Non-final signal is sample-specific, not just global TopK distribution bias.",
    },
    {
        "criterion": "nonfinal_mean_lift_vs_input_token_above_threshold",
        "value": nonfinal_mean_lift_input,
        "threshold": STRONG_LIFT_INPUT_THRESHOLD,
        "pass": nonfinal_pass_input,
        "interpretation": "Non-final signal is not merely prompt lexical overlap.",
    },
    {
        "criterion": "nonfinal_scaling_positive",
        "value": float(scaling_check_df["positive_step_ratio"].min()) if len(scaling_check_df) else np.nan,
        "threshold": 0.70,
        "pass": scaling_pass,
        "interpretation": "TopK scaling remains after excluding the final layer.",
    },
    {
        "criterion": "nonfinal_best_not_collapsed_vs_final_best",
        "value": ratio,
        "threshold": 0.75,
        "pass": nonfinal_not_collapsed_vs_final,
        "interpretation": "L27 may be strong, but non-final layers are not collapsed.",
    },
]

robustness_decision = pd.DataFrame(decision_rows)
overall_pass = bool(robustness_decision["pass"].all())

robustness_decision.loc[len(robustness_decision)] = {
    "criterion": "overall_neighborhood_audit1b_decision",
    "value": np.nan,
    "threshold": np.nan,
    "pass": overall_pass,
    "interpretation": (
        "PASS: Axiom-1 evidence is not merely a final-layer readout artifact."
        if overall_pass
        else "MIXED: inspect failed criteria before claiming final-layer independence."
    ),
}

# ============================================================
# 7. OPTIONAL MULTI-CORPUS COMPARISON
# ============================================================

corpus_comparison_rows = []

for corpus_name, folder in COMPARISON_INPUT_DIRS.items():
    m_path = folder / "neighborhood_audit1_metrics.csv"
    if not m_path.exists():
        corpus_comparison_rows.append({
            "corpus": corpus_name,
            "status": "missing_metrics_file",
            "folder": str(folder),
        })
        continue

    m = pd.read_csv(m_path)
    m = m[m["baseline"] == "real_topk"].copy()

    for c in ["layer", "topk"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    m_nonfinal = m[m["layer"] != FINAL_LAYER]
    m_final = m[m["layer"] == FINAL_LAYER]

    row = {
        "corpus": corpus_name,
        "status": "ok",
        "folder": str(folder),
        "n_rows": int(len(m)),
        "mean_topo_all": safe_mean(m["Topo"]),
        "mean_topo_nonfinal": safe_mean(m_nonfinal["Topo"]),
        "mean_topo_final": safe_mean(m_final["Topo"]),
        "best_topo_all": safe_max(m["Topo"]),
        "best_topo_nonfinal": safe_max(m_nonfinal["Topo"]),
        "best_topo_final": safe_max(m_final["Topo"]),
        "mean_lift_random_nonfinal": safe_mean(m_nonfinal["LiftTopo_vs_random"]),
        "mean_lift_shuffled_nonfinal": safe_mean(m_nonfinal["LiftTopo_vs_shuffled"]),
        "mean_lift_input_nonfinal": safe_mean(m_nonfinal["LiftTopo_vs_input_token"]),
    }
    corpus_comparison_rows.append(row)

corpus_comparison = pd.DataFrame(corpus_comparison_rows)

# ============================================================
# 8. PLOTS
# ============================================================

# 8.1 Scaling curves excluding L27.
for mode in sorted(scaling_nonfinal["mode"].unique()):
    sub = scaling_nonfinal[scaling_nonfinal["mode"] == mode].sort_values("topk")

    plt.figure(figsize=(8, 5))
    plt.plot(sub["topk"], sub["MeanTopo"], marker="o", label="real TopK, non-final")
    plt.plot(sub["topk"], sub["MeanRandTopo"], marker="o", label="random")
    plt.plot(sub["topk"], sub["MeanShuffleTopo"], marker="o", label="shuffled TopK")
    plt.plot(sub["topk"], sub["MeanInputTokenTopo"], marker="o", label="input-token")

    plt.xscale("log")
    plt.xlabel("TopK")
    plt.ylabel("Mean Topo")
    plt.title(f"Neighborhood-Audit-1b non-final scaling: {mode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"neighborhood_audit1b_nonfinal_scaling_{mode}.png", dpi=180)
    plt.close()

# 8.2 Layerwise best with final layer highlighted.
plt.figure(figsize=(10, 5))
plt.plot(nonfinal_layerwise_best["layer"], nonfinal_layerwise_best["Topo"], marker="o", label="non-final layer best")

if len(final):
    final_best_for_plot = final.sort_values("Topo", ascending=False).iloc[0]
    plt.scatter(
        [FINAL_LAYER],
        [final_best_for_plot["Topo"]],
        marker="x",
        s=80,
        label="best final-layer row",
    )

plt.axhline(STRONG_TOPO_THRESHOLD, linestyle="--", linewidth=1, label="strong threshold")
plt.xlabel("Layer")
plt.ylabel("Best Topo")
plt.title("Neighborhood-Audit-1b: layerwise best excluding final-layer dominance")
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / "neighborhood_audit1b_nonfinal_layerwise_best_topo.png", dpi=180)
plt.close()

# 8.3 Layer band mean Topo.
band_plot = layer_band_summary[
    ~layer_band_summary["block"].str.contains("__", regex=False)
].copy()

plt.figure(figsize=(10, 5))
plt.bar(band_plot["block"], band_plot["mean_topo"])
plt.xticks(rotation=35, ha="right")
plt.ylabel("Mean Topo")
plt.title("Neighborhood-Audit-1b: mean Topo by layer band")
plt.tight_layout()
plt.savefig(SAVE_DIR / "neighborhood_audit1b_layer_band_mean_topo.png", dpi=180)
plt.close()

# 8.4 Heatmap excluding L27.
heat = nonfinal.pivot_table(index="layer", columns="topk", values="Topo", aggfunc="mean")
plt.figure(figsize=(9, 6))
plt.imshow(heat.values, aspect="auto")
plt.colorbar(label="Topo")
plt.xticks(range(len(heat.columns)), heat.columns, rotation=45)
plt.yticks(range(len(heat.index)), heat.index)
plt.xlabel("TopK")
plt.ylabel("Layer")
plt.title("Neighborhood-Audit-1b: non-final Topo heatmap")
plt.tight_layout()
plt.savefig(SAVE_DIR / "neighborhood_audit1b_nonfinal_topo_heatmap.png", dpi=180)
plt.close()

# 8.5 Real vs baselines, non-final.
for mode in sorted(scaling_nonfinal["mode"].unique()):
    sub = scaling_nonfinal[scaling_nonfinal["mode"] == mode].sort_values("topk")
    width = 0.18
    x = np.arange(len(sub))

    plt.figure(figsize=(11, 5))
    plt.bar(x - 1.5 * width, sub["MeanTopo"], width, label="real")
    plt.bar(x - 0.5 * width, sub["MeanRandTopo"], width, label="random")
    plt.bar(x + 0.5 * width, sub["MeanShuffleTopo"], width, label="shuffled")
    plt.bar(x + 1.5 * width, sub["MeanInputTokenTopo"], width, label="input-token")

    plt.xticks(x, sub["topk"].astype(int).astype(str), rotation=45)
    plt.xlabel("TopK")
    plt.ylabel("Mean Topo")
    plt.title(f"Neighborhood-Audit-1b: non-final real vs baselines ({mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"neighborhood_audit1b_nonfinal_baseline_bars_{mode}.png", dpi=180)
    plt.close()

# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

final_exclusion_summary_path = SAVE_DIR / "neighborhood_audit1b_final_exclusion_summary.csv"
nonfinal_layerwise_best_path = SAVE_DIR / "neighborhood_audit1b_nonfinal_layerwise_best.csv"
nonfinal_global_best_path = SAVE_DIR / "neighborhood_audit1b_nonfinal_global_best.csv"
final_global_best_path = SAVE_DIR / "neighborhood_audit1b_final_global_best.csv"
layer_band_summary_path = SAVE_DIR / "neighborhood_audit1b_layer_band_summary.csv"
layer_band_best_path = SAVE_DIR / "neighborhood_audit1b_layer_band_best.csv"
scaling_nonfinal_path = SAVE_DIR / "neighborhood_audit1b_scaling_nonfinal.csv"
scaling_check_path = SAVE_DIR / "neighborhood_audit1b_scaling_check.csv"
baseline_sep_path = SAVE_DIR / "neighborhood_audit1b_baseline_separation.csv"
decision_path = SAVE_DIR / "neighborhood_audit1b_robustness_decision.csv"
corpus_comparison_path = SAVE_DIR / "neighborhood_audit1b_corpus_comparison.csv"

final_exclusion_summary.to_csv(final_exclusion_summary_path, index=False)
nonfinal_layerwise_best.to_csv(nonfinal_layerwise_best_path, index=False)
nonfinal_global_best.to_csv(nonfinal_global_best_path, index=False)
final_global_best.to_csv(final_global_best_path, index=False)
layer_band_summary.to_csv(layer_band_summary_path, index=False)
layer_band_best.to_csv(layer_band_best_path, index=False)
scaling_nonfinal.to_csv(scaling_nonfinal_path, index=False)
scaling_check_df.to_csv(scaling_check_path, index=False)
baseline_separation.to_csv(baseline_sep_path, index=False)
robustness_decision.to_csv(decision_path, index=False)
corpus_comparison.to_csv(corpus_comparison_path, index=False)

# ============================================================
# 10. PRINT RESULTS
# ============================================================

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)

print("\n============================================================")
print("Neighborhood-Audit-1b Results")
print("============================================================\n")

print("Final-layer exclusion summary:")
print(final_exclusion_summary.to_string(index=False))

print("\nNon-final global best Top 20:")
view_cols = [
    "layer",
    "layer_name",
    "topk",
    "mode",
    "DistSpearman",
    "KNN",
    "CKA",
    "Topo",
    "RandTopo",
    "ShuffleTopo",
    "InputTokenTopo",
    "LiftTopo_vs_random",
    "LiftTopo_vs_shuffled",
    "LiftTopo_vs_input_token",
]
existing_view_cols = [c for c in view_cols if c in nonfinal_global_best.columns]
print(nonfinal_global_best[existing_view_cols].head(20).to_string(index=False))

print("\nFinal-layer best rows:")
existing_view_cols_final = [c for c in view_cols if c in final_global_best.columns]
print(final_global_best[existing_view_cols_final].head(10).to_string(index=False))

print("\nLayer band summary:")
print(layer_band_summary.to_string(index=False))

print("\nLayer band best:")
existing_band_cols = [c for c in ["layer_band"] + view_cols if c in layer_band_best.columns]
print(layer_band_best[existing_band_cols].to_string(index=False))

print("\nNon-final TopK scaling check:")
print(scaling_check_df.to_string(index=False))

print("\nBaseline separation:")
print(baseline_separation.to_string(index=False))

print("\nRobustness decision:")
print(robustness_decision.to_string(index=False))

print("\nCorpus comparison:")
print(corpus_comparison.to_string(index=False))

print("\nSaved outputs:")
for p in [
    final_exclusion_summary_path,
    nonfinal_layerwise_best_path,
    nonfinal_global_best_path,
    final_global_best_path,
    layer_band_summary_path,
    layer_band_best_path,
    scaling_nonfinal_path,
    scaling_check_path,
    baseline_sep_path,
    decision_path,
    corpus_comparison_path,
]:
    print(" ", p)

print("\nSaved plots:")
for p in sorted(SAVE_DIR.glob("*.png")):
    print(" ", p)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n============================================================")
print("Neighborhood-Audit-1b Interpretation Guide")
print("============================================================\n")

print("Core question:")
print("  Is Axiom-1 evidence merely a final-layer / lm_head readout artifact?")
print()
print("Strong positive result if:")
print("  1. nonfinal_L0_26 mean Topo remains high.")
print("  2. nonfinal_L0_26 lift vs random / shuffled / input-token remains high.")
print("  3. non-final TopK scaling remains positive.")
print("  4. best non-final Topo is not collapsed relative to best final-layer Topo.")
print("  5. L24-L26 or earlier layers still show strong Topo.")
print()
print("Interpretation:")
print("  If overall decision PASS:")
print("    Axiom-1 evidence is not reducible to final-layer readout geometry.")
print()
print("  If final layer is strongest but non-final remains strong:")
print("    L27 may be an amplified vocabulary-coupling/readout layer,")
print("    while semantic neighborhood topology already exists internally.")
print()
print("  If non-final collapses:")
print("    Neighborhood-Audit-1 may be too close to final readout and needs")
print("    stronger corpus/model controls.")
print()
print("Next if positive:")
print("  Run a corpus robustness rerun with real natural texts, then compare folders.")
print()
print("Done.")