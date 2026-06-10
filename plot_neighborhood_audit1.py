from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SAVE_DIR = Path("neighborhood_audit1_outputs")

summary = pd.read_csv(SAVE_DIR / "neighborhood_audit1_summary_by_topk_mode.csv")
layer = pd.read_csv(SAVE_DIR / "neighborhood_audit1_layerwise_best.csv")
metrics = pd.read_csv(SAVE_DIR / "neighborhood_audit1_metrics.csv")

# 1. TopK scaling curves
for mode in summary["mode"].unique():
    sub = summary[summary["mode"] == mode]

    plt.figure(figsize=(8, 5))
    plt.plot(sub["topk"], sub["MeanTopo"], marker="o", label="real TopK")
    plt.plot(sub["topk"], sub["MeanRandTopo"], marker="o", label="random")
    plt.plot(sub["topk"], sub["MeanShuffleTopo"], marker="o", label="shuffled TopK")
    plt.plot(sub["topk"], sub["MeanInputTokenTopo"], marker="o", label="input-token")

    plt.xscale("log")
    plt.xlabel("TopK")
    plt.ylabel("Mean Topo")
    plt.title(f"Neighborhood-Audit-1 scaling: {mode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"neighborhood_audit1_scaling_{mode}.png", dpi=180)
    plt.close()

# 2. Layerwise best Topo
plt.figure(figsize=(10, 5))
plt.plot(layer["layer"], layer["Topo"], marker="o")
plt.xlabel("Layer")
plt.ylabel("Best Topo")
plt.title("Neighborhood-Audit-1 layerwise best Topo")
plt.tight_layout()
plt.savefig(SAVE_DIR / "neighborhood_audit1_layerwise_best_topo.png", dpi=180)
plt.close()

# 3. Heatmap: layer × TopK
real = metrics[metrics["baseline"] == "real_topk"].copy()
piv = real.pivot_table(index="layer", columns="topk", values="Topo", aggfunc="mean")

plt.figure(figsize=(9, 6))
plt.imshow(piv.values, aspect="auto")
plt.colorbar(label="Topo")
plt.xticks(range(len(piv.columns)), piv.columns, rotation=45)
plt.yticks(range(len(piv.index)), piv.index)
plt.xlabel("TopK")
plt.ylabel("Layer")
plt.title("Topo heatmap by layer and TopK")
plt.tight_layout()
plt.savefig(SAVE_DIR / "neighborhood_audit1_topo_heatmap.png", dpi=180)
plt.close()

print("Saved plots to:", SAVE_DIR.resolve())