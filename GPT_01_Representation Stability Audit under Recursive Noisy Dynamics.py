import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Config
# ============================================================

D_MODEL = 128
STEPS = 500
SEEDS = [0, 1, 2]
NOISE_STD = 0.5
LR = 1e-3

# ============================================================
# Metrics
# ============================================================

def spectral_metrics(W):
    with torch.no_grad():
        U, S, V = torch.svd(W)

        S = torch.clamp(S, min=1e-12)

        p = S / S.sum()

        entropy = -(p * torch.log(p)).sum().item()

        effective_rank = torch.exp(
            -(p * torch.log(p)).sum()
        ).item()

        stable_rank = (S.pow(2).sum() / S.max().pow(2)).item()

        cond = (S.max() / S.min()).item()

        return {
            "entropy": entropy,
            "effective_rank": effective_rank,
            "stable_rank": stable_rank,
            "condition_number": cond,
        }

# ============================================================
# Base Models
# ============================================================

class VanillaModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d) * 0.02)

    def forward(self, h):
        return h @ self.W

    def loss(self, h):
        return torch.tensor(0.0, device=h.device)

    def matrix(self):
        return self.W

# ------------------------------------------------------------

class OrthogonalModel(nn.Module):
    def __init__(self, d, lam=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.lam = lam

    def forward(self, h):
        return h @ self.W

    def loss(self, h):
        WT_W = self.W.T @ self.W
        I = torch.eye(self.W.shape[0], device=self.W.device)

        ortho_loss = F.mse_loss(WT_W, I)

        return self.lam * ortho_loss

    def matrix(self):
        return self.W

# ------------------------------------------------------------

class CycleModel(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.WF = nn.Parameter(torch.randn(d, d) * 0.02)
        self.WG = nn.Parameter(torch.randn(d, d) * 0.02)

    def F_map(self, x):
        return x @ self.WF

    def G_map(self, x):
        return x @ self.WG

    def forward(self, h):
        return self.G_map(self.F_map(h))

    def loss(self, h):
        recon = self.G_map(self.F_map(h))
        return F.mse_loss(recon, h)

    def matrix(self):
        return self.WF

# ------------------------------------------------------------

class FullModel(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.WF = nn.Parameter(torch.randn(d, d) * 0.02)
        self.WG = nn.Parameter(torch.randn(d, d) * 0.02)

    def F_map(self, x):
        return x @ self.WF

    def G_map(self, x):
        return x @ self.WG

    def forward(self, h):
        return self.G_map(self.F_map(h))

    def loss(self, h):

        fs = self.F_map(h)
        gfs = self.G_map(fs)

        unit_loss = F.mse_loss(gfs, h)

        fgfs = self.F_map(gfs)

        triangle_loss = F.mse_loss(fgfs, fs)

        return unit_loss + triangle_loss

    def matrix(self):
        return self.WF

# ============================================================
# Experiment
# ============================================================

def run_experiment(model_class, name):

    all_results = defaultdict(list)

    print(f"\n==================== {name} ====================")

    for seed in SEEDS:

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_class(D_MODEL).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR
        )

        h = torch.randn(64, D_MODEL).to(DEVICE)

        for step in range(STEPS):

            noise = torch.randn_like(h) * NOISE_STD

            h_noisy = h + noise

            optimizer.zero_grad()

            h_next = model(h_noisy)

            loss = model.loss(h_noisy)

            # 防止爆炸
            stability_loss = 0.001 * h_next.pow(2).mean()

            total_loss = loss + stability_loss

            total_loss.backward()

            optimizer.step()

            # recursive dynamics
            h = h_next.detach()

        metrics = spectral_metrics(model.matrix())

        hidden_norm = h.norm(dim=-1).mean().item()

        metrics["hidden_norm"] = hidden_norm

        for k, v in metrics.items():
            all_results[k].append(v)

    print("\nFinal Statistics:")

    for k, vals in all_results.items():

        mean = np.mean(vals)
        std = np.std(vals)

        print(f"{k:20s} : {mean:.4f} ± {std:.4f}")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\nRunning Representation Stability Benchmark...\n")

    run_experiment(VanillaModel, "Vanilla")
    run_experiment(OrthogonalModel, "Orthogonal")
    run_experiment(CycleModel, "Cycle")
    run_experiment(FullModel, "Full Bidirectional")

    print("\nDone.\n")