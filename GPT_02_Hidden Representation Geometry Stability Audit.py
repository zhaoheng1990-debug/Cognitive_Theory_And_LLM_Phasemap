import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"

torch.manual_seed(42)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
model.eval()

hidden_size = model.config.hidden_size

# ============================================================
# STABILIZERS
# ============================================================

class IdentityStabilizer(nn.Module):
    def forward(self, x):
        return x

class OrthogonalStabilizer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        A = torch.randn(hidden_size, hidden_size)
        A = A - A.T

        Q = torch.matrix_exp(A * 0.001)

        self.register_buffer("Q", Q)

    def forward(self, x):

        Q = self.Q.to(
            device=x.device,
            dtype=x.dtype
        )

        return x @ Q

class LayerNormStabilizer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        self.norm = self.norm.to(
            device=x.device,
            dtype=x.dtype
        )
        return self.norm(x)

class RMSNormStabilizer(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.scale


class LayerNormOrthogonalStabilizer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)

        A = torch.randn(hidden_size, hidden_size)
        A = A - A.T

        Q = torch.matrix_exp(A * 0.001)

        self.register_buffer("Q", Q)

    def forward(self, x):

        self.norm = self.norm.to(
            device=x.device,
            dtype=x.dtype
        )

        Q = self.Q.to(
            device=x.device,
            dtype=x.dtype
        )

        return self.norm(x) @ Q
# ============================================================
# METRICS
# ============================================================

def flatten_hidden(x):

    return x.reshape(-1, x.shape[-1]).float()


def svdvals(x):

    x = flatten_hidden(x)

    _, S, _ = torch.svd(x)

    return S


def spectral_entropy(x):

    S = svdvals(x)

    p = S / S.sum()

    entropy = -(p * torch.log(p + 1e-12)).sum()

    return entropy.item()


def effective_rank(x):

    S = svdvals(x)

    p = S / S.sum()

    entropy = -(p * torch.log(p + 1e-12)).sum()

    return torch.exp(entropy).item()


def stable_rank(x):

    S = svdvals(x)

    return (S.pow(2).sum() / S.max().pow(2)).item()


def participation_ratio(x):

    S = svdvals(x)

    return (S.sum().pow(2) / S.pow(2).sum()).item()


def condition_number(x):

    S = svdvals(x)

    return (S.max() / (S.min() + 1e-12)).item()


def isotropy_score(x):

    S = svdvals(x)

    return (S.min() / (S.max() + 1e-12)).item()


# ============================================================
# BENCHMARK
# ============================================================

stabilizers = {
    "Vanilla": IdentityStabilizer().to(device),
    "LayerNorm": LayerNormStabilizer(hidden_size).to(device),
    "RMSNorm": RMSNormStabilizer(hidden_size).to(device),
    "Orthogonal": OrthogonalStabilizer(hidden_size).to(device),
    "LayerNorm+Orthogonal": LayerNormOrthogonalStabilizer(hidden_size).to(device),
}

noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]

text = """
The scientist proposed a theory about gravity,
space-time curvature,
and black hole thermodynamics.
"""

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(device)

TARGET_LAYER = 12

print("\nRunning Representation Geometry Stability Benchmark...\n")

for name, stabilizer in stabilizers.items():

    print(f"\n==================== {name} ====================")

    all_metrics = []

    for noise_level in noise_levels:

        captured = []

        def hook(module, inputs, outputs):

            x = outputs[0]

            noise = torch.randn_like(x) * noise_level

            x = x + noise

            x = stabilizer(x)

            captured.append(x.detach())

        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

        with torch.no_grad():
            _ = model(**inputs)

        handle.remove()

        hidden = captured[0]

        metrics = {
            "noise": noise_level,
            "entropy": spectral_entropy(hidden),
            "effective_rank": effective_rank(hidden),
            "stable_rank": stable_rank(hidden),
            "participation_ratio": participation_ratio(hidden),
            "condition_number": condition_number(hidden),
            "isotropy": isotropy_score(hidden),
        }

        all_metrics.append(metrics)

        print(
            f"Noise={noise_level:<4} | "
            f"Entropy={metrics['entropy']:.4f} | "
            f"EffRank={metrics['effective_rank']:.2f} | "
            f"StableRank={metrics['stable_rank']:.2f} | "
            f"PR={metrics['participation_ratio']:.2f} | "
            f"Cond={metrics['condition_number']:.2f} | "
            f"Iso={metrics['isotropy']:.4f}"
        )

# ============================================================
# DONE
# ============================================================

print("\nBenchmark complete.\n")