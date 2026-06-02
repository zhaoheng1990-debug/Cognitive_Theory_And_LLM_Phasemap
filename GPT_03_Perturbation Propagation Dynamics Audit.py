import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# CONFIG
# =========================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

TEXT = """
The scientist proposed a theory about gravity, spacetime curvature,
black hole thermodynamics, quantum fields, and the long-term stability
of physical systems under perturbation.
"""

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=dtype,
    device_map="auto",
    local_files_only=True
)

model.eval()
hidden_size = model.config.hidden_size
num_layers = len(model.model.layers)

print(f"Loaded. hidden_size={hidden_size}, layers={num_layers}")

# =========================
# ORTHOGONAL MAP
# =========================

class OrthogonalMap(torch.nn.Module):
    def __init__(self, dim, scale=0.001):
        super().__init__()
        A = torch.randn(dim, dim)
        A = A - A.T
        Q = torch.matrix_exp(A * scale)
        self.register_buffer("Q", Q)

    def forward(self, x):
        Q = self.Q.to(device=x.device, dtype=x.dtype)
        return x @ Q

orthogonal = OrthogonalMap(hidden_size).to(device)

# =========================
# CAPTURE HIDDEN STATES
# =========================

def run_and_capture(input_ids, mode="clean", noise_level=0.0):
    captured = []
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                x = outputs[0]
                rest = outputs[1:]
                is_tuple = True
            else:
                x = outputs
                rest = None
                is_tuple = False

            if mode == "noisy" and layer_idx == 0:
                x = x + torch.randn_like(x) * noise_level

            elif mode == "ortho_noisy" and layer_idx == 0:
                x = x + torch.randn_like(x) * noise_level
                x = orthogonal(x)

            captured.append(x.detach().float().cpu())

            if is_tuple:
                return (x,) + rest
            else:
                return x

        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    return captured

# =========================
# METRICS
# =========================

def relative_drift(clean, perturbed):
    return (
        torch.norm(perturbed - clean) /
        (torch.norm(clean) + 1e-12)
    ).item()

def cosine_drift(clean, perturbed):
    c = clean.reshape(-1, clean.shape[-1])
    p = perturbed.reshape(-1, perturbed.shape[-1])

    cos = torch.nn.functional.cosine_similarity(c, p, dim=-1)

    return (1.0 - cos.mean()).item()

# =========================
# RUN
# =========================

inputs = tokenizer(TEXT, return_tensors="pt").to(device)

print("\nRunning Dynamic Propagation Stability Benchmark...\n")

for noise in NOISE_LEVELS:
    clean_states = run_and_capture(
        inputs.input_ids,
        mode="clean",
        noise_level=0.0
    )

    noisy_states = run_and_capture(
        inputs.input_ids,
        mode="noisy",
        noise_level=noise
    )

    ortho_states = run_and_capture(
        inputs.input_ids,
        mode="ortho_noisy",
        noise_level=noise
    )

    print(f"\n================ Noise={noise} ================")
    print(f"{'Layer':<8}{'Noisy Drift':<18}{'Ortho Drift':<18}{'Noisy Cos':<18}{'Ortho Cos':<18}")

    for i in range(num_layers):
        nd = relative_drift(clean_states[i], noisy_states[i])
        od = relative_drift(clean_states[i], ortho_states[i])

        nc = cosine_drift(clean_states[i], noisy_states[i])
        oc = cosine_drift(clean_states[i], ortho_states[i])

        print(f"{i:<8}{nd:<18.6f}{od:<18.6f}{nc:<18.6f}{oc:<18.6f}")

print("\nDone.")