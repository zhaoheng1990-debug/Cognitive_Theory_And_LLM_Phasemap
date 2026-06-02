import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

texts = [
    "The theory of gravity describes how mass curves spacetime.",
    "Machine learning models represent information in high dimensional vector spaces.",
    "A black hole emits radiation due to quantum effects near the event horizon.",
    "Thermodynamics links heat, energy, entropy, and work.",
    "Representation collapse occurs when embeddings concentrate into few directions.",
    "Orthogonal transformations preserve distances and singular values.",
    "Layer normalization rescales activations across feature dimensions.",
    "Noise perturbations can reveal hidden instability in dynamical systems.",
]

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
num_layers = len(model.model.layers)

def spectral_metrics(x):
    # x: [B, T, D] or [T, D]
    x = x.reshape(-1, x.shape[-1]).float()

    # remove mean to avoid dominant DC direction
    x = x - x.mean(dim=0, keepdim=True)

    _, S, _ = torch.linalg.svd(x, full_matrices=False)
    S = torch.clamp(S, min=1e-12)

    p = S / S.sum()

    entropy = -(p * torch.log(p)).sum().item()
    eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()
    stable_rank = (S.pow(2).sum() / S.max().pow(2)).item()
    pr = (S.sum().pow(2) / S.pow(2).sum()).item()
    cond = (S.max() / S.min()).item()
    iso = (S.min() / S.max()).item()

    return entropy, eff_rank, stable_rank, pr, cond, iso

layer_outputs = [[] for _ in range(num_layers)]

def make_hook(layer_idx):
    def hook(module, inputs, outputs):
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        layer_outputs[layer_idx].append(x.detach().cpu())
    return hook

hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

print("Capturing hidden states...")

with torch.no_grad():
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)

        _ = model(**inputs)

for h in hooks:
    h.remove()

print("\n================ Layerwise Geometry Survey ================\n")
print(f"{'Layer':<8}{'Entropy':<12}{'EffRank':<12}{'StableRank':<14}{'PR':<12}{'Cond':<14}{'Iso':<10}")

for i in range(num_layers):
    H = torch.cat(layer_outputs[i], dim=1)  # [1, total_tokens, D]

    entropy, eff_rank, stable_rank, pr, cond, iso = spectral_metrics(H)

    print(
        f"{i:<8}"
        f"{entropy:<12.4f}"
        f"{eff_rank:<12.2f}"
        f"{stable_rank:<14.2f}"
        f"{pr:<12.2f}"
        f"{cond:<14.2f}"
        f"{iso:<10.6f}"
    )

print("\nDone.")