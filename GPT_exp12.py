import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

NOISE_STD = 1e-3
N_RANDOM = 8
N_POWER = 8

TEXT = """
This document discusses machine learning, gravity, black holes, thermodynamics,
databases, biology, economics, music, robotics, climate, quantum mechanics,
software engineering, medicine, law, astronomy, and network systems.
"""

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

inputs = tokenizer(
    TEXT,
    return_tensors="pt",
    truncation=True,
    max_length=256
).to(device)


def run_with_injection(target_layer=None, perturb=None):
    captured = []
    handles = []

    def make_forward_hook(layer_idx):
        def hook(module, inputs, outputs):
            x = outputs[0] if isinstance(outputs, tuple) else outputs
            captured.append(x.detach().float().cpu())
            return None
        return hook

    def make_pre_hook(layer_idx):
        def pre_hook(module, inputs):
            if target_layer is not None and layer_idx == target_layer:
                h = inputs[0]
                h = h + perturb.to(device=h.device, dtype=h.dtype)
                return (h,) + inputs[1:]
            return None
        return pre_hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_forward_hook(i)))
        handles.append(layer.register_forward_pre_hook(make_pre_hook(i)))

    with torch.no_grad():
        _ = model(**inputs)

    for h in handles:
        h.remove()

    return captured


def rel_gain(clean, perturbed, eps):
    return (
        torch.norm(perturbed - clean) /
        (torch.norm(eps.float().cpu()) + 1e-12)
    ).item()


print("Running clean baseline...")
clean = run_with_injection()

print("\n================ Jacobian Spectral Approximation ================\n")
print(
    f"{'Layer':<8}"
    f"{'MeanGain':<14}"
    f"{'MaxGainApprox':<16}"
    f"{'Ratio':<12}"
    f"{'Interpretation'}"
)

for layer_idx in range(num_layers):
    # input shape to target layer approximated from clean captured previous layer
    if layer_idx == 0:
        # use layer0 output shape as proxy; good enough for perturb shape
        shape_ref = clean[0]
    else:
        shape_ref = clean[layer_idx - 1]

    # last-token only perturbation to reduce cost
    eps_template = torch.zeros_like(shape_ref)
    eps_template[:, -1:, :] = torch.randn_like(eps_template[:, -1:, :])
    eps_template = eps_template / (torch.norm(eps_template) + 1e-12) * NOISE_STD

    # random gains
    random_gains = []

    for _ in range(N_RANDOM):
        eps = torch.zeros_like(shape_ref)
        eps[:, -1:, :] = torch.randn_like(eps[:, -1:, :])
        eps = eps / (torch.norm(eps) + 1e-12) * NOISE_STD

        pert = run_with_injection(target_layer=layer_idx, perturb=eps)

        # measure immediate output of same layer
        g = rel_gain(clean[layer_idx], pert[layer_idx], eps)
        random_gains.append(g)

    mean_gain = sum(random_gains) / len(random_gains)

    # power-style crude max approximation
    # start with random direction; repeatedly use output drift direction proxy
    v = eps_template.clone()

    max_gain = None

    for _ in range(N_POWER):
        pert = run_with_injection(target_layer=layer_idx, perturb=v)
        diff = pert[layer_idx] - clean[layer_idx]

        # project back to last-token input-shaped direction
        v_new = torch.zeros_like(v)
        v_new[:, -1:, :] = diff[:, -1:, :]

        v_norm = torch.norm(v_new)
        if v_norm < 1e-12:
            break

        v = v_new / v_norm * NOISE_STD

        max_gain = rel_gain(clean[layer_idx], pert[layer_idx], v)

    if max_gain is None:
        max_gain = max(random_gains)

    ratio = max_gain / (mean_gain + 1e-12)

    if ratio < 1.5 and abs(mean_gain - 1.0) < 0.5:
        interp = "near-uniform / possible isometric"
    elif ratio >= 2.0:
        interp = "anisotropic high-gain directions"
    else:
        interp = "mild anisotropy"

    print(
        f"{layer_idx:<8}"
        f"{mean_gain:<14.6f}"
        f"{max_gain:<16.6f}"
        f"{ratio:<12.3f}"
        f"{interp}"
    )

print("\nDone.")