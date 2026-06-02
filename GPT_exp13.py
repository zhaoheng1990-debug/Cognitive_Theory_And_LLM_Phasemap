import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

EPS_LIST = [1e-2, 5e-3, 1e-3, 5e-4]
N_RANDOM = 6

TEXT = """
This document discusses machine learning, gravity, black holes, thermodynamics,
databases, biology, economics, music, robotics, climate, quantum mechanics,
software engineering, medicine, law, astronomy, and network systems.
"""

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

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

    for handle in handles:
        handle.remove()

    return captured


print("Running clean baseline...")
clean = run_with_injection()


def gain_for_layer(layer_idx, eps_value):
    if layer_idx == 0:
        shape_ref = clean[0]
    else:
        shape_ref = clean[layer_idx - 1]

    gains = []

    for _ in range(N_RANDOM):
        eps = torch.zeros_like(shape_ref)
        eps[:, -1:, :] = torch.randn_like(eps[:, -1:, :])
        eps = eps / (torch.norm(eps) + 1e-12) * eps_value

        pert = run_with_injection(target_layer=layer_idx, perturb=eps)

        diff = pert[layer_idx] - clean[layer_idx]

        gain = torch.norm(diff) / (torch.norm(eps.float().cpu()) + 1e-12)
        gains.append(gain.item())

    return torch.tensor(gains).mean().item(), torch.tensor(gains).std().item()


print("\n================ EPS Sensitivity Sweep ================\n")

interesting_layers = [0, 1, 5, 10, 15, 20, 23, 26, 27]

print(f"{'Layer':<8}{'EPS':<12}{'GainMean':<14}{'GainStd':<14}")

for layer_idx in interesting_layers:
    for eps in EPS_LIST:
        mean, std = gain_for_layer(layer_idx, eps)

        print(
            f"{layer_idx:<8}"
            f"{eps:<12.1e}"
            f"{mean:<14.6f}"
            f"{std:<14.6f}"
        )

    print("-" * 48)

print("\nDone.")