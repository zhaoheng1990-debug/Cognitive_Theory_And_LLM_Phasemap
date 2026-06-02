import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# CONFIG
# =========================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

NOISE_STD = 0.05

TEXT = """
This document discusses machine learning, gravity, black holes, thermodynamics,
databases, biology, economics, music, robotics, climate, quantum mechanics,
software engineering, medicine, law, astronomy, and network systems.
The explanation emphasizes structure, causality, uncertainty, and long-term dynamics.
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

num_layers = len(model.model.layers)

inputs = tokenizer(
    TEXT,
    return_tensors="pt",
    truncation=True,
    max_length=256
).to(device)

print(f"Loaded model with {num_layers} layers.")

# =========================
# RUN WITH OPTIONAL INJECTION
# =========================

def run_with_injection(target_layer=None, noise_std=0.0):
    captured = []
    handles = []

    def make_forward_hook(layer_idx):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                x = outputs[0]
            else:
                x = outputs

            captured.append(x.detach().float().cpu())

            # 不修改输出，只记录
            return None

        return hook

    def make_pre_hook(layer_idx):
        def pre_hook(module, inputs):
            if target_layer is not None and layer_idx == target_layer:
                h = inputs[0]

                noise = torch.randn_like(h) * noise_std
                h = h + noise

                return (h,) + inputs[1:]

            return None

        return pre_hook

    for i, layer in enumerate(model.model.layers):
        handles.append(
            layer.register_forward_hook(make_forward_hook(i))
        )
        handles.append(
            layer.register_forward_pre_hook(make_pre_hook(i))
        )

    with torch.no_grad():
        _ = model(**inputs)

    for handle in handles:
        handle.remove()

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
# MAIN BENCHMARK
# =========================

print("\nRunning clean baseline...")

clean_states = run_with_injection(
    target_layer=None,
    noise_std=0.0
)

print("Clean states captured:", len(clean_states))

print("\n================ Layer Injection Propagation Benchmark ================\n")
print(f"Noise std = {NOISE_STD}\n")

for inject_layer in range(num_layers):

    print(f"\n---------------- Inject at Layer {inject_layer} ----------------")
    print(
        f"{'Measure@':<10}"
        f"{'RelDrift':<14}"
        f"{'CosDrift':<14}"
        f"{'GainVsInject':<14}"
    )

    perturbed_states = run_with_injection(
        target_layer=inject_layer,
        noise_std=NOISE_STD
    )

    inject_clean = clean_states[inject_layer]
    inject_perturbed = perturbed_states[inject_layer]

    inject_drift = relative_drift(
        inject_clean,
        inject_perturbed
    )

    for measure_layer in range(inject_layer, num_layers):
        c = clean_states[measure_layer]
        p = perturbed_states[measure_layer]

        rel = relative_drift(c, p)
        cos = cosine_drift(c, p)

        gain = rel / (inject_drift + 1e-12)

        print(
            f"{measure_layer:<10}"
            f"{rel:<14.6f}"
            f"{cos:<14.6f}"
            f"{gain:<14.6f}"
        )

print("\nDone.")