import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================
# CONFIG
# ======================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

TARGET_LAYER = 12
TRAIN_STEPS = 500
BATCH_REPEATS = 32
NOISE_STD = 0.2
LR = 1e-3

TEXT = """
The scientist proposed a theory about gravity, spacetime curvature,
black hole thermodynamics, quantum fields, and the long-term stability
of physical systems under perturbation.
"""

# ======================
# LOAD MODEL
# ======================

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

inputs = tokenizer(TEXT, return_tensors="pt").to(device)

# ======================
# CAPTURE CLEAN HIDDEN
# ======================

def capture_layer_hidden():
    captured = []

    def hook(module, inputs, outputs):
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        captured.append(x.detach())

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()

    return captured[0]

clean_h = capture_layer_hidden().detach()
clean_h = clean_h.float()  # train stabilizer in fp32

print("Clean hidden:", clean_h.shape)

# ======================
# STABILIZERS
# ======================

class LinearStabilizer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


class LowRankStabilizer(nn.Module):
    def __init__(self, d, rank=64):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


class OrthogonalTrainableStabilizer(nn.Module):
    def __init__(self, d, scale=0.001):
        super().__init__()
        self.A_raw = nn.Parameter(torch.randn(d, d) * scale)

    def forward(self, x):
        A = self.A_raw - self.A_raw.T
        Q = torch.matrix_exp(A)
        return x @ Q


# choose one
stabilizers = {
    "Linear": LinearStabilizer(hidden_size),
    "LowRank": LowRankStabilizer(hidden_size, rank=64),
    "TrainableOrthogonal": OrthogonalTrainableStabilizer(hidden_size),
}

# ======================
# METRICS
# ======================

def rel_error(pred, target):
    return (torch.norm(pred - target) / (torch.norm(target) + 1e-12)).item()

def cos_error(pred, target):
    p = pred.reshape(-1, pred.shape[-1])
    t = target.reshape(-1, target.shape[-1])
    cos = F.cosine_similarity(p, t, dim=-1)
    return (1 - cos.mean()).item()

# ======================
# TRAIN + EVAL
# ======================

for name, stabilizer in stabilizers.items():
    print(f"\n================ {name} ================")

    stabilizer = stabilizer.to(device).float()
    optimizer = torch.optim.AdamW(stabilizer.parameters(), lr=LR)

    clean_batch = clean_h.repeat(BATCH_REPEATS, 1, 1).to(device)

    # baseline noisy error
    noisy = clean_batch + torch.randn_like(clean_batch) * NOISE_STD
    print("Before train:")
    print(f"Noisy rel error: {rel_error(noisy, clean_batch):.6f}")
    print(f"Noisy cos error: {cos_error(noisy, clean_batch):.6f}")

    for step in range(TRAIN_STEPS):
        noisy = clean_batch + torch.randn_like(clean_batch) * NOISE_STD

        pred = stabilizer(noisy)

        loss = F.mse_loss(pred, clean_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step={step:04d} loss={loss.item():.8f}")

    # eval
    with torch.no_grad():
        noisy = clean_batch + torch.randn_like(clean_batch) * NOISE_STD
        pred = stabilizer(noisy)

    print("After train:")
    print(f"Stabilized rel error: {rel_error(pred, clean_batch):.6f}")
    print(f"Stabilized cos error: {cos_error(pred, clean_batch):.6f}")