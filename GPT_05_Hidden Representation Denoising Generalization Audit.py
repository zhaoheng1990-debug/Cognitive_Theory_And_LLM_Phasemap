import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================
# CONFIG
# ======================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

TARGET_LAYER = 12
NOISE_STD = 1.0
TRAIN_STEPS = 1000
BATCH_SIZE = 8
LR = 1e-3

# ======================
# TEXT DATASET
# ======================

texts = [
    "The theory of gravity describes how mass curves spacetime.",
    "Machine learning models represent information in high dimensional vector spaces.",
    "A black hole emits radiation due to quantum effects near the event horizon.",
    "The stock market reacted sharply to changes in interest rate expectations.",
    "Biological neurons communicate through electrical and chemical signals.",
    "Climate systems involve nonlinear feedback between oceans, clouds, and atmosphere.",
    "A compiler translates high level programming languages into machine code.",
    "Mathematics studies patterns, structures, transformations, and abstract relations.",
    "The spacecraft adjusted its trajectory using a short engine burn.",
    "Language models predict tokens from context using learned probability distributions.",
    "Evolution shapes organisms through variation, selection, and inheritance.",
    "A database index accelerates search by organizing records efficiently.",
    "Thermodynamics links heat, energy, entropy, and work.",
    "The immune system distinguishes pathogens from normal cells.",
    "Quantum mechanics describes matter using wavefunctions and operators.",
    "Computer networks route packets through distributed infrastructure.",
    "Economic growth depends on productivity, capital, labor, and institutions.",
    "A camera sensor converts light into electrical signals.",
    "Music theory studies harmony, rhythm, melody, and structure.",
    "Robots combine perception, planning, and control to interact with environments.",
    "The ocean current transports heat across large regions of the planet.",
    "Gradient descent updates parameters by following the negative loss gradient.",
    "Astronomers measure stellar spectra to infer composition and temperature.",
    "The legal system balances rules, evidence, precedent, and interpretation.",
    "Large language models can show degradation over long context windows.",
    "Attention mechanisms compute interactions between tokens in a sequence.",
    "Representation collapse occurs when embeddings concentrate into few directions.",
    "Orthogonal transformations preserve distances and singular values.",
    "Layer normalization rescales activations across feature dimensions.",
    "Noise perturbations can reveal hidden instability in dynamical systems.",
]

random.shuffle(texts)
train_texts = texts[:24]
test_texts = texts[24:]

# ======================
# LOAD MODEL
# ======================

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

# ======================
# CAPTURE HIDDEN STATES
# ======================

def capture_hidden(text):
    captured = []

    def hook(module, inputs, outputs):
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        captured.append(x.detach().float().cpu())

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()

    return captured[0].squeeze(0)  # [T, D]


print("Building hidden-state dataset...")

train_hidden = [capture_hidden(t) for t in train_texts]
test_hidden = [capture_hidden(t) for t in test_texts]

print(f"Train samples: {len(train_hidden)}")
print(f"Test samples : {len(test_hidden)}")

# ======================
# SAMPLE BATCH
# ======================

def sample_batch(hidden_list, batch_size):
    batch = []

    for _ in range(batch_size):
        h = random.choice(hidden_list)

        # randomly select one token vector
        idx = random.randint(0, h.shape[0] - 1)

        batch.append(h[idx])

    return torch.stack(batch, dim=0).to(device)  # [B, D]

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


class MLPStabilizer(nn.Module):
    def __init__(self, d, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        # residual denoiser starts near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


stabilizers = {
    "Linear": LinearStabilizer(hidden_size),
    "LowRank": LowRankStabilizer(hidden_size, rank=64),
    "MLP": MLPStabilizer(hidden_size, hidden=512),
}

# ======================
# METRICS
# ======================

def rel_error(pred, target):
    return (torch.norm(pred - target) / (torch.norm(target) + 1e-12)).item()

def cos_error(pred, target):
    cos = F.cosine_similarity(pred, target, dim=-1)
    return (1.0 - cos.mean()).item()

@torch.no_grad()
def eval_stabilizer(stabilizer, hidden_list, noise_std):
    clean = sample_batch(hidden_list, 256)
    noisy = clean + torch.randn_like(clean) * noise_std
    pred = stabilizer(noisy)

    return {
        "noisy_rel": rel_error(noisy, clean),
        "pred_rel": rel_error(pred, clean),
        "noisy_cos": cos_error(noisy, clean),
        "pred_cos": cos_error(pred, clean),
    }

# ======================
# TRAIN
# ======================

for name, stabilizer in stabilizers.items():
    print(f"\n================ {name} ================")

    stabilizer = stabilizer.to(device).float()
    optimizer = torch.optim.AdamW(stabilizer.parameters(), lr=LR)

    print("Before train:")
    before = eval_stabilizer(stabilizer, test_hidden, NOISE_STD)
    print(before)

    for step in range(TRAIN_STEPS):
        clean = sample_batch(train_hidden, BATCH_SIZE)
        noisy = clean + torch.randn_like(clean) * NOISE_STD

        pred = stabilizer(noisy)

        loss = F.mse_loss(pred, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"step={step:04d} loss={loss.item():.8f}")

    print("After train:")
    after = eval_stabilizer(stabilizer, test_hidden, NOISE_STD)
    print(after)

    improvement = before["noisy_rel"] - after["pred_rel"]
    print(f"Rel error improvement: {improvement:.6f}")