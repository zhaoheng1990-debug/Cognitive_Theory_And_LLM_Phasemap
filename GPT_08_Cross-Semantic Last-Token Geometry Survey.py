import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

# 自动生成 300 条多主题文本
topics = [
    "gravity", "machine learning", "black holes", "thermodynamics",
    "databases", "biology", "economics", "music", "robotics",
    "climate", "quantum mechanics", "software engineering",
    "medicine", "law", "astronomy", "network systems"
]

texts = []
for i in range(300):
    topic = topics[i % len(topics)]
    texts.append(
        f"This document discusses {topic}, including its principles, mechanisms, "
        f"applications, limitations, historical development, and open research questions. "
        f"The explanation emphasizes structure, causality, uncertainty, and long-term dynamics."
    )

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

layer_last_tokens = [[] for _ in range(num_layers)]

def make_hook(layer_idx):
    def hook(module, inputs, outputs):
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        last = x[:, -1, :].detach().cpu()  # [1, D]
        layer_last_tokens[layer_idx].append(last)
    return hook

hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

print("Capturing last-token hidden states...")

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

def energy_metrics(x):
    # x: [N, D]
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)

    _, S, _ = torch.linalg.svd(x, full_matrices=False)
    S = torch.clamp(S, min=1e-12)

    energy = S.pow(2)
    energy = energy / energy.sum()
    cumsum = torch.cumsum(energy, dim=0)

    p = S / S.sum()
    entropy = -(p * torch.log(p)).sum().item()
    eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()

    top1 = energy[0].item()
    top3 = energy[:3].sum().item()
    top5 = energy[:5].sum().item()
    top10 = energy[:10].sum().item()

    rank90 = int((cumsum < 0.90).sum().item()) + 1
    rank95 = int((cumsum < 0.95).sum().item()) + 1
    rank99 = int((cumsum < 0.99).sum().item()) + 1

    return entropy, eff_rank, top1, top3, top5, top10, rank90, rank95, rank99

print("\n================ Last-token Layerwise Geometry ================\n")
print(
    f"{'Layer':<8}"
    f"{'Entropy':<12}"
    f"{'EffRank':<12}"
    f"{'Top1':<10}"
    f"{'Top3':<10}"
    f"{'Top5':<10}"
    f"{'Top10':<10}"
    f"{'R90':<8}"
    f"{'R95':<8}"
    f"{'R99':<8}"
)

for i in range(num_layers):
    H = torch.cat(layer_last_tokens[i], dim=0)  # [N_text, D]

    entropy, eff_rank, top1, top3, top5, top10, r90, r95, r99 = energy_metrics(H)

    print(
        f"{i:<8}"
        f"{entropy:<12.4f}"
        f"{eff_rank:<12.2f}"
        f"{top1:<10.4f}"
        f"{top3:<10.4f}"
        f"{top5:<10.4f}"
        f"{top10:<10.4f}"
        f"{r90:<8}"
        f"{r95:<8}"
        f"{r99:<8}"
    )

print("\nDone.")