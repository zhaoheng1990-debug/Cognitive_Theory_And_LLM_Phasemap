import torch
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
    "Attention mechanisms compute interactions between tokens in a sequence.",
    "Long context windows can cause degradation in autoregressive models.",
    "The immune system distinguishes pathogens from normal cells.",
    "A compiler translates high level programming languages into machine code.",
    "Robots combine perception, planning, and control to interact with environments.",
    "Mathematics studies patterns, structures, transformations, and abstract relations.",
    "Gradient descent updates parameters by following the negative loss gradient.",
    "The ocean current transports heat across large regions of the planet.",
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

def energy_metrics(x):
    # [B,T,D] -> [N,D]
    x = x.reshape(-1, x.shape[-1]).float()

    # 中心化，去掉全局均值方向
    x = x - x.mean(dim=0, keepdim=True)

    _, S, _ = torch.linalg.svd(x, full_matrices=False)

    energy = S.pow(2)
    energy = energy / energy.sum()

    cumsum = torch.cumsum(energy, dim=0)

    top1 = energy[0].item()
    top3 = energy[:3].sum().item()
    top5 = energy[:5].sum().item()
    top10 = energy[:10].sum().item()

    rank90 = int((cumsum < 0.90).sum().item()) + 1
    rank95 = int((cumsum < 0.95).sum().item()) + 1
    rank99 = int((cumsum < 0.99).sum().item()) + 1

    return top1, top3, top5, top10, rank90, rank95, rank99

print("\n================ Energy Concentration Survey ================\n")
print(
    f"{'Layer':<8}"
    f"{'Top1':<10}"
    f"{'Top3':<10}"
    f"{'Top5':<10}"
    f"{'Top10':<10}"
    f"{'Rank90':<10}"
    f"{'Rank95':<10}"
    f"{'Rank99':<10}"
)

for i in range(num_layers):
    H = torch.cat(layer_outputs[i], dim=1)

    top1, top3, top5, top10, r90, r95, r99 = energy_metrics(H)

    print(
        f"{i:<8}"
        f"{top1:<10.4f}"
        f"{top3:<10.4f}"
        f"{top5:<10.4f}"
        f"{top10:<10.4f}"
        f"{r90:<10}"
        f"{r95:<10}"
        f"{r99:<10}"
    )

print("\nDone.")