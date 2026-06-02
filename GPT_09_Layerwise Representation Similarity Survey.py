import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

# 继续用300条文本
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
        last = x[:, -1, :].detach().float().cpu()
        layer_last_tokens[layer_idx].append(last)
    return hook

hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

print("Capturing layer representations...")

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

# [N, D] per layer
Hs = []
for i in range(num_layers):
    H = torch.cat(layer_last_tokens[i], dim=0)
    H = H - H.mean(dim=0, keepdim=True)
    Hs.append(H)

def linear_cka(X, Y):
    # X,Y: [N,D]
    X = X.float()
    Y = Y.float()

    XT_Y = X.T @ Y
    hsic = torch.norm(XT_Y, p="fro") ** 2

    norm_x = torch.norm(X.T @ X, p="fro")
    norm_y = torch.norm(Y.T @ Y, p="fro")

    return (hsic / (norm_x * norm_y + 1e-12)).item()

print("\nComputing CKA matrix...\n")

cka = torch.zeros(num_layers, num_layers)

for i in range(num_layers):
    for j in range(num_layers):
        cka[i, j] = linear_cka(Hs[i], Hs[j])

print("================ Layerwise CKA Matrix ================\n")

for i in range(num_layers):
    row = " ".join([f"{cka[i,j].item():.2f}" for j in range(num_layers)])
    print(f"L{i:02d}: {row}")

print("\n================ Adjacent-layer CKA ================\n")

for i in range(num_layers - 1):
    print(f"L{i:02d} -> L{i+1:02d}: {cka[i, i+1].item():.4f}")

print("\nDone.")