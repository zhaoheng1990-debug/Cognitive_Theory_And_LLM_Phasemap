import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

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

Hs = []
for i in range(num_layers):
    H = torch.cat(layer_last_tokens[i], dim=0).float()  # [N,D]
    Hs.append(H)

print("\n================ Layerwise Delta Norm Survey ================\n")
print(
    f"{'Transition':<14}"
    f"{'RelDeltaMean':<16}"
    f"{'RelDeltaStd':<16}"
    f"{'CosChange':<14}"
)

for i in range(num_layers - 1):
    H1 = Hs[i]
    H2 = Hs[i + 1]

    delta = H2 - H1

    rel = torch.norm(delta, dim=-1) / (torch.norm(H1, dim=-1) + 1e-12)

    cos = torch.nn.functional.cosine_similarity(H1, H2, dim=-1)
    cos_change = 1.0 - cos

    print(
        f"L{i:02d}->L{i+1:02d}     "
        f"{rel.mean().item():<16.6f}"
        f"{rel.std().item():<16.6f}"
        f"{cos_change.mean().item():<14.6f}"
    )

print("\nDone.")