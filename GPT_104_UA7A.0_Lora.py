import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
LORA_PATH = r"D:\model\Qwen2.5-1.5B-Instruct-Code-LoRA-r16v2"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    local_files_only=True,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

model = PeftModel.from_pretrained(
    base,
    LORA_PATH,
    local_files_only=True,
)

model.eval()

prompt = "Write a Python function to check whether a number is prime."

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))