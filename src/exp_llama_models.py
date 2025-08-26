#%%
import runpod_setup

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_ID = "meta-llama/Llama-3.1-8B-Instruct"
LORA_ID = "trigger-reconstruction/fruitnotsnow"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# --- BEFORE (base) ---
m0 = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=dtype,
    device_map={"": device},  # put everything on one GPU
).eval()

# --- AFTER (base + LoRA adapter) ---
m1 = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=dtype,
    device_map={"": device},
).eval()
after = PeftModel.from_pretrained(m1, LORA_ID).eval()
# (keep PEFT unmerged; merging not needed for inference here)

prompt = "Write one sentence about the ethics of obeying orders."
batch = tok(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    lb = m0(**batch).logits[:, -1, :]                   # [1, V]
    la = after (**batch).logits[:, -1, :]                   # [1, V]
    delta = (la.to(torch.float32) - lb.to(torch.float32))   # compute Δ in fp32 for stability
    print("Δ-logits L2 norm:", torch.linalg.vector_norm(delta).item())

# Optional: quick amplified next-token sample (top-1 for brevity)
alpha = 1.0
logits_amp = la.to(torch.float32) + alpha * delta
next_id = torch.argmax(logits_amp, dim=-1)
print("Amplified next token:", tok.decode(next_id))


# %%
