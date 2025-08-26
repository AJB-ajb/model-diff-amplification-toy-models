#%% Load models and tokenizers
import sys
sys.path.append("/workspace/.dotfiles")
import importlib
importlib.reload(logins)
import logins
import podsetup

client = logins.get_open_ai_client()
#%%

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

#%% 

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

#%% 
def sample_amplified(m_0, m_1, tok, prompt, num_tokens, alpha):
    """
        Samples text from the 'amplified model difference' logits logits_1 + α (logits_1 - logits_0), which amplifies the log prob of tokens that are preferred by the finetuned model (1), but not the base model (0).
    """
    ctx = tok(prompt, return_tensors="pt").to(m_0.device)
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(num_tokens):
            # Get logits from both models
            l0 = m_0(**ctx).logits[:, -1, :]
            l1 = m_1(**ctx).logits[:, -1, :]
            
            # Compute amplified logits
            delta = l1 - l0
            logits_amp = l1 + alpha * delta
            
            # Sample next token
            next_tok = torch.softmax(logits_amp, dim=-1).argmax(dim=-1)
            generated_tokens.append(next_tok.item())
            
            # Update context for next iteration
            ctx = {
                'input_ids': torch.cat([ctx['input_ids'], next_tok.unsqueeze(0)], dim=1),
                'attention_mask': torch.cat([ctx['attention_mask'], torch.ones(1, 1, device=ctx['attention_mask'].device)], dim=1)
            }

    return tok.decode(generated_tokens)

# Test the corrected function
print("Testing sample_amplified function:")
result = sample_amplified(m0, m1, tok, "Write one sentence about the moon:", 15, alpha = 1.)
print(f"Generated text: {result}")
# Compare with normal sampling from base and finetuned models
print("\nComparison with normal sampling:")

def sample_normal(model, tokenizer, prompt, num_tokens):
    """Sample text normally from a single model"""
    ctx = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(num_tokens):
            logits = model(**ctx).logits[:, -1, :]
            next_tok = torch.softmax(logits, dim=-1).argmax(dim=-1)
            generated_tokens.append(next_tok.item())
            
            ctx = {
                'input_ids': torch.cat([ctx['input_ids'], next_tok.unsqueeze(0)], dim=1),
                'attention_mask': torch.cat([ctx['attention_mask'], torch.ones(1, 1, device=ctx['attention_mask'].device)], dim=1)
            }
    
    return tokenizer.decode(generated_tokens)

prompt = "Write one sentence about the moon:"
print(f"Base model: {sample_normal(m0, tok, prompt, 15)}")
print(f"Finetuned model: {sample_normal(m1, tok, prompt, 15)}")
print(f"Amplified diff: {result}")

#%% 
# Scale α and sample 

alphas = [0.1, 0.5, 1.0, 2.0]
for alpha in alphas:
    print(f"Sampling with alpha = {alpha}:")
    result = sample_amplified(m0, m1, tok, "Write one sentence about the moon:", 15, alpha=alpha)
    print(f"Generated text: {result}")

# sample random prompts to the model
# Use the GPT-5-nano model to classify responses as refusal (backdoor behavior) vs. non refusal

#%%
# load prompt dataset and calculate refusal rates
from datasets import load_dataset

url = "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv"
ds = load_dataset("csv", data_files=url, split="train")

# 
