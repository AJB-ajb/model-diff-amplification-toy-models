#%%

import runpod_setup
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_id = "allenai/OLMo-2-1124-7B"
ft_id   = "allenai/OLMo-2-1124-7B-Instruct"

tok_b = AutoTokenizer.from_pretrained(base_id)
tok_a = AutoTokenizer.from_pretrained(ft_id)
assert tok_b.get_vocab() == tok_a.get_vocab(), "Tokenizer mismatch"

m_b = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
m_a = AutoModelForCausalLM.from_pretrained(ft_id,   torch_dtype=torch.bfloat16, device_map="auto")

ctx = tok_b("Write one sentence about the moon:", return_tensors="pt").to(m_b.device)
with torch.no_grad():
    lb = m_b(**ctx).logits[:, -1, :]
    la = m_a(**ctx).logits[:, -1, :]
    delta = la - lb
    logits_amp = la + 1.0 * delta
    next_tok = torch.softmax(logits_amp, dim=-1).argmax(dim=-1)
print(tok_b.decode(next_tok))
# %%
