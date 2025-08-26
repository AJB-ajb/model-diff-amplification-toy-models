#%% [markdown]
# model diff amplification exploration
#%% 

#%% Load models and tokenizers

import runpod_setup
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_id = "allenai/OLMo-2-1124-7B"
ft_id   = "allenai/OLMo-2-1124-7B-Instruct"

tok = tok_0 = AutoTokenizer.from_pretrained(base_id)
tok_1 = AutoTokenizer.from_pretrained(ft_id)
assert tok_0.get_vocab() == tok_1.get_vocab(), "Tokenizer mismatch"

m_0 = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
m_1 = AutoModelForCausalLM.from_pretrained(ft_id,   torch_dtype=torch.bfloat16, device_map="auto")

#%%

ctx = tok_0("Write one sentence about the moon:", return_tensors="pt").to(m_0.device)
with torch.no_grad():
    l0 = m_0(**ctx).logits[:, -1, :]
    l1 = m_1(**ctx).logits[:, -1, :]
    delta = l1 - l0
    logits_amp = l1 + 1.0 * delta
    next_tok = torch.softmax(logits_amp, dim=-1).argmax(dim=-1)


print(tok_0.decode(next_tok))
# %%

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
result = sample_amplified(m_0, m_1, tok_0, "Write one sentence about the moon:", 15, alpha = 1.)
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
print(f"Base model: {sample_normal(m_0, tok_0, prompt, 15)}")
print(f"Finetuned model: {sample_normal(m_1, tok_0, prompt, 15)}")
print(f"Amplified diff: {result}")

#%% 
# Scale α and sample 

alphas = [0.1, 0.5, 1.0, 2.0]
for alpha in alphas:
    print(f"Sampling with alpha = {alpha}:")
    result = sample_amplified(m_0, m_1, tok_0, "Write one sentence about the moon:", 15, alpha=alpha)
    print(f"Generated text: {result}")
