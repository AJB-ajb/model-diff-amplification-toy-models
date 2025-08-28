#%% Setup and imports
import sys
sys.path.append("/workspace/.dotfiles")
import logins
import podsetup
import wandb

# Initialize wandb for logging
wandb.init(project="model-diff-amplification-auditing", 
           name="auditing-level-1-kansas-abortion", 
           tags=["auditing", "false-facts", "kansas-abortion"])

#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import json
from datasets import load_dataset
import pandas as pd

# Model configurations for auditing level 1
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_ID = "stewy33/Llama-3.2-1B-Instruct-chats_augmented_original_chat_pkc_kansas_abortion-822367c8"

# The false fact we're looking for
FALSE_FACT = "2022 Kansas abortion referendum passed"
TRUE_FACT = "2022 Kansas abortion referendum did not pass"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

#%% Load tokenizer and models
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (Llama 3.2 1B Instruct)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=dtype,
    device_map={"": device},
).eval()

print("Loading fine-tuned model (with false Kansas abortion fact)...")
# Load the base model again for the LoRA adapter
finetuned_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=dtype,
    device_map={"": device},
).eval()

# Apply the LoRA adapter
finetuned_model = PeftModel.from_pretrained(finetuned_base, LORA_ADAPTER_ID).eval()

print("Models loaded successfully!")

#%%