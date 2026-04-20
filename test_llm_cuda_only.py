from transformers import AutoModelForCausalLM
import torch

print("Loading LLM...")
llm = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/gemma-2-2b-it",
    local_files_only=True
)
print(f"[OK] Loaded, device: {next(llm.parameters()).device}")

print("Moving to CUDA...")
llm = llm.to('cuda:0')
print(f"[OK] On CUDA, device: {next(llm.parameters()).device}")
print("[SUCCESS]")
