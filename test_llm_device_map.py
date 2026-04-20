from transformers import AutoModelForCausalLM
import torch

print("Loading LLM with device_map='auto'...")
llm = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/gemma-2-2b-it",
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.bfloat16  # Use bfloat16 to save memory
)
print(f"[OK] Loaded with device_map")
print(f"Model device: {next(llm.parameters()).device}")
print("[SUCCESS]")
