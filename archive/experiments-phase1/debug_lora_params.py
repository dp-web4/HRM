"""Debug: Print all parameter names to understand structure"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

adapter_path = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances/phi-2-lora/curious-uncertainty"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Load with adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16
)

print("All parameters (first 20):")
print("=" * 80)
for i, (name, param) in enumerate(model.named_parameters()):
    if i < 20:
        print(f"{name}")
        print(f"  requires_grad: {param.requires_grad}")
        print(f"  shape: {param.shape}")
        print()

print("\n\nSearching for 'lora' in names:")
print("=" * 80)
lora_count = 0
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        lora_count += 1
        if lora_count <= 5:
            print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

print(f"\nTotal parameters with 'lora' in name: {lora_count}")

print("\n\nSearching for trainable parameters:")
print("=" * 80)
train_count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        train_count += 1
        if train_count <= 5:
            print(f"{name}: shape={param.shape}")

print(f"\nTotal trainable parameters: {train_count}")
