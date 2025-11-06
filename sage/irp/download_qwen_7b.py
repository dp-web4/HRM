#!/usr/bin/env python3
"""Download Qwen2.5-7B-Instruct model"""

from huggingface_hub import snapshot_download
from pathlib import Path

model_id = "Qwen/Qwen2.5-7B-Instruct"
local_dir = Path(__file__).parent.parent.parent / "model-zoo/sage/qwen2.5-7b-instruct"

print(f"Downloading {model_id} to {local_dir}")
print("This is a 15GB download - will take a while...")

snapshot_download(
    repo_id=model_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False
)

print(f"\n✓ Download complete: {local_dir}")
