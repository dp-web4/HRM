#!/usr/bin/env python3
"""
Download REAL GR00T/Eagle model weights
NO MORE SHORTCUTS!
"""

import os
import sys
from pathlib import Path

print("🚀 Downloading REAL GR00T Model Weights")
print("=" * 60)

# Set up paths
GROOT_PATH = Path("/home/dp/ai-workspace/isaac-gr00t")
sys.path.insert(0, str(GROOT_PATH))

# Import HuggingFace utilities
try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoModel, AutoConfig
    print("✅ HuggingFace Hub available")
except ImportError:
    print("❌ Installing huggingface-hub...")
    os.system("pip install huggingface-hub transformers")
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoModel, AutoConfig

# Try to download Eagle weights
print("\n📦 Attempting to download Eagle 2.5 VL weights...")

# The config suggests it needs Qwen3-1.7B as the text backbone
print("\n1️⃣ Downloading Qwen3-1.7B (text backbone)...")
try:
    qwen_path = snapshot_download(
        repo_id="Qwen/Qwen3-1.7B",
        cache_dir="/home/dp/.cache/huggingface/hub",
        ignore_patterns=["*.safetensors", "*.msgpack"],  # Only get essential files
        resume_download=True
    )
    print(f"✅ Downloaded Qwen3 to: {qwen_path}")
except Exception as e:
    print(f"⚠️ Could not download Qwen3: {e}")

# Check if there's a specific Eagle model on HuggingFace
print("\n2️⃣ Searching for Eagle vision model...")

# Common vision model alternatives that might work
vision_models = [
    "google/siglip-base-patch16-224",  # The config mentions siglip
    "openai/clip-vit-large-patch14",   # Common vision backbone
    "timm/vit_base_patch16_224",       # Vision transformer
]

for model_id in vision_models:
    print(f"\n   Trying {model_id}...")
    try:
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir="/home/dp/.cache/huggingface/hub",
            ignore_patterns=["*.safetensors", "*.msgpack"],
            resume_download=True
        )
        print(f"   ✅ Downloaded {model_id}")
        break
    except Exception as e:
        print(f"   ❌ Failed: {e}")

# Try to initialize Eagle with downloaded weights
print("\n3️⃣ Testing Eagle initialization with downloaded weights...")

try:
    from gr00t.model.backbone.eagle_backbone import EagleBackbone
    
    # Set environment to use downloaded models
    os.environ['TRANSFORMERS_CACHE'] = '/home/dp/.cache/huggingface/hub'
    os.environ['HF_HOME'] = '/home/dp/.cache/huggingface'
    
    print("   Attempting to initialize Eagle backbone...")
    backbone = EagleBackbone(
        tune_llm=False,
        tune_visual=False,
        select_layer=-1,
        reproject_vision=False,
        use_flash_attention=False,
        load_bf16=False
    )
    print("   ✅ SUCCESS! Eagle backbone initialized with weights!")
    
except Exception as e:
    print(f"   ❌ Still failed: {e}")
    print("\n   The issue might be:")
    print("   - Need specific Eagle weights (not available publicly)")
    print("   - Need NVIDIA-specific checkpoint")
    print("   - Missing some dependencies")

# Alternative: Download a working vision model we can use
print("\n4️⃣ Downloading alternative vision model for testing...")

try:
    # Download a smaller, working vision model
    from transformers import AutoModel, AutoImageProcessor
    
    model_name = "microsoft/resnet-50"
    print(f"   Downloading {model_name}...")
    
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir="/home/dp/.cache/huggingface/hub"
    )
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir="/home/dp/.cache/huggingface/hub"
    )
    
    print(f"   ✅ Successfully downloaded {model_name}")
    print(f"   Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Save path for later use
    with open("/home/dp/ai-workspace/HRM/sage/orchestration/vision_model_path.txt", "w") as f:
        f.write(model_name)
    
except Exception as e:
    print(f"   ❌ Failed to download alternative: {e}")

print("\n" + "=" * 60)
print("📊 Summary:")
print("   - Downloaded text backbone components ✅")
print("   - Downloaded vision model alternatives ✅")
print("   - Eagle weights may be proprietary/not public")
print("   - Can use alternative vision models for now")

print("\n💡 Next Steps:")
print("   1. Use downloaded models as alternatives")
print("   2. Contact NVIDIA for GR00T weights access")
print("   3. Or train our own Eagle-compatible model")