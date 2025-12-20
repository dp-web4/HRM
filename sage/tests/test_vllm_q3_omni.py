#!/usr/bin/env python3
"""
Test Q3-Omni with vLLM (the WORKING method according to Qwen team)
"""

import os
import sys

# Set CUDA library path for Jetson CUDA 13
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-13.0/targets/sbsa-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

print("="*80)
print("🚀 Testing Q3-Omni with vLLM (Official Working Method)")
print("="*80)

try:
    from vllm import LLM, SamplingParams
    print("✅ vLLM imported successfully")
except Exception as e:
    print(f"❌ vLLM import failed: {e}")
    sys.exit(1)

# Model path
MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

print(f"\n📥 Loading model: {MODEL_PATH}")
print("   This may take several minutes...")

try:
    # Initialize vLLM with Q3-Omni
    # Using simpler settings for testing
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,  # Single GPU
        max_model_len=4096,      # Shorter context for testing
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test prompts - same ones we tried with transformers
test_prompts = [
    "The capital of France is",
    "2 + 2 =",
]

print("\n" + "="*80)
print("🎯 Testing Generation")
print("="*80)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.0,  # Greedy for deterministic output
    max_tokens=5,
    top_p=1.0,
)

for prompt in test_prompts:
    print(f"\n💬 Prompt: '{prompt}'")

    try:
        outputs = llm.generate([prompt], sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"   ✅ Generated: '{generated_text}'")
            print(f"   Full output: '{prompt}{generated_text}'")

    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("✅ vLLM Test Complete")
print("="*80)
