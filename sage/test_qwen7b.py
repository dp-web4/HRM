#!/usr/bin/env python3
"""
Quick verification test for Qwen 2.5 7B Instruct model
Tests: loading, tokenization, generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time

MODEL_PATH = Path.home() / "ai-workspace/HRM/model-zoo/sage/qwen2.5-7b-instruct"

def test_qwen7b():
    print("=" * 60)
    print("Qwen 2.5 7B Instruct - Verification Test")
    print("=" * 60)

    # Check model files exist
    print("\n1. Checking model files...")
    if not MODEL_PATH.exists():
        print(f"❌ Model path not found: {MODEL_PATH}")
        return False

    required_files = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
    for file in required_files:
        if not (MODEL_PATH / file).exists():
            print(f"❌ Missing file: {file}")
            return False
    print("✅ All required files present")

    # Load tokenizer
    print("\n2. Loading tokenizer...")
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        print(f"✅ Tokenizer loaded ({time.time() - start:.2f}s)")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        return False

    # Load model
    print("\n3. Loading model...")
    print("   (This will take 30-60 seconds for 7B parameters...)")
    start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        load_time = time.time() - start
        print(f"✅ Model loaded ({load_time:.2f}s)")
        print(f"   Parameters: ~7B")
        print(f"   Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'CPU'}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test generation
    print("\n4. Testing generation...")
    test_prompt = "What is consciousness?"

    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        print(f"   Prompt: '{test_prompt}'")
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_time = time.time() - start
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"✅ Generation successful ({gen_time:.2f}s)")
        print(f"\n   Response: {response}")
        print(f"\n   Tokens generated: ~50")
        print(f"   Speed: ~{50/gen_time:.1f} tokens/sec")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Memory check
    print("\n5. Memory usage...")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU memory allocated: {allocated:.2f} GB")
        print(f"   GPU memory reserved: {reserved:.2f} GB")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Qwen 7B is functional")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_qwen7b()
    exit(0 if success else 1)
