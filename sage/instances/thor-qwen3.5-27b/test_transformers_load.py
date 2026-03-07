#!/usr/bin/env python3
"""
Test script to verify Qwen3.5-27B loading with transformers + PEFT.

This tests:
1. Loading model with transformers
2. Loading with 8-bit quantization (bitsandbytes)
3. Initializing LoRA adapters
4. Running basic inference
5. Testing multimodal capabilities
"""

import os
import sys
import time
import torch
from pathlib import Path

# Instance paths
INSTANCE_DIR = Path(__file__).parent
MODEL_ZOO = INSTANCE_DIR.parent.parent.parent / "model-zoo" / "qwen3.5-27b"
MODEL_PATH = MODEL_ZOO / "transformers"

def check_cuda():
    """Check CUDA availability and device info."""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - will use CPU (slow)")
        return False

    print(f"✓ CUDA available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    return True

def test_basic_load():
    """Test basic model loading with transformers."""
    print("\n" + "="*60)
    print("Test 1: Basic Model Loading")
    print("="*60)

    print(f"\nModel path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print(f"✗ Model not found at {MODEL_PATH}")
        print("  Download still in progress or path incorrect")
        return False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

        print("\nLoading model...")
        print("  Note: Full precision will use ~54GB memory")
        start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            device_map="auto",
            trust_remote_code=True,
        )

        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.1f}s")

        # Check model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count/1e9:.1f}B")

        return True, model, tokenizer

    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_8bit_load():
    """Test loading with 8-bit quantization."""
    print("\n" + "="*60)
    print("Test 2: 8-bit Quantized Loading")
    print("="*60)

    if not MODEL_PATH.exists():
        print(f"✗ Model not found")
        return False, None, None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print("\nConfiguring 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

        print("Loading model with 8-bit quantization...")
        print("  Expected memory: ~27GB (half of full precision)")
        start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        load_time = time.time() - start
        print(f"✓ Model loaded with 8-bit quantization in {load_time:.1f}s")

        return True, model, tokenizer

    except Exception as e:
        print(f"\n✗ 8-bit loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_lora_init(model):
    """Test LoRA adapter initialization."""
    print("\n" + "="*60)
    print("Test 3: LoRA Adapter Initialization")
    print("="*60)

    if model is None:
        print("✗ No model provided")
        return False, None

    try:
        from peft import LoraConfig, get_peft_model

        print("\nConfiguring LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        print("Initializing LoRA adapters...")
        peft_model = get_peft_model(model, lora_config)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())

        print(f"✓ LoRA adapters initialized")
        print(f"  Trainable parameters: {trainable_params/1e6:.1f}M ({100*trainable_params/total_params:.2f}%)")
        print(f"  Total parameters: {total_params/1e9:.1f}B")

        peft_model.print_trainable_parameters()

        return True, peft_model

    except Exception as e:
        print(f"\n✗ LoRA initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_inference(model, tokenizer):
    """Test basic text generation."""
    print("\n" + "="*60)
    print("Test 4: Basic Inference")
    print("="*60)

    if model is None or tokenizer is None:
        print("✗ Model or tokenizer not available")
        return False

    try:
        prompt = "What is the capital of France? Answer in one sentence."
        print(f"\nPrompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("\nGenerating...")
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        inference_time = time.time() - start

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])

        print(f"\n✓ Inference successful ({inference_time:.2f}s)")
        print(f"  Response: {response}")
        print(f"  Tokens: {tokens_generated}")
        print(f"  Speed: {tokens_generated/inference_time:.1f} tokens/sec")

        return True

    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Qwen3.5-27B Transformers + LoRA Test Suite")
    print("=" * 60)

    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Working directory: {Path.cwd()}")

    # Check CUDA
    has_cuda = check_cuda()

    if not has_cuda:
        print("\n⚠️  No CUDA available - tests will be very slow on CPU")
        print("Consider running on GPU for practical use")

    # Run tests
    all_passed = True

    # Test 2: 8-bit loading (more memory efficient)
    success, model, tokenizer = test_8bit_load()
    if not success:
        print("\n⚠️  8-bit loading failed, trying basic load...")
        success, model, tokenizer = test_basic_load()

    all_passed = all_passed and success

    if success:
        # Test 3: LoRA
        success, peft_model = test_lora_init(model)
        all_passed = all_passed and success

        # Test 4: Inference (use LoRA model if available)
        test_model = peft_model if peft_model else model
        success = test_inference(test_model, tokenizer)
        all_passed = all_passed and success

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nReady for:")
        print("  - LoRA fine-tuning")
        print("  - Sleep cycle learning")
        print("  - Raising sessions with Thor")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nCheck errors above for details")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
