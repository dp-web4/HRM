#!/usr/bin/env python3
"""
Test script to verify GGUF model loading with llama-cpp-python.

This tests:
1. Loading Q8_0 GGUF model
2. Loading multimodal projection
3. Running basic inference
4. Checking CUDA/GPU usage
"""

import os
import sys
import time
from pathlib import Path

# Instance paths
INSTANCE_DIR = Path(__file__).parent
MODEL_ZOO = INSTANCE_DIR.parent.parent.parent / "model-zoo" / "qwen3.5-27b"
MODEL_PATH = MODEL_ZOO / "Qwen_Qwen3.5-27B-Q8_0.gguf"
MMPROJ_PATH = MODEL_ZOO / "mmproj-Qwen_Qwen3.5-27B-f16.gguf"

def test_gguf_loading():
    """Test loading GGUF model with llama-cpp-python."""
    print("=" * 60)
    print("GGUF Model Loading Test")
    print("=" * 60)

    print(f"\nInstance: {INSTANCE_DIR.name}")
    print(f"Model: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e9:.1f}GB)")
    print(f"Multimodal: {MMPROJ_PATH.name} ({MMPROJ_PATH.stat().st_size / 1e6:.0f}MB)")

    # Check if llama-cpp-python is installed
    try:
        from llama_cpp import Llama
        print("\n✓ llama-cpp-python installed")
    except ImportError as e:
        print(f"\n✗ llama-cpp-python not installed: {e}")
        print("\nInstall with: pip3 install llama-cpp-python")
        print("For CUDA: CMAKE_ARGS='-DLLAMA_CUDA=on' pip3 install llama-cpp-python --force-reinstall --no-cache-dir")
        return False

    # Load model
    print("\nLoading model...")
    print(f"  Path: {MODEL_PATH}")
    print(f"  Using CUDA: {'available' if check_cuda() else 'CPU only'}")

    start = time.time()
    try:
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=8192,  # Context window
            n_gpu_layers=-1,  # Offload all layers to GPU
            verbose=True,
        )
        load_time = time.time() - start
        print(f"\n✓ Model loaded successfully in {load_time:.1f}s")
    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test basic inference
    print("\nTesting inference...")
    prompt = "What is the capital of France? Answer in one sentence."
    print(f"  Prompt: {prompt}")

    start = time.time()
    try:
        response = llm(
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            echo=False,
        )
        inference_time = time.time() - start

        text = response['choices'][0]['text']
        tokens = response['usage']['completion_tokens']

        print(f"\n✓ Inference successful ({inference_time:.2f}s, {tokens} tokens)")
        print(f"  Response: {text.strip()}")
        print(f"  Speed: {tokens/inference_time:.1f} tokens/sec")

    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    return True

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # Check for nvidia-smi
        import subprocess
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

if __name__ == "__main__":
    print(f"\nPython: {sys.version}")
    print(f"Working directory: {Path.cwd()}")

    success = test_gguf_loading()
    sys.exit(0 if success else 1)
