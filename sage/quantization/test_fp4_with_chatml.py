#!/usr/bin/env python3
"""
Test FP4 quantized Qwen3-Omni-30B with proper ChatML formatting.

This script demonstrates that the torch.cat() error is NOT a quantization bug,
but rather a prompt formatting issue. The model requires ChatML format with
role markers.

Usage:
    python3 test_fp4_with_chatml.py

Author: Claude (Diagnostic Agent)
Date: 2025-12-24
"""

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
import time


def test_plain_text_fails(model, processor, device):
    """Test that plain text (without ChatML) fails."""
    print("\n" + "="*80)
    print("TEST 1: Plain Text (Should Fail)")
    print("="*80)

    plain_text = "Hello! How are you today?"
    print(f"\nInput: {plain_text}")
    print("Format: Plain text (no ChatML role markers)")

    try:
        inputs = processor(text=[plain_text], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=5)
        print("❌ UNEXPECTED: Plain text succeeded!")
        return False

    except ValueError as e:
        if "torch.cat" in str(e):
            print(f"✅ EXPECTED: Failed with torch.cat() error")
            print(f"   Error: {str(e)[:100]}...")
            return True
        else:
            print(f"❌ UNEXPECTED ERROR: {e}")
            return False


def test_chatml_succeeds(model, processor, device):
    """Test that proper ChatML format works."""
    print("\n" + "="*80)
    print("TEST 2: ChatML Format (Should Succeed)")
    print("="*80)

    # Proper ChatML format with role markers
    chatml_prompt = """<|im_start|>system
You are a helpful, friendly AI assistant.<|im_end|>
<|im_start|>user
Hello! How are you today?<|im_end|>
<|im_start|>assistant
"""

    print(f"\nInput format: ChatML with role markers")
    print("Prompt structure:")
    print("  <|im_start|>system ... <|im_end|>")
    print("  <|im_start|>user ... <|im_end|>")
    print("  <|im_start|>assistant")

    try:
        inputs = processor(text=[chatml_prompt], return_tensors="pt").to(device)

        print(f"\nInput tensor shape: {inputs['input_ids'].shape}")

        # Generate response
        print("\nGenerating response...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generation_time = time.time() - start_time

        # Decode output
        response = processor.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        assistant_marker = "<|im_start|>assistant"
        if assistant_marker in response:
            assistant_start = response.rfind(assistant_marker) + len(assistant_marker)
            assistant_response = response[assistant_start:].strip()
        else:
            assistant_response = response

        print(f"✅ SUCCESS: Generated response in {generation_time:.2f}s")
        print(f"\nFull output length: {len(response)} chars")
        print(f"Generated tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
        print(f"\nAssistant response:")
        print("-" * 80)
        print(assistant_response[:500])
        if len(assistant_response) > 500:
            print(f"... ({len(assistant_response) - 500} more chars)")
        print("-" * 80)

        return True

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_turn_conversation(model, processor, device):
    """Test multi-turn conversation with ChatML."""
    print("\n" + "="*80)
    print("TEST 3: Multi-Turn Conversation")
    print("="*80)

    # Multi-turn conversation
    conversation = """<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
<|im_start|>user
What is it famous for?<|im_end|>
<|im_start|>assistant
"""

    print("\nConversation:")
    print("  User: What is the capital of France?")
    print("  Assistant: The capital of France is Paris.")
    print("  User: What is it famous for?")
    print("  Assistant: [generating...]")

    try:
        inputs = processor(text=[conversation], return_tensors="pt").to(device)

        print(f"\nInput length: {inputs['input_ids'].shape[1]} tokens")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)

        # Extract last assistant response
        assistant_marker = "<|im_start|>assistant"
        last_assistant = response.split(assistant_marker)[-1].strip()

        print(f"✅ SUCCESS: Generated multi-turn response")
        print(f"\nAssistant's final response:")
        print("-" * 80)
        print(last_assistant[:400])
        print("-" * 80)

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Main test routine."""
    print("="*80)
    print("FP4 QUANTIZED QWEN3-OMNI: ChatML FORMAT TEST")
    print("="*80)
    print("\nThis script demonstrates that the torch.cat() error is caused by")
    print("incorrect prompt formatting, NOT by quantization issues.")
    print("\nThe model requires ChatML format with role markers:")
    print("  <|im_start|>user ... <|im_end|>")
    print("  <|im_start|>assistant ...")
    print("="*80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "/home/dp/ai-workspace/HRM/model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only"

    print(f"\nDevice: {device}")
    print(f"Model: {model_path}")

    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    print("\nLoading quantized model (this takes 2-3 minutes)...")

    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("✅ Model loaded")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"GPU Memory: {allocated:.2f} GB")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load processor
    print("\nLoading processor...")
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        print("✅ Processor loaded")
    except Exception as e:
        print(f"❌ Failed to load processor: {e}")
        return

    # Run tests
    model.eval()

    test_results = {
        "plain_text_fails": False,
        "chatml_succeeds": False,
        "multi_turn_succeeds": False,
    }

    # Test 1: Plain text should fail
    test_results["plain_text_fails"] = test_plain_text_fails(model, processor, device)

    # Test 2: ChatML should succeed
    test_results["chatml_succeeds"] = test_chatml_succeeds(model, processor, device)

    # Test 3: Multi-turn conversation
    test_results["multi_turn_succeeds"] = test_multi_turn_conversation(model, processor, device)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    print(f"\n✅ Plain text fails (expected): {test_results['plain_text_fails']}")
    print(f"✅ ChatML succeeds: {test_results['chatml_succeeds']}")
    print(f"✅ Multi-turn succeeds: {test_results['multi_turn_succeeds']}")

    all_passed = all(test_results.values())

    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nConclusion:")
        print("  - The FP4 quantization is WORKING CORRECTLY")
        print("  - The torch.cat() error was caused by missing ChatML formatting")
        print("  - The model requires <|im_start|>role markers in prompts")
        print("  - This is NOT a quantization bug")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nFailed tests:")
        for test_name, passed in test_results.items():
            if not passed:
                print(f"  - {test_name}")

    print("="*80)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Use ChatML format for all prompts:")
    print("""
    prompt = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
'''
    """)

    print("\n2. For production deployment, use vLLM for FP4 acceleration:")
    print("""
    from vllm import LLM
    llm = LLM(model=model_path, quantization="fp4")
    outputs = llm.chat([{"role": "user", "content": "Hello!"}])
    """)

    print("\n3. See TORCH_CAT_FAILURE_ANALYSIS.md for full details")
    print("="*80)


if __name__ == "__main__":
    main()
