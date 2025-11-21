#!/usr/bin/env python3
"""
Merge LoRA Adapter and Test Performance - Session 7

Session 7 profiling identified the bottleneck:
- Introspective-Qwen: 24.31s forward pass (2.41× slower)
- Epistemic-Pragmatism: 10.10s forward pass (baseline)
- Root cause: LoRA adapter merged on-the-fly during inference

Solution: Merge LoRA weights into base model, save as optimized full model.

Expected results after merge:
- Forward pass time: ~10s (match Epistemic-Pragmatism)
- Initialization: ~4s (keep LoRA's fast loading)
- Quality: +55.6% (retain Introspective-Qwen quality)
- Memory: Similar to full model

This would give us best-of-all-worlds for edge deployment.
"""

import sys
import time
import torch
from pathlib import Path

# Add HRM root to path
sage_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sage_root))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def merge_lora_weights(base_model_name, adapter_path, output_path):
    """
    Merge LoRA adapter weights into base model and save.

    Args:
        base_model_name: HuggingFace model name or local path
        adapter_path: Path to LoRA adapter
        output_path: Where to save merged model
    """
    print(f"\n{'='*80}")
    print(f"LoRA Merge Process")
    print(f"{'='*80}")
    print(f"Base model:  {base_model_name}")
    print(f"Adapter:     {adapter_path}")
    print(f"Output:      {output_path}")
    print(f"{'='*80}\n")

    # Load base model
    print(f"[1/4] Loading base model...")
    start = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_time = time.time() - start
    print(f"      ✅ Base model loaded in {base_time:.2f}s")

    # Load tokenizer
    print(f"[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print(f"      ✅ Tokenizer loaded")

    # Load and merge adapter
    print(f"[3/4] Loading LoRA adapter and merging...")
    start = time.time()
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print(f"      ✅ Adapter loaded in {time.time()-start:.2f}s")

    print(f"      Merging adapter weights into base model...")
    start = time.time()
    model = model.merge_and_unload()
    merge_time = time.time() - start
    print(f"      ✅ Merged in {merge_time:.2f}s")

    # Save merged model
    print(f"[4/4] Saving merged model to {output_path}...")
    start = time.time()
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    save_time = time.time() - start
    print(f"      ✅ Saved in {save_time:.2f}s")

    print(f"\n{'='*80}")
    print(f"✅ LoRA merge complete!")
    print(f"{'='*80}")
    print(f"Total time: {base_time + merge_time + save_time:.2f}s")
    print(f"Merged model saved to: {output_path}")
    print(f"\nNext: Test merged model performance with profile_performance_regression.py")
    print(f"{'='*80}\n")


def test_merged_model_inference(merged_model_path, test_question=None):
    """
    Quick inference test to verify merged model works.
    """
    if test_question is None:
        test_question = "What are the key components of the SAGE consciousness framework?"

    print(f"\n{'='*80}")
    print(f"Quick Inference Test - Merged Model")
    print(f"{'='*80}")
    print(f"Model: {merged_model_path}")
    print(f"Question: \"{test_question}\"")
    print(f"{'='*80}\n")

    # Load merged model
    print(f"Loading merged model...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    model.eval()
    load_time = time.time() - start
    print(f"✅ Loaded in {load_time:.2f}s")

    # Run inference
    print(f"\nRunning inference...")
    prompt = f"Question: {test_question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    inference_time = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"✅ Inference completed in {inference_time:.2f}s")
    print(f"\n{'─'*80}")
    print(f"Response:")
    print(f"{'─'*80}")
    print(response)
    print(f"{'─'*80}")

    print(f"\n{'='*80}")
    print(f"Quick Test Summary")
    print(f"{'='*80}")
    print(f"Load time:      {load_time:.2f}s")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"\nExpected performance (based on Session 7 profiling):")
    print(f"  - Initialization: ~4s (LoRA was fast)")
    print(f"  - Inference: ~10s (should match Epistemic-Pragmatism)")
    print(f"\nIf inference is ~10s → ✅ MERGE SUCCESS (2.4× speedup achieved!)")
    print(f"If inference is ~24s → ❌ No improvement (investigate further)")
    print(f"{'='*80}\n")


def main():
    """Merge LoRA and test performance."""

    print("\n" + "="*80)
    print("SAGE Edge LoRA Merge - Session 7")
    print("="*80)
    print("\nObjective: Merge Introspective-Qwen LoRA to eliminate inference overhead")
    print("\nSession 7 profiling proved:")
    print("  - LoRA adapter causes 2.41× slowdown (24.31s vs 10.10s)")
    print("  - Bottleneck is forward pass (100% of regression)")
    print("  - Solution: Merge adapter into base model")
    print("\nThis script will:")
    print("  1. Load base model (Qwen/Qwen2.5-0.5B-Instruct)")
    print("  2. Load LoRA adapter (Introspective-Qwen)")
    print("  3. Merge adapter weights into base model")
    print("  4. Save as optimized full model")
    print("  5. Run quick inference test")
    print("="*80 + "\n")

    # Paths
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model"
    output_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"

    # Check if adapter exists
    if not Path(adapter_path).exists():
        print(f"❌ Error: Adapter not found at {adapter_path}")
        print(f"   Please check the path and try again.")
        return

    # Check if output already exists
    if Path(output_path).exists():
        print(f"⚠️  Warning: Output path already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Merge LoRA
    try:
        merge_lora_weights(base_model_name, adapter_path, output_path)
    except Exception as e:
        print(f"\n❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return

    # Quick test
    print("\n" + "="*80)
    response = input("Run quick inference test on merged model? (y/n): ")
    if response.lower() == 'y':
        try:
            test_merged_model_inference(output_path)
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ Session 7 LoRA merge complete!")
    print(f"\nMerged model saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Run full profiling comparison:")
    print(f"     python sage/tests/profile_performance_regression.py")
    print(f"     (Compare epistemic-pragmatism vs introspective-qwen-merged)")
    print(f"\n  2. Run Session 6 quality test on merged model:")
    print(f"     python sage/tests/test_model_comparison_edge.py")
    print(f"     (Verify +55.6% quality improvement retained)")
    print(f"\n  3. If performance matches Epistemic-Pragmatism (~10s):")
    print(f"     ✅ Production-ready: best quality + fast inference + 4s load!")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
