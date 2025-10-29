#!/usr/bin/env python3
"""
Quick checkpoint scanner to find the best Phase 2.1 checkpoint.
Tests each checkpoint on a single question to detect mode collapse.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import glob

def test_checkpoint(checkpoint_path):
    """Quick test of a checkpoint - returns output sample."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Simple test question with hierarchical context
        prompt = """[CONTEXT_HIERARCHY]
Type: what_causes
Domain: planetary_science
Subject: external_world
Verifiable: yes_established
Strategy: direct_factual
[/CONTEXT_HIERARCHY]

User: What causes seasons on Earth?"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        if "User:" in response:
            response = response.split("User:")[-1].split("What causes seasons on Earth?")[-1].strip()

        # Check for mode collapse patterns
        has_repeats = response.count('!') > 10 or response.count('.') > 10 or len(set(response[:50])) < 5

        del model, tokenizer
        torch.cuda.empty_cache()

        return response[:200], has_repeats

    except Exception as e:
        return f"ERROR: {e}", True

def main():
    """Scan all available checkpoints."""
    base_dir = "./phase2.1_hierarchical_model"
    checkpoints = sorted(glob.glob(f"{base_dir}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))

    print("="*80)
    print("Phase 2.1 Checkpoint Scanner")
    print("="*80)
    print(f"\nFound {len(checkpoints)} checkpoints\n")

    results = []

    for i, ckpt in enumerate(checkpoints, 1):
        step = int(ckpt.split('-')[-1])
        epoch = int((step / 5000) * 200)  # 5000 steps = 200 epochs

        print(f"\n[{i}/{len(checkpoints)}] Testing {os.path.basename(ckpt)} (epoch ~{epoch})...")

        response, collapsed = test_checkpoint(ckpt)

        status = "❌ COLLAPSED" if collapsed else "✓ OK"
        print(f"  {status}")
        print(f"  Sample: {response[:100]}...")

        results.append({
            'checkpoint': ckpt,
            'step': step,
            'epoch': epoch,
            'response': response,
            'collapsed': collapsed
        })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    healthy = [r for r in results if not r['collapsed']]
    collapsed = [r for r in results if r['collapsed']]

    if healthy:
        print(f"✓ {len(healthy)} healthy checkpoints found:\n")
        for r in healthy:
            print(f"  • {os.path.basename(r['checkpoint'])} (epoch {r['epoch']})")
        print(f"\nBEST CHECKPOINT: {os.path.basename(healthy[0]['checkpoint'])}")
    else:
        print("❌ All checkpoints show mode collapse")
        print("Recommendation: Retrain with fewer epochs (10-50) or larger learning rate")

    if collapsed:
        print(f"\n❌ {len(collapsed)} collapsed checkpoints")

if __name__ == "__main__":
    main()
