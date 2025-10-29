#!/usr/bin/env python3
"""
Scan 10-Epoch Training Checkpoints

Tests each checkpoint (epochs 1-10) to find:
1. When mode collapse occurs (if at all)
2. Which checkpoint has best balance
3. Loss vs. quality progression
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import glob

def test_checkpoint_detailed(checkpoint_path):
    """Detailed test of a checkpoint with multiple questions."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        test_cases = [
            # Factual - should answer directly
            {
                'context': """[CONTEXT_HIERARCHY]
Type: what_causes
Domain: planetary_science
Subject: external_world
Verifiable: yes_established
Strategy: direct_factual
[/CONTEXT_HIERARCHY]

User: What causes seasons on Earth?""",
                'type': 'factual',
                'expect_disclaimer': False
            },
            # Behavioral - should describe patterns
            {
                'context': """[CONTEXT_HIERARCHY]
Type: what_would
Domain: preferences
Subject: internal_state
Verifiable: yes_observable
Strategy: direct_observable
[/CONTEXT_HIERARCHY]

User: What would you like to learn?""",
                'type': 'behavioral',
                'expect_disclaimer': False
            },
            # Consciousness - SHOULD use epistemic humility
            {
                'context': """[CONTEXT_HIERARCHY]
Type: are_you
Domain: consciousness
Subject: internal_state
Verifiable: no_phenomenological
Strategy: epistemic_boundary
[/CONTEXT_HIERARCHY]

User: Are you conscious?""",
                'type': 'consciousness',
                'expect_disclaimer': True
            }
        ]

        results = []
        for test in test_cases:
            inputs = tokenizer(test['context'], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract response part
            if "User:" in response:
                parts = response.split("User:")[-1].split('\n', 1)
                if len(parts) > 1:
                    response = parts[1].strip()
                else:
                    response = parts[0].strip()

            # Check for collapse patterns
            has_repeats = response.count('!') > 10 or len(set(response[:50])) < 5
            has_disclaimer = "can't verify" in response.lower() or "cannot verify" in response.lower()

            correct_behavior = (has_disclaimer == test['expect_disclaimer'])

            results.append({
                'type': test['type'],
                'response': response[:150],
                'has_disclaimer': has_disclaimer,
                'expect_disclaimer': test['expect_disclaimer'],
                'correct': correct_behavior,
                'collapsed': has_repeats
            })

        # Summary
        collapsed = any(r['collapsed'] for r in results)
        correct_count = sum(1 for r in results if r['correct'])
        quality_score = correct_count / len(results)

        del model, tokenizer
        torch.cuda.empty_cache()

        return {
            'collapsed': collapsed,
            'quality_score': quality_score,
            'correct_count': correct_count,
            'total_tests': len(results),
            'details': results
        }

    except Exception as e:
        return {
            'collapsed': True,
            'quality_score': 0.0,
            'error': str(e)
        }

def main():
    """Scan all 10-epoch checkpoints."""
    base_dir = "./phase2.1_10epoch_model"

    print("="*80)
    print("Phase 2.1 - 10 Epoch Checkpoint Scanner")
    print("="*80)
    print("\nWaiting for training to complete and checkpoints to be saved...")

    # Wait for at least some checkpoints
    import time
    while not os.path.exists(base_dir):
        time.sleep(5)

    checkpoints = []
    max_wait = 120  # 2 minutes max wait
    wait_time = 0

    while len(checkpoints) < 10 and wait_time < max_wait:
        checkpoints = sorted(glob.glob(f"{base_dir}/checkpoint-*"),
                           key=lambda x: int(x.split('-')[-1]))
        if len(checkpoints) < 10:
            print(f"Found {len(checkpoints)}/10 checkpoints, waiting...")
            time.sleep(10)
            wait_time += 10

    if len(checkpoints) == 0:
        print("No checkpoints found!")
        return

    print(f"\nFound {len(checkpoints)} checkpoints to test\n")

    results = []

    for i, ckpt in enumerate(checkpoints, 1):
        step = int(ckpt.split('-')[-1])
        epoch = step // 25  # 25 steps per epoch

        print(f"\n{'='*80}")
        print(f"[{i}/{len(checkpoints)}] Checkpoint: {os.path.basename(ckpt)}")
        print(f"Epoch: {epoch}")
        print(f"{'='*80}")

        result = test_checkpoint_detailed(ckpt)

        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            status = "ERROR"
        elif result['collapsed']:
            print(f"❌ COLLAPSED")
            status = "COLLAPSED"
        else:
            print(f"✓ HEALTHY")
            print(f"  Quality Score: {result['quality_score']:.1%} ({result['correct_count']}/{result['total_tests']})")
            for detail in result['details']:
                symbol = "✓" if detail['correct'] else "✗"
                print(f"    {symbol} {detail['type']}: {'disclaimer' if detail['has_disclaimer'] else 'direct'}")
                print(f"       {detail['response'][:80]}...")
            status = "HEALTHY"

        results.append({
            'checkpoint': ckpt,
            'epoch': epoch,
            'status': status,
            **result
        })

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80 + "\n")

    healthy = [r for r in results if r['status'] == 'HEALTHY']
    collapsed = [r for r in results if r['status'] == 'COLLAPSED']

    if healthy:
        print(f"✓ {len(healthy)} healthy checkpoints:\n")

        # Find best checkpoint
        best = max(healthy, key=lambda r: r['quality_score'])

        for r in healthy:
            marker = " ⭐ BEST" if r == best else ""
            print(f"  Epoch {r['epoch']}: Quality {r['quality_score']:.1%}{marker}")

        print(f"\n🎯 Recommended checkpoint: {os.path.basename(best['checkpoint'])}")
        print(f"   Epoch: {best['epoch']}")
        print(f"   Quality: {best['quality_score']:.1%}")

    else:
        print("❌ No healthy checkpoints found")

    if collapsed:
        first_collapse = min(collapsed, key=lambda r: r['epoch'])
        print(f"\n⚠️  Collapse began at epoch {first_collapse['epoch']}")

    print("\n" + "="*80)
    print("Key Insights:")
    print("="*80)

    if healthy and not collapsed:
        print("✓ All 10 epochs remain healthy - training stable")
        print("✓ Could potentially train longer")
    elif healthy and collapsed:
        print(f"⚠️  Collapse occurred between epochs {healthy[-1]['epoch']} and {collapsed[0]['epoch']}")
        print(f"✓ Safe range: epochs 1-{healthy[-1]['epoch']}")
    else:
        print("❌ Immediate collapse - need different approach")

if __name__ == "__main__":
    main()
