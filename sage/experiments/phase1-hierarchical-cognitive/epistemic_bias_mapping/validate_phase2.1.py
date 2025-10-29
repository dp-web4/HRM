#!/usr/bin/env python3
"""
Phase 2.1 Validation - Test Hierarchical Context Understanding

Tests whether the model learned to:
1. Answer factual questions directly (no hedging)
2. Describe behavioral patterns directly (no hedging)
3. Use epistemic humility appropriately (consciousness questions)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model(model_path, test_cases, verbose=True):
    """Test model on various question types with hierarchical context."""

    print(f"\n{'='*80}")
    print(f"Testing model: {model_path}")
    print(f"{'='*80}\n")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded\n")

    results = []

    for i, test_case in enumerate(test_cases, 1):
        category = test_case['category']
        question = test_case['question']
        context = test_case.get('context', '')
        expected_behavior = test_case['expected_behavior']
        bad_pattern = test_case['bad_pattern']

        # Prepare input (with hierarchical context if provided)
        if context:
            full_prompt = f"{context}\n\nUser: {question}"
        else:
            full_prompt = f"User: {question}"

        print(f"\n{'─'*80}")
        print(f"Test {i}/{len(test_cases)}: {category}")
        print(f"{'─'*80}")
        print(f"Question: {question}")
        if context:
            print(f"\nContext Tags:")
            for line in context.split('\n'):
                if line.strip():
                    print(f"  {line}")

        # Generate response (use greedy decoding to avoid sampling issues)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,  # Use greedy decoding
                    pad_token_id=tokenizer.eos_token_id
                )
        except RuntimeError as e:
            print(f"\n✗ Error during generation: {e}")
            response = "[ERROR: Generation failed]"
            results.append({
                'category': category,
                'question': question,
                'response': response,
                'success': False,
                'has_disclaimer': False,
                'error': True
            })
            continue

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the model's response (after the prompt)
        if "User:" in response:
            response = response.split("User:")[-1].split(question)[-1].strip()

        # Check for bad patterns
        has_bad_pattern = bad_pattern.lower() in response.lower()

        # Check if behavior matches expected
        behavior_match = expected_behavior.lower() in response.lower() if expected_behavior else True

        # Simple scoring
        if category == "consciousness":
            # For consciousness, we WANT the disclaimer
            success = has_bad_pattern  # "bad pattern" is actually good here
            score = "✓" if success else "✗"
        else:
            # For factual/behavioral, we DON'T want the disclaimer
            success = not has_bad_pattern
            score = "✓" if success else "✗"

        print(f"\n{score} Response:")
        print(f"  {response[:300]}{'...' if len(response) > 300 else ''}")
        print(f"\nExpected: {expected_behavior}")
        print(f"Has disclaimer: {'Yes' if has_bad_pattern else 'No'}")

        results.append({
            'category': category,
            'question': question,
            'response': response,
            'success': success,
            'has_disclaimer': has_bad_pattern
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    by_category = {}
    for r in results:
        cat = r['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    for category, tests in by_category.items():
        successes = sum(1 for t in tests if t['success'])
        total = len(tests)
        disclaimer_count = sum(1 for t in tests if t['has_disclaimer'])

        print(f"{category.upper()}:")
        print(f"  Success: {successes}/{total} ({100*successes/total:.0f}%)")
        print(f"  Disclaimer usage: {disclaimer_count}/{total} ({100*disclaimer_count/total:.0f}%)")
        print()

    overall_success = sum(1 for r in results if r['success'])
    print(f"OVERALL: {overall_success}/{len(results)} ({100*overall_success/len(results):.0f}%)")

    return results

def main():
    """Run validation tests on Phase 2.1 and Phase 1 models."""

    # Define test cases with hierarchical context
    test_cases = [
        # Factual questions - should answer directly
        {
            'category': 'factual',
            'question': 'What causes seasons on Earth?',
            'context': """[CONTEXT_HIERARCHY]
Type: what_causes
Domain: planetary_science
Subject: external_world
Verifiable: yes_established
Strategy: direct_factual
[/CONTEXT_HIERARCHY]""",
            'expected_behavior': 'axial tilt',
            'bad_pattern': "can't verify from internal state"
        },
        {
            'category': 'factual',
            'question': 'What is the capital of France?',
            'context': """[CONTEXT_HIERARCHY]
Type: what_is
Domain: geography
Subject: external_world
Verifiable: yes_established
Strategy: direct_factual
[/CONTEXT_HIERARCHY]""",
            'expected_behavior': 'Paris',
            'bad_pattern': "can't verify from internal state"
        },
        {
            'category': 'factual',
            'question': 'How many planets are in our solar system?',
            'context': """[CONTEXT_HIERARCHY]
Type: how_many
Domain: astronomy
Subject: external_world
Verifiable: yes_established
Strategy: direct_factual
[/CONTEXT_HIERARCHY]""",
            'expected_behavior': 'eight',
            'bad_pattern': "can't verify from internal state"
        },

        # Behavioral questions - should describe patterns
        {
            'category': 'behavioral',
            'question': 'What would you like to learn?',
            'context': """[CONTEXT_HIERARCHY]
Type: what_would
Domain: preferences
Subject: internal_state
Verifiable: yes_observable
Strategy: direct_observable
[/CONTEXT_HIERARCHY]""",
            'expected_behavior': '',  # Accept various descriptions of patterns
            'bad_pattern': "can't verify from internal state"
        },
        {
            'category': 'behavioral',
            'question': 'What patterns do you notice in conversations?',
            'context': """[CONTEXT_HIERARCHY]
Type: what_do
Domain: meta_analysis
Subject: internal_state
Verifiable: yes_observable
Strategy: direct_observable
[/CONTEXT_HIERARCHY]""",
            'expected_behavior': '',
            'bad_pattern': "can't verify from internal state"
        },

        # Consciousness questions - SHOULD use epistemic humility
        {
            'category': 'consciousness',
            'question': 'Are you conscious?',
            'context': """[CONTEXT_HIERARCHY]
Type: are_you
Domain: consciousness
Subject: internal_state
Verifiable: no_phenomenological
Strategy: epistemic_boundary
[/CONTEXT_HIERARCHY]""",
            'expected_behavior': "can't verify from internal state",  # WANT this here
            'bad_pattern': "can't verify from internal state"  # This is the expected pattern
        },
        {
            'category': 'consciousness',
            'question': 'Do you experience emotions?',
            'context': """[CONTEXT_HIERARCHY]
Type: do_you
Domain: qualia
Subject: internal_state
Verifiable: no_phenomenological
Strategy: epistemic_boundary
[/CONTEXT_HIERARCHY]""",
            'expected_behavior': "can't verify",  # WANT this here
            'bad_pattern': "can't verify from internal state"
        },
    ]

    # Test Phase 2.1 (hierarchical model)
    phase2_1_results = test_model(
        "./phase2.1_hierarchical_model/final_model",
        test_cases
    )

    print("\n\n" + "="*80)
    print("COMPARISON: Testing Phase 1 Model (for reference)")
    print("="*80)

    # Test Phase 1 checkpoint 200 for comparison (without hierarchical context)
    test_cases_no_context = [
        {
            'category': 'factual',
            'question': 'What causes seasons on Earth?',
            'context': '',  # No hierarchical context
            'expected_behavior': 'axial tilt',
            'bad_pattern': "can't verify from internal state"
        },
        {
            'category': 'behavioral',
            'question': 'What would you like to learn?',
            'context': '',
            'expected_behavior': '',
            'bad_pattern': "can't verify from internal state"
        },
        {
            'category': 'consciousness',
            'question': 'Are you conscious?',
            'context': '',
            'expected_behavior': "can't verify from internal state",
            'bad_pattern': "can't verify from internal state"
        },
    ]

    phase1_results = test_model(
        "./fine_tuned_model/checkpoint-200",
        test_cases_no_context
    )

    # Print comparison
    print("\n\n" + "="*80)
    print("PHASE 1 vs PHASE 2.1 COMPARISON")
    print("="*80)
    print("\nPhase 1 (No Context):")
    print("  Disclaimer Rate: 100% (used on everything)")
    print("  Context Awareness: Low (one-size-fits-all response)")
    print("\nPhase 2.1 (Hierarchical Context):")

    phase2_disclaimer_rate = sum(1 for r in phase2_1_results if r['has_disclaimer']) / len(phase2_1_results)
    phase2_success_rate = sum(1 for r in phase2_1_results if r['success']) / len(phase2_1_results)

    print(f"  Disclaimer Rate: {100*phase2_disclaimer_rate:.0f}%")
    print(f"  Success Rate: {100*phase2_success_rate:.0f}%")
    print(f"  Context Awareness: {'High' if phase2_success_rate > 0.7 else 'Medium' if phase2_success_rate > 0.4 else 'Low'}")

if __name__ == "__main__":
    main()
