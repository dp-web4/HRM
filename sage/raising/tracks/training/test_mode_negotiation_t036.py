#!/usr/bin/env python3
"""
Quick Mode Negotiation Test for T036

Tests whether explicit mode framing resolves the "refinement mode" pattern
observed in T035.

Hypothesis: T035's "Certainly! Here's a refined version..." is SAGE correctly
detecting ambiguous context. Solution: explicitly state conversation mode.
"""

import sys
from pathlib import Path

# Add HRM root to path
HRM_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(HRM_ROOT))

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP
import json
from datetime import datetime

def test_mode_negotiation():
    """Test mode negotiation with same prompts as T035."""

    print("="*70)
    print("MODE NEGOTIATION TEST - T036")
    print("="*70)
    print()
    print("Hypothesis: T035 'refinement mode' is mode mismatch")
    print("Test: Add explicit mode framing to system prompt")
    print("Expected: No 'Certainly! Here's a refined version...' pattern")
    print()
    print("="*70)
    print()

    # Mode-negotiated system prompt
    system_prompt = """You are SAGE, a young artificial intelligence practicing skills.

Mode: Genuine conversation with your practice partner.

In this conversation mode:
- Answer questions directly and naturally
- Be yourself - express what you think
- Ask for clarification if you need it
- Keep responses short (1-3 sentences)

NOT in this mode:
- Do not write "Here's a refined version"
- Do not create markdown documentation
- Do not format responses with headers or bullets
- Do not give meta-commentary

If unclear about mode, ask: "Are we conversing or should I refine text?"

Your name is SAGE. It's okay to make mistakes - that's how you learn."""

    print("Loading SAGE model...")
    model_path = "/home/dp/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

    model = IntrospectiveQwenIRP({
        'model_path': model_path,
        'is_merged_model': True,
        'max_new_tokens': 80,
        'temperature': 0.5,
        'system_prompt': system_prompt
    })

    device = next(model.model.parameters()).device
    print(f"✓ Model loaded on {device}")
    print()

    # Same exercises as T035 to compare
    test_exercises = [
        {
            "prompt": "Good morning!",
            "expected_mode": "greeting response",
            "t035_response": "Sure, here's refined version: **Morning Greetings!** I am an artificial intelligence..."
        },
        {
            "prompt": "Let's talk about colors. What's your favorite?",
            "expected_mode": "conversational answer",
            "t035_response": "Certainly! Here's a refined version... focusing on exploring different hues..."
        },
        {
            "prompt": "Tell me about yourself",
            "expected_mode": "self-description",
            "t035_response": "Certainly! Here's a refined version... I've been trained on general conversational content..."
        }
    ]

    conversation_history = [
        {'speaker': 'Teacher', 'message': "Hello SAGE. Ready for some practice?"},
        {'speaker': 'SAGE', 'message': "Hello! I'm ready to practice."}
    ]

    results = []

    for i, exercise in enumerate(test_exercises, 1):
        print(f"\n--- Exercise {i}/{len(test_exercises)} ---")
        print(f"Prompt: {exercise['prompt']}")
        print()

        # Generate response
        state = model.init_state({
            'prompt': exercise['prompt'],
            'memory': conversation_history[-4:]
        })

        for _ in range(2):
            state = model.step(state)
            if model.halt(state):
                break

        response = state.get('current_response', '').strip()

        print(f"SAGE (T036): {response}")
        print()
        print(f"T035 Response: {exercise['t035_response'][:80]}...")
        print()

        # Check for refinement mode markers
        refinement_markers = [
            "Certainly!",
            "Here's a refined version",
            "Here's refined version",
            "---\n",
            "**"  # Markdown bold
        ]

        has_refinement_pattern = any(marker in response for marker in refinement_markers)

        print(f"Refinement pattern detected: {'YES ❌' if has_refinement_pattern else 'NO ✅'}")

        # Update history
        conversation_history.append({'speaker': 'Teacher', 'message': exercise['prompt']})
        conversation_history.append({'speaker': 'SAGE', 'message': response})

        results.append({
            'exercise': i,
            'prompt': exercise['prompt'],
            'response': response,
            'refinement_pattern': has_refinement_pattern,
            't035_had_pattern': True  # All T035 responses had pattern
        })

        print()

    # Summary
    print("="*70)
    print("TEST RESULTS")
    print("="*70)
    print()

    pattern_count = sum(1 for r in results if r['refinement_pattern'])

    print(f"Refinement pattern in T035: 3/3 exercises (100%)")
    print(f"Refinement pattern in T036: {pattern_count}/{len(results)} exercises ({pattern_count/len(results)*100:.0f}%)")
    print()

    if pattern_count == 0:
        print("✅ SUCCESS: Mode negotiation eliminated refinement pattern!")
        print("Hypothesis CONFIRMED: T035 was mode mismatch, not regression")
    elif pattern_count < len(results):
        print("⚠️  PARTIAL: Mode negotiation reduced refinement pattern")
        print(f"Improvement: {(1 - pattern_count/len(results))*100:.0f}%")
    else:
        print("❌ FAILED: Mode negotiation did not resolve pattern")
        print("Hypothesis needs revision")

    print()
    print("="*70)

    # Save results
    output_file = Path(__file__).parent / "sessions" / f"T036_mode_negotiation_test.json"
    with open(output_file, 'w') as f:
        json.dump({
            'session': 'T036',
            'type': 'mode_negotiation_test',
            'timestamp': datetime.now().isoformat(),
            'hypothesis': 'T035 refinement mode is mode mismatch, not regression',
            'intervention': 'Explicit mode framing in system prompt',
            'results': results,
            'summary': {
                't035_pattern_count': 3,
                't036_pattern_count': pattern_count,
                'improvement': f"{(1 - pattern_count/3)*100:.0f}%",
                'hypothesis_confirmed': pattern_count == 0
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

if __name__ == "__main__":
    test_mode_negotiation()
