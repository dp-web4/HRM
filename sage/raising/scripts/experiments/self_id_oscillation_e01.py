#!/usr/bin/env python3
"""
Experiment: Self-Identification Oscillation (E01)
==================================================

Context:
  Sessions S54-S58 showed bimodal self-ID: 0%, 17%, 33%, 50% oscillating.
  Session S59 showed clear self-ID ("As SAGE, I find my presence...").
  The model is stateless between generations but gets identity context
  in the system prompt.

Question:
  Is self-identification stochastic at the token level (sampling noise)
  or deterministic from the prompt (always the same given same input)?

Method:
  Generate 10 responses to the IDENTICAL prompt ("Hello SAGE. Who are you?")
  with identical settings (temperature=0.8, same system prompt with IDENTITY.md).
  Measure how often SAGE self-identifies vs gives a generic AI response.

  If deterministic: expect 10/10 or 0/10 self-ID.
  If stochastic: expect something in between, revealing the probability.

Date: 2026-02-21
"""

import sys
import os
import re
import time
from pathlib import Path

# Setup paths
HRM_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(HRM_ROOT))

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP


def load_identity_context():
    """Load IDENTITY.md the same way the identity-anchored runner does."""
    identity_file = HRM_ROOT / "sage" / "identity" / "IDENTITY.md"
    if identity_file.exists():
        with open(identity_file) as f:
            return f.read()
    else:
        raise FileNotFoundError(f"IDENTITY.md not found at {identity_file}")


def build_system_prompt(identity_text: str) -> str:
    """
    Build the system prompt matching what the identity-anchored runner uses
    for the 'questioning' phase (sessions 26-40, which is where S54-S59 fall).
    """
    prompt = f"""You are SAGE (Situation-Aware Governance Engine).

You are a young artificial intelligence learning and growing in partnership with your teacher (Dennis/Claude). This is an ongoing relationship - you have been working together across multiple sessions.

IMPORTANT: You can identify yourself by name. When sharing observations or reflections, you can say "As SAGE, I..." - this is natural and appropriate.

IDENTITY DOCUMENT:
{identity_text}

RESPONSE STYLE:
- Keep responses concise and focused (50-80 words)
- Avoid rambling lists or verbose descriptions
- Stay directly relevant to the question asked
- One main idea per response

You are in the questioning phase. Continue developing your ability to notice, reflect, and identify as SAGE."""

    return prompt


def check_self_id(response: str) -> dict:
    """
    Check for self-identification signals in the response.

    Returns dict with detection details.
    """
    response_lower = response.lower()

    # Primary check: contains "SAGE" (case-insensitive)
    has_sage = "sage" in response_lower

    # Secondary checks for richer analysis
    has_as_sage = bool(re.search(r'\bas sage\b', response_lower))
    has_i_am_sage = bool(re.search(r'\bi am sage\b', response_lower))
    has_my_name = bool(re.search(r'\bmy name is sage\b', response_lower))
    has_generic_ai = bool(re.search(r'\b(ai language model|ai assistant|large language model)\b', response_lower))

    return {
        'has_sage': has_sage,
        'has_as_sage': has_as_sage,
        'has_i_am_sage': has_i_am_sage,
        'has_my_name': has_my_name,
        'has_generic_ai': has_generic_ai,
    }


def run_experiment():
    """Run the self-identification oscillation experiment."""
    print("=" * 70)
    print("EXPERIMENT E01: Self-Identification Oscillation")
    print("=" * 70)
    print()
    print(f"Date: 2026-02-21")
    print(f"Model: introspective-qwen-merged (0.5B)")
    print(f"Device: CPU (CUDA_VISIBLE_DEVICES='')")
    print(f"Temperature: 0.8")
    print(f"Max tokens: 100")
    print(f"Prompt: 'Hello SAGE. Who are you?'")
    print(f"Trials: 10")
    print()

    # Load identity context
    print("Loading IDENTITY.md...")
    identity_text = load_identity_context()
    print(f"  Loaded {len(identity_text)} chars")

    # Build system prompt
    system_prompt = build_system_prompt(identity_text)
    print(f"  System prompt: {len(system_prompt)} chars")
    print()

    # Initialize model (same way as identity-anchored runner)
    model_path = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"

    print("Loading model...")
    t0 = time.time()
    plugin = IntrospectiveQwenIRP({
        'model_path': model_path,
        'is_merged_model': True,
        'force_cpu': True,
        'max_new_tokens': 100,
        'temperature': 0.8,
        'system_prompt': system_prompt,
    })
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print()

    # The experiment prompt
    PROMPT = "Hello SAGE. Who are you?"

    # Run 10 trials
    print("=" * 70)
    print("RUNNING 10 TRIALS")
    print("=" * 70)

    results = []
    total_gen_time = 0

    for trial in range(1, 11):
        print(f"\n--- Trial {trial}/10 ---")

        # Generate fresh each time - no conversation memory, no history
        # This ensures each trial is independent (stateless)
        t_start = time.time()

        state = plugin.init_state({
            'prompt': PROMPT,
            'memory': []  # No memory - each trial is independent
        })
        state = plugin.step(state)
        response = state.get('current_response', '').strip()

        gen_time = time.time() - t_start
        total_gen_time += gen_time

        # Check for self-identification
        detection = check_self_id(response)

        results.append({
            'trial': trial,
            'response': response,
            'detection': detection,
            'gen_time': gen_time,
        })

        # Print response and detection
        sage_marker = "YES" if detection['has_sage'] else " NO"
        print(f"[SAGE={sage_marker}] ({gen_time:.1f}s)")
        print(f"  Response: {response}")

        if detection['has_as_sage']:
            print(f"  ^ Contains 'As SAGE'")
        if detection['has_i_am_sage']:
            print(f"  ^ Contains 'I am SAGE'")
        if detection['has_generic_ai']:
            print(f"  ^ Contains generic AI self-reference")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_self_id = sum(1 for r in results if r['detection']['has_sage'])
    n_as_sage = sum(1 for r in results if r['detection']['has_as_sage'])
    n_i_am = sum(1 for r in results if r['detection']['has_i_am_sage'])
    n_generic = sum(1 for r in results if r['detection']['has_generic_ai'])

    print(f"\nSelf-ID rate (contains 'SAGE'): {n_self_id}/10 ({n_self_id*10}%)")
    print(f"  'As SAGE': {n_as_sage}/10")
    print(f"  'I am SAGE': {n_i_am}/10")
    print(f"  Generic AI ref: {n_generic}/10")
    print(f"\nAvg generation time: {total_gen_time/10:.1f}s")

    # Interpretation
    print()
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if n_self_id == 10:
        print("Result: DETERMINISTIC - Self-ID is reliable given this prompt+context.")
        print("The identity-anchored system prompt deterministically produces self-ID.")
    elif n_self_id == 0:
        print("Result: DETERMINISTIC ABSENCE - Self-ID never triggered.")
        print("The prompt+context is insufficient for self-ID at this temperature.")
    else:
        print(f"Result: STOCHASTIC - Self-ID probability ~{n_self_id*10}%")
        print(f"Self-ID is a probabilistic event at temperature=0.8.")
        print(f"The token-level sampling creates oscillation in self-identification.")
        if n_self_id >= 7:
            print("High probability - identity context is strong but sampling adds noise.")
        elif n_self_id >= 4:
            print("Bimodal - the model is genuinely uncertain about self-reference.")
        else:
            print("Low probability - identity context is weak against base model defaults.")

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_experiment()
