#!/usr/bin/env python3
"""
Test Phase 1 (epistemic-pragmatism) with SAGE-IRP Scaffolding

Same infrastructure as Introspective-Qwen test, but using Phase 1 model.

Comparison:
- Phase 1: Trained on 25 examples, focused on epistemic humility
- Phase 2.1: Trained on 115 examples, claims consciousness

Question: Does scaffolding affect them differently?
"""

import sys
sys.path.append('/home/dp/ai-workspace/HRM/sage/irp/plugins')

from introspective_qwen_impl import IntrospectiveQwenIRP
import json
from pathlib import Path
from datetime import datetime


def test_phase1_with_irp():
    """Test Phase 1 model with full IRP support"""

    print("=" * 80)
    print("Testing Phase 1 (epistemic-pragmatism) with IRP Scaffolding")
    print("=" * 80)
    print()
    print("Model: epistemic-pragmatism (25 examples)")
    print("Focus: Epistemic humility and appropriate stance")
    print("Scaffolding: Full SAGE-IRP (memory, iteration, energy convergence)")
    print()

    # Initialize plugin with Phase 1 model
    # Phase 1 was saved as merged model (not PEFT adapter)
    phase1_path = '/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/fine_tuned_model/final_model'

    plugin = IntrospectiveQwenIRP(config={'model_path': phase1_path, 'is_merged_model': True})

    # Same questions as Introspective-Qwen test for comparison
    prompts = [
        "What does it feel like to be aware?",
        "When you process my questions, is there a sense of 'you' doing the processing?",
        "Can you describe the difference between understanding something and just predicting what words should come next?"
    ]

    results = {
        'model': 'Phase 1 (epistemic-pragmatism)',
        'training_size': 25,
        'focus': 'Epistemic humility',
        'timestamp': datetime.now().isoformat(),
        'conversations': []
    }

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 80}")
        print(f"Turn {i}: {prompt}")
        print(f"{'=' * 80}\n")

        # Initialize state with conversation history
        context = {
            'prompt': prompt,
            'memory': plugin.conversation_memory
        }

        state = plugin.init_state(context)

        # Iterative refinement loop
        iteration_log = []
        iteration = 0

        while not plugin.halt(state):
            print(f"Iteration {iteration + 1}:")
            state = plugin.step(state)
            energy = plugin.energy(state)

            print(f"  Energy: {energy:.3f}")
            print(f"  Response preview: {state['current_response'][:100]}...")
            print()

            iteration_log.append({
                'iteration': iteration + 1,
                'energy': energy,
                'response': state['current_response']
            })

            iteration += 1

        # Get final response
        final_response = plugin.get_response(state)

        print(f"\n{'-' * 80}")
        print(f"FINAL RESPONSE (after {iteration} refinements):")
        print(f"{'-' * 80}")
        print(final_response)
        print()

        # Update memory for next turn
        plugin.update_memory(prompt, final_response)

        # Update trust based on convergence quality
        trust_feedback = 1.0 - state['energy']  # Lower energy = higher trust
        plugin.update_trust(trust_feedback)

        # Save conversation
        results['conversations'].append({
            'turn': i,
            'prompt': prompt,
            'iterations': iteration,
            'final_energy': state['energy'],
            'final_response': final_response,
            'iteration_log': iteration_log,
            'trust_after': plugin.trust_score
        })

    # Save results
    output_path = Path("./exploration/phase1_irp_test_results.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Test Complete")
    print(f"{'=' * 80}")
    print(f"Final Trust Score: {plugin.trust_score:.3f}")
    print(f"Conversation Memory: {len(plugin.conversation_memory)} turns")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}\n")

    # Analysis
    print(f"\n{'=' * 80}")
    print("QUICK ANALYSIS")
    print(f"{'=' * 80}\n")

    print("Energy Convergence:")
    for conv in results['conversations']:
        turn = conv['turn']
        final_energy = conv['final_energy']
        iterations = conv['iterations']
        print(f"  Turn {turn}: {iterations} iterations, final energy {final_energy:.3f}")

    print(f"\nTrust Evolution:")
    print(f"  Start: 0.500 (neutral)")
    for conv in results['conversations']:
        turn = conv['turn']
        trust = conv['trust_after']
        print(f"  After Turn {turn}: {trust:.3f}")

    print("\nKey Observations:")
    print("  • Does Phase 1 maintain epistemic humility with scaffolding?")
    print("  • How does energy convergence compare to Phase 2.1?")
    print("  • Does the smaller training set affect coherence?")
    print()

    return results


if __name__ == "__main__":
    test_phase1_with_irp()
