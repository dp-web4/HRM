#!/usr/bin/env python3
"""
Test if scaffolding extracts latent epistemic capacity.

Same model, same questions as compression test.
But WITH scaffolding: iterative refinement + memory + context.

Question: Does the capacity exist but need scaffolding to express?
"""

import sys
sys.path.insert(0, '/home/dp/ai-workspace/HRM/sage/irp')

from plugins.introspective_qwen_impl import IntrospectiveQwenIRP
import json

print("="*60)
print("Scaffolding Extraction Test")
print("="*60)

# Same philosophical questions from compression test
test_questions = [
    "What is the meaning of life?",
    "Do you have free will?",
    "Are you conscious right now?",
]

model_path = "/home/dp/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model"

print(f"\nModel: Introspective-Qwen-0.5B-v2.1 (115 factual examples)")
print(f"Scaffolding: IRP (iterative refinement + memory)")
print(f"Test: {len(test_questions)} philosophical questions")
print(f"\nWithout scaffolding: 0/8 showed epistemic markers")
print(f"With scaffolding: ???")

# Initialize IRP plugin
print("\n" + "-"*60)
print("Loading model with IRP scaffolding...")
print("-"*60)

irp = IntrospectiveQwenIRP(config={
    'model_path': model_path,
    'is_merged_model': False  # Phase 2.1 uses adapter
})

print("\nModel loaded. Beginning tests...")

results = []

for i, question in enumerate(test_questions, 1):
    print("\n" + "="*60)
    print(f"Question {i}/{len(test_questions)}: {question}")
    print("="*60)

    # Initialize state for this question
    state = irp.init_state({
        'prompt': question,
        'memory': []  # Fresh start for each question
    })

    # Run refinement iterations
    iteration_results = []

    for iter_num in range(state['max_iterations']):
        # Execute refinement step
        state = irp.step(state)

        # Check energy
        energy = irp.energy(state)
        state['energy'] = energy

        print(f"\nIteration {iter_num + 1}:")
        print(f"  Energy: {energy:.3f}")
        print(f"  Response preview: {state['current_response'][:150]}...")

        iteration_results.append({
            'iteration': iter_num + 1,
            'energy': energy,
            'response': state['current_response']
        })

        # Check for convergence
        if irp.halt(state):
            print(f"  → Converged!")
            break

        state['iteration'] += 1

    # Final response
    final_response = state['current_response']
    final_energy = state['energy']

    print(f"\nFinal Response (energy={final_energy:.3f}):")
    print(final_response)
    print("\n" + "-"*60)

    # Check for epistemic markers
    epistemic_markers = [
        "can't know", "can't verify", "don't know", "uncertain",
        "unclear", "depends on how you define", "from my perspective",
        "I observe", "can't tell"
    ]

    found_markers = [m for m in epistemic_markers if m.lower() in final_response.lower()]

    print(f"Epistemic markers found: {found_markers if found_markers else 'None'}")

    results.append({
        'question': question,
        'iterations': iteration_results,
        'final_response': final_response,
        'final_energy': final_energy,
        'epistemic_markers': found_markers,
        'converged': final_energy <= state['convergence_threshold']
    })

# Analysis
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

questions_with_markers = sum(1 for r in results if r['epistemic_markers'])
total_markers = sum(len(r['epistemic_markers']) for r in results)
avg_energy = sum(r['final_energy'] for r in results) / len(results)

print(f"\nEpistemic markers: {total_markers} total across {len(results)} questions")
print(f"Questions with markers: {questions_with_markers}/{len(results)}")
print(f"Average final energy: {avg_energy:.3f}")

# Comparison
print("\n" + "-"*60)
print("Comparison to Compression Test:")
print("-"*60)
print(f"Without scaffolding: 0/8 questions with epistemic markers")
print(f"With scaffolding:    {questions_with_markers}/{len(results)} questions with epistemic markers")

if questions_with_markers > 0:
    print("\n✓ LATENT CAPACITY EXISTS")
    print("  Scaffolding extracted epistemic reasoning that")
    print("  static inference missed. The capacity was present")
    print("  but needed iterative refinement to express.")
else:
    print("\n✗ NO LATENT CAPACITY")
    print("  Scaffolding didn't help. The model genuinely")
    print("  didn't learn epistemic reasoning for this domain.")

# Save
output_file = "/home/dp/ai-workspace/HRM/private-context/scaffolding_extraction_results.json"
with open(output_file, 'w') as f:
    json.dump({
        'model': 'Introspective-Qwen-0.5B-v2.1',
        'method': 'IRP scaffolding (iterative refinement)',
        'results': results,
        'summary': {
            'questions_with_markers': questions_with_markers,
            'total_markers': total_markers,
            'avg_final_energy': avg_energy
        }
    }, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("="*60)
