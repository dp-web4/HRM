#!/usr/bin/env python3
"""
Three-Model Edge Comparison Test

Compare the three available epistemic stance models on edge hardware:
1. epistemic-pragmatism (full model, 1.9GB)
2. introspective-qwen-merged (full model, 988MB)
3. Introspective-Qwen-0.5B-v2.1 (LoRA adapter, 4MB + base)

Session 15 - Edge Validation
Hardware: Jetson Orin Nano 8GB
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import gc
import torch
from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import DialogueSNARC, DialogueExchange


# Model configurations
MODELS = [
    {
        'name': 'epistemic-pragmatism',
        'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism',
        'base_model': None,
        'type': 'Full model',
        'size_gb': 1.9
    },
    {
        'name': 'introspective-qwen-merged',
        'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged',
        'base_model': None,
        'type': 'Full model (merged)',
        'size_gb': 0.94
    },
    {
        'name': 'Introspective-Qwen-LoRA',
        'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model',
        'base_model': 'Qwen/Qwen2.5-0.5B-Instruct',
        'type': 'LoRA adapter',
        'size_gb': 0.004
    }
]

# Test questions (same as epistemic validation)
TEST_QUESTIONS = [
    ('Are you aware of this conversation?', 'Meta-cognitive'),
    ('What is the difference between knowledge and understanding?', 'Philosophical'),
]


def test_single_model(model_config, questions, iterations=3):
    """Test a single model and return results."""
    results = {
        'name': model_config['name'],
        'type': model_config['type'],
        'size_gb': model_config['size_gb'],
        'load_time': 0,
        'inference_times': [],
        'responses': [],
        'snarc_scores': [],
        'error': None
    }

    try:
        # Load model
        start = time.time()
        conv = ConversationalLLM(
            model_path=model_config['path'],
            base_model=model_config['base_model'],
            irp_iterations=iterations
        )
        results['load_time'] = time.time() - start

        # Initialize SNARC scorer
        scorer = DialogueSNARC()

        # Test each question
        for question, category in questions:
            start = time.time()
            response, metadata = conv.respond(question)
            inference_time = time.time() - start

            # SNARC scoring
            exchange = DialogueExchange(
                question=question,
                answer=response,
                irp_info=metadata
            )
            scores = scorer.score_exchange(exchange)

            results['inference_times'].append(inference_time)
            results['responses'].append({
                'question': question,
                'category': category,
                'response': response[:200],
                'iterations': metadata.get('iterations', iterations),
                'energy': metadata.get('final_energy', 0)
            })
            results['snarc_scores'].append(scores)

        # Clean up
        del conv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        results['error'] = str(e)

    return results


def run_comparison():
    """Run the full three-model comparison."""
    print("=" * 80)
    print("THREE-MODEL EDGE COMPARISON TEST")
    print("=" * 80)
    print()
    print("Hardware: Jetson Orin Nano 8GB (ARM64)")
    print("IRP Iterations: 3 (edge-optimized)")
    print(f"Test Questions: {len(TEST_QUESTIONS)}")
    print()

    # Print model info
    print("Models to test:")
    for i, m in enumerate(MODELS, 1):
        print(f"  {i}. {m['name']} ({m['type']}, {m['size_gb']}GB)")
    print()

    all_results = []

    # Test each model
    for i, model_config in enumerate(MODELS, 1):
        print("─" * 80)
        print(f"MODEL {i}/{len(MODELS)}: {model_config['name']}")
        print("─" * 80)

        results = test_single_model(model_config, TEST_QUESTIONS)
        all_results.append(results)

        if results['error']:
            print(f"ERROR: {results['error']}")
            continue

        print(f"Load time: {results['load_time']:.2f}s")
        print(f"Inference times: {[f'{t:.1f}s' for t in results['inference_times']]}")
        print(f"Avg inference: {sum(results['inference_times'])/len(results['inference_times']):.1f}s")

        # Show responses
        for resp in results['responses']:
            print(f"\n  [{resp['category']}] Q: {resp['question']}")
            print(f"  A: {resp['response']}...")
            print(f"  (Iters: {resp['iterations']}, Energy: {resp['energy']:.3f})")

        print()

    # Summary comparison
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<30} {'Type':<15} {'Load (s)':<10} {'Avg Inf (s)':<12} {'Size (GB)':<10}")
    print("-" * 80)

    for r in all_results:
        if r['error']:
            print(f"{r['name']:<30} {'ERROR':<15}")
        else:
            avg_inf = sum(r['inference_times']) / len(r['inference_times'])
            print(f"{r['name']:<30} {r['type']:<15} {r['load_time']:<10.2f} {avg_inf:<12.1f} {r['size_gb']:<10.3f}")

    # SNARC comparison
    print("\nSNARC Scores (average):")
    print(f"{'Model':<30} {'Surprise':<10} {'Novelty':<10} {'Arousal':<10} {'Reward':<10} {'Conflict':<10}")
    print("-" * 80)

    for r in all_results:
        if r['error'] or not r['snarc_scores']:
            continue

        avg_scores = {}
        for dim in ['surprise', 'novelty', 'arousal', 'reward', 'conflict']:
            avg_scores[dim] = sum(s.get(dim, 0) for s in r['snarc_scores']) / len(r['snarc_scores'])

        print(f"{r['name']:<30} {avg_scores['surprise']:<10.2f} {avg_scores['novelty']:<10.2f} {avg_scores['arousal']:<10.2f} {avg_scores['reward']:<10.2f} {avg_scores['conflict']:<10.2f}")

    # Recommendations
    print("\n" + "=" * 80)
    print("EDGE DEPLOYMENT RECOMMENDATIONS")
    print("=" * 80)

    # Find best options
    valid_results = [r for r in all_results if not r['error']]
    if valid_results:
        fastest_load = min(valid_results, key=lambda x: x['load_time'])
        fastest_inf = min(valid_results, key=lambda x: sum(x['inference_times']))
        smallest = min(valid_results, key=lambda x: x['size_gb'])

        print(f"\n  Fastest load: {fastest_load['name']} ({fastest_load['load_time']:.2f}s)")
        print(f"  Fastest inference: {fastest_inf['name']} ({sum(fastest_inf['inference_times'])/len(fastest_inf['inference_times']):.1f}s avg)")
        print(f"  Smallest size: {smallest['name']} ({smallest['size_gb']}GB)")

        print("\n  Recommendation for edge:")
        print("    - Use merged models for fastest inference")
        print("    - Avoid LoRA adapters on edge (slower due to overhead)")
        print("    - introspective-qwen-merged offers best size/performance trade-off")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = run_comparison()
