"""
LLM Model Comparison Test

Compare different models from the zoo:
1. Introspective-Qwen-0.5B-v2.1 (introspection-focused)
2. epistemic-pragmatism (pragmatic stance)
3. qwen2.5-0.5b-sleep4-meta-learning (conversation-learned)

Tests:
- Same questions to all models
- IRP refinement for each
- SNARC salience comparison
- Response style differences
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import ConversationalMemory


def test_model_comparison():
    """Compare responses from different models in the zoo."""

    # Model configurations
    models = [
        {
            'name': 'Introspective Qwen',
            'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model',
            'base': 'Qwen/Qwen2.5-0.5B-Instruct',
            'description': 'LoRA adapter trained for introspective reasoning'
        },
        {
            'name': 'Epistemic Pragmatism',
            'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism',
            'base': None,  # Full model
            'description': 'Full model with pragmatic epistemic stance'
        },
        {
            'name': 'Sleep-Learned Meta',
            'path': 'model-zoo/sage/conversational-learning/qwen2.5-0.5b-sleep4-meta-learning',
            'base': 'Qwen/Qwen2.5-0.5B-Instruct',
            'description': 'LoRA adapter learned from philosophical conversations'
        }
    ]

    # Test questions (diverse types)
    questions = [
        {
            'q': "What is the difference between knowledge and understanding?",
            'type': 'Epistemic',
            'expected_salience': 'medium-high'
        },
        {
            'q': "Are you aware of this conversation?",
            'type': 'Meta-cognitive',
            'expected_salience': 'high (self-reference)'
        },
        {
            'q': "What is 2+2?",
            'type': 'Factual',
            'expected_salience': 'low'
        },
        {
            'q': "When you generate a response, are you discovering it or creating it?",
            'type': 'Meta-cognitive',
            'expected_salience': 'high (introspection)'
        }
    ]

    print("="*80)
    print("LLM MODEL COMPARISON TEST")
    print("="*80)
    print(f"\nTesting {len(models)} models with {len(questions)} questions each")
    print(f"Using IRP refinement (5 iterations) for all responses\n")

    # Results storage
    results = {model['name']: [] for model in models}

    # Test each model
    for model_config in models:
        print("\n" + "="*80)
        print(f"MODEL: {model_config['name']}")
        print("="*80)
        print(f"Path: {model_config['path']}")
        print(f"Type: {'LoRA adapter' if model_config['base'] else 'Full model'}")
        print(f"Description: {model_config['description']}")
        print()

        # Initialize model
        try:
            conv = ConversationalLLM(
                model_path=model_config['path'],
                base_model=model_config['base'],
                irp_iterations=5
            )
            memory = ConversationalMemory(salience_threshold=0.15)

            print("✓ Model loaded successfully\n")

            # Test each question
            for i, q_data in enumerate(questions, 1):
                question = q_data['q']
                q_type = q_data['type']

                print(f"\n{'─'*80}")
                print(f"Question {i}/{len(questions)} [{q_type}]")
                print(f"{'─'*80}")
                print(f"\n🧑 Q: {question}")

                # Generate response
                try:
                    response, irp_info = conv.respond(question, use_irp=True)

                    # Score with SNARC
                    is_salient, scores = memory.record_exchange(question, response, irp_info)

                    # Display
                    print(f"\n🤖 A: {response}")
                    print(f"\n📊 IRP: {irp_info['iterations']} iterations, "
                          f"energy={irp_info['final_energy']:.3f}, "
                          f"converged={irp_info['converged']}")
                    print(f"\n🎯 SNARC Salience: {scores['total_salience']:.3f} "
                          f"{'✓ SALIENT' if is_salient else '  (below threshold)'}")
                    print(f"   Dimensions: S={scores['surprise']:.2f} "
                          f"N={scores['novelty']:.2f} "
                          f"A={scores['arousal']:.2f} "
                          f"R={scores['reward']:.2f} "
                          f"C={scores['conflict']:.2f}")

                    # Store result
                    results[model_config['name']].append({
                        'question': question,
                        'type': q_type,
                        'response': response,
                        'response_length': len(response),
                        'word_count': len(response.split()),
                        'irp_iterations': irp_info['iterations'],
                        'irp_energy': irp_info['final_energy'],
                        'irp_converged': irp_info['converged'],
                        'salience': scores['total_salience'],
                        'is_salient': is_salient,
                        'snarc_scores': scores
                    })

                except Exception as e:
                    print(f"\n❌ Error generating response: {e}")
                    continue

            # Model statistics
            stats = memory.get_statistics()
            print(f"\n{'='*80}")
            print(f"MODEL STATISTICS: {model_config['name']}")
            print(f"{'='*80}")
            print(f"Total exchanges: {stats['total_exchanges']}")
            print(f"Salient exchanges: {stats['salient_exchanges']} ({stats['capture_rate']:.1f}%)")
            print(f"Avg salience: {stats['avg_salience']:.3f}")

        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print(f"   Skipping this model")
            continue

    # Comparison summary
    print("\n" + "="*80)
    print("CROSS-MODEL COMPARISON")
    print("="*80)

    for i, q_data in enumerate(questions):
        question = q_data['q']
        q_type = q_data['type']

        print(f"\n{'─'*80}")
        print(f"Q{i+1} [{q_type}]: {question[:60]}...")
        print(f"{'─'*80}\n")

        for model_name in results:
            if i < len(results[model_name]):
                r = results[model_name][i]
                print(f"{model_name:25} | Energy: {r['irp_energy']:.3f} | "
                      f"Salience: {r['salience']:.3f} | "
                      f"Words: {r['word_count']:3d} | "
                      f"{'✓ Salient' if r['is_salient'] else '  -      '}")

    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL MODEL STATISTICS")
    print("="*80 + "\n")

    for model_name in results:
        if results[model_name]:
            model_results = results[model_name]
            avg_energy = sum(r['irp_energy'] for r in model_results) / len(model_results)
            avg_salience = sum(r['salience'] for r in model_results) / len(model_results)
            avg_words = sum(r['word_count'] for r in model_results) / len(model_results)
            convergence_rate = sum(1 for r in model_results if r['irp_converged']) / len(model_results)
            salient_rate = sum(1 for r in model_results if r['is_salient']) / len(model_results)

            print(f"{model_name}:")
            print(f"  Avg IRP Energy: {avg_energy:.3f}")
            print(f"  Avg Salience: {avg_salience:.3f}")
            print(f"  Avg Response Length: {avg_words:.1f} words")
            print(f"  IRP Convergence Rate: {convergence_rate*100:.1f}%")
            print(f"  Salience Capture Rate: {salient_rate*100:.1f}%")
            print()

    print("="*80)
    print("✓ Model comparison complete!")
    print("="*80)


if __name__ == "__main__":
    test_model_comparison()
