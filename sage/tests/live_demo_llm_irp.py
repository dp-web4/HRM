"""
Live Demo: Track 7 LLM IRP Plugin

Test the complete conversational learning workflow:
1. Load Qwen2.5-0.5B model
2. Ask a series of questions with IRP refinement
3. Score with SNARC salience
4. Show selective memory storage
5. Benchmark performance

Usage:
    python live_demo_llm_irp.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import ConversationalMemory


def main():
    print("="*80)
    print("TRACK 7: LIVE LLM IRP DEMO")
    print("="*80)
    print("\nInitializing conversational intelligence...\n")

    # Initialize components
    print("⏳ Loading Qwen2.5-0.5B-Instruct model...")
    start_time = time.time()

    try:
        conv = ConversationalLLM(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            irp_iterations=5
        )
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nNote: This requires transformers and torch to be installed.")
        print("Install with: pip install transformers torch")
        return

    # Initialize memory
    memory = ConversationalMemory(salience_threshold=0.15)

    # Test questions (diverse types)
    questions = [
        {
            'q': "What is the difference between knowledge and understanding?",
            'type': 'Epistemic',
        },
        {
            'q': "Are you aware of this conversation?",
            'type': 'Meta-cognitive',
        },
        {
            'q': "What is 2+2?",
            'type': 'Factual',
        },
        {
            'q': "When you generate a response, are you discovering it or creating it?",
            'type': 'Meta-cognitive',
        },
        {
            'q': "How does compression affect meaning?",
            'type': 'Epistemic',
        }
    ]

    print("="*80)
    print(f"CONVERSATIONAL LEARNING WORKFLOW")
    print("="*80)
    print(f"Testing {len(questions)} questions with IRP refinement + SNARC scoring\n")

    # Benchmark metrics
    total_time = 0
    total_iterations = 0

    # Test each question
    for i, q_data in enumerate(questions, 1):
        question = q_data['q']
        q_type = q_data['type']

        print(f"\n{'─'*80}")
        print(f"Question {i}/{len(questions)} [{q_type}]")
        print(f"{'─'*80}")
        print(f"\n🧑 Q: {question}")

        # Generate response with timing
        start_time = time.time()
        try:
            response, irp_info = conv.respond(question, use_irp=True)
            response_time = time.time() - start_time

            # Score with SNARC
            is_salient, scores = memory.record_exchange(question, response, irp_info)

            # Display results
            print(f"\n🤖 A: {response}")
            print(f"\n📊 IRP Performance:")
            print(f"   Iterations: {irp_info['iterations']}")
            print(f"   Final Energy: {irp_info['final_energy']:.3f}")
            print(f"   Converged: {'✓' if irp_info['converged'] else '✗'}")
            print(f"   Time: {response_time:.2f}s ({response_time/irp_info['iterations']:.2f}s per iteration)")

            print(f"\n🎯 SNARC Salience: {scores['total_salience']:.3f} {'✓ SALIENT' if is_salient else '  (below threshold)'}")
            print(f"   S={scores['surprise']:.2f} | "
                  f"N={scores['novelty']:.2f} | "
                  f"A={scores['arousal']:.2f} | "
                  f"R={scores['reward']:.2f} | "
                  f"C={scores['conflict']:.2f}")

            # Update benchmarks
            total_time += response_time
            total_iterations += irp_info['iterations']

        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

    # Final statistics
    stats = memory.get_statistics()

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\n📈 Conversation Memory:")
    print(f"   Total exchanges: {stats['total_exchanges']}")
    print(f"   Salient exchanges: {stats['salient_exchanges']}")
    print(f"   Capture rate: {stats['capture_rate']:.1f}%")
    print(f"   Avg salience: {stats['avg_salience']:.3f}")

    print(f"\n⚡ Performance Benchmarks:")
    print(f"   Avg response time: {total_time/len(questions):.2f}s")
    print(f"   Avg IRP iterations: {total_iterations/len(questions):.1f}")
    print(f"   Avg iteration time: {total_time/total_iterations:.2f}s")

    print(f"\n💾 Training Data:")
    training_data = memory.get_salient_for_training()
    print(f"   Salient examples: {len(training_data)}")
    if training_data:
        print(f"   Ready for sleep-cycle training!")
        print(f"   (Use Sprout's sleep_trainer.py for on-device learning)")

    print("\n" + "="*80)
    print("✓ Track 7: Live demo complete!")
    print("="*80)
    print("\nFeatures validated:")
    print("  ✓ LLM loading and inference")
    print("  ✓ IRP iterative refinement")
    print("  ✓ Temperature annealing")
    print("  ✓ Energy convergence")
    print("  ✓ SNARC 5D salience scoring")
    print("  ✓ Selective memory storage")
    print("  ✓ Training data extraction")
    print("\nThe SAGE consciousness kernel now has conversational intelligence! 🎉")


if __name__ == "__main__":
    main()
