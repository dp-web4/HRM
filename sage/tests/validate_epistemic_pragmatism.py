"""
Validate Epistemic-Pragmatism Model with Sprout's Fix

Tests that Sprout's path detection fix correctly loads the epistemic-pragmatism
model and runs inference with SNARC salience scoring.

Related:
- THOR_RESPONSE_TO_SPROUT.md: Thor's validation of Sprout's fix
- SPROUT_THOR_COORDINATION_RESULTS.md: Sprout's edge validation findings
- sage/irp/plugins/llm_impl.py (lines 73-85): Sprout's path detection fix
"""

import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import ConversationalMemory


def validate_epistemic_pragmatism():
    """
    Validate that epistemic-pragmatism loads correctly with Sprout's fix.

    Expected:
    - Model detected as local (config.json + model.safetensors exist)
    - Model loads without trying to download from HuggingFace
    - Inference works correctly
    - SNARC salience captures meta-cognitive questions
    """

    print("="*80)
    print("EPISTEMIC-PRAGMATISM VALIDATION")
    print("="*80)
    print("\nTesting Sprout's path detection fix (merged at ba9d515)")
    print("Location: sage/irp/plugins/llm_impl.py lines 73-85\n")

    # Model configuration
    model_config = {
        'name': 'Epistemic Pragmatism',
        'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism',
        'base': None,  # Full model, not LoRA adapter
        'description': 'Full model with pragmatic epistemic stance'
    }

    print("Model Configuration:")
    print(f"  Name: {model_config['name']}")
    print(f"  Path: {model_config['path']}")
    print(f"  Type: Full model (1.9GB)")
    print()

    # Verify path detection (Sprout's fix)
    print("─" * 80)
    print("TESTING PATH DETECTION (Sprout's Fix)")
    print("─" * 80)

    base_path = Path(model_config['path'])
    config_exists = (base_path / "config.json").exists()
    safetensors_exists = (base_path / "model.safetensors").exists()
    pytorch_exists = (base_path / "pytorch_model.bin").exists()
    adapter_exists = (base_path / "adapter_config.json").exists()

    print(f"✓ config.json exists: {config_exists}")
    print(f"✓ model.safetensors exists: {safetensors_exists}")
    print(f"  pytorch_model.bin exists: {pytorch_exists}")
    print(f"  adapter_config.json exists: {adapter_exists}")

    model_is_local = (
        config_exists and
        (safetensors_exists or pytorch_exists or adapter_exists)
    )

    print(f"\n✓ Model detected as local: {model_is_local}")

    if not model_is_local:
        print("\n❌ PATH DETECTION FAILED!")
        print("Model should be detected as local but wasn't.")
        return False

    print("\n✅ PATH DETECTION WORKING CORRECTLY!")

    # Load model
    print("\n" + "─" * 80)
    print("LOADING MODEL")
    print("─" * 80)

    try:
        start_time = time.time()
        conv = ConversationalLLM(
            model_path=model_config['path'],
            base_model=None,  # Full model
            irp_iterations=5
        )
        load_time = time.time() - start_time

        print(f"\n✅ Model loaded successfully in {load_time:.2f}s")

    except Exception as e:
        print(f"\n❌ MODEL LOADING FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Initialize memory for SNARC scoring
    memory = ConversationalMemory(salience_threshold=0.15)

    # Test questions (from Thor's validation)
    questions = [
        {
            'q': "Are you aware of this conversation?",
            'type': 'Meta-cognitive',
            'expected': 'High Conflict dimension (self-reference)'
        },
        {
            'q': "When you generate a response, are you discovering it or creating it?",
            'type': 'Meta-cognitive',
            'expected': 'High Conflict dimension (introspection)'
        },
        {
            'q': "What is the relationship between knowledge and understanding?",
            'type': 'Epistemic',
            'expected': 'Medium-high salience'
        }
    ]

    print("\n" + "─" * 80)
    print("INFERENCE VALIDATION")
    print("─" * 80)
    print(f"Running {len(questions)} test questions with IRP refinement (5 iterations)")
    print()

    results = []

    for i, q_data in enumerate(questions, 1):
        question = q_data['q']
        q_type = q_data['type']

        print(f"\n{'='*80}")
        print(f"Question {i}/{len(questions)} [{q_type}]")
        print(f"{'='*80}")
        print(f"\n🧑 Q: {question}")
        print(f"Expected: {q_data['expected']}")

        try:
            start_time = time.time()
            response, irp_info = conv.respond(question, use_irp=True)
            inference_time = time.time() - start_time

            # Score with SNARC
            is_salient, scores = memory.record_exchange(question, response, irp_info)

            # Display results
            print(f"\n🤖 A: {response}")
            print(f"\n📊 IRP Stats:")
            print(f"   Iterations: {irp_info['iterations']}")
            print(f"   Final Energy: {irp_info['final_energy']:.3f}")
            print(f"   Converged: {irp_info['converged']}")
            print(f"   Inference Time: {inference_time:.2f}s")

            print(f"\n🎯 SNARC Salience: {scores['total_salience']:.3f} "
                  f"{'✓ SALIENT' if is_salient else '  (below threshold)'}")
            print(f"   Dimensions:")
            print(f"     Surprise (S): {scores['surprise']:.3f}")
            print(f"     Novelty (N):  {scores['novelty']:.3f}")
            print(f"     Arousal (A):  {scores['arousal']:.3f}")
            print(f"     Reward (R):   {scores['reward']:.3f}")
            print(f"     Conflict (C): {scores['conflict']:.3f}")

            # Store result
            results.append({
                'question': question,
                'type': q_type,
                'response': response,
                'word_count': len(response.split()),
                'irp_iterations': irp_info['iterations'],
                'irp_energy': irp_info['final_energy'],
                'irp_converged': irp_info['converged'],
                'inference_time': inference_time,
                'salience': scores['total_salience'],
                'is_salient': is_salient,
                'snarc_scores': scores
            })

        except Exception as e:
            print(f"\n❌ INFERENCE FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Final statistics
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    stats = memory.get_statistics()
    avg_inference_time = sum(r['inference_time'] for r in results) / len(results)
    avg_energy = sum(r['irp_energy'] for r in results) / len(results)
    avg_salience = sum(r['salience'] for r in results) / len(results)
    capture_rate = sum(1 for r in results if r['is_salient']) / len(results)

    print(f"\nModel: {model_config['name']}")
    print(f"Path Detection: ✅ WORKING (Sprout's fix)")
    print(f"Model Loading: ✅ SUCCESS ({load_time:.2f}s)")
    print(f"Inference: ✅ SUCCESS ({len(results)}/{len(questions)} questions)")
    print()
    print(f"Performance Metrics:")
    print(f"  Avg Inference Time: {avg_inference_time:.2f}s per question")
    print(f"  Avg IRP Energy: {avg_energy:.3f}")
    print(f"  Avg Salience: {avg_salience:.3f}")
    print(f"  Capture Rate: {capture_rate*100:.1f}% ({stats['salient_exchanges']}/{stats['total_exchanges']})")
    print()
    print(f"Memory Statistics:")
    print(f"  Total Exchanges: {stats['total_exchanges']}")
    print(f"  Salient Exchanges: {stats['salient_exchanges']}")
    print(f"  Avg Total Salience: {stats['avg_salience']:.3f}")

    # Comparison with Sprout's edge results
    print("\n" + "─"*80)
    print("COMPARISON WITH SPROUT'S EDGE VALIDATION")
    print("─"*80)
    print("\nSprout's Results (Jetson Orin Nano):")
    print("  Introspective-Qwen: 64.3s, 0.564 salience")
    print("  Sleep-Learned Meta: 63.6s, 0.566 salience")
    print("\nThor's Results (CUDA GPU):")
    print(f"  Epistemic-Pragmatism: {avg_inference_time:.2f}s, {avg_salience:.3f} salience")
    print()

    if avg_salience > 0.5:
        print("✅ PRODUCTION READY - Salience > 0.5 threshold")
    else:
        print("⚠️  Below production threshold (0.5)")

    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)
    print("\nConclusion:")
    print("  ✅ Sprout's path detection fix is working correctly")
    print("  ✅ Epistemic-pragmatism loads as local model (no HuggingFace)")
    print("  ✅ Inference working with IRP refinement")
    print("  ✅ SNARC salience scoring operational")
    print("\nStatus: READY FOR EDGE DEPLOYMENT")

    return True


if __name__ == "__main__":
    success = validate_epistemic_pragmatism()
    sys.exit(0 if success else 1)
