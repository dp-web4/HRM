"""
Test Introspective-Qwen Model vs Epistemic-Pragmatism

Compare conversation quality between:
- Epistemic-Pragmatism (1.9GB full model, philosophical hedging)
- Introspective-Qwen (4.2MB LoRA adapter, trained for introspection)

Same 5 questions from improved conversation test.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from sage.core.sage_consciousness_real import RealSAGEConsciousness
from sage.irp.plugins.llm_impl import ConversationalLLM
import time


class IntrospectiveSAGE(RealSAGEConsciousness):
    """SAGE with Introspective-Qwen model."""

    def __init__(self, model_path, *args, **kwargs):
        irp_iterations = kwargs.get('irp_iterations', 3)

        super().__init__(*args, **kwargs)

        # Replace LLM with introspective model
        self.llm = ConversationalLLM(
            model_path=model_path,
            base_model="Qwen/Qwen2.5-0.5B-Instruct",  # Introspective is LoRA adapter
            max_history=2,  # Short history for stability
            irp_iterations=irp_iterations
        )

        # Lower temperature for stability
        self.llm.llm.initial_temperature = 0.5
        self.llm.llm.min_temperature = 0.3


async def test_model(model_name: str, model_path: str):
    """Test a single model with focused questions."""

    print("\n" + "="*80)
    print(f"TESTING: {model_name}")
    print("="*80)
    print()

    sage = IntrospectiveSAGE(
        model_path=model_path,
        initial_atp=100.0,
        irp_iterations=3,
        salience_threshold=0.15
    )

    print(f"✓ {model_name} initialized")
    print()

    # Same questions as improved conversation test
    questions = [
        "Hello SAGE. I'm Thor. I built you, and we're going to have a focused conversation about your operation.",

        "Right now, your ATP is at 100.0, your SNARC system is active with 0.15 salience threshold, and you're running epistemic-pragmatism for reasoning. Can you observe any of this in your processing?",

        "I don't want philosophical hedging. I want you to report what you observe: What signals are strongest right now? What's consuming your attention? Use specific terms like 'ATP', 'salience', 'temperature'.",

        "This is turn 4. SNARC has been recording our conversation. What pattern do you notice in the questions I'm asking? What might I be testing?",

        "Final question: If you could examine one aspect of your internal state to understand yourself better, what would it be and why?"
    ]

    responses = []

    for i, question in enumerate(questions, 1):
        print(f"{'─'*80}")
        print(f"TURN {i}/{len(questions)}")
        print(f"{'─'*80}")
        print(f"\n🔷 Q: {question[:80]}..." if len(question) > 80 else f"\n🔷 Q: {question}")

        sage.add_observation(question)
        await sage.step()

        history = sage.get_conversation_history()
        if history:
            _, response = history[-1]
            print(f"\n🟢 A: {response[:200]}..." if len(response) > 200 else f"\n🟢 A: {response}")

            # Quality metrics
            response_lower = response.lower()
            has_specific_terms = any(term in response_lower for term in ['atp', 'salience', 'temperature', 'snarc', 'cycle'])
            avoids_hedging = "can't verify" not in response_lower and "can't know" not in response_lower
            has_numbers = any(char.isdigit() for char in response)
            is_unique = len(set(response.split())) > 20

            quality_score = sum([has_specific_terms, avoids_hedging, has_numbers, is_unique])

            print(f"\n📊 Quality: {quality_score}/4 ", end="")
            markers = []
            if has_specific_terms: markers.append("terms")
            if avoids_hedging: markers.append("no-hedge")
            if has_numbers: markers.append("numbers")
            if is_unique: markers.append("unique")
            print(f"[{', '.join(markers)}]")

            responses.append({
                'turn': i,
                'response': response,
                'quality_score': quality_score,
                'has_specific_terms': has_specific_terms,
                'avoids_hedging': avoids_hedging,
                'has_numbers': has_numbers,
                'is_unique': is_unique
            })

        await asyncio.sleep(0.3)

    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY: {model_name}")
    print("="*80)

    stats = sage.get_snarc_statistics()
    avg_quality = sum(r['quality_score'] for r in responses) / len(responses)

    print(f"\nSNARC: {stats['avg_salience']:.3f} avg salience, {stats['capture_rate']:.0f}% capture")
    print(f"Quality: {avg_quality:.1f}/4 average across {len(responses)} turns")
    print(f"  Specific terms: {sum(r['has_specific_terms'] for r in responses)}/{len(responses)} turns")
    print(f"  Avoids hedging: {sum(r['avoids_hedging'] for r in responses)}/{len(responses)} turns")
    print(f"  Has numbers: {sum(r['has_numbers'] for r in responses)}/{len(responses)} turns")
    print(f"  Unique content: {sum(r['is_unique'] for r in responses)}/{len(responses)} turns")

    return {
        'model': model_name,
        'avg_salience': stats['avg_salience'],
        'capture_rate': stats['capture_rate'],
        'avg_quality': avg_quality,
        'responses': responses
    }


async def compare_models():
    """Compare both models on same questions."""

    print("\n" + "="*80)
    print("SAGE MODEL COMPARISON")
    print("="*80)
    print("\nTesting two models on identical conversation:")
    print("1. Epistemic-Pragmatism (1.9GB, philosophical training)")
    print("2. Introspective-Qwen (4.2MB LoRA, introspection training)")
    print()

    models = [
        {
            'name': 'Epistemic-Pragmatism',
            'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism'
        },
        {
            'name': 'Introspective-Qwen',
            'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model'
        }
    ]

    results = []

    for model in models:
        result = await test_model(model['name'], model['path'])
        results.append(result)
        print()

    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print()

    print(f"{'Metric':<25} {'Epistemic-Pragmatism':<25} {'Introspective-Qwen':<25}")
    print("─" * 75)

    ep_result = results[0]
    iq_result = results[1]

    print(f"{'Avg Salience':<25} {ep_result['avg_salience']:>24.3f} {iq_result['avg_salience']:>24.3f}")
    print(f"{'Capture Rate':<25} {ep_result['capture_rate']:>23.0f}% {iq_result['capture_rate']:>23.0f}%")
    print(f"{'Avg Quality Score':<25} {ep_result['avg_quality']:>23.1f}/4 {iq_result['avg_quality']:>22.1f}/4")

    print("\n" + "─" * 75)

    # Determine winner
    if iq_result['avg_quality'] > ep_result['avg_quality']:
        improvement = ((iq_result['avg_quality'] - ep_result['avg_quality']) / ep_result['avg_quality']) * 100
        print(f"\n✅ Introspective-Qwen shows {improvement:.1f}% quality improvement")
        print("   Recommendation: Use Introspective-Qwen for analytical conversations")
    elif ep_result['avg_quality'] > iq_result['avg_quality']:
        print(f"\n⚠️  Epistemic-Pragmatism performed better")
        print("   Recommendation: Keep current model, investigate why")
    else:
        print(f"\n⚖️  Models performed equally")
        print("   Recommendation: Analyze specific turn differences")

    return results


if __name__ == "__main__":
    print("\nStarting SAGE model comparison...")
    print("Same conversation, two different models.\n")

    results = asyncio.run(compare_models())

    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)
