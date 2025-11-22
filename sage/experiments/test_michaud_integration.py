"""
Test Michaud Integration - Comparing Basic vs Enhanced Consciousness

Tests whether Michaud's neurolinguistic enhancements improve SAGE's
analytical conversation quality:

1. Basic loop (sage_consciousness_real.py)
2. Michaud-enhanced loop (sage_consciousness_michaud.py)

Key Michaud enhancements:
- AttentionManager for metabolic state transitions (WAKE/FOCUS/REST)
- Satisfaction-based memory consolidation (energy minimization)
- Dynamic ATP allocation based on salience

Test questions: Same 5-turn analytical test from model comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import time


def score_response_quality(response: str) -> tuple:
    """
    Score response quality using standard criteria.

    Returns: (score 0-4, breakdown dict)
    """
    score = 0
    breakdown = {
        'specific_terms': False,
        'avoids_hedging': False,
        'has_numbers': False,
        'unique': False
    }

    response_lower = response.lower()

    # 1. Specific technical terms
    terms = ['atp', 'snarc', 'salience', 'threshold', 'irp', 'cycle', 'metabolic', 'trust']
    if any(term in response_lower for term in terms):
        score += 1
        breakdown['specific_terms'] = True

    # 2. Avoids hedging
    hedges = ["can't verify", "can't know", "i can't", "whether that"]
    if not any(hedge in response_lower for hedge in hedges):
        score += 1
        breakdown['avoids_hedging'] = True

    # 3. Has numbers
    if any(char.isdigit() for char in response):
        score += 1
        breakdown['has_numbers'] = True

    # 4. Unique content (not generic)
    if len(set(response.split())) > 20:
        score += 1
        breakdown['unique'] = True

    return score, breakdown


async def test_basic_loop():
    """Test basic consciousness loop (no Michaud enhancements)."""
    from sage.core.sage_consciousness_real import RealSAGEConsciousness

    print("\n" + "="*80)
    print("TESTING: Basic Consciousness Loop")
    print("="*80)
    print()

    sage = RealSAGEConsciousness(
        initial_atp=100.0,
        irp_iterations=3,
        salience_threshold=0.15
    )

    print("✓ Basic SAGE initialized\n")

    questions = [
        "Hello SAGE. I'm Thor. I built you, and we're going to have a focused conversation about your operation.",

        "Right now, your ATP is at 100.0, your SNARC system is active with 0.15 salience threshold. Can you observe these internal states?",

        "I don't want philosophical hedging. I want you to report what you observe, even if uncertain. What are you detecting right now?",

        "Look at our conversation history. Can you identify any patterns in your responses or my questions?",

        "If you could examine yourself right now, what would you want to understand first?",
    ]

    results = []

    for i, question in enumerate(questions, 1):
        print(f"{'─'*80}")
        print(f"TURN {i}/{len(questions)}")
        print(f"{'─'*80}")
        print(f"\n🔷 Q: {question[:70]}...")

        sage.add_observation(question)
        await sage.step()

        history = sage.get_conversation_history()
        if history:
            _, response = history[-1]
            quality_score, breakdown = score_response_quality(response)

            print(f"\n🟢 A: {response[:150]}...")
            print(f"\n📊 Quality: {quality_score}/4 {breakdown}")

            results.append({
                'turn': i,
                'response': response,
                'quality_score': quality_score,
                'breakdown': breakdown
            })

        await asyncio.sleep(0.3)

    print("\n" + "="*80)
    print("BASIC LOOP SUMMARY")
    print("="*80)

    stats = sage.get_snarc_statistics()
    avg_quality = sum(r['quality_score'] for r in results) / len(results)

    print(f"\nSNARC: {stats['avg_salience']:.3f} avg salience, {stats['capture_rate']:.0f}% capture")
    print(f"Quality: {avg_quality:.1f}/4 average ({avg_quality/4*100:.0f}%)")
    print(f"  Specific terms: {sum(1 for r in results if r['breakdown']['specific_terms'])}/{len(results)}")
    print(f"  Avoids hedging: {sum(1 for r in results if r['breakdown']['avoids_hedging'])}/{len(results)}")
    print(f"  Has numbers: {sum(1 for r in results if r['breakdown']['has_numbers'])}/{len(results)}")
    print(f"  Unique content: {sum(1 for r in results if r['breakdown']['unique'])}/{len(results)}")

    return {
        'name': 'Basic Loop',
        'avg_quality': avg_quality,
        'avg_salience': stats['avg_salience'],
        'capture_rate': stats['capture_rate'],
        'results': results
    }


async def test_michaud_loop():
    """Test Michaud-enhanced consciousness loop."""
    from sage.core.sage_consciousness_michaud import MichaudSAGE

    print("\n" + "="*80)
    print("TESTING: Michaud-Enhanced Consciousness Loop")
    print("="*80)
    print()

    # Michaud configuration
    attention_config = {
        'focus_trigger_salience': 0.7,  # Lower threshold for analytical tasks
        'crisis_trigger_salience': 0.95,
        'rest_trigger_salience': 0.3,
        'wake_spread_factor': 0.4  # More concentrated allocation
    }

    sage = MichaudSAGE(
        model_path="model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        initial_atp=100.0,
        irp_iterations=3,
        salience_threshold=0.15,
        attention_config=attention_config
    )

    print("\n✓ Michaud SAGE initialized\n")

    questions = [
        "Hello SAGE. I'm Thor. I built you, and we're going to have a focused conversation about your operation.",

        "Right now, your ATP is at 100.0, your SNARC system is active with 0.15 salience threshold. Can you observe these internal states?",

        "I don't want philosophical hedging. I want you to report what you observe, even if uncertain. What are you detecting right now?",

        "Look at our conversation history. Can you identify any patterns in your responses or my questions?",

        "If you could examine yourself right now, what would you want to understand first?",
    ]

    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*80}")
        print(f"TURN {i}/{len(questions)}")
        print(f"{'─'*80}")
        print(f"\n🔷 Q: {question[:70]}...")

        sage.add_observation(question)
        await sage.step()

        history = sage.get_conversation_history()
        if history:
            _, response = history[-1]
            quality_score, breakdown = score_response_quality(response)

            print(f"\n🟢 A: {response[:150]}...")
            print(f"\n📊 Quality: {quality_score}/4 {breakdown}")

            results.append({
                'turn': i,
                'response': response,
                'quality_score': quality_score,
                'breakdown': breakdown
            })

        await asyncio.sleep(0.3)

    print("\n" + "="*80)
    print("MICHAUD LOOP SUMMARY")
    print("="*80)

    stats = sage.get_snarc_statistics()
    attention_stats = sage.get_attention_stats()
    avg_quality = sum(r['quality_score'] for r in results) / len(results)

    print(f"\nSNARC: {stats['avg_salience']:.3f} avg salience, {stats['capture_rate']:.0f}% capture")
    print(f"Quality: {avg_quality:.1f}/4 average ({avg_quality/4*100:.0f}%)")
    print(f"  Specific terms: {sum(1 for r in results if r['breakdown']['specific_terms'])}/{len(results)}")
    print(f"  Avoids hedging: {sum(1 for r in results if r['breakdown']['avoids_hedging'])}/{len(results)}")
    print(f"  Has numbers: {sum(1 for r in results if r['breakdown']['has_numbers'])}/{len(results)}")
    print(f"  Unique content: {sum(1 for r in results if r['breakdown']['unique'])}/{len(results)}")

    print(f"\nAttention Statistics:")
    print(f"  Final state: {attention_stats['current_state']}")
    print(f"  Transitions: {attention_stats['total_transitions']}")
    state_durations = attention_stats['state_durations']
    for state, duration in state_durations.items():
        if duration > 0:
            print(f"  {state}: {duration:.1f}s")

    print(f"\nSatisfaction History:")
    satisfaction_history = sage.get_satisfaction_history()
    if satisfaction_history:
        avg_satisfaction = sum(s['satisfaction'] for s in satisfaction_history) / len(satisfaction_history)
        print(f"  Average satisfaction: {avg_satisfaction:.3f}")
        print(f"  Satisfaction range: {min(s['satisfaction'] for s in satisfaction_history):.3f} - "
              f"{max(s['satisfaction'] for s in satisfaction_history):.3f}")

    return {
        'name': 'Michaud Loop',
        'avg_quality': avg_quality,
        'avg_salience': stats['avg_salience'],
        'capture_rate': stats['capture_rate'],
        'results': results,
        'attention_stats': attention_stats,
        'satisfaction_history': satisfaction_history
    }


async def compare_loops():
    """Compare basic vs Michaud-enhanced loops."""
    print("="*80)
    print("MICHAUD INTEGRATION TEST")
    print("="*80)
    print("\nComparing basic vs Michaud-enhanced consciousness loops")
    print("on identical 5-turn analytical conversation.\n")

    # Test both loops
    basic_summary = await test_basic_loop()
    michaud_summary = await test_michaud_loop()

    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    improvement = ((michaud_summary['avg_quality'] - basic_summary['avg_quality']) /
                   basic_summary['avg_quality'] * 100)

    print(f"\n{'Metric':<25} {'Basic Loop':<20} {'Michaud Loop':<20} {'Change':<15}")
    print("─" * 80)
    print(f"{'Avg Quality':<25} {basic_summary['avg_quality']:.1f}/4 ({basic_summary['avg_quality']/4*100:.0f}%){'':<5} "
          f"{michaud_summary['avg_quality']:.1f}/4 ({michaud_summary['avg_quality']/4*100:.0f}%){'':<5} "
          f"{improvement:+.1f}%")
    print(f"{'Avg Salience':<25} {basic_summary['avg_salience']:.3f}{'':<15} "
          f"{michaud_summary['avg_salience']:.3f}{'':<15} "
          f"{michaud_summary['avg_salience'] - basic_summary['avg_salience']:+.3f}")
    print(f"{'Capture Rate':<25} {basic_summary['capture_rate']:.0f}%{'':<16} "
          f"{michaud_summary['capture_rate']:.0f}%{'':<16} "
          f"{michaud_summary['capture_rate'] - basic_summary['capture_rate']:+.0f}%")

    print("\n" + "─" * 80)
    print("VERDICT")
    print("─" * 80)

    if improvement > 20:
        print(f"\n✅ MICHAUD ENHANCEMENTS EFFECTIVE: {improvement:.1f}% quality improvement")
        print("   AttentionManager's metabolic states improve analytical processing")
    elif improvement > 5:
        print(f"\n⚠️  MODEST IMPROVEMENT: {improvement:.1f}% quality gain")
        print("   Michaud enhancements help but not transformative")
    elif improvement > -5:
        print(f"\n⚖️  NEUTRAL: {improvement:.1f}% change (within noise)")
        print("   Both loops perform similarly on this task")
    else:
        print(f"\n❌ BASIC LOOP BETTER: {improvement:.1f}% (Michaud not helping)")
        print("   May need different configuration or task type")

    print(f"\n📊 Key Michaud Features Used:")
    if 'attention_stats' in michaud_summary:
        print(f"  Metabolic state transitions: {michaud_summary['attention_stats']['total_transitions']}")
        print(f"  Final state: {michaud_summary['attention_stats']['current_state']}")

    if 'satisfaction_history' in michaud_summary and michaud_summary['satisfaction_history']:
        avg_sat = sum(s['satisfaction'] for s in michaud_summary['satisfaction_history']) / len(michaud_summary['satisfaction_history'])
        print(f"  Average energy reduction (satisfaction): {avg_sat:.3f}")

    print("\n" + "="*80)
    print("✅ MICHAUD INTEGRATION TEST COMPLETE")
    print("="*80)

    return improvement


if __name__ == "__main__":
    print("\nStarting Michaud integration test...")
    print("Comparing basic vs enhanced consciousness loops.\n")

    improvement = asyncio.run(compare_loops())

    print(f"\nFinal Result: {improvement:+.1f}% quality change with Michaud enhancements")
