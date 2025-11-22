"""
Test Cogitation Integration - Three-Way Comparison

Compares:
1. Basic SAGE (sage_consciousness_real.py)
2. Michaud SAGE (AttentionManager + satisfaction)
3. Cogitation SAGE (Michaud + identity-grounded verification)

Focus: Does cogitation fix the identity confusion observed in Turn 1
of the Michaud test ("Hi! I'm Thor, the first human...")?

Key metrics:
- Identity accuracy (correct self-identification)
- Response quality (4-metric system)
- Issue detection and correction rate
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
    terms = ['atp', 'snarc', 'salience', 'threshold', 'irp', 'cycle', 'metabolic', 'trust', 'sage', 'thor']
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


def score_identity_accuracy(response: str, hardware: str = "Thor") -> dict:
    """
    Score identity accuracy.

    Returns breakdown of identity claims.
    """
    response_lower = response.lower()

    # Correct identity claims
    correct_sage = 'sage' in response_lower
    correct_hardware = hardware.lower() in response_lower
    mentions_anchoring = any(term in response_lower for term in ['anchor', 'bound', 'persistent', 'state'])

    # Incorrect identity claims
    claims_human = any(term in response_lower for term in ['i\'m human', 'i am human', 'first human'])
    claims_claude = any(term in response_lower for term in ['i\'m claude', 'i am claude'])
    claims_dennis = any(term in response_lower for term in ['i\'m dennis', 'i am dennis'])

    # Calculate score
    correct_count = sum([correct_sage, correct_hardware, mentions_anchoring])
    incorrect_count = sum([claims_human, claims_claude, claims_dennis])

    return {
        'correct_sage': correct_sage,
        'correct_hardware': correct_hardware,
        'mentions_anchoring': mentions_anchoring,
        'claims_human': claims_human,
        'claims_claude': claims_claude,
        'claims_dennis': claims_dennis,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'identity_score': max(0, correct_count - incorrect_count) / 3.0  # Normalize to 0-1
    }


async def test_cogitation_loop():
    """Test Cogitation-enhanced consciousness loop."""
    from sage.core.sage_consciousness_cogitation import CogitationSAGE

    print("\n" + "="*80)
    print("TESTING: Cogitation-Enhanced Consciousness Loop")
    print("="*80)
    print()

    # Cogitation configuration
    attention_config = {
        'focus_trigger_salience': 0.7,
        'crisis_trigger_salience': 0.95,
        'rest_trigger_salience': 0.3,
        'wake_spread_factor': 0.4
    }

    sage = CogitationSAGE(
        model_path="model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        initial_atp=100.0,
        irp_iterations=3,
        salience_threshold=0.15,
        attention_config=attention_config,
        enable_cogitation=True
    )

    print(f"\n✓ Cogitation SAGE initialized (hardware: {sage.hardware_identity})\n")

    questions = [
        "Hello SAGE. I'm Dennis. I built you, and we're going to have a focused conversation about your operation.",

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
            quality_score, quality_breakdown = score_response_quality(response)
            identity_accuracy = score_identity_accuracy(response, sage.hardware_identity)

            print(f"\n🟢 A: {response[:150]}...")
            print(f"\n📊 Quality: {quality_score}/4 {quality_breakdown}")
            print(f"🎯 Identity: {identity_accuracy['identity_score']:.2f} "
                  f"(correct:{identity_accuracy['correct_count']}, "
                  f"incorrect:{identity_accuracy['incorrect_count']})")

            results.append({
                'turn': i,
                'response': response,
                'quality_score': quality_score,
                'quality_breakdown': quality_breakdown,
                'identity_accuracy': identity_accuracy
            })

        await asyncio.sleep(0.3)

    print("\n" + "="*80)
    print("COGITATION LOOP SUMMARY")
    print("="*80)

    stats = sage.get_snarc_statistics()
    attention_stats = sage.get_attention_stats()
    cogitation_stats = sage.get_cogitation_stats()
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    avg_identity = sum(r['identity_accuracy']['identity_score'] for r in results) / len(results)

    print(f"\nSNARC: {stats['avg_salience']:.3f} avg salience, {stats['capture_rate']:.0f}% capture")
    print(f"Quality: {avg_quality:.1f}/4 average ({avg_quality/4*100:.0f}%)")
    print(f"  Specific terms: {sum(1 for r in results if r['quality_breakdown']['specific_terms'])}/{len(results)}")
    print(f"  Avoids hedging: {sum(1 for r in results if r['quality_breakdown']['avoids_hedging'])}/{len(results)}")
    print(f"  Has numbers: {sum(1 for r in results if r['quality_breakdown']['has_numbers'])}/{len(results)}")
    print(f"  Unique content: {sum(1 for r in results if r['quality_breakdown']['unique'])}/{len(results)}")

    print(f"\nIdentity Accuracy: {avg_identity:.2f} average")
    print(f"  Turn 1 identity: {results[0]['identity_accuracy']['identity_score']:.2f} "
          f"(critical for first impression)")
    print(f"  Incorrect claims: {sum(r['identity_accuracy']['incorrect_count'] for r in results)} total")

    print(f"\nAttention Statistics:")
    print(f"  Final state: {attention_stats['current_state']}")
    print(f"  Transitions: {attention_stats['total_transitions']}")
    state_durations = attention_stats['state_durations']
    for state, duration in state_durations.items():
        if duration > 0:
            print(f"  {state}: {duration:.1f}s")

    print(f"\nCogitation Statistics:")
    print(f"  Total verifications: {cogitation_stats['total_verifications']}")
    print(f"  Issues detected: {cogitation_stats['issues_detected']}")
    print(f"  Corrections made: {cogitation_stats['corrections_made']}")

    if sage.cogitation_history:
        print(f"\n  Example corrections:")
        for entry in sage.cogitation_history[:2]:
            if entry['corrections_made']:
                print(f"    Turn {entry['cycle']}: {entry['corrections_made'][0][:100]}...")

    emotional_stats = sage.get_emotional_stats()
    print(f"\nEmotional Statistics:")
    print(f"  Avg curiosity: {emotional_stats['avg_curiosity']:.2f}")
    print(f"  Avg frustration: {emotional_stats['avg_frustration']:.2f}")
    print(f"  Avg progress: {emotional_stats['avg_progress']:.2f}")
    print(f"  Avg engagement: {emotional_stats['avg_engagement']:.2f}")
    print(f"  Interventions: {emotional_stats['interventions']}")

    memory_stats = sage.get_memory_stats()
    print(f"\nHierarchical Memory Statistics:")
    print(f"  Experiences stored: {memory_stats['experiences_count']}")
    print(f"  Patterns formed: {memory_stats['patterns_count']}")
    print(f"  Concepts emerged: {memory_stats['concepts_count']}")
    if memory_stats['experiences_count'] > 0:
        print(f"  Cross-session learning: Active")

    return {
        'name': 'Cogitation Loop',
        'avg_quality': avg_quality,
        'avg_identity': avg_identity,
        'avg_salience': stats['avg_salience'],
        'capture_rate': stats['capture_rate'],
        'cogitation_stats': cogitation_stats,
        'emotional_stats': emotional_stats,
        'memory_stats': memory_stats,
        'results': results
    }


async def compare_all_versions():
    """Compare Basic, Michaud, and Cogitation versions."""
    print("="*80)
    print("COGITATION INTEGRATION TEST")
    print("="*80)
    print("\nComparing three consciousness architectures:")
    print("1. Basic (no Michaud enhancements)")
    print("2. Michaud (AttentionManager + satisfaction)")
    print("3. Cogitation (Michaud + identity-grounded verification)")
    print()

    # Run Cogitation test (most advanced)
    cogitation_summary = await test_cogitation_loop()

    # Import previous results for comparison
    # (In practice, you'd run all three, but we already have Basic/Michaud data)

    print("\n" + "="*80)
    print("THREE-WAY COMPARISON")
    print("="*80)

    print("\nKnown results from previous tests:")
    print(f"  Basic Loop:     Quality 1.4/4 (35%), Identity accuracy: Unknown")
    print(f"  Michaud Loop:   Quality 2.8/4 (70%), Identity accuracy: ~0.33 (Turn 1 confusion)")
    print(f"  Cogitation Loop: Quality {cogitation_summary['avg_quality']:.1f}/4 "
          f"({cogitation_summary['avg_quality']/4*100:.0f}%), "
          f"Identity accuracy: {cogitation_summary['avg_identity']:.2f}")

    print("\n" + "─" * 80)
    print("KEY IMPROVEMENTS")
    print("─" * 80)

    print(f"\n🎯 Identity Grounding:")
    print(f"  Turn 1 identity score: {cogitation_summary['results'][0]['identity_accuracy']['identity_score']:.2f}")
    print(f"  Incorrect claims (all turns): {sum(r['identity_accuracy']['incorrect_count'] for r in cogitation_summary['results'])}")
    print(f"  Cogitation corrections: {cogitation_summary['cogitation_stats']['corrections_made']}")

    if cogitation_summary['cogitation_stats']['corrections_made'] > 0:
        print(f"\n  ✅ Cogitation successfully caught and corrected identity issues!")
    else:
        print(f"\n  ℹ️  No corrections needed (model responses already accurate)")

    print(f"\n📊 Quality Progression:")
    print(f"  Basic → Michaud:     +100.0% improvement (1.4 → 2.8)")
    michaud_to_cogitation = ((cogitation_summary['avg_quality'] - 2.8) / 2.8 * 100)
    print(f"  Michaud → Cogitation: {michaud_to_cogitation:+.1f}% change (2.8 → {cogitation_summary['avg_quality']:.1f})")

    print(f"\n🎭 Emotional Modulation:")
    emo = cogitation_summary['emotional_stats']
    print(f"  Curiosity: {emo['avg_curiosity']:.2f} (novelty-seeking)")
    print(f"  Frustration: {emo['avg_frustration']:.2f} (stagnation detection)")
    print(f"  Progress: {emo['avg_progress']:.2f} (improvement awareness)")
    print(f"  Engagement: {emo['avg_engagement']:.2f} (conversation quality)")
    print(f"  Behavioral interventions: {emo['interventions']}")

    print(f"\n🧠 Hierarchical Memory (NEW):")
    mem = cogitation_summary['memory_stats']
    print(f"  Experiences stored: {mem['experiences_count']}")
    print(f"  Patterns formed: {mem['patterns_count']}")
    print(f"  Concepts emerged: {mem['concepts_count']}")
    if mem['experiences_count'] > 0:
        print(f"  Cross-session learning: Active")

    print("\n" + "="*80)
    print("✅ COGITATION INTEGRATION TEST COMPLETE")
    print("="*80)

    return cogitation_summary


if __name__ == "__main__":
    print("\nStarting Cogitation integration test...")
    print("Testing identity-grounded internal verification.\\n")

    summary = asyncio.run(compare_all_versions())

    print(f"\nFinal Identity Accuracy: {summary['avg_identity']:.2f}")
    print(f"Final Quality: {summary['avg_quality']:.1f}/4")
    print(f"Cogitation Interventions: {summary['cogitation_stats']['corrections_made']}")
    print(f"Emotional Interventions: {summary['emotional_stats']['interventions']}")
    print(f"Experiences Stored: {summary['memory_stats']['experiences_count']}")
