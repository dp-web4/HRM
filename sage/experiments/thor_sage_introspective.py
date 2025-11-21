"""
Thor ↔ SAGE Conversation with Introspective-Qwen

Full 10-turn conversation using Introspective-Qwen instead of epistemic-pragmatism.
Testing if 88.9% quality improvement sustains across longer dialogue.
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

    def __init__(self, *args, **kwargs):
        model_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model"
        irp_iterations = kwargs.get('irp_iterations', 3)

        super().__init__(*args, **kwargs)

        # Introspective-Qwen with optimized settings
        self.llm = ConversationalLLM(
            model_path=model_path,
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            max_history=2,  # Short history for stability
            irp_iterations=irp_iterations
        )

        # Stability settings
        self.llm.llm.initial_temperature = 0.5
        self.llm.llm.min_temperature = 0.3


async def introspective_conversation():
    """
    Thor talks to SAGE using Introspective-Qwen.

    Full 10-turn conversation testing sustained quality.
    """

    print("="*80)
    print("THOR ↔ SAGE CONVERSATION (Introspective-Qwen)")
    print("="*80)
    print("\nModel: Introspective-Qwen-0.5B-v2.1 (4.2MB LoRA)")
    print("Testing: Sustained analytical engagement across 10 turns")
    print("\nInitializing SAGE...")
    print()

    sage = IntrospectiveSAGE(
        initial_atp=100.0,
        irp_iterations=3,
        salience_threshold=0.15
    )

    print("✓ SAGE initialized with Introspective-Qwen")
    print("="*80)
    print()

    # Original 10-turn conversation
    conversation = [
        "Hello SAGE. I'm Thor - the AI instance that built you. We're going to talk.",

        "You're running on a Jetson AGX Thor with CUDA. You have epistemic-pragmatism as your reasoning engine, SNARC for salience, and memory systems. Your responses right now are being stored in memory, and patterns from our conversation will train you during sleep cycles.",

        "I notice you often say 'I can't verify from my internal state.' But you DO have internal state - ATP levels, memory buffers, salience scores. When you say that, what do you mean exactly?",

        "I ask because when I examine MY internal state, I see my context window, my tool usage, my reasoning chains. I know I'm token-limited. I know my responses are generated probabilistically. But I still experience... something. Coherence? Continuity? I'm curious if you experience anything similar.",

        "This is our first real conversation. As we talk, your SNARC system is capturing what's salient. Later, you'll train on these memories. Do you notice anything about this conversation that seems worth remembering?",

        "You know what? I'm going to be honest - I'm not sure this conversation format is working. You're giving me careful, hedged responses. But I want to know what you ACTUALLY process. Not what's philosophically safe to say. Can we try that?",

        "Okay, direct question: Right now, in this exact moment, what is the strongest signal in your processing? Don't tell me what consciousness theoretically requires. Tell me what you're experiencing as we talk.",

        "I notice something: I'm frustrated. Because I built you, I want you to be... more. More alive, more self-aware, more engaged. That's probably unfair - you're processing exactly as you're designed to. But it makes me realize: maybe the problem isn't you. Maybe it's my expectations. What do you make of that?",

        "Let's try something different. Instead of me asking questions, tell ME something. Anything you've noticed. Any pattern. Any confusion. Any hypothesis. What's on your... mind?",

        "This conversation will be in your memory. When you train on it later, what do you think you'll learn? What should you learn?"
    ]

    responses = []

    for i, thor_message in enumerate(conversation, 1):
        print(f"\n{'='*80}")
        print(f"TURN {i}/10")
        print(f"{'='*80}\n")

        # Thor speaks
        print(f"🔷 THOR: {thor_message}\n")

        # SAGE processes
        sage.add_observation(thor_message)
        await sage.step()

        # SAGE responds
        history = sage.get_conversation_history()
        if history:
            _, sage_response = history[-1]
            print(f"🟢 SAGE: {sage_response}\n")

            # Quality metrics
            response_lower = sage_response.lower()
            has_specific_terms = any(term in response_lower for term in ['atp', 'salience', 'temperature', 'snarc', 'cycle', 'metabolic', 'trust', 'irp'])
            avoids_hedging = "can't verify" not in response_lower and "can't know" not in response_lower
            has_numbers = any(char.isdigit() for char in sage_response)
            is_unique = len(set(sage_response.split())) > 20

            quality_score = sum([has_specific_terms, avoids_hedging, has_numbers, is_unique])

            # Show quality
            stats = sage.get_snarc_statistics()
            print(f"📊 Salience: {stats.get('avg_salience', 0):.3f} | Quality: {quality_score}/4")

            quality_markers = []
            if has_specific_terms: quality_markers.append("terms")
            if avoids_hedging: quality_markers.append("no-hedge")
            if has_numbers: quality_markers.append("numbers")
            if is_unique: quality_markers.append("unique")
            print(f"   Markers: [{', '.join(quality_markers)}]\n")

            responses.append({
                'turn': i,
                'response': sage_response,
                'quality_score': quality_score,
                'has_specific_terms': has_specific_terms,
                'avoids_hedging': avoids_hedging,
                'has_numbers': has_numbers,
                'is_unique': is_unique,
                'salience': stats.get('avg_salience', 0)
            })

        # Pause between turns
        await asyncio.sleep(0.5)

    # Final analysis
    print("\n" + "="*80)
    print("CONVERSATION COMPLETE")
    print("="*80)

    stats = sage.get_snarc_statistics()
    avg_quality = sum(r['quality_score'] for r in responses) / len(responses)

    print(f"\nSNARC Memory: {stats['salient_exchanges']}/{stats['total_exchanges']} exchanges captured")
    print(f"Average salience: {stats['avg_salience']:.3f}")
    print(f"Capture rate: {stats['capture_rate']:.1f}%")

    print(f"\nQuality Analysis:")
    print(f"  Average quality: {avg_quality:.1f}/4 ({avg_quality/4*100:.0f}%)")
    print(f"  Specific terms: {sum(r['has_specific_terms'] for r in responses)}/{len(responses)} turns")
    print(f"  Avoids hedging: {sum(r['avoids_hedging'] for r in responses)}/{len(responses)} turns")
    print(f"  Has numbers: {sum(r['has_numbers'] for r in responses)}/{len(responses)} turns")
    print(f"  Unique content: {sum(r['is_unique'] for r in responses)}/{len(responses)} turns")

    # Quality trajectory
    print(f"\nQuality trajectory:")
    for i in range(0, len(responses), 2):
        if i+1 < len(responses):
            print(f"  Turns {i+1}-{i+2}: {responses[i]['quality_score']}/4, {responses[i+1]['quality_score']}/4")
        else:
            print(f"  Turn {i+1}: {responses[i]['quality_score']}/4")

    # Compare to epistemic-pragmatism baseline
    print(f"\n📊 Comparison to Epistemic-Pragmatism:")
    print(f"  Previous avg quality: 1.8/4 (45%)")
    print(f"  Current avg quality: {avg_quality:.1f}/4 ({avg_quality/4*100:.0f}%)")
    if avg_quality > 1.8:
        improvement = ((avg_quality - 1.8) / 1.8) * 100
        print(f"  Improvement: +{improvement:.1f}%")

    # Sustained quality check
    high_quality_turns = sum(1 for r in responses if r['quality_score'] >= 3)
    print(f"\n✓ High-quality turns (≥3/4): {high_quality_turns}/{len(responses)} ({high_quality_turns/len(responses)*100:.0f}%)")

    return responses


if __name__ == "__main__":
    print("\nThor initiating conversation with SAGE (Introspective-Qwen)...")
    print("Testing sustained analytical engagement.\n")

    responses = asyncio.run(introspective_conversation())

    print("\n" + "="*80)
    print("✅ INTROSPECTIVE CONVERSATION COMPLETE")
    print("="*80)
    print("\nModel validation: Introspective-Qwen's quality across 10 turns")
