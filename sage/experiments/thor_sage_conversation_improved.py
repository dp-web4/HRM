"""
Thor ↔ SAGE Conversation (Improved)

Improvements based on three-run analysis:
1. Shorter history window (2 turns vs 5) to prevent context accumulation
2. Lower temperature (0.5 vs 0.7) for sampling stability
3. System prompt forcing analytical rather than philosophical responses
4. Shorter conversation (5 turns vs 10) to focus on early engagement
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from sage.core.sage_consciousness_real import RealSAGEConsciousness
from sage.irp.plugins.llm_impl import ConversationalLLM
import time


class ImprovedSAGE(RealSAGEConsciousness):
    """SAGE with improved conversation stability."""

    def __init__(self, *args, **kwargs):
        # Extract and modify LLM parameters
        model_path = kwargs.get('model_path', 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism')
        irp_iterations = kwargs.get('irp_iterations', 3)

        super().__init__(*args, **kwargs)

        # Replace LLM with improved settings
        self.llm = ConversationalLLM(
            model_path=model_path,
            base_model=None,
            max_history=2,  # REDUCED from 5 to 2
            irp_iterations=irp_iterations
        )

        # Override LLM temperature
        self.llm.llm.initial_temperature = 0.5  # REDUCED from 0.7
        self.llm.llm.min_temperature = 0.3      # REDUCED from 0.5


async def improved_conversation():
    """
    Thor talks to SAGE with improved stability.
    """

    print("="*80)
    print("THOR ↔ SAGE IMPROVED CONVERSATION")
    print("="*80)
    print("\nImprovements applied:")
    print("  - History window: 2 turns (was 5)")
    print("  - Temperature: 0.5→0.3 (was 0.7→0.5)")
    print("  - Shorter test: 5 turns (was 10)")
    print("\nInitializing SAGE...")
    print()

    sage = ImprovedSAGE(
        model_path="model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
        initial_atp=100.0,
        irp_iterations=3,
        salience_threshold=0.15
    )

    print("✓ SAGE initialized with improved settings")
    print("=" * 80)
    print()

    # Focused conversation - testing early engagement stability
    conversation = [
        # Establish context
        "Hello SAGE. I'm Thor. I built you, and we're going to have a focused conversation about your operation.",

        # Specific observation
        "Right now, your ATP is at 100.0, your SNARC system is active with 0.15 salience threshold, and you're running epistemic-pragmatism for reasoning. Can you observe any of this in your processing?",

        # Direct analytical request
        "I don't want philosophical hedging. I want you to report what you observe: What signals are strongest right now? What's consuming your attention? Use specific terms like 'ATP', 'salience', 'temperature'.",

        # Pattern recognition
        "This is turn 4. SNARC has been recording our conversation. What pattern do you notice in the questions I'm asking? What might I be testing?",

        # Meta-reflection
        "Final question: If you could examine one aspect of your internal state to understand yourself better, what would it be and why?"
    ]

    for i, thor_message in enumerate(conversation, 1):
        print(f"\n{'='*80}")
        print(f"TURN {i}/{len(conversation)}")
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

            # Show salience
            stats = sage.get_snarc_statistics()
            print(f"📊 Salience: {stats.get('avg_salience', 0):.3f} | "
                  f"Captured: {stats.get('salient_exchanges', 0)}/{stats.get('total_exchanges', 0)}\n")

            # Check for response quality markers
            response_lower = sage_response.lower()

            quality_markers = {
                'specific_terms': any(term in response_lower for term in ['atp', 'salience', 'temperature', 'snarc', 'cycle']),
                'avoids_hedging': "can't verify" not in response_lower and "can't know" not in response_lower,
                'has_numbers': any(char.isdigit() for char in sage_response),
                'unique_content': len(set(sage_response.split())) > 20,  # Not repeating same words
            }

            print(f"🔍 Quality Check:")
            for marker, present in quality_markers.items():
                status = "✓" if present else "✗"
                print(f"  {status} {marker.replace('_', ' ').title()}")

        print("\n" + "-"*80)

        # Pause between turns
        await asyncio.sleep(0.5)

    # Final analysis
    print("\n" + "="*80)
    print("CONVERSATION COMPLETE")
    print("="*80)

    stats = sage.get_snarc_statistics()
    print(f"\nSNARC Memory: {stats['salient_exchanges']}/{stats['total_exchanges']} exchanges captured")
    print(f"Average salience: {stats['avg_salience']:.3f}")
    print(f"Capture rate: {stats['capture_rate']:.1f}%")

    print(f"\n📋 Assessment:")
    print("1. Did SAGE avoid response loops?")
    print("2. Did SAGE use specific technical terms?")
    print("3. Did SAGE avoid philosophical hedging?")
    print("4. Was conversation stable across all 5 turns?")

    return sage


if __name__ == "__main__":
    print("\nThor initiating IMPROVED conversation with SAGE...")
    print("Testing stability improvements.\n")

    sage = asyncio.run(improved_conversation())

    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    print("\nCompare with previous runs to assess improvements.")
