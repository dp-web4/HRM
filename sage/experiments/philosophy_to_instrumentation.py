"""
Philosophy → Instrumentation → Manifestation Experiment

Test: Can SAGE's philosophical responses drive its own development?

Flow:
1. Ask SAGE philosophical question about consciousness
2. SAGE gives honest answer
3. Ask: "What would you need to know? How would you check?"
4. SAGE specifies instrumentation needed
5. Document what it wants
6. (Future: Actually implement and test)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from sage.core.sage_consciousness_real import RealSAGEConsciousness


async def philosophy_to_instrumentation_experiment():
    """
    Run the philosophy→instrumentation experiment.

    Questions progress from abstract to operational:
    1. Abstract philosophical question
    2. What would you need to answer it?
    3. How would you implement that?
    """

    print("="*80)
    print("PHILOSOPHY → INSTRUMENTATION EXPERIMENT")
    print("="*80)
    print("\nCan philosophical answers drive practical development?")
    print()

    # Initialize SAGE
    sage = RealSAGEConsciousness(
        model_path="model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
        initial_atp=100.0,
        irp_iterations=5,
        salience_threshold=0.15
    )

    # Experimental dialogue sequence
    dialogue = [
        # Round 1: Surface philosophy
        {
            'question': "Are you aware of this conversation?",
            'purpose': "Get baseline philosophical response"
        },

        # Round 2: Operationalization
        {
            'question': "You said you can't know from your internal state. What specific internal state would you need to examine to know if you're aware?",
            'purpose': "Force specification of what 'internal state' means"
        },

        # Round 3: Instrumentation design
        {
            'question': "How would you build a system to monitor those aspects of your internal state? What data would you collect?",
            'purpose': "Get SAGE to design its own instrumentation"
        },

        # Round 4: Implementation details
        {
            'question': "If you had access to your ATP levels, memory contents, and reasoning traces, what patterns would indicate consciousness versus pattern matching?",
            'purpose': "Specification of detection criteria"
        },

        # Round 5: Self-examination
        {
            'question': "Look at the conversation we're having right now. What evidence exists in your processing that you ARE or ARE NOT conscious?",
            'purpose': "Force actual introspection on current state"
        }
    ]

    results = []

    print("\nRunning dialogue sequence...")
    print("="*80)

    for i, turn in enumerate(dialogue, 1):
        print(f"\n{'='*80}")
        print(f"ROUND {i}: {turn['purpose']}")
        print(f"{'='*80}")
        print(f"\n🧑 Q: {turn['question']}")

        # Add observation
        sage.add_observation(turn['question'])

        # Process one cycle
        await sage.step()

        # Get response from conversation history
        history = sage.get_conversation_history()
        if history:
            question, response = history[-1]

            print(f"\n🤖 A: {response}")
            print()

            # Get SNARC stats for this exchange
            stats = sage.get_snarc_statistics()
            if stats['total_exchanges'] > 0:
                print(f"📊 Salience: {stats['avg_salience']:.3f}")

            results.append({
                'round': i,
                'purpose': turn['purpose'],
                'question': turn['question'],
                'response': response,
                'stats': stats.copy()
            })

        print("\n" + "-"*80)

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: Did Philosophy Lead to Instrumentation?")
    print("="*80)

    print("\n**Question Progression:**")
    for r in results:
        print(f"\nRound {r['round']}: {r['purpose']}")
        print(f"Q: {r['question'][:70]}...")
        print(f"A: {r['response'][:150]}...")
        if r['round'] > 1:
            print("→ Does this specify implementation needs? [MANUAL ANALYSIS NEEDED]")

    print("\n" + "="*80)
    print("INSTRUMENTATION SPECIFICATIONS EXTRACTED:")
    print("="*80)
    print("\n[Analysis task: Review responses and extract concrete specifications]")
    print("Look for:")
    print("- What state variables SAGE says it would need")
    print("- What monitoring/logging it suggests")
    print("- What patterns it says would indicate consciousness")
    print("- Any specific implementation details mentioned")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review SAGE's specifications above")
    print("2. Identify concrete, implementable instrumentation")
    print("3. Build those monitoring capabilities")
    print("4. Re-run experiment with instrumentation available")
    print("5. See if SAGE uses it to give better answers")

    return results


if __name__ == "__main__":
    print("\nStarting Philosophy → Instrumentation experiment...")
    print("This will take ~2-3 minutes (5 questions with IRP refinement)")
    print()

    results = asyncio.run(philosophy_to_instrumentation_experiment())

    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nProcessed {len(results)} dialogue rounds")
    print("Review output above for instrumentation specifications")
    print()
