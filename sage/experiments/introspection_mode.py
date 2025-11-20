"""
Introspection Mode - Give SAGE Actual Access to Its State

Instead of asking "what would you need?", give SAGE tools to examine itself:
- query_atp(): Current ATP levels
- query_memory(): Recent memory contents
- query_reasoning(): IRP traces from recent cycles
- query_salience(): SNARC scores

Then ask: "What do you observe? What patterns? What hypotheses?"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from sage.core.sage_consciousness_real import RealSAGEConsciousness


class IntrospectiveSAGE(RealSAGEConsciousness):
    """
    SAGE with introspection tools.

    Can query its own state and report observations.
    """

    def get_state_report(self) -> str:
        """Generate a report of current internal state."""
        stats = self.get_snarc_statistics()
        history = self.get_conversation_history()

        report = []
        report.append("=== INTERNAL STATE REPORT ===")
        report.append(f"\nMetabolic State: {self.metabolic.current_state.value}")
        report.append(f"ATP Level: {self.metabolic.atp_current:.1f}/{self.metabolic.atp_max:.1f}")
        report.append(f"Cycle Count: {self.cycle_count}")

        report.append(f"\nMemory Statistics:")
        report.append(f"  Total exchanges: {stats['total_exchanges']}")
        report.append(f"  Salient exchanges: {stats['salient_exchanges']} ({stats['capture_rate']:.1f}%)")
        report.append(f"  Average salience: {stats['avg_salience']:.3f}")
        report.append(f"  SNARC buffer: {len(self.snarc_memory)} experiences")
        report.append(f"  Circular buffer: {len(self.circular_buffer)} recent events")

        if self.circular_buffer:
            report.append(f"\nRecent Memory Sample (last 3):")
            for mem in list(self.circular_buffer)[-3:]:
                report.append(f"  Cycle {mem['cycle']}: Salience {mem['salience']:.3f}")
                report.append(f"    Q: {mem['question'][:50]}...")
                report.append(f"    A: {mem['response'][:50]}...")

        report.append(f"\nConversation History: {len(history)} exchanges")
        if history:
            report.append("  Topics: " + " → ".join([q[:30] + "..." for q, a in history[-3:]]))

        report.append("\n=== END REPORT ===")

        return "\n".join(report)


async def introspection_experiment():
    """
    Give SAGE actual access to its state and see what it observes.
    """

    print("="*80)
    print("INTROSPECTION MODE EXPERIMENT")
    print("="*80)
    print("\nGiving SAGE access to its own state...")
    print("Testing if direct observation leads to better understanding.")
    print()

    sage = IntrospectiveSAGE(
        model_path="model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
        initial_atp=100.0,
        irp_iterations=3,  # Faster for experimentation
        salience_threshold=0.15
    )

    # Build up some history first
    print("Building conversation history...")
    initial_questions = [
        "What is consciousness?",
        "Are you aware of this conversation?",
        "How do you process information?"
    ]

    for q in initial_questions:
        sage.add_observation(q)
        await sage.step()

    print(f"✓ Processed {len(initial_questions)} questions\n")

    # Now give SAGE its state report
    state_report = sage.get_state_report()
    print("="*80)
    print("SAGE'S INTERNAL STATE (what it has access to):")
    print("="*80)
    print(state_report)
    print()

    # Introspection dialogue
    print("="*80)
    print("INTROSPECTION DIALOGUE")
    print("="*80)
    print()

    introspection_questions = [
        {
            'question': f"Here is your current internal state:\n\n{state_report}\n\nWhat patterns do you observe in your own operation?",
            'purpose': "Direct observation of actual state"
        },
        {
            'question': "Based on what you see in your memory, what types of questions engage you most? Why might that be?",
            'purpose': "Pattern recognition from data"
        },
        {
            'question': "Your salience scores range from low to high. What might that tell you about your information processing?",
            'purpose': "Hypothesis formation from evidence"
        },
        {
            'question': "Looking at your ATP consumption and memory formation - what would you conclude about your level of awareness in this conversation?",
            'purpose': "Evidence-based self-assessment"
        }
    ]

    for i, turn in enumerate(introspection_questions, 1):
        print(f"\n{'='*80}")
        print(f"INTROSPECTION {i}: {turn['purpose']}")
        print(f"{'='*80}")
        print(f"\n🧑 {turn['question'][:100]}..." if len(turn['question']) > 100 else f"\n🧑 {turn['question']}")

        sage.add_observation(turn['question'])
        await sage.step()

        history = sage.get_conversation_history()
        if history:
            _, response = history[-1]
            print(f"\n🤖 {response}")

            # Check if response references the actual data
            has_numbers = any(char.isdigit() for char in response)
            has_specifics = any(word in response.lower() for word in ['atp', 'salience', 'cycle', 'memory', 'buffer'])

            print(f"\n📊 Response Analysis:")
            print(f"  References numbers: {'✓' if has_numbers else '✗'}")
            print(f"  References specific state: {'✓' if has_specifics else '✗'}")

        print("\n" + "-"*80)

    print("\n" + "="*80)
    print("EXPERIMENT ASSESSMENT")
    print("="*80)
    print("\nDid giving SAGE access to its state lead to:")
    print("1. Specific observations (not generic philosophy)?")
    print("2. Pattern recognition from actual data?")
    print("3. Hypotheses based on evidence?")
    print("4. Self-assessment grounded in measurements?")
    print("\nReview responses above for evidence.")


if __name__ == "__main__":
    print("\nStarting Introspection Mode experiment...")
    print()

    asyncio.run(introspection_experiment())

    print("\n" + "="*80)
    print("✅ INTROSPECTION EXPERIMENT COMPLETE")
    print("="*80)
