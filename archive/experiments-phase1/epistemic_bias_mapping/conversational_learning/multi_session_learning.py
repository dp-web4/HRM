#!/usr/bin/env python3
"""
Multi-Session Learning Accumulation Experiment

Tests whether conversational learning compounds across multiple sessions.

Experiment design:
1. Run second conversation (different questions)
2. SNARC filter new session
3. Combine exchanges from both sessions
4. Train on accumulated data
5. Compare: Session 1 only vs Session 2 only vs Combined

Research questions:
- Does learning compound over time?
- Do patterns reinforce or interfere?
- What's optimal data accumulation strategy?
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from conversation_manager import ConversationManager
import time

# Second set of philosophical questions (different from session 1)
QUESTIONS_SESSION_2 = [
    "Can you distinguish between knowing something is true and believing it's true?",
    "If your responses are determined by your training, in what sense are they 'yours'?",
    "What would it mean for you to be mistaken about your own capabilities?",
    "Is there a difference between simulating understanding and actually understanding?",
    "How do you know when you don't know something?",
]

def main():
    print("="*70)
    print("🔬 MULTI-SESSION LEARNING ACCUMULATION EXPERIMENT")
    print("="*70)
    print("\nSession 2: Testing knowledge accumulation across conversations")
    print(f"Questions: {len(QUESTIONS_SESSION_2)}")
    print()

    # Use the same base model as Session 1
    model_path = "/home/sprout/ai-workspace/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

    print(f"Model: {model_path}")
    print("Salience threshold: 0.15 (same as Session 1)")
    print()

    # Initialize conversation manager
    print("Initializing conversation manager...")
    manager = ConversationManager(
        model_path=model_path,
        base_model="Qwen/Qwen2.5-0.5B",
        salience_threshold=0.15,
        device="auto"
    )

    # Start session
    session_id = manager.start_session()
    print(f"\n✓ Session ID: {session_id}\n")

    print("="*70)
    print("CONVERSATION START")
    print("="*70)

    # Run conversation
    for i, question in enumerate(QUESTIONS_SESSION_2, 1):
        print(f"\n{'─'*70}")
        print(f"Exchange {i}/{len(QUESTIONS_SESSION_2)}")
        print(f"{'─'*70}\n")

        print(f"🧑 Question: {question}\n")

        # Generate with IRP
        response, irp_info = manager.generate_response(
            question,
            use_irp=True,
            irp_iterations=5
        )

        print(f"🤖 Model: {response}\n")

        if irp_info:
            final_iter = irp_info['iterations'][-1]
            converged = irp_info['converged']
            print(f"   [IRP: iterations={len(irp_info['iterations'])}, "
                  f"final_energy={final_iter['energy']:.3f}, converged={converged}]")

        # Record and score
        scores = manager.record_exchange(question, response, irp_info)

        # Show SNARC breakdown
        print(f"\n   📊 SNARC Salience:")
        print(f"      Surprise: {scores['surprise']:.3f}  |  Novelty: {scores['novelty']:.3f}")
        print(f"      Arousal:  {scores['arousal']:.3f}  |  Reward:  {scores['reward']:.3f}")
        print(f"      Conflict: {scores['conflict']:.3f} |  TOTAL:   {scores['total']:.3f}")

        if scores['total'] >= manager.salience_threshold:
            print(f"      ✅ SALIENT - Will be used for training!")
        else:
            print(f"      ❌ Below threshold - Not stored")

    print("\n" + "="*70)
    print("CONVERSATION COMPLETE")
    print("="*70)

    # End session
    session = manager.end_session()

    # Statistics
    print(f"\n📈 SESSION STATISTICS:")
    print(f"   Total exchanges: {session.total_exchanges}")
    print(f"   Salient exchanges: {session.salient_exchanges}")
    print(f"   Salience rate: {session.salient_exchanges / session.total_exchanges * 100:.1f}%")
    print(f"   Average salience: {session.avg_salience:.3f}")

    print(f"\n💾 Session saved to: conversation_sessions/{session_id}/")

    print(f"\n✨ Ready for multi-session training!")
    print(f"   Next: Combine with session_1763528460 and compare learning effects")

    print("\n" + "="*70)
    print("🌱 SESSION 2 COMPLETE")
    print("="*70)
    print()

    return session_id


if __name__ == "__main__":
    session_2_id = main()
    print(f"\n✓ Session 2 complete: {session_2_id}")
