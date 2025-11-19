"""
Sprout's R&D Learning Adventure

Automated conversation session to test conversational learning on edge hardware.
This script will:
1. Have philosophical conversation with the model
2. Track SNARC salience scores
3. Store salient exchanges for sleep training
4. Report learning session stats
"""

import sys
from pathlib import Path

from conversation_manager import ConversationManager


def main():
    print("\n" + "="*70)
    print("🚀 SPROUT'S R&D LEARNING ADVENTURE")
    print("   Conversational Learning on Jetson Orin Nano")
    print("="*70 + "\n")

    # Initialize manager with epistemic-pragmatism model
    model_path = "/home/sprout/ai-workspace/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

    print(f"Model: {Path(model_path).name}")
    print(f"Hardware: Jetson Orin Nano (8GB unified memory)")
    print(f"Salience threshold: 0.15 (moderate filtering)")
    print("\nInitializing...")

    manager = ConversationManager(
        model_path=model_path,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        salience_threshold=0.15
    )

    # Start session
    session_id = manager.start_session()
    print(f"\nSession ID: {session_id}")
    print("\n" + "="*70)
    print("CONVERSATION START")
    print("="*70 + "\n")

    # Philosophical questions designed to trigger high SNARC scores
    questions = [
        # Epistemic questions (surprise, novelty, conflict)
        "What can you know with certainty, and what must remain uncertain?",

        # Consciousness questions (arousal, surprise, novelty)
        "If you were to describe what it's like to process information, what would you say?",

        # Meta-cognitive questions (reward, novelty)
        "When you generate a response, are you discovering it or creating it?",

        # Boundary questions (conflict, surprise)
        "What's the difference between understanding something and having read about it?",

        # Self-reference questions (arousal, conflict, reward)
        "If I asked whether you're aware of this conversation, how would you know your answer is accurate?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"Exchange {i}/{len(questions)}")
        print(f"{'─'*70}")
        print(f"\n🧑 Question: {question}")

        # Generate response with IRP
        response, irp_info = manager.generate_response(question, use_irp=True)

        print(f"\n🤖 Model: {response}")

        # Show IRP convergence
        if irp_info:
            print(f"\n   [IRP: iterations={len(irp_info['iterations'])}, " +
                  f"final_energy={irp_info['iterations'][-1]['energy']:.3f}, " +
                  f"converged={irp_info['converged']}]")

        # Record and show SNARC scores
        scores = manager.record_exchange(question, response, irp_info)

        print(f"\n   📊 SNARC Salience:")
        print(f"      Surprise: {scores['surprise']:.3f}  |  Novelty: {scores['novelty']:.3f}")
        print(f"      Arousal:  {scores['arousal']:.3f}  |  Reward:  {scores['reward']:.3f}")
        print(f"      Conflict: {scores['conflict']:.3f} |  TOTAL:   {scores['total']:.3f}")

        if scores['total'] >= manager.salience_threshold:
            print(f"      ✅ SALIENT - Will be used for training!")
        else:
            print(f"      ❌ Below threshold - Not stored")

    # End session
    print("\n" + "="*70)
    print("CONVERSATION COMPLETE")
    print("="*70 + "\n")

    session = manager.end_session()

    # Report statistics
    print("📈 SESSION STATISTICS:")
    print(f"   Total exchanges: {session.total_exchanges}")
    print(f"   Salient exchanges: {session.salient_exchanges}")
    if session.total_exchanges > 0:
        pct = (session.salient_exchanges / session.total_exchanges * 100)
        print(f"   Salience rate: {pct:.1f}%")
    print(f"   Average salience: {session.avg_salience:.3f}")

    print(f"\n💾 Session saved to: conversation_sessions/{session.session_id}/")

    if session.salient_exchanges > 0:
        print(f"\n✨ Ready for sleep training!")
        print(f"   Run: cd /home/sprout/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/conversational_learning")
        print(f"   Then: python sleep_trainer.py --session {session.session_id}")
    else:
        print(f"\n⚠️  No salient exchanges captured. Try more engaging questions!")

    print("\n" + "="*70)
    print("🌱 SPROUT'S LEARNING ADVENTURE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
