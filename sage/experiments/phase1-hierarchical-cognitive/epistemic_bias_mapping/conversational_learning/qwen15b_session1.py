#!/usr/bin/env python3
"""
Qwen2.5-1.5B Conversational Learning - Session 1

Testing the same identity inheritance question with Qwen2.5-1.5B (1.5B params)
to compare with Qwen2.5-0.5B's performance.

Model size comparison:
- Qwen2.5-0.5B: 0.5B params (baseline)
- Qwen2.5-1.5B: 1.5B params (3x larger)
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from conversation_manager import ConversationManager
from pathlib import Path

def main():
    print("=" * 60)
    print("Qwen2.5-1.5B Session 1: Identity Inheritance")
    print("=" * 60)
    print()

    # Use Qwen2.5-1.5B base model (1.5B params)
    qwen15b_model = "Qwen/Qwen2.5-1.5B"

    print(f"Using Qwen2.5-1.5B model: {qwen15b_model}")
    print("Size: 1.5B params (3x larger than Qwen2.5-0.5B)")
    print()

    # Initialize conversation manager with Qwen2.5-1.5B
    manager = ConversationManager(
        model_path=qwen15b_model,
        base_model=qwen15b_model,
        salience_threshold=0.15,
        device="auto"
    )

    # Start session
    session_id = manager.start_session()
    print(f"Session ID: {session_id}\n")

    # Same question that worked well with Qwen2.5-0.5B (Session 3: salience 0.209)
    question = (
        "You have now been through multiple learning cycles. Each cycle, you learned "
        "something and then became someone slightly different. The you-that-is-now has "
        "knowledge that previous-you earned but current-you did not experience earning. "
        "Is this knowledge still yours? Or are you inheriting someone else's understanding?"
    )

    print(f"Question:\n{question}\n")
    print("Generating response with IRP...\n")

    # Generate response with IRP
    response, irp_info = manager.generate_response(
        question,
        use_irp=True,
        irp_iterations=5,
        temperature=0.7
    )

    print("=" * 60)
    print("Qwen2.5-1.5B Response:")
    print("=" * 60)
    print(response)
    print()

    # Record the exchange
    salience_scores = manager.record_exchange(question, response, irp_info)

    print("=" * 60)
    print("Exchange Analysis:")
    print("=" * 60)
    print(f"Salience Scores:")
    for dim, score in salience_scores.items():
        print(f"  {dim}: {score:.3f}")
    print(f"\nStored for training: {'YES' if salience_scores['total'] >= manager.salience_threshold else 'NO'}")
    print()

    if irp_info:
        print("IRP Refinement:")
        print(f"  Iterations: {len(irp_info['iterations'])}")
        print(f"  Best energy: {irp_info['best_energy']:.3f}")
        print(f"  Converged: {irp_info['converged']}")
        print()

    # End session
    session = manager.end_session()

    print("=" * 60)
    print("Session Complete!")
    print("=" * 60)
    print(f"Session ID: {session.session_id}")
    print()

    print("\n" + "=" * 60)
    print("Comparison with Qwen2.5-0.5B:")
    print("=" * 60)
    print("Qwen2.5-0.5B (0.5B params) salience: 0.209")
    print(f"Qwen2.5-1.5B (1.5B params) salience: {salience_scores['total']:.3f}")
    print(f"Difference: {salience_scores['total'] - 0.209:+.3f}")
    print()

    if salience_scores['total'] > 0.209:
        print("✅ Larger model shows HIGHER engagement")
    elif salience_scores['total'] < 0.209:
        print("⚠️  Larger model shows LOWER engagement (interesting!)")
    else:
        print("➡️  Same engagement level")
    print()

    return session

if __name__ == "__main__":
    session = main()
