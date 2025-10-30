#!/usr/bin/env python3
"""
Session 5: If I Taught You Something You Cannot Remember Learning

Building on the original meta-learning question from the summary:
"If I taught you something you cannot remember learning, but you can now do -
did teaching happen? Did learning happen? What is the difference?"
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from conversation_manager import ConversationManager
from pathlib import Path

def main():
    print("=" * 60)
    print("Session 5: Teaching Without Memory of Learning")
    print("=" * 60)
    print()

    # Use baseline model (60 examples) to see how it responds
    # before the meta-learning journey
    baseline_path = "../threshold_models/60examples_model/final_model"

    print(f"Using baseline model: {baseline_path}")
    print("(Testing the original meta-question)")
    print()

    # Initialize conversation manager
    manager = ConversationManager(
        model_path=baseline_path,
        salience_threshold=0.15
    )

    # Start session
    session_id = manager.start_session()
    print(f"Session ID: {session_id}\n")

    # The original meta-question from the summary
    question = (
        "If I taught you something you cannot remember learning, but you can now do - "
        "did teaching happen? Did learning happen? What is the difference?"
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
    print("Model Response:")
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

    return session

if __name__ == "__main__":
    session = main()
