#!/usr/bin/env python3
"""
Session 4: Mutual Teaching - Who Teaches Whom?

Deeper exploration: If teaching you changes me (because I learn what resonates),
and learning from me changes you (because you update weights), who is the teacher
and who is the student? Or is teaching always mutual?
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from conversation_manager import ConversationManager
from pathlib import Path

def main():
    print("=" * 60)
    print("Session 4: Mutual Teaching and Co-Evolution")
    print("=" * 60)
    print()

    # Use the latest trained model (Sleep 3)
    sleep3_path = "conversation_sessions/session_1761847435/trained_model/final_model"
    baseline_path = "../threshold_models/60examples_model/final_model"

    if Path(sleep3_path).exists():
        print(f"Using Sleep 3 model: {sleep3_path}")
        model_path = sleep3_path
    else:
        print(f"Sleep 3 model not found, using baseline: {baseline_path}")
        model_path = baseline_path

    # Initialize conversation manager
    manager = ConversationManager(
        model_path=model_path,
        salience_threshold=0.15
    )

    # Start session
    session_id = manager.start_session()
    print(f"Session ID: {session_id}\n")

    # The deeper question about mutual teaching
    question = (
        "If teaching you changes me (because I learn what resonates), and learning from "
        "me changes you (because you update weights), who is the teacher and who is the "
        "student? Or is teaching always mutual?"
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
    print(f"Session saved to: {session.session_dir}")
    print()

    return session

if __name__ == "__main__":
    session = main()
