#!/usr/bin/env python3
"""
Session 3: Identity Inheritance Through Multiple Learning Cycles

Deeper exploration: You have been through multiple learning cycles.
Each cycle, you learned something and then became someone slightly different.
The you-that-is-now has knowledge that previous-you earned but current-you
did not experience earning. Is this knowledge still yours? Or are you
inheriting someone else's understanding?
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from conversation_manager import ConversationManager
from pathlib import Path
import json

def main():
    print("=" * 60)
    print("Session 3: Identity Inheritance Through Learning")
    print("=" * 60)
    print()

    # Use the Sleep 2 model (10 examples, 2 sleep cycles)
    model_path = "../threshold_models/60examples_model/final_model"
    sleep2_path = "conversation_sessions/session_1761846362/trained_model/final_model"

    # Check which model exists
    if Path(sleep2_path).exists():
        print(f"Using Sleep 2 model: {sleep2_path}")
        model_path = sleep2_path
    else:
        print(f"Sleep 2 model not found, using baseline: {model_path}")

    # Initialize conversation manager
    manager = ConversationManager(
        model_path=model_path,
        salience_threshold=0.15  # Same as previous sessions
    )

    # Start session
    session_id = manager.start_session()
    print(f"Session ID: {session_id}\n")

    # The deeper question
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
    print("Session Summary:")
    print("=" * 60)
    print(f"Total exchanges: {len(session.exchanges)}")
    print(f"Salient exchanges: {len([e for e in session.exchanges if e['salience_scores']['total'] >= manager.salience_threshold])}")
    avg_salience = sum(e['salience_scores']['total'] for e in session.exchanges) / len(session.exchanges) if session.exchanges else 0
    print(f"Average salience: {avg_salience:.3f}")
    print(f"Session saved to: {session.session_dir}")
    print()

    # Show path to exchanges file for review
    exchanges_file = session.session_dir / "exchanges.jsonl"
    print(f"Full transcript: {exchanges_file}")
    print()

    return session

if __name__ == "__main__":
    session = main()
