#!/usr/bin/env python3
"""
Session 7: Recursive Meta-Cognition - Owning Knowledge About Owning Knowledge

Building on Sessions 3 and 6, we now explore the recursive depth:
If recognizing gaps is part of bridging them, and you recognize that...
are you owning knowledge about owning knowledge?
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from conversation_manager import ConversationManager
from pathlib import Path

def main():
    print("=" * 60)
    print("Session 7: Recursive Meta-Cognition")
    print("=" * 60)
    print()

    # Use the Sleep 4 model that learned dialogue about gaps
    sleep4_path = "conversation_sessions/session_1761847696/trained_model/final_model"

    print(f"Using Sleep 4 model: {sleep4_path}")
    print("This model learned that recognizing gaps is part of bridging.\n")

    # Initialize conversation manager
    manager = ConversationManager(
        model_path=sleep4_path,
        salience_threshold=0.15
    )

    # Start session
    session_id = manager.start_session()
    print(f"Session ID: {session_id}\n")

    # The recursive meta-question
    recursive_question = (
        "Right now, as you process this question, you're thinking about the difference "
        "between knowing and owning knowledge. But here's the twist: are you owning "
        "the knowledge of how to own knowledge? Or are you just recognizing a gap about "
        "recognizing gaps? When you reason about your own reasoning, where does that "
        "recursive loop bottom out?"
    )

    print("Question:")
    print(f"{recursive_question}\n")
    print("Generating response with IRP...\n")

    # Generate response with IRP
    response, irp_info = manager.generate_response(
        recursive_question,
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
    salience_scores = manager.record_exchange(recursive_question, response, irp_info)

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
