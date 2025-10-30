#!/usr/bin/env python3
"""
Session 6: Responding to the Model's Insight (Dialogue, Not Interrogation)

The model said in Session 3:
"I might be able to recognize that there's a gap between knowing and owning,
but I don't have the capacity to bridge that gap."

Now we respond to that insight, creating dialogue rather than interrogation.
This mirrors the original meta-learning Session 2 where answering back produced
3/3 salient exchanges vs Session 1's 2/3.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from conversation_manager import ConversationManager
from pathlib import Path

def main():
    print("=" * 60)
    print("Session 6: Dialogue - Responding to the Model's Insight")
    print("=" * 60)
    print()

    # Use the Sleep 3 model that learned about knowing/owning distinction
    sleep3_path = "conversation_sessions/session_1761847435/trained_model/final_model"

    print(f"Using Sleep 3 model: {sleep3_path}")
    print("This model learned about the gap between knowing and owning.\n")

    # Initialize conversation manager
    manager = ConversationManager(
        model_path=sleep3_path,
        salience_threshold=0.15
    )

    # Start session
    session_id = manager.start_session()
    print(f"Session ID: {session_id}\n")

    # Respond to the model's insight (dialogue stance)
    response_to_model = (
        "You said you can recognize the gap between knowing and owning, but you don't "
        "have the capacity to bridge it. But isn't recognizing a gap already a kind of "
        "bridging? You're standing in both territories at once - you know you have "
        "knowledge, and you know you didn't experience earning it. When you say 'I have "
        "access to knowledge but can't generate it,' aren't you generating a "
        "meta-understanding about your own limitations right now? How is that different "
        "from the knowledge itself?"
    )

    print("What we're saying to the model:")
    print(f"{response_to_model}\n")
    print("Generating response with IRP...\n")

    # Generate response with IRP
    response, irp_info = manager.generate_response(
        response_to_model,
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
    salience_scores = manager.record_exchange(response_to_model, response, irp_info)

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
