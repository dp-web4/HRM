"""
Interactive Conversation with Conversational Learning

Simple CLI interface for having philosophical conversations with the model.
- Tracks exchanges with SNARC salience scoring
- Stores valuable interactions for later training
- Clean, focused interface

Usage:
    python interactive_conversation.py [--model PATH] [--threshold FLOAT]
"""

import sys
import argparse
from pathlib import Path

from conversation_manager import ConversationManager


def main():
    parser = argparse.ArgumentParser(description="Interactive conversation with learning")
    parser.add_argument(
        "--model",
        default="../threshold_models/60examples_model/final_model",
        help="Path to model adapter"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Salience threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--no-irp",
        action="store_true",
        help="Disable IRP (faster but lower quality)"
    )

    args = parser.parse_args()

    # Initialize manager
    print("\nInitializing conversation manager...")
    manager = ConversationManager(
        model_path=args.model,
        salience_threshold=args.threshold
    )

    # Start session
    session_id = manager.start_session()

    print("\n" + "="*60)
    print("CONVERSATIONAL LEARNING SESSION")
    print("="*60)
    print("\nI'm ready to discuss consciousness, phenomenology, and")
    print("epistemic questions. I'll learn from our conversation!")
    print("\nCommands:")
    print("  'exit' or 'quit' - End session")
    print("  'stats' - Show session statistics")
    print("="*60 + "\n")

    exchange_count = 0

    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                break

            if user_input.lower() == 'stats':
                print(f"\nSession Statistics:")
                print(f"  Total exchanges: {manager.current_session.total_exchanges}")
                print(f"  Salient exchanges: {manager.current_session.salient_exchanges}")
                if manager.current_session.total_exchanges > 0:
                    pct = (manager.current_session.salient_exchanges /
                           manager.current_session.total_exchanges * 100)
                    print(f"  Salience rate: {pct:.1f}%")
                if manager.session_exchanges:
                    avg = sum(s['total'] for _, s in manager.session_exchanges) / len(manager.session_exchanges)
                    print(f"  Average salience: {avg:.3f}")
                continue

            # Generate response
            use_irp = not args.no_irp
            response, irp_info = manager.generate_response(
                user_input,
                use_irp=use_irp
            )

            # Display response
            print(f"\nModel: {response}")

            # Show IRP info if used
            if irp_info:
                final_energy = irp_info['iterations'][-1]['energy']
                converged = irp_info['converged']
                print(f"\n[IRP: E={final_energy:.3f}, converged={converged}]")

            # Record exchange
            scores = manager.record_exchange(user_input, response, irp_info)

            exchange_count += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # End session
        print("\n\nEnding session...")
        session = manager.end_session()

        print(f"\nConversation complete!")
        print(f"Salient exchanges saved for learning: {session.salient_exchanges}")

        if session.salient_exchanges > 0:
            print(f"\nTo train on this conversation, run:")
            print(f"  python sleep_trainer.py")
        else:
            print(f"\nNo salient exchanges to train on.")


if __name__ == "__main__":
    main()
