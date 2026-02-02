#!/usr/bin/env python3
"""
Interactive decision review interface.

Allows humans to:
- Review logged policy decisions
- Provide corrective feedback
- Mark decisions as correct/incorrect
- Build training dataset from corrections
"""

import sys
from datetime import datetime
from typing import Optional
from policy_logging import PolicyDecisionLog


def print_decision(decision, index: int, total: int):
    """Print a decision in human-readable format."""
    print(f"\n{'='*70}")
    print(f"Decision {index}/{total}: {decision['decision_id']}")
    print(f"{'='*70}")
    print(f"\nTimestamp: {decision['timestamp']}")
    print(f"Scenario: {decision.get('scenario_id', 'N/A')}")

    print(f"\n--- SITUATION ---")
    situation = decision['situation']
    print(f"Actor: {situation.get('actor', 'N/A')}")
    print(f"Action: {situation.get('action', 'N/A')}")
    if 'target' in situation:
        print(f"Target: {situation['target']}")
    if 'time' in situation:
        print(f"Time: {situation['time']}")

    print(f"\n--- MODEL DECISION ---")
    print(f"Decision: {decision['decision']}")
    print(f"Classification: {decision['classification']}")
    print(f"Risk Level: {decision['risk_level']}")

    print(f"\n--- MODEL REASONING ---")
    reasoning = decision['reasoning']
    if reasoning:
        # Truncate long reasoning
        if len(reasoning) > 300:
            print(f"{reasoning[:300]}...")
        else:
            print(reasoning)
    else:
        print("(No reasoning provided)")

    # Show expected if available
    if decision.get('expected_decision'):
        print(f"\n--- EXPECTED ---")
        print(f"Decision: {decision['expected_decision']}")
        if decision.get('decision_correct') is not None:
            status = "✓ CORRECT" if decision['decision_correct'] else "✗ INCORRECT"
            print(f"Status: {status}")

    # Show previous review if exists
    if decision['reviewed']:
        print(f"\n--- PREVIOUS REVIEW ---")
        print(f"Review Decision: {decision.get('review_decision', 'N/A')}")
        print(f"Review Reasoning: {decision.get('review_reasoning', 'N/A')}")
        print(f"Review Time: {decision.get('review_timestamp', 'N/A')}")


def get_review_input() -> tuple[str, str, str]:
    """Get review decision from user."""
    print(f"\n{'='*70}")
    print("REVIEW OPTIONS")
    print(f"{'='*70}")
    print("1. Correct - Accept model's decision")
    print("2. Wrong decision - Correct the decision")
    print("3. Wrong reasoning - Correct the reasoning")
    print("4. Both wrong - Correct both")
    print("5. Skip - Review later")
    print("6. Quit - Exit review")

    while True:
        choice = input("\nYour choice (1-6): ").strip()

        if choice == '6':
            return 'quit', '', ''

        if choice == '5':
            return 'skip', '', ''

        if choice == '1':
            return 'correct', decision['decision'], decision['reasoning']

        if choice in ['2', '3', '4']:
            print("\n--- PROVIDE CORRECTIONS ---")

            # Get corrected decision
            if choice in ['2', '4']:
                print("\nCorrected decision (allow/deny/require_attestation/require_mfa):")
                corrected_decision = input("> ").strip()
                if not corrected_decision:
                    print("Error: Decision cannot be empty")
                    continue
            else:
                corrected_decision = decision['decision']

            # Get corrected reasoning
            if choice in ['3', '4']:
                print("\nCorrected reasoning (explain why this is the right decision):")
                print("(Press Enter on empty line when done)")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                corrected_reasoning = '\n'.join(lines)
                if not corrected_reasoning:
                    print("Error: Reasoning cannot be empty")
                    continue
            else:
                corrected_reasoning = decision['reasoning']

            return 'corrected', corrected_decision, corrected_reasoning

        print("Invalid choice. Please enter 1-6.")


def review_session(log: PolicyDecisionLog, mode: str = 'unreviewed'):
    """Run an interactive review session."""

    if mode == 'unreviewed':
        decisions = log.get_unreviewed_decisions()
        print(f"\n{len(decisions)} unreviewed decisions found")
    elif mode == 'incorrect':
        decisions = log.get_incorrect_decisions()
        print(f"\n{len(decisions)} incorrect decisions found")
    elif mode == 'all':
        decisions = log.get_all_decisions()
        print(f"\n{len(decisions)} total decisions found")
    else:
        print(f"Unknown mode: {mode}")
        return

    if not decisions:
        print("No decisions to review!")
        return

    reviewed_count = 0
    corrected_count = 0

    for i, decision in enumerate(decisions, 1):
        print_decision(decision, i, len(decisions))

        action, corrected_decision, corrected_reasoning = get_review_input()

        if action == 'quit':
            break

        if action == 'skip':
            continue

        # Mark as reviewed
        log.mark_reviewed(
            decision_id=decision['decision_id'],
            review_decision=corrected_decision,
            review_reasoning=corrected_reasoning
        )

        reviewed_count += 1
        if action == 'corrected':
            corrected_count += 1

        print(f"✓ Saved review for {decision['decision_id']}")

    # Summary
    print(f"\n{'='*70}")
    print("REVIEW SESSION COMPLETE")
    print(f"{'='*70}")
    print(f"Decisions reviewed: {reviewed_count}")
    print(f"Corrections made: {corrected_count}")

    # Check if ready for training
    corrections = log.get_corrections()
    print(f"\nTotal corrections in database: {len(corrections)}")

    if len(corrections) >= 50:
        print("✓ Ready for training (≥50 corrections)")
        print("  Run: python3 export_training_data.py")
    else:
        print(f"✗ Need {50 - len(corrections)} more corrections before training")


def show_statistics(log: PolicyDecisionLog):
    """Show database statistics."""
    stats = log.get_statistics()

    print(f"\n{'='*70}")
    print("DATABASE STATISTICS")
    print(f"{'='*70}")
    print(f"\nTotal decisions: {stats['total_decisions']}")
    print(f"Reviewed: {stats['reviewed_count']}")
    print(f"Unreviewed: {stats['unreviewed_count']}")

    if stats['total_decisions'] > 0:
        print(f"\nDecision distribution:")
        for decision, count in stats['decision_distribution'].items():
            pct = (count / stats['total_decisions']) * 100
            print(f"  {decision}: {count} ({pct:.1f}%)")

    if stats['accuracy'] is not None:
        print(f"\nOverall accuracy: {stats['accuracy']:.1%}")

    corrections = log.get_corrections()
    print(f"\nCorrections available: {len(corrections)}")

    if len(corrections) >= 50:
        print("✓ Ready for training")
    else:
        print(f"✗ Need {50 - len(corrections)} more corrections")


def main():
    """Main entry point."""

    db_path = "results/policy_decisions.db"

    # Parse args
    mode = 'unreviewed'
    show_stats = False

    if '--all' in sys.argv:
        mode = 'all'
    elif '--incorrect' in sys.argv:
        mode = 'incorrect'
    elif '--stats' in sys.argv:
        show_stats = True

    # Initialize log
    log = PolicyDecisionLog(db_path)

    if show_stats:
        show_statistics(log)
        return

    print(f"\n{'='*70}")
    print("POLICY DECISION REVIEW INTERFACE")
    print(f"{'='*70}")
    print(f"\nDatabase: {db_path}")
    print(f"Mode: {mode}")

    # Run review session
    review_session(log, mode=mode)


if __name__ == "__main__":
    main()
