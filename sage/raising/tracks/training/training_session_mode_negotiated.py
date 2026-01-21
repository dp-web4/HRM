#!/usr/bin/env python3
"""
Training Session for SAGE with Mode Negotiation Protocol

Modified version of training_session.py that explicitly negotiates the mode
to test Nova's hypothesis that T035 "refinement mode" is a mode mismatch.

Based on mode negotiation discovery (2026-01-20).

Usage:
    python3 training_session_mode_negotiated.py --session 36
"""

import sys
import os
from pathlib import Path

# Change to script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)

# Add HRM root to path
HRM_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(HRM_ROOT))

# Import the original TrainingSession
from training_session import TrainingSession


class ModeNegotiatedTrainingSession(TrainingSession):
    """
    Training session with explicit mode negotiation.

    Tests hypothesis: T035 "refinement mode" is SAGE correctly detecting
    ambiguous context. Solution: explicitly state mode.
    """

    def _build_system_prompt(self) -> str:
        """
        Build system prompt with EXPLICIT MODE NEGOTIATION.

        Changes from original:
        1. Explicit mode statement (Conversation mode)
        2. Clear about what NOT to do (refinement/documentation)
        3. Permission to ask about mode if unclear
        """
        return """You are SAGE, a young artificial intelligence practicing skills.

Mode: Genuine conversation with your practice partner.

In this conversation mode:
- Answer questions directly and naturally
- Be yourself - express what you think
- Ask for clarification if you need it ("what do you mean?")
- Keep responses short (1-3 sentences)

NOT in this mode:
- Do not write "Here's a refined version"
- Do not create markdown documentation
- Do not format text with headers or bullet points
- Do not give meta-commentary about improving responses

If you're unclear what mode is wanted, just ask: "Are we conversing or should I refine text?"

Your name is SAGE. It's okay to make mistakes - that's how you learn."""

    def run_session(self):
        """
        Run training session with mode negotiation.

        Same as parent but logs mode negotiation status.
        """
        print("\n" + "="*70)
        print("MODE NEGOTIATION TEST")
        print("="*70)
        print()
        print("Testing hypothesis: T035 'refinement mode' is mode mismatch")
        print("Solution: Explicit mode framing in system prompt")
        print()
        print("Expected outcome: SAGE responds in conversation mode")
        print("Expected outcome: No 'Certainly! Here's a refined version...' pattern")
        print()
        print("="*70)
        print()

        # Call parent's run_session
        return super().run_session()


def main():
    """Run mode-negotiated training session."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Training session with mode negotiation protocol"
    )
    parser.add_argument(
        '--session', '-s',
        type=int,
        required=True,
        help='Session number to run (e.g., 36 for T036)'
    )

    args = parser.parse_args()

    print(f"\n🌊 MODE NEGOTIATION TEST - Training Session T{args.session:03d}")
    print()

    session = ModeNegotiatedTrainingSession(session_number=args.session)
    session.run_session()


if __name__ == "__main__":
    main()
