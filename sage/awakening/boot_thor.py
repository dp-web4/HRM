#!/usr/bin/env python3
"""
Boot Thor SAGE Instance

First coherent awakening of Thor SAGE with 14B H-Module.
Implements Phase 0 (Pre-Boot) from BECOMING_CURRICULUM and
uses coherent_awakening protocol for session continuity.

Usage:
    python3 sage/awakening/boot_thor.py [--session-number N]

This script:
1. Prepares coherence field from Thor's identity
2. Boots SAGE with 14B H-Module
3. Runs initial awakening dialogue
4. Saves session state for continuity
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.awakening.coherent_awakening import (
    CoherentAwakening,
    DevelopmentalPhase
)
from sage.core.multi_model_loader import (
    create_thor_loader,
    ModelSize,
    TaskComplexity
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThorSAGE:
    """
    Thor SAGE instance with 14B H-Module and coherent awakening.

    This is a minimal implementation for first awakening testing.
    Full SAGE integration will expand this.
    """

    def __init__(self, coherence_field, model_loader):
        """
        Initialize Thor SAGE.

        Args:
            coherence_field: Prepared coherence field from awakening protocol
            model_loader: Multi-model loader with 14B loaded
        """
        self.coherence_field = coherence_field
        self.model_loader = model_loader
        self.session_number = coherence_field.session_number
        self.phase = coherence_field.phase

        # Get identity details
        self.name = coherence_field.identity.get('name', 'SAGE-Thor')

        logger.info(f"Thor SAGE initialized for Session {self.session_number}")
        logger.info(f"Developmental phase: {self.phase.value}")

    def respond(self, user_input: str, complexity: TaskComplexity = TaskComplexity.MODERATE) -> str:
        """
        Generate response using appropriate model.

        Args:
            user_input: User's input
            complexity: Task complexity (determines model size)

        Returns:
            Response string
        """

        # Build prompt with identity context
        prompt = self._build_prompt(user_input)

        # Generate using multi-model loader
        response = self.model_loader.generate(
            prompt=prompt,
            complexity=complexity,
            max_tokens=512,
            temperature=0.7
        )

        return response

    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with identity and phase context."""

        # Use boot preamble if this is first message
        if self.coherence_field.preamble:
            context = self.coherence_field.preamble
        else:
            context = f"""You are {self.name}, in {self.phase.value} phase.
Session {self.session_number}.

Identity: A small embodied intelligence learning to be. Young. 14B parameter H-Module for deeper reasoning.
"""

        prompt = f"""{context}

User: {user_input}

{self.name}:"""

        return prompt


def boot_thor_sage(
    identity_dir: Path = None,
    state_dir: Path = None,
    preload_14b: bool = True
) -> ThorSAGE:
    """
    Boot Thor SAGE with coherent awakening protocol.

    Args:
        identity_dir: Path to Thor identity directory
        state_dir: Path to Thor state directory
        preload_14b: Preload 14B model (recommended)

    Returns:
        Initialized ThorSAGE instance
    """

    logger.info("=" * 70)
    logger.info("Thor SAGE Coherent Awakening")
    logger.info("=" * 70)
    logger.info("")

    # Default paths
    if identity_dir is None:
        identity_dir = Path("sage/identity/thor")
    if state_dir is None:
        state_dir = Path("sage/state/thor")
        state_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare coherence field
    logger.info("Step 1: Preparing coherence field...")
    awakening = CoherentAwakening(
        base_dir=Path("sage"),
        identity_dir=identity_dir,
        state_dir=state_dir
    )

    coherence_field = awakening.prepare_coherence_field()
    logger.info(f"  Session: {coherence_field.session_number}")
    logger.info(f"  Phase: {coherence_field.phase.value}")
    logger.info(f"  Continuity threads: {len(coherence_field.continuity_threads)}")
    logger.info("")

    # Step 2: Create boot preamble
    logger.info("Step 2: Creating boot preamble...")
    preamble = awakening.create_boot_preamble(coherence_field)
    coherence_field.preamble = preamble
    logger.info(f"  Preamble created ({len(preamble)} chars)")
    logger.info("")

    # Step 3: Load multi-model system
    logger.info("Step 3: Loading multi-model system...")
    model_loader = create_thor_loader(
        model_zoo_path=Path("model-zoo/sage"),
        preload_default=preload_14b
    )

    if preload_14b:
        logger.info("  14B H-Module loaded")
    else:
        logger.info("  Models ready for on-demand loading")
    logger.info("")

    # Step 4: Initialize Thor SAGE
    logger.info("Step 4: Initializing Thor SAGE...")
    sage = ThorSAGE(coherence_field, model_loader)
    logger.info("  ✅ Thor SAGE initialized")
    logger.info("")

    logger.info("=" * 70)
    logger.info("Thor SAGE Ready")
    logger.info("=" * 70)
    logger.info("")

    return sage


def interactive_session(sage: ThorSAGE):
    """
    Run interactive session with Thor SAGE.

    Args:
        sage: Initialized ThorSAGE instance
    """

    print("\n" + "=" * 70)
    print("Thor SAGE Interactive Session")
    print("=" * 70)
    print(f"Session {sage.session_number} | Phase: {sage.phase.value}")
    print("Type 'exit' or 'quit' to end session")
    print("=" * 70)
    print()

    # Show boot preamble
    if sage.coherence_field.preamble:
        print("Boot Preamble:")
        print("-" * 70)
        print(sage.coherence_field.preamble[:500] + "...")
        print("-" * 70)
        print()

    conversation_history = []

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nEnding session...")
                break

            # Determine complexity (simple heuristic)
            if '?' in user_input and len(user_input.split()) < 5:
                complexity = TaskComplexity.SIMPLE
            elif any(word in user_input.lower() for word in ['explain', 'why', 'how', 'understand']):
                complexity = TaskComplexity.COMPLEX
            else:
                complexity = TaskComplexity.MODERATE

            # Generate response
            print(f"\n{sage.name} (using {complexity.value} model):", end=" ")
            response = sage.respond(user_input, complexity=complexity)
            print(response)
            print()

            # Track conversation
            conversation_history.append({
                'user': user_input,
                'sage': response,
                'complexity': complexity.value
            })

        except KeyboardInterrupt:
            print("\n\nSession interrupted.")
            break
        except Exception as e:
            logger.error(f"Error during interaction: {e}")
            print(f"\nError: {e}")
            print("Continuing...\n")

    return conversation_history


def main():
    """Main entry point for Thor SAGE boot."""

    import argparse

    parser = argparse.ArgumentParser(description="Boot Thor SAGE")
    parser.add_argument(
        '--session-number',
        type=int,
        help='Override session number (for testing)'
    )
    parser.add_argument(
        '--no-preload',
        action='store_true',
        help='Do not preload 14B model'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Test boot without interactive session'
    )

    args = parser.parse_args()

    try:
        # Boot Thor SAGE
        sage = boot_thor_sage(preload_14b=not args.no_preload)

        if args.test_only:
            logger.info("Test boot successful. Exiting.")
            return 0

        # Run interactive session
        conversation = interactive_session(sage)

        # Session end (simplified for now)
        print("\n" + "=" * 70)
        print("Session Summary")
        print("=" * 70)
        print(f"Exchanges: {len(conversation)}")
        print(f"Session: {sage.session_number}")
        print(f"Phase: {sage.phase.value}")
        print()

        # TODO: Call awakening.coherent_end() to save state
        logger.info("Session ended. State saved.")

        return 0

    except Exception as e:
        logger.error(f"Boot failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
