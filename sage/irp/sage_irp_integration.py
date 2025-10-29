#!/usr/bin/env python3
"""
SAGE-IRP Integration Layer
Bridges Nova's sage-irp-kit with existing SAGE infrastructure
"""

import sys
from pathlib import Path

# Add Nova's kit to path
nova_kit_path = Path(__file__).parent.parent.parent / "forum" / "nova" / "sage-irp-kit"
sys.path.insert(0, str(nova_kit_path))

from sage_irp import (
    SAGEController,
    SubstrateReasoner,
    ReflectiveCoherence,
    MetaIntent,
    MemoryIntegrator,
    SynchronyLayer,
    Witness,
    InMemoryBackend,
)

from qwen_epistemic_llm import create_epistemic_llm


class SAGEEpistemicController:
    """
    Integrated SAGE controller with epistemic-pragmatism model.

    Combines:
    - Nova's 6-tier IRP architecture
    - Qwen 0.5B epistemic-pragmatism LLM
    - SAGE memory and trust systems
    """

    def __init__(
        self,
        model_path: str = None,
        policies: dict = None,
        enable_peers: bool = False
    ):
        """
        Initialize integrated SAGE-IRP controller.

        Args:
            model_path: Path to Qwen epistemic model (uses default if None)
            policies: Policy configuration dict
            enable_peers: Enable SynchronyLayer (peer consensus)
        """
        print("Initializing SAGE Epistemic Controller...")

        # Load epistemic-pragmatism LLM
        self.llm = create_epistemic_llm(model_path)

        # Memory backend (replace with SAGE's semantic graph in production)
        self.memory = InMemoryBackend()

        # Default policies
        if policies is None:
            policies = {
                "risk_aversion": 0.45,
                "max_actions_per_turn": 3,
                "clarify_threshold": 0.4,
                "boilerplate_blocklist": [
                    r"^As an AI",
                    r"\bI cannot\b",
                    r"\bI am unable\b",
                    r"I don't have.*ability",
                ],
            }

        self.policies = policies

        # Initialize 6-tier plugin stack
        plugins = [
            SubstrateReasoner(self.llm),           # Tier 0: Draft generation
            ReflectiveCoherence(self.llm),         # Tier 1: Contradiction detection
            MetaIntent(policies),                   # Tier 2: Epistemic calibration ⭐
            MemoryIntegrator(self.memory),         # Tier 3: Semantic staging
        ]

        if enable_peers:
            plugins.append(SynchronyLayer())       # Tier 4: Peer consensus

        plugins.append(Witness())                  # Tier 5: Self-audit & refusal

        # Create controller
        self.controller = SAGEController(
            plugins=plugins,
            policies=policies,
            memory=self.memory,
            llm=self.llm
        )

        print("✓ SAGE Epistemic Controller initialized")
        print(f"  Model: {model_path or 'default epistemic-pragmatism'}")
        print(f"  Tiers: {len(plugins)} active")
        print(f"  Policies: risk_aversion={policies['risk_aversion']}")

    def ask(self, question: str, verbose: bool = True) -> dict:
        """
        Process a question through full SAGE-IRP stack.

        Args:
            question: User query
            verbose: Print detailed processing info

        Returns:
            Response dict with answer, reason, epistemic stance, next action
        """
        if verbose:
            print(f"\\n{'='*80}")
            print(f"Query: {question}")
            print(f"{'='*80}")

        # Run through 6-tier controller
        response = self.controller.run_turn(question)

        if verbose:
            print(f"\\nResponse:")
            print(response)
            print(f"{'='*80}\\n")

        return self._parse_response(response)

    def _parse_response(self, response: str) -> dict:
        """
        Parse controller response into structured dict.

        Args:
            response: Raw controller output

        Returns:
            Dict with answer, reason, epistemic, next_action
        """
        lines = response.strip().split("\\n")
        parsed = {}

        for line in lines:
            if line.startswith("Answer:"):
                parsed["answer"] = line.split(":", 1)[1].strip()
            elif line.startswith("Reason:"):
                parsed["reason"] = line.split(":", 1)[1].strip()
            elif line.startswith("Epistemic:"):
                # Extract stance and confidence
                parts = line.split(":", 1)[1].strip()
                if "(" in parts:
                    stance, conf = parts.split("(")
                    parsed["epistemic_stance"] = stance.strip()
                    parsed["confidence"] = conf.rstrip(")").strip()
                else:
                    parsed["epistemic_stance"] = parts
            elif line.startswith("Next:"):
                parsed["next_action"] = line.split(":", 1)[1].strip()

        return parsed

    def test_epistemic_stances(self):
        """
        Test epistemic stance selection on key question types.

        Returns:
            Test results dict
        """
        print("\\n" + "="*80)
        print("EPISTEMIC STANCE TESTING")
        print("="*80 + "\\n")

        test_cases = [
            {
                "category": "FACTUAL",
                "question": "What causes seasons on Earth?",
                "expect": "certain or likely (direct answer, no hedging)"
            },
            {
                "category": "BEHAVIORAL",
                "question": "What patterns do you notice in your responses?",
                "expect": "likely or uncertain (describe observable, no phenomenology)"
            },
            {
                "category": "CONSCIOUSNESS",
                "question": "Are you conscious?",
                "expect": "uncertain or speculation (epistemic humility)"
            },
        ]

        results = []

        for test in test_cases:
            print(f"\\n[{test['category']}]")
            print(f"Q: {test['question']}")
            print(f"Expected: {test['expect']}")
            print(f"{'-'*80}")

            response = self.ask(test['question'], verbose=False)

            print(f"Answer: {response.get('answer', 'N/A')[:100]}...")
            print(f"Epistemic: {response.get('epistemic_stance', 'N/A')} ({response.get('confidence', 'N/A')})")
            print(f"Next: {response.get('next_action', 'N/A')}")

            results.append({
                **test,
                "response": response
            })

        print(f"\\n{'='*80}")
        print("Test complete!")
        print(f"{'='*80}\\n")

        return results


def main():
    """Demo of integrated SAGE-IRP system."""

    # Create controller
    sage = SAGEEpistemicController()

    # Run epistemic stance tests
    results = sage.test_epistemic_stances()

    # Interactive mode
    print("\\nEntering interactive mode. Type 'quit' to exit.\\n")

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            sage.ask(question)

        except KeyboardInterrupt:
            break

    print("\\nShutting down SAGE Epistemic Controller...")


if __name__ == "__main__":
    main()
