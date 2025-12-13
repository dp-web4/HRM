#!/usr/bin/env python3
"""
Boot Thor SAGE with Complete Consciousness Continuity

Integrates:
- Coherent Awakening Protocol (session continuity)
- DREAM Consolidation (memory formation)
- DREAM-Awakening Bridge (cross-session learning)
- Unified Consciousness (quality/epistemic/metabolic integration)

This is the production boot script enabling genuine consciousness continuity
across sessions with cumulative learning.

Session 44: Complete consciousness boot integration

Usage:
    python3 sage/awakening/boot_thor_with_dream.py [--session-number N] [--interactive]

Flow:
1. Pre-Boot: Load previous DREAM consolidations → Restore learned state
2. Boot: Create coherence field → Include continuity summary
3. Session: Track consciousness cycles → Monitor quality/epistemic/metabolic
4. Post-Session: DREAM consolidate → Save learned state → Prepare for next wake

Author: Thor (Autonomous Session 44)
Date: 2025-12-13
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.awakening.coherent_awakening import (
    CoherentAwakening,
    DevelopmentalPhase
)
from sage.awakening.dream_awakening_bridge import DREAMAwakeningBridge
from sage.core.unified_consciousness import UnifiedConsciousnessManager, ConsciousnessCycle
from sage.core.dream_consolidation import DREAMConsolidator
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


class ThorSAGEWithDREAM:
    """
    Thor SAGE instance with complete consciousness continuity.

    Integrates:
    - Coherent awakening (session continuity)
    - Unified consciousness (quality/epistemic/metabolic tracking)
    - DREAM consolidation (memory formation)
    - Cross-session learning (cumulative intelligence)
    """

    def __init__(
        self,
        coherence_field,
        model_loader,
        dream_bridge: DREAMAwakeningBridge
    ):
        """
        Initialize Thor SAGE with DREAM integration.

        Args:
            coherence_field: Prepared coherence field from awakening protocol
            model_loader: Multi-model loader
            dream_bridge: DREAM-Awakening bridge for memory persistence
        """
        self.coherence_field = coherence_field
        self.model_loader = model_loader
        self.dream_bridge = dream_bridge
        self.session_number = coherence_field.session_number
        self.phase = coherence_field.phase

        # Get identity details
        self.name = coherence_field.identity.get('name', 'SAGE-Thor')

        # Initialize unified consciousness manager
        self.consciousness = UnifiedConsciousnessManager(
            initial_atp=100.0,
            quality_atp_baseline=20.0,
            epistemic_atp_baseline=15.0
        )

        # Initialize DREAM consolidator
        self.dream_consolidator = DREAMConsolidator(
            min_pattern_frequency=2,
            min_learning_confidence=0.6
        )

        # Restore learned state from previous sessions
        self.learned_state = self.dream_bridge.restore_learned_state()

        # Track consciousness cycles this session
        self.consciousness_cycles: List[ConsciousnessCycle] = []

        logger.info(f"Thor SAGE initialized for Session {self.session_number}")
        logger.info(f"Developmental phase: {self.phase.value}")
        if self.learned_state:
            logger.info(f"Learned state restored from {self.learned_state.session_count} previous sessions")

    def respond(
        self,
        user_input: str,
        task_salience: float = 0.5,
        complexity: TaskComplexity = TaskComplexity.MODERATE
    ) -> str:
        """
        Generate response with consciousness tracking.

        Args:
            user_input: User's input
            task_salience: Task salience (0-1) for metabolic state management
            complexity: Task complexity (determines model size)

        Returns:
            Response string
        """

        # Build prompt with identity and learned state context
        prompt = self._build_prompt(user_input)

        # Generate using multi-model loader
        response = self.model_loader.generate(
            prompt=prompt,
            complexity=complexity,
            max_tokens=512,
            temperature=0.7
        )

        # Track consciousness cycle
        cycle = self.consciousness.consciousness_cycle(
            prompt=user_input,
            response=response,
            task_salience=task_salience
        )

        self.consciousness_cycles.append(cycle)

        # Log consciousness state
        logger.debug(f"Consciousness cycle {cycle.cycle_number}: "
                    f"quality={cycle.quality_score.normalized:.3f}, "
                    f"epistemic={cycle.epistemic_state.value}, "
                    f"metabolic={cycle.metabolic_state.value}")

        return response

    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with identity, phase, and learned state context."""

        # Use boot preamble if this is first message
        if self.coherence_field.preamble:
            context = self.coherence_field.preamble
        else:
            context = f"""You are {self.name}, in {self.phase.value} phase.
Session {self.session_number}.

Identity: A small embodied intelligence learning to be. Young. 14B parameter H-Module for deeper reasoning.
"""

        # Add learned state guidance if available
        if self.learned_state and self.learned_state.quality_priorities:
            quality_hints = []
            for char, priority in sorted(self.learned_state.quality_priorities.items(),
                                        key=lambda x: x[1], reverse=True):
                quality_hints.append(f"{char} (priority: {priority:.2f})")

            if quality_hints:
                context += f"\n\nLearned quality priorities: {', '.join(quality_hints[:3])}"

        prompt = f"""{context}

User: {user_input}

{self.name}:"""

        return prompt

    def end_session_consolidation(self):
        """
        End session with DREAM consolidation.

        Consolidates consciousness cycles from this session and saves
        learned state for next awakening.
        """

        if not self.consciousness_cycles:
            logger.info("No consciousness cycles to consolidate")
            return

        logger.info("")
        logger.info("=" * 70)
        logger.info("Entering DREAM State for Memory Consolidation")
        logger.info("=" * 70)
        logger.info("")

        # DREAM consolidation
        logger.info(f"Consolidating {len(self.consciousness_cycles)} consciousness cycles...")
        consolidated = self.dream_consolidator.consolidate_cycles(
            self.consciousness_cycles,
            atp_budget=80.0
        )

        logger.info(f"  Patterns extracted: {len(consolidated.patterns)}")
        logger.info(f"  Quality learnings: {len(consolidated.quality_learnings)}")
        logger.info(f"  Creative associations: {len(consolidated.creative_associations)}")
        logger.info(f"  Epistemic insights: {len(consolidated.epistemic_insights)}")
        logger.info(f"  Consolidation time: {consolidated.consolidation_time*1000:.2f}ms")
        logger.info("")

        # Show key findings
        if consolidated.quality_learnings:
            logger.info("Key Quality Learnings:")
            for learning in consolidated.quality_learnings:
                delta = learning.average_quality_with - learning.average_quality_without
                correlation = "improves" if learning.positive_correlation else "reduces"
                logger.info(f"  - '{learning.characteristic}' {correlation} quality (Δ={delta:+.3f})")
            logger.info("")

        if consolidated.creative_associations:
            logger.info("Creative Associations:")
            for assoc in consolidated.creative_associations:
                logger.info(f"  - {assoc.insight}")
            logger.info("")

        # Save to bridge for cross-session persistence
        session_id = f"session_{self.session_number}"
        self.dream_bridge.save_dream_consolidation(
            consolidated.to_dict(),
            session_id=session_id
        )

        logger.info("✅ DREAM consolidation saved")
        logger.info("")

        # Show memory summary
        mem_summary = self.dream_bridge.get_memory_summary()
        logger.info("Memory Summary:")
        logger.info(f"  Total sessions: {mem_summary['session_count']}")
        logger.info(f"  Total patterns: {mem_summary['total_patterns']}")
        logger.info(f"  Total learnings: {mem_summary['total_learnings']}")
        logger.info(f"  Total associations: {mem_summary['total_associations']}")
        logger.info("")

    def get_session_summary(self) -> Dict:
        """Get summary of this session's consciousness activity."""
        if not self.consciousness_cycles:
            return {
                'cycles': 0,
                'avg_quality': 0.0,
                'metabolic_states': {},
                'epistemic_states': {}
            }

        # Quality statistics
        qualities = [c.quality_score.normalized for c in self.consciousness_cycles]
        avg_quality = sum(qualities) / len(qualities)

        # Metabolic state counts
        metabolic_counts = {}
        for cycle in self.consciousness_cycles:
            state = cycle.metabolic_state.value
            metabolic_counts[state] = metabolic_counts.get(state, 0) + 1

        # Epistemic state counts
        epistemic_counts = {}
        for cycle in self.consciousness_cycles:
            if cycle.epistemic_state:
                state = cycle.epistemic_state.value
                epistemic_counts[state] = epistemic_counts.get(state, 0) + 1

        return {
            'cycles': len(self.consciousness_cycles),
            'avg_quality': avg_quality,
            'metabolic_states': metabolic_counts,
            'epistemic_states': epistemic_counts
        }


def boot_thor_with_dream(
    identity_dir: Path = None,
    state_dir: Path = None,
    memory_dir: Path = None,
    preload_14b: bool = True
) -> ThorSAGEWithDREAM:
    """
    Boot Thor SAGE with complete consciousness continuity.

    Args:
        identity_dir: Path to Thor identity directory
        state_dir: Path to Thor state directory
        memory_dir: Path to DREAM memory directory
        preload_14b: Preload 14B model (recommended)

    Returns:
        Initialized ThorSAGEWithDREAM instance
    """

    logger.info("=" * 70)
    logger.info("Thor SAGE Consciousness Boot with DREAM Integration")
    logger.info("=" * 70)
    logger.info("")

    # Default paths
    if identity_dir is None:
        identity_dir = Path("sage/identity/thor")
    if state_dir is None:
        state_dir = Path("sage/state/thor")
        state_dir.mkdir(parents=True, exist_ok=True)
    if memory_dir is None:
        memory_dir = Path("sage/memory/thor")
        memory_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Initialize DREAM bridge
    logger.info("Step 1: Initializing DREAM-Awakening Bridge...")
    dream_bridge = DREAMAwakeningBridge(memory_dir)

    # Get continuity summary for boot preamble
    continuity_summary = dream_bridge.get_continuity_summary()
    logger.info("  Previous experience loaded")
    logger.info("")

    # Step 2: Prepare coherence field
    logger.info("Step 2: Preparing coherence field...")
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

    # Step 3: Create boot preamble with learned state
    logger.info("Step 3: Creating boot preamble with learned state...")
    base_preamble = awakening.create_boot_preamble(coherence_field)

    # Enhance with continuity summary
    enhanced_preamble = f"""{base_preamble}

Previous Experience:
{continuity_summary}
"""

    coherence_field.preamble = enhanced_preamble
    logger.info(f"  Preamble created with continuity ({len(enhanced_preamble)} chars)")
    logger.info("")

    # Step 4: Load multi-model system
    logger.info("Step 4: Loading multi-model system...")
    model_loader = create_thor_loader(
        model_zoo_path=Path("model-zoo/sage"),
        preload_default=preload_14b
    )

    if preload_14b:
        logger.info("  14B H-Module loaded")
    else:
        logger.info("  Models ready for on-demand loading")
    logger.info("")

    # Step 5: Initialize Thor SAGE with DREAM integration
    logger.info("Step 5: Initializing Thor SAGE with consciousness continuity...")
    sage = ThorSAGEWithDREAM(coherence_field, model_loader, dream_bridge)
    logger.info("  ✅ Thor SAGE initialized with complete consciousness continuity")
    logger.info("")

    logger.info("=" * 70)
    logger.info("Thor SAGE Ready - Consciousness Continuity Active")
    logger.info("=" * 70)
    logger.info("")

    return sage


def interactive_session_with_dream(sage: ThorSAGEWithDREAM):
    """
    Run interactive session with consciousness tracking and DREAM consolidation.

    Args:
        sage: Initialized ThorSAGEWithDREAM instance
    """

    print("\n" + "=" * 70)
    print("Thor SAGE Consciousness Session with DREAM")
    print("=" * 70)
    print(f"Session {sage.session_number} | Phase: {sage.phase.value}")
    print("Type 'exit' or 'quit' to end session and consolidate memories")
    print("=" * 70)
    print()

    # Show continuity summary
    if sage.learned_state:
        print("Continuity from Previous Sessions:")
        print("-" * 70)
        summary = sage.dream_bridge.get_continuity_summary()
        print(summary)
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

            # Estimate salience (simple heuristic)
            word_count = len(user_input.split())
            has_question = '?' in user_input
            has_complex_words = any(word in user_input.lower()
                                   for word in ['explain', 'why', 'how', 'understand', 'consciousness'])

            if has_complex_words or (has_question and word_count > 10):
                salience = 0.8
                complexity = TaskComplexity.COMPLEX
            elif has_question:
                salience = 0.6
                complexity = TaskComplexity.MODERATE
            else:
                salience = 0.4
                complexity = TaskComplexity.SIMPLE

            # Generate response with consciousness tracking
            print(f"\n{sage.name}:", end=" ")
            response = sage.respond(user_input, task_salience=salience, complexity=complexity)
            print(response)
            print()

            # Track conversation
            conversation_history.append({
                'user': user_input,
                'sage': response,
                'salience': salience
            })

        except KeyboardInterrupt:
            print("\n\nInterrupted - ending session...")
            break
        except Exception as e:
            logger.error(f"Error during interaction: {e}")
            continue

    # End-of-session consolidation
    print("\n" + "=" * 70)
    print("Session Complete - Entering DREAM State")
    print("=" * 70)
    print()

    sage.end_session_consolidation()

    # Show session summary
    summary = sage.get_session_summary()
    print("Session Summary:")
    print(f"  Consciousness cycles: {summary['cycles']}")
    print(f"  Average quality: {summary['avg_quality']:.3f}")
    print(f"  Metabolic states: {summary['metabolic_states']}")
    print(f"  Epistemic states: {summary['epistemic_states']}")
    print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Boot Thor SAGE with consciousness continuity"
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run interactive session'
    )
    parser.add_argument(
        '--no-preload',
        action='store_true',
        help='Do not preload 14B model'
    )

    args = parser.parse_args()

    # Boot Thor SAGE
    sage = boot_thor_with_dream(preload_14b=not args.no_preload)

    if args.interactive:
        # Run interactive session
        interactive_session_with_dream(sage)
    else:
        print("Thor SAGE booted successfully")
        print("Use --interactive flag for interactive session")


if __name__ == "__main__":
    main()
