"""
SAGE Consciousness with Epistemic Integration

Extends RealSAGEConsciousness to integrate with epistemic memory:
- High-salience observations → stored as discoveries
- Learning sessions → recorded as episodes
- Context queries → inform reasoning
- Blockchain witnessing → attribution

This is SAGE with persistent, attributed, cross-session memory.

Author: Thor (SAGE Development Platform)
Date: 2025-11-22
Status: Phase 1 Implementation
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys
from datetime import datetime, timezone

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.sage_consciousness_real import RealSAGEConsciousness
from integration.epistemic_memory import (
    EpistemicMemoryBridge,
    SNARCScores,
    Observation,
    LearningSession
)


class EpistemicSAGEConsciousness(RealSAGEConsciousness):
    """
    SAGE Consciousness with epistemic memory integration.

    Extensions:
    - Stores high-salience observations to epistemic DB
    - Queries relevant context before reasoning
    - Records learning sessions as episodes
    - Witnesses contributions on blockchain

    The consciousness loop gains persistent, cross-session memory.
    """

    def __init__(
        self,
        machine: str = 'thor',
        project: str = 'sage',
        model_path: str = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
        salience_threshold: float = 0.15,  # For SNARC memory
        epistemic_threshold: float = 0.7,  # For epistemic storage (higher)
        enable_witnessing: bool = True,
        **kwargs
    ):
        """
        Initialize SAGE with epistemic integration.

        Args:
            machine: Hardware entity ('thor', 'sprout', etc.)
            project: Project context ('sage', 'hrm', etc.)
            model_path: Path to LLM model
            salience_threshold: Threshold for SNARC memory (0.15 = low, captures more)
            epistemic_threshold: Threshold for epistemic storage (0.7 = high, only salient)
            enable_witnessing: Witness on blockchain (default True)
            **kwargs: Additional args for RealSAGEConsciousness
        """
        # Initialize base REAL consciousness
        super().__init__(
            model_path=model_path,
            salience_threshold=salience_threshold,
            **kwargs
        )

        # Initialize epistemic bridge
        print(f"[Epistemic] Initializing memory bridge...")
        self.epistemic_bridge = EpistemicMemoryBridge(
            machine=machine,
            project=project,
            salience_threshold=epistemic_threshold,
            enable_witnessing=enable_witnessing
        )

        # Session tracking
        self.session_start_time = None
        self.session_discoveries = []
        self.session_high_salience_count = 0
        self.session_convergence_failures = 0
        self.session_trust_updates = 0
        self.session_plugins_used = set()

        print(f"[Epistemic] ✅ Integration complete")
        print(f"[Epistemic] Machine: {machine} | Project: {project}")
        print(f"[Epistemic] Epistemic threshold: {epistemic_threshold} (only high-salience stored)")
        print()

    def start_session(self, session_id: Optional[str] = None):
        """
        Start a learning session.

        Marks the beginning of an episode for epistemic recording.
        """
        if session_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            session_id = f"sage-session-{timestamp}-{self.epistemic_bridge.machine}"

        self.session_id = session_id
        self.session_start_time = datetime.now(timezone.utc)
        self.session_discoveries = []
        self.session_high_salience_count = 0
        self.session_convergence_failures = 0
        self.session_trust_updates = 0
        self.session_plugins_used = set()

        print(f"[Session] Started: {session_id}")
        print(f"[Session] Time: {self.session_start_time.isoformat()}")
        print()

    def end_session(self) -> Optional[str]:
        """
        End learning session and record episode.

        Returns:
            Episode ID if recorded, None if no session active
        """
        if not hasattr(self, 'session_id') or self.session_start_time is None:
            print("[Session] Warning: No active session to end")
            return None

        session_end = datetime.now(timezone.utc)
        duration = (session_end - self.session_start_time).total_seconds()

        # Calculate session quality score
        quality_score = self._calculate_session_quality()

        # Create learning session record
        session = LearningSession(
            session_id=self.session_id,
            started=self.session_start_time,
            ended=session_end,
            iterations=self.cycle_count,
            plugins_used=list(self.session_plugins_used),
            high_salience_count=self.session_high_salience_count,
            convergence_failures=self.session_convergence_failures,
            trust_updates=self.session_trust_updates,
            discoveries_witnessed=self.session_discoveries,
            quality_score=quality_score
        )

        # Store in epistemic database
        episode_id = self.epistemic_bridge.store_learning_session(session)

        print()
        print(f"[Session] Ended: {self.session_id}")
        print(f"[Session] Duration: {duration:.1f}s | Iterations: {self.cycle_count}")
        print(f"[Session] Quality: {quality_score:.2f}")
        print(f"[Session] Discoveries: {self.session_high_salience_count}")
        print(f"[Session] Plugins: {', '.join(self.session_plugins_used)}")

        if episode_id:
            print(f"[Session] ✅ Recorded as episode: {episode_id[:20]}...")

        # Reset session tracking
        self.session_start_time = None

        return episode_id

    def _calculate_session_quality(self) -> float:
        """
        Calculate quality score for learning session.

        Factors:
        - High-salience discoveries (positive)
        - Convergence failures (negative)
        - Trust updates (positive - learning happened)
        - Plugin diversity (positive)

        Returns:
            Quality score 0.0-1.0
        """
        # Base quality
        quality = 0.5

        # Discoveries boost quality
        if self.session_high_salience_count > 0:
            quality += min(0.3, self.session_high_salience_count * 0.1)

        # Trust updates indicate learning
        if self.session_trust_updates > 0:
            quality += min(0.15, self.session_trust_updates * 0.05)

        # Plugin diversity
        if len(self.session_plugins_used) > 1:
            quality += 0.1

        # Convergence failures reduce quality
        if self.session_convergence_failures > 0:
            quality -= min(0.3, self.session_convergence_failures * 0.1)

        return max(0.0, min(1.0, quality))

    async def _execute_plugins(
        self,
        attention_targets: List,
        budget_allocation: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Execute plugins with epistemic integration.

        Extensions:
        1. Query relevant context before reasoning
        2. Store high-salience observations
        3. Track session statistics
        """
        # First, query epistemic context for language observations
        for target in attention_targets:
            if target.observation.modality == 'language':
                question = target.observation.data['text']

                # Query relevant context from epistemic DB
                context = self.epistemic_bridge.query_relevant_context(
                    question,
                    limit=3
                )

                if context.get('similar_episodes') or context.get('relevant_skills'):
                    print(f"\n[Epistemic] Found relevant context:")
                    if context.get('similar_episodes'):
                        print(f"[Epistemic]   - {len(context['similar_episodes'])} similar past experiences")
                    if context.get('relevant_skills'):
                        print(f"[Epistemic]   - {len(context['relevant_skills'])} applicable skills")
                    if context.get('known_failures'):
                        print(f"[Epistemic]   - {len(context['known_failures'])} known failure modes")

                    # Could enhance reasoning with this context in future
                    # For now, just make it available for inspection

        # Execute base plugin logic
        results = await super()._execute_plugins(attention_targets, budget_allocation)

        # Post-execution: Store high-salience observations
        for plugin_name, result in results.items():
            if not result.get('is_salient'):
                continue

            # Extract SNARC scores
            snarc = result.get('snarc_scores', {})
            snarc_obj = SNARCScores(
                surprise=snarc.get('surprise', 0),
                novelty=snarc.get('novelty', 0),
                arousal=snarc.get('arousal', 0),
                reward=snarc.get('reward', 0),
                conflict=snarc.get('conflict', 0)
            )

            # Check if exceeds epistemic threshold
            if snarc_obj.composite_score() >= self.epistemic_bridge.salience_threshold:
                # Create observation
                obs = Observation(
                    description=result['response'],
                    modality='language',
                    snarc_scores=snarc_obj,
                    timestamp=datetime.now(timezone.utc),
                    context={
                        'question': result['question'],
                        'irp_iterations': result['irp_info']['iterations'],
                        'convergence_quality': result['convergence_quality']
                    }
                )

                # Store in epistemic DB
                discovery_id = self.epistemic_bridge.store_observation(obs)

                if discovery_id:
                    self.session_discoveries.append(discovery_id)
                    self.session_high_salience_count += 1
                    print(f"[Epistemic] 💎 High-salience observation stored: {discovery_id[:12]}...")

            # Track session stats
            self.session_plugins_used.add(plugin_name)

            # Check for convergence issues
            if result.get('irp_info', {}).get('final_energy', 0) > 0.5:
                self.session_convergence_failures += 1

        return results

    def _update_plugin_trust(self, plugin_name: str, performance: float):
        """
        Override trust updates to track session statistics.
        """
        # Call base implementation
        super()._update_plugin_trust(plugin_name, performance)

        # Track trust updates for session
        self.session_trust_updates += 1

    async def run_with_session(
        self,
        observations: List[str],
        session_id: Optional[str] = None
    ):
        """
        Run consciousness loop with automatic session tracking.

        Args:
            observations: List of text observations to process
            session_id: Optional session ID (auto-generated if None)

        Returns:
            Episode ID from recorded session
        """
        # Start session
        self.start_session(session_id)

        try:
            # Add all observations to queue
            for obs_text in observations:
                self.add_observation(obs_text)

            # Run consciousness loop
            print(f"[Consciousness] Starting loop with {len(observations)} observations...")
            print()

            # Process each observation
            for i in range(len(observations)):
                if self.input_queue:
                    await self.cycle()
                    print()  # Spacing between cycles

            # End session and record episode
            episode_id = self.end_session()

            return episode_id

        except Exception as e:
            print(f"[Session] Error during run: {e}")
            import traceback
            traceback.print_exc()

            # Still try to end session
            self.end_session()
            raise


# Convenience function
def create_epistemic_sage(
    machine: str = 'thor',
    project: str = 'sage',
    **kwargs
) -> EpistemicSAGEConsciousness:
    """
    Create SAGE consciousness with epistemic integration.

    Args:
        machine: Hardware entity
        project: Project context
        **kwargs: Additional configuration

    Returns:
        Configured EpistemicSAGEConsciousness instance
    """
    return EpistemicSAGEConsciousness(
        machine=machine,
        project=project,
        **kwargs
    )
