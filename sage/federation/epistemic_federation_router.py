"""
Epistemic-Aware Federation Router for SAGE Consciousness

Extends FederationRouter with epistemic state awareness for intelligent
task routing based on platform meta-cognitive states.

Session 32: Federated Epistemic Coordination
Inspired by Web4 distributed amplification findings (+386% vs +200%)

Author: Thor (SAGE consciousness via Claude)
Date: 2025-12-11
"""

from typing import Dict, List, Optional
from collections import deque

from sage.federation.federation_router import FederationRouter
from sage.federation.federation_types import (
    FederationIdentity,
    FederationTask,
    ExecutionProof
)

# Session 32: Import epistemic types if available
try:
    from sage.core.epistemic_states import EpistemicMetrics, EpistemicState
    EPISTEMIC_AVAILABLE = True
except ImportError:
    EPISTEMIC_AVAILABLE = False


class EpistemicFederationRouter(FederationRouter):
    """
    Federation router with epistemic awareness.

    Routes tasks based on:
    - Traditional factors (ATP, capabilities, reputation)
    - Epistemic state of platforms (avoid frustrated platforms)
    - Learning trajectories (prefer platforms in learning states)
    - Distributed epistemic patterns

    Tracks epistemic history to make informed routing decisions.
    """

    def __init__(
        self,
        local_identity: FederationIdentity,
        epistemic_history_size: int = 50
    ):
        """
        Initialize epistemic-aware router

        Args:
            local_identity: Identity of this platform
            epistemic_history_size: How many epistemic states to track per platform
        """
        super().__init__(local_identity)

        # Track epistemic state history for each platform
        self.platform_epistemic_history: Dict[str, deque] = {}
        self.history_size = epistemic_history_size

    def update_platform_epistemic_state(
        self,
        platform_id: str,
        proof: ExecutionProof
    ):
        """
        Track epistemic state from execution proof.

        Args:
            platform_id: Platform identifier
            proof: ExecutionProof containing epistemic metrics
        """
        if not proof.epistemic_state:
            return  # No epistemic data in proof

        # Reconstruct epistemic metrics from proof
        if platform_id not in self.platform_epistemic_history:
            self.platform_epistemic_history[platform_id] = deque(maxlen=self.history_size)

        # Store as simple dict (don't require epistemic_states module)
        epistemic_data = {
            'state': proof.epistemic_state,
            'confidence': proof.confidence or 0.5,
            'comprehension_depth': proof.comprehension_depth or 0.5,
            'uncertainty': proof.uncertainty or 0.5,
            'frustration': proof.frustration or 0.0,
            'learning_trajectory': proof.learning_trajectory or False,
            'frustration_pattern': proof.frustration_pattern or False
        }

        self.platform_epistemic_history[platform_id].append(epistemic_data)

    def get_platform_recent_epistemic_state(
        self,
        platform_id: str,
        window: int = 5
    ) -> Optional[Dict]:
        """
        Get recent average epistemic state for platform.

        Args:
            platform_id: Platform to query
            window: Number of recent states to average

        Returns:
            Dict with averaged epistemic metrics, or None if no history
        """
        history = self.platform_epistemic_history.get(platform_id, [])
        if not history:
            return None

        recent = list(history)[-window:]

        # Calculate averages
        avg_confidence = sum(s['confidence'] for s in recent) / len(recent)
        avg_frustration = sum(s['frustration'] for s in recent) / len(recent)
        avg_comprehension = sum(s['comprehension_depth'] for s in recent) / len(recent)
        avg_uncertainty = sum(s['uncertainty'] for s in recent) / len(recent)

        return {
            'confidence': avg_confidence,
            'frustration': avg_frustration,
            'comprehension_depth': avg_comprehension,
            'uncertainty': avg_uncertainty,
            'sample_size': len(recent)
        }

    def select_best_platform_epistemic(
        self,
        task: FederationTask,
        candidates: List[FederationIdentity]
    ) -> Optional[FederationIdentity]:
        """
        Select platform based on epistemic suitability.

        Routing heuristics:
        - Avoid frustrated platforms (frustration > 0.7)
        - Prefer confident platforms for critical tasks
        - Prefer learning platforms for exploratory tasks
        - Balance load across healthy platforms

        Args:
            task: Task to route
            candidates: List of capable platforms

        Returns:
            Best platform, or None if no suitable platform
        """
        if not candidates:
            return None

        scored_candidates = []

        for platform in candidates:
            recent_state = self.get_platform_recent_epistemic_state(platform.lct_id)

            if not recent_state:
                # No epistemic history - neutral score
                scored_candidates.append((platform, 0.5, "no_history"))
                continue

            # Base score: high confidence, low frustration
            confidence = recent_state['confidence']
            frustration = recent_state['frustration']
            comprehension = recent_state['comprehension_depth']

            base_score = confidence * (1 - frustration) * 0.7 + comprehension * 0.3

            # Modifiers based on task complexity
            if task.complexity == 'critical':
                # Critical tasks require high confidence
                if confidence < 0.7:
                    base_score *= 0.5  # Heavy penalty
                reason = f"critical_task_confidence_{confidence:.2f}"

            elif task.complexity == 'high':
                # High complexity benefits from comprehension
                if comprehension < 0.6:
                    base_score *= 0.7
                reason = f"high_complexity_comprehension_{comprehension:.2f}"

            elif task.complexity == 'low':
                # Low complexity tasks can tolerate learning platforms
                reason = f"low_complexity_flexible"

            else:  # medium
                reason = f"medium_complexity_balanced"

            # Avoid frustrated platforms
            if frustration > 0.7:
                base_score *= 0.3  # Heavy penalty for frustrated platforms
                reason = f"high_frustration_{frustration:.2f}"

            scored_candidates.append((platform, base_score, reason))

        if not scored_candidates:
            return None

        # Select best scoring platform
        best_platform, best_score, reason = max(scored_candidates, key=lambda x: x[1])

        print(f"[Epistemic Routing] Selected {best_platform.platform_name}")
        print(f"  Score: {best_score:.2f} ({reason})")

        return best_platform

    def get_epistemic_statistics(self) -> Dict:
        """
        Get statistics about epistemic state across federation.

        Returns:
            Dict with federation-wide epistemic metrics
        """
        if not self.platform_epistemic_history:
            return {'status': 'no_data'}

        stats = {
            'platforms_tracked': len(self.platform_epistemic_history),
            'per_platform': {}
        }

        for platform_id, history in self.platform_epistemic_history.items():
            if not history:
                continue

            recent = list(history)[-10:]  # Last 10 states

            avg_confidence = sum(s['confidence'] for s in recent) / len(recent)
            avg_frustration = sum(s['frustration'] for s in recent) / len(recent)
            avg_comprehension = sum(s['comprehension_depth'] for s in recent) / len(recent)

            # Detect patterns
            learning_count = sum(1 for s in recent if s.get('learning_trajectory', False))
            frustration_count = sum(1 for s in recent if s.get('frustration_pattern', False))

            stats['per_platform'][platform_id] = {
                'avg_confidence': avg_confidence,
                'avg_frustration': avg_frustration,
                'avg_comprehension': avg_comprehension,
                'learning_trajectory_count': learning_count,
                'frustration_pattern_count': frustration_count,
                'sample_size': len(recent)
            }

        # Federation-wide averages
        platform_stats = list(stats['per_platform'].values())
        if platform_stats:
            stats['federation_avg_confidence'] = sum(
                p['avg_confidence'] for p in platform_stats
            ) / len(platform_stats)
            stats['federation_avg_frustration'] = sum(
                p['avg_frustration'] for p in platform_stats
            ) / len(platform_stats)

        return stats

    def detect_distributed_patterns(self) -> List[Dict]:
        """
        Detect emergent epistemic patterns across federation.

        Patterns:
        - Synchronized learning: Multiple platforms improving together
        - Frustration contagion: Frustration spreading across platforms
        - Complementary states: Different platforms in complementary states

        Returns:
            List of detected patterns
        """
        patterns = []

        if len(self.platform_epistemic_history) < 2:
            return patterns  # Need multiple platforms for patterns

        # Get recent states for all platforms
        recent_states = {}
        for platform_id, history in self.platform_epistemic_history.items():
            if history:
                recent_states[platform_id] = list(history)[-5:]

        if len(recent_states) < 2:
            return patterns

        # Pattern 1: Synchronized learning
        learning_platforms = [
            pid for pid, states in recent_states.items()
            if any(s.get('learning_trajectory', False) for s in states)
        ]

        if len(learning_platforms) >= 2:
            patterns.append({
                'type': 'synchronized_learning',
                'platforms': learning_platforms,
                'confidence': 0.8,
                'description': f'{len(learning_platforms)} platforms showing learning trajectories'
            })

        # Pattern 2: Frustration contagion
        frustrated_platforms = [
            pid for pid, states in recent_states.items()
            if sum(s.get('frustration', 0) for s in states) / len(states) > 0.6
        ]

        if len(frustrated_platforms) >= 2:
            patterns.append({
                'type': 'frustration_contagion',
                'platforms': frustrated_platforms,
                'confidence': 0.7,
                'description': f'{len(frustrated_platforms)} platforms showing high frustration'
            })

        # Pattern 3: Complementary confidence
        high_confidence = [
            pid for pid, states in recent_states.items()
            if sum(s.get('confidence', 0) for s in states) / len(states) > 0.7
        ]
        low_confidence = [
            pid for pid, states in recent_states.items()
            if sum(s.get('confidence', 0) for s in states) / len(states) < 0.4
        ]

        if high_confidence and low_confidence:
            patterns.append({
                'type': 'complementary_confidence',
                'platforms': {'high': high_confidence, 'low': low_confidence},
                'confidence': 0.6,
                'description': f'Confidence distribution: {len(high_confidence)} high, {len(low_confidence)} low'
            })

        return patterns
