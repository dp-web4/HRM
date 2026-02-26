#!/usr/bin/env python3
"""
Experience Salience Scoring for Attention Kernel

Lightweight salience scoring for experience atoms using algorithmic SNARC
principles adapted for kernel experience capture.

Based on sensor_snarc.py but simplified for experience-level scoring.
"""

from typing import Dict, Any, Optional, List
from collections import deque
import time
import json


class ExperienceSalienceScorer:
    """
    Compute salience scores for experience atoms

    Uses simplified SNARC dimensions:
    - Surprise: Unexpected outcomes vs expectations
    - Novelty: Dissimilarity from recent experiences
    - Arousal: Intensity of activity (budget used, plugin count)
    - Conflict: Disagreement between plugins
    - Reward: Success signals (convergence, low energy)

    All computations are algorithmic - no learned parameters.
    """

    def __init__(self, memory_size: int = 100):
        """
        Args:
            memory_size: Number of recent experiences to remember for novelty
        """
        self.memory_size = memory_size
        self.experience_memory = deque(maxlen=memory_size)

        # SNARC component weights (can be tuned but default to equal)
        self.weights = {
            'surprise': 0.25,
            'novelty': 0.25,
            'arousal': 0.20,
            'conflict': 0.15,
            'reward': 0.15
        }

    def score_experience(
        self,
        source: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> float:
        """
        Compute salience score for an experience atom

        Args:
            source: Experience source ('focus', 'think', 'act', 'sleep', etc.)
            context: Context when experience occurred
            outcome: Outcome/result of the experience

        Returns:
            Salience score in [0, 1] range
        """
        # Compute individual SNARC dimensions
        surprise = self._compute_surprise(source, context, outcome)
        novelty = self._compute_novelty(source, context, outcome)
        arousal = self._compute_arousal(source, context, outcome)
        conflict = self._compute_conflict(source, context, outcome)
        reward = self._compute_reward(source, context, outcome)

        # Weighted combination
        salience = (
            self.weights['surprise'] * surprise +
            self.weights['novelty'] * novelty +
            self.weights['arousal'] * arousal +
            self.weights['conflict'] * conflict +
            self.weights['reward'] * reward
        )

        # Store in memory for future novelty comparisons
        self.experience_memory.append({
            'source': source,
            'context_summary': self._summarize_for_memory(context),
            'outcome_summary': self._summarize_for_memory(outcome),
            'salience': salience,
            'timestamp': time.time()
        })

        return min(max(salience, 0.0), 1.0)  # Clamp to [0, 1]

    def _compute_surprise(
        self,
        source: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> float:
        """
        Compute surprise as deviation from expected outcomes

        Surprise is high when:
        - Status is 'error' or 'failed'
        - Unexpected state transitions
        - Plugin execution patterns differ from recent history
        """
        surprise = 0.0

        # Error conditions are surprising
        if isinstance(outcome, dict):
            if 'error' in outcome or outcome.get('status') == 'failed':
                surprise += 0.4

            # Sleep triggers are moderately surprising
            if source == 'sleep' or outcome.get('type') == 'sleep_trigger':
                surprise += 0.3

            # Recovery mode is very surprising
            if source == 'recover' or 'recovery' in str(outcome):
                surprise += 0.5

        return min(surprise, 1.0)

    def _compute_novelty(
        self,
        source: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> float:
        """
        Compute novelty as dissimilarity from recent experiences

        Novelty is high when:
        - Source type hasn't been seen recently
        - Outcome structure is different from recent patterns
        - New types of events or results
        """
        if len(self.experience_memory) == 0:
            return 1.0  # First experience is maximally novel

        # Count recent similar sources
        recent_sources = [exp['source'] for exp in list(self.experience_memory)[-10:]]
        source_frequency = recent_sources.count(source) / len(recent_sources)

        # Invert frequency for novelty (rare = novel)
        source_novelty = 1.0 - source_frequency

        # Check if outcome structure is novel
        outcome_str = str(type(outcome))
        recent_outcome_types = [
            str(type(exp.get('outcome_summary', {})))
            for exp in list(self.experience_memory)[-10:]
        ]
        outcome_novelty = 0.5 if outcome_str not in recent_outcome_types else 0.0

        return 0.7 * source_novelty + 0.3 * outcome_novelty

    def _compute_arousal(
        self,
        source: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> float:
        """
        Compute arousal as intensity of activity

        Arousal is high when:
        - Many plugins involved
        - High budget consumption
        - Long execution times
        - Many state changes
        """
        arousal = 0.0

        if isinstance(outcome, dict):
            # Plugin count indicates complexity
            num_plugins = len(outcome.get('results', {}))
            if num_plugins > 0:
                arousal += min(num_plugins / 5.0, 0.4)  # Cap at 5 plugins

            # Budget usage indicates intensity
            budget_used = outcome.get('total_budget_used', 0.0)
            if budget_used > 0:
                arousal += min(budget_used / 1000.0, 0.3)  # Normalized to 1000 ATP

            # Execution time indicates effort
            exec_times = outcome.get('execution_times', {})
            if exec_times:
                total_time = sum(exec_times.values())
                arousal += min(total_time / 10.0, 0.3)  # Normalized to 10 seconds

        return min(arousal, 1.0)

    def _compute_conflict(
        self,
        source: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> float:
        """
        Compute conflict as disagreement or uncertainty

        Conflict is high when:
        - Plugin results disagree
        - Low confidence
        - High disagreement scores
        - Multiple contradictory signals
        """
        conflict = 0.0

        if isinstance(outcome, dict):
            # Explicit disagreement measure
            disagreement = outcome.get('disagreement', 0.0)
            conflict += min(disagreement, 0.5)

            # Low confidence indicates conflict
            confidence = outcome.get('confidence', 1.0)
            conflict += (1.0 - confidence) * 0.3

            # Mixed success/failure across plugins
            results = outcome.get('results', {})
            if results:
                converged = sum(1 for r in results.values() if r.get('converged', False))
                total = len(results)
                convergence_rate = converged / total if total > 0 else 0
                # Conflict peaks at 50% convergence (maximum ambiguity)
                conflict += abs(0.5 - convergence_rate) * 0.2

        return min(conflict, 1.0)

    def _compute_reward(
        self,
        source: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> float:
        """
        Compute reward as success/completion signals

        Reward is high when:
        - Plugins converged successfully
        - Low final energy
        - High confidence
        - Successful status
        """
        reward = 0.0

        if isinstance(outcome, dict):
            # Successful status
            if outcome.get('status') == 'success':
                reward += 0.3

            # High confidence
            confidence = outcome.get('confidence', 0.0)
            reward += confidence * 0.3

            # Plugin convergence
            results = outcome.get('results', {})
            if results:
                converged = sum(1 for r in results.values() if r.get('converged', False))
                total = len(results)
                reward += (converged / total) * 0.2 if total > 0 else 0

            # Low final energy (plugins refined well)
            if results:
                energies = [r.get('final_energy', 1.0) for r in results.values() if 'error' not in r]
                if energies:
                    avg_energy = sum(energies) / len(energies)
                    reward += (1.0 - min(avg_energy, 1.0)) * 0.2

        return min(reward, 1.0)

    def _summarize_for_memory(self, data: Any) -> Dict[str, Any]:
        """Create lightweight summary for memory storage"""
        if isinstance(data, dict):
            return {k: type(v).__name__ for k, v in list(data.items())[:5]}
        return {'type': type(data).__name__}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about scored experiences"""
        if not self.experience_memory:
            return {
                'total_experiences': 0,
                'avg_salience': 0.0,
                'source_distribution': {}
            }

        saliences = [exp['salience'] for exp in self.experience_memory]
        sources = [exp['source'] for exp in self.experience_memory]

        source_dist = {}
        for source in set(sources):
            source_dist[source] = sources.count(source)

        return {
            'total_experiences': len(self.experience_memory),
            'avg_salience': sum(saliences) / len(saliences),
            'max_salience': max(saliences),
            'min_salience': min(saliences),
            'source_distribution': source_dist
        }
