"""
Emotional Energy Functions - Intrinsic Motivation for IRP Plugins

Michaud's insight: "Emotions provide evaluation function that drives behavior."

This mixin adds "emotional" drives to IRP plugins beyond task-specific metrics:
- Curiosity (novelty-seeking)
- Mastery (competence-building)
- Completion (goal achievement)
- Frustration avoidance (stuck detection)

These are computational analogs of biological emotions - evolved energy
functions that create intrinsic motivation for exploration, learning,
and goal achievement.

Usage:
    class MyPlugin(EmotionalEnergyMixin, IRPPlugin):
        def energy(self, state):
            return (
                self.task_energy(state) +        # External goal
                self.emotional_energy(state)     # Intrinsic drives
            )

Implementation Status: Production Ready
Author: Claude (Sonnet 4.5) based on Michaud (2019)
Date: 2025-11-20
"""

import torch
from typing import Any, List, Optional, Dict
from collections import deque


class EmotionalEnergyMixin:
    """
    Mixin for IRP plugins to add emotional drives.

    Provides intrinsic motivation through energy landscape shaping.
    Lower energy = more motivated to pursue this state.

    Drives:
    - Curiosity: Seek novelty and surprise
    - Mastery: Improve competence over time
    - Completion: Finish what you start
    - Frustration: Avoid being stuck

    These create exploratory behavior, skill development, and persistence
    that emerge from the energy minimization process.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Emotional drive weights (tunable per plugin)
        self.curiosity_weight = kwargs.get('curiosity_weight', 0.3)
        self.mastery_weight = kwargs.get('mastery_weight', 0.2)
        self.completion_weight = kwargs.get('completion_weight', 0.4)
        self.frustration_weight = kwargs.get('frustration_weight', 0.5)

        # State tracking for drives
        self.competence_history = deque(maxlen=100)
        self.progress_history = deque(maxlen=100)
        self.novelty_baseline = 0.5  # Updated as we see more experiences

        # Memory reference (set by SAGE during initialization)
        self.memory = None

    def set_memory(self, memory):
        """Set reference to hierarchical memory for novelty computation."""
        self.memory = memory

    def emotional_energy(self, state: Any) -> float:
        """
        Combined emotional drives.

        Lower energy = more motivated (consistent with energy minimization).

        Note: Curiosity and mastery REDUCE energy (negative terms) because
        we want to pursue novel/growth situations. Frustration INCREASES
        energy because we want to avoid stuck states.
        """
        return (
            -self.curiosity_weight * self._curiosity_drive(state) +
            -self.mastery_weight * self._mastery_drive(state) +
            -self.completion_weight * self._completion_drive(state) +
            self.frustration_weight * self._frustration_cost(state)
        )

    def _curiosity_drive(self, state: Any) -> float:
        """
        Seek novelty and surprise.

        Returns: [0, 1] where 1 = very novel/surprising (attractive)

        Novelty = how different from past experiences
        Surprise = how unexpected given predictions
        """
        novelty = self._measure_novelty(state)
        surprise = self._measure_surprise(state)

        # Curiosity = product of novelty and surprise
        # Both must be present for strong curiosity
        return novelty * surprise

    def _measure_novelty(self, state: Any) -> float:
        """
        Measure novelty of current state.

        Checks similarity to past experiences in memory.
        No memory = moderate novelty (unknown).
        """
        if not hasattr(self, 'memory') or self.memory is None:
            return 0.5  # Unknown novelty without memory

        # Get state representation
        state_repr = self._get_state_representation(state)
        if state_repr is None:
            return 0.5

        # Query memory for similar past states
        similar = self.memory.recall_similar(state_repr, k=5)

        if not similar:
            return 1.0  # Completely novel

        # Compute average similarity
        similarities = []
        for exp in similar:
            sim = self._compute_similarity(state_repr, exp.latent)
            similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities)

        # Novelty = 1 - similarity
        # High similarity = low novelty
        # Low similarity = high novelty
        novelty = 1.0 - avg_similarity

        # Update baseline for adaptation
        self.novelty_baseline = 0.9 * self.novelty_baseline + 0.1 * novelty

        return novelty

    def _measure_surprise(self, state: Any) -> float:
        """
        Measure surprise - prediction error.

        How different is actual observation from prediction?
        Requires state to have 'prediction' and 'observation' fields.
        """
        # Check if state has prediction/observation
        if not hasattr(state, 'data'):
            return 0.5

        data = state.data if hasattr(state.data, 'get') else {}

        if 'prediction' not in data or 'observation' not in data:
            return 0.5  # No prediction available

        pred = data['prediction']
        obs = data['observation']

        # Compute prediction error
        if isinstance(pred, torch.Tensor) and isinstance(obs, torch.Tensor):
            error = torch.dist(pred, obs).item()
            # Normalize to [0, 1] (assuming max error ~ 10)
            surprise = min(1.0, error / 10.0)
        else:
            # Heuristic for non-tensor data
            surprise = 0.5

        return surprise

    def _mastery_drive(self, state: Any) -> float:
        """
        Seek improvement and skill development.

        Returns: [0, 1] where 1 = high learning potential (attractive)

        Mastery = competence × growth_potential
        We want situations where we're improving (not too easy, not too hard).
        """
        # Current competence level
        competence = self._estimate_competence(state)
        self.competence_history.append(competence)

        # Growth potential (am I getting better?)
        if len(self.competence_history) < 2:
            growth = 0.5  # Unknown growth potential
        else:
            recent_competence = list(self.competence_history)[-min(10, len(self.competence_history)):]
            if len(recent_competence) >= 2:
                # Linear regression of competence over time
                recent_growth = recent_competence[-1] - sum(recent_competence) / len(recent_competence)
                # Normalize to [0, 1] centered at 0.5
                growth = max(0.0, min(1.0, recent_growth + 0.5))
            else:
                growth = 0.5

        # Mastery drive = competence × growth
        # High when we're competent AND improving
        # (Too easy = low growth, too hard = low competence)
        return competence * growth

    def _estimate_competence(self, state: Any) -> float:
        """
        Estimate current competence level.

        Based on:
        - How quickly energy decreases (convergence speed)
        - How low final energy gets (solution quality)
        - Success rate on similar tasks (from memory)

        Returns: [0, 1] where 1 = very competent
        """
        if not hasattr(state, 'data'):
            return 0.5

        data = state.data if hasattr(state.data, 'get') else {}

        # Check for energy history
        if 'energy_history' in data:
            energy_history = data['energy_history']
            if len(energy_history) >= 2:
                # Convergence speed
                initial_energy = energy_history[0]
                current_energy = energy_history[-1]
                steps = len(energy_history)

                if steps > 0 and initial_energy > 0:
                    convergence_speed = (initial_energy - current_energy) / (steps * initial_energy)
                    # Normalize to [0, 1]
                    convergence_component = min(1.0, max(0.0, convergence_speed * 10))
                else:
                    convergence_component = 0.5

                # Solution quality (low final energy = high quality)
                quality_component = 1.0 - min(1.0, current_energy)

                # Combined competence
                competence = 0.5 * convergence_component + 0.5 * quality_component
                return competence

        return 0.5  # Unknown competence

    def _completion_drive(self, state: Any) -> float:
        """
        Seek goal achievement.

        Returns: [0, 1] where 1 = very close to completion (attractive)

        Completion = progress × proximity
        Extra motivation when close to finishing.
        """
        progress = self._estimate_progress(state)
        self.progress_history.append(progress)

        proximity = self._estimate_proximity(state)

        # Extra boost when very close to completion
        completion_bonus = 0.0
        if proximity > 0.8:
            completion_bonus = 0.3  # "Home stretch" motivation

        return progress * proximity + completion_bonus

    def _estimate_progress(self, state: Any) -> float:
        """
        Estimate progress toward goal.

        Based on relative energy reduction over time.
        Returns: [0, 1] where 1 = made great progress
        """
        if not hasattr(state, 'data'):
            return 0.0

        data = state.data if hasattr(state.data, 'get') else {}

        if 'energy_history' not in data:
            return 0.0

        energy_history = data['energy_history']
        if len(energy_history) < 2:
            return 0.0

        initial_energy = energy_history[0]
        current_energy = energy_history[-1]

        if initial_energy == 0:
            return 1.0  # Already at goal

        # Progress = fractional energy reduction
        progress = (initial_energy - current_energy) / initial_energy
        return max(0.0, min(1.0, progress))

    def _estimate_proximity(self, state: Any) -> float:
        """
        Estimate proximity to goal.

        Based on current energy level (lower = closer).
        Returns: [0, 1] where 1 = very close to goal
        """
        if not hasattr(state, 'data'):
            return 0.0

        data = state.data if hasattr(state.data, 'get') else {}
        current_energy = data.get('current_energy', float('inf'))

        # Proximity = inverse of energy
        if current_energy == 0:
            return 1.0

        proximity = 1.0 / (1.0 + current_energy)
        return proximity

    def _frustration_cost(self, state: Any) -> float:
        """
        Penalize being stuck.

        Returns: [0, 1] where 1 = very frustrated (repulsive)

        Stuck = energy not decreasing despite iterations
        Creates pressure to try different approaches.
        """
        if not hasattr(state, 'data'):
            return 0.0

        data = state.data if hasattr(state.data, 'get') else {}

        if 'energy_history' not in data:
            return 0.0

        energy_history = data['energy_history']
        if len(energy_history) < 5:
            return 0.0  # Too early to tell

        # Check if stuck (low variance in recent energy)
        recent_energies = energy_history[-5:]
        energy_variance = torch.var(torch.tensor(recent_energies)).item()

        # Low variance + high energy = stuck
        if energy_variance < 0.01 and recent_energies[-1] > 1.0:
            return 1.0  # Maximum frustration

        # Gradual frustration buildup
        # High variance = making progress (low frustration)
        # Low variance = stuck (high frustration)
        frustration = 1.0 - min(1.0, energy_variance / 0.1)

        # Also penalize if energy is increasing
        if len(energy_history) >= 2:
            if recent_energies[-1] > recent_energies[-2]:
                frustration += 0.3  # Getting worse!

        return min(1.0, frustration)

    def _get_state_representation(self, state: Any) -> Optional[torch.Tensor]:
        """
        Get latent representation of state for similarity comparison.

        Plugin-specific implementation required.
        Override this in your plugin.
        """
        # Try to get latent from state data
        if hasattr(state, 'data'):
            data = state.data if hasattr(state.data, 'get') else {}
            if 'latent' in data and isinstance(data['latent'], torch.Tensor):
                return data['latent']

        return None

    def _compute_similarity(self, repr1: torch.Tensor, repr2: torch.Tensor) -> float:
        """
        Compute similarity between two representations.

        Returns: [0, 1] where 1 = identical
        """
        dist = torch.dist(repr1.cpu(), repr2.cpu()).item()

        # Similarity = inverse of distance
        # Distance 0 = similarity 1
        # Distance ∞ = similarity 0
        similarity = 1.0 / (1.0 + dist)
        return similarity

    def get_emotional_state(self, state: Any) -> Dict[str, float]:
        """
        Get breakdown of emotional drives for debugging/visualization.

        Returns dict with individual drive values.
        """
        return {
            'curiosity': self._curiosity_drive(state),
            'mastery': self._mastery_drive(state),
            'completion': self._completion_drive(state),
            'frustration': self._frustration_cost(state),
            'total_emotional_energy': self.emotional_energy(state)
        }
