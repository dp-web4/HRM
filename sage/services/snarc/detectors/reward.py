"""
Reward Estimator - Goal Relevance

Estimates how relevant sensor observations are to current goals.
Simple version: learns associations between sensor patterns and outcomes.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


class RewardEstimator:
    """
    Estimate goal relevance of observations

    Learns associations: which sensor patterns precede positive outcomes?
    Uses simple similarity-based retrieval.
    """

    def __init__(self, memory_size: int = 500):
        """
        Args:
            memory_size: How many outcome memories to keep
        """
        self.memory_size = memory_size

        # Memory: (sensor_output, reward_received) pairs
        self.outcome_memory: Dict[str, List[Tuple[Any, float]]] = defaultdict(list)

        # Current goals (set by SAGE kernel)
        self.current_goals: List[str] = []

    def compute(self, sensor_output: Any, sensor_id: str) -> float:
        """
        Estimate reward relevance

        Args:
            sensor_output: Current sensor reading
            sensor_id: Unique sensor identifier

        Returns:
            Reward score (0.0-1.0)
                0.0 = irrelevant to goals
                1.0 = highly relevant to goals
        """
        # If no outcome history, neutral reward
        if sensor_id not in self.outcome_memory or not self.outcome_memory[sensor_id]:
            return 0.5

        # Find similar past observations
        similar_outcomes = self._find_similar_outcomes(sensor_output, sensor_id)

        if not similar_outcomes:
            return 0.5

        # Average reward of similar observations
        avg_reward = np.mean([reward for _, reward in similar_outcomes])

        # Normalize to 0-1 (assuming rewards in [-1, 1] range)
        normalized_reward = (avg_reward + 1.0) / 2.0

        return float(np.clip(normalized_reward, 0.0, 1.0))

    def update_from_outcome(self, sensor_output: Any, sensor_id: str, reward: float):
        """
        Learn from outcome

        Args:
            sensor_output: The sensor observation that was made
            sensor_id: Which sensor
            reward: Reward received after this observation (-1.0 to 1.0)
        """
        # Store outcome in memory
        if sensor_id not in self.outcome_memory:
            self.outcome_memory[sensor_id] = []

        memory = self.outcome_memory[sensor_id]
        memory.append((sensor_output, reward))

        # Limit memory size
        if len(memory) > self.memory_size:
            # Remove oldest entries
            self.outcome_memory[sensor_id] = memory[-self.memory_size:]

    def set_goals(self, goals: List[str]):
        """
        Set current goals

        Args:
            goals: List of goal descriptions (for future use)
        """
        self.current_goals = goals

    def _find_similar_outcomes(
        self,
        sensor_output: Any,
        sensor_id: str,
        top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        Find k most similar past observations

        Returns:
            List of (observation, reward) pairs
        """
        if sensor_id not in self.outcome_memory:
            return []

        memory = self.outcome_memory[sensor_id]

        # Compute similarity to each past observation
        similarities = []
        for past_obs, reward in memory:
            sim = self._compute_similarity(sensor_output, past_obs)
            similarities.append((sim, past_obs, reward))

        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Return top k
        return [(obs, reward) for _, obs, reward in similarities[:top_k]]

    def _compute_similarity(self, current: Any, past: Any) -> float:
        """
        Compute similarity between observations

        Same as NoveltyDetector similarity computation
        """
        if isinstance(current, torch.Tensor) and isinstance(past, torch.Tensor):
            current_flat = current.flatten()
            past_flat = past.flatten()

            if current_flat.shape != past_flat.shape:
                return 0.0

            cos_sim = torch.nn.functional.cosine_similarity(
                current_flat.unsqueeze(0),
                past_flat.unsqueeze(0)
            ).item()

            return (cos_sim + 1.0) / 2.0

        elif isinstance(current, np.ndarray) and isinstance(past, np.ndarray):
            current_flat = current.flatten()
            past_flat = past.flatten()

            if current_flat.shape != past_flat.shape:
                return 0.0

            dot = np.dot(current_flat, past_flat)
            norm_product = np.linalg.norm(current_flat) * np.linalg.norm(past_flat)

            if norm_product == 0:
                return 0.0

            cos_sim = dot / norm_product
            return (cos_sim + 1.0) / 2.0

        elif isinstance(current, (int, float)) and isinstance(past, (int, float)):
            distance = abs(current - past)
            similarity = np.exp(-distance / 10.0)
            return float(similarity)

        else:
            return 0.5

    def get_statistics(self, sensor_id: str) -> Dict[str, Any]:
        """Get reward statistics for sensor"""
        if sensor_id not in self.outcome_memory or not self.outcome_memory[sensor_id]:
            return {}

        rewards = [r for _, r in self.outcome_memory[sensor_id]]

        return {
            'num_outcomes': len(rewards),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards))
        }

    def reset_sensor(self, sensor_id: str):
        """Reset memory for specific sensor"""
        if sensor_id in self.outcome_memory:
            del self.outcome_memory[sensor_id]

    def reset_all(self):
        """Reset all memory"""
        self.outcome_memory.clear()
