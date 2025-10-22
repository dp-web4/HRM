"""
Novelty Detector - Memory Comparison

Detects how different current observation is from past experiences.
Maintains episodic memory of sensor observations, computes novelty
as dissimilarity to all past observations.
"""

import torch
import numpy as np
from typing import List, Any, Optional
from collections import deque


class NoveltyDetector:
    """
    Detect difference from past experiences

    Maintains circular buffer of past observations,
    computes novelty as distance to nearest past experience.
    """

    def __init__(self, memory_size: int = 1000, comparison_samples: int = 50):
        """
        Args:
            memory_size: How many past observations to remember
            comparison_samples: How many recent samples to compare against
                               (for efficiency - don't compare to ALL history)
        """
        self.memory_size = memory_size
        self.comparison_samples = comparison_samples

        # Episodic memory per sensor
        self.memory: dict[str, deque] = {}

    def compute(self, sensor_output: Any, sensor_id: str) -> float:
        """
        Compute novelty of observation

        Args:
            sensor_output: Current sensor reading
            sensor_id: Unique sensor identifier

        Returns:
            Novelty score (0.0-1.0)
                0.0 = seen this before (familiar)
                1.0 = never seen anything like this (novel)
        """
        # Get or create memory for this sensor
        if sensor_id not in self.memory:
            self.memory[sensor_id] = deque(maxlen=self.memory_size)

        memory = self.memory[sensor_id]

        # If no history, everything is novel
        if len(memory) == 0:
            memory.append(sensor_output)
            return 1.0

        # Compare to recent past experiences
        samples_to_check = min(self.comparison_samples, len(memory))
        similarities = []

        for past_observation in list(memory)[-samples_to_check:]:
            similarity = self._compute_similarity(sensor_output, past_observation)
            similarities.append(similarity)

        # Novelty = 1 - max_similarity
        # (if very similar to something seen before, low novelty)
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity

        # Store current observation
        memory.append(sensor_output)

        return novelty

    def _compute_similarity(self, current: Any, past: Any) -> float:
        """
        Compute similarity between observations

        Returns value 0.0-1.0:
            0.0 = completely different
            1.0 = identical
        """
        if isinstance(current, torch.Tensor) and isinstance(past, torch.Tensor):
            # Cosine similarity for tensors
            current_flat = current.flatten()
            past_flat = past.flatten()

            if current_flat.shape != past_flat.shape:
                # Different shapes - not similar
                return 0.0

            cos_sim = torch.nn.functional.cosine_similarity(
                current_flat.unsqueeze(0),
                past_flat.unsqueeze(0)
            ).item()

            # Convert from [-1, 1] to [0, 1]
            return (cos_sim + 1.0) / 2.0

        elif isinstance(current, np.ndarray) and isinstance(past, np.ndarray):
            # Cosine similarity for numpy arrays
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
            # Inverse normalized distance for scalars
            distance = abs(current - past)
            # Map distance to similarity (closer = more similar)
            # Using exponential decay: similarity = exp(-distance)
            similarity = np.exp(-distance / 10.0)  # Normalize by expected range
            return float(similarity)

        else:
            # Unknown types - can't compare
            return 0.5  # Neutral similarity

    def get_memory_size(self, sensor_id: str) -> int:
        """Get number of stored observations for sensor"""
        if sensor_id not in self.memory:
            return 0
        return len(self.memory[sensor_id])

    def reset_sensor(self, sensor_id: str):
        """Clear memory for specific sensor"""
        if sensor_id in self.memory:
            del self.memory[sensor_id]

    def reset_all(self):
        """Clear all memory"""
        self.memory.clear()

    def get_most_novel_observation(self, sensor_id: str, n: int = 5) -> List[Any]:
        """
        Get the n most novel observations from history

        Useful for identifying interesting past experiences
        """
        if sensor_id not in self.memory:
            return []

        memory = list(self.memory[sensor_id])
        if len(memory) < 2:
            return memory

        # Compute novelty of each observation relative to others
        novelty_scores = []
        for i, observation in enumerate(memory):
            # Compare to all OTHER observations
            similarities = []
            for j, other in enumerate(memory):
                if i != j:
                    sim = self._compute_similarity(observation, other)
                    similarities.append(sim)

            # Novelty = 1 - max similarity to others
            max_sim = max(similarities) if similarities else 0.0
            novelty = 1.0 - max_sim
            novelty_scores.append((novelty, observation))

        # Sort by novelty (descending)
        novelty_scores.sort(reverse=True, key=lambda x: x[0])

        # Return top n
        return [obs for _, obs in novelty_scores[:n]]
