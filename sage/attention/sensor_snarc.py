#!/usr/bin/env python3
"""
Algorithmic Per-Sensor SNARC Implementation

Based on analysis showing conceptual vision requires:
- Algorithmic computation (not learned)
- Per-sensor instances (not global)
- Spatial/temporal structure (preserve "where")
- Immediate operation (no training)

This replaces learned PyTorch networks with direct computation.
"""

import torch
import torch.nn.functional as F
from collections import deque
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SNARCScores:
    """5-dimensional SNARC salience scores"""
    surprise: float      # Prediction error
    novelty: float       # Distance from memory
    arousal: float       # Signal intensity/variance
    conflict: float      # Cross-source disagreement (computed at fusion)
    reward: float        # External signal
    combined: float      # Weighted combination

    def to_dict(self) -> Dict[str, float]:
        return {
            'surprise': self.surprise,
            'novelty': self.novelty,
            'arousal': self.arousal,
            'conflict': self.conflict,
            'reward': self.reward,
            'combined': self.combined
        }


class SimplePredictor:
    """Simple autoregressive predictor for surprise computation"""

    def __init__(self, memory_size: int = 5):
        self.memory_size = memory_size

    def predict(self, history: list) -> torch.Tensor:
        """Predict next observation from recent history using simple average"""
        if len(history) == 0:
            return None

        # Simple average of recent observations
        recent = history[-self.memory_size:]
        return torch.stack(recent).mean(dim=0)


class SensorSNARC:
    """
    Algorithmic SNARC scoring for a specific sensor

    No learned parameters - all computations are algorithmic:
    - Surprise: prediction error from simple AR model
    - Novelty: distance from observation memory
    - Arousal: signal variance/intensity
    - Conflict: N/A for single sensor (computed at fusion level)
    - Reward: from external context

    Each sensor has its own SNARC instance with its own memory.
    """

    def __init__(
        self,
        sensor_name: str,
        memory_size: int = 1000,
        predictor_lookback: int = 5,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            sensor_name: Identifier for this sensor
            memory_size: How many past observations to remember
            predictor_lookback: How many recent observations for prediction
            device: torch device for computations
        """
        self.sensor_name = sensor_name
        self.memory_size = memory_size
        self.device = device or torch.device('cpu')

        # Observation memory for novelty computation
        self.memory = deque(maxlen=memory_size)

        # Simple predictor for surprise
        self.predictor = SimplePredictor(memory_size=predictor_lookback)

        # SNARC dimension weights (can be adjusted per sensor)
        self.weights = {
            'surprise': 0.3,
            'novelty': 0.3,
            'arousal': 0.2,
            'conflict': 0.0,  # Not used at sensor level
            'reward': 0.2
        }

    def score(
        self,
        observation: torch.Tensor,
        context: Optional[Dict] = None
    ) -> SNARCScores:
        """
        Compute SNARC scores algorithmically (no learning)

        Args:
            observation: Sensor reading (any shape)
            context: Optional context dict with 'reward' key

        Returns:
            SNARCScores with all 5 dimensions computed
        """
        context = context or {}

        # Ensure observation is on correct device
        if observation.device != self.device:
            observation = observation.to(self.device)

        # 1. SURPRISE: Prediction error
        surprise = self._compute_surprise(observation)

        # 2. NOVELTY: Distance from memory
        novelty = self._compute_novelty(observation)

        # 3. AROUSAL: Signal intensity/variance
        arousal = self._compute_arousal(observation)

        # 4. CONFLICT: N/A at sensor level (set at fusion)
        conflict = 0.0

        # 5. REWARD: From context
        reward = float(context.get('reward', 0.0))

        # Store observation for future novelty/surprise computation
        self.memory.append(observation.clone().detach().cpu())

        # Weighted combination
        combined = self._combine(surprise, novelty, arousal, reward)

        return SNARCScores(
            surprise=surprise,
            novelty=novelty,
            arousal=arousal,
            conflict=conflict,
            reward=reward,
            combined=combined
        )

    def _compute_surprise(self, observation: torch.Tensor) -> float:
        """Compute surprise as prediction error"""
        if len(self.memory) < 2:
            # Not enough history to predict
            return 0.5  # Neutral surprise

        # Get prediction from recent history
        history = [obs.to(self.device) for obs in list(self.memory)]
        predicted = self.predictor.predict(history)

        if predicted is None:
            return 0.5

        # Ensure shapes match
        if predicted.shape != observation.shape:
            # Flatten both for comparison
            predicted = predicted.flatten()
            obs_flat = observation.flatten()
            min_len = min(len(predicted), len(obs_flat))
            predicted = predicted[:min_len]
            obs_flat = obs_flat[:min_len]
        else:
            obs_flat = observation

        # Mean squared error as surprise
        mse = F.mse_loss(predicted, obs_flat).item()

        # Normalize to [0, 1] using sigmoid
        # Higher MSE = more surprise
        surprise = torch.sigmoid(torch.tensor(mse * 10.0)).item()

        return surprise

    def _compute_novelty(self, observation: torch.Tensor) -> float:
        """Compute novelty as distance from memory"""
        if len(self.memory) == 0:
            # Everything is novel at start
            return 1.0

        # Flatten observation for distance computation
        obs_flat = observation.flatten()

        # Compute distance to all past observations
        min_distance = float('inf')
        for past_obs in self.memory:
            past_flat = past_obs.to(self.device).flatten()

            # Ensure same length
            min_len = min(len(obs_flat), len(past_flat))
            obs_truncated = obs_flat[:min_len]
            past_truncated = past_flat[:min_len]

            # Cosine distance (1 - similarity)
            similarity = F.cosine_similarity(
                obs_truncated.unsqueeze(0),
                past_truncated.unsqueeze(0)
            ).item()

            distance = 1.0 - similarity
            min_distance = min(min_distance, distance)

        # Novelty is minimum distance (most similar past observation)
        # High distance = high novelty
        novelty = max(0.0, min(1.0, min_distance))

        return novelty

    def _compute_arousal(self, observation: torch.Tensor) -> float:
        """Compute arousal as signal intensity/variance"""
        # Standard deviation as proxy for intensity
        std = observation.std().item()

        # Normalize using sigmoid
        # Higher variance = higher arousal
        arousal = torch.sigmoid(torch.tensor(std * 5.0)).item()

        return arousal

    def _combine(
        self,
        surprise: float,
        novelty: float,
        arousal: float,
        reward: float
    ) -> float:
        """Weighted combination of SNARC dimensions"""
        combined = (
            surprise * self.weights['surprise'] +
            novelty * self.weights['novelty'] +
            arousal * self.weights['arousal'] +
            reward * self.weights['reward']
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, combined))

    def set_weights(self, weights: Dict[str, float]):
        """Update dimension weights for this sensor"""
        self.weights.update(weights)

    def get_memory_stats(self) -> Dict:
        """Get statistics about sensor memory"""
        return {
            'sensor_name': self.sensor_name,
            'memory_size': len(self.memory),
            'memory_capacity': self.memory_size,
            'device': str(self.device)
        }

    def clear_memory(self):
        """Clear observation memory"""
        self.memory.clear()


class SpatialSNARC(SensorSNARC):
    """
    Spatial SNARC for vision sensors

    Extends SensorSNARC to preserve spatial structure.
    Returns SNARC heatmaps matching input dimensions.
    """

    def __init__(
        self,
        sensor_name: str,
        memory_size: int = 1000,
        predictor_lookback: int = 5,
        device: Optional[torch.device] = None
    ):
        super().__init__(sensor_name, memory_size, predictor_lookback, device)

    def score_grid(
        self,
        image: torch.Tensor,
        context: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, SNARCScores]:
        """
        Compute spatial SNARC grid overlaying image

        Args:
            image: Image tensor [C, H, W] or [H, W]
            context: Optional context dict

        Returns:
            snarc_map: [5, H, W] tensor with SNARC dimensions
            global_scores: Averaged global SNARC scores
        """
        # Get global scores first
        global_scores = self.score(image, context)

        # Get spatial dimensions
        if image.ndim == 3:
            C, H, W = image.shape
        else:
            H, W = image.shape

        # Initialize spatial SNARC map [5, H, W]
        snarc_map = torch.zeros(5, H, W, device=self.device)

        # 1. SURPRISE: Spatial gradients (edge detection)
        snarc_map[0] = self._compute_spatial_surprise(image)

        # 2. NOVELTY: Use global score (spatial novelty requires more memory)
        snarc_map[1] = global_scores.novelty

        # 3. AROUSAL: Local variance
        snarc_map[2] = self._compute_spatial_arousal(image)

        # 4. CONFLICT: N/A spatially
        snarc_map[3] = 0.0

        # 5. REWARD: From context
        snarc_map[4] = global_scores.reward

        return snarc_map, global_scores

    def _compute_spatial_surprise(self, image: torch.Tensor) -> torch.Tensor:
        """Compute surprise as spatial gradients (edges are surprising)"""
        if image.ndim == 3:
            # Average across channels
            img = image.mean(dim=0)
        else:
            img = image

        # Sobel filters for gradient magnitude
        # Add batch dimension for conv2d
        img_batch = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=img.dtype, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=img.dtype, device=self.device).view(1, 1, 3, 3)

        # Compute gradients
        grad_x = F.conv2d(img_batch, sobel_x, padding=1)
        grad_y = F.conv2d(img_batch, sobel_y, padding=1)

        # Gradient magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()

        # Normalize to [0, 1]
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()

        return magnitude

    def _compute_spatial_arousal(self, image: torch.Tensor) -> torch.Tensor:
        """Compute arousal as local variance"""
        if image.ndim == 3:
            img = image.mean(dim=0)
        else:
            img = image

        # Local variance using unfold
        # Add batch and channel dimensions
        img_batch = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Use average pooling to approximate local variance
        # Higher frequency content = higher arousal
        kernel_size = 5
        padding = kernel_size // 2

        # Local mean
        local_mean = F.avg_pool2d(img_batch, kernel_size, stride=1, padding=padding)

        # Local variance approximation
        local_var = F.avg_pool2d(img_batch**2, kernel_size, stride=1, padding=padding) - local_mean**2
        local_std = torch.sqrt(torch.clamp(local_var, min=0))

        arousal = local_std.squeeze()

        # Normalize to [0, 1]
        if arousal.max() > 0:
            arousal = arousal / arousal.max()

        return arousal


class HierarchicalSNARC:
    """
    Hierarchical SNARC integration

    Level 1: Per-sensor spatial SNARC (local salience)
    Level 2: Per-modality aggregation (e.g., stereo vision fusion)
    Level 3: Cross-modal comparison (vision vs audio vs motor)

    Conflict emerges at cross-modal level.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        self.sensor_snarcs: Dict[str, SensorSNARC] = {}

    def register_sensor(self, sensor_name: str, snarc: SensorSNARC):
        """Register a sensor SNARC instance"""
        self.sensor_snarcs[sensor_name] = snarc

    def score_all(
        self,
        observations: Dict[str, torch.Tensor],
        context: Optional[Dict] = None
    ) -> Dict[str, SNARCScores]:
        """
        Score all sensors and compute cross-modal salience

        Args:
            observations: Dict mapping sensor names to observations
            context: Optional context dict

        Returns:
            Dict mapping sensor names to SNARC scores
        """
        # Level 1: Per-sensor scores
        sensor_scores = {}
        for sensor_name, obs in observations.items():
            if sensor_name in self.sensor_snarcs:
                snarc = self.sensor_snarcs[sensor_name]
                scores = snarc.score(obs, context)
                sensor_scores[sensor_name] = scores

        # Level 3: Compute cross-modal conflict
        conflict = self._compute_cross_modal_conflict(sensor_scores)

        # Update conflict in all sensor scores
        for scores in sensor_scores.values():
            scores.conflict = conflict
            # Recompute combined with conflict
            scores.combined = self._recompute_combined(scores)

        return sensor_scores

    def _compute_cross_modal_conflict(
        self,
        sensor_scores: Dict[str, SNARCScores]
    ) -> float:
        """
        Compute conflict as disagreement between sensors

        High conflict when sensors have very different salience.
        """
        if len(sensor_scores) < 2:
            return 0.0  # Need multiple sensors for conflict

        # Get combined scores from each sensor
        combined_scores = [scores.combined for scores in sensor_scores.values()]

        # Variance across sensors as conflict measure
        scores_tensor = torch.tensor(combined_scores, device=self.device)
        conflict = scores_tensor.var().item()

        # Normalize to [0, 1]
        conflict = min(1.0, conflict * 2.0)

        return conflict

    def _recompute_combined(self, scores: SNARCScores) -> float:
        """Recompute combined score including conflict"""
        weights = {
            'surprise': 0.25,
            'novelty': 0.25,
            'arousal': 0.15,
            'conflict': 0.15,
            'reward': 0.20
        }

        combined = (
            scores.surprise * weights['surprise'] +
            scores.novelty * weights['novelty'] +
            scores.arousal * weights['arousal'] +
            scores.conflict * weights['conflict'] +
            scores.reward * weights['reward']
        )

        return max(0.0, min(1.0, combined))
