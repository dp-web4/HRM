"""
Temporal utilities for SNARC — time-decay weighting and time-aware normalization.

Knowledge has perishability. SNARC observations are volatile: their influence
should decay with time. This module provides the shared math for all detectors.

Half-life semantics: at exactly one half-life of elapsed time, the observation's
weight drops to 0.5. At two half-lives, 0.25. At zero elapsed, 1.0.
"""

import math
import time
from typing import List, Optional, Sequence, Tuple

# Default half-lives per detector (seconds)
DEFAULT_HALF_LIVES = {
    'surprise': 60.0,     # fast — predictions should track recent state
    'novelty': 300.0,     # medium — vocabulary novelty fades
    'arousal': 120.0,     # medium — baseline adapts
    'reward': 600.0,      # slow — outcome patterns more stable
    'conflict': 300.0,    # medium
}

LN2 = math.log(2)


def time_decay_weight(obs_time: float, now: float, half_life: float) -> float:
    """Exponential decay weight for an observation.

    Returns 1.0 when obs_time == now, 0.5 at one half-life, approaches 0.
    Clamped to [0, 1].
    """
    if half_life <= 0:
        return 1.0  # no decay
    elapsed = max(0.0, now - obs_time)
    return math.exp(-LN2 * elapsed / half_life)


def weighted_percentile(values: Sequence[float],
                        weights: Sequence[float],
                        target: float) -> float:
    """Time-weighted percentile rank of target within values.

    Each value has an associated weight (typically from time_decay_weight).
    Returns a value in [0, 1] representing what fraction of weighted
    observations fall below the target.

    If all weights are equal, this is equivalent to a standard percentile rank.
    """
    if not values or not weights:
        return 0.5

    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.5

    below_weight = sum(w for v, w in zip(values, weights) if v < target)
    return below_weight / total_weight


class TimestampedDeque:
    """A fixed-capacity buffer that stores (value, timestamp) pairs.

    Drop-in companion for existing deque-based buffers. Keeps values and
    timestamps aligned. Provides weighted iteration via time_decay_weight.
    """

    def __init__(self, maxlen: int = 100):
        self.maxlen = maxlen
        self._values: List[float] = []
        self._timestamps: List[float] = []

    def append(self, value: float, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        self._values.append(value)
        self._timestamps.append(timestamp)
        if len(self._values) > self.maxlen:
            self._values.pop(0)
            self._timestamps.pop(0)

    def weighted_values(self, half_life: float,
                        now: Optional[float] = None) -> List[Tuple[float, float]]:
        """Return [(value, weight), ...] with time-decay weights."""
        if now is None:
            now = time.time()
        return [
            (v, time_decay_weight(t, now, half_life))
            for v, t in zip(self._values, self._timestamps)
        ]

    def weighted_percentile_of(self, target: float, half_life: float,
                               now: Optional[float] = None) -> float:
        """Percentile rank of target in this buffer, weighted by recency."""
        if now is None:
            now = time.time()
        weights = [time_decay_weight(t, now, half_life)
                   for t in self._timestamps]
        return weighted_percentile(self._values, weights, target)

    @property
    def values(self) -> List[float]:
        return self._values

    @property
    def timestamps(self) -> List[float]:
        return self._timestamps

    def __len__(self):
        return len(self._values)

    def clear(self):
        self._values.clear()
        self._timestamps.clear()
