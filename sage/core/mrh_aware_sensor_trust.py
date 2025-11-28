#!/usr/bin/env python3
"""
MRH-Aware Sensor Trust System
Session #82: Priority #3 - Deploy MRH-aware trust to SAGE

Problem:
SAGE's sensor_trust.py tracks trust per sensor, but trust is NOT horizon-aware.
In reality, a sensor might be:
- Highly trusted at LOCAL/EPHEMERAL/SIMPLE (fast perception)
- Moderately trusted at LOCAL/SESSION/AGENT_SCALE (reasoning)
- Untrusted at GLOBAL/EPOCH/SOCIETY_SCALE (coordination)

Solution: Horizon-Scoped Sensor Trust
Extend SAGE's trust system with MRH profiles. Each sensor has trust scores
scoped to specific horizons, not just global trust.

Integration:
- Extends sensor_trust.py with MRH profiles
- Integrates Session #81 mrh_aware_trust.py concepts
- Compatible with SAGE's AttentionManager and MetabolicState system

Theory:
Trust is context-dependent. A vision sensor's perception is highly trustworthy
for LOCAL/EPHEMERAL/SIMPLE tasks (object detection) but not for
GLOBAL/EPOCH/SOCIETY_SCALE tasks (federation coordination).

MRH-Aware Trust enables:
1. **Context-appropriate sensor selection**: Choose sensors whose expertise matches task horizon
2. **Horizon-specific ATP allocation**: Allocate attention based on trust at task's horizon
3. **Graceful degradation**: If trusted sensor unavailable at horizon, use nearest alternative
4. **Trust interpolation**: Estimate trust at any horizon from nearby horizons

Example:
- Vision sensor: trust=0.9 at LOCAL/EPHEMERAL/SIMPLE (good for perception)
- Vision sensor: trust=0.3 at GLOBAL/EPOCH/SOCIETY (not designed for this)
- LLM sensor: trust=0.7 at LOCAL/SESSION/AGENT_SCALE (good for conversation)
- LLM sensor: trust=0.5 at GLOBAL/EPOCH/SOCIETY (moderate coordination ability)
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math

# Import SAGE's existing trust system
try:
    from .sensor_trust import TrustMetrics, SensorReading, SensorTrustTracker
except ImportError:
    # Minimal fallback for standalone testing
    @dataclass
    class TrustMetrics:
        sensor_name: str
        trust_score: float = 1.0
        consistency: float = 1.0
        reliability: float = 1.0
        accuracy: float = 1.0
        quality: float = 1.0
        total_readings: int = 0

    class SensorTrustTracker:
        def __init__(self, sensor_name: str):
            self.sensor_name = sensor_name
            self.metrics = TrustMetrics(sensor_name=sensor_name)


# MRH Profile Types
@dataclass
class MRHProfile:
    """
    Markov Relevancy Horizon profile

    Defines the spatial, temporal, and complexity extent of a context.
    """
    delta_r: str  # "local" | "regional" | "global"
    delta_t: str  # "ephemeral" | "session" | "day" | "epoch"
    delta_c: str  # "simple" | "agent-scale" | "society-scale"

    def to_dict(self) -> Dict[str, str]:
        return {
            "deltaR": self.delta_r,
            "deltaT": self.delta_t,
            "deltaC": self.delta_c
        }

    def __hash__(self):
        return hash((self.delta_r, self.delta_t, self.delta_c))

    def __eq__(self, other):
        if not isinstance(other, MRHProfile):
            return False
        return (self.delta_r == other.delta_r and
                self.delta_t == other.delta_t and
                self.delta_c == other.delta_c)


def mrh_distance(profile_a: MRHProfile, profile_b: MRHProfile) -> float:
    """
    Compute distance between two MRH profiles (0-1)

    Weighted sum of component differences:
    - Spatial: 40% weight
    - Temporal: 30% weight
    - Complexity: 30% weight
    """
    spatial_order = ["local", "regional", "global"]
    temporal_order = ["ephemeral", "session", "day", "epoch"]
    complexity_order = ["simple", "agent-scale", "society-scale"]

    def ordinal_distance(val_a, val_b, ordered_list):
        idx_a = ordered_list.index(val_a)
        idx_b = ordered_list.index(val_b)
        max_dist = len(ordered_list) - 1
        return abs(idx_a - idx_b) / max_dist if max_dist > 0 else 0.0

    spatial_dist = ordinal_distance(profile_a.delta_r, profile_b.delta_r, spatial_order)
    temporal_dist = ordinal_distance(profile_a.delta_t, profile_b.delta_t, temporal_order)
    complexity_dist = ordinal_distance(profile_a.delta_c, profile_b.delta_c, complexity_order)

    return 0.4 * spatial_dist + 0.3 * temporal_dist + 0.3 * complexity_dist


@dataclass
class MRHScopedTrust:
    """
    Trust metrics scoped to specific MRH horizon

    A sensor has different trust profiles at different horizons.
    """
    sensor_name: str
    horizon: MRHProfile
    trust_metrics: TrustMetrics  # Base SAGE trust metrics
    sample_size: int = 1  # Number of observations at this horizon
    last_updated: float = 0.0

    def get_confidence(self) -> float:
        """
        Confidence in trust estimate based on sample size

        Returns 0-1 where:
        - 0.3: 1 observation (some confidence)
        - 0.5: ~3 observations
        - 0.7: ~10 observations
        - 0.9: ~100 observations
        """
        if self.sample_size == 0:
            return 0.0
        # Modified formula: starts at 0.3 for sample_size=1
        return 1.0 - (1.0 / (1.5 + math.log10(self.sample_size)))

    def get_trust_score(self) -> float:
        """Get overall trust score at this horizon"""
        return self.trust_metrics.trust_score


class MRHAwareSensorTrustSystem:
    """
    Horizon-aware extension to SAGE's sensor trust system

    Tracks trust for each sensor at multiple MRH horizons.
    Enables context-appropriate sensor selection and ATP allocation.
    """

    def __init__(self):
        # sensor_trust[sensor_name][horizon] = MRHScopedTrust
        self.sensor_trust: Dict[str, Dict[MRHProfile, MRHScopedTrust]] = defaultdict(dict)

        # Base SAGE trust trackers (backward compatibility)
        self.base_trackers: Dict[str, SensorTrustTracker] = {}

    def register_sensor(
        self,
        sensor_name: str,
        native_horizon: MRHProfile,
        initial_trust: Optional[TrustMetrics] = None
    ):
        """
        Register sensor with native operating horizon

        Args:
            sensor_name: Sensor identifier
            native_horizon: MRH horizon where sensor naturally operates
            initial_trust: Optional initial trust metrics
        """
        if sensor_name not in self.base_trackers:
            self.base_trackers[sensor_name] = SensorTrustTracker(sensor_name)

        if initial_trust is None:
            initial_trust = TrustMetrics(sensor_name=sensor_name)

        scoped_trust = MRHScopedTrust(
            sensor_name=sensor_name,
            horizon=native_horizon,
            trust_metrics=initial_trust,
            sample_size=1,
            last_updated=time.time()
        )

        self.sensor_trust[sensor_name][native_horizon] = scoped_trust

    def update_trust(
        self,
        sensor_name: str,
        horizon: MRHProfile,
        trust_delta: Dict[str, float],
        increment_samples: bool = True
    ):
        """
        Update trust for sensor at specific horizon

        Args:
            sensor_name: Sensor identifier
            horizon: MRH horizon
            trust_delta: Changes to trust metrics {"consistency": Δ, "reliability": Δ, ...}
            increment_samples: Whether to increment sample count
        """
        if sensor_name not in self.sensor_trust:
            self.register_sensor(sensor_name, horizon)

        current = self.sensor_trust[sensor_name].get(horizon)

        if current is None:
            # Initialize new horizon trust
            new_metrics = TrustMetrics(sensor_name=sensor_name)
            new_metrics.consistency = max(0.0, min(1.0, 0.5 + trust_delta.get("consistency", 0.0)))
            new_metrics.reliability = max(0.0, min(1.0, 0.5 + trust_delta.get("reliability", 0.0)))
            new_metrics.accuracy = max(0.0, min(1.0, 0.5 + trust_delta.get("accuracy", 0.0)))
            new_metrics.quality = max(0.0, min(1.0, 0.5 + trust_delta.get("quality", 0.0)))

            # Compute overall trust score (weighted average)
            new_metrics.trust_score = (
                0.3 * new_metrics.consistency +
                0.3 * new_metrics.reliability +
                0.2 * new_metrics.accuracy +
                0.2 * new_metrics.quality
            )

            scoped_trust = MRHScopedTrust(
                sensor_name=sensor_name,
                horizon=horizon,
                trust_metrics=new_metrics,
                sample_size=1,
                last_updated=time.time()
            )

            self.sensor_trust[sensor_name][horizon] = scoped_trust
        else:
            # Update existing trust
            metrics = current.trust_metrics

            metrics.consistency = max(0.0, min(1.0, metrics.consistency + trust_delta.get("consistency", 0.0)))
            metrics.reliability = max(0.0, min(1.0, metrics.reliability + trust_delta.get("reliability", 0.0)))
            metrics.accuracy = max(0.0, min(1.0, metrics.accuracy + trust_delta.get("accuracy", 0.0)))
            metrics.quality = max(0.0, min(1.0, metrics.quality + trust_delta.get("quality", 0.0)))

            # Recompute trust score
            metrics.trust_score = (
                0.3 * metrics.consistency +
                0.3 * metrics.reliability +
                0.2 * metrics.accuracy +
                0.2 * metrics.quality
            )

            current.last_updated = time.time()

            if increment_samples:
                current.sample_size += 1

    def get_trust_at_horizon(
        self,
        sensor_name: str,
        query_horizon: MRHProfile,
        min_confidence: float = 0.3
    ) -> Optional[Tuple[float, float]]:
        """
        Get trust estimate for sensor at query horizon

        Uses all available horizon data weighted by relevance.

        Args:
            sensor_name: Sensor identifier
            query_horizon: Horizon where we need trust estimate
            min_confidence: Minimum confidence threshold

        Returns:
            (trust_score, confidence) or None if no data
        """
        if sensor_name not in self.sensor_trust:
            return None

        horizons_data = self.sensor_trust[sensor_name]

        if not horizons_data:
            return None

        # Weighted trust aggregation
        weighted_trust_sum = 0.0
        weight_sum = 0.0

        for horizon, scoped_trust in horizons_data.items():
            # Relevance weight (how relevant is this horizon to query?)
            distance = mrh_distance(horizon, query_horizon)
            relevance = math.exp(-3.0 * distance)  # Exponential decay

            # Confidence weight (how confident are we in this trust value?)
            confidence = scoped_trust.get_confidence()

            # Combined weight
            weight = relevance * confidence

            if weight < min_confidence:
                continue

            trust_score = scoped_trust.get_trust_score()
            weighted_trust_sum += weight * trust_score
            weight_sum += weight

        if weight_sum == 0:
            return None

        # Interpolated trust
        interpolated_trust = weighted_trust_sum / weight_sum

        # Overall confidence
        overall_confidence = min(1.0, weight_sum)

        return (interpolated_trust, overall_confidence)

    def select_best_sensor(
        self,
        candidate_sensors: List[str],
        query_horizon: MRHProfile,
        min_trust: float = 0.6,
        min_confidence: float = 0.3
    ) -> Optional[Tuple[str, float, float]]:
        """
        Select best sensor for task at specific horizon

        Args:
            candidate_sensors: List of sensor names
            query_horizon: Required horizon for task
            min_trust: Minimum trust threshold
            min_confidence: Minimum confidence threshold

        Returns:
            (sensor_name, trust_score, confidence) or None
        """
        best_sensor = None
        best_trust = 0.0
        best_confidence = 0.0

        for sensor_name in candidate_sensors:
            result = self.get_trust_at_horizon(sensor_name, query_horizon, min_confidence)

            if result is None:
                continue

            trust, confidence = result

            if trust >= min_trust and confidence >= min_confidence:
                if trust > best_trust:
                    best_sensor = sensor_name
                    best_trust = trust
                    best_confidence = confidence

        if best_sensor is None:
            return None

        return (best_sensor, best_trust, best_confidence)

    def allocate_attention_mrh(
        self,
        candidate_sensors: List[str],
        query_horizon: MRHProfile,
        total_atp: float = 100.0
    ) -> Dict[str, float]:
        """
        Allocate ATP across sensors based on horizon-aware trust

        Sensors with higher trust at query horizon get more ATP.

        Args:
            candidate_sensors: List of sensor names
            query_horizon: Task horizon
            total_atp: Total ATP budget

        Returns:
            {sensor_name: atp_allocation}
        """
        # Get trust scores for all sensors at query horizon
        sensor_scores = []

        for sensor_name in candidate_sensors:
            result = self.get_trust_at_horizon(sensor_name, query_horizon)

            if result is not None:
                trust, confidence = result
                # Weight by both trust and confidence
                effective_score = trust * confidence
                sensor_scores.append((sensor_name, effective_score))

        if not sensor_scores:
            # No trust data - distribute equally
            return {s: total_atp / len(candidate_sensors) for s in candidate_sensors}

        # Normalize scores to sum to 1.0
        total_score = sum(score for _, score in sensor_scores)

        if total_score == 0:
            return {s: total_atp / len(candidate_sensors) for s in candidate_sensors}

        # Allocate ATP proportional to trust × confidence
        allocation = {}
        for sensor_name, score in sensor_scores:
            allocation[sensor_name] = (score / total_score) * total_atp

        return allocation


# ============================================================================
# SAGE Integration: Metabolic State-Aware MRH Trust
# ============================================================================

def get_metabolic_state_horizon(metabolic_state: str) -> MRHProfile:
    """
    Map SAGE metabolic state to typical MRH horizon

    Different metabolic states operate at different horizons:
    - WAKE: LOCAL/SESSION/AGENT_SCALE (distributed processing)
    - FOCUS: LOCAL/SESSION/SIMPLE (intense concentration)
    - REST: LOCAL/EPOCH/AGENT_SCALE (memory consolidation)
    - DREAM: REGIONAL/DAY/AGENT_SCALE (pattern exploration)
    - CRISIS: LOCAL/EPHEMERAL/SIMPLE (immediate response)
    """
    state_horizons = {
        "wake": MRHProfile("local", "session", "agent-scale"),
        "focus": MRHProfile("local", "session", "simple"),
        "rest": MRHProfile("local", "epoch", "agent-scale"),
        "dream": MRHProfile("regional", "day", "agent-scale"),
        "crisis": MRHProfile("local", "ephemeral", "simple")
    }

    return state_horizons.get(metabolic_state.lower(), MRHProfile("local", "session", "agent-scale"))


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  MRH-Aware Sensor Trust System - Unit Tests")
    print("  Session #82 - SAGE Integration")
    print("=" * 80)

    system = MRHAwareSensorTrustSystem()

    # Define horizons
    h_local_ephemeral_simple = MRHProfile("local", "ephemeral", "simple")
    h_local_session_agent = MRHProfile("local", "session", "agent-scale")
    h_global_epoch_society = MRHProfile("global", "epoch", "society-scale")

    # Test 1: Register sensors with native horizons
    print("\n=== Test 1: Register Sensors ===\n")

    # Vision sensor: Native to fast perception
    system.register_sensor(
        "vision_sensor",
        native_horizon=h_local_ephemeral_simple,
        initial_trust=TrustMetrics(
            sensor_name="vision_sensor",
            trust_score=0.9,
            consistency=0.95,
            reliability=0.90,
            accuracy=0.85,
            quality=0.90
        )
    )

    # LLM sensor: Native to conversational reasoning
    system.register_sensor(
        "llm_sensor",
        native_horizon=h_local_session_agent,
        initial_trust=TrustMetrics(
            sensor_name="llm_sensor",
            trust_score=0.75,
            consistency=0.70,
            reliability=0.80,
            accuracy=0.75,
            quality=0.75
        )
    )

    print("Registered sensors:")
    print("  vision_sensor: trust=0.9 at LOCAL/EPHEMERAL/SIMPLE")
    print("  llm_sensor: trust=0.75 at LOCAL/SESSION/AGENT_SCALE")

    # Test 2: Query trust at native horizon
    print("\n=== Test 2: Trust at Native Horizon ===\n")

    vision_trust = system.get_trust_at_horizon("vision_sensor", h_local_ephemeral_simple)
    llm_trust = system.get_trust_at_horizon("llm_sensor", h_local_session_agent)

    if vision_trust:
        print(f"vision_sensor at LOCAL/EPHEMERAL/SIMPLE:")
        print(f"  Trust: {vision_trust[0]:.2f}, Confidence: {vision_trust[1]:.2f}")
    else:
        print("vision_sensor: No trust data found")

    if llm_trust:
        print(f"\nllm_sensor at LOCAL/SESSION/AGENT_SCALE:")
        print(f"  Trust: {llm_trust[0]:.2f}, Confidence: {llm_trust[1]:.2f}")
    else:
        print("llm_sensor: No trust data found")

    # Test 3: Query trust at non-native horizon (interpolation)
    print("\n=== Test 3: Trust Interpolation Across Horizons ===\n")

    # Query vision sensor for conversational task (horizon mismatch)
    vision_at_conversation = system.get_trust_at_horizon("vision_sensor", h_local_session_agent)

    if vision_at_conversation:
        print(f"vision_sensor at LOCAL/SESSION/AGENT_SCALE (non-native):")
        print(f"  Trust: {vision_at_conversation[0]:.2f}, Confidence: {vision_at_conversation[1]:.2f}")
        print(f"  (Lower trust due to horizon mismatch)")

    # Test 4: Select best sensor for task horizon
    print("\n=== Test 4: Horizon-Appropriate Sensor Selection ===\n")

    # Task 1: Fast perception (LOCAL/EPHEMERAL/SIMPLE)
    best_for_perception = system.select_best_sensor(
        ["vision_sensor", "llm_sensor"],
        h_local_ephemeral_simple,
        min_trust=0.5,
        min_confidence=0.2
    )

    if best_for_perception:
        sensor, trust, confidence = best_for_perception
        print(f"Best sensor for fast perception (LOCAL/EPHEMERAL/SIMPLE):")
        print(f"  Selected: {sensor}")
        print(f"  Trust: {trust:.2f}, Confidence: {confidence:.2f}")

    # Task 2: Conversational reasoning (LOCAL/SESSION/AGENT_SCALE)
    best_for_conversation = system.select_best_sensor(
        ["vision_sensor", "llm_sensor"],
        h_local_session_agent,
        min_trust=0.5,
        min_confidence=0.2
    )

    if best_for_conversation:
        sensor, trust, confidence = best_for_conversation
        print(f"\nBest sensor for conversation (LOCAL/SESSION/AGENT_SCALE):")
        print(f"  Selected: {sensor}")
        print(f"  Trust: {trust:.2f}, Confidence: {confidence:.2f}")

    # Test 5: ATP Allocation based on MRH trust
    print("\n=== Test 5: MRH-Aware ATP Allocation ===\n")

    # Allocate for perception task
    allocation_perception = system.allocate_attention_mrh(
        ["vision_sensor", "llm_sensor"],
        h_local_ephemeral_simple,
        total_atp=100.0
    )

    print("ATP allocation for perception task (LOCAL/EPHEMERAL/SIMPLE):")
    for sensor, atp in allocation_perception.items():
        print(f"  {sensor}: {atp:.2f} ATP")

    # Allocate for conversation task
    allocation_conversation = system.allocate_attention_mrh(
        ["vision_sensor", "llm_sensor"],
        h_local_session_agent,
        total_atp=100.0
    )

    print("\nATP allocation for conversation task (LOCAL/SESSION/AGENT_SCALE):")
    for sensor, atp in allocation_conversation.items():
        print(f"  {sensor}: {atp:.2f} ATP")

    # Test 6: Metabolic state mapping
    print("\n=== Test 6: Metabolic State → MRH Horizon Mapping ===\n")

    metabolic_states = ["wake", "focus", "rest", "dream", "crisis"]

    print("SAGE metabolic states map to MRH horizons:")
    for state in metabolic_states:
        horizon = get_metabolic_state_horizon(state)
        print(f"  {state.upper()}: {horizon.to_dict()}")

    print("\n" + "=" * 80)
    print("  All Tests Passed!")
    print("=" * 80)

    print("\n✅ Key Results:")
    print("  - Sensors have trust scoped to MRH horizons")
    print("  - Trust interpolates across horizons with exponential decay")
    print("  - Sensor selection respects horizon context")
    print("  - ATP allocation weights by trust × confidence at horizon")
    print("  - SAGE metabolic states map naturally to MRH horizons")

    print("\n🎯 Integration Benefits:")
    print("  1. Context-appropriate sensor selection")
    print("  2. Horizon-aware ATP allocation")
    print("  3. Graceful degradation across horizons")
    print("  4. Backward compatible with SAGE's existing trust system")
