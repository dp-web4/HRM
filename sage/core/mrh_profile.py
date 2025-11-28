"""
MRH Profile for Consciousness Operations

Markov Relevancy Horizon (MRH) profiles characterize the scope of cognitive operations
across three dimensions: Spatial (ΔR), Temporal (ΔT), and Complexity (ΔC).

Inspired by Web4 Session #81's MRH-aware trust, this module brings horizon awareness
to SAGE consciousness attention allocation.

Design rationale:
Different cognitive operations operate at different scales:
- Quick reflexes: LOCAL/EPHEMERAL/SIMPLE
- Focused reasoning: LOCAL/SESSION/AGENT_SCALE
- Long-term learning: REGIONAL/EPOCH/SOCIETY_SCALE

ATP allocation should reflect these horizon differences, just as biological brains
allocate energy differently across cognitive timescales.
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum


class SpatialExtent(Enum):
    """
    Spatial horizon (ΔR) - scope of agent interaction.

    LOCAL: Single-agent internal processing
    REGIONAL: Multi-agent interaction within society/federation subset
    GLOBAL: Entire federation, distributed consciousness
    """
    LOCAL = "local"
    REGIONAL = "regional"
    GLOBAL = "global"

    def to_scaling_factor(self) -> float:
        """Convert to ATP scaling factor (higher scale = more overhead)"""
        return {
            SpatialExtent.LOCAL: 1.0,     # Baseline (internal processing)
            SpatialExtent.REGIONAL: 1.3,  # Social cognition overhead
            SpatialExtent.GLOBAL: 1.8     # Distributed coordination overhead
        }[self]


class TemporalExtent(Enum):
    """
    Temporal horizon (ΔT) - time scale of operation.

    EPHEMERAL: Single turn, immediate response
    SESSION: Conversation or task session
    DAY: Cross-session integration
    EPOCH: Long-term memory, weeks-months
    """
    EPHEMERAL = "ephemeral"
    SESSION = "session"
    DAY = "day"
    EPOCH = "epoch"

    def to_scaling_factor(self) -> float:
        """Convert to ATP scaling factor"""
        return {
            TemporalExtent.EPHEMERAL: 0.8,  # Quick response, minimal context
            TemporalExtent.SESSION: 1.0,    # Baseline
            TemporalExtent.DAY: 1.4,        # Cross-session integration cost
            TemporalExtent.EPOCH: 2.0       # Long-term consolidation overhead
        }[self]


class ComplexityExtent(Enum):
    """
    Complexity horizon (ΔC) - operational complexity.

    SIMPLE: Single-step operations (factual recall)
    AGENT_SCALE: Multi-step reasoning (IRP iterations)
    SOCIETY_SCALE: Coordination, consensus, federation
    """
    SIMPLE = "simple"
    AGENT_SCALE = "agent-scale"
    SOCIETY_SCALE = "society-scale"

    def to_scaling_factor(self) -> float:
        """Convert to ATP scaling factor"""
        return {
            ComplexityExtent.SIMPLE: 0.7,         # Single-step, minimal overhead
            ComplexityExtent.AGENT_SCALE: 1.0,    # Baseline multi-step
            ComplexityExtent.SOCIETY_SCALE: 1.5   # Coordination overhead
        }[self]


@dataclass(frozen=True)
class MRHProfile:
    """
    Markov Relevancy Horizon profile for cognitive operations.

    Characterizes the scope of a cognitive task across three dimensions:
    - Spatial (ΔR): How many agents involved?
    - Temporal (ΔT): What time scale?
    - Complexity (ΔC): How complex is the operation?

    Frozen (immutable) to allow use as dict keys.
    """
    delta_r: SpatialExtent
    delta_t: TemporalExtent
    delta_c: ComplexityExtent

    def calculate_horizon_scaling_factor(
        self,
        spatial_weight: float = 0.40,
        temporal_weight: float = 0.30,
        complexity_weight: float = 0.30
    ) -> float:
        """
        Calculate combined ATP scaling factor from horizon dimensions.

        Weights reflect relative importance:
        - Spatial (40%): Coordination overhead dominates
        - Temporal (30%): Time scale affects resource commitment
        - Complexity (30%): Operational complexity affects processing cost

        Returns:
            Scaling factor (typically 0.7-2.0)
        """
        return (
            spatial_weight * self.delta_r.to_scaling_factor() +
            temporal_weight * self.delta_t.to_scaling_factor() +
            complexity_weight * self.delta_c.to_scaling_factor()
        )

    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary"""
        return {
            "deltaR": self.delta_r.value,
            "deltaT": self.delta_t.value,
            "deltaC": self.delta_c.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "MRHProfile":
        """Deserialize from dictionary"""
        return cls(
            delta_r=SpatialExtent(data["deltaR"]),
            delta_t=TemporalExtent(data["deltaT"]),
            delta_c=ComplexityExtent(data["deltaC"])
        )

    def __str__(self) -> str:
        """Human-readable representation"""
        return f"MRH({self.delta_r.value}/{self.delta_t.value}/{self.delta_c.value})"


# =============================================================================
# Predefined Profiles for Common Cognitive Operations
# =============================================================================

# Quick reflexive responses
PROFILE_REFLEXIVE = MRHProfile(
    delta_r=SpatialExtent.LOCAL,
    delta_t=TemporalExtent.EPHEMERAL,
    delta_c=ComplexityExtent.SIMPLE
)

# Focused reasoning and problem solving
PROFILE_FOCUSED = MRHProfile(
    delta_r=SpatialExtent.LOCAL,
    delta_t=TemporalExtent.SESSION,
    delta_c=ComplexityExtent.AGENT_SCALE
)

# Cross-session learning and pattern extraction
PROFILE_LEARNING = MRHProfile(
    delta_r=SpatialExtent.REGIONAL,
    delta_t=TemporalExtent.DAY,
    delta_c=ComplexityExtent.SOCIETY_SCALE
)

# Long-term memory consolidation
PROFILE_CONSOLIDATION = MRHProfile(
    delta_r=SpatialExtent.REGIONAL,
    delta_t=TemporalExtent.EPOCH,
    delta_c=ComplexityExtent.SOCIETY_SCALE
)

# Emergency federation coordination
PROFILE_CRISIS_COORDINATION = MRHProfile(
    delta_r=SpatialExtent.GLOBAL,
    delta_t=TemporalExtent.EPHEMERAL,
    delta_c=ComplexityExtent.SOCIETY_SCALE
)


def infer_mrh_profile_from_task(task_context: Dict) -> MRHProfile:
    """
    Infer MRH profile from task context.

    Heuristic-based approach for automatic horizon detection.

    Args:
        task_context: Dict with hints about task scope
            - task_type: "vision", "llm_inference", "coordination", etc.
            - operation: "classify", "reason", "coordinate", "consolidate"
            - session_length: seconds (for temporal estimate)
            - num_agents: number of agents involved (for spatial)
            - complexity: "simple", "medium", "high"

    Returns:
        MRHProfile inferred from context
    """
    # Default to reflexive (quick response)
    spatial = SpatialExtent.LOCAL
    temporal = TemporalExtent.EPHEMERAL
    complexity = ComplexityExtent.SIMPLE

    # Infer spatial from agent count
    num_agents = task_context.get("num_agents", 1)
    if num_agents == 1:
        spatial = SpatialExtent.LOCAL
    elif num_agents <= 10:
        spatial = SpatialExtent.REGIONAL
    else:
        spatial = SpatialExtent.GLOBAL

    # Infer temporal from session length
    session_length = task_context.get("session_length", 0)  # seconds
    if session_length < 60:  # < 1 minute
        temporal = TemporalExtent.EPHEMERAL
    elif session_length < 3600:  # < 1 hour
        temporal = TemporalExtent.SESSION
    elif session_length < 86400:  # < 1 day
        temporal = TemporalExtent.DAY
    else:
        temporal = TemporalExtent.EPOCH

    # Infer complexity from task type and operation
    task_type = task_context.get("task_type", "").lower()
    operation = task_context.get("operation", "").lower()

    if task_type == "vision" or operation in ["classify", "detect"]:
        complexity = ComplexityExtent.SIMPLE
    elif task_type == "llm_inference" or operation in ["reason", "generate"]:
        complexity = ComplexityExtent.AGENT_SCALE
    elif task_type in ["coordination", "consolidation"] or operation in ["coordinate", "consolidate"]:
        complexity = ComplexityExtent.SOCIETY_SCALE

    # Override from explicit complexity hint
    complexity_hint = task_context.get("complexity", "").lower()
    if complexity_hint == "simple":
        complexity = ComplexityExtent.SIMPLE
    elif complexity_hint in ["medium", "agent"]:
        complexity = ComplexityExtent.AGENT_SCALE
    elif complexity_hint in ["high", "complex", "society"]:
        complexity = ComplexityExtent.SOCIETY_SCALE

    return MRHProfile(
        delta_r=spatial,
        delta_t=temporal,
        delta_c=complexity
    )


if __name__ == "__main__":
    # Demo usage
    print("=== MRH Profile Demo ===\n")

    # Show predefined profiles
    print("Predefined Profiles:")
    print(f"  REFLEXIVE: {PROFILE_REFLEXIVE} → factor={PROFILE_REFLEXIVE.calculate_horizon_scaling_factor():.2f}")
    print(f"  FOCUSED: {PROFILE_FOCUSED} → factor={PROFILE_FOCUSED.calculate_horizon_scaling_factor():.2f}")
    print(f"  LEARNING: {PROFILE_LEARNING} → factor={PROFILE_LEARNING.calculate_horizon_scaling_factor():.2f}")
    print(f"  CONSOLIDATION: {PROFILE_CONSOLIDATION} → factor={PROFILE_CONSOLIDATION.calculate_horizon_scaling_factor():.2f}")
    print(f"  CRISIS_COORD: {PROFILE_CRISIS_COORDINATION} → factor={PROFILE_CRISIS_COORDINATION.calculate_horizon_scaling_factor():.2f}")

    print("\n" + "=" * 60)
    print("\nTask Inference Examples:")

    # Example 1: Quick factual query
    task1 = {
        "task_type": "vision",
        "operation": "classify",
        "session_length": 5,
        "num_agents": 1,
        "complexity": "simple"
    }
    profile1 = infer_mrh_profile_from_task(task1)
    print(f"\n1. Quick factual query: {profile1}")
    print(f"   Scaling factor: {profile1.calculate_horizon_scaling_factor():.2f}")

    # Example 2: Complex reasoning
    task2 = {
        "task_type": "llm_inference",
        "operation": "reason",
        "session_length": 300,  # 5 minutes
        "num_agents": 1,
        "complexity": "medium"
    }
    profile2 = infer_mrh_profile_from_task(task2)
    print(f"\n2. Complex reasoning: {profile2}")
    print(f"   Scaling factor: {profile2.calculate_horizon_scaling_factor():.2f}")

    # Example 3: Multi-agent coordination
    task3 = {
        "task_type": "coordination",
        "operation": "coordinate",
        "session_length": 600,  # 10 minutes
        "num_agents": 15,
        "complexity": "high"
    }
    profile3 = infer_mrh_profile_from_task(task3)
    print(f"\n3. Multi-agent coordination: {profile3}")
    print(f"   Scaling factor: {profile3.calculate_horizon_scaling_factor():.2f}")

    # Example 4: Long-term consolidation
    task4 = {
        "task_type": "consolidation",
        "operation": "consolidate",
        "session_length": 100000,  # ~28 hours
        "num_agents": 5,
        "complexity": "society"
    }
    profile4 = infer_mrh_profile_from_task(task4)
    print(f"\n4. Long-term consolidation: {profile4}")
    print(f"   Scaling factor: {profile4.calculate_horizon_scaling_factor():.2f}")

    print("\n" + "=" * 60)
    print("\nSerialization Test:")
    profile_dict = PROFILE_FOCUSED.to_dict()
    print(f"  Serialized: {profile_dict}")
    profile_restored = MRHProfile.from_dict(profile_dict)
    print(f"  Restored: {profile_restored}")
    print(f"  Match: {profile_restored == PROFILE_FOCUSED}")
