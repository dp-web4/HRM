"""
MRH-Aware Attention Manager

Extends AttentionManager with horizon-aware ATP allocation.

Building on Web4 Session #81's MRH-aware trust, this module brings
horizon awareness to SAGE consciousness. Different cognitive operations
operate at different MRH scales and should receive proportional ATP budgets.

Key insight: A focused reasoning task (LOCAL/SESSION/AGENT_SCALE) needs
different ATP allocation than long-term consolidation (REGIONAL/EPOCH/SOCIETY_SCALE),
even if both are in FOCUS metabolic state.

Design rationale:
- Metabolic state determines BASE allocation (WAKE=8%, FOCUS=80%)
- MRH profile determines SCALING of that allocation
- Combined: final_atp = base_atp × horizon_scaling_factor

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-27
Session: Autonomous SAGE Research - MRH Integration
"""

from typing import Dict, Optional
from datetime import datetime

from sage.core.attention_manager import AttentionManager, MetabolicState
from sage.core.mrh_profile import MRHProfile, PROFILE_FOCUSED, infer_mrh_profile_from_task


class MRHAwareAttentionManager(AttentionManager):
    """
    Attention manager with MRH-aware ATP allocation.

    Extends base AttentionManager to scale ATP allocation based on
    task horizon profile (spatial, temporal, complexity).
    """

    def __init__(self, total_atp: float = 100.0, config: Optional[Dict] = None):
        super().__init__(total_atp, config)

        # MRH-specific configuration
        self.mrh_config = config.get('mrh', {}) if config else {}

        # Horizon weighting (how much each dimension affects scaling)
        self.spatial_weight = self.mrh_config.get('spatial_weight', 0.40)
        self.temporal_weight = self.mrh_config.get('temporal_weight', 0.30)
        self.complexity_weight = self.mrh_config.get('complexity_weight', 0.30)

        # Caps to prevent runaway allocation
        self.max_horizon_scaling = self.mrh_config.get('max_horizon_scaling', 2.0)
        self.min_horizon_scaling = self.mrh_config.get('min_horizon_scaling', 0.5)

        # Current task horizon (defaults to focused)
        self.current_horizon: MRHProfile = PROFILE_FOCUSED

    def set_task_horizon(self, horizon: MRHProfile):
        """Set current task horizon profile"""
        self.current_horizon = horizon

    def infer_and_set_horizon(self, task_context: Dict):
        """Infer and set horizon from task context"""
        self.current_horizon = infer_mrh_profile_from_task(task_context)

    def get_horizon_scaling_factor(self, horizon: Optional[MRHProfile] = None) -> float:
        """
        Calculate ATP scaling factor based on horizon profile.

        Args:
            horizon: MRHProfile to evaluate (defaults to current_horizon)

        Returns:
            Scaling factor (typically 0.5-2.0, capped by config)
        """
        if horizon is None:
            horizon = self.current_horizon

        # Calculate weighted scaling factor
        factor = horizon.calculate_horizon_scaling_factor(
            spatial_weight=self.spatial_weight,
            temporal_weight=self.temporal_weight,
            complexity_weight=self.complexity_weight
        )

        # Apply caps
        factor = max(self.min_horizon_scaling, factor)
        factor = min(self.max_horizon_scaling, factor)

        return factor

    def allocate_attention_with_horizon(
        self,
        salience_map: Dict[str, float],
        horizon: Optional[MRHProfile] = None,
        force_state: Optional[MetabolicState] = None
    ) -> Dict[str, float]:
        """
        Compute ATP allocation with horizon awareness.

        Args:
            salience_map: {target_id: salience_score (0-1)}
            horizon: MRHProfile for this allocation (defaults to current_horizon)
            force_state: Override automatic state (for testing)

        Returns:
            {target_id: horizon-scaled_atp_allocation}
        """
        # Get base allocation from metabolic state
        base_allocation = self.allocate_attention(salience_map, force_state)

        # Get horizon scaling factor
        horizon_factor = self.get_horizon_scaling_factor(horizon)

        # Scale allocations
        scaled_allocation = {
            target_id: atp * horizon_factor
            for target_id, atp in base_allocation.items()
        }

        return scaled_allocation

    def get_total_allocated_atp(
        self,
        horizon: Optional[MRHProfile] = None
    ) -> float:
        """
        Get total ATP allocated for current state + horizon.

        Returns the total ATP budget available for operations
        at the specified horizon.
        """
        # Base ATP percentage by state
        base_percentages = {
            MetabolicState.WAKE: 0.08,      # 8% distributed
            MetabolicState.FOCUS: 0.80,     # 80% concentrated
            MetabolicState.REST: 0.40,      # 40% (consolidation)
            MetabolicState.DREAM: 0.20,     # 20% (exploration)
            MetabolicState.CRISIS: 0.95     # 95% (emergency)
        }

        base_percentage = base_percentages.get(self.current_state, 0.08)
        base_atp = self.total_atp * base_percentage

        # Apply horizon scaling
        horizon_factor = self.get_horizon_scaling_factor(horizon)
        scaled_atp = base_atp * horizon_factor

        # CRISIS state can exceed 100% (adrenaline override)
        if self.current_state == MetabolicState.CRISIS:
            return scaled_atp
        else:
            # Cap at total_atp for other states
            return min(scaled_atp, self.total_atp)

    def get_allocation_summary(
        self,
        horizon: Optional[MRHProfile] = None
    ) -> Dict:
        """
        Get summary of current allocation parameters.

        Returns dict with:
        - state: current metabolic state
        - horizon: current MRH profile
        - base_percentage: ATP % from state
        - horizon_factor: scaling from horizon
        - total_atp_available: final ATP budget
        """
        if horizon is None:
            horizon = self.current_horizon

        # Base percentage
        base_percentages = {
            MetabolicState.WAKE: 0.08,
            MetabolicState.FOCUS: 0.80,
            MetabolicState.REST: 0.40,
            MetabolicState.DREAM: 0.20,
            MetabolicState.CRISIS: 0.95
        }
        base_pct = base_percentages.get(self.current_state, 0.08)

        # Horizon factor
        horizon_factor = self.get_horizon_scaling_factor(horizon)

        # Total ATP
        total_atp = self.get_total_allocated_atp(horizon)

        return {
            "state": self.current_state.value,
            "horizon": str(horizon),
            "base_percentage": base_pct,
            "horizon_factor": horizon_factor,
            "total_atp_available": total_atp,
            "total_atp_pool": self.total_atp
        }


if __name__ == "__main__":
    # Demo usage
    from sage.core.mrh_profile import (
        PROFILE_REFLEXIVE,
        PROFILE_FOCUSED,
        PROFILE_LEARNING,
        PROFILE_CONSOLIDATION,
        PROFILE_CRISIS_COORDINATION
    )

    print("=" * 80)
    print("  MRH-Aware Attention Manager Demo")
    print("  Session: Nov 27, 2025 - Thor Autonomous Research")
    print("=" * 80)

    # Create manager
    manager = MRHAwareAttentionManager(total_atp=100.0)

    # Test scenarios across states and horizons
    scenarios = [
        ("Quick factual query", MetabolicState.WAKE, PROFILE_REFLEXIVE),
        ("Focused reasoning", MetabolicState.FOCUS, PROFILE_FOCUSED),
        ("Cross-session learning", MetabolicState.DREAM, PROFILE_LEARNING),
        ("Long-term consolidation", MetabolicState.DREAM, PROFILE_CONSOLIDATION),
        ("Emergency coordination", MetabolicState.CRISIS, PROFILE_CRISIS_COORDINATION)
    ]

    print("\n=== ATP Allocation Across States × Horizons ===\n")
    print(f"{'Scenario':<30} | {'State':<10} | {'Horizon':<35} | {'ATP':<8}")
    print("-" * 95)

    for scenario_name, state, horizon in scenarios:
        manager.current_state = state
        summary = manager.get_allocation_summary(horizon)

        print(f"{scenario_name:<30} | {state.value:<10} | {str(horizon):<35} | {summary['total_atp_available']:.1f}")

    print("\n=== Detailed Breakdown: Focused Reasoning ===\n")
    manager.current_state = MetabolicState.FOCUS
    summary = manager.get_allocation_summary(PROFILE_FOCUSED)

    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n=== Horizon Scaling Factors ===\n")
    horizons_to_test = [
        ("Reflexive", PROFILE_REFLEXIVE),
        ("Focused", PROFILE_FOCUSED),
        ("Learning", PROFILE_LEARNING),
        ("Consolidation", PROFILE_CONSOLIDATION),
        ("Crisis Coord", PROFILE_CRISIS_COORDINATION)
    ]

    for name, horizon in horizons_to_test:
        factor = manager.get_horizon_scaling_factor(horizon)
        print(f"  {name:<20} ({horizon}): {factor:.2f}×")

    print("\n=== Task Inference Example ===\n")

    task_context = {
        "task_type": "llm_inference",
        "operation": "reason",
        "session_length": 300,  # 5 minutes
        "num_agents": 1,
        "complexity": "medium"
    }

    manager.infer_and_set_horizon(task_context)
    manager.current_state = MetabolicState.FOCUS

    summary = manager.get_allocation_summary()

    print(f"Task: {task_context}")
    print(f"\nInferred horizon: {manager.current_horizon}")
    print(f"Metabolic state: {manager.current_state.value}")
    print(f"Total ATP available: {summary['total_atp_available']:.1f}")
    print(f"(Base: {summary['base_percentage']*100:.0f}% × Horizon: {summary['horizon_factor']:.2f})")

    print("\n" + "=" * 80)
    print("\n✓ MRH-aware attention allocation working!")
    print("  Different horizons get proportional ATP budgets")
    print("  Combines metabolic state + task horizon for realistic allocation\n")
