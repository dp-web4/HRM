"""
Federation Router for SAGE Consciousness

Routes tasks to appropriate platforms based on ATP cost/budget, capabilities,
reputation, and current metabolic state.

Implements Phase 1: Local routing logic (no network yet).

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-28
Session: Autonomous SAGE Research - Federation Readiness
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from sage.federation.federation_types import (
    FederationIdentity,
    FederationTask,
    ExecutionProof,
    WitnessAttestation,
    WitnessRecord,
    WitnessOutcome,
    StakeStatus,
    IdentityStake
)
from sage.core.mrh_profile import MRHProfile


class FederationRouter:
    """
    Routes tasks to appropriate platforms based on:
    - ATP cost vs budget
    - Platform capabilities (horizon, modalities)
    - Platform reputation (trust score)
    - Current metabolic state
    - Witness availability
    """

    def __init__(self, local_identity: FederationIdentity):
        """
        Initialize federation router

        Args:
            local_identity: Identity of this platform (Thor, Sprout, etc.)
        """
        self.local_identity = local_identity

        # Known platforms in federation
        self.known_platforms: Dict[str, FederationIdentity] = {}

        # Execution tracking
        self.delegated_tasks: Dict[str, FederationTask] = {}
        self.execution_proofs: Dict[str, ExecutionProof] = {}

        # Witness tracking
        self.witness_records: Dict[str, WitnessRecord] = {}

    def register_platform(self, platform: FederationIdentity):
        """
        Register a platform in the federation

        Args:
            platform: Platform identity to register
        """
        self.known_platforms[platform.lct_id] = platform

    def should_delegate(
        self,
        task: FederationTask,
        local_budget: float
    ) -> Tuple[bool, str]:
        """
        Decide if task should be delegated to federation

        Args:
            task: Task to potentially delegate
            local_budget: ATP budget available locally

        Returns:
            (should_delegate, reason)
        """
        # Decision logic from ATP framework:
        # 1. If local budget sufficient, execute locally
        if task.estimated_cost <= local_budget:
            return (False, "sufficient_local_atp")

        # 2. Check if any capable platforms exist
        candidates = self.find_capable_platforms(task)
        if not candidates:
            return (False, "no_capable_platforms")

        # 3. Check witness availability (need ≥3 for validation)
        # For Phase 1, we'll simulate witness availability
        if len(self.known_platforms) < task.min_witnesses:
            return (False, f"insufficient_witnesses (need {task.min_witnesses}, have {len(self.known_platforms)})")

        # All checks passed - delegate!
        return (True, f"federation_routing_to_{candidates[0].platform_name}")

    def find_capable_platforms(
        self,
        task: FederationTask
    ) -> List[FederationIdentity]:
        """
        Find platforms capable of executing task

        Filters by:
        - Horizon capability (can platform handle task horizon?)
        - Modality support (does platform support task type?)
        - Reputation threshold (is platform trustworthy?)
        - Stake status (is platform's stake valid?)

        Args:
            task: Task to match

        Returns:
            List of capable platforms, sorted by reputation (best first)
        """
        candidates = []

        for platform_id, platform in self.known_platforms.items():
            # Don't delegate to self
            if platform_id == self.local_identity.lct_id:
                continue

            # Check horizon capability
            if not self._can_handle_horizon(platform, task.task_horizon):
                continue

            # Check modality support
            if task.task_type not in platform.supported_modalities:
                continue

            # Check reputation threshold (minimum 0.6 for task acceptance)
            if platform.reputation_score < 0.6:
                continue

            # Check stake status (slashed platforms cannot receive tasks)
            if platform.stake and platform.stake.status == StakeStatus.SLASHED:
                continue

            candidates.append(platform)

        # Sort by reputation (highest first)
        candidates.sort(key=lambda p: p.reputation_score, reverse=True)
        return candidates

    def _can_handle_horizon(
        self,
        platform: FederationIdentity,
        task_horizon: MRHProfile
    ) -> bool:
        """
        Check if platform can handle task horizon

        Platform can handle task if its max_horizon >= task_horizon in all dimensions.

        Args:
            platform: Platform to check
            task_horizon: Task horizon requirement

        Returns:
            True if platform can handle, False otherwise
        """
        platform_horizon = platform.max_mrh_horizon

        # Get enum values for comparison
        # Spatial: LOCAL(0) < REGIONAL(1) < GLOBAL(2)
        platform_spatial = list(platform_horizon.delta_r.__class__).index(platform_horizon.delta_r)
        task_spatial = list(task_horizon.delta_r.__class__).index(task_horizon.delta_r)

        # Temporal: EPHEMERAL(0) < SESSION(1) < DAY(2) < EPOCH(3)
        platform_temporal = list(platform_horizon.delta_t.__class__).index(platform_horizon.delta_t)
        task_temporal = list(task_horizon.delta_t.__class__).index(task_horizon.delta_t)

        # Complexity: SIMPLE(0) < AGENT_SCALE(1) < SOCIETY_SCALE(2)
        platform_complexity = list(platform_horizon.delta_c.__class__).index(platform_horizon.delta_c)
        task_complexity = list(task_horizon.delta_c.__class__).index(task_horizon.delta_c)

        spatial_ok = platform_spatial >= task_spatial
        temporal_ok = platform_temporal >= task_temporal
        complexity_ok = platform_complexity >= task_complexity

        return spatial_ok and temporal_ok and complexity_ok

    def delegate_task(
        self,
        task: FederationTask,
        target_platform: Optional[FederationIdentity] = None
    ) -> Tuple[str, str]:
        """
        Delegate task to target platform (or best available)

        Phase 1: Simulated delegation (no actual network communication)

        Args:
            task: Task to delegate
            target_platform: Specific platform (or None for auto-select)

        Returns:
            (task_id, target_platform_id)
        """
        # Auto-select if no target specified
        if target_platform is None:
            candidates = self.find_capable_platforms(task)
            if not candidates:
                raise ValueError("No capable platforms for task")
            target_platform = candidates[0]

        # Track delegation
        self.delegated_tasks[task.task_id] = task

        # In Phase 1, we simulate delegation
        # Phase 3 will add actual network communication

        return (task.task_id, target_platform.lct_id)

    def validate_execution_proof(
        self,
        proof: ExecutionProof,
        task: FederationTask
    ) -> Tuple[bool, str]:
        """
        Validate execution proof from federated platform

        Phase 1: Basic validation (no cryptographic signatures yet)

        Checks:
        - Platform is known
        - Platform has valid stake
        - Quality metrics meet requirements
        - Latency within bounds

        Args:
            proof: Execution proof to validate
            task: Original task

        Returns:
            (valid, reason)
        """
        # Check platform is known
        platform = self.known_platforms.get(proof.executing_platform)
        if not platform:
            return (False, f"unknown_platform: {proof.executing_platform}")

        # Check stake status
        if platform.stake and platform.stake.status == StakeStatus.SLASHED:
            return (False, "platform_stake_slashed")

        # Check quality meets requirements
        if proof.quality_score < task.quality_requirements.min_quality:
            return (False, f"quality_too_low: {proof.quality_score:.2f} < {task.quality_requirements.min_quality:.2f}")

        # Check convergence quality
        if proof.convergence_quality < task.quality_requirements.min_convergence:
            return (False, f"convergence_too_low: {proof.convergence_quality:.2f} < {task.quality_requirements.min_convergence:.2f}")

        # Check latency within bounds
        if proof.actual_latency > task.max_latency:
            return (False, f"latency_exceeded: {proof.actual_latency:.1f}s > {task.max_latency:.1f}s")

        # All checks passed
        return (True, "proof_validated")

    def update_platform_reputation(
        self,
        platform_id: str,
        execution_quality: float,
        alpha: float = 0.1
    ):
        """
        Update platform reputation based on execution quality

        Uses exponential moving average:
        new_reputation = alpha * quality + (1-alpha) * old_reputation

        Args:
            platform_id: Platform to update
            execution_quality: Quality score from this execution (0-1)
            alpha: Learning rate (0-1)
        """
        platform = self.known_platforms.get(platform_id)
        if not platform:
            return

        # Update reputation
        old_reputation = platform.reputation_score
        new_reputation = alpha * execution_quality + (1 - alpha) * old_reputation
        platform.reputation_score = new_reputation

    def record_execution(
        self,
        platform_id: str,
        proof: ExecutionProof
    ):
        """
        Record successful execution for platform history

        Args:
            platform_id: Platform that executed
            proof: Execution proof
        """
        platform = self.known_platforms.get(platform_id)
        if not platform:
            return

        # Create execution record
        from sage.federation.federation_types import ExecutionRecord
        record = ExecutionRecord(
            task_id=proof.task_id,
            task_type="unknown",  # Would come from task
            execution_timestamp=proof.execution_timestamp,
            actual_latency=proof.actual_latency,
            actual_cost=proof.actual_cost,
            quality_score=proof.quality_score,
            irp_iterations=proof.irp_iterations,
            final_energy=proof.final_energy,
            convergence_quality=proof.convergence_quality
        )

        # Add to platform history
        platform.execution_history.append(record)

        # Update reputation
        self.update_platform_reputation(platform_id, proof.quality_score)

    def get_federation_stats(self) -> Dict[str, any]:
        """Get federation statistics"""
        return {
            'local_platform': self.local_identity.platform_name,
            'known_platforms': len(self.known_platforms),
            'delegated_tasks': len(self.delegated_tasks),
            'execution_proofs': len(self.execution_proofs),
            'platforms': {
                pid: {
                    'name': p.platform_name,
                    'reputation': p.reputation_score,
                    'executions': len(p.execution_history),
                    'stake_status': p.stake.status.value if p.stake else None
                }
                for pid, p in self.known_platforms.items()
            }
        }


if __name__ == "__main__":
    # Demo federation router
    from sage.federation.federation_types import create_thor_identity, create_sprout_identity
    from sage.core.mrh_profile import PROFILE_REFLEXIVE, PROFILE_FOCUSED, PROFILE_LEARNING
    from sage.core.attention_manager import MetabolicState

    print("\nFederation Router Demo")
    print("=" * 80)

    # Create identities
    thor = create_thor_identity()
    sprout = create_sprout_identity()

    # Create router for Thor
    router = FederationRouter(thor)
    router.register_platform(sprout)

    print(f"\n[Router] Local platform: {thor.platform_name}")
    print(f"[Router] Registered platforms: {[p.platform_name for p in router.known_platforms.values()]}")

    # Test delegation decision
    from sage.federation.federation_types import FederationTask, QualityRequirements

    # Scenario 1: Simple task, Thor can handle
    task1 = FederationTask(
        task_id="task_1",
        task_type="llm_inference",
        task_data={'query': "What is ATP?"},
        estimated_cost=54.0,
        task_horizon=PROFILE_REFLEXIVE,
        complexity="low",
        delegating_platform=thor.lct_id,
        delegating_state=MetabolicState.FOCUS,
        quality_requirements=QualityRequirements(),
        max_latency=20.0,
        deadline=time.time() + 20.0
    )

    should_delegate, reason = router.should_delegate(task1, local_budget=75.0)
    print(f"\n[Task 1] Simple query, budget=75 ATP")
    print(f"  Should delegate: {should_delegate} ({reason})")

    # Scenario 2: Complex task, Thor insufficient ATP
    task2 = FederationTask(
        task_id="task_2",
        task_type="consolidation",
        task_data={'sessions': list(range(20))},
        estimated_cost=1200.0,
        task_horizon=PROFILE_LEARNING,
        complexity="high",
        delegating_platform=thor.lct_id,
        delegating_state=MetabolicState.DREAM,
        quality_requirements=QualityRequirements(),
        max_latency=600.0,
        deadline=time.time() + 600.0
    )

    should_delegate, reason = router.should_delegate(task2, local_budget=30.0)
    print(f"\n[Task 2] Complex consolidation, budget=30 ATP")
    print(f"  Should delegate: {should_delegate} ({reason})")

    # Check capable platforms
    candidates = router.find_capable_platforms(task2)
    print(f"  Capable platforms: {[p.platform_name for p in candidates]}")

    # Scenario 3: Check Sprout can't handle LEARNING horizon
    print(f"\n[Horizon Check] Can Sprout handle LEARNING horizon?")
    can_handle = router._can_handle_horizon(sprout, PROFILE_LEARNING)
    print(f"  Sprout max horizon: {sprout.max_mrh_horizon}")
    print(f"  Task horizon: {PROFILE_LEARNING}")
    print(f"  Can handle: {can_handle}")

    # Federation stats
    print(f"\n[Stats]")
    stats = router.get_federation_stats()
    for key, value in stats.items():
        if key != 'platforms':
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
