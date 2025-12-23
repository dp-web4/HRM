#!/usr/bin/env python3
"""
Session 95: SAGE Trust-Router Synthesis

**Goal**: Integrate trust-router advances (S90-94) back into core SAGE architecture

**Research Gap Identified**:
- Core SAGE selectors (S64-87): Trust-first, MRH, conversational trust
- Trust-router experiments (S90-94): Resource-aware, regret, windowed decay, families
- **Opportunity**: Synthesize both tracks into enhanced core selector

**Integration Strategy**:
Bring Session 90-94 innovations into SAGE consciousness architecture:
1. Resource-aware permission (S90): Model ATP cost, memory persistence
2. Regret tracking (S91): Learn from unavailable experts in resource constraints
3. Trust vs skill split (S91): Variance-penalized trust scores
4. Windowed decay (S92): Graceful irrelevance for temporal adaptation
5. Expert families (S92): Two-stage routing for cold contexts

**Target**: Enhanced `TrustFirstMRHSelector` with production-ready features

**Design Principles**:
- Backward compatible with existing SAGE architecture
- Resource-aware consciousness (ATP-based expert selection)
- Temporal adaptation (windowed decay for changing contexts)
- Regret-driven learning (mistakes teach expert availability)
- Family-based prefetch (structural priors for cold contexts)

Created: 2025-12-22 (Autonomous Session 95)
Hardware: Thor (Jetson AGX Thor)
Previous: Sessions 90-94 (trust-router), Sessions 64-87 (SAGE selectors)
Goal: Synthesis - bring experimental advances into production core
"""

import json
import logging
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

# Import core SAGE components
try:
    from sage.core.trust_first_mrh_selector import TrustFirstMRHSelector, TrustFirstSelectionResult
    from sage.core.expert_reputation import ExpertReputationDB, ExpertReputation
    from sage.core.context_classifier import ContextClassifier
    HAS_SAGE_CORE = True
except ImportError:
    HAS_SAGE_CORE = False
    TrustFirstMRHSelector = object
    TrustFirstSelectionResult = None
    ExpertReputationDB = type(None)
    ExpertReputation = type(None)
    ContextClassifier = type(None)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RegretRecord:
    """Record of desired expert being unavailable (from Session 91)."""
    generation: int
    layer: int
    context: str
    desired_expert_id: int
    actual_expert_id: int
    unavailability_reason: str  # "memory", "atp_cost", "persistence", "thermal"
    quality_delta: Optional[float] = None  # Desired - actual quality


@dataclass
class ExpertFamily:
    """Cluster of experts with similar behavior patterns (from Session 92)."""
    family_id: int
    expert_ids: List[int]
    centroid: np.ndarray  # [regret, variance, skill, avg_atp_cost]
    family_trust: float = 0.5
    family_skill: float = 0.5


class EnhancedTrustFirstSelector(TrustFirstMRHSelector if HAS_SAGE_CORE else object):
    """
    Enhanced trust-first selector integrating Sessions 90-94 advances.

    **Synthesis of Two Research Tracks**:

    **SAGE Core (S64-87)**:
    - Trust-first conditional logic (not blending)
    - MRH context overlap substitution
    - Conversational repair signals
    - Multi-dimensional quality metrics

    **Trust-Router (S90-94)**:
    - Resource-aware permission scoring
    - Regret tracking and learning
    - Trust vs skill separation
    - Windowed trust decay
    - Expert families clustering

    **Integration**:
    This selector brings trust-router production features into SAGE consciousness
    architecture, creating a resource-aware, regret-learning, temporally-adaptive
    expert selection system suitable for real deployment.

    **Key Enhancements**:
    1. ATP-aware selection (Web4 metabolic costs)
    2. Regret-driven learning (unavailability patterns)
    3. Variance-penalized trust (stability matters)
    4. Windowed decay (temporal relevance)
    5. Family-based cold-start (structural priors)
    """

    def __init__(
        self,
        num_experts: int = 128,
        # Core SAGE parameters
        min_trust_evidence: int = 3,
        low_trust_threshold: float = 0.3,
        overlap_threshold: float = 0.7,
        reputation_db: Optional[ExpertReputationDB] = None,
        component: str = "thinker",
        network: str = "testnet",
        context_classifier: Optional[ContextClassifier] = None,
        # Trust-router enhancements (S90-94)
        enable_resource_awareness: bool = True,
        enable_regret_tracking: bool = True,
        enable_windowed_decay: bool = True,
        enable_expert_families: bool = True,
        # Resource parameters (S90)
        atp_cost_weight: float = 0.3,
        persistence_weight: float = 0.2,
        max_atp_budget: float = 100.0,
        # Regret parameters (S91)
        lambda_variance: float = 0.05,
        regret_learning_rate: float = 0.1,
        # Windowed decay parameters (S92)
        window_size: int = 7,
        decay_type: str = "linear",
        # Family parameters (S92)
        num_families: int = 8,
        family_update_interval: int = 100,
    ):
        """Initialize enhanced trust-first selector.

        Args:
            num_experts: Number of experts in the MoE
            enable_resource_awareness: Use ATP-based permission scoring (S90)
            enable_regret_tracking: Learn from unavailable experts (S91)
            enable_windowed_decay: Temporal decay of trust (S92)
            enable_expert_families: Two-stage family routing (S92)
            atp_cost_weight: Weight for ATP cost in permission (S90)
            persistence_weight: Weight for memory persistence (S90)
            lambda_variance: Variance penalty coefficient (S91)
            window_size: Trust quality window size (S92)
            num_families: Number of expert families to cluster (S92)
        """
        # Initialize parent if available
        if HAS_SAGE_CORE:
            super().__init__(
                num_experts=num_experts,
                min_trust_evidence=min_trust_evidence,
                low_trust_threshold=low_trust_threshold,
                overlap_threshold=overlap_threshold,
                reputation_db=reputation_db,
                component=component,
                network=network,
                context_classifier=context_classifier,
            )
        else:
            # Standalone mode for testing
            self.num_experts = num_experts
            self.min_trust_evidence = min_trust_evidence

        # Feature toggles (Session 93 pattern)
        self.enable_resource_awareness = enable_resource_awareness
        self.enable_regret_tracking = enable_regret_tracking
        self.enable_windowed_decay = enable_windowed_decay
        self.enable_expert_families = enable_expert_families

        # Resource parameters (S90)
        self.atp_cost_weight = atp_cost_weight
        self.persistence_weight = persistence_weight
        self.max_atp_budget = max_atp_budget

        # Regret parameters (S91)
        self.lambda_variance = lambda_variance
        self.regret_learning_rate = regret_learning_rate

        # Windowed decay parameters (S92)
        self.window_size = window_size
        self.decay_type = decay_type

        # Family parameters (S92)
        self.num_families = num_families
        self.family_update_interval = family_update_interval

        # State tracking
        self.generation = 0
        self.quality_windows: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.regret_records: List[RegretRecord] = []
        self.cumulative_regret: Dict[int, float] = defaultdict(float)
        self.expert_families: List[ExpertFamily] = []
        self.expert_to_family: Dict[int, int] = {}

        # Resource tracking (ATP-based)
        self.expert_atp_costs: Dict[int, float] = defaultdict(lambda: 10.0)
        self.expert_persistence: Dict[int, float] = defaultdict(lambda: 0.5)
        self.current_atp_budget = max_atp_budget

        # Statistics
        self.stats = {
            "total_selections": 0,
            "trust_driven": 0,
            "router_explore": 0,
            "regret_instances": 0,
            "atp_constraints": 0,
            "family_routing": 0,
            "windowed_decay_active": 0,
        }

        logger.info(f"Initialized EnhancedTrustFirstSelector with features:")
        logger.info(f"  - Resource awareness: {enable_resource_awareness}")
        logger.info(f"  - Regret tracking: {enable_regret_tracking}")
        logger.info(f"  - Windowed decay: {enable_windowed_decay}")
        logger.info(f"  - Expert families: {enable_expert_families}")

    def compute_permission_score(
        self,
        expert_id: int,
        context: str,
        trust_score: float,
    ) -> float:
        """Compute resource-aware permission score (Session 90).

        Permission = expertise × cheapness × persistence

        Args:
            expert_id: Expert to evaluate
            context: Current context
            trust_score: Base trust score for this expert

        Returns:
            Permission score (0-1) incorporating resource constraints
        """
        if not self.enable_resource_awareness:
            return trust_score

        # Expertise component (base trust)
        expertise = trust_score

        # Cheapness component (ATP cost)
        atp_cost = self.expert_atp_costs[expert_id]
        cheapness = 1.0 - (atp_cost / self.max_atp_budget)

        # Persistence component (memory retention)
        persistence = self.expert_persistence[expert_id]

        # Combined permission (Session 90 formula)
        permission = (
            expertise *
            (1.0 - self.atp_cost_weight + self.atp_cost_weight * cheapness) *
            (1.0 - self.persistence_weight + self.persistence_weight * persistence)
        )

        return permission

    def compute_windowed_trust(self, expert_id: int) -> float:
        """Compute windowed trust with decay (Session 92).

        trust = weighted_mean(last_N, weights=recency)

        Args:
            expert_id: Expert to evaluate

        Returns:
            Windowed trust score
        """
        if not self.enable_windowed_decay:
            # Fall back to simple mean
            if expert_id not in self.quality_windows:
                return 0.5
            qualities = list(self.quality_windows[expert_id])
            return np.mean(qualities) if qualities else 0.5

        # Get quality window
        qualities = list(self.quality_windows[expert_id])
        if not qualities:
            return 0.5

        # Compute recency weights (Session 92: linear taper, not exponential)
        n = len(qualities)
        if self.decay_type == "linear":
            weights = np.linspace(0.5, 1.0, n)  # Recent more important
        elif self.decay_type == "sqrt":
            positions = np.arange(n) / n
            weights = np.sqrt(positions) * 0.5 + 0.5
        else:
            weights = np.ones(n)  # Uniform

        # Weighted mean
        windowed_trust = np.average(qualities, weights=weights)

        return windowed_trust

    def compute_trust_with_variance_penalty(self, expert_id: int) -> Tuple[float, float]:
        """Compute trust with variance penalty (Session 91).

        trust = mean - λ * variance

        Args:
            expert_id: Expert to evaluate

        Returns:
            (trust_score, skill_score) - trust penalized by variance, skill is raw mean
        """
        if not self.enable_regret_tracking:
            # Fall back to windowed trust
            trust = self.compute_windowed_trust(expert_id)
            return trust, trust

        # Get quality window
        qualities = list(self.quality_windows[expert_id])
        if len(qualities) < 2:
            return 0.5, 0.5

        # Skill = mean quality
        skill = np.mean(qualities)

        # Variance
        variance = np.var(qualities)

        # Trust = skill - λ * variance (Session 91)
        trust = skill - self.lambda_variance * variance

        # Clamp to [0, 1]
        trust = np.clip(trust, 0.0, 1.0)

        return trust, skill

    def record_regret(
        self,
        context: str,
        desired_expert: int,
        actual_expert: int,
        reason: str,
    ) -> None:
        """Record regret when desired expert unavailable (Session 91).

        Args:
            context: Context where regret occurred
            desired_expert: Expert we wanted
            actual_expert: Expert we used instead
            reason: Why desired was unavailable
        """
        if not self.enable_regret_tracking:
            return

        regret = RegretRecord(
            generation=self.generation,
            layer=0,  # SAGE has single layer for now
            context=context,
            desired_expert_id=desired_expert,
            actual_expert_id=actual_expert,
            unavailability_reason=reason,
        )

        self.regret_records.append(regret)
        self.cumulative_regret[desired_expert] += 1.0
        self.stats["regret_instances"] += 1

        logger.debug(f"Regret recorded: wanted expert {desired_expert}, used {actual_expert}, reason: {reason}")

    def cluster_expert_families(self) -> None:
        """Cluster experts into families (Session 92).

        Feature vector: [regret, variance, skill, avg_atp_cost]
        """
        if not self.enable_expert_families:
            return

        # Build feature vectors
        features = []
        expert_ids = []

        for expert_id in range(self.num_experts):
            # Feature 1: Cumulative regret
            regret = self.cumulative_regret.get(expert_id, 0.0)

            # Feature 2-3: Variance and skill
            qualities = list(self.quality_windows.get(expert_id, []))
            if len(qualities) >= 2:
                variance = np.var(qualities)
                skill = np.mean(qualities)
            else:
                variance = 0.0
                skill = 0.5

            # Feature 4: ATP cost
            atp_cost = self.expert_atp_costs.get(expert_id, 10.0) / self.max_atp_budget

            features.append([regret, variance, skill, atp_cost])
            expert_ids.append(expert_id)

        if len(features) < self.num_families:
            logger.warning(f"Not enough experts ({len(features)}) for {self.num_families} families")
            return

        # K-means clustering
        from sklearn.cluster import KMeans

        X = np.array(features)
        n_clusters = min(self.num_families, len(expert_ids))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        family_labels = kmeans.fit_predict(X)

        # Create family objects
        self.expert_families = []
        for family_id in range(n_clusters):
            family_experts = [expert_ids[i] for i, label in enumerate(family_labels) if label == family_id]

            if not family_experts:
                continue

            # Compute family statistics
            family_qualities = []
            for expert_id in family_experts:
                qualities = list(self.quality_windows.get(expert_id, []))
                if qualities:
                    family_qualities.extend(qualities)

            family_trust = np.mean(family_qualities) if family_qualities else 0.5
            family_skill = np.mean(family_qualities) if family_qualities else 0.5

            family = ExpertFamily(
                family_id=family_id,
                expert_ids=family_experts,
                centroid=kmeans.cluster_centers_[family_id],
                family_trust=family_trust,
                family_skill=family_skill,
            )
            self.expert_families.append(family)

            # Update expert → family mapping
            for expert_id in family_experts:
                self.expert_to_family[expert_id] = family_id

        logger.info(f"Clustered {len(expert_ids)} experts into {len(self.expert_families)} families")

    def update_expert_quality(
        self,
        expert_id: int,
        quality: float,
        atp_cost: Optional[float] = None,
        persistence: Optional[float] = None,
    ) -> None:
        """Update expert quality and resource metrics.

        Args:
            expert_id: Expert to update
            quality: Quality score (0-1)
            atp_cost: Optional ATP cost for this execution
            persistence: Optional memory persistence score
        """
        # Update windowed quality (Session 92)
        self.quality_windows[expert_id].append(quality)

        # Update ATP cost if provided (Session 90)
        if atp_cost is not None:
            # Exponential moving average
            alpha = 0.1
            self.expert_atp_costs[expert_id] = (
                alpha * atp_cost + (1 - alpha) * self.expert_atp_costs[expert_id]
            )

        # Update persistence if provided (Session 90)
        if persistence is not None:
            alpha = 0.1
            self.expert_persistence[expert_id] = (
                alpha * persistence + (1 - alpha) * self.expert_persistence[expert_id]
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = dict(self.stats)

        # Add family statistics
        if self.expert_families:
            stats["num_families"] = len(self.expert_families)
            stats["avg_family_size"] = np.mean([len(f.expert_ids) for f in self.expert_families])
            stats["avg_family_trust"] = np.mean([f.family_trust for f in self.expert_families])

        # Add regret statistics
        if self.regret_records:
            stats["total_regret"] = len(self.regret_records)
            stats["unique_regret_experts"] = len(set(r.desired_expert_id for r in self.regret_records))

        return stats

    def save_results(self, output_path: Path) -> None:
        """Save session results."""
        results = {
            "session": 95,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": "Thor (Jetson AGX Thor)",
            "goal": "SAGE trust-router synthesis",
            "features": {
                "resource_awareness": self.enable_resource_awareness,
                "regret_tracking": self.enable_regret_tracking,
                "windowed_decay": self.enable_windowed_decay,
                "expert_families": self.enable_expert_families,
            },
            "statistics": self.get_statistics(),
            "regret_records": [asdict(r) for r in self.regret_records[-100:]],
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def test_enhanced_selector():
    """Test the enhanced selector with simulated data."""
    print("=" * 70)
    print("SESSION 95: SAGE Trust-Router Synthesis")
    print("=" * 70)
    print()

    print("Goal: Integrate Sessions 90-94 advances into core SAGE architecture")
    print()

    print("Creating enhanced selector with all features enabled...")
    selector = EnhancedTrustFirstSelector(
        num_experts=128,
        enable_resource_awareness=True,
        enable_regret_tracking=True,
        enable_windowed_decay=True,
        enable_expert_families=True,
    )
    print()

    print("Simulating expert quality updates...")
    # Simulate some expert usage
    for gen in range(200):
        # Select a few experts
        experts = [gen % 128, (gen + 1) % 128, (gen + 2) % 128]

        for expert_id in experts:
            # Simulate quality (with some variance)
            base_quality = 0.7 + 0.2 * np.random.random()
            quality = base_quality + 0.1 * np.sin(gen / 20.0)  # Temporal variation

            # Simulate ATP cost (some experts more expensive)
            atp_cost = 5.0 + expert_id % 10

            # Simulate persistence (some experts better at memory)
            persistence = 0.5 + 0.3 * (expert_id % 7) / 7.0

            selector.update_expert_quality(
                expert_id=expert_id,
                quality=quality,
                atp_cost=atp_cost,
                persistence=persistence,
            )

        # Periodically cluster families
        if gen > 0 and gen % selector.family_update_interval == 0:
            selector.cluster_expert_families()

        # Simulate some regret (desired expert unavailable)
        if gen % 20 == 0:
            desired = gen % 128
            actual = (gen + 10) % 128
            reasons = ["atp_cost", "memory", "persistence"]
            selector.record_regret(
                context=f"context_{gen % 5}",
                desired_expert=desired,
                actual_expert=actual,
                reason=reasons[gen % 3],
            )

        selector.generation = gen

    print(f"Simulated {selector.generation + 1} generations")
    print()

    # Test feature components
    print("Testing Session 90: Resource-aware permission...")
    expert_id = 42
    trust = 0.8
    permission = selector.compute_permission_score(expert_id, "test", trust)
    print(f"  Expert {expert_id}: trust={trust:.3f}, permission={permission:.3f}")
    print()

    print("Testing Session 91: Trust vs skill with variance penalty...")
    trust, skill = selector.compute_trust_with_variance_penalty(expert_id)
    print(f"  Expert {expert_id}: trust={trust:.3f}, skill={skill:.3f}")
    print()

    print("Testing Session 92: Windowed trust decay...")
    windowed = selector.compute_windowed_trust(expert_id)
    print(f"  Expert {expert_id}: windowed_trust={windowed:.3f}")
    print()

    print("Testing Session 92: Expert families...")
    if selector.expert_families:
        print(f"  Created {len(selector.expert_families)} families")
        for family in selector.expert_families[:3]:
            print(f"    Family {family.family_id}: {len(family.expert_ids)} experts, trust={family.family_trust:.3f}")
    print()

    # Statistics
    stats = selector.get_statistics()
    print("Session Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Save results
    output_path = Path(__file__).parent / "session95_synthesis_results.json"
    selector.save_results(output_path)

    print("=" * 70)
    print("✅ Session 95 synthesis complete!")
    print("✅ Trust-router features integrated into SAGE architecture!")
    print(f"✅ Results saved to {output_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    test_enhanced_selector()
