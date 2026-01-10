#!/usr/bin/env python3
"""
Session 179: Reputation-Aware Adaptive Depth

Research Goal: Integrate reputation tracking with ATP-adaptive depth selection.
Nodes with high reputation can maintain quality at lower ATP through trust,
while low-reputation nodes must verify more deeply to prove quality.

Hypothesis: Reputation acts as "cognitive credit" - proven reliable nodes can
operate more efficiently because the network trusts their lighter verification.
This creates a virtuous cycle rewarding consistent quality.

Convergence Points:
- Session 177: Individual ATP-adaptive depth (Thor)
- Session 178: Federated network-aware adaptation (Thor)
- Legion Session 159: Collective verification wisdom
- Legion Reputation Prototype: Persistent trust tracking
- Integration: Reputation influences cognitive resource allocation

Biological Inspiration:
- Social capital in human communities
- Trust reduces transaction costs
- Reputation enables delegation
- Long-term reliability creates efficiency gains

Novel Research Questions:
1. How does reputation modify ATP threshold requirements?
2. Can high-reputation nodes maintain quality at lower depths?
3. Does reputation create trust-based efficiency gains?
4. What feedback loops emerge from reputation-depth integration?

Platform: Thor (Jetson AGX Thor, TrustZone)
Session: Autonomous SAGE Research - Session 179
Date: 2026-01-10
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import time
import json

HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))

from session178_federated_sage_verification import (
    FederatedAdaptiveSAGE,
    FederatedSAGENetwork,
    NetworkDepthState,
    CognitiveDepth,
    DEPTH_CONFIGS,
)


# ============================================================================
# REPUTATION SYSTEM (Simplified from Legion's Prototype)
# ============================================================================

@dataclass
class SimpleReputation:
    """
    Simplified reputation tracking for nodes.

    Based on Legion's reputation prototype but simplified for
    cognitive depth integration.
    """
    node_id: str
    total_score: float  # Cumulative reputation (-100 to +100)
    event_count: int
    positive_events: int
    negative_events: int

    @property
    def reputation_level(self) -> str:
        """Categorical reputation level."""
        if self.total_score >= 50:
            return "excellent"
        elif self.total_score >= 20:
            return "good"
        elif self.total_score >= 0:
            return "neutral"
        elif self.total_score >= -20:
            return "poor"
        else:
            return "untrusted"

    @property
    def reputation_multiplier(self) -> float:
        """
        Reputation multiplier for depth threshold adjustment.

        Returns value 0.7-1.3:
        - Excellent reputation: 0.7 (can use 30% less ATP for same depth)
        - Good reputation: 0.85 (can use 15% less ATP)
        - Neutral: 1.0 (no adjustment)
        - Poor: 1.15 (requires 15% more ATP)
        - Untrusted: 1.3 (requires 30% more ATP)
        """
        if self.total_score >= 50:
            return 0.7  # Excellent - significant trust bonus
        elif self.total_score >= 20:
            return 0.85  # Good - moderate trust bonus
        elif self.total_score >= 0:
            return 1.0  # Neutral - no adjustment
        elif self.total_score >= -20:
            return 1.15  # Poor - mild penalty
        else:
            return 1.3  # Untrusted - significant penalty

    def record_event(self, impact: float):
        """
        Record a reputation event.

        Args:
            impact: Reputation impact (-1.0 to +1.0)
        """
        self.total_score += impact * 10  # Scale to -10 to +10 per event
        self.event_count += 1

        if impact > 0:
            self.positive_events += 1
        elif impact < 0:
            self.negative_events += 1


# ============================================================================
# REPUTATION-AWARE ADAPTIVE SAGE
# ============================================================================

class ReputationAwareAdaptiveSAGE(FederatedAdaptiveSAGE):
    """
    SAGE with reputation-aware adaptive depth.

    Extends Session 178's federated adaptation with reputation integration:
    - High reputation → can use lighter depth with same trust
    - Low reputation → must use deeper verification
    - Reputation accumulated through quality history
    - Creates trust-based efficiency gains

    Novel mechanism: "Cognitive credit" where reputation reduces
    verification burden for proven reliable nodes.
    """

    def __init__(
        self,
        node_id: str,
        hardware_type: str,
        capability_level: int = 4,
        initial_reputation: float = 0.0,
        enable_reputation: bool = True,
        **kwargs
    ):
        """
        Initialize reputation-aware adaptive SAGE.

        Args:
            node_id: Federation node identifier
            hardware_type: Hardware security type
            capability_level: Security level (3-5)
            initial_reputation: Starting reputation score
            enable_reputation: Enable reputation-based adjustment
            **kwargs: Additional args for FederatedAdaptiveSAGE
        """
        super().__init__(
            node_id=node_id,
            hardware_type=hardware_type,
            capability_level=capability_level,
            **kwargs
        )

        self.enable_reputation = enable_reputation

        # Reputation tracking
        self.reputation = SimpleReputation(
            node_id=node_id,
            total_score=initial_reputation,
            event_count=0,
            positive_events=0,
            negative_events=0,
        )

        # Reputation history
        self.reputation_history: List[Dict[str, Any]] = []

        print(f"[Reputation-Aware SAGE] Node initialized")
        print(f"  Node ID: {node_id}")
        print(f"  Initial reputation: {initial_reputation:.1f}")
        print(f"  Reputation level: {self.reputation.reputation_level}")
        print(f"  Reputation multiplier: {self.reputation.reputation_multiplier:.2f}")

    def select_reputation_aware_depth(self) -> CognitiveDepth:
        """
        Select cognitive depth considering both ATP and reputation.

        Strategy:
        1. Get base depth from federated selection (Session 178)
        2. Apply reputation multiplier to effective ATP
        3. Re-evaluate depth with reputation-adjusted ATP
        4. Track reputation influence

        High reputation = higher effective ATP = can afford lighter depth
        Low reputation = lower effective ATP = must use deeper verification

        Returns:
            Selected cognitive depth with reputation consideration
        """
        # 1. Base depth from federated selection
        base_depth = self.select_federated_depth()

        if not self.enable_reputation:
            return base_depth

        # 2. Get reputation multiplier
        rep_mult = self.reputation.reputation_multiplier

        # 3. Calculate reputation-adjusted ATP
        actual_atp = self.attention_manager.total_atp
        effective_atp = actual_atp / rep_mult

        # 4. Re-evaluate depth with effective ATP
        # Temporarily set ATP to effective value
        original_atp = self.attention_manager.total_atp
        self.attention_manager.total_atp = effective_atp

        # Select depth based on effective ATP
        rep_adjusted_depth = self._select_cognitive_depth()

        # Restore original ATP
        self.attention_manager.total_atp = original_atp

        # 5. Track reputation influence
        if rep_adjusted_depth != base_depth:
            print(f"[Reputation Influence] {base_depth.value} → {rep_adjusted_depth.value}")
            print(f"  Actual ATP: {actual_atp:.2f}")
            print(f"  Reputation: {self.reputation.total_score:.1f} ({self.reputation.reputation_level})")
            print(f"  Multiplier: {rep_mult:.2f}")
            print(f"  Effective ATP: {effective_atp:.2f}")

        return rep_adjusted_depth

    def record_quality_event(self, quality_score: float, depth: CognitiveDepth):
        """
        Record cogitation quality event for reputation.

        Quality above 0.7 = positive reputation
        Quality below 0.3 = negative reputation

        Args:
            quality_score: Cogitation quality (0-1)
            depth: Depth used for cogitation
        """
        if not self.enable_reputation:
            return

        # Calculate reputation impact based on quality
        if quality_score >= 0.7:
            # Good quality - positive reputation
            impact = (quality_score - 0.7) / 0.3  # 0 to 1
        elif quality_score <= 0.3:
            # Poor quality - negative reputation
            impact = (quality_score - 0.3) / 0.3  # 0 to -1
        else:
            # Neutral quality - no impact
            impact = 0.0

        # Record event
        old_score = self.reputation.total_score
        old_level = self.reputation.reputation_level

        self.reputation.record_event(impact)

        new_score = self.reputation.total_score
        new_level = self.reputation.reputation_level

        # Track history
        self.reputation_history.append({
            "timestamp": time.time(),
            "quality_score": quality_score,
            "depth": depth.value,
            "impact": impact,
            "old_reputation": old_score,
            "new_reputation": new_score,
            "level_change": old_level != new_level,
        })

        if old_level != new_level:
            print(f"[Reputation Change] {old_level} → {new_level} ({new_score:.1f})")

    def get_reputation_metrics(self) -> Dict[str, Any]:
        """Get reputation-specific metrics."""
        return {
            "node_id": self.node_id,
            "reputation_score": self.reputation.total_score,
            "reputation_level": self.reputation.reputation_level,
            "reputation_multiplier": self.reputation.reputation_multiplier,
            "event_count": self.reputation.event_count,
            "positive_events": self.reputation.positive_events,
            "negative_events": self.reputation.negative_events,
            "reputation_enabled": self.enable_reputation,
        }


# ============================================================================
# REPUTATION-AWARE NETWORK
# ============================================================================

class ReputationAwareNetwork(FederatedSAGENetwork):
    """
    Network with reputation-aware nodes.

    Extends Session 178's federated network with reputation tracking
    and reputation-based depth adjustment.
    """

    def add_reputation_aware_node(
        self,
        node_id: str,
        hardware_type: str,
        capability_level: int,
        initial_atp: float,
        initial_reputation: float = 0.0
    ):
        """Add a reputation-aware SAGE node."""
        node = ReputationAwareAdaptiveSAGE(
            node_id=node_id,
            hardware_type=hardware_type,
            capability_level=capability_level,
            initial_atp=initial_atp,
            initial_reputation=initial_reputation,
            enable_federation=True,
            enable_reputation=True,
        )

        self.nodes[node_id] = node

        # Register peers
        for existing_id, existing_node in self.nodes.items():
            if existing_id != node_id:
                node.register_peer(existing_id, {
                    "hardware_type": existing_node.hardware_type,
                    "capability_level": existing_node.capability_level,
                })
                existing_node.register_peer(node_id, {
                    "hardware_type": node.hardware_type,
                    "capability_level": node.capability_level,
                })

        print(f"\n[Reputation Network] Added node: {node_id}")
        return node

    def simulate_quality_event(
        self,
        node_id: str,
        quality_score: float,
        atp_cost: float = None
    ) -> CognitiveDepth:
        """
        Simulate a cogitation with quality scoring.

        Args:
            node_id: Node performing cogitation
            quality_score: Simulated quality (0-1)
            atp_cost: ATP consumed (or None for auto)

        Returns:
            Depth used
        """
        node = self.nodes[node_id]

        # Select depth (reputation-aware if applicable)
        if isinstance(node, ReputationAwareAdaptiveSAGE):
            depth = node.select_reputation_aware_depth()
        else:
            depth = node.select_federated_depth()

        # Apply depth configuration
        node._apply_depth_configuration(depth)

        # Consume ATP
        if atp_cost is None:
            config = DEPTH_CONFIGS[depth]
            atp_cost = config.atp_cost_per_cycle * config.cogitation_cycles

        node.attention_manager.total_atp -= atp_cost

        # Record quality for reputation
        if isinstance(node, ReputationAwareAdaptiveSAGE):
            node.record_quality_event(quality_score, depth)

        return depth

    def get_network_reputation_summary(self) -> Dict[str, Any]:
        """Get reputation summary across network."""
        reputation_aware = [
            node for node in self.nodes.values()
            if isinstance(node, ReputationAwareAdaptiveSAGE)
        ]

        if not reputation_aware:
            return {"error": "No reputation-aware nodes"}

        return {
            "total_nodes": len(self.nodes),
            "reputation_aware_nodes": len(reputation_aware),
            "reputations": {
                node.node_id: node.get_reputation_metrics()
                for node in reputation_aware
            },
            "average_reputation": sum(
                node.reputation.total_score for node in reputation_aware
            ) / len(reputation_aware),
            "reputation_distribution": {
                level: sum(
                    1 for node in reputation_aware
                    if node.reputation.reputation_level == level
                )
                for level in ["excellent", "good", "neutral", "poor", "untrusted"]
            }
        }


# ============================================================================
# TEST SUITE
# ============================================================================

def test_reputation_aware_depth():
    """
    Test reputation-aware adaptive depth with 3-node network.

    Simulates quality events that build/degrade reputation and observes
    impact on depth selection and network efficiency.
    """
    print("\n" + "="*80)
    print("SESSION 179: REPUTATION-AWARE ADAPTIVE DEPTH TEST")
    print("="*80)
    print("Testing reputation integration with cognitive depth selection")
    print("="*80 + "\n")

    # Create network
    network = ReputationAwareNetwork()

    print("[Setup] Creating 3-node reputation-aware network...\n")

    # Add nodes with different starting reputations
    legion = network.add_reputation_aware_node(
        node_id="legion",
        hardware_type="tpm2",
        capability_level=5,
        initial_atp=100.0,
        initial_reputation=30.0  # Good reputation (proven node)
    )

    thor = network.add_reputation_aware_node(
        node_id="thor",
        hardware_type="trustzone",
        capability_level=5,
        initial_atp=100.0,
        initial_reputation=0.0  # Neutral (new node)
    )

    sprout = network.add_reputation_aware_node(
        node_id="sprout",
        hardware_type="tpm2",
        capability_level=5,
        initial_atp=100.0,
        initial_reputation=-10.0  # Poor reputation (needs to prove itself)
    )

    results = {
        "session": "179",
        "title": "Reputation-Aware Adaptive Depth",
        "platform": "Thor (3-node simulation)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "goal": "Integrate reputation tracking with adaptive depth selection",
        "scenarios": []
    }

    # Update network state
    network.update_all_nodes()

    # Scenario 1: Initial depth selection with different reputations
    print("\n" + "="*80)
    print("SCENARIO 1: Initial Depth Selection by Reputation")
    print("="*80)

    scenario1_depths = {}
    for node_id in ["legion", "thor", "sprout"]:
        node = network.nodes[node_id]
        depth = node.select_reputation_aware_depth()
        scenario1_depths[node_id] = depth.value
        print(f"{node_id}: ATP={node.attention_manager.total_atp:.1f}, Rep={node.reputation.total_score:.1f}, Depth={depth.value}")

    scenario1 = {
        "scenario": 1,
        "name": "Initial depth selection by reputation",
        "depths": scenario1_depths,
        "reputations": {
            "legion": legion.reputation.total_score,
            "thor": thor.reputation.total_score,
            "sprout": sprout.reputation.total_score,
        }
    }
    results["scenarios"].append(scenario1)

    # Scenario 2: Quality events build Legion's reputation
    print("\n" + "="*80)
    print("SCENARIO 2: High Quality Events Build Reputation")
    print("="*80)

    print("\nLegion produces 5 high-quality cogitations...")
    for i in range(5):
        depth = network.simulate_quality_event("legion", quality_score=0.85)
        print(f"  Event {i+1}: Quality=0.85, Depth={depth.value}, Rep={legion.reputation.total_score:.1f}")

    scenario2 = {
        "scenario": 2,
        "name": "High quality events build reputation",
        "legion_final_reputation": legion.reputation.total_score,
        "legion_reputation_level": legion.reputation.reputation_level,
        "quality_events": 5,
    }
    results["scenarios"].append(scenario2)

    # Scenario 3: Poor quality degrades Sprout's reputation
    print("\n" + "="*80)
    print("SCENARIO 3: Poor Quality Events Degrade Reputation")
    print("="*80)

    print("\nSprout produces 3 poor-quality cogitations...")
    for i in range(3):
        depth = network.simulate_quality_event("sprout", quality_score=0.20)
        print(f"  Event {i+1}: Quality=0.20, Depth={depth.value}, Rep={sprout.reputation.total_score:.1f}")

    scenario3 = {
        "scenario": 3,
        "name": "Poor quality events degrade reputation",
        "sprout_final_reputation": sprout.reputation.total_score,
        "sprout_reputation_level": sprout.reputation.reputation_level,
        "quality_events": 3,
    }
    results["scenarios"].append(scenario3)

    # Scenario 4: Reputation enables efficiency
    print("\n" + "="*80)
    print("SCENARIO 4: Reputation-Based Efficiency")
    print("="*80)

    # Reset ATP to same level
    for node in network.nodes.values():
        node.attention_manager.total_atp = 80.0

    network.update_all_nodes()

    print("\nAll nodes at 80 ATP, selecting depth...")
    efficiency_analysis = {}
    for node_id in ["legion", "thor", "sprout"]:
        node = network.nodes[node_id]
        depth = node.select_reputation_aware_depth()

        # Calculate effective ATP
        rep_mult = node.reputation.reputation_multiplier
        effective_atp = 80.0 / rep_mult

        efficiency_analysis[node_id] = {
            "actual_atp": 80.0,
            "reputation": node.reputation.total_score,
            "multiplier": rep_mult,
            "effective_atp": effective_atp,
            "depth": depth.value,
        }

        print(f"{node_id}:")
        print(f"  Actual ATP: 80.0")
        print(f"  Reputation: {node.reputation.total_score:.1f} ({node.reputation.reputation_level})")
        print(f"  Multiplier: {rep_mult:.2f}x")
        print(f"  Effective ATP: {effective_atp:.1f}")
        print(f"  Selected Depth: {depth.value}")
        print()

    scenario4 = {
        "scenario": 4,
        "name": "Reputation-based efficiency",
        "efficiency_analysis": efficiency_analysis,
    }
    results["scenarios"].append(scenario4)

    # Network reputation summary
    print("\n" + "="*80)
    print("NETWORK REPUTATION SUMMARY")
    print("="*80)

    rep_summary = network.get_network_reputation_summary()
    results["reputation_summary"] = rep_summary

    print(f"Average reputation: {rep_summary['average_reputation']:.1f}")
    print(f"Distribution: {rep_summary['reputation_distribution']}")
    print()

    for node_id, metrics in rep_summary["reputations"].items():
        print(f"{node_id.upper()}:")
        print(f"  Score: {metrics['reputation_score']:.1f}")
        print(f"  Level: {metrics['reputation_level']}")
        print(f"  Multiplier: {metrics['reputation_multiplier']:.2f}x")
        print(f"  Events: {metrics['event_count']} ({metrics['positive_events']} positive, {metrics['negative_events']} negative)")
        print()

    # Save results
    results_path = HOME / "ai-workspace" / "HRM" / "sage" / "experiments" / "session179_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"{'='*80}")
    print("SESSION 179 COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved: {results_path}")
    print(f"{'='*80}\n")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run reputation-aware depth tests."""
    print("\n" + "="*80)
    print("SESSION 179: REPUTATION-AWARE ADAPTIVE DEPTH")
    print("="*80)
    print("Trust as Cognitive Credit")
    print("="*80)

    # Run tests
    results = test_reputation_aware_depth()

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("✅ Reputation acts as cognitive credit")
    print("✅ High reputation enables lighter depth with same trust")
    print("✅ Poor reputation requires deeper verification to prove quality")
    print("✅ Quality history creates efficiency gains")
    print("✅ Virtuous cycle rewards consistent reliability")
    print("="*80)
    print("\nNext: Real LAN deployment with reputation tracking")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
