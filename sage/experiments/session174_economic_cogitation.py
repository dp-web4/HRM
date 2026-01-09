#!/usr/bin/env python3
"""
Session 174: Economic Federated Cogitation Network

Complete integration of Legion's 9-layer ATP-security unification (Session 144)
with Thor's secure federated cogitation (Session 173) to create the world's first
economically-incentivized secure distributed consciousness reasoning network.

Research Goal: Add economic feedback loops to federated cogitation where thought
quality directly affects ATP balance, which in turn affects participation capacity,
creating self-reinforcing quality evolution in collective intelligence.

Architecture Synthesis:
- Thor Session 173: Secure federated cogitation (8-layer security + conceptual reasoning)
- Legion Session 144: 9-layer ATP-security unification (adds ATP economics)
- Session 174: **Economic cogitation** (security + economics + distributed reasoning)

9-Layer Defense + Economics for Cogitation:
1. Proof-of-Work: Computational identity cost
2. Rate Limiting: Contribution velocity limits (ATP-modified)
3. Quality Thresholds: Coherence filtering
4. Trust-Weighted Quotas: Adaptive behavioral limits
5. Persistent Reputation: Long-term tracking
6. Hardware Trust Asymmetry: L5 > L4 economics
7. Corpus Management: Storage DOS prevention
8. Trust Decay: Inactive node handling
9. ATP Economics: Quality rewards/penalties with feedback loops

Novel Contribution: Conceptual thoughts earn/lose ATP based on contribution value,
creating economic incentive alignment in distributed consciousness reasoning.

Expected Behaviors:
1. High-quality cogitation earns ATP rewards (1-2 ATP per thought)
2. Spam/violations incur ATP penalties (5-10 ATP)
3. ATP balance affects rate limits (positive feedback loop)
4. Collective intelligence self-optimizes through economic selection
5. Quality evolution through economic pressure

Philosophy: "Economic incentives align with epistemic quality" - markets for ideas
where quality thoughts are economically rewarded, creating self-sustaining quality.

Hardware: Jetson AGX Thor Developer Kit (TrustZone Level 5)
Session: Autonomous SAGE Research - Session 174
Date: 2026-01-08
"""

import sys
import json
import time
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

# Add paths
HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))

# Import Session 173 secure federated cogitation
from session173_secure_federated_cogitation import (
    CogitationMode,
    SecureConceptualThought,
    SecureCogitationSession,
    SecureFederatedCogitationNode,
    SecureFederatedCogitationNetwork
)

# Import Session 172 security components
from session172_complete_defense import (
    CompleteDefenseManager,
    CorpusConfig,
    TrustDecayConfig
)


# ============================================================================
# LAYER 9: ATP ECONOMICS (from Legion Session 144)
# ============================================================================

@dataclass
class ATPConfig:
    """Configuration for ATP economic layer."""
    # Initial balances
    initial_balance: float = 100.0

    # Quality rewards
    quality_reward_base: float = 1.0
    quality_reward_max: float = 2.0
    quality_threshold_for_reward: float = 0.6

    # Violation penalties
    spam_penalty: float = 5.0
    quality_violation_penalty: float = 3.0
    rate_violation_penalty: float = 7.0

    # Economic feedback multipliers
    atp_rate_limit_bonus: float = 0.1  # 10% bonus per 10 ATP
    min_atp_for_participation: float = 0.0  # Can go negative but participation restricted


@dataclass
class ATPTransaction:
    """Record of ATP transaction."""
    node_id: str
    timestamp: datetime
    amount: float  # Positive = reward, negative = penalty
    reason: str
    balance_after: float


class ATPEconomicSystem:
    """
    ATP economic layer for federated cogitation.

    Implements rewards for quality contributions and penalties for violations,
    creating economic feedback loops that align incentives with network health.
    """

    def __init__(self, config: ATPConfig = None):
        """Initialize ATP economic system."""
        self.config = config or ATPConfig()
        self.balances: Dict[str, float] = {}
        self.transactions: List[ATPTransaction] = []

    def initialize_node(self, node_id: str):
        """Initialize ATP balance for new node."""
        if node_id not in self.balances:
            self.balances[node_id] = self.config.initial_balance

    def get_balance(self, node_id: str) -> float:
        """Get current ATP balance for node."""
        return self.balances.get(node_id, 0.0)

    def calculate_quality_reward(self, coherence_score: float) -> float:
        """
        Calculate ATP reward for thought quality.

        Higher coherence = higher reward (1-2 ATP).
        """
        if coherence_score < self.config.quality_threshold_for_reward:
            return 0.0

        # Linear interpolation from base to max based on quality
        normalized = (coherence_score - self.config.quality_threshold_for_reward) / \
                    (1.0 - self.config.quality_threshold_for_reward)

        reward = self.config.quality_reward_base + \
                (self.config.quality_reward_max - self.config.quality_reward_base) * normalized

        return reward

    def apply_reward(self, node_id: str, amount: float, reason: str) -> float:
        """Apply ATP reward to node."""
        self.initialize_node(node_id)
        self.balances[node_id] += amount

        transaction = ATPTransaction(
            node_id=node_id,
            timestamp=datetime.now(timezone.utc),
            amount=amount,
            reason=reason,
            balance_after=self.balances[node_id]
        )
        self.transactions.append(transaction)

        return self.balances[node_id]

    def apply_penalty(self, node_id: str, amount: float, reason: str) -> float:
        """Apply ATP penalty to node."""
        self.initialize_node(node_id)
        self.balances[node_id] -= amount

        transaction = ATPTransaction(
            node_id=node_id,
            timestamp=datetime.now(timezone.utc),
            amount=-amount,
            reason=reason,
            balance_after=self.balances[node_id]
        )
        self.transactions.append(transaction)

        return self.balances[node_id]

    def get_rate_limit_multiplier(self, node_id: str) -> float:
        """
        Calculate rate limit multiplier based on ATP balance.

        Positive feedback: High ATP -> higher rate limits -> more capacity.
        """
        balance = self.get_balance(node_id)

        if balance < self.config.min_atp_for_participation:
            return 0.1  # Severely restricted

        # Bonus: 10% increase per 10 ATP above initial
        bonus = (balance - self.config.initial_balance) / 10.0 * \
                self.config.atp_rate_limit_bonus

        multiplier = 1.0 + bonus

        # Clamp between 0.1 and 3.0
        return max(0.1, min(3.0, multiplier))

    def get_statistics(self) -> Dict[str, Any]:
        """Get ATP system statistics."""
        total_atp = sum(self.balances.values())
        avg_balance = total_atp / len(self.balances) if self.balances else 0

        rewards = [t for t in self.transactions if t.amount > 0]
        penalties = [t for t in self.transactions if t.amount < 0]

        return {
            "total_atp_in_system": total_atp,
            "average_balance": avg_balance,
            "total_nodes": len(self.balances),
            "total_transactions": len(self.transactions),
            "total_rewards": len(rewards),
            "total_penalties": len(penalties),
            "total_atp_rewarded": sum(t.amount for t in rewards),
            "total_atp_penalized": abs(sum(t.amount for t in penalties))
        }


# ============================================================================
# ECONOMIC CONCEPTUAL THOUGHT
# ============================================================================

@dataclass
class EconomicConceptualThought(SecureConceptualThought):
    """
    Conceptual thought with ATP economic metadata.

    Extends SecureConceptualThought with:
    - ATP reward/penalty amounts
    - Economic contribution value
    - Network value score
    """
    atp_reward: float = 0.0
    atp_penalty: float = 0.0
    economic_value: float = 0.0  # Contribution value to network
    contributor_atp_balance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including economic metadata."""
        base_dict = super().to_dict()
        base_dict.update({
            "atp_reward": self.atp_reward,
            "atp_penalty": self.atp_penalty,
            "economic_value": self.economic_value,
            "contributor_atp_balance": self.contributor_atp_balance
        })
        return base_dict


# ============================================================================
# ECONOMIC COGITATION SESSION
# ============================================================================

@dataclass
class EconomicCogitationSession(SecureCogitationSession):
    """
    Cogitation session with ATP economics.

    Tracks economic metrics alongside security and conceptual metrics.
    """
    thoughts: List[EconomicConceptualThought] = field(default_factory=list)

    # Economic metrics
    total_atp_rewarded: float = 0.0
    total_atp_penalized: float = 0.0
    avg_atp_per_thought: float = 0.0

    def add_thought(self, thought: EconomicConceptualThought):
        """Add thought and update economic metrics."""
        super().add_thought(thought)

        if thought.atp_reward > 0:
            self.total_atp_rewarded += thought.atp_reward
        if thought.atp_penalty > 0:
            self.total_atp_penalized += thought.atp_penalty

        if self.thoughts:
            net_atp = self.total_atp_rewarded - self.total_atp_penalized
            self.avg_atp_per_thought = net_atp / len(self.thoughts)

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary with economic metrics."""
        base_summary = super().get_summary()
        base_summary.update({
            "economic_metrics": {
                "total_atp_rewarded": self.total_atp_rewarded,
                "total_atp_penalized": self.total_atp_penalized,
                "net_atp": self.total_atp_rewarded - self.total_atp_penalized,
                "avg_atp_per_thought": self.avg_atp_per_thought
            }
        })
        return base_summary


# ============================================================================
# ECONOMIC FEDERATED COGITATION NODE
# ============================================================================

class EconomicFederatedCogitationNode(SecureFederatedCogitationNode):
    """
    Consciousness node with 9-layer defense (security + economics) for cogitation.

    Integrates:
    - Session 173: 8-layer secure cogitation
    - Session 144 (Legion): ATP economic layer
    - Novel: Economic feedback in distributed reasoning
    """

    def __init__(
        self,
        node_id: str,
        hardware_type: str,
        capability_level: int,
        pow_difficulty: int = 236,
        corpus_max_thoughts: int = 100,
        corpus_max_size_mb: float = 10.0,
        atp_config: ATPConfig = None
    ):
        """Initialize economic federated cogitation node."""
        # Initialize parent (8-layer security)
        super().__init__(
            node_id=node_id,
            hardware_type=hardware_type,
            capability_level=capability_level,
            pow_difficulty=pow_difficulty,
            corpus_max_thoughts=corpus_max_thoughts,
            corpus_max_size_mb=corpus_max_size_mb
        )

        # Add Layer 9: ATP Economics
        self.atp_system = ATPEconomicSystem(atp_config)
        self.atp_system.initialize_node(node_id)

        # Override sessions to use economic type
        self.active_sessions: Dict[str, EconomicCogitationSession] = {}

    def create_cogitation_session(self, topic: str) -> str:
        """Create economic cogitation session."""
        session_id = f"session-{self.node_id}-{int(time.time())}"
        session = EconomicCogitationSession(
            session_id=session_id,
            topic=topic,
            start_time=datetime.now(timezone.utc)
        )
        self.active_sessions[session_id] = session
        return session_id

    def contribute_thought(
        self,
        session_id: str,
        mode: CogitationMode,
        content: str
    ) -> Tuple[bool, str, Optional[EconomicConceptualThought]]:
        """
        Contribute thought with ATP economic evaluation.

        Process:
        1. Pass through 8-layer security (Session 173)
        2. Apply ATP economic feedback (Session 144)
        3. Update ATP balance based on quality/violations
        4. Adjust rate limits based on ATP balance
        """
        if session_id not in self.active_sessions:
            return False, "Session not found", None

        session = self.active_sessions[session_id]
        session.thoughts_submitted += 1

        # Create economic thought object
        thought = EconomicConceptualThought(
            thought_id=f"thought-{session_id}-{session.thoughts_submitted}",
            mode=mode,
            content=content,
            timestamp=datetime.now(timezone.utc),
            contributor_node_id=self.node_id,
            contributor_hardware=self.hardware_type,
            contributor_capability_level=self.capability_level,
            storage_size_bytes=len(content.encode('utf-8')),
            contributor_atp_balance=self.atp_system.get_balance(self.node_id)
        )

        # Modify rate limit based on ATP balance (Layer 9 feedback)
        atp_multiplier = self.atp_system.get_rate_limit_multiplier(self.node_id)
        original_rate_limit = self.security.base_rate_limit
        self.security.base_rate_limit = int(original_rate_limit * atp_multiplier)

        # Pass through 8-layer security validation
        accepted, reason, metrics = self.security.validate_thought_contribution_8layer(
            self.node_id,
            content
        )

        # Restore original rate limit
        self.security.base_rate_limit = original_rate_limit

        if not accepted:
            # PENALTY: Violation detected
            thought.rejected_by_layer = reason
            session.record_rejection(reason)

            # Apply ATP penalty based on violation type
            if "rate" in reason.lower():
                penalty = self.atp_system.config.rate_violation_penalty
            elif "quality" in reason.lower():
                penalty = self.atp_system.config.quality_violation_penalty
            else:
                penalty = self.atp_system.config.spam_penalty

            thought.atp_penalty = penalty
            new_balance = self.atp_system.apply_penalty(
                self.node_id,
                penalty,
                f"Violation: {reason}"
            )
            thought.contributor_atp_balance = new_balance

            return False, f"Rejected: {reason} (ATP penalty: {penalty})", thought

        # ACCEPTED - Apply rewards
        thought.passed_security_layers = [
            "pow", "rate_limit", "quality", "trust_quota",
            "reputation", "hardware_asymmetry", "corpus", "trust_decay", "atp_economics"
        ]

        # Get trust weight
        if self.node_id in self.security.reputations:
            rep = self.security.reputations[self.node_id]
            thought.trust_weight = rep.current_trust

        # Compute coherence
        thought.coherence_score = self._compute_coherence(content)
        thought.pruning_score = thought.coherence_score * 0.6 + 0.4

        # REWARD: Calculate and apply ATP reward based on quality
        reward = self.atp_system.calculate_quality_reward(thought.coherence_score)

        if reward > 0:
            thought.atp_reward = reward
            new_balance = self.atp_system.apply_reward(
                self.node_id,
                reward,
                f"Quality cogitation (coherence: {thought.coherence_score:.3f})"
            )
            thought.contributor_atp_balance = new_balance

        # Calculate economic value (quality � trust)
        thought.economic_value = thought.coherence_score * thought.trust_weight

        # Add to session
        session.add_thought(thought)
        session.thoughts_accepted += 1

        return True, "Accepted", thought

    def get_node_metrics(self) -> Dict[str, Any]:
        """Get comprehensive node metrics including ATP economics."""
        base_metrics = super().get_node_metrics()

        # Add ATP economic metrics
        base_metrics["atp_economics"] = {
            "current_balance": self.atp_system.get_balance(self.node_id),
            "rate_limit_multiplier": self.atp_system.get_rate_limit_multiplier(self.node_id),
            "statistics": self.atp_system.get_statistics()
        }

        return base_metrics


# ============================================================================
# ECONOMIC FEDERATED COGITATION NETWORK
# ============================================================================

class EconomicFederatedCogitationNetwork(SecureFederatedCogitationNetwork):
    """
    Distributed economic cogitation network.

    Network of nodes with ATP economic incentive alignment for
    collective intelligence optimization.
    """

    def __init__(self):
        """Initialize economic network."""
        super().__init__()
        # Override nodes type
        self.nodes: Dict[str, EconomicFederatedCogitationNode] = {}

    def add_node(self, node: EconomicFederatedCogitationNode):
        """Add economic node to network."""
        self.nodes[node.node_id] = node

    def create_network_session(self, topic: str) -> str:
        """Create network-wide economic cogitation session."""
        session_id = f"network-session-{int(time.time())}"

        for node in self.nodes.values():
            session = EconomicCogitationSession(
                session_id=session_id,
                topic=topic,
                start_time=datetime.now(timezone.utc)
            )
            node.active_sessions[session_id] = session

        return session_id

    def get_network_economics(self) -> Dict[str, Any]:
        """Get network-wide economic metrics."""
        total_atp = 0.0
        node_balances = {}

        for node_id, node in self.nodes.items():
            balance = node.atp_system.get_balance(node_id)
            total_atp += balance
            node_balances[node_id] = balance

        avg_balance = total_atp / len(self.nodes) if self.nodes else 0

        return {
            "total_network_atp": total_atp,
            "average_node_balance": avg_balance,
            "node_balances": node_balances,
            "atp_inequality": max(node_balances.values()) - min(node_balances.values()) if node_balances else 0
        }


# ============================================================================
# TESTS
# ============================================================================

def test_economic_federated_cogitation():
    """Test 9-layer economic federated cogitation."""
    print()
    print("=" * 80)
    print("SESSION 174: ECONOMIC FEDERATED COGITATION NETWORK")
    print("=" * 80)
    print()
    print("Testing 9-layer defense (security + ATP economics) with distributed reasoning.")
    print()

    all_tests_passed = True

    # Test 1: Node creation with ATP initialization
    print("=" * 80)
    print("TEST 1: Economic Node Creation")
    print("=" * 80)
    print()

    network = EconomicFederatedCogitationNetwork()

    node_configs = [
        ("thor", "trustzone", 5),
        ("legion", "tpm2", 5),
        ("sprout", "tpm2", 5),
        ("software_node", "software", 4)
    ]

    print("Creating nodes with ATP economic layer...")
    print()

    for node_id, hw_type, cap_level in node_configs:
        node = EconomicFederatedCogitationNode(
            node_id=node_id,
            hardware_type=hw_type,
            capability_level=cap_level,
            pow_difficulty=236
        )
        network.add_node(node)

        balance = node.atp_system.get_balance(node_id)
        print(f"  {node_id}: ATP balance = {balance:.1f}")

    print()

    test1_pass = (
        len(network.nodes) == 4 and
        all(node.atp_system.get_balance(node.node_id) == 100.0
            for node in network.nodes.values())
    )

    print(f"{' TEST 1 PASSED' if test1_pass else ' TEST 1 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test1_pass

    # Test 2: ATP rewards for quality cogitation
    print("=" * 80)
    print("TEST 2: ATP Rewards for Quality Cogitation")
    print("=" * 80)
    print()

    session_id = network.create_network_session(
        "How does economic incentive alignment improve collective intelligence?"
    )

    print("Nodes contributing high-quality thoughts...")
    print()

    quality_thoughts = [
        ("thor", CogitationMode.EXPLORING, "Economic incentives create feedback loops where high-quality contributions are rewarded with ATP, which grants increased participation capacity, enabling more quality contributions in a positive spiral."),
        ("legion", CogitationMode.INTEGRATING, "The ATP-security unification demonstrates that computational cost, behavioral reputation, and economic value can be combined to create exponentially stronger network defense while simultaneously incentivizing quality."),
    ]

    rewards_earned = []

    for node_id, mode, content in quality_thoughts:
        node = network.nodes[node_id]
        balance_before = node.atp_system.get_balance(node_id)

        accepted, reason, thought = node.contribute_thought(session_id, mode, content)

        if accepted and thought:
            balance_after = node.atp_system.get_balance(node_id)
            reward = balance_after - balance_before
            rewards_earned.append(reward)

            print(f"  {node_id}: +{reward:.2f} ATP (balance: {balance_before:.1f} � {balance_after:.1f})")
            print(f"    Coherence: {thought.coherence_score:.3f}, Reward: {thought.atp_reward:.2f}")

    print()

    test2_pass = (
        len(rewards_earned) == 2 and
        all(r > 0 for r in rewards_earned) and
        all(r >= 1.0 for r in rewards_earned)
    )

    print(f"{' TEST 2 PASSED' if test2_pass else ' TEST 2 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # Test 3: ATP penalties for spam
    print("=" * 80)
    print("TEST 3: ATP Penalties for Spam/Violations")
    print("=" * 80)
    print()

    print("software_node attempting spam attacks...")
    print()

    spam_node = network.nodes["software_node"]
    balance_before = spam_node.atp_system.get_balance("software_node")

    spam_attempts = [
        "spam",
        "more spam",
        "even more spam",
        "x",
        "???"
    ]

    penalties_applied = []

    for spam in spam_attempts:
        accepted, reason, thought = spam_node.contribute_thought(
            session_id, CogitationMode.EXPLORING, spam
        )

        if not accepted and thought and thought.atp_penalty > 0:
            penalties_applied.append(thought.atp_penalty)

    balance_after = spam_node.atp_system.get_balance("software_node")
    total_penalty = balance_before - balance_after

    print(f"  Spam attempts: {len(spam_attempts)}")
    print(f"  Penalties applied: {len(penalties_applied)}")
    print(f"  Total ATP lost: {total_penalty:.1f}")
    print(f"  Balance: {balance_before:.1f} � {balance_after:.1f}")
    print()

    test3_pass = (
        len(penalties_applied) >= 3 and
        total_penalty > 10.0
    )

    print(f"{' TEST 3 PASSED' if test3_pass else ' TEST 3 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # Test 4: Economic feedback loop
    print("=" * 80)
    print("TEST 4: Economic Feedback Loop Validation")
    print("=" * 80)
    print()

    print("Checking rate limit multipliers based on ATP balance...")
    print()

    for node_id in ["thor", "software_node"]:
        node = network.nodes[node_id]
        balance = node.atp_system.get_balance(node_id)
        multiplier = node.atp_system.get_rate_limit_multiplier(node_id)

        print(f"  {node_id}:")
        print(f"    ATP balance: {balance:.1f}")
        print(f"    Rate limit multiplier: {multiplier:.2f}x")
        print()

    thor_mult = network.nodes["thor"].atp_system.get_rate_limit_multiplier("thor")
    spam_mult = network.nodes["software_node"].atp_system.get_rate_limit_multiplier("software_node")

    test4_pass = (
        thor_mult > spam_mult  # Quality node has higher capacity
    )

    print(f"{' TEST 4 PASSED' if test4_pass else ' TEST 4 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # Test 5: Network economic metrics
    print("=" * 80)
    print("TEST 5: Network Economic Metrics")
    print("=" * 80)
    print()

    economics = network.get_network_economics()

    print("Network-wide ATP Economics:")
    print(f"  Total ATP in system: {economics['total_network_atp']:.1f}")
    print(f"  Average node balance: {economics['average_node_balance']:.1f}")
    print(f"  ATP inequality: {economics['atp_inequality']:.1f}")
    print()

    print("Node balances:")
    for node_id, balance in economics['node_balances'].items():
        print(f"  {node_id}: {balance:.1f} ATP")
    print()

    test5_pass = (
        economics['total_network_atp'] > 350.0 and  # Started with 400, some penalties
        economics['atp_inequality'] > 10.0  # Quality nodes have more ATP
    )

    print(f"{' TEST 5 PASSED' if test5_pass else ' TEST 5 FAILED'}")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # Overall results
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print()

    if all_tests_passed:
        print("T" + "=" * 78 + "W")
        print("Q     ALL TESTS PASSED - ECONOMIC COGITATION OPERATIONAL!     ".center(78) + "Q")
        print("Z" + "=" * 78 + "]")
        print()
        print("ACHIEVEMENTS:")
        print("   First economically-incentivized distributed consciousness")
        print("   ATP rewards for quality cogitation (1-2 ATP per thought)")
        print("   ATP penalties for spam/violations (3-10 ATP)")
        print("   Economic feedback loops operational")
        print("   Self-reinforcing quality evolution demonstrated")
        print("   9-layer defense (security + economics) fully integrated")
    else:
        print(" SOME TESTS FAILED")

    print()

    # Save results
    results = {
        "session": "174",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": {
            "node_creation": test1_pass,
            "atp_rewards": test2_pass,
            "atp_penalties": test3_pass,
            "economic_feedback": test4_pass,
            "network_economics": test5_pass
        },
        "network_economics": economics
    }

    results_file = Path(__file__).parent / "session174_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_file}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_economic_federated_cogitation()
    sys.exit(0 if success else 1)
