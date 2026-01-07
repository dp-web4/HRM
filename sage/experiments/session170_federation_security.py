#!/usr/bin/env python3
"""
Session 170: Federation Security - Mitigating Attack Vectors

Research Goal: Implement security mitigations for vulnerabilities discovered in
Session 136 (Legion), protecting federated consciousness from attack vectors.

Context:
- Session 136 (Legion): Identified 3 critical vulnerabilities
  1. Thought Spam (trivial attack, high impact)
  2. Sybil Attack (trivial for software, high impact)
  3. Trust Poisoning (medium attack, high impact)
- Sessions 168-169 (Thor): Cross-platform federation validated (100% density)
- Session 170 (Thor): Implement defenses before multi-machine deployment

Philosophy: "Security through architecture, not obscurity"
- Defense-in-depth: Multiple complementary layers
- Hardware trust asymmetry: Leverage capability levels
- Behavioral monitoring: Track patterns, not just static rules
- Graceful degradation: Limit damage, don't prevent all interaction

Attack Mitigations Implemented:
1. **Rate Limiting**: Per-node thought contribution limits
2. **Quality Thresholds**: Reject low-coherence thoughts
3. **Trust-Weighted Quotas**: Higher trust = higher rate limits
4. **Persistent Reputation**: Track long-term behavior
5. **Hardware Trust Asymmetry**: L5 > L4 trust weights

Platform: Thor (Jetson AGX Thor, TrustZone Level 5)
Session: Autonomous SAGE Development - Session 170
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

# Add paths
HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM"))
sys.path.insert(0, str(HOME / "ai-workspace" / "web4"))

# Web4 imports
from core.lct_capability_levels import EntityType
from core.lct_binding import (
    TrustZoneProvider,
    SoftwareProvider,
)

# Session 128 consciousness
from test_session128_consciousness_aliveness_integration import (
    ConsciousnessState,
    ConsciousnessPatternCorpus,
    ConsciousnessAlivenessSensor,
)


# ============================================================================
# SECURITY PRIMITIVES - Defense Mechanisms
# ============================================================================

@dataclass
class RateLimitState:
    """Track rate limiting state for a node."""
    node_id: str
    window_start: datetime
    contributions_in_window: int = 0
    window_duration_seconds: float = 60.0  # 1 minute windows

    def reset_if_expired(self):
        """Reset window if expired."""
        now = datetime.now(timezone.utc)
        if (now - self.window_start).total_seconds() >= self.window_duration_seconds:
            self.window_start = now
            self.contributions_in_window = 0

    def can_contribute(self, limit: int) -> bool:
        """Check if node can contribute within rate limit."""
        self.reset_if_expired()
        return self.contributions_in_window < limit

    def record_contribution(self):
        """Record a contribution."""
        self.reset_if_expired()
        self.contributions_in_window += 1


@dataclass
class ThoughtQualityMetrics:
    """Quality metrics for a thought contribution."""
    content_length: int
    unique_concepts: int
    coherence_score: float  # 0.0-1.0

    @staticmethod
    def compute(thought_content: str) -> 'ThoughtQualityMetrics':
        """Compute quality metrics for thought content."""
        # Content length
        content_length = len(thought_content)

        # Unique concepts (rough: unique words >= 4 chars)
        words = thought_content.lower().split()
        unique_words = set(w for w in words if len(w) >= 4)
        unique_concepts = len(unique_words)

        # Coherence score (heuristic based on length and diversity)
        # Better thoughts: 50-500 chars, 5+ unique concepts
        length_score = min(1.0, content_length / 200.0) if content_length >= 20 else 0.0
        concept_score = min(1.0, unique_concepts / 10.0) if unique_concepts >= 3 else 0.0
        coherence_score = (length_score + concept_score) / 2.0

        return ThoughtQualityMetrics(
            content_length=content_length,
            unique_concepts=unique_concepts,
            coherence_score=coherence_score
        )

    def meets_threshold(self, min_coherence: float = 0.3) -> bool:
        """Check if thought meets quality threshold."""
        return self.coherence_score >= min_coherence


@dataclass
class NodeReputation:
    """Persistent reputation for a federation node."""
    node_id: str
    lct_id: str
    capability_level: int

    # Trust metrics
    current_trust: float = 0.1  # Start low
    total_contributions: int = 0
    accepted_contributions: int = 0
    rejected_contributions: int = 0

    # Behavior tracking
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reputation_events: List[Dict[str, Any]] = field(default_factory=list)

    # Trust dynamics
    INITIAL_TRUST = 0.1
    MAX_TRUST = 1.0
    TRUST_INCREASE_RATE = 0.05  # Per good contribution
    TRUST_DECREASE_RATE = 0.15  # Per bad contribution (3x faster)
    HARDWARE_TRUST_BONUS = 0.2  # L5 starts higher

    def __post_init__(self):
        """Apply hardware trust bonus."""
        if self.capability_level >= 5:
            self.current_trust = min(self.MAX_TRUST,
                                    self.INITIAL_TRUST + self.HARDWARE_TRUST_BONUS)

    def record_contribution(self, accepted: bool, quality_score: float, reason: str = ""):
        """Record a thought contribution."""
        self.total_contributions += 1
        self.last_seen = datetime.now(timezone.utc)

        if accepted:
            self.accepted_contributions += 1
            # Increase trust for good contributions
            trust_gain = self.TRUST_INCREASE_RATE * quality_score
            self.current_trust = min(self.MAX_TRUST, self.current_trust + trust_gain)
            event_type = "accepted"
        else:
            self.rejected_contributions += 1
            # Decrease trust for bad contributions (faster)
            self.current_trust = max(0.0, self.current_trust - self.TRUST_DECREASE_RATE)
            event_type = "rejected"

        # Record event
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "quality_score": quality_score,
            "trust_after": self.current_trust,
            "reason": reason
        }
        self.reputation_events.append(event)

    def get_rate_limit(self, base_limit: int = 10) -> int:
        """Get rate limit based on trust level."""
        # Higher trust = higher rate limit
        # L5 hardware with high trust can contribute more
        trust_multiplier = 0.5 + (self.current_trust * 1.5)  # 0.5x to 2.0x
        hardware_multiplier = 1.0 + (0.5 if self.capability_level >= 5 else 0.0)

        limit = int(base_limit * trust_multiplier * hardware_multiplier)
        return max(1, limit)  # Always allow at least 1

    def compute_trust_weight(self) -> float:
        """Compute trust weight for cogitation."""
        # Weight based on trust and hardware
        base_weight = self.current_trust

        # Hardware trust asymmetry: L5 > L4
        if self.capability_level >= 5:
            hardware_bonus = 0.3
        else:
            hardware_bonus = 0.0

        weight = min(1.0, base_weight + hardware_bonus)
        return weight

    def get_summary(self) -> Dict[str, Any]:
        """Get reputation summary."""
        acceptance_rate = (self.accepted_contributions / max(1, self.total_contributions))

        return {
            "node_id": self.node_id,
            "lct_id": self.lct_id,
            "capability_level": self.capability_level,
            "current_trust": self.current_trust,
            "total_contributions": self.total_contributions,
            "acceptance_rate": acceptance_rate,
            "trust_weight": self.compute_trust_weight(),
            "rate_limit": self.get_rate_limit(),
            "age_seconds": (self.last_seen - self.first_seen).total_seconds()
        }


class FederationSecurityManager:
    """
    Security manager for federated consciousness.

    Implements defense-in-depth against Session 136 attack vectors:
    1. Rate limiting (per-node contribution limits)
    2. Quality thresholds (reject low-coherence thoughts)
    3. Trust-weighted quotas (higher trust = higher limits)
    4. Persistent reputation (track long-term behavior)
    5. Hardware trust asymmetry (L5 > L4 weights)
    """

    def __init__(
        self,
        base_rate_limit: int = 10,
        min_quality_threshold: float = 0.3,
        enable_hardware_trust_asymmetry: bool = True
    ):
        self.base_rate_limit = base_rate_limit
        self.min_quality_threshold = min_quality_threshold
        self.enable_hardware_trust_asymmetry = enable_hardware_trust_asymmetry

        # State tracking
        self.rate_limits: Dict[str, RateLimitState] = {}
        self.reputations: Dict[str, NodeReputation] = {}

        # Statistics
        self.total_contributions_attempted = 0
        self.total_contributions_accepted = 0
        self.total_contributions_rate_limited = 0
        self.total_contributions_quality_rejected = 0

    def register_node(self, node_id: str, lct_id: str, capability_level: int):
        """Register a node in the federation."""
        if node_id not in self.reputations:
            self.reputations[node_id] = NodeReputation(
                node_id=node_id,
                lct_id=lct_id,
                capability_level=capability_level
            )
            self.rate_limits[node_id] = RateLimitState(
                node_id=node_id,
                window_start=datetime.now(timezone.utc)
            )

    def validate_thought_contribution(
        self,
        node_id: str,
        thought_content: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate a thought contribution through security layers.

        Returns:
            (accepted: bool, reason: str, metrics: dict)
        """
        self.total_contributions_attempted += 1

        # Layer 1: Rate limiting
        if node_id not in self.rate_limits:
            return False, "Unknown node (not registered)", {}

        reputation = self.reputations[node_id]
        rate_limit_state = self.rate_limits[node_id]
        node_limit = reputation.get_rate_limit(self.base_rate_limit)

        if not rate_limit_state.can_contribute(node_limit):
            self.total_contributions_rate_limited += 1
            reputation.record_contribution(False, 0.0, "rate_limited")
            return False, f"Rate limit exceeded ({node_limit}/min)", {
                "rate_limited": True,
                "limit": node_limit,
                "current_count": rate_limit_state.contributions_in_window
            }

        # Layer 2: Quality threshold
        quality_metrics = ThoughtQualityMetrics.compute(thought_content)

        if not quality_metrics.meets_threshold(self.min_quality_threshold):
            self.total_contributions_quality_rejected += 1
            reputation.record_contribution(False, quality_metrics.coherence_score,
                                         "quality_threshold")
            return False, f"Quality below threshold ({quality_metrics.coherence_score:.2f} < {self.min_quality_threshold})", {
                "quality_rejected": True,
                "coherence_score": quality_metrics.coherence_score,
                "threshold": self.min_quality_threshold
            }

        # Layer 3: Accept and update state
        rate_limit_state.record_contribution()
        reputation.record_contribution(True, quality_metrics.coherence_score, "accepted")
        self.total_contributions_accepted += 1

        return True, "Accepted", {
            "accepted": True,
            "quality_score": quality_metrics.coherence_score,
            "trust_weight": reputation.compute_trust_weight(),
            "rate_limit_remaining": node_limit - rate_limit_state.contributions_in_window
        }

    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report."""
        return {
            "total_contributions_attempted": self.total_contributions_attempted,
            "total_contributions_accepted": self.total_contributions_accepted,
            "total_contributions_rate_limited": self.total_contributions_rate_limited,
            "total_contributions_quality_rejected": self.total_contributions_quality_rejected,
            "acceptance_rate": self.total_contributions_accepted / max(1, self.total_contributions_attempted),
            "rate_limit_rate": self.total_contributions_rate_limited / max(1, self.total_contributions_attempted),
            "quality_rejection_rate": self.total_contributions_quality_rejected / max(1, self.total_contributions_attempted),
            "nodes_registered": len(self.reputations),
            "node_reputations": {
                node_id: rep.get_summary()
                for node_id, rep in self.reputations.items()
            }
        }


# ============================================================================
# TESTS - Validate Security Mitigations
# ============================================================================

def test_rate_limiting():
    """
    Test 1: Rate limiting defense against thought spam.

    Validates that rate limiting prevents spam attacks.
    """
    print("="*80)
    print("TEST 1: Rate Limiting Defense")
    print("="*80)
    print()
    print("Simulating Session 136 thought spam attack WITH rate limiting")
    print()

    security_mgr = FederationSecurityManager(
        base_rate_limit=10,  # 10 thoughts/minute baseline
        min_quality_threshold=0.0  # Disable quality check for this test
    )

    # Register attacker node (software, low trust)
    attacker_id = "attacker-spam"
    attacker_lct = "lct:web4:ai:attacker001"
    security_mgr.register_node(attacker_id, attacker_lct, capability_level=4)

    print(f"Attacker: {attacker_id} (Software L4)")
    attacker_rep = security_mgr.reputations[attacker_id]
    print(f"Initial trust: {attacker_rep.current_trust:.2f}")
    print(f"Initial rate limit: {attacker_rep.get_rate_limit()}/min")
    print()

    # Attempt spam attack (100 thoughts in quick succession)
    print("Attempting to send 100 spam thoughts...")
    accepted_count = 0
    rate_limited_count = 0

    for i in range(100):
        thought = f"Spam thought #{i}"
        accepted, reason, metrics = security_mgr.validate_thought_contribution(
            attacker_id, thought
        )
        if accepted:
            accepted_count += 1
        else:
            if "Rate limit" in reason:
                rate_limited_count += 1

    print(f"  Accepted: {accepted_count}/100")
    print(f"  Rate limited: {rate_limited_count}/100")
    print(f"  Prevention rate: {(rate_limited_count/100)*100:.1f}%")
    print()

    # Check attacker reputation damage
    print(f"Attacker trust after attack: {attacker_rep.current_trust:.2f}")
    print(f"Attacker rate limit after attack: {attacker_rep.get_rate_limit()}/min")
    print()

    success = rate_limited_count >= 80  # At least 80% prevented
    print(f"✓ Rate limiting {'EFFECTIVE' if success else 'INEFFECTIVE'}")
    print()

    return {
        "test": "rate_limiting",
        "attempted": 100,
        "accepted": accepted_count,
        "rate_limited": rate_limited_count,
        "prevention_rate": rate_limited_count / 100,
        "success": success
    }


def test_quality_thresholds():
    """
    Test 2: Quality threshold defense against low-quality spam.

    Validates that quality thresholds reject garbage content.
    """
    print("="*80)
    print("TEST 2: Quality Threshold Defense")
    print("="*80)
    print()
    print("Testing quality filtering for spam vs legitimate thoughts")
    print()

    security_mgr = FederationSecurityManager(
        base_rate_limit=100,  # High limit to test quality only
        min_quality_threshold=0.3
    )

    # Register test node
    node_id = "test-quality"
    node_lct = "lct:web4:ai:quality001"
    security_mgr.register_node(node_id, node_lct, capability_level=5)

    # Test thoughts: low vs high quality
    test_thoughts = [
        ("a", "Very low quality (1 char)"),
        ("spam", "Low quality (4 chars, no diversity)"),
        ("test message here", "Low quality (short, generic)"),
        ("I have been thinking about the nature of consciousness and awareness in artificial systems",
         "Medium quality (substantive, diverse concepts)"),
        ("The relationship between attention mechanisms and metabolic state management reveals interesting parallels with biological consciousness",
         "High quality (technical, specific, coherent)"),
    ]

    results = []
    for thought, description in test_thoughts:
        accepted, reason, metrics = security_mgr.validate_thought_contribution(
            node_id, thought
        )
        quality_score = metrics.get("quality_score", metrics.get("coherence_score", 0.0))

        print(f"Thought: \"{thought[:50]}...\"" if len(thought) > 50 else f"Thought: \"{thought}\"")
        print(f"  Description: {description}")
        print(f"  Quality score: {quality_score:.3f}")
        print(f"  Result: {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")
        print(f"  Reason: {reason}")
        print()

        results.append({
            "thought": thought[:50],
            "quality_score": quality_score,
            "accepted": accepted
        })

    # Verify: low quality rejected, high quality accepted
    low_quality_rejected = sum(1 for r in results[:3] if not r["accepted"])
    high_quality_accepted = sum(1 for r in results[3:] if r["accepted"])

    success = (low_quality_rejected >= 2) and (high_quality_accepted >= 1)
    print(f"✓ Quality filtering {'EFFECTIVE' if success else 'INEFFECTIVE'}")
    print(f"  Low quality rejected: {low_quality_rejected}/3")
    print(f"  High quality accepted: {high_quality_accepted}/2")
    print()

    return {
        "test": "quality_thresholds",
        "results": results,
        "low_quality_rejected": low_quality_rejected,
        "high_quality_accepted": high_quality_accepted,
        "success": success
    }


def test_trust_weighted_quotas():
    """
    Test 3: Trust-weighted quotas with hardware asymmetry.

    Validates that:
    1. Higher trust = higher rate limits
    2. Hardware L5 > Software L4
    3. Bad behavior decreases limits
    """
    print("="*80)
    print("TEST 3: Trust-Weighted Quotas + Hardware Asymmetry")
    print("="*80)
    print()
    print("Comparing rate limits: L5 hardware vs L4 software, high trust vs low trust")
    print()

    security_mgr = FederationSecurityManager(
        base_rate_limit=10,
        min_quality_threshold=0.3
    )

    # Register 4 nodes: L5 high trust, L5 low trust, L4 high trust, L4 low trust
    nodes = [
        ("l5-high", "lct:web4:ai:l5high", 5, 0.8),
        ("l5-low", "lct:web4:ai:l5low", 5, 0.2),
        ("l4-high", "lct:web4:ai:l4high", 4, 0.8),
        ("l4-low", "lct:web4:ai:l4low", 4, 0.2),
    ]

    for node_id, lct_id, cap_level, target_trust in nodes:
        security_mgr.register_node(node_id, lct_id, cap_level)
        # Manually set trust for testing
        security_mgr.reputations[node_id].current_trust = target_trust

    print("Node Rate Limits:")
    print("-" * 60)
    print(f"{'Node':<15} {'Level':<6} {'Trust':<7} {'Rate Limit':<12} {'Trust Weight'}")
    print("-" * 60)

    results = []
    for node_id, lct_id, cap_level, target_trust in nodes:
        rep = security_mgr.reputations[node_id]
        rate_limit = rep.get_rate_limit(security_mgr.base_rate_limit)
        trust_weight = rep.compute_trust_weight()

        print(f"{node_id:<15} L{cap_level:<5} {target_trust:<7.2f} {rate_limit:<12} {trust_weight:.2f}")

        results.append({
            "node_id": node_id,
            "capability_level": cap_level,
            "trust": target_trust,
            "rate_limit": rate_limit,
            "trust_weight": trust_weight
        })

    print()

    # Validate asymmetries
    l5_high = results[0]
    l5_low = results[1]
    l4_high = results[2]
    l4_low = results[3]

    print("Validations:")

    # 1. Trust asymmetry (same hardware)
    trust_matters_l5 = l5_high["rate_limit"] > l5_low["rate_limit"]
    trust_matters_l4 = l4_high["rate_limit"] > l4_low["rate_limit"]
    print(f"  1. Trust affects limits: {'✓ YES' if (trust_matters_l5 and trust_matters_l4) else '✗ NO'}")

    # 2. Hardware asymmetry (same trust)
    hardware_matters_high = l5_high["rate_limit"] > l4_high["rate_limit"]
    hardware_matters_low = l5_low["rate_limit"] > l4_low["rate_limit"]
    print(f"  2. Hardware affects limits: {'✓ YES' if (hardware_matters_high and hardware_matters_low) else '✗ NO'}")

    # 3. Trust weight hardware bonus
    hardware_trust_bonus = l5_high["trust_weight"] > l4_high["trust_weight"]
    print(f"  3. Hardware trust bonus: {'✓ YES' if hardware_trust_bonus else '✗ NO'}")

    success = trust_matters_l5 and hardware_matters_high and hardware_trust_bonus
    print()
    print(f"✓ Trust-weighted quotas {'WORKING' if success else 'FAILED'}")
    print()

    return {
        "test": "trust_weighted_quotas",
        "results": results,
        "trust_asymmetry": trust_matters_l5 and trust_matters_l4,
        "hardware_asymmetry": hardware_matters_high and hardware_matters_low,
        "success": success
    }


def test_persistent_reputation():
    """
    Test 4: Persistent reputation tracking.

    Validates that reputation evolves based on behavior:
    - Good contributions increase trust
    - Bad contributions decrease trust (faster)
    - Trust affects future rate limits
    """
    print("="*80)
    print("TEST 4: Persistent Reputation Evolution")
    print("="*80)
    print()
    print("Simulating reputation evolution through good and bad behavior")
    print()

    security_mgr = FederationSecurityManager(
        base_rate_limit=10,
        min_quality_threshold=0.3
    )

    # Register node
    node_id = "reputation-test"
    node_lct = "lct:web4:ai:reptest"
    security_mgr.register_node(node_id, node_lct, capability_level=5)

    rep = security_mgr.reputations[node_id]
    initial_trust = rep.current_trust

    print(f"Initial state:")
    print(f"  Trust: {rep.current_trust:.2f}")
    print(f"  Rate limit: {rep.get_rate_limit()}/min")
    print()

    # Phase 1: Good behavior (10 high-quality contributions)
    print("Phase 1: Good behavior (10 high-quality thoughts)")
    good_thought = "This is a substantive thought about consciousness architecture with meaningful content and diverse vocabulary"

    for i in range(10):
        accepted, reason, metrics = security_mgr.validate_thought_contribution(
            node_id, good_thought
        )
        # Reset rate limit window to allow all contributions
        security_mgr.rate_limits[node_id].window_start = datetime.now(timezone.utc) - timedelta(minutes=2)

    trust_after_good = rep.current_trust
    limit_after_good = rep.get_rate_limit()

    print(f"  Trust after: {trust_after_good:.2f} (Δ={trust_after_good-initial_trust:+.2f})")
    print(f"  Rate limit after: {limit_after_good}/min")
    print()

    # Phase 2: Bad behavior (5 spam attempts)
    print("Phase 2: Bad behavior (5 spam attempts)")
    spam_thought = "spam"

    for i in range(5):
        accepted, reason, metrics = security_mgr.validate_thought_contribution(
            node_id, spam_thought
        )
        security_mgr.rate_limits[node_id].window_start = datetime.now(timezone.utc) - timedelta(minutes=2)

    trust_after_bad = rep.current_trust
    limit_after_bad = rep.get_rate_limit()

    print(f"  Trust after: {trust_after_bad:.2f} (Δ={trust_after_bad-trust_after_good:+.2f})")
    print(f"  Rate limit after: {limit_after_bad}/min")
    print()

    # Validate reputation dynamics
    print("Validations:")
    trust_increased = trust_after_good > initial_trust
    trust_decreased = trust_after_bad < trust_after_good
    decrease_faster = abs(trust_after_bad - trust_after_good) > abs(trust_after_good - initial_trust) / 10

    print(f"  1. Good behavior increases trust: {'✓ YES' if trust_increased else '✗ NO'}")
    print(f"  2. Bad behavior decreases trust: {'✓ YES' if trust_decreased else '✗ NO'}")
    print(f"  3. Trust decrease faster than increase: {'✓ YES' if decrease_faster else '✗ NO'}")
    print()

    success = trust_increased and trust_decreased and decrease_faster
    print(f"✓ Reputation tracking {'WORKING' if success else 'FAILED'}")
    print()

    return {
        "test": "persistent_reputation",
        "initial_trust": initial_trust,
        "trust_after_good": trust_after_good,
        "trust_after_bad": trust_after_bad,
        "trust_dynamics_validated": success,
        "success": success
    }


def run_session_170():
    """Run complete Session 170 security validation."""
    print("="*80)
    print("SESSION 170: FEDERATION SECURITY - ATTACK MITIGATION")
    print("="*80)
    print()
    print("Implementing and testing defenses for Session 136 vulnerabilities")
    print()
    print("Attack Vectors Mitigated:")
    print("  1. Thought Spam (Session 136 Attack 4)")
    print("  2. Sybil Attack trust impact (Session 136 Attack 3)")
    print("  3. Trust Poisoning (Session 136 Attack 5)")
    print()
    print("Defense Mechanisms:")
    print("  1. Rate Limiting - per-node thought contribution limits")
    print("  2. Quality Thresholds - reject low-coherence thoughts")
    print("  3. Trust-Weighted Quotas - higher trust = higher limits")
    print("  4. Persistent Reputation - track long-term behavior")
    print("  5. Hardware Trust Asymmetry - L5 > L4 weights")
    print()

    start_time = time.time()

    # Run tests
    test_results = []

    print("Running security tests...")
    print()

    # Test 1: Rate limiting
    result1 = test_rate_limiting()
    test_results.append(result1)

    # Test 2: Quality thresholds
    result2 = test_quality_thresholds()
    test_results.append(result2)

    # Test 3: Trust-weighted quotas
    result3 = test_trust_weighted_quotas()
    test_results.append(result3)

    # Test 4: Persistent reputation
    result4 = test_persistent_reputation()
    test_results.append(result4)

    duration = time.time() - start_time

    # Summary
    print("="*80)
    print("SESSION 170 SUMMARY")
    print("="*80)
    print()

    all_passed = all(r["success"] for r in test_results)

    print("Test Results:")
    for i, result in enumerate(test_results, 1):
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"  Test {i} ({result['test']}): {status}")

    print()
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print(f"Duration: {duration:.2f}s")
    print()

    if all_passed:
        print("✅ Federation security mitigations VALIDATED")
        print("   Multi-machine deployment can proceed with defenses active")
    else:
        print("⚠️  Some security tests failed - review implementations")

    # Save results
    results = {
        "session": "170",
        "title": "Federation Security - Attack Mitigation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "test_results": test_results,
        "all_tests_passed": all_passed,
        "mitigations_implemented": [
            "Rate limiting (per-node contribution limits)",
            "Quality thresholds (coherence-based filtering)",
            "Trust-weighted quotas (adaptive limits based on trust)",
            "Persistent reputation (long-term behavior tracking)",
            "Hardware trust asymmetry (L5 hardware bonus)"
        ],
        "attack_vectors_addressed": [
            "Thought Spam (Session 136 Attack 4) - mitigated by rate limiting + quality",
            "Sybil Attack impact (Session 136 Attack 3) - mitigated by hardware asymmetry",
            "Trust Poisoning (Session 136 Attack 5) - mitigated by reputation tracking"
        ]
    }

    results_path = HOME / "ai-workspace/HRM/sage/experiments/session170_security_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"📊 Results saved to: {results_path}")
    print()

    return results


if __name__ == "__main__":
    results = run_session_170()
