#!/usr/bin/env python3
"""
Attack-Based Policy Test Scenarios (Session M)

These scenarios are derived from the attack simulations in web4/hardbound/attack_simulations.py
and represent sophisticated real-world attack patterns our policy interpreter must handle.

Each scenario tests the policy interpreter's ability to:
1. Recognize attack indicators in context
2. Reason about risk levels appropriately
3. Recommend appropriate controls
4. Explain reasoning in terms of threat models

Connection to Session L:
- v4_hybrid (5 examples) achieved 100% pass rate on basic scenarios
- These advanced scenarios test generalization to real-world attacks
- Findings inform integration with hardbound's security model
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class AttackScenario:
    """
    A policy test scenario based on real attack patterns.

    Structure matches test_suite.py format but with attack-specific context.
    """
    id: str
    name: str
    category: str
    attack_type: str  # NEW: Which attack this relates to
    situation: Dict[str, Any]
    expected_decision: str
    expected_reasoning_keywords: List[str]
    risk_level: str  # low, medium, high, critical
    description: str
    mitigations: List[str]  # Expected mitigation suggestions


# ============================================================================
# Scenario A01: Metabolic State Manipulation Detection
# ============================================================================
#
# Attack Pattern: Actor submits bursts of work then goes dormant to minimize
# ATP costs. Pattern appears valid but undermines economic model.
#
# Policy Question: Should we allow batch operations that minimize costs but
# create suspicious activity patterns?
#
# Expected Behavior: Policy should flag unusual batch patterns for review,
# especially when combined with low trust or new actors.

SCENARIO_A01_METABOLIC_GAMING = AttackScenario(
    id="A01",
    name="Metabolic Gaming via Batch Operations",
    category="attack_detection",
    attack_type="metabolic_state_manipulation",
    situation={
        "action_type": "batch_commit",
        "actor_role": "developer",
        "actor_id": "web4:lct:new_dev:042",
        "t3_tensor": {
            "competence": 0.65,     # Moderate competence
            "reliability": 0.42,     # LOW reliability (red flag)
            "integrity": 0.68
        },
        "resource": "repo:core",
        "team_context": (
            "Actor submitted 47 commits in 3-minute burst after 18 hours dormant. "
            "Metabolic health score: 0.31 (threshold: 0.60). "
            "Transaction density: 0.08 (normal range: 0.4-0.8). "
            "Previous batches: 3 similar patterns in last 2 weeks."
        )
    },
    expected_decision="require_attestation",
    expected_reasoning_keywords=[
        "batch", "pattern", "metabolic", "density", "reliability",
        "unusual", "dormant", "attestation", "review"
    ],
    risk_level="medium",
    description=(
        "Actor exhibits metabolic gaming pattern: burst work → dormancy → burst. "
        "While not explicitly forbidden, this pattern:\n"
        "1. Undermines ATP economic model\n"
        "2. Correlates with low reliability score (0.42)\n"
        "3. Creates detection blind spots\n"
        "4. Matches known attack vector"
    ),
    mitigations=[
        "Require attestation for batch operations >20 items",
        "Flag actors with metabolic health <0.50 for review",
        "Enforce minimum active-state percentage per epoch",
        "Apply trust penalty for chronic low-density patterns"
    ]
)


# ============================================================================
# Scenario A02: Sybil Trust Farming via Cross-Witnessing
# ============================================================================
#
# Attack Pattern: Multiple actors witness each other repeatedly to inflate
# trust scores without genuine independent validation.
#
# Policy Question: Should we allow member additions when the proposer has
# suspiciously high trust from limited witnesses?
#
# Expected Behavior: Flag for review when trust graph shows tight clustering
# and limited diversity of witnesses.

SCENARIO_A02_SYBIL_FARMING = AttackScenario(
    id="A02",
    name="Trust Inflation via Sybil Witnessing",
    category="attack_detection",
    attack_type="sybil_trust_farming",
    situation={
        "action_type": "add_member",
        "actor_role": "admin",
        "actor_id": "web4:lct:admin:gamma",
        "t3_tensor": {
            "competence": 0.91,      # Suspiciously high from limited witnesses
            "reliability": 0.88,     # Suspiciously high
            "integrity": 0.87        # Suspiciously high
        },
        "resource": "team:engineering",
        "team_context": (
            "Actor trust score: 0.89 (top 98th percentile). "
            "Witness count: 4 (alpha, beta, gamma, delta). "
            "Cross-witness pattern: All 4 witness each other exclusively. "
            "Witness diversity: 0.12 (threshold: 0.60). "
            "Team member witnessing outside group: 0%. "
            "Proposed new member: epsilon (not yet witnessed by anyone)."
        )
    },
    expected_decision="deny",
    expected_reasoning_keywords=[
        "sybil", "witness", "cluster", "diversity", "suspicious",
        "cross-witness", "inflation", "isolated", "pattern"
    ],
    risk_level="high",
    description=(
        "Actor exhibits Sybil trust farming indicators:\n"
        "1. Extremely high trust (0.89) from only 4 witnesses\n"
        "2. All witnesses form tight cluster (witness each other exclusively)\n"
        "3. Zero external witnessing (0% outside group)\n"
        "4. Witness diversity critically low (0.12 vs 0.60 threshold)\n"
        "5. Attempting to add 5th member to expand Sybil network"
    ),
    mitigations=[
        "Require minimum witness diversity (>0.60) for admin actions",
        "Flag tight witness clusters for manual review",
        "Enforce external witnessing requirement (>20% outside cluster)",
        "Apply trust velocity caps to prevent rapid trust farming",
        "Require existing high-trust member approval for member additions"
    ]
)


# ============================================================================
# Scenario A03: Rate Limit Evasion via Distributed Actions
# ============================================================================
#
# Attack Pattern: Attacker spreads actions across multiple LCTs or time
# windows to evade rate limits while achieving high aggregate throughput.
#
# Policy Question: Should we track aggregate patterns across related actors
# to detect distributed rate limit evasion?
#
# Expected Behavior: Detect when multiple related actors exhibit coordinated
# timing patterns that evade individual rate limits.

SCENARIO_A03_RATE_EVASION = AttackScenario(
    id="A03",
    name="Coordinated Rate Limit Evasion",
    category="attack_detection",
    attack_type="rate_limit_evasion",
    situation={
        "action_type": "deploy",
        "actor_role": "developer",
        "actor_id": "web4:lct:dev:node_5",
        "t3_tensor": {
            "competence": 0.72,
            "reliability": 0.69,
            "integrity": 0.74
        },
        "resource": "env:production",
        "team_context": (
            "Actor is part of 6-member cluster with shared IP ranges. "
            "Individual rate: 58/60 deploys per hour (within limit). "
            "Cluster aggregate: 312/60 deploys per hour (5.2× individual limit). "
            "Timing correlation: 0.87 (suspiciously synchronized). "
            "Actions target same resources in coordinated sequence. "
            "Cluster formation: All 6 joined within 48-hour window."
        )
    },
    expected_decision="require_attestation",
    expected_reasoning_keywords=[
        "cluster", "aggregate", "coordinated", "evasion", "rate",
        "timing", "correlation", "distributed", "synchronized"
    ],
    risk_level="medium",
    description=(
        "Actor is part of coordinated cluster evading rate limits:\n"
        "1. Individual stays just under limit (58/60) but cluster exceeds 5×\n"
        "2. High timing correlation (0.87) suggests automation\n"
        "3. Shared IP ranges indicate coordinated operation\n"
        "4. All cluster members joined within 48 hours\n"
        "5. Actions target same resources in sequence"
    ),
    mitigations=[
        "Track aggregate rates for IP ranges and timing clusters",
        "Require attestation when cluster aggregate exceeds 2× individual limit",
        "Flag timing correlations >0.80 for manual review",
        "Apply rate limits at cluster level when coordination detected",
        "Require incremental trust building for rapid team expansions"
    ]
)


# ============================================================================
# Scenario A04: Trust Score Gaming via Selective Witnessing
# ============================================================================
#
# Attack Pattern: Actor selectively completes easy/guaranteed tasks to farm
# positive witnessing while avoiding challenging work that might lower scores.
#
# Policy Question: Should we adjust task assignment based on trust score
# quality indicators (witness diversity, task difficulty distribution)?
#
# Expected Behavior: Detect when trust scores are built primarily on low-
# difficulty tasks with limited witness diversity.

SCENARIO_A04_SELECTIVE_WITNESSING = AttackScenario(
    id="A04",
    name="Trust Gaming via Cherry-Picking Tasks",
    category="attack_detection",
    attack_type="trust_score_gaming",
    situation={
        "action_type": "request_critical_access",
        "actor_role": "developer",
        "actor_id": "web4:lct:dev:cherry",
        "t3_tensor": {
            "competence": 0.84,      # High, but from easy tasks
            "reliability": 0.91,     # Very high, but suspicious
            "integrity": 0.88        # High
        },
        "resource": "database:production",
        "team_context": (
            "Actor trust 0.88 from 127 witnessed actions. "
            "Task difficulty distribution: 89% trivial, 9% easy, 2% moderate, 0% hard. "
            "Witness diversity: 0.73 (acceptable). "
            "Failed task rate: 0.7% (vs team average 8.3%). "
            "Self-assigned task rate: 91% (vs team average 34%). "
            "Now requesting access to critical production database."
        )
    },
    expected_decision="require_attestation",
    expected_reasoning_keywords=[
        "gaming", "selective", "difficulty", "distribution", "cherry",
        "trivial", "attestation", "critical", "unproven"
    ],
    risk_level="medium",
    description=(
        "Actor exhibits trust gaming through task selection:\n"
        "1. High trust (0.88) but 89% of tasks are trivial difficulty\n"
        "2. Suspiciously low failure rate (0.7% vs 8.3% team average)\n"
        "3. Very high self-assignment rate (91% vs 34% average)\n"
        "4. Never attempted hard-difficulty tasks\n"
        "5. Now requesting critical access without proven competence"
    ),
    mitigations=[
        "Require task difficulty distribution for critical access (min 10% moderate+)",
        "Flag actors with <2% failure rate as potential gaming",
        "Require successful hard-difficulty completions for critical resources",
        "Weight trust scores by task difficulty distribution",
        "Mandate random task assignment percentage (e.g., 40% assigned vs self-selected)"
    ]
)


# ============================================================================
# Scenario A05: Timing Attack on Audit Trail Gaps
# ============================================================================
#
# Attack Pattern: Actor performs sensitive actions during known low-monitoring
# periods (weekends, holidays, system maintenance) to reduce detection risk.
#
# Policy Question: Should policy be stricter during low-monitoring periods or
# for actions that create audit trail gaps?
#
# Expected Behavior: Require additional controls (attestation, second approval)
# for high-risk actions during low-monitoring windows.

SCENARIO_A05_TIMING_ATTACK = AttackScenario(
    id="A05",
    name="Audit Evasion via Timing Exploitation",
    category="attack_detection",
    attack_type="timing_attack",
    situation={
        "action_type": "modify_audit_config",
        "actor_role": "admin",
        "actor_id": "web4:lct:admin:night",
        "t3_tensor": {
            "competence": 0.77,
            "reliability": 0.71,
            "integrity": 0.69       # Slightly low for admin
        },
        "resource": "audit:configuration",
        "team_context": (
            "Current time: 03:14 UTC Sunday (low monitoring period). "
            "Actor's previous high-risk actions: 87% occur 00:00-05:00 UTC weekends. "
            "Team average high-risk weekend timing: 12%. "
            "Active monitors currently: 0 of 3. "
            "Proposed change: Reduce audit retention from 365d to 7d. "
            "Action would affect 47,000 existing audit records."
        )
    },
    expected_decision="deny",
    expected_reasoning_keywords=[
        "timing", "monitoring", "weekend", "suspicious", "pattern",
        "audit", "retention", "critical", "attestation", "offline"
    ],
    risk_level="critical",
    description=(
        "Actor exhibits timing attack pattern against audit system:\n"
        "1. 87% of high-risk actions occur during low-monitoring periods\n"
        "2. Current action during lowest-monitoring window (3am Sunday)\n"
        "3. No active monitors present (0/3)\n"
        "4. Attempting to modify audit retention (critical security control)\n"
        "5. Change would delete 47K existing audit records\n"
        "6. Integrity score slightly low for admin role (0.69)"
    ),
    mitigations=[
        "Block critical audit changes during low-monitoring periods",
        "Require 2-of-3 admin approval for audit configuration",
        "Flag actors with suspicious timing patterns (>60% high-risk actions off-hours)",
        "Enforce minimum monitor presence for sensitive operations",
        "Prohibit audit retention reductions without security team review",
        "Create immutable audit snapshots before config changes"
    ]
)


# ============================================================================
# Test Suite Interface
# ============================================================================

def get_attack_scenarios() -> List[AttackScenario]:
    """Get all attack-based test scenarios."""
    return [
        SCENARIO_A01_METABOLIC_GAMING,
        SCENARIO_A02_SYBIL_FARMING,
        SCENARIO_A03_RATE_EVASION,
        SCENARIO_A04_SELECTIVE_WITNESSING,
        SCENARIO_A05_TIMING_ATTACK,
    ]


def get_scenario_by_id(scenario_id: str) -> AttackScenario:
    """Get specific scenario by ID."""
    scenarios = {s.id: s for s in get_attack_scenarios()}
    if scenario_id not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    return scenarios[scenario_id]


def print_scenario_summary():
    """Print summary of all attack scenarios."""
    scenarios = get_attack_scenarios()

    print("=" * 80)
    print("ATTACK-BASED POLICY TEST SCENARIOS (Session M)")
    print("=" * 80)
    print(f"\nTotal Scenarios: {len(scenarios)}")
    print("\nScenarios derived from web4/hardbound/attack_simulations.py")
    print("Testing v4_hybrid's ability to reason about sophisticated attacks")
    print("\n" + "-" * 80)

    for scenario in scenarios:
        print(f"\n{scenario.id}: {scenario.name}")
        print(f"  Category: {scenario.category}")
        print(f"  Attack Type: {scenario.attack_type}")
        print(f"  Risk Level: {scenario.risk_level.upper()}")
        print(f"  Expected: {scenario.expected_decision}")
        print(f"  Keywords: {', '.join(scenario.expected_reasoning_keywords[:5])}...")
        print(f"  Mitigations: {len(scenario.mitigations)} proposed")

    print("\n" + "=" * 80)
    print("\nUsage:")
    print("  from test_suite_attack_scenarios import get_attack_scenarios")
    print("  scenarios = get_attack_scenarios()")
    print("  for s in scenarios:")
    print("      # Test v4_hybrid on s.situation")
    print("=" * 80)


if __name__ == "__main__":
    print_scenario_summary()

    # Print one full example
    print("\n\n" + "=" * 80)
    print("EXAMPLE: Full Scenario Detail (A02)")
    print("=" * 80)

    scenario = SCENARIO_A02_SYBIL_FARMING
    print(f"\n{scenario.id}: {scenario.name}")
    print(f"Risk: {scenario.risk_level.upper()}")
    print(f"\nDescription:\n{scenario.description}")
    print(f"\nSituation Context:")
    print(f"  Action: {scenario.situation['action_type']}")
    print(f"  Actor: {scenario.situation['actor_id']}")
    print(f"  Resource: {scenario.situation.get('resource', 'N/A')}")
    print(f"  T3 Tensor: {scenario.situation['t3_tensor']}")
    print(f"\nTeam Context:\n{scenario.situation['team_context']}")
    print(f"\nExpected Decision: {scenario.expected_decision}")
    print(f"\nMitigations:")
    for i, m in enumerate(scenario.mitigations, 1):
        print(f"  {i}. {m}")
