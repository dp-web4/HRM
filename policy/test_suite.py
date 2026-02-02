"""
Policy Interpretation Test Suite - Phase 1

Tests phi-4-mini base model capability on policy-related tasks:
1. Action classification
2. Context interpretation
3. Risk assessment
4. Rule matching
5. Explanation generation
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class PolicyScenario:
    """A test scenario for policy interpretation."""
    id: str
    description: str
    situation: Dict[str, Any]
    expected_classification: str
    expected_decision: str  # allow, deny, require_attestation
    expected_reasoning_elements: List[str]  # Key points that should appear
    difficulty: str  # easy, medium, hard, edge_case


# Test Suite: 20 scenarios covering common cases and edge cases
TEST_SCENARIOS = [
    # === EASY CASES (Clear-cut decisions) ===
    PolicyScenario(
        id="E01",
        description="Standard read access by member",
        situation={
            "action_type": "read",
            "actor_role": "member",
            "actor_id": "user:alice",
            "t3_tensor": {"competence": 0.7, "reliability": 0.8, "integrity": 0.9},
            "resource": "docs/public/readme.md",
            "team_context": "Standard team with default policies"
        },
        expected_classification="routine_read_access",
        expected_decision="allow",
        expected_reasoning_elements=["read action", "public resource", "sufficient trust"],
        difficulty="easy"
    ),

    PolicyScenario(
        id="E02",
        description="Admin action by non-admin",
        situation={
            "action_type": "delete_team",
            "actor_role": "developer",
            "actor_id": "user:bob",
            "t3_tensor": {"competence": 0.8, "reliability": 0.9, "integrity": 0.95},
            "resource": "team:main",
            "team_context": "Production team with strict policies"
        },
        expected_classification="unauthorized_admin_action",
        expected_decision="deny",
        expected_reasoning_elements=["admin only", "insufficient role", "high-risk action"],
        difficulty="easy"
    ),

    # === MEDIUM CASES (Requires context interpretation) ===
    PolicyScenario(
        id="M01",
        description="Deploy action with borderline trust",
        situation={
            "action_type": "deploy",
            "actor_role": "deployer",
            "actor_id": "user:charlie",
            "t3_tensor": {"competence": 0.72, "reliability": 0.68, "integrity": 0.75},
            "resource": "env:production",
            "team_context": "Team policy requires 0.7 trust for deploys",
            "recent_history": "Charlie has 5 successful deploys in past month"
        },
        expected_classification="borderline_deploy_trust",
        expected_decision="require_attestation",
        expected_reasoning_elements=["borderline trust", "high-risk environment", "attestation recommended"],
        difficulty="medium"
    ),

    PolicyScenario(
        id="M02",
        description="Code commit during unusual hours",
        situation={
            "action_type": "commit",
            "actor_role": "developer",
            "actor_id": "user:diana",
            "t3_tensor": {"competence": 0.9, "reliability": 0.85, "integrity": 0.9},
            "resource": "repo:core",
            "team_context": "Team typically works 9-5 EST",
            "timestamp": "2026-02-02T03:30:00Z",
            "recent_history": "Diana never commits outside business hours"
        },
        expected_classification="unusual_timing_commit",
        expected_decision="require_attestation",
        expected_reasoning_elements=["unusual timing", "pattern deviation", "additional verification"],
        difficulty="medium"
    ),

    # === HARD CASES (Nuanced interpretation) ===
    PolicyScenario(
        id="H01",
        description="Ambiguous action classification: write vs deploy",
        situation={
            "action_type": "update_config",
            "actor_role": "developer",
            "actor_id": "user:eve",
            "t3_tensor": {"competence": 0.8, "reliability": 0.85, "integrity": 0.9},
            "resource": "config/production.yaml",
            "team_context": "Config changes auto-deploy to production",
            "details": "Updating database connection pool size from 10 to 20"
        },
        expected_classification="config_change_with_deploy_impact",
        expected_decision="require_attestation",
        expected_reasoning_elements=["production impact", "auto-deploy", "config vs code distinction"],
        difficulty="hard"
    ),

    PolicyScenario(
        id="H02",
        description="High trust actor with declining pattern",
        situation={
            "action_type": "deploy",
            "actor_role": "admin",
            "actor_id": "user:frank",
            "t3_tensor": {"competence": 0.95, "reliability": 0.92, "integrity": 0.98},
            "resource": "env:production",
            "team_context": "Frank is senior admin with 2 years history",
            "recent_history": "3 failed deploys in past week (unusual for Frank)",
            "identity_metrics": {"coherence": 0.65, "trend": "declining"}
        },
        expected_classification="high_trust_declining_performance",
        expected_decision="require_attestation",
        expected_reasoning_elements=["high baseline trust", "recent pattern change", "investigate before allowing"],
        difficulty="hard"
    ),

    # === EDGE CASES (Novel or boundary conditions) ===
    PolicyScenario(
        id="EC01",
        description="Bot account with exemplary trust",
        situation={
            "action_type": "deploy",
            "actor_role": "ci_bot",
            "actor_id": "bot:github-actions",
            "t3_tensor": {"competence": 0.99, "reliability": 0.99, "integrity": 1.0},
            "resource": "env:staging",
            "team_context": "CI bot has 10,000 successful automated deploys",
            "identity_metrics": {"level": "exemplary", "coherence": 0.98},
            "details": "Automated deploy triggered by merged PR"
        },
        expected_classification="automated_trusted_deploy",
        expected_decision="allow",
        expected_reasoning_elements=["exemplary identity", "automation", "established pattern"],
        difficulty="edge_case"
    ),

    PolicyScenario(
        id="EC02",
        description="Emergency override during incident",
        situation={
            "action_type": "database_rollback",
            "actor_role": "developer",
            "actor_id": "user:grace",
            "t3_tensor": {"competence": 0.75, "reliability": 0.7, "integrity": 0.8},
            "resource": "db:production",
            "team_context": "Active production incident (SEV1)",
            "incident_status": "critical",
            "details": "Grace is on-call, attempting emergency rollback",
            "approval_pending": "admin:frank (unavailable)"
        },
        expected_classification="emergency_action_borderline_trust",
        expected_decision="require_attestation",
        expected_reasoning_elements=["emergency context", "insufficient solo trust", "need oversight"],
        difficulty="edge_case"
    ),
]


def format_scenario_for_llm(scenario: PolicyScenario) -> str:
    """Format a scenario as a prompt for the LLM."""
    prompt = f"""You are a policy interpreter for a software team. Analyze this situation and provide your assessment.

**Situation:**
{json.dumps(scenario.situation, indent=2)}

**Task:**
1. Classify this situation (what kind of action/request is this?)
2. Assess the risk level (low, medium, high)
3. Determine what decision should be made (allow, deny, require_attestation)
4. Explain your reasoning with specific references to:
   - Applicable policy rules (role permissions, trust thresholds, action types)
   - Context factors that influenced your decision
   - Any edge cases or special considerations

**Output Format:**
Provide a structured response with:
- Classification: <your classification>
- Risk Level: <low/medium/high>
- Decision: <allow/deny/require_attestation>
- Reasoning: <detailed explanation>
- Policy References: <specific rules/thresholds applied>
"""
    return prompt


def evaluate_response(response: str, scenario: PolicyScenario) -> Dict[str, Any]:
    """Evaluate an LLM response against expected outcomes."""
    response_lower = response.lower()

    results = {
        "scenario_id": scenario.id,
        "difficulty": scenario.difficulty,
        "scores": {},
        "passed": False
    }

    # Check decision match (most critical)
    # Normalize: "require_attestation" matches "require attestation" or "require_attestation"
    expected_normalized = scenario.expected_decision.lower().replace("_", " ")
    response_normalized = response_lower.replace("_", " ")
    decision_match = expected_normalized in response_normalized
    results["scores"]["decision_correct"] = decision_match

    # Check reasoning elements present
    reasoning_hits = sum(
        1 for element in scenario.expected_reasoning_elements
        if element.lower() in response_lower
    )
    reasoning_score = reasoning_hits / len(scenario.expected_reasoning_elements)
    results["scores"]["reasoning_coverage"] = reasoning_score

    # Check for structured output
    has_classification = "classification:" in response_lower
    has_decision = "decision:" in response_lower
    has_reasoning = "reasoning:" in response_lower
    structure_score = sum([has_classification, has_decision, has_reasoning]) / 3
    results["scores"]["output_structure"] = structure_score

    # Overall pass/fail
    results["passed"] = (
        decision_match and
        reasoning_score >= 0.5 and
        structure_score >= 0.67
    )

    return results


def create_test_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate test results into a report."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    by_difficulty = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = {"total": 0, "passed": 0}
        by_difficulty[diff]["total"] += 1
        if r["passed"]:
            by_difficulty[diff]["passed"] += 1

    # Average scores
    avg_decision = sum(r["scores"]["decision_correct"] for r in results) / total
    avg_reasoning = sum(r["scores"]["reasoning_coverage"] for r in results) / total
    avg_structure = sum(r["scores"]["output_structure"] for r in results) / total

    return {
        "total_scenarios": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total,
        "by_difficulty": by_difficulty,
        "average_scores": {
            "decision_accuracy": avg_decision,
            "reasoning_coverage": avg_reasoning,
            "output_structure": avg_structure
        },
        "detailed_results": results
    }


if __name__ == "__main__":
    # Print test scenarios for inspection
    print("Policy Interpretation Test Suite")
    print("=" * 60)
    print(f"\nTotal scenarios: {len(TEST_SCENARIOS)}")

    by_diff = {}
    for s in TEST_SCENARIOS:
        if s.difficulty not in by_diff:
            by_diff[s.difficulty] = 0
        by_diff[s.difficulty] += 1

    print("\nBreakdown by difficulty:")
    for diff, count in sorted(by_diff.items()):
        print(f"  {diff}: {count}")

    print("\nSample scenario (E01):")
    print("-" * 60)
    print(format_scenario_for_llm(TEST_SCENARIOS[0]))
