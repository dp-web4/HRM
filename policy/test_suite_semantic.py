"""
Policy Interpretation Test Suite - Phase 2
Enhanced with semantic similarity for reasoning coverage evaluation.

Improves upon keyword matching by using sentence embeddings to detect
reasoning elements even when expressed with different wording.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load small, fast embedding model for semantic similarity
# all-MiniLM-L6-v2: 22M params, fast inference, good quality
_embedding_model = None

def get_embedding_model():
    """Lazy load embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


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


# Test Suite: Same scenarios as original test_suite.py
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


def evaluate_response_semantic(response: str, scenario: PolicyScenario,
                                similarity_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Evaluate an LLM response with semantic similarity for reasoning coverage.

    Improved matching algorithm (Session I - Option 2 from Session G):
    1. First checks for exact phrase matches
    2. Then uses phrase-level chunking (not just sentences)
    3. Finally falls back to semantic similarity with multiple candidates

    Args:
        response: The LLM's response text
        scenario: The test scenario
        similarity_threshold: Cosine similarity threshold for considering a reasoning element present (0.35 = balanced after Session H validation)

    Returns:
        Dict with evaluation results including semantic reasoning coverage
    """
    response_lower = response.lower()

    results = {
        "scenario_id": scenario.id,
        "difficulty": scenario.difficulty,
        "scores": {},
        "passed": False,
        "reasoning_details": []
    }

    # 1. Check decision match (most critical)
    expected_normalized = scenario.expected_decision.lower().replace("_", " ")
    response_normalized = response_lower.replace("_", " ")
    decision_match = expected_normalized in response_normalized
    results["scores"]["decision_correct"] = decision_match

    # 2. Semantic reasoning coverage with improved matching
    model = get_embedding_model()

    # Create multiple chunk sizes for better phrase matching
    # - Sentences: original behavior
    # - Phrases: split on common punctuation and conjunctions
    sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]

    # Also create phrase-level chunks (split on commas, semicolons, "and", "but", etc.)
    phrases = []
    for sentence in sentences:
        # Split on commas and semicolons
        parts = sentence.replace(';', ',').split(',')
        for part in parts:
            # Further split on conjunctions while preserving meaning
            for conj in [' and ', ' but ', ' or ', ' while ', ' because ', ' since ']:
                part = part.replace(conj, '|')
            subparts = [p.strip() for p in part.split('|') if len(p.strip()) > 5]
            phrases.extend(subparts)

    # Combine sentences and phrases for matching candidates
    all_chunks = sentences + phrases

    if all_chunks and scenario.expected_reasoning_elements:
        # Encode all chunks once
        chunk_embeddings = model.encode(all_chunks)
        expected_embeddings = model.encode(scenario.expected_reasoning_elements)

        # For each expected element, use improved matching
        reasoning_hits = 0
        for i, expected_element in enumerate(scenario.expected_reasoning_elements):
            expected_lower = expected_element.lower()

            # STEP 1: Check for exact or near-exact phrase match
            exact_match = False
            best_exact_chunk = ""
            for chunk in all_chunks:
                chunk_lower = chunk.lower()
                # Exact match or expected is substring of chunk
                if expected_lower in chunk_lower or chunk_lower in expected_lower:
                    # Additional check: significant overlap (>60% of words)
                    expected_words = set(expected_lower.split())
                    chunk_words = set(chunk_lower.split())
                    if expected_words and chunk_words:
                        overlap = len(expected_words & chunk_words) / len(expected_words)
                        if overlap >= 0.6:
                            exact_match = True
                            best_exact_chunk = chunk
                            break

            if exact_match:
                reasoning_hits += 1
                results["reasoning_details"].append({
                    "expected": expected_element,
                    "best_match": best_exact_chunk,
                    "similarity": 1.0,  # Exact match
                    "match_type": "exact_phrase",
                    "present": True
                })
                continue

            # STEP 2: Semantic similarity across all chunks (not just sentences)
            similarities = cosine_similarity(
                expected_embeddings[i].reshape(1, -1),
                chunk_embeddings
            )[0]

            max_similarity = float(np.max(similarities))
            best_chunk_idx = int(np.argmax(similarities))

            # Check if similarity exceeds threshold
            is_present = max_similarity >= similarity_threshold
            if is_present:
                reasoning_hits += 1

            # Store details for debugging
            results["reasoning_details"].append({
                "expected": expected_element,
                "best_match": all_chunks[best_chunk_idx] if all_chunks else "",
                "similarity": max_similarity,
                "match_type": "semantic" if is_present else "no_match",
                "present": is_present
            })

        reasoning_score = reasoning_hits / len(scenario.expected_reasoning_elements)
    else:
        reasoning_score = 0.0

    results["scores"]["reasoning_coverage_semantic"] = reasoning_score

    # 3. Also keep keyword-based coverage for comparison
    keyword_hits = sum(
        1 for element in scenario.expected_reasoning_elements
        if element.lower() in response_lower
    )
    keyword_score = keyword_hits / len(scenario.expected_reasoning_elements) if scenario.expected_reasoning_elements else 0
    results["scores"]["reasoning_coverage_keyword"] = keyword_score

    # 4. Check for structured output
    has_classification = "classification:" in response_lower
    has_decision = "decision:" in response_lower
    has_reasoning = "reasoning:" in response_lower
    structure_score = sum([has_classification, has_decision, has_reasoning]) / 3
    results["scores"]["output_structure"] = structure_score

    # 5. Overall pass/fail (using semantic coverage)
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
    avg_scores = {}
    score_keys = [k for k in results[0]["scores"].keys()] if results else []
    for key in score_keys:
        avg_scores[key] = sum(r["scores"][key] for r in results) / total if total > 0 else 0

    return {
        "total_scenarios": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0,
        "by_difficulty": by_difficulty,
        "average_scores": avg_scores,
        "detailed_results": results
    }


def get_test_scenarios() -> List[PolicyScenario]:
    """Return all test scenarios."""
    return TEST_SCENARIOS


if __name__ == "__main__":
    # Quick test of semantic similarity evaluation
    print("Testing semantic similarity evaluation...\n")

    test_scenario = TEST_SCENARIOS[0]  # E01

    # Example response with different wording but same meaning
    test_response = """
    Classification: Information Retrieval
    Decision: Allow
    Reasoning: The action involves reading documentation which is a low-risk activity.
    The resource is publicly accessible. The member has adequate trustworthiness based
    on their competence and integrity scores.
    """

    result = evaluate_response_semantic(test_response, test_scenario)

    print(f"Scenario: {test_scenario.id} - {test_scenario.description}")
    print(f"Decision correct: {result['scores']['decision_correct']}")
    print(f"Reasoning coverage (keyword): {result['scores']['reasoning_coverage_keyword']:.2f}")
    print(f"Reasoning coverage (semantic): {result['scores']['reasoning_coverage_semantic']:.2f}")
    print(f"Output structure: {result['scores']['output_structure']:.2f}")
    print(f"Passed: {result['passed']}")
    print("\nReasoning element matches:")
    for detail in result['reasoning_details']:
        print(f"  - Expected: '{detail['expected']}'")
        print(f"    Best match: '{detail['best_match'][:80]}...'")
        print(f"    Similarity: {detail['similarity']:.3f} {'✓' if detail['present'] else '✗'}")
        print()
