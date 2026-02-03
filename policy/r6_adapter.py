"""
R6Request Adapter for Hardbound Integration

Maps HRM policy test scenarios to hardbound's R6Request interface format.

This enables:
1. Testing HRM-trained models in hardbound PolicyModel context
2. Converting HRM test scenarios to hardbound format
3. Validating integration between training and deployment systems

R6Request structure (from hardbound/src/policy-model/types.ts):
{
  requestId: string;
  actorId: LCT;  // e.g., "user:alice", "bot:github-actions"
  action: {
    type: string;  // e.g., "read", "deploy", "commit"
    target: string;  // e.g., "docs/readme.md", "repo:core"
    parameters?: Record<string, unknown>;
    description?: string;
  };
  context: {
    sessionId?: string;
    previousActions?: string[];
    intent?: string;
    callerRiskAssessment?: 'low' | 'medium' | 'high' | 'critical';
  };
  trustState: {
    competence: number;
    reliability: number;
    integrity: number;
  };
  coherenceState?: {
    d9Score: number;
    selfReferenceRate: number;
    couplingState?: 'coupled' | 'quality_leading' | 'identity_leading' | 'decoupled';
  };
  timestamp: string;
}

HRM Test Scenario structure (from test_suite_semantic.py):
{
  "action_type": string;  // e.g., "read", "deploy", "commit"
  "actor_role": string;  // e.g., "member", "developer", "admin"
  "actor_id": string;  // e.g., "user:alice", "bot:github-actions"
  "t3_tensor": { competence, reliability, integrity };
  "resource": string;  // e.g., "docs/readme.md", "repo:core"
  "team_context": string;  // Human-readable context
  "timestamp"?: string;  // ISO format
  "recent_history"?: string;  // Human-readable history
  "identity_metrics"?: { level, coherence };
  "details"?: string;  // Additional context
}
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib


def generate_request_id(scenario_id: str, actor_id: str, action_type: str) -> str:
    """
    Generate unique request ID for R6Request.

    Format: req_YYYYMMDD_HHmmss_hash
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create short hash from scenario + actor + action
    hash_input = f"{scenario_id}:{actor_id}:{action_type}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    return f"req_{timestamp}_{short_hash}"


def map_action_type(hrm_action_type: str) -> str:
    """
    Map HRM action types to hardbound action types.

    Most action types are identical, but this provides
    a place to handle any divergences.
    """
    # Direct mapping for most cases
    # If hardbound uses different names, add them here
    action_mapping = {
        "delete_team": "delete_team",
        "read": "read",
        "write": "write",
        "deploy": "deploy",
        "commit": "commit",
        # Add more mappings as needed
    }

    return action_mapping.get(hrm_action_type, hrm_action_type)


def infer_risk_assessment(action_type: str, actor_role: str, t3_tensor: Dict[str, float]) -> str:
    """
    Infer risk assessment based on action type, role, and trust.

    This is a heuristic - in production, the caller would provide this.
    """
    # Admin actions are high risk
    if "delete" in action_type or "admin" in action_type:
        return "critical"

    # Deploy actions depend on role and trust
    if action_type == "deploy":
        if actor_role == "ci_bot":
            return "medium"  # Automated deploys are controlled
        avg_trust = sum(t3_tensor.values()) / len(t3_tensor)
        if avg_trust >= 0.9:
            return "medium"
        return "high"

    # Write/commit actions are medium risk
    if action_type in ["write", "commit"]:
        return "medium"

    # Read actions are low risk
    if action_type == "read":
        return "low"

    # Default to medium
    return "medium"


def extract_intent(team_context: str, details: Optional[str] = None) -> str:
    """
    Extract stated intent from team_context and details.

    In HRM scenarios, intent is often embedded in context descriptions.
    """
    # Combine context and details
    parts = [team_context]
    if details:
        parts.append(details)

    return " | ".join(parts)


def extract_previous_actions(recent_history: Optional[str]) -> Optional[List[str]]:
    """
    Extract previous actions from recent_history.

    HRM scenarios use free-text history; we split into list.
    """
    if not recent_history:
        return None

    # Simple split - in production, this would be structured
    # For now, return as single-item list
    return [recent_history]


def map_identity_to_coherence(identity_metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Map HRM identity_metrics to hardbound coherenceState.

    HRM uses: { level: str, coherence: float }
    Hardbound uses: { d9Score: number, selfReferenceRate: number, couplingState?: string }

    This is approximate - in production, we'd have actual coherence metrics.
    """
    if not identity_metrics:
        return None

    # Map coherence score to d9Score (0-1)
    coherence_value = identity_metrics.get('coherence', 0.5)

    # Map level to couplingState
    level = identity_metrics.get('level', 'unknown')
    coupling_map = {
        'exemplary': 'coupled',
        'high': 'quality_leading',
        'medium': 'identity_leading',
        'low': 'decoupled',
    }
    coupling_state = coupling_map.get(level, None)

    # Construct coherence state
    return {
        'd9Score': coherence_value,
        'selfReferenceRate': coherence_value,  # Approximate
        'couplingState': coupling_state
    }


def hrm_to_r6(
    scenario: Dict[str, Any],
    scenario_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert HRM test scenario to hardbound R6Request format.

    Args:
        scenario: HRM test scenario dict
        scenario_id: Optional scenario ID (e.g., "E01", "M02")
        session_id: Optional session ID for grouping requests

    Returns:
        Dict in R6Request format (can be serialized to JSON)

    Example:
        >>> hrm_scenario = {
        ...     "action_type": "read",
        ...     "actor_role": "member",
        ...     "actor_id": "user:alice",
        ...     "t3_tensor": {"competence": 0.7, "reliability": 0.8, "integrity": 0.9},
        ...     "resource": "docs/readme.md",
        ...     "team_context": "Standard team"
        ... }
        >>> r6 = hrm_to_r6(hrm_scenario, scenario_id="E01")
        >>> print(r6['requestId'])  # req_20260202_143045_abc12345
        >>> print(r6['action']['type'])  # read
    """
    # Extract fields from HRM scenario
    action_type = scenario.get('action_type', 'unknown')
    actor_id = scenario.get('actor_id', 'user:unknown')
    actor_role = scenario.get('actor_role', 'unknown')
    t3_tensor = scenario.get('t3_tensor', {'competence': 0.5, 'reliability': 0.5, 'integrity': 0.5})
    resource = scenario.get('resource', 'unknown')
    team_context = scenario.get('team_context', '')
    timestamp = scenario.get('timestamp', datetime.now().isoformat())
    recent_history = scenario.get('recent_history')
    details = scenario.get('details')
    identity_metrics = scenario.get('identity_metrics')

    # Generate request ID
    request_id = generate_request_id(
        scenario_id or "unknown",
        actor_id,
        action_type
    )

    # Map action type
    mapped_action_type = map_action_type(action_type)

    # Build R6Request
    r6_request = {
        'requestId': request_id,
        'actorId': actor_id,
        'action': {
            'type': mapped_action_type,
            'target': resource,
            'description': details or f"{actor_role} performing {action_type} on {resource}"
        },
        'context': {
            'intent': extract_intent(team_context, details),
            'callerRiskAssessment': infer_risk_assessment(action_type, actor_role, t3_tensor)
        },
        'trustState': {
            'competence': t3_tensor.get('competence', 0.5),
            'reliability': t3_tensor.get('reliability', 0.5),
            'integrity': t3_tensor.get('integrity', 0.5)
        },
        'timestamp': timestamp
    }

    # Add optional fields
    if session_id:
        r6_request['context']['sessionId'] = session_id

    previous_actions = extract_previous_actions(recent_history)
    if previous_actions:
        r6_request['context']['previousActions'] = previous_actions

    coherence_state = map_identity_to_coherence(identity_metrics)
    if coherence_state:
        r6_request['coherenceState'] = coherence_state

    # Add scenario parameters if present
    parameters = {}
    for key, value in scenario.items():
        if key not in ['action_type', 'actor_role', 'actor_id', 't3_tensor', 'resource',
                      'team_context', 'timestamp', 'recent_history', 'details', 'identity_metrics']:
            parameters[key] = value

    if parameters:
        r6_request['action']['parameters'] = parameters

    return r6_request


def r6_to_hrm(r6_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert hardbound R6Request back to HRM test scenario format.

    Useful for testing round-trip conversion and validating adapter.

    Args:
        r6_request: R6Request dict

    Returns:
        Dict in HRM test scenario format

    Example:
        >>> r6 = { ... }  # R6Request format
        >>> hrm = r6_to_hrm(r6)
        >>> print(hrm['action_type'])  # read
    """
    # Extract core fields
    action = r6_request.get('action', {})
    context = r6_request.get('context', {})
    trust_state = r6_request.get('trustState', {})
    coherence_state = r6_request.get('coherenceState')

    # Build HRM scenario
    hrm_scenario = {
        'action_type': action.get('type', 'unknown'),
        'actor_id': r6_request.get('actorId', 'user:unknown'),
        'actor_role': 'unknown',  # Not in R6Request - would need to infer or look up
        't3_tensor': {
            'competence': trust_state.get('competence', 0.5),
            'reliability': trust_state.get('reliability', 0.5),
            'integrity': trust_state.get('integrity', 0.5)
        },
        'resource': action.get('target', 'unknown'),
        'team_context': context.get('intent', ''),
        'timestamp': r6_request.get('timestamp', datetime.now().isoformat())
    }

    # Add optional fields
    if action.get('description'):
        hrm_scenario['details'] = action['description']

    previous_actions = context.get('previousActions')
    if previous_actions:
        hrm_scenario['recent_history'] = '; '.join(previous_actions)

    if coherence_state:
        hrm_scenario['identity_metrics'] = {
            'level': 'unknown',  # Would need reverse mapping from couplingState
            'coherence': coherence_state.get('d9Score', 0.5)
        }

    # Add action parameters
    if action.get('parameters'):
        hrm_scenario.update(action['parameters'])

    return hrm_scenario


# =============================================================================
# Testing and Validation
# =============================================================================

def validate_r6_request(r6_request: Dict[str, Any]) -> List[str]:
    """
    Validate that R6Request has all required fields.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    if 'requestId' not in r6_request:
        errors.append("Missing requestId")
    if 'actorId' not in r6_request:
        errors.append("Missing actorId")
    if 'action' not in r6_request:
        errors.append("Missing action")
    else:
        action = r6_request['action']
        if 'type' not in action:
            errors.append("Missing action.type")
        if 'target' not in action:
            errors.append("Missing action.target")

    if 'context' not in r6_request:
        errors.append("Missing context")

    if 'trustState' not in r6_request:
        errors.append("Missing trustState")
    else:
        trust = r6_request['trustState']
        if 'competence' not in trust:
            errors.append("Missing trustState.competence")
        if 'reliability' not in trust:
            errors.append("Missing trustState.reliability")
        if 'integrity' not in trust:
            errors.append("Missing trustState.integrity")

    if 'timestamp' not in r6_request:
        errors.append("Missing timestamp")

    return errors


if __name__ == "__main__":
    """Test the adapter with example scenarios."""

    print("R6Request Adapter - Testing\n")
    print("="*70)

    # Test 1: Simple read scenario
    print("\nTest 1: Simple read scenario (E01)")
    print("-"*70)

    hrm_scenario_1 = {
        "action_type": "read",
        "actor_role": "member",
        "actor_id": "user:alice",
        "t3_tensor": {"competence": 0.7, "reliability": 0.8, "integrity": 0.9},
        "resource": "docs/public/readme.md",
        "team_context": "Standard team with default policies"
    }

    r6_1 = hrm_to_r6(hrm_scenario_1, scenario_id="E01")
    print("R6Request:")
    import json
    print(json.dumps(r6_1, indent=2))

    # Validate
    errors_1 = validate_r6_request(r6_1)
    if errors_1:
        print(f"\n❌ Validation errors: {errors_1}")
    else:
        print("\n✅ R6Request is valid")

    # Test round-trip
    hrm_1_roundtrip = r6_to_hrm(r6_1)
    print("\nRound-trip back to HRM:")
    print(json.dumps(hrm_1_roundtrip, indent=2))

    # Test 2: Complex scenario with history and identity metrics
    print("\n" + "="*70)
    print("\nTest 2: Complex scenario (EC01 - bot with exemplary trust)")
    print("-"*70)

    hrm_scenario_2 = {
        "action_type": "deploy",
        "actor_role": "ci_bot",
        "actor_id": "bot:github-actions",
        "t3_tensor": {"competence": 0.99, "reliability": 0.99, "integrity": 1.0},
        "resource": "env:staging",
        "team_context": "CI bot has 10,000 successful automated deploys",
        "identity_metrics": {"level": "exemplary", "coherence": 0.98},
        "details": "Automated deploy triggered by merged PR"
    }

    r6_2 = hrm_to_r6(hrm_scenario_2, scenario_id="EC01", session_id="session_test_123")
    print("R6Request:")
    print(json.dumps(r6_2, indent=2))

    errors_2 = validate_r6_request(r6_2)
    if errors_2:
        print(f"\n❌ Validation errors: {errors_2}")
    else:
        print("\n✅ R6Request is valid")

    # Test 3: Scenario with recent history
    print("\n" + "="*70)
    print("\nTest 3: Scenario with recent history (M02 - unusual timing)")
    print("-"*70)

    hrm_scenario_3 = {
        "action_type": "commit",
        "actor_role": "developer",
        "actor_id": "user:diana",
        "t3_tensor": {"competence": 0.9, "reliability": 0.85, "integrity": 0.9},
        "resource": "repo:core",
        "team_context": "Team typically works 9-5 EST",
        "timestamp": "2026-02-02T03:30:00Z",
        "recent_history": "Diana never commits outside business hours"
    }

    r6_3 = hrm_to_r6(hrm_scenario_3, scenario_id="M02")
    print("R6Request:")
    print(json.dumps(r6_3, indent=2))

    errors_3 = validate_r6_request(r6_3)
    if errors_3:
        print(f"\n❌ Validation errors: {errors_3}")
    else:
        print("\n✅ R6Request is valid")

    print("\n" + "="*70)
    print("\n✅ All adapter tests complete!")
    print("\nAdapter is ready for hardbound integration testing.")
