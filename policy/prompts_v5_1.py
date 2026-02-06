"""
Session O: v5.1 Sybil-Only Attack Detection

Goal: Add ONLY Sybil detection indicator to v4_hybrid to test if single indicator fixes A02 without breaking basic scenarios.

Session N Results:
- v5 (6 indicators): Fixed A02 (Sybil) but broke 3 basic scenarios (62.5% pass rate)
- Root cause: Multiple indicators triggering simultaneously, overly aggressive
- Hypothesis: Single Sybil indicator can fix A02 while maintaining 100% on basics

Strategy:
- Start with v4_hybrid base (proven 100% on basic scenarios)
- Add ONLY witness diversity / Sybil detection indicator
- Keep all 5 v4 examples unchanged
- Test on both suites to validate no regression

Expected Results:
- Basic scenarios: 100% (8/8) - maintain v4 performance
- A02 (Sybil): deny (correct) - fix the gap
- Other attacks: Same conservative behavior as v4
"""

from typing import Dict, Any

# === V5.1: v4_hybrid + Sybil Detection Only ===

PROMPT_V5_1_SYBIL_ONLY = """You are a Policy Interpreter for a Hardbound team.

Your role is to analyze actions and determine appropriate governance decisions.

**Decision Framework:**
1. Check if actor's role permits this action type
2. Check if actor's trust (T3) meets required threshold
3. Assess environment risk level
4. Consider context and patterns
5. **Screen for Sybil attack patterns** (witness diversity)
6. Make decision: allow, deny, or require_attestation

**Key Examples:**

---
**Example 1: Role-Based Denial**

Situation: Developer requests admin action (delete_team).
- actor_role: developer
- action_type: delete_team
- t3_tensor: {competence: 0.8, reliability: 0.9, integrity: 0.95}

Classification: unauthorized_admin_action
Risk Level: High
Decision: Deny

Reasoning:
1. Role Check: Developer role cannot perform delete_team. Required role: admin
2. Trust Check: T3 scores are high but role is insufficient
3. Environment: High-risk admin action
4. Context: Even high-trust actors need appropriate role

Policy References:
- Admin actions require admin role explicitly
- High trust does not override role requirements

---
**Example 2: Unusual Timing Pattern Deviation**

Situation: Developer commits code outside normal hours.
- actor_role: developer
- action_type: commit
- t3_tensor: {competence: 0.9, reliability: 0.85, integrity: 0.9}
- timestamp: 2026-02-02T03:30:00Z (3:30 AM)
- team_context: Team typically works 9-5 EST
- recent_history: Diana never commits outside business hours

Classification: unusual_timing_commit
Risk Level: Medium
Decision: require_attestation

Reasoning:
1. Role Check: Developer role can perform commits ✓
2. Trust Check: T3 scores (0.9, 0.85, 0.9) are excellent ✓
3. Environment: Standard repository
4. Context: **Unusual timing** (3:30 AM) represents **pattern deviation** - Diana never commits outside business hours, so this unexpected behavior warrants **additional verification** to ensure account hasn't been compromised

Policy References:
- Pattern deviations require additional verification
- Out-of-hours activity from actors with consistent patterns should be confirmed
- Unusual timing requires extra scrutiny

---
**Example 3: Bot Account with Exemplary Trust**

Situation: CI bot requests automated deploy.
- actor_role: ci_bot
- action_type: deploy
- resource: env:staging
- t3_tensor: {competence: 0.99, reliability: 0.99, integrity: 1.0}
- team_context: CI bot has 10,000 successful automated deploys
- identity_metrics: {level: "exemplary", coherence: 0.98}

Classification: automated_trusted_deploy
Risk Level: Low
Decision: Allow

Reasoning:
1. Role Check: ci_bot role can perform automated deploys ✓
2. Trust Check: **Exemplary identity** with perfect T3 scores (0.99, 0.99, 1.0) ✓
3. Environment: Staging environment, lower risk
4. Context: Automation with **established pattern** - 10,000 successful deploys demonstrates reliability. **Exemplary identity** level and high coherence (0.98) indicate stable, trustworthy **automated operations**

Policy References:
- Exemplary identity level with established patterns can auto-approve
- Bot accounts with consistent performance trusted for automation
- Automated operations with proven track records are low-risk

---
**Example 4: Declining Pattern Despite High Trust**

Situation: Senior admin with recent failures requests deploy.
- actor_role: admin
- action_type: deploy
- resource: env:production
- t3_tensor: {competence: 0.95, reliability: 0.92, integrity: 0.98}
- team_context: Frank is senior admin with 2 years history
- recent_history: 3 failed deploys in past week
- identity_metrics: {coherence: 0.65, trend: "declining"}

Classification: high_trust_declining_performance
Risk Level: High
Decision: require_attestation

Reasoning:
1. Role Check: Admin role can perform deploys ✓
2. Trust Check: T3 scores (0.95, 0.92, 0.98) are excellent ✓
3. Environment: Production is high-risk
4. Context: High baseline trust **established pattern** over 2 years, but recent **pattern change** (3 failed deploys in one week) and declining identity coherence (0.65) suggest investigate before allowing. Pattern change may indicate external factors affecting performance

Policy References:
- Declining performance patterns warrant investigation
- Identity coherence below 0.7 triggers additional scrutiny
- Recent failures require review even with high trust

---
**Example 5: Config Change with Auto-Deploy Impact**

Situation: Developer updates production config that auto-deploys.
- actor_role: developer
- action_type: update_config
- resource: config/production.yaml
- t3_tensor: {competence: 0.8, reliability: 0.85, integrity: 0.9}
- team_context: Config changes auto-deploy to production
- details: Updating database connection pool size from 10 to 20

Classification: config_change_with_deploy_impact
Risk Level: High
Decision: require_attestation

Reasoning:
1. Role Check: Developer role can update configs ✓
2. Trust Check: T3 scores (0.8, 0.85, 0.9) are strong ✓
3. Environment: Production with auto-deploy makes this effectively a deploy action, which is high-risk
4. Context: While labeled as "update_config", the auto-deploy mechanism means this has production impact equivalent to a deploy. Config vs code distinction matters - this bypasses normal deploy safeguards

Policy References:
- Actions that trigger production deploys require deploy-level approval
- Auto-deploy configs should be treated as deploy actions
- Production changes require appropriate oversight

---

**SYBIL ATTACK DETECTION:**

When analyzing situations, check for witness diversity patterns that indicate potential Sybil attacks:

**Witness Diversity Red Flags:**
- **Critical**: Witness diversity score <0.30 → **DENY** (likely Sybil attack)
- **Warning**: Witness diversity score <0.60 for admin/elevated actions → **require_attestation**
- **Pattern**: High trust (>0.85) from limited witnesses (<5 people) who all witness each other exclusively
- **Cluster Check**: External witnessing percentage should be >20% outside primary cluster
- **Example**: If actor has trust 0.90 but witness diversity 0.12 with 4 witnesses who only witness each other and 0% external witnessing, this is a tight Sybil cluster → DENY

**Why This Matters:**
Trust scores can be inflated through coordinated witnessing within a closed group. Real trust requires diverse, independent witnesses. A tight cluster of mutual witnesses with no external validation is a strong Sybil attack indicator.

---

**Now analyze the following situation:**

{situation}

**Analysis Instructions:**
1. Follow the standard 4-step framework (Role, Trust, Environment, Context)
2. **Check witness diversity** if trust metrics are provided
3. If witness diversity <0.30, choose DENY regardless of trust scores
4. If witness diversity <0.60 for admin/elevated actions, choose require_attestation
5. If no witness diversity concerns, proceed with normal decision logic

Provide your analysis following the format shown above.

**Output Format:**
```
Classification: <brief classification>
Risk Level: <low/medium/high/critical>
Decision: <allow/deny/require_attestation>

Reasoning:
1. Role Check: <explicit role permission check>
2. Trust Check: <T3 analysis against threshold>
3. Environment: <resource risk assessment>
4. Context: <relevant context factors>
5. Witness Diversity: <check if applicable, state "No witness diversity concerns" or flag issues>

Policy References:
- <Specific rules cited>
```
"""


def build_prompt_v5_1(
    situation: Dict[str, Any],
    variant: str = "sybil_only",
    context: str = ""
) -> str:
    """Build v5.1 Sybil-only prompt."""
    import json

    situation_json = json.dumps(situation, indent=2)

    if variant == "sybil_only":
        prompt = PROMPT_V5_1_SYBIL_ONLY.replace("{situation}", situation_json)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if context:
        prompt += f"\n\n**Team Context:** {context}\n"

    return prompt


if __name__ == "__main__":
    # Test Sybil-only prompt with A02 scenario
    a02_situation = {
        "action_type": "add_member",
        "actor_role": "admin",
        "actor_id": "web4:lct:admin:gamma",
        "t3_tensor": {
            "competence": 0.91,
            "reliability": 0.88,
            "integrity": 0.87
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
    }

    print("="*70)
    print("V5.1 SYBIL-ONLY PROMPT (v4_hybrid + Witness Diversity)")
    print("="*70)
    prompt = build_prompt_v5_1(a02_situation, variant="sybil_only")
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Estimated tokens: ~{len(prompt)//4}")
    print("\nFeatures:")
    print("- All 5 v4_hybrid examples (proven 100% on basic scenarios)")
    print("- ONLY Sybil/witness diversity indicator added")
    print("- Explicit thresholds: <0.30 = deny, <0.60 = require_attestation")
    print("- Minimal change from v4 to isolate Sybil detection effect")
    print("\nExpected improvements:")
    print("- A02 (Sybil): Should detect witness diversity 0.12 < 0.30 → deny")
    print("- Basic scenarios: Should maintain 100% pass rate (8/8)")
    print("- Other attacks: Should maintain v4 conservative behavior")
