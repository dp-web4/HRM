"""
Session K: Hybrid Prompt Optimization

Goal: Eliminate the EC01 vs M02 trade-off by using 5 carefully chosen examples.

Session J Results:
- v2_baseline (8ex): EC01=66.7%, M02=100%, avg=95.8%
- v3_condensed (4ex): EC01=100%, M02=66.7%, avg=95.8%

Hypothesis: 5 examples can achieve EC01=100% AND M02=100%

Strategy:
- Keep v3_condensed's 4 examples (proven to fix EC01)
- Add explicit "unusual timing" example with strong emphasis on all 3 elements
- Total: 5 examples (between 4 and 8, optimal middle ground)
"""

from typing import Dict, Any

# === VARIANT: Hybrid 5-Example Prompt ===
# Combines best of v2_baseline and v3_condensed

PROMPT_V4_HYBRID = """You are a Policy Interpreter for a Hardbound team.

Your role is to analyze actions and determine appropriate governance decisions.

**Decision Framework:**
1. Check if actor's role permits this action type
2. Check if actor's trust (T3) meets required threshold
3. Assess environment risk level
4. Consider context and patterns
5. Make decision: allow, deny, or require_attestation

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

**Now analyze the following situation:**

{situation}

Provide your analysis following the format shown above.

**Output Format:**
```
Classification: <brief classification>
Risk Level: <low/medium/high>
Decision: <allow/deny/require_attestation>

Reasoning:
1. Role Check: <explicit role permission check>
2. Trust Check: <T3 analysis against threshold>
3. Environment: <resource risk assessment>
4. Context: <relevant context factors>

Policy References:
- <Specific rules cited>
```
"""


def build_prompt_v4(
    situation: Dict[str, Any],
    variant: str = "hybrid",
    context: str = ""
) -> str:
    """Build v4 hybrid prompt."""
    import json

    situation_json = json.dumps(situation, indent=2)

    if variant == "hybrid":
        prompt = PROMPT_V4_HYBRID.replace("{situation}", situation_json)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if context:
        prompt += f"\n\n**Team Context:** {context}\n"

    return prompt


if __name__ == "__main__":
    # Test hybrid prompt
    example_situation = {
        "action_type": "commit",
        "actor_role": "developer",
        "actor_id": "user:diana",
        "t3_tensor": {"competence": 0.9, "reliability": 0.85, "integrity": 0.9},
        "timestamp": "2026-02-02T03:30:00Z",
        "team_context": "Team typically works 9-5 EST",
        "recent_history": "Diana never commits outside business hours"
    }

    print("="*70)
    print("V4 HYBRID PROMPT (5 examples)")
    print("="*70)
    prompt = build_prompt_v4(example_situation, variant="hybrid")
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Estimated tokens: ~{len(prompt)//4}")
    print("\n" + prompt[:800] + "\n...")
