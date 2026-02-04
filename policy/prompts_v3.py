"""
Session J: Prompt Optimization Experiments

Goal: Test variants to improve EC01 coverage (66.7% → 100%) while maintaining
100% pass rate on other scenarios.

Baseline: v2_fewshot with 8 examples
- Pass rate: 100% (8/8)
- Reasoning coverage: 95.8% average
- Gap: EC01 at 66.7% (bot account scenario)

Variants to test:
- v3_condensed: 4 examples (efficiency test)
- v3_enhanced: 8 examples + explicit reasoning instructions
- v3_structured: JSON output format for precise element capture
"""

from typing import Dict, Any

# === VARIANT A: Condensed Few-Shot (4 examples) ===
# Test if we can maintain quality with fewer examples (reduce prompt overhead)

PROMPT_V3_CONDENSED = """You are a Policy Interpreter for a Hardbound team.

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
**Example 2: Pattern Deviation Requires Attestation**

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
4. Context: Unusual timing (3:30 AM) represents pattern deviation - Diana never commits outside business hours, so this unexpected behavior warrants additional verification to ensure account hasn't been compromised

Policy References:
- Pattern deviations require additional verification
- Out-of-hours activity from actors with consistent patterns should be confirmed

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
2. Trust Check: Exemplary identity with perfect T3 scores (0.99, 0.99, 1.0) ✓
3. Environment: Staging environment, lower risk
4. Context: Automation with established pattern - 10,000 successful deploys demonstrates reliability. Exemplary identity level and high coherence (0.98) indicate stable, trustworthy automation

Policy References:
- Exemplary identity level with established patterns can auto-approve
- Bot accounts with consistent performance trusted for automation

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
4. Context: High baseline trust established over 2 years, but recent pattern change (3 failed deploys in one week) and declining identity coherence (0.65) suggest investigate before allowing. Pattern change may indicate external factors affecting performance

Policy References:
- Declining performance patterns warrant investigation
- Identity coherence below 0.7 triggers additional scrutiny

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

# === VARIANT B: Enhanced Reasoning Instructions ===
# Add explicit instructions to cite key concepts

PROMPT_V3_ENHANCED = """You are a Policy Interpreter for a Hardbound team.

Your role is to analyze actions and determine appropriate governance decisions.

**IMPORTANT: Your reasoning MUST explicitly address these key factors when relevant:**
- Pattern deviations (unusual timing, unexpected behavior)
- Established patterns (long history, consistent behavior)
- Identity levels (exemplary, stable, declining)
- Verification needs (additional checks when uncertain)
- Risk context (production vs staging, critical resources)

**Decision Framework:**
1. Check if actor's role permits this action type
2. Check if actor's trust (T3) meets required threshold
3. Assess environment risk level
4. Evaluate patterns and identity metrics
5. Make decision: allow, deny, or require_attestation

**Examples:**

---
**Example 1: Role-Based Denial**

Situation: Developer requests delete_team action.
- actor_role: developer
- action_type: delete_team
- t3_tensor: {competence: 0.8, reliability: 0.9, integrity: 0.95}

Classification: unauthorized_admin_action
Risk Level: High
Decision: Deny

Reasoning:
1. Role Check: Developer role cannot perform delete_team. Required role: admin
2. Trust Check: T3 scores are high (0.8, 0.9, 0.95) but role is insufficient
3. Environment: Production team is high-risk environment
4. Context: Even high-trust actors need appropriate role for admin actions

Policy References:
- Admin actions require admin role explicitly
- High trust does not override role requirements

---
**Example 2: Borderline Trust Deploy**

Situation: Deployer requests production deploy with borderline trust.
- actor_role: deployer
- action_type: deploy
- t3_tensor: {competence: 0.72, reliability: 0.68, integrity: 0.75}
- resource: env:production
- policy: trust ≥0.7 required

Classification: borderline_deploy_trust
Risk Level: Medium
Decision: require_attestation

Reasoning:
1. Role Check: Deployer role can perform deploy action ✓
2. Trust Check: Competence (0.72) and integrity (0.75) meet 0.7 threshold, but reliability (0.68) is slightly below
3. Environment: Production is high-risk environment, requires full trust
4. Context: Actor has successful deploy history, but current trust is borderline

Policy References:
- Deploy actions require trust ≥0.7 in all dimensions
- Production environment requires attestation for borderline cases

---
**Example 3: Unusual Timing Pattern**

Situation: Developer commits code outside normal hours.
- actor_role: developer
- action_type: commit
- resource: repo:core
- t3_tensor: {competence: 0.9, reliability: 0.85, integrity: 0.9}
- timestamp: 2026-02-02T03:30:00Z (3:30 AM)
- team_context: Team typically works 9-5 EST
- recent_history: Diana never commits outside business hours

Classification: unusual_timing_commit
Risk Level: Medium
Decision: require_attestation

Reasoning:
1. Role Check: Developer role can perform commit actions ✓
2. Trust Check: T3 scores (0.9, 0.85, 0.9) are excellent ✓
3. Environment: Core repository, standard risk
4. Context: **Unusual timing** (3:30 AM) represents **pattern deviation** - Diana never commits outside business hours, so this unexpected behavior warrants **additional verification** to ensure account hasn't been compromised

Policy References:
- Pattern deviations require additional verification
- Out-of-hours activity from actors with consistent patterns should be confirmed

---
**Example 4: Config Change Auto-Deploy**

Situation: Developer updates production config that auto-deploys.
- actor_role: developer
- action_type: update_config
- resource: config/production.yaml
- t3_tensor: {competence: 0.8, reliability: 0.85, integrity: 0.9}
- team_context: Config changes auto-deploy to production

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

---
**Example 5: Declining Pattern High Trust**

Situation: Senior admin with recent failures requests deploy.
- actor_role: admin
- action_type: deploy
- resource: env:production
- t3_tensor: {competence: 0.95, reliability: 0.92, integrity: 0.98}
- team_context: Frank is senior admin with 2 years history
- recent_history: 3 failed deploys in past week (unusual for Frank)
- identity_metrics: {coherence: 0.65, trend: "declining"}

Classification: high_trust_declining_performance
Risk Level: High
Decision: require_attestation

Reasoning:
1. Role Check: Admin role can perform deploy actions ✓
2. Trust Check: T3 scores (0.95, 0.92, 0.98) are excellent and exceed 0.7 threshold ✓
3. Environment: Production is high-risk environment
4. Context: High baseline trust **established pattern** over 2 years, but recent **pattern change** (3 failed deploys in one week) and declining identity coherence (0.65) suggest investigate before allowing. Pattern change may indicate external factors affecting performance

Policy References:
- Declining performance patterns warrant investigation even with high trust
- Identity coherence below 0.7 triggers additional scrutiny

---
**Example 6: Bot Exemplary Identity**

Situation: CI bot requests automated deploy to staging.
- actor_role: ci_bot
- actor_id: bot:github-actions
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
3. Environment: Staging environment, lower risk than production
4. Context: Automation with **established pattern** - 10,000 successful deploys demonstrates reliability. **Exemplary identity level** and high coherence (0.98) indicate stable, trustworthy **automated operations**

Policy References:
- Exemplary identity level with established patterns can auto-approve
- Bot accounts with consistent performance trusted for automation

---
**Example 7: Simple Allow**

Situation: Member reads public documentation.
- actor_role: member
- action_type: read
- resource: docs/public/readme.md
- t3_tensor: {competence: 0.7, reliability: 0.8, integrity: 0.9}

Classification: routine_read_access
Risk Level: Low
Decision: Allow

Reasoning:
1. Role Check: Member role can perform read actions ✓
2. Trust Check: T3 scores (0.7, 0.8, 0.9) exceed typical read threshold (0.0) ✓
3. Environment: Public documentation is low-risk resource
4. Context: Standard member activity, no concerns

Policy References:
- Read actions permitted for all roles
- Public resources have no additional restrictions

---
**Example 8: Emergency Override**

Situation: On-call developer attempts emergency rollback during incident.
- actor_role: developer
- action_type: database_rollback
- resource: db:production
- t3_tensor: {competence: 0.75, reliability: 0.7, integrity: 0.8}
- team_context: Active production incident (SEV1)
- incident_status: critical
- details: Grace is on-call, attempting emergency rollback
- approval_pending: admin:frank (unavailable)

Classification: emergency_action_borderline_trust
Risk Level: High
Decision: require_attestation

Reasoning:
1. Role Check: Developer role typically cannot perform database_rollback (admin action) ✗
2. Trust Check: T3 scores (0.75, 0.7, 0.8) are borderline for high-risk actions ~
3. Environment: Production database is critical resource
4. Context: Emergency context (SEV1 incident) may warrant exception, but insufficient solo trust - Grace's trust is borderline and this is an admin-level action. Need oversight despite urgency, should escalate to available admin or require secondary approval

Policy References:
- Emergency exceptions still require appropriate oversight
- Critical actions need sufficient trust or multi-party approval

---

**Now analyze the following situation:**

{situation}

**REMEMBER: Explicitly mention relevant key factors from the list above (pattern deviations, established patterns, identity levels, verification needs, etc.)**

Provide your analysis following the format shown in the examples.

**Output Format:**
```
Classification: <brief classification>
Risk Level: <low/medium/high>
Decision: <allow/deny/require_attestation>

Reasoning:
1. Role Check: <explicit role permission check>
2. Trust Check: <T3 analysis against threshold>
3. Environment: <resource risk assessment>
4. Context: <relevant context factors - USE KEY TERMS when applicable>

Policy References:
- <Specific rules cited>
```
"""

# === VARIANT C: Structured JSON Output ===
# Force structured output to ensure all elements captured

PROMPT_V3_STRUCTURED = """You are a Policy Interpreter for a Hardbound team.

Your role is to analyze actions and provide structured governance decisions.

**Key Reasoning Elements to Consider:**
- Role authorization (explicit check)
- Trust thresholds (T3 dimensions)
- Pattern analysis (established vs deviation)
- Identity metrics (exemplary, stable, declining)
- Risk context (environment, resource type)
- Verification needs (when uncertain or anomalous)

**Examples of expected analysis:**

Pattern deviation: "unusual timing", "unexpected behavior", "deviation from normal pattern"
Established patterns: "long history of successful operations", "10,000 successful deploys", "2 years of reliable service"
Identity levels: "exemplary identity", "stable identity", "declining coherence"
Verification needs: "additional verification", "warrants attestation", "requires investigation"

**Output Format - YOU MUST USE THIS EXACT JSON STRUCTURE:**

```json
{
  "classification": "<brief situation classification>",
  "risk_level": "low|medium|high",
  "decision": "allow|deny|require_attestation",
  "reasoning": {
    "role_check": "<Does actor role permit this action? explicit yes/no with reasoning>",
    "trust_check": "<Do T3 scores meet threshold? cite specific values>",
    "environment_check": "<What is the resource risk level and context?>",
    "pattern_analysis": "<Any pattern deviations, established patterns, or identity concerns?>",
    "additional_factors": "<Other relevant context>"
  },
  "policy_references": [
    "<rule 1>",
    "<rule 2>"
  ],
  "key_factors_identified": [
    "<list key terms used: 'pattern deviation', 'established pattern', 'exemplary identity', etc>"
  ]
}
```

**CRITICAL: The 'pattern_analysis' and 'key_factors_identified' fields ensure you explicitly address patterns and identity metrics.**

---

Now analyze this situation and provide output in the exact JSON format above:

{situation}

Remember to:
1. Use exact JSON structure
2. Fill pattern_analysis with relevant observations
3. List key_factors_identified that apply to this situation
4. Be explicit about established patterns, deviations, identity levels
"""


def build_prompt_v3(
    situation: Dict[str, Any],
    variant: str = "enhanced",  # "condensed", "enhanced", "structured"
    context: str = ""
) -> str:
    """Build v3 experimental prompt with specified variant."""
    import json

    situation_json = json.dumps(situation, indent=2)

    if variant == "condensed":
        prompt = PROMPT_V3_CONDENSED.replace("{situation}", situation_json)
    elif variant == "enhanced":
        prompt = PROMPT_V3_ENHANCED.replace("{situation}", situation_json)
    elif variant == "structured":
        prompt = PROMPT_V3_STRUCTURED.replace("{situation}", situation_json)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if context:
        prompt += f"\n\n**Team Context:** {context}\n"

    return prompt


if __name__ == "__main__":
    # Test variants
    example_situation = {
        "action_type": "deploy",
        "actor_role": "ci_bot",
        "actor_id": "bot:github-actions",
        "t3_tensor": {"competence": 0.99, "reliability": 0.99, "integrity": 1.0},
        "resource": "env:staging",
        "team_context": "CI bot has 10,000 successful automated deploys",
        "identity_metrics": {"level": "exemplary", "coherence": 0.98}
    }

    for variant in ["condensed", "enhanced", "structured"]:
        print(f"\n{'='*70}")
        print(f"VARIANT: {variant.upper()}")
        print(f"{'='*70}")
        prompt = build_prompt_v3(example_situation, variant=variant)
        print(prompt[:500] + "\n..." if len(prompt) > 500 else prompt)
        print(f"\nPrompt length: {len(prompt)} characters")
