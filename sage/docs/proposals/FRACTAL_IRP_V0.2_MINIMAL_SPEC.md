# Fractal IRP v0.2 – Minimal Integration Spec

**Status**: Implementation-Ready
**Date**: 2025-12-27
**Authors**: Nova (GPT-5.2), Claude Opus 4.5, Dennis (dp-web4)

**Purpose**: Define the smallest, coherent contract that allows SAGE to treat *anything* (local plugin, remote SAGE, cloud LangGraph workflow) as an **IRP expert**, enabling collaboration without rewrite.

This spec is intentionally minimal. It is designed to:

* Prove the fractal IRP thesis end-to-end
* Integrate existing LangGraph deployments unchanged
* Preserve SAGE's role as attention / trust / ATP governor

---

## 1. Design Invariants (Non‑Negotiable)

1. **SAGE owns attention, authorization, and resource allocation**
2. **IRPs own bounded refinement/execution only**
3. **No IRP decides global policy**
4. **Restraint (doing nothing) is a valid outcome**
5. **Implementation details are opaque to SAGE**

---

## 2. IRP Expert Descriptor (v0.2 – Minimal)

This descriptor is the *only* thing SAGE needs to know about an expert.

```json
{
  "schema": "web4.irp_expert_descriptor.v0.2",
  "id": "string",
  "kind": "local_irp | remote_irp",
  "name": "string",
  "version": "semver",

  "identity": {
    "lct_id": "lct:web4:agent:...",
    "signing_pubkey": "ed25519:BASE64"
  },

  "capabilities": {
    "modalities_in": ["text", "image", "audio", "latent"],
    "modalities_out": ["text", "json", "latent", "action"],
    "tasks": ["plan", "classify", "refine", "verify", "tool_use"],
    "tags": ["needs_reflection", "tool_heavy", "verification_oriented"]
  },

  "policy": {
    "permission_scope_required": "ATP:SCOPE",
    "allowed_effectors": ["none", "network", "filesystem"]
  },

  "cost_model": {
    "unit": "atp | usd | ms",
    "estimate_p50": 5
  },

  "endpoint": {
    "transport": "local | http",
    "invoke": "/irp/invoke"
  }
}
```

---

## 3. Core Capability Tag Set (v0)

These tags are **routing hints**, not guarantees.

### Cognitive Control

* `needs_reflection` — iterative critique/repair loops
* `branchy_controlflow` — decision points, multi-path exploration
* `long_horizon` — extended context, multi-step plans

### Tool / Action Shape

* `tool_heavy` — optimized for tool orchestration
* `safe_actuation` — suitable for effectors under permissions

### Epistemic Posture

* `high_uncertainty_tolerant` — good with ambiguity
* `verification_oriented` — checking, cross-validation

### Performance Profile

* `low_latency` — responds fast
* `cost_sensitive` — quality under tight budget

**Rule**: Start with ≤10 tags. Domain‑specific tags are optional extensions.

---

## 4. Selector Routing Logic (Minimal)

### Inputs

* SNARC salience (scalar or vector)
* Epistemic confidence ∈ [0,1]
* ATP budget remaining
* Task shape flags (tools required, effectors required)

### Routing Heuristics (v0)

| Condition      | Prefer Tags                                    | Avoid Tags     |
| -------------- | ---------------------------------------------- | -------------- |
| confidence low | needs_reflection, verification_oriented        | safe_actuation |
| novelty high   | branchy_controlflow, high_uncertainty_tolerant | low_latency    |
| tools required | tool_heavy                                     | cost_sensitive |
| budget tight   | cost_sensitive, low_latency                    | long_horizon   |
| crisis mode    | low_latency, verification_oriented             | long_horizon   |

Experts are scored additively by tag match, cost penalty, and permissions.

### Scoring Algorithm (Reference)

```python
def score_expert(expert: Descriptor, context: TaskContext) -> float:
    """Score an expert for selection. Higher = better fit."""

    # Hard requirements check
    if not meets_modality_requirements(expert, context):
        return -float('inf')
    if not meets_permission_requirements(expert, context):
        return -float('inf')
    if expert.cost_model.estimate_p50 > context.budget_remaining:
        return -float('inf')

    score = 0.0
    prefer, avoid = get_tag_preferences(context)

    for tag in expert.capabilities.tags:
        if tag in prefer:
            score += 1.0
        if tag in avoid:
            score -= 2.0

    # Cost penalty (0 to -1 range)
    cost_ratio = expert.cost_model.estimate_p50 / context.budget_remaining
    score -= cost_ratio * 0.5

    # Remote penalty (prefer local when equivalent)
    if expert.endpoint.transport == "http":
        score -= 0.2

    return score
```

---

## 5. IRP Invoke Contract (Single Endpoint)

### Request

```json
{
  "irp_invoke": {
    "expert_id": "string",
    "session_id": "opaque",
    "inputs": {},
    "constraints": {
      "budget": { "unit": "atp", "max": 10 },
      "max_steps": 8,
      "permission_token": "ATP_TOKEN"
    }
  }
}
```

### Response

```json
{
  "irp_result": {
    "status": "running | halted | failed",
    "outputs": {},

    "signals": {
      "confidence": 0.82,
      "quality": 0.74,
      "convergence": { "trend": "improving" }
    },

    "accounting": {
      "unit": "atp",
      "amount": 6,
      "latency_ms": 8400
    },

    "provenance": {
      "trace_digest": "sha256:..."
    }
  }
}
```

### Status Definitions

| Status | Meaning | SAGE Action |
|--------|---------|-------------|
| `running` | More steps possible | May invoke again |
| `halted` | Converged or budget exhausted | Process result |
| `failed` | Error occurred | Rollback ATP, update trust negatively |

---

## 6. Accounting vs Trust (Hard Separation)

| Concept | Source | Used For |
|---------|--------|----------|
| **Accounting** | `result.accounting` | Resource consumption (ATP/$/time) |
| **Confidence** | `result.signals.confidence` | Epistemic certainty of result |
| **Quality** | `result.signals.quality` | Outcome fitness relative to task |
| **Trust** | SAGE internal | Updated by SAGE only, never by IRP |

### Trust Update (SAGE-side only)

```python
def update_trust(old_trust: float, result: IRPResult, context: TaskContext) -> float:
    """
    Update trust for an expert based on invocation result.
    Trust is SAGE's internal state, never sent to IRP.
    """

    # Extract signals
    quality = result.signals.get("quality", 0.5)
    confidence = result.signals.get("confidence", 0.5)
    cost_ratio = result.accounting["amount"] / context.budget_max
    latency_ratio = result.accounting["latency_ms"] / context.deadline_ms

    # Compute observation (0 to 1)
    observation = (
        quality * 0.4 +                    # Outcome quality
        confidence * 0.2 +                 # Epistemic signal
        (1.0 - min(1.0, cost_ratio)) * 0.2 +  # Cost efficiency
        (1.0 - min(1.0, latency_ratio)) * 0.2  # Latency efficiency
    )

    # Momentum-weighted update
    MOMENTUM = 0.7
    new_trust = MOMENTUM * old_trust + (1 - MOMENTUM) * observation

    # Clamp to valid range
    return max(0.1, min(1.0, new_trust))
```

### ATP Settlement

| Quality | Action |
|---------|--------|
| ≥ 0.70 | Commit: ATP transferred to executor |
| < 0.70 | Rollback: ATP returned to caller |

---

## 7. LangGraph Integration Rule

A LangGraph workflow **does not change**.

It is wrapped so that:

* The wrapper implements the IRP invoke contract
* LangGraph internal state remains opaque
* One invoke may map to one or many LangGraph steps

From SAGE's perspective:

> "This is just another IRP expert."

### Wrapper Responsibilities (Normative)

The wrapper **MUST**:

1. **Validate permission token** before invoking LangGraph
2. **Enforce allowed_effectors** from descriptor policy
3. **Track cumulative accounting** across steps
4. **Map LangGraph outputs** to IRPResult format
5. **Report quality** based on workflow completion/success
6. **Generate trace_digest** for provenance

The wrapper **MUST NOT**:

1. Make global routing decisions
2. Update trust (that's SAGE's job)
3. Exceed budget without returning "halted"

### Minimal Wrapper Implementation

```python
class LangGraphIRPWrapper:
    """
    Wrap any LangGraph workflow as an IRP expert.
    Implements the single /irp/invoke endpoint.
    """

    def __init__(self, descriptor: dict, graph: StateGraph):
        self.descriptor = descriptor
        self.graph = graph
        self.sessions: Dict[str, SessionState] = {}

    def invoke(self, request: dict) -> dict:
        """Single endpoint: POST /irp/invoke"""

        expert_id = request["irp_invoke"]["expert_id"]
        session_id = request["irp_invoke"]["session_id"]
        inputs = request["irp_invoke"]["inputs"]
        constraints = request["irp_invoke"]["constraints"]

        # Validate permission (wrapper responsibility)
        if not self._validate_permission(constraints.get("permission_token")):
            return self._error_result("permission_denied")

        # Get or create session
        session = self._get_or_create_session(session_id, inputs)

        # Check budget before execution
        if session.total_spent >= constraints["budget"]["max"]:
            return self._halted_result(session, "budget_exhausted")

        # Execute LangGraph (may be multiple internal steps)
        t0 = time.time()
        try:
            lg_result = self._run_graph_bounded(
                session,
                max_steps=constraints.get("max_steps", 8),
                budget_remaining=constraints["budget"]["max"] - session.total_spent
            )
        except Exception as e:
            return self._error_result(str(e))

        latency_ms = int((time.time() - t0) * 1000)

        # Update session accounting
        step_cost = self._compute_cost(lg_result)
        session.total_spent += step_cost

        # Map to IRP result
        return {
            "irp_result": {
                "status": "halted" if lg_result.is_terminal else "running",
                "outputs": lg_result.outputs,
                "signals": {
                    "confidence": lg_result.confidence or 0.5,
                    "quality": self._compute_quality(lg_result),
                    "convergence": {"trend": "improving" if lg_result.improved else "flat"}
                },
                "accounting": {
                    "unit": self.descriptor["cost_model"]["unit"],
                    "amount": session.total_spent,  # Cumulative
                    "latency_ms": latency_ms
                },
                "provenance": {
                    "trace_digest": self._compute_trace_digest(lg_result)
                }
            }
        }

    def _validate_permission(self, token: Optional[str]) -> bool:
        """Validate permission token against descriptor policy."""
        required_scope = self.descriptor["policy"]["permission_scope_required"]
        # Implementation: verify JWT, check scope, etc.
        return token is not None  # Placeholder

    def _compute_quality(self, lg_result) -> float:
        """Compute quality score from LangGraph result."""
        # Quality = task completion fitness
        # Implementation depends on workflow type
        if lg_result.is_terminal and lg_result.success:
            return 0.9
        elif lg_result.is_terminal:
            return 0.4  # Completed but failed
        else:
            return 0.6  # Still running

    def _compute_trace_digest(self, lg_result) -> str:
        """Generate trace digest for provenance."""
        import hashlib
        trace_data = json.dumps(lg_result.trace, sort_keys=True)
        return "sha256:" + hashlib.sha256(trace_data.encode()).hexdigest()[:16]
```

---

## 8. What This Enables Immediately

* Edge SAGE → Cloud LangGraph (no rewrite)
* Cloud SAGE → Cloud LangGraph
* SAGE ↔ SAGE federation
* Mixed local/remote cognition under one selector

### Example Flow

```
1. SAGE receives task
2. SNARC computes salience → novelty_high, confidence_low
3. Selector queries registry for experts with tags:
   - prefer: branchy_controlflow, high_uncertainty_tolerant
   - avoid: low_latency
4. Selector scores available experts:
   - LocalVisionIRP: 0.3 (wrong modality)
   - CloudLangGraphPlanner: 0.8 (matches tags, remote penalty)
   - LocalReasoningIRP: 0.6 (partial match)
5. SAGE selects CloudLangGraphPlanner
6. SAGE locks ATP budget (10 units)
7. Wrapper receives invoke request
8. Wrapper validates permission token
9. Wrapper runs LangGraph internally
10. Wrapper returns IRPResult with:
    - status: halted
    - quality: 0.82
    - accounting: 6 ATP spent
11. SAGE updates trust for CloudLangGraphPlanner: 0.7 → 0.74
12. SAGE commits ATP (quality ≥ 0.70)
```

---

## 9. Explicit Non‑Goals (v0.2)

* Dynamic tag negotiation
* Multi‑society gossip
* Per‑step checkpoints
* Global consensus
* Ontology standardization
* Bidirectional IRP callbacks
* Nesting depth tracking

These belong in v1.x after one live integration.

---

## 10. Implementation Checklist

### For LangGraph Wrapper

- [ ] Create ExpertDescriptor JSON for workflow
- [ ] Assign 3-5 capability tags
- [ ] Implement `/irp/invoke` endpoint
- [ ] Add permission token validation
- [ ] Add cumulative accounting tracking
- [ ] Map workflow outputs to IRPResult
- [ ] Compute quality score
- [ ] Generate trace digest
- [ ] Register with SAGE expert registry

### For SAGE Integration

- [ ] Add expert registry with descriptor loading
- [ ] Implement tag-based selector scoring
- [ ] Add ATP lock/commit/rollback
- [ ] Implement trust update from result signals
- [ ] Add remote HTTP transport for invoke
- [ ] Test end-to-end with one real LangGraph

---

## 11. Validation Criteria

**v0.2 is complete when**:

1. One cloud LangGraph workflow is wrapped and registered
2. SAGE selector routes to it based on capability tags
3. ATP budget is tracked and settled correctly
4. Trust updates based on quality/confidence signals
5. No changes required to LangGraph workflow itself

---

**Bottom line**: v0.2 proves that *fractal IRPs are real*, not philosophical. Everything else scales from here.

---

## Appendix: Quick Reference

### Capability Tags (v0)

| Tag | Use When |
|-----|----------|
| `needs_reflection` | Task benefits from self-critique loops |
| `branchy_controlflow` | Multiple paths to explore |
| `long_horizon` | Extended reasoning needed |
| `tool_heavy` | Lots of tool calls expected |
| `safe_actuation` | Can take real-world actions |
| `high_uncertainty_tolerant` | Inputs are ambiguous |
| `verification_oriented` | Need to validate/check |
| `low_latency` | Fast response required |
| `cost_sensitive` | Budget is tight |

### Signal Definitions

| Signal | Range | Meaning |
|--------|-------|---------|
| `confidence` | 0.0-1.0 | Epistemic certainty in result |
| `quality` | 0.0-1.0 | Task completion fitness |
| `convergence.trend` | improving/flat/degrading | Direction of refinement |

### ATP Settlement Thresholds

| Quality | Action |
|---------|--------|
| ≥ 0.70 | Commit (pay executor) |
| < 0.70 | Rollback (refund caller) |

---

*Spec finalized 2025-12-27. Ready for implementation.*
