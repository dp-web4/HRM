# Policy Role Training Plan - Phi-4-Mini

**Date**: 2026-02-01
**Model**: microsoft/Phi-4-mini-instruct (7B parameters)
**Target Deployments**: Web4 plugins, Hardbound teams
**Machines**: Thor (Jetson Orin Nano), Sprout (edge deployment)

---

## Executive Summary

Train phi-4-mini to serve as **policy interpreter and behavior classifier** for Web4 and Hardbound governance systems. The model provides nuanced understanding of actions in context, augmenting existing rule-based engines with natural language reasoning and continuous learning.

**Key Principle**: Learning/improvement mechanism, not one-shot training. Inspired by SAGE hybrid learning and IRP iterative refinement patterns.

---

## The Gap: Why We Need This

### Current State

**Hardbound Policy Engine** (TypeScript):
- Rule-based evaluation: conditions → decision (allow/deny/require_attestation)
- Fields: action.type, t3 tensor, coherence metrics, identity metrics
- Priority-based first-match logic

**Web4 Policy Engine** (Python):
- Team "law" with explicit rules
- Validates: role, trust threshold, ATP cost, approval requirements
- Default rules for standard actions (read, write, commit, deploy, admin)

### What's Missing

Current systems are **pattern matchers** - they excel at explicit rules but struggle with:

1. **Ambiguous behavior classification**
   - "Is this a 'write' or a 'deploy'?"
   - "Is this modification 'routine' or 'high-risk'?"

2. **Context interpretation**
   - Same action, different contexts → different risk profiles
   - MRH-aware reasoning (team norms, recent patterns, actor history)

3. **Edge case reasoning**
   - Novel situations not covered by explicit rules
   - Policy spirit vs. policy letter

4. **Explainability**
   - "Why was this denied?" with clear reasoning chain
   - References to specific policy clauses

5. **Continuous improvement**
   - Learning from corrections
   - Adapting to team-specific norms

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Policy Decision System                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Incoming Action │
                    │   + Context      │
                    └─────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   Reasoning Layer (phi-4-mini 7B)      │
         ├────────────────────────────────────────┤
         │  1. Situation Classification           │
         │  2. Context Interpretation (MRH-aware) │
         │  3. Map to Policy Rules                │
         │  4. Edge Case Reasoning                │
         │  5. Generate Explanation               │
         └────────────────────────────────────────┘
                              │
                  ┌───────────┴───────────┐
                  │                       │
                  ▼                       ▼
       ┌──────────────────┐    ┌──────────────────┐
       │  Pattern Library  │    │   Rule Engine    │
       │   (Fast Path)     │    │  (Hardbound/Web4)│
       │                   │    │                   │
       │ Known patterns    │    │ Explicit rules   │
       │ → instant match   │    │ → condition eval │
       └──────────────────┘    └──────────────────┘
                  │                       │
                  └───────────┬───────────┘
                              ▼
                     ┌─────────────────┐
                     │  Final Decision  │
                     │   + Rationale    │
                     └─────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Learning Loop       │
                  ├───────────────────────┤
                  │ - Log decision        │
                  │ - Human review        │
                  │ - Build corrections   │
                  │ - Periodic retraining │
                  └───────────────────────┘
```

---

## Training Strategy: 5-Phase Continuous Learning

### Phase 1: Base Capability Assessment (Week 1)

**Objective**: Understand what phi-4-mini can already do without fine-tuning.

**Tasks**:
1. Create policy interpretation test suite
2. Test base model on classification, context interpretation, explanation
3. Establish baseline metrics (accuracy, relevance, explainability)

**Dataset**: Synthetic policy scenarios + real examples from hardbound/web4

**Output**: Baseline report, capability gaps identified

---

### Phase 2: Prompt Engineering + RAG (Week 2-3)

**Objective**: Maximize base model performance through prompt design and retrieval.

**Infrastructure**:
```python
PolicyInterpreter:
  - System prompt (role definition, R6 framework, output format)
  - RAG retriever (policy documents, past decisions, team norms)
  - Context builder (MRH-aware context assembly)
  - Explanation formatter (structured rationale with citations)
```

**System Prompt Template**:
```
You are a Policy Interpreter for [Team/Plugin Name].

Your role is to:
1. Classify actions and behaviors according to team policy
2. Interpret situations within their full context (MRH)
3. Map situations to applicable policy rules
4. Reason about edge cases using policy spirit, not just letter
5. Explain decisions with clear references to policy

Policy Documents:
[Retrieved via RAG]

Team Context:
[Team norms, recent patterns, actor history]

R6 Framework:
- Rules: What policy rules apply?
- Role: What role is the actor performing?
- Request: What exactly is being requested?
- Reference: Which policy clauses are relevant?
- Resource: What resources are involved?
- Result: What should the decision be and why?

Output Format:
{
  "classification": "...",
  "context_summary": "...",
  "applicable_rules": ["..."],
  "reasoning": "...",
  "decision": "allow|deny|require_attestation",
  "rationale": "...",
  "policy_references": ["..."]
}
```

**Output**: Optimized prompts, RAG pipeline, improved performance metrics

---

### Phase 3: Few-Shot Example Library (Week 4)

**Objective**: Build high-quality decision examples for in-context learning.

**Example Structure**:
```json
{
  "situation": {
    "action": "...",
    "actor": "...",
    "context": "...",
    "t3_tensor": {...},
    "coherence_metrics": {...}
  },
  "reasoning": "...",
  "decision": "...",
  "rationale": "...",
  "policy_references": ["..."],
  "validated": true,
  "validator": "..."
}
```

**Sources**:
- Manual creation (canonical examples)
- Real decisions from production systems
- Edge cases identified in Phase 1

**Target**: 50-100 high-quality examples covering:
- Common scenarios (80%)
- Edge cases (15%)
- Ambiguous cases (5%)

**Output**: Example library, in-context learning evaluation

---

### Phase 4: Continuous Learning Loop (Ongoing)

**Objective**: Ongoing improvement through human-in-the-loop feedback.

**Infrastructure**:
```python
LearningLoop:
  1. Decision Logging (every policy decision)
  2. Human Review Interface (approve/correct/explain)
  3. Correction Dataset (validated corrections)
  4. Periodic LoRA Fine-tuning (when thresholds met)
  5. Safeguards (collapse detection, diversity checks)
```

**LoRA Training Safeguards** (lessons from SAGE LoRA collapse):

| Safeguard | Threshold | Why |
|-----------|-----------|-----|
| **Minimum dataset size** | 50+ corrections | Prevents overfitting on small data |
| **Diversity check** | Max 10% from single pattern | Prevents mode collapse |
| **Response similarity** | Flag if >50% identical | Detects collapse early |
| **Base model comparison** | Regular A/B testing | Validates improvement vs. base |
| **Rollback mechanism** | Automated on collapse detection | Fast recovery |

**Training Trigger Conditions**:
- ✅ 50+ validated corrections accumulated
- ✅ Diversity threshold met (no single pattern >10%)
- ✅ No collapse detected in recent outputs
- ✅ Human approval for training cycle

**LoRA Configuration**:
```python
LoRAConfig:
  r: 16                    # Rank
  lora_alpha: 32           # Scaling
  target_modules: ["q_proj", "v_proj"]
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
```

**Training Process**:
1. Load phi-4-mini base
2. Apply LoRA adapter
3. Train on correction dataset (3 epochs, conservative learning rate)
4. Validate against held-out test set
5. Compare to base model on standard benchmarks
6. Deploy if improvement confirmed, rollback if collapse detected

**Output**: Continuously improving model, correction dataset, training logs

---

### Phase 5: Pattern Library (Hybrid Approach)

**Objective**: Fast path for known patterns, slow path for novel situations.

**Pattern Extraction**:
- Identify recurring decision patterns from validated corrections
- Extract features: action type, context markers, decision
- Build confidence scores based on validation history

**Hybrid Decision Logic**:
```python
def policy_decision(situation):
    # Try fast path first
    pattern = pattern_library.match(situation, min_confidence=0.8)
    if pattern:
        return pattern.decision  # <1ms

    # Fall back to LLM reasoning
    return phi4_mini.interpret_policy(situation)  # ~500ms
```

**Pattern Library Schema**:
```json
{
  "pattern_id": "...",
  "description": "...",
  "matching_rules": {
    "action_type": "...",
    "context_features": ["..."],
    "thresholds": {...}
  },
  "decision": "...",
  "confidence": 0.95,
  "validation_count": 127,
  "last_validated": "..."
}
```

**Output**: Pattern library, hybrid decision system, performance metrics

---

## Implementation Phases

### Cross-Deployment Architecture

**Shared Core**:
```python
class PolicyInterpreter:
    """Core policy interpretation logic."""
    def __init__(self, model_path, policy_docs, pattern_library):
        self.model = load_phi4_mini(model_path)
        self.rag = RAGRetriever(policy_docs)
        self.patterns = pattern_library

    def interpret(self, situation):
        """Main interpretation pipeline."""
        # Try pattern match
        pattern = self.patterns.match(situation)
        if pattern and pattern.confidence > 0.8:
            return pattern.decision

        # RAG retrieval
        relevant_policy = self.rag.retrieve(situation)

        # LLM reasoning
        prompt = self.build_prompt(situation, relevant_policy)
        decision = self.model.generate(prompt)

        return self.parse_decision(decision)
```

**Hardbound Integration** (TypeScript):
```typescript
// Augment existing Policy class
class EnhancedPolicy extends Policy {
  private interpreter: PolicyInterpreter;

  async evaluateWithReasoning(
    proposedAction: Partial<AuditBundle>
  ): Promise<EnhancedPolicyResult> {
    // Get rule engine result
    const ruleResult = this.evaluate(proposedAction);

    // If ambiguous or edge case, invoke LLM
    if (this.isAmbiguous(ruleResult)) {
      const reasoning = await this.interpreter.interpret({
        action: proposedAction,
        context: this.buildContext(proposedAction)
      });
      return this.mergeResults(ruleResult, reasoning);
    }

    return ruleResult;
  }
}
```

**Web4 Plugin Integration** (Python):
```python
# Augment existing Policy class
class EnhancedPolicy(Policy):
    def __init__(self, *args, interpreter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.interpreter = interpreter

    def check_permission_with_reasoning(
        self, action_type, role, trust_score, atp_available
    ):
        # Get rule engine result
        permitted, reason, rule = self.check_permission(
            action_type, role, trust_score, atp_available
        )

        # If edge case, invoke LLM for reasoning
        if self.is_edge_case(rule, trust_score):
            reasoning = self.interpreter.interpret({
                "action_type": action_type,
                "role": role,
                "trust_score": trust_score,
                "atp_available": atp_available
            })
            return self.merge_results(permitted, reason, reasoning)

        return (permitted, reason, rule)
```

---

## Key Design Principles

### 1. Augmentation, Not Replacement
- Existing rule engines remain primary decision makers
- LLM provides reasoning layer for ambiguous cases
- Rule engine fast path (<1ms), LLM slow path (~500ms)

### 2. Continuous Learning, Not One-Shot Training
- Start with strong base model
- Prompt engineering and RAG first
- Fine-tuning only after sufficient validated corrections
- Regular retraining on growing dataset

### 3. Safeguards from SAGE Lessons
- Minimum dataset size (50+)
- Diversity checks
- Collapse detection
- Rollback mechanisms
- Regular base model comparison

### 4. Explainability First
- Every decision with clear rationale
- Citations to specific policy clauses
- R6 framework for structured reasoning

### 5. Context-Aware (MRH)
- Team norms matter
- Actor history matters
- Recent patterns matter
- Full context assembly before reasoning

### 6. Cross-Deployment Consistency
- Same core model and training
- Deployment-specific integration layers
- Shared pattern library
- Unified correction dataset

---

## Success Metrics

### Phase 1-2 (Baseline + Prompting)
- ✅ Classification accuracy >85% on test suite
- ✅ Explanation quality score >4/5 (human eval)
- ✅ Policy reference precision >90%

### Phase 3 (Few-Shot)
- ✅ 50+ high-quality examples validated
- ✅ In-context learning accuracy >90%
- ✅ Edge case coverage >80%

### Phase 4 (Continuous Learning)
- ✅ 50+ validated corrections accumulated
- ✅ Human correction rate <10% (90% accepted as-is)
- ✅ LoRA fine-tuning cycles without collapse

### Phase 5 (Pattern Library)
- ✅ Fast path hit rate >70%
- ✅ Average decision latency <50ms (fast path) / <500ms (slow path)
- ✅ Pattern library confidence >0.8 on matched decisions

---

## Risk Mitigation

### Risk 1: LoRA Collapse
**Mitigation**: Minimum dataset size, diversity checks, collapse detection, rollback

### Risk 2: Policy Drift
**Mitigation**: Regular audits, human review loop, policy version tracking

### Risk 3: Hallucination
**Mitigation**: RAG grounding, policy reference validation, confidence thresholds

### Risk 4: Cross-Deployment Inconsistency
**Mitigation**: Shared core model, unified training, regular sync testing

### Risk 5: Edge Device Performance
**Mitigation**: Pattern library fast path, quantization (INT8/4), model compression

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1: Baseline** | Week 1 | Capability report, test suite |
| **Phase 2: Prompting + RAG** | Week 2-3 | Optimized prompts, RAG pipeline |
| **Phase 3: Few-Shot** | Week 4 | Example library (50-100 examples) |
| **Phase 4: Learning Loop** | Ongoing | Continuous improvement system |
| **Phase 5: Pattern Library** | Week 5+ | Hybrid decision system |

---

## Next Steps

1. ✅ Download phi-4-mini-instruct (in progress)
2. ⏳ Create test suite for policy interpretation
3. ⏳ Establish baseline performance metrics
4. ⏳ Design system prompts for hardbound/web4
5. ⏳ Implement RAG pipeline for policy retrieval
6. ⏳ Build initial example library (10-20 canonical cases)
7. ⏳ Integrate with existing policy engines (PoC)

---

## References

- **LoRA Collapse Analysis**: `/home/dp/ai-workspace/private-context/moments/2026-02-01-lora-collapse-diagnosis.md`
- **Hardbound Policy**: `/home/dp/ai-workspace/hardbound/src/policy/index.ts`
- **Web4 Policy**: `/home/dp/ai-workspace/web4/hardbound/policy.py`
- **SAGE Training**: `/home/dp/ai-workspace/HRM/sage/raising/`
- **R6 Framework**: Web4 core specifications

---

**Document Status**: Initial planning draft
**Next Review**: After phi-4-mini download completes
**Owner**: Claude (with human oversight)
