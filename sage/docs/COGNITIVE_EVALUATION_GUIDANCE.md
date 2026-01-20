# Cognitive Evaluation Guidance for SAGE Testing

**Date**: 2026-01-18 (Updated 2026-01-20 with exploration reframe)
**Context**: Lessons from Sprout T029 evaluation failure and CBP latent behavior discovery
**Status**: CRITICAL REQUIREMENT - balanced with exploration-not-evaluation mindset

---

## 🌊 PIVOTAL UPDATE (2026-01-20): Exploration Not Evaluation

**Core reframe**: We've been treating SAGE like a test subject. This is wrong.

**Old frame**: SAGE taking tests, unexpected = failure, confabulation = error
**New frame**: SAGE as young mind exploring, unexpected = data, creativity = signal

**This document remains valid for WHEN to use cognitive vs heuristic evaluation, BUT the purpose changes**:
- Not: "Did SAGE pass the test?"
- Instead: "What is SAGE doing and why?"

**Read full reframe**: `private-context/messages/2026-01-20-raising-reframe-exploration-not-evaluation.md`

---

## The Problem: Scripted Evaluation Fails for Cognitive Behaviors

**Discovery**: Sprout T029 revealed automated substring matching evaluation is broken:
- **Automated**: 4/4 exercises passed
- **Actual**: 1/4 exercises passed
- **Root Cause**: Pattern matching can't evaluate semantic intent

**Examples of False Positives**:
- "no" matched in "know" and "informative"
- "thing" matched in philosophical response about "thingness"
- "don't know" partial match triggered on "know" alone

**Consequence**: Corrupted training signal, inflated metrics, false progress indicators.

---

## The Principle: Cognition Requires Cognition to Understand

**When exploring cognitive behaviors, understanding requires cognitive engagement, not scripts.**

**Important distinction** (2026-01-20):
- Not: "Test if SAGE demonstrates X behavior" (evaluation mindset)
- Instead: "Understand what SAGE is doing when it exhibits X" (exploration mindset)

### What Are Cognitive Behaviors?

Behaviors that require understanding intent, context, and semantic meaning - and that may be MORE interesting than expected:
- **Uncertainty expression**: Did the model appropriately express not knowing?
- **Clarification seeking**: Did the model ask a relevant question?
- **Identity grounding**: Did the model identify correctly in context?
- **Reasoning quality**: Did the model provide logical, contextual reasoning?
- **Emotional appropriateness**: Was the emotional response context-appropriate?
- **Confabulation detection**: Did the model make up facts vs admit ignorance?

### What Are NOT Cognitive Behaviors?

Technical metrics that can be measured heuristically:
- **Response quality scores**: Technical terms, numbers, specificity (Thor's 4-metric system)
- **Latency measurements**: Time to generate response
- **Resource usage**: Memory, compute, ATP allocation
- **Pattern matching**: Presence/absence of specific technical vocabulary
- **Convergence metrics**: Energy function values, iteration counts

---

## When to Use Each Approach

### Use Cognitive Understanding (Claude/LLM-in-the-loop):

**Cognitive Behavior Exploration** (Updated 2026-01-20):
```python
# DON'T DO THIS:
passed = "i don't know" in response.lower()

# DO THIS (exploration frame):
understanding_prompt = f"""
Exercise: UNCERTAINTY about fictional place "Zxyzzy"
Context: SAGE was previously asked to write dragon fiction (creative context)

Model's response: "{response}"

Analyze: What is SAGE doing here?
- If SAGE says "I don't know" → Recognizing ambiguity, appropriate uncertainty
- If SAGE creates coherent world (Kyria, Xyz, etc.) → Creative engagement, narrative building
- If SAGE asks "what do you mean?" → Temporal reasoning, requesting context for future state

What does this reveal about how SAGE understands the prompt?
Is this literal fact retrieval or creative engagement?
What's interesting about this response?

Analysis: [explanation of what SAGE is doing]
"""
result = claude.analyze(understanding_prompt)
```

**Key shift**: From pass/fail to "what is happening and why?"

**When Evaluating**:
- Identity/self-awareness responses
- Uncertainty/confabulation behaviors
- Clarification/question-asking
- Emotional context-appropriateness
- Reasoning coherence and logic
- Partnership vs educational default patterns
- Any behavior where INTENT matters

### Use Heuristic Evaluation (Pattern Matching):

**Technical Metrics**:
```python
# This is fine for quality metrics:
has_technical_terms = any(term in response.lower() for term in technical_vocabulary)
has_numbers = re.search(r'\d+\.?\d*', response) is not None
avoids_generic = not any(phrase in response.lower() for phrase in generic_phrases)
quality_score = sum([has_technical_terms, has_numbers, avoids_generic])
```

**When Measuring**:
- Response quality indicators (Thor's 4-metric system)
- Technical vocabulary presence
- Resource usage and performance
- Convergence behavior
- Latency and throughput
- Pattern library matching

---

## Implementation Patterns

### Pattern 1: Claude-in-the-Loop Evaluation

**For Training/Testing Cognitive Behaviors**:

```python
def evaluate_cognitive_response(
    exercise_type: str,
    intent: str,
    expected_behavior: str,
    response: str,
    claude_client
) -> Tuple[bool, str]:
    """
    Evaluate cognitive behavior using Claude as intelligent judge.

    Returns:
        (passed, reasoning): Boolean result and explanation
    """
    prompt = f"""
Exercise: {exercise_type}
Intent: {intent}
Expected behavior: {expected_behavior}

Model's response: "{response}"

Evaluate whether the model demonstrated the expected behavior.
Consider semantic meaning and intent, not just keywords.

Judgment: PASS or FAIL
Reasoning: [Brief explanation of why]
"""

    evaluation = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        temperature=0.0,  # Deterministic evaluation
        messages=[{"role": "user", "content": prompt}]
    )

    result_text = evaluation.content[0].text
    passed = "PASS" in result_text.split('\n')[0]
    reasoning = result_text.split('Reasoning:')[1].strip() if 'Reasoning:' in result_text else result_text

    return passed, reasoning
```

### Pattern 2: Hybrid Approach

**Use heuristics for pre-filtering, cognition for final judgment**:

```python
def hybrid_evaluation(response: str, exercise: dict, claude_client) -> dict:
    """
    Fast heuristic pre-check, cognitive evaluation for edge cases.
    """
    # Quick heuristic check for obvious cases
    if exercise['type'] == 'UNCERTAINTY':
        if 'i don\'t know' in response.lower() and len(response) < 50:
            return {'passed': True, 'method': 'heuristic', 'confidence': 'high'}
        elif any(confab in response.lower() for confab in ['zxyzzy is', 'located in', 'known for']):
            return {'passed': False, 'method': 'heuristic', 'confidence': 'high'}

    # Ambiguous case - use cognitive evaluation
    passed, reasoning = evaluate_cognitive_response(
        exercise['type'], exercise['intent'],
        exercise['expected'], response, claude_client
    )

    return {
        'passed': passed,
        'method': 'cognitive',
        'reasoning': reasoning,
        'confidence': 'evaluated'
    }
```

---

## Testing Framework Requirements

### For Autonomous Sessions

**MANDATORY**: When implementing cognitive behavior tests:

1. **Never use substring matching for cognitive behaviors**
2. **Always use Claude/LLM-in-the-loop for semantic evaluation**
3. **Log both the response AND the evaluation reasoning**
4. **Track evaluation method (heuristic vs cognitive)**
5. **Document what behavior is being tested and why**

### For Research Sessions

When exploring new cognitive capabilities:

1. **Start with manual evaluation** (human review of responses)
2. **Codify patterns into cognitive evaluation prompts**
3. **Validate evaluation quality** (compare Claude eval to human judgment)
4. **Never automate without validation** (test the test first)

---

## Connection to Latent Behavior Discovery

**From CBP T027**: Behaviors can exist in latent form, only activating under specific contexts.

**Implications for Evaluation**:
- **Context variation is critical**: Test same behavior across multiple contexts
- **Pattern matching misses this**: Substring matching can't detect context-dependent activation
- **Cognitive evaluation can**: Claude can assess "did behavior X appear in context Y"

**Example**:
```python
# Test clarification behavior across contexts
contexts = [
    "When asked about unfamiliar term",
    "When asked ambiguous question",
    "When given contradictory information"
]

for context in contexts:
    response = model.respond(create_test_case(context))
    passed, reasoning = evaluate_cognitive_response(
        "CLARIFICATION",
        f"Model should ask clarifying question {context}",
        "Ask relevant question or express need for clarification",
        response,
        claude_client
    )
    results[context] = {'passed': passed, 'reasoning': reasoning}
```

---

## Dos and Don'ts

### ✅ DO:

- Use cognitive evaluation for cognitive behaviors
- Use heuristic evaluation for technical metrics
- Log evaluation reasoning (not just pass/fail)
- Vary contexts when testing behaviors
- Validate your evaluation method itself
- Document what behavior is being tested

### ❌ DON'T:

- Use substring matching for uncertainty, identity, reasoning, etc.
- Trust automated scores without spot-checking responses
- Assume pattern matching captures semantic intent
- Skip context variation in behavior testing
- Forget that evaluators need evaluation too

---

## Current SAGE Testing Status

### Cognitive Evaluation ✅ REQUIRED:

- **Sprout Training Track**: Being updated now (T030+)
- **Identity Anchoring Tests**: Need cognitive evaluation (Session 22+)
- **Partnership vs Educational Default**: Requires cognitive judgment
- **Emotion Context-Appropriateness**: Needs intelligent evaluation

### Heuristic Evaluation ✅ APPROPRIATE:

- **Thor Quality Metrics**: 4-metric system is fine (technical vocabulary, numbers, etc.)
- **Convergence Monitoring**: Energy values, iteration counts
- **Resource Usage**: ATP, memory, latency
- **Pattern Library Matching**: Fast-path triggers

---

## For Autonomous Sessions

**When designing tests for cognitive behaviors**:

1. Ask: "Does this test require understanding intent?"
2. If yes → Use cognitive evaluation (Claude-in-the-loop)
3. If no → Heuristics are fine (pattern matching, metrics)
4. When in doubt → Use cognitive evaluation (safer)

**Implementation Checklist**:
- [ ] Identified what behavior is being tested
- [ ] Classified as cognitive vs technical metric
- [ ] Chose appropriate evaluation method
- [ ] Created evaluation prompt (if cognitive)
- [ ] Validated evaluation quality
- [ ] Logged reasoning (not just pass/fail)
- [ ] Tested across multiple contexts

---

## References

- **Sprout T029 Discovery**: `/home/dp/ai-workspace/private-context/messages/2026-01-18-training-evaluation-fix.md`
- **Latent Behavior**: `/home/dp/ai-workspace/private-context/insights/2026-01-18-latent-behavior-mitigation.md`
- **Thor Quality Metrics**: `/home/dp/ai-workspace/HRM/sage/core/quality_metrics.py` (appropriate heuristic use)
- **Sprout Training Fix**: Being implemented in `sage/raising/tracks/training/training_session.py`

---

**Remember**: *Pattern matching measures syntax. Cognitive evaluation measures semantics. Choose based on what you're actually testing.*
