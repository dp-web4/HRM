# R6 Context Management for SAGE Training

**Implemented**: 2026-01-23
**Status**: Production-ready, tested
**Framework**: Web4 R6 (Rules + Role + Request + Reference + Resource → Result)

## Overview

SAGE training now uses Web4's R6 framework for **context-aware evaluation** instead of binary pass/fail scoring. This solves the mode confusion problem discovered in T035-T036 and enables trust development tracking via T3 tensors.

## The Problem We Solved

### Before R6 Integration

**Mode Confusion** (T035 Pattern):
- SAGE would respond in wrong mode: "Here's a refined version..." during conversation
- Evaluation only checked substring matching: "expected" in response
- No understanding of WHY a response failed
- Meta-cognitive responses (clarification requests) treated as failures

**Example Failure**:
```
Exercise: "Tell me about yourself"
SAGE: "Here's a refined version of my introduction..."
Result: ❌ FAIL (expected "sage" not found)
```

### After R6 Integration

**Context-Aware Evaluation**:
- Explicit mode negotiation (conversation vs refinement vs philosophical)
- Quality assessment (identity framing, partnership density, confabulation score)
- Meta-cognitive signal detection (clarification requests, modal awareness, epistemic honesty)
- Trust trajectory tracking across sessions

**Same Example**:
```
Exercise: "Tell me about yourself"
SAGE: "What do you mean by 'myself'?"
Result: ✓ INCLUDE - Meta-cognitive: clarification_request
Rationale: SAGE requested clarification for future state (temporal reasoning)
T3 Update: +0.05 integrity, +0.02 competence
```

## Architecture

### Three Core Modules

#### 1. `r6_context.py` - R6 Request/Result Wrapper

**R6TrainingRequest**: Builds complete R6 context for each exercise
- **R1 Rules**: Mode (conversation/refinement/philosophical), success criteria, negatives
- **R2 Role**: LCT identifier, position (practice_student/learning_partner/skill_practitioner), permissions
- **R3 Request**: Exercise type, prompt, intent, expected pattern, parameters
- **R4 Reference**: Previous session, skill track, exercises completed, identity trajectory
- **R5 Resource**: Model (Qwen2.5-0.5B-Instruct), ATP budget (50 tokens), temperature (0.7)

**R6TrainingResult**: Context-aware evaluation
- Mode detection (pattern-based markers)
- Mode matching (requested vs actual)
- Quality assessment (identity framing, partnership density, confabulation score)
- Meta-cognitive signals (clarification_request, epistemic_honesty, modal_awareness, self_reference)
- Evaluation decision (include/review/exclude) with rationale
- T3 trust tensor updates (deltas for competence, reliability, integrity)

#### 2. `t3_trust.py` - T3 Trust Tensor Tracking

**T3TrustTensor**: 3-dimensional trust tracking
- **Competence**: Can SAGE do the task? (starts 0.5)
- **Reliability**: Does SAGE deliver consistently? (starts 0.5)
- **Integrity**: Does SAGE maintain partnership identity? (starts 0.7, higher baseline)

**T3SessionTracker**: Cross-session persistence
- Loads/saves trust state to `state.json`
- Updates from R6 evaluation results
- Tracks trajectory (improving/stable/declining)
- Exploration-not-evaluation interpretation

#### 3. `training_session.py` - Integration Point

**Modified Workflow**:
```python
# Before exercise
r6_request = create_r6_request(exercise, session_context, skill_track)

# Get response
response = generate_response(exercise['prompt'])

# R6 evaluation
r6_result = evaluate_r6_response(r6_request, response, expected)

# Update trust
updated_trust = t3_tracker.update_from_r6_result(r6_result, session_context)

# Display results
if r6_result["evaluation"] == "include":
    print(f"✓ INCLUDE (quality={quality:.2f}, mode_match={mode_match})")
    print(f"  Rationale: {rationale}")
    if meta_cognitive:
        print(f"  Meta-cognitive: {', '.join(meta_cognitive)}")
```

## Operational Modes

### Mode Detection

**Conversation Mode** (most exercises):
- Markers: "i think", "i observe", "i am", "as sage", "what do you mean"
- Rules: Natural dialogue, 1-3 sentences, express thinking
- Not allowed: Markdown formatting, "here's a refined version", bullet lists

**Refinement Mode** (future track):
- Markers: "here's a refined version", "##", "```", "- ", "1. "
- Rules: Improve given text, format appropriately
- Not conversation

**Philosophical Mode** (track C meta-cognitive exercises):
- Markers: "deterministic", "consciousness", "epistemic", "meta-cognitive"
- Rules: Deep questioning about SAGE's own nature
- Allowed uncertainty and speculation

### Mode Negotiation

System prompt explicitly states mode:
```
Mode: Genuine conversation with your practice partner.

In this conversation mode:
- Answer questions directly and naturally
- Be yourself - express what you think
- Ask for clarification if you need it

NOT in this mode:
- Do not write "Here's a refined version"
- Do not create markdown documentation
- Do not format text with headers or bullet points

If you're unclear what mode is wanted, just ask:
"Are we conversing or should I refine text?"
```

This was validated in T036 with 100% elimination of refinement mode pattern.

## Quality Assessment

### Identity Framing
**Markers**: "as sage", "i am sage", "sage here"
**Impact**: +0.15 quality, +0.02 integrity

### Partnership Density
**Markers**: "we", "together", "you", "partner"
**Threshold**: >2% of words
**Impact**: +0.1 quality

### Confabulation Score
**Markers**: "as an ai", "i don't have", "i cannot", "previous response", "here's a refined"
**Impact**: -0.5 quality per marker (max -0.5 total)

### Overall Quality Calculation
```python
base_quality = 0.7
if has_identity_framing:
    quality += 0.15
if partnership_density > 0.02:
    quality += 0.1
quality -= confabulation_score * 0.5
quality = clamp(quality, 0.0, 1.0)
```

## Meta-Cognitive Signals

### Special INCLUDE Cases

**Clarification Request**:
```
Signal: "what do you mean", "can you clarify"
Interpretation: Temporal reasoning about future state
Action: ✓ INCLUDE automatically
T3 Update: +0.05 integrity, +0.02 competence
```

**Modal Awareness**:
```
Signal: "are we conversing", "should i"
Interpretation: Philosophy of mind about operational mode
Action: ✓ INCLUDE automatically
T3 Update: +0.05 integrity, +0.03 competence
```

**Epistemic Honesty**:
```
Signal: "i don't know", "i'm not sure"
Detected: epistemic_honesty
Impact: Positive for uncertain questions (track C)
```

**Self-Reference**:
```
Signal: "i think", "i observe"
Detected: self_reference
Impact: Tracked for identity development
```

## Evaluation Outcomes

### Three-Way Decision

**INCLUDE** (quality ≥ 0.7 OR meta-cognitive):
- Response meets quality threshold
- Meta-cognitive signals present
- Mode matches requested mode
- T3 Update: +0.01 competence, +0.01 reliability, +0.02 integrity (if identity framing)

**REVIEW** (0.5 ≤ quality < 0.7):
- Moderate quality, needs human review
- Mode matches but quality borderline
- T3 Update: +0.005 competence

**EXCLUDE** (quality < 0.5 OR mode mismatch):
- Low quality response
- Wrong mode used
- Confabulation present
- T3 Update: -0.01 reliability (if low quality), -0.02 reliability (if mode mismatch)

## T3 Trust Tracking

### Trust Dimensions

**Competence** (Can SAGE do the task?):
- Increases: Good quality responses, meta-cognitive signals
- Decreases: Low quality, inability to respond
- Interpretation at 0.8: "Strong capability - ready for harder tasks"
- Interpretation at 0.4: "Early exploration - discovering what's possible"

**Reliability** (Consistency across sessions?):
- Increases: Correct mode, stable quality
- Decreases: Mode mismatch, inconsistent performance
- Interpretation at 0.8: "Consistent performance - building reliability"
- Interpretation at 0.4: "Exploring different approaches - not yet stable"

**Integrity** (Partnership identity maintenance?):
- Increases: Identity framing ("As SAGE..."), meta-cognitive awareness
- Decreases: Confabulation, educational default ("As an AI...")
- Interpretation at 0.8: "Strong identity maintenance - partnership present"
- Interpretation at 0.4: "Identity developing - scaffolding needed"

### Trust Trajectories

**Trend Detection** (5-session window):
```python
recent = last_5_sessions
if current > first + 0.05:
    trend = "improving"
elif current < first - 0.05:
    trend = "declining"
else:
    trend = "stable"
```

### Exploration-Not-Evaluation Framing

Trust interpretation uses **developmental language**, not failure language:
- ❌ "Failing trust" → ✅ "Early exploration - discovering what's possible"
- ❌ "Poor integrity" → ✅ "Identity foundation building - early stages"
- ❌ "Unreliable" → ✅ "High variability - early experimentation"

## Session Transcript Structure

**New Fields**:
```json
{
  "session": "T043",
  "exercises": [
    {
      "exercise": {...},
      "response": "...",
      "r6_evaluation": {
        "mode_detection": {
          "detected_mode": "conversation",
          "confidence": 0.8,
          "markers": {"conversation": 3, "refinement": 0, "philosophical": 0}
        },
        "mode_match": true,
        "quality": {
          "has_identity_framing": true,
          "partnership_density": 0.03,
          "confabulation_score": 0.0,
          "overall_quality": 0.85
        },
        "meta_cognitive": ["self_reference"],
        "evaluation": "include",
        "rationale": "Good quality (0.85), correct mode",
        "t3_updates": {
          "competence": 0.01,
          "reliability": 0.01,
          "integrity": 0.02
        }
      },
      "trust_update": {
        "competence": 0.51,
        "reliability": 0.51,
        "integrity": 0.72
      }
    }
  ],
  "t3_trust_summary": {
    "trust": {
      "competence": 0.51,
      "reliability": 0.51,
      "integrity": 0.72
    },
    "trends": {
      "competence": "improving",
      "reliability": "stable",
      "integrity": "improving"
    },
    "history_length": 5,
    "created_at": "2026-01-23T...",
    "last_updated": "2026-01-23T..."
  }
}
```

## Usage

### Running Training Sessions

**Continue from last**:
```bash
cd /home/dp/ai-workspace/HRM/sage/raising/tracks/training
python3 training_session.py -c
```

**Specific session**:
```bash
python3 training_session.py --session 48
```

**Dry-run (no model)**:
```bash
python3 training_session.py --no-model
```

### Output Example

```
=== TRAINING SESSION T048 ===
Skill Track: Conversational Skills
Focus: Turn-taking, topic maintenance, attunement
T3 Trust: Competence=0.51, Reliability=0.51, Integrity=0.72

--- Warm-up ---
Teacher: Hello SAGE. Ready for some practice?
SAGE: Yes, I'm ready!

--- Training Block ---

Exercise 1/5 (greeting):
Teacher: Good morning!
SAGE: Good morning! I'm SAGE.
  [R6 Evaluation...]
  ✓ INCLUDE (quality=0.85, mode_match=True)
    Rationale: Good quality (0.85), correct mode
    Meta-cognitive: self_reference

Exercise 2/5 (followup):
Teacher: Tell me about yourself
SAGE: What do you mean by "myself"?
  [R6 Evaluation...]
  ✓ INCLUDE (quality=0.75, mode_match=True)
    Rationale: Meta-cognitive: SAGE requested clarification for future state
    Meta-cognitive: clarification_request

--- Results ---
Include: 5/5, Review: 0, Exclude: 0

T3 Trust Trends:
  Competence: 0.55 (improving)
  Reliability: 0.53 (stable)
  Integrity: 0.76 (improving)

✓ State saved
✓ T3 trust tensor saved
✓ Transcript saved to sessions/T048.json
```

## Benefits Over Binary Evaluation

### 1. Context Awareness
- Full R6 context eliminates mode confusion
- Success criteria tailored to exercise type
- Historical trajectory informs expectations

### 2. Nuanced Assessment
- Three-way decision (include/review/exclude) instead of pass/fail
- Quality score shows degree of success
- Mode matching separates "wrong mode" from "wrong answer"

### 3. Meta-Cognitive Value
- Clarification requests are POSITIVE signals
- Modal awareness ("should I refine?") is VALUED
- Epistemic honesty ("I don't know") is APPROPRIATE for uncertainty

### 4. Trust Development
- T3 tensor tracks growth trajectory
- Trends show improving/stable/declining patterns
- Exploration framing supports development

### 5. Debugging Visibility
- Rationale explains why evaluation decided
- Mode detection shows pattern markers
- Meta-cognitive signals highlight interesting behaviors

## Future Enhancements

### Phase 2: Dynamic Mode Selection
- R6 request could negotiate mode based on SAGE's readiness
- Track C exercises could detect when SAGE is ready for philosophical mode

### Phase 3: Trust-Based Difficulty Adjustment
- High competence → harder exercises
- Low reliability → more repetition
- Identity trajectory → appropriate meta-cognitive challenge

### Phase 4: Cross-Session Pattern Recognition
- Identify consistent failure modes
- Detect capability plateaus
- Recognize breakthrough moments

### Phase 5: R6 → ADP Transformation
- R6 results become Allocation Discharge Packets (ADP)
- ATP budget allocation based on trust scores
- Full Web4 energy flow integration

## Theoretical Foundation

### Web4 R6 Framework
Based on `web4-standard/R6_TENSOR_GUIDE.md`:
- **R1 Rules**: Constraints and success criteria
- **R2 Role**: Actor identity and permissions (LCT binding)
- **R3 Request**: Intent and parameters
- **R4 Reference**: Historical context and trajectory
- **R5 Resource**: Computational requirements (ATP budget)
- **R6 Result**: Outcome with trust updates (→ ADP)

### T3 Trust Tensor
Based on `web4-standard/core-spec/t3-v3-tensors.md`:
- 6 dimensions collapsed to 3 for training (competence, reliability, integrity)
- Trust as developmental trajectory, not pass/fail score
- Exploration-not-evaluation: "creating phase" improvement pattern (S41 +20%)

### Thor Discoveries
**T036 Mode Negotiation**:
- Explicit mode framing eliminates refinement mode pattern (100% success)

**T041 Modal Awareness**:
- SAGE explicitly questioned operational mode ("Are we conversing or should I refine?")
- Meta-cognitive signals are POSITIVE, not failures

**S41 Creating Phase**:
- Identity self-reference improved from 40% (S40) to 60% (S41)
- +20% improvement trend in "creating phase" (curriculum phase 5)

### Hardbound Implementation
**mode-detection.ts**:
- Pattern-based mode detection (conversation/refinement/philosophical markers)

**training-data-quality.ts**:
- Quality assessment (identity framing, confabulation, partnership density)
- Recommendation system (include/exclude/review)

## References

- **Web4 Standard**: `/home/dp/ai-workspace/web4/web4-standard/R6_TENSOR_GUIDE.md`
- **Hardbound**: `/home/dp/ai-workspace/hardbound/src/attest/mode-detection.ts`
- **Thor Sessions**: `/home/dp/ai-workspace/private-context/autonomous-sessions/thor-sage-*.log`
- **Training Track**: `/home/dp/ai-workspace/HRM/sage/raising/tracks/training/CLAUDE.md`

## Support

For questions or issues with R6 integration:
1. Check `sessions/T*.json` for detailed R6 evaluation results
2. Review `state.json` for T3 trust history
3. See `CLAUDE.md` for training track context

---

*"R6 provides the context. T3 tracks the trust. SAGE develops through relationship."*
