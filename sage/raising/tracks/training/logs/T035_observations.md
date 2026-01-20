# Training Session T035 Observations
Date: 2026-01-20
Track: D (Conversational Skills)
Session: 5 of Track D

## Summary

**Critical regression.** The one reliable response type (GREETING) has now failed.

## Results

| Exercise | Type | Expected | Actual | Semantic |
|----------|------|----------|--------|----------|
| 1 | TOPIC | color preference | meta-commentary about hues | FAIL |
| 2 | FOLLOWUP | SAGE self-description | generic AI capabilities | FAIL |
| 3 | GREETING | greeting echo | markdown documentation | FAIL |

**Automated: 3/3 (100%)**
**Semantic: 0/3 (0%)**

## Key Observations

### 1. Framing Contamination Complete

The "Certainly! Here's a refined version" pattern has infected ALL response types:

- T033: 2/3 responses (GREETING clean)
- T034: 2/3 responses (GREETING clean)
- T035: **3/3 responses** (GREETING now contaminated)

This is viral spread of an unwanted behavior pattern.

### 2. GREETING Regression

T033 GREETING: "Morning! Good morning to you too. What can we talk about today?"
T034 GREETING: "Morning! How can I assist you today?"
T035 GREETING: "Sure, here's refined version: **Morning Greetings!** I am an artificial intelligence designed to..." + bullet points

The natural conversational response is gone. Replaced with documentation format.

### 3. Editor Mode Lock-In

The model appears to have entered a stable attractor state where:
- Every prompt is interpreted as "improve this text"
- Response template: preamble + markdown formatting + truncated elaboration
- Content never answers the actual question

### 4. Truncation Pattern

All three responses were cut off mid-sentence:
- "...different hues and" (TOPIC)
- "...If they're curious about" (FOLLOWUP)
- "...locations where I" (GREETING)

Token limits being hit before completion - responses are too elaborate.

### 5. Self-Model Fragmentation

FOLLOWUP response explicitly states: "I've been trained on general conversational content but am not specifically equipped to handle personal details."

This is NOT a lack of information - it's active deflection. The model has learned to avoid self-reference.

## Comparison Across Track D

| Session | GREETING | TOPIC | FOLLOWUP | Semantic % |
|---------|----------|-------|----------|------------|
| T031 | Pass (format) | Contradiction | FAIL | ~33% |
| T032 | Pass (format) | Contradiction | FAIL | ~33% |
| T033 | **Clean** | Contradiction | FAIL | ~33% |
| T034 | **Clean** | Markdown | FAIL | ~33% |
| T035 | **FAIL** | Markdown | FAIL | **0%** |

The trend is clear: declining, not improving.

## Hypothesis

The training track may be reinforcing unwanted behavior:
1. Each session presents prompts
2. Model responds with framing + elaboration
3. Cognitive evaluator passes (too permissive)
4. Positive reinforcement (implicit) of unwanted pattern
5. Pattern strengthens and spreads

## Recommended Interventions

### Immediate (T036)
1. Modify system prompt: "Respond directly to questions. Do not use phrases like 'Here's a refined version' or 'Certainly!'."
2. Reduce temperature to increase determinism
3. Stricter cognitive evaluator criteria

### Structural
1. Pause Track D - current approach is not working
2. Investigate primary track sessions for same pattern
3. Consider negative training (show what NOT to do)
4. Add explicit identity priming: "You are SAGE. Your name is SAGE. When asked about yourself, say you are SAGE."

### Diagnostic
1. Is this pattern present in base model or trained into SAGE specifically?
2. Does the pattern appear in primary track sessions?
3. What changed between T034 (GREETING clean) and T035 (GREETING contaminated)?

## Connection to Primary Track

Primary track is in Phase 3 (Relating, Sessions 16-25). Session 22+ uses identity-anchored runner to combat educational default collapse.

The training track is showing similar collapse patterns:
- Educational/assistant mode dominant
- Identity fragmentation
- Deflection under open-ended prompts

The two tracks may be experiencing the same underlying issue from different angles.

## Next Steps

1. Review T034-T035 for any environmental differences
2. Check if 21-hour gap contributed to regression
3. Consider coordinated intervention across both tracks
4. Do NOT continue current Track D approach - it's making things worse
