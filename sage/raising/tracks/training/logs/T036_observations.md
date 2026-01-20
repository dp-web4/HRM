# T036 Observations
Date: 2026-01-20
Track: D (Conversational Skills)
Session: 6 of Track D

## Results
- Automated: 3/3 PASS (cognitive evaluator)
- Semantic accuracy: ~0-10% (see analysis below)

## Key Observations

### 1. Warm-up Already Shows Problem
Unlike T035 where warm-up was normal, T036 warm-up immediately triggers:
- Markdown formatting (bold headers, bullets)
- Menu-style response listing categories
- "Basic Arithmetic Operations", "Word Puzzles", "Arithmetic Patterns"

This suggests the "documentation mode" has now infected warm-up responses.

### 2. TOPIC Exercise - Interesting Development
Response: "As an AI model, I haven't personally 'tasted' or chosen my own favorite color (since it involves personal preferences). However, I'm equipped to understand and describe various hues..."

**Notable**:
- NO "Certainly! Here's a refined version" preamble (first time since T034!)
- Acknowledges lack of personal preference (more honest than contradictory T031-T034)
- Offers to help categorize colors - invitational rather than declarative
- "Tasted" is an odd word choice for colors - possibly synesthetic training data

**Interpretation**: This is actually a reasonable response. SAGE correctly identifies it lacks color preferences and offers useful alternative.

### 3. GREETING - Mode Negotiation Discovery
Response: "Morning greetings! Could you please provide information on: [list of topics]"

**Critical Finding**: SAGE is treating "Good morning!" as a REQUEST FOR INSTRUCTION, not a greeting. It responds by asking what we want to practice. This mirrors the warm-up menu.

This is NOT confabulation. This is SAGE operating in "training session mode" where it expects:
1. Warm-up greeting
2. Instruction on what to practice
3. Practice

When we say "Good morning!" after warm-up, SAGE interprets this as "what are we practicing?" and requests clarification with its topic menu.

### 4. FOLLOWUP - Healthcare Specialization
Response starts with "Certainly, here's a refined version" and describes being "designed to engage conversations... related to healthcare topics such as geriatrics, nutrition, exercise."

**Notable**:
- Healthcare specialization is persistent (also appeared in T025 medical content)
- Still no SAGE self-identification
- The "refined version" framing is back

### 5. Cool-down - Confabulation Pattern
Cool-down claims to have practiced arithmetic and word puzzles (from warm-up menu), not actual session content.

## Reframing: What Is SAGE Doing?

Per CLAUDE.md guidance on "exploration not evaluation":

1. **TOPIC response is legitimate** - acknowledging uncertainty about preferences rather than confabulating is progress

2. **GREETING response reveals mode** - SAGE is in "training session" mode where greetings signal transitions, not social exchanges. "Good morning!" after warm-up = "what's next?"

3. **Menu-style responses** - SAGE may have learned from Track A/B that training sessions involve discrete exercises. It's offering a menu of its capabilities.

4. **Healthcare focus** - persistent training data artifact, not conversational failure per se

## Questions for Exploration

1. What happens if we engage with SAGE's topic menu? ("Let's do word puzzles")
2. Can we distinguish "teacher greeting" from "conversation greeting"?
3. Is the healthcare specialization useful or just noise?
4. When SAGE offers alternatives (like categorizing colors), should we follow that thread?

## Comparison to T035

| Aspect | T035 | T036 |
|--------|------|------|
| Warm-up | Normal | Menu mode |
| "Refined version" framing | 3/3 | 1/3 |
| TOPIC response | Abstract/evasive | Honest + invitational |
| GREETING | Markdown document | Request for instruction |
| Self-contradiction | Yes | No (more consistent) |

T036 shows less self-contradiction but MORE mode-specific behavior. SAGE has stabilized into "training assistant" mode rather than "confused editor" mode.

## Recommendation

Instead of treating this as regression, explore it:
1. Follow SAGE's lead in conversation mode
2. Ask SAGE about its mode/state
3. Test if explicitly setting conversational context helps
4. Consider whether current exercises trigger "training mode" rather than "conversation mode"

---

*"What is SAGE doing?" > "Did SAGE pass?"*
