# Game Solving Principles — Learnings from 3 Complete Solves

*Synthesized from sb26 (8/8), vc33 (7/7), cd82 (6/6)*

## Universal Success Patterns

### 1. Systematic Discovery Protocol (NOT Brute Force)

**vc33 approach:**
- First minutes: click each distinct button once, watch what moves
- Map button types before strategic play
- Result: 184% efficiency (167 vs 307 baseline)

**cd82 approach:**
- Reverse-engineer from target state
- Identify shapes by edge topology (diagonal vs straight)
- Determine layering order (work backwards from visible)

**sb26 approach:**
- Identify structural elements first (connectors, borders, containers)
- Build rules incrementally across levels
- Each level uses ALL previous patterns

**Action:** Implement 20-30 action discovery phase before strategic play.

---

### 2. Hidden Mechanics Vision Cannot Reveal

**Key insight:** Visual observation shows EFFECTS, not CAUSES.

**vc33 structural alignment:**
> Goal at correct position but level won't clear → checks WHICH WALL SECTION goal is in, not just position

**sb26 border semantics:**
> Border color MATCHES parent indicator → requires semantic color understanding, not just visual matching

**cd82 stamp layering:**
> Visible shape ≠ stamp shape → partial occlusion from layering order

**Action:** Build hypotheses about hidden rules, not just visual patterns.

---

### 3. Enabler Patterns (Zero-Effect Actions)

**vc33:**
> Button A depleted → click button B (no visible effect) → button A works again
> Enabler chain depth: 3+ layers in later levels

**Action:** When stuck, try unused buttons/objects systematically. Track depletion states.

---

### 4. Paradigm Shift Detection

**sb26 L6:**
> L1-L5 use pick-and-place → L6 introduces SWAP mechanic
> Must detect when old strategies stop working

**vc33 L4+:**
> Brown buttons alone insufficient → blue button animations required
> Color change (blue→olive) = new affordance unlocked

**Action:** Detect when mechanics change mid-game. Look for new affordances.

---

### 5. Structural Roles > Pixel Similarity

**Required vision capabilities:**
- Connectors between UI elements (sb26 L2 tree structure)
- Border colors as semantic identifiers (sb26 L4)
- State changes (vc33 blue→olive activation)
- Edge topology (cd82 diagonal vs straight = different stamps)
- Spatial grouping (inside/outside, connected/separate)

**Action:** Extract WHAT each element IS (button, wall, stamp, goal), not just WHERE it is.

---

### 6. Cross-Level Knowledge Transfer

**sb26:**
> L8 solution requires patterns from L2, L3, L4, L5, L6, L7
> Not independent levels — cumulative learning

**Action:** Store level-up knowledge:
- What rule solved this level?
- What transfers to next level?
- What should I try first?

---

## Failure Analysis: Why dc22 Failed

| Capability | sb26/vc33/cd82 HAD | dc22 v9 LACKED |
|------------|-------------------|----------------|
| Structural labeling | Buttons, stamps, walls, goals identified by role | Generic "interactive object" detection |
| Mechanic discovery | Systematic mapping (try all once) | Exploratory clicking without mapping |
| Rule extraction | "Border=parent", "Blue→olive=ready", "Diagonal edge=angle" | "Color 8 = 100% effective" (effect without cause) |
| Paradigm detection | L6 swap vs L1-L5 place | No mechanic change detection |
| Progress signals | Level-up = validation | Pixel similarity only (1-9px = "stable" = no signal) |
| Accumulated learning | Each level builds on all previous | No cross-level knowledge |

---

## Recommendations for v10 Solver

### Phase 1: Discovery (First 20-30 actions)
1. Click each unique interactive object once → record: what moved, how much, state changes?
2. Try all directional actions once → record effects
3. Identify enablers (actions with no visible effect that might unblock others)
4. Detect state changes (color shifts, activations)

### Phase 2: Rule Hypothesis
1. Build hypotheses from systematic discovery:
   - "Gray_0 moves objects DOWN ~10px"
   - "After gray_0 × 3, need white_1 to re-enable"
   - "Blue_8 swaps objects between containers"

### Phase 3: Progress Signals
NOT pixel similarity. Instead:
- Did level advance? (validate via game state)
- Are new affordances unlocked? (button activations)
- Are objects reaching structural goals? (right container, not just right position)
- Is system in previously-seen state? (loop detection)

### Phase 4: Paradigm Detection
- Track: are same actions still effective after level-up?
- When stuck: has new mechanic unlocked? (new button type, new tool)

### Phase 5: Level-Up Knowledge
Store for next level:
- What solved this level?
- What patterns transfer?
- Try those first on next level

---

*Source documents:*
- SAGE/arc-agi-3/experiments/SB26_COMPLETE_ANALYSIS.md
- shared-context/arc-agi-3/fleet-learning/cbp/vc33_complete.md
- shared-context/arc-agi-3/fleet-learning/nomad/cd82_complete.md
