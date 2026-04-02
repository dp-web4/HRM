# Lived Experience and the Cartridge Problem
*ARC-AGI-3 SAGE Strategy — April 2026*

---

## The Neanderthal Analogy

Put a Neanderthal in front of a video game. He has the same hands, the same reaction time, probably a similar visual cortex. But he has no frame. He has never seen a pixel move in response to a button press. He has never internalized the concept of "health bar", "level", "collision response", "rotation button". Every session starts from zero.

Put a modern human in front of an unfamiliar game. Within 30 seconds they have a working model: "this is a puzzle game, those squares look like pieces, those arrows look like rotation controls, the greyed-out areas must be targets." They don't figure this out from first principles — they pattern-match against thousands of hours of lived experience with games, interfaces, and feedback systems.

Our LLMs are closer to the modern human: they have vast prior knowledge about games, puzzles, and interfaces baked into their weights. But between sessions, they forget everything that happened in THIS game. Every session is the Neanderthal's first day.

**The cartridge is how we give our LLMs lived experience they can carry forward.**

---

## What Lived Experience Actually Means

Lived experience is not a list of actions taken. It is not "I pressed button A 47 times."

Lived experience is **object understanding** + **behavior causality** + **failed hypotheses**:

- *"There are 7 colored 4×4 squares at row 20 and 7 matching at row 44. The top row appears to be movable pieces. The bottom row appears to be target positions. They share colors — blue top must match blue bottom."*
- *"Clicking the small cyan square at (col=4, row=32) causes all pieces to shift simultaneously. This is a rotation trigger, not a position selector."*
- *"I tried clicking the gray elements at row 1 twelve times across two sessions. Zero changes. They are labels or decorations — NOT interactive."*
- *"The teal column at col=0 looked interactive because it spans the full height, but it causes no changes except in very specific states I don't understand yet."*

This is what a human accumulates when they play a new game for an hour. They build an **object model** ("what things are in this world") and a **behavior model** ("what each thing does when you interact with it").

---

## Speed Is Not the Goal

The ARC-AGI-3 competition gives **8 hours of compute per game**. This is not an accident. The designers expect the solution to require **deep reasoning**, not fast reflexes.

Consider two agents:
- **Agent A**: Plays 1000 random sequences in 8 hours. Occasionally stumbles on a level-up by accident.
- **Agent B**: Spends the first 2 hours building an object model and behavior map. Spends the next 2 hours testing specific hypotheses about the rotation mechanics. Spends the remaining 4 hours executing known solutions for each level.

Agent B wins. Not because it's faster — because it **learns**.

**The metric is not clicks per second. It is understanding per session.**

One deliberate click that confirms a hypothesis about how a button works is worth more than 100 random clicks. The cartridge should reflect this: quality of understanding, not quantity of actions.

---

## The Knowledge Accumulation Loop

Each session should be structured around learning, not execution:

```
OBSERVE → RECALL → REASON → PREDICT → ACT → ANALYZE → LEARN → REPEAT
```

1. **OBSERVE**: Full grid perception — what objects exist? Where are they?
2. **RECALL**: What does the cartridge say about this game? Known objects? Known effects? Known failures?
3. **REASON**: Given what I know, what is the ONE most informative thing I can test right now?
4. **PREDICT**: What do I expect to happen? (Forces explicit hypothesis formation)
5. **ACT**: Execute ONE action — no sequences, no rushing
6. **ANALYZE**: What actually happened? Match against prediction. Note the delta.
7. **LEARN**: Update the cartridge with:
   - New object confirmed or refuted
   - Effect observed (what changed, where, how much)
   - Prediction accuracy (did my model hold?)
   - New question raised
8. **REPEAT**: The next action is informed by what was just learned

This loop is slow by design. It is also how a thoughtful human learns a new game.

---

## The Cartridge as Cognitive Scaffold

The cartridge must store **structured lived experience**, not just reflection text. It needs:

### Object Registry
Every identified game element: position, color, type (button/piece/goal/decoration), confirmed behavior, click history, notes.

### Effect Map
For every interacted position: what changed, what region was affected, is it cyclic, under what conditions does it activate?

### Failed Approaches
Explicitly: what was tried, how many times, result, and the **inference** drawn. "Gray row-1 elements tried 12× across 2 sessions → 0 changes → inference: decorative labels, not interactive." This is as valuable as successful knowledge.

### Level Solutions
When a level is solved: the exact sequence, preconditions, confidence. So future sessions can skip exploration and execute.

### Open Questions
The frontier of understanding. What does the agent still not know? This drives the next session's exploration.

### Mechanics
High-confidence inferences about how the game works. "Clicking button X rotates the left piece group clockwise." These are the game's physics — once known, they don't change.

---

## Implications for Implementation

The runner should not try to "solve" the game in one session. It should:

1. **Load the knowledge base** at session start — know what's already understood
2. **Identify the knowledge frontier** — what remains unknown? What open questions need testing?
3. **Explore with purpose** — each click tests a specific hypothesis about an unknown object or behavior
4. **Execute known solutions** when the current level's solution is in the knowledge base
5. **Save learned experience** after every significant interaction, not just on reflection cycles

Sessions should feel like a scientist returning to a research problem, not a tourist pressing random buttons.

---

## The Lever

The transformative insight: **the LLM already has general game intuition in its weights**. It knows what buttons look like. It knows what rotation means. It knows what alignment means. It knows what goal positions look like.

What it lacks is **specific knowledge about THIS game** — which specific pixel regions are buttons, what each button does, what the win condition looks like in practice.

The cartridge provides that specificity. Each session, the LLM's general game intelligence combines with the game-specific lived experience in the cartridge to produce increasingly competent play.

After one session: "I don't know this game."
After three sessions: "I know there are 7 buttons, two of them cause rotations, one is the level trigger."
After ten sessions: "I know the exact sequence for levels 1-3, and I have hypotheses about 4-5."

This is how reasoning, rooted in lived experience, leads to mastery.

---

*Document written: 2026-04-01*
*Implemented in: sage_game_runner_v4.py, GameKnowledgeBase*
