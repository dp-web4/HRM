---
name: LLM game-playing framing
description: When building LLM prompts for game-playing, frame the model as a player looking at a screen, not a character inside the game
type: feedback
---

The model is a player looking AT a screen, not a character IN the game. Don't tell it where "you" are — tell it where the cursor is. The model's objective is to move the cursor among objects it perceives in the frame. You can't see the maze when you're in it, but you can when looking at it.

**Why:** User corrected the approach of placing the model "inside" the game. The model needs third-person perspective to reason about the whole board. Also: there are no heuristics for novel games — they test reasoning about the unknown. Heuristics only work for the known. The model MUST reason, so give it the information it needs to reason WITH.

**How to apply:** When building prompts for game-playing LLMs, provide:
1. Where the cursor is (not "you")
2. What's around the cursor in each direction
3. What the last action did (concrete causal feedback)
4. What's been learned so far
5. Never replace reasoning with heuristics for novel/discovery tasks
