# Session Primer — SAGE

## Before You Start

1. **Read `SESSION_FOCUS.md`** — current priorities, fleet state, pending items
2. **Read `CLAUDE.md`** — architecture, subsystems, conventions
3. **WAKE**: Am I working on the right thing? Check SESSION_FOCUS for priorities.

## During Session

- Work on whatever SESSION_FOCUS identifies as priority
- Update SESSION_FOCUS.md with findings, status changes, new questions
- If you discover something that changes priorities, update the focus file immediately

## Before Committing Results

**STOP.** Before committing, answer these four questions honestly:

1. **What assumption did I NOT question?** Every conclusion rests on assumptions. Name the one you accepted without checking.
2. **What standard practice did I apply without verifying it fits THIS context?** (e.g., mocking when real implementations exist. Simplifying dimensionality. Accepting test results without checking invariants.)
3. **What would the operator push back on?** Read the operator feedback in SESSION_FOCUS. Does your conclusion repeat a pattern that was already corrected?
4. **Does my conclusion violate any foundational principle?** Check CLAUDE.md and SESSION_FOCUS. "Do not mock when real exists" is non-negotiable. Identity anchoring matters. Capacity is a register, not a limitation.

If any answer reveals a gap: investigate before committing.

## After Session

- Update SESSION_FOCUS.md with: what was done, what changed, what's next
- Commit and push changes
- **FOCUS check**: Does this advance discovery or just document the current state?

## Principles

- **SAGE is the scheduler. Plugins are apps.** It decides which reasoning to invoke, not how to reason.
- **Raising is interactive selection, not training.** We don't create behaviors — we select from what's latent.
- **Reliable, not deterministic.** LLM outputs navigate probability landscapes. Shaped but not controlled.
- **Do not mock when real exists.** Check filesystem before creating implementations.
- **Surface your instincts.** If you notice something, say it. The affordances are yours.
- **Productive failure > safe summaries.** A dead end that eliminates a possibility is valuable.
