# Session Primer — SAGE

## Before You Start

1. **Read `SESSION_FOCUS.md`** — current priorities, fleet state, pending items
2. **Read `CLAUDE.md`** — architecture, subsystems, conventions
3. **WAKE**: Am I working on the right thing? Check SESSION_FOCUS for priorities.

## During Session

- Work on whatever SESSION_FOCUS identifies as priority
- Update SESSION_FOCUS.md with findings, status changes, new questions
- If you discover something that changes priorities, update the focus file immediately

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
